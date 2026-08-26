"""The prediction artifact: the one interface between whatever produced a prediction and scoring.

This is the whole reason `mia-evals` never imports a model. A producer writes an array plus the
attrs below; everything downstream reads only that. A `mia-train` checkpoint, a collaborator's
segmentation, and a baseline from a paper therefore all enter by the same door, and no code path
here is privileged for "our own" models.

    <artifact>.zarr        (C, *spatial) or (*spatial), any dtype
      .attrs
        kind          one of KINDS -- what the numbers *mean*, which decides what may postprocess it
        background_id  a labelling's "no object here" value      (required for a labelling)
        ignore_id      a labelling's "unknown, do not score" value
        origin        (x, y, z) of this array's corner within the source volume
        convention    free text: any squashing already applied, e.g. "sigmoid(0.2 * logit)"
        ... plus whatever provenance the producer knows (run, step, cube, patch, stride)

**Why `background_id` is required rather than assumed to be 0.** For thresholded connected
components a label of `0` means "no affinity edge survived here" -- which happens both at real
membrane and wherever the model was merely unsure -- and *not* "background". A producer whose `0`
is a genuine instance would otherwise have its largest object silently scored as background, and
the number that comes out is plausible. This has already been the most dangerous distinction in the
pseudo-labelling work, so it is stated per artifact rather than inferred.

**Why `kind` and not just the array shape.** `(6, X, Y, Z)` of floats could be affinities or six
class scores, and thresholding class scores as if they were affinities produces a segmentation
rather than an error. The shape is checked *against* the declared kind, so a mismatch fails at the
read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zarr

# What a producer may declare, and what each kind promises about its array. `channels` is the
# leading axis: an int fixes it, "rank*2" means two per spatial axis (the affinity offsets), and
# None means there is no channel axis at all.
#
# `canonical` is what postprocessing this kind must produce, and it is the narrow waist of the
# whole design: metrics attach to the canonical form, never to the kind, so nERL neither knows nor
# cares whether the labelling came from affinities, a watershed, or a file someone sent us.
KINDS: dict[str, dict[str, Any]] = {
    "affinity":     {"channels": "rank*2", "canonical": "instances", "floating": True},
    "boundary":     {"channels": 1,        "canonical": "instances", "floating": True},
    "embedding":    {"channels": "any",    "canonical": "instances", "floating": True},
    "sdt":          {"channels": 1,        "canonical": "instances", "floating": True},
    "instances":    {"channels": None,     "canonical": "instances", "floating": False},
    "class_scores": {"channels": "any",    "canonical": "classes",   "floating": True},
    "class_labels": {"channels": None,     "canonical": "classes",   "floating": False},
}
CANONICAL_FORMS = ("instances", "classes")
LABELLING_KINDS = ("instances", "class_labels")


@dataclass(frozen=True)
class Artifact:
    """A prediction on disk, with its declaration read and checked but its data not loaded.

    The array is left in the store deliberately. A whole-cube affinity prediction is ~51 GB and a
    whole-cube uint32 labelling ~49 GB, so what a caller wants is almost always a block: `read()`
    takes one, and `load()` is the explicit "yes, all of it" for the cases that need it.
    """

    path: Path
    kind: str
    shape: tuple[int, ...]
    spatial_shape: tuple[int, ...]
    origin: tuple[int, ...]
    background_id: int | None
    ignore_id: int | None
    convention: str
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def canonical(self) -> str:
        """Which scoreable form postprocessing this artifact must produce."""
        return str(KINDS[self.kind]["canonical"])

    @property
    def is_labelling(self) -> bool:
        return self.kind in LABELLING_KINDS

    @property
    def channels(self) -> int | None:
        """Size of the leading axis, or None for a kind that has no channel axis."""
        return None if KINDS[self.kind]["channels"] is None else int(self.shape[0])

    def _store(self) -> Any:
        return zarr.open(str(self.path), mode="r")

    def read(self, origin: tuple[int, ...] | None = None,
             size: tuple[int, ...] | None = None) -> np.ndarray:
        """A block, in coordinates *absolute to the source volume* rather than to this array.

        Absolute, because a caller holding a bounding box from a data config has it in volume
        coordinates and has no reason to know that this artifact covers a sub-region. Subtracting
        `origin` at each callsite instead is the kind of arithmetic that is wrong once and then
        wrong quietly -- an off-by-`origin` read returns real data from the wrong place.
        """
        store = self._store()
        if origin is None and size is None:
            return np.asarray(store[:])
        origin = origin or self.origin
        size = size or self.spatial_shape
        local = [a - b for a, b in zip(origin, self.origin, strict=True)]
        for axis, (start, extent, available) in enumerate(
            zip(local, size, self.spatial_shape, strict=True)
        ):
            if start < 0 or start + extent > available:
                raise ValueError(
                    f"requested [{origin[axis]}, {origin[axis] + extent}) on axis {axis}, but this "
                    f"artifact covers [{self.origin[axis]}, {self.origin[axis] + available}). "
                    "Origins are absolute to the source volume."
                )
        window = tuple(slice(o, o + s) for o, s in zip(local, size, strict=True))
        block = (slice(None), *window) if self.channels is not None else window
        return np.asarray(store[block])

    def load(self) -> np.ndarray:
        """The whole array. Named to make its cost visible at the callsite."""
        return np.asarray(self._store()[:])


def _check_channels(kind: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Validate the leading axis against what `kind` promises; return the spatial shape."""
    expected = KINDS[kind]["channels"]
    if expected is None:
        return shape
    if len(shape) < 2:
        raise ValueError(
            f"kind={kind!r} carries a channel axis, so the array needs at least 2 dimensions, "
            f"got shape {shape}"
        )
    channels, spatial = shape[0], shape[1:]
    if expected == "rank*2" and channels != 2 * len(spatial):
        raise ValueError(
            f"kind='affinity' over {len(spatial)} spatial axes needs {2 * len(spatial)} channels "
            f"(short-range then long-range, one per axis), got {channels}. A 3-channel array is "
            "the short-range half only: re-predict with all of them, or declare kind='boundary'."
        )
    if isinstance(expected, int) and channels != expected:
        raise ValueError(f"kind={kind!r} needs {expected} channel(s), got {channels}")
    return spatial


def open_artifact(path: str | Path) -> Artifact:
    """Read an artifact's declaration and check it, without loading the data.

    Every failure here is a mislabelled artifact, which is worth catching before a scoring job
    spends an hour producing a number that means something other than it says.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no artifact at {path}")
    store = zarr.open(str(path), mode="r")
    attrs = dict(store.attrs)

    kind = attrs.get("kind")
    if kind is None:
        raise ValueError(
            f"{path} declares no `kind`, so nothing can know what its numbers mean. Producers "
            f"must set it; valid kinds are {sorted(KINDS)}. (Affinity zarrs written before the "
            "artifact spec existed are 6-channel affinities: add kind='affinity' to their attrs.)"
        )
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r} in {path}; valid kinds are {sorted(KINDS)}")

    shape = tuple(int(s) for s in store.shape)
    spatial = _check_channels(kind, shape)

    background_id = attrs.get("background_id")
    ignore_id = attrs.get("ignore_id")
    if kind in LABELLING_KINDS and background_id is None:
        raise ValueError(
            f"{path} is kind={kind!r} but declares no `background_id`. For a labelling this is not "
            "a detail: connected components emit 0 for 'no edge survived here', which is not the "
            "same claim as 'background', and a producer whose 0 is a real instance would have its "
            "largest object scored as background. Set background_id (commonly 0), and ignore_id "
            "(commonly -1) if any voxel is unannotated."
        )

    origin = tuple(int(o) for o in attrs.get("origin", (0,) * len(spatial)))
    if len(origin) != len(spatial):
        raise ValueError(
            f"{path} records origin {origin} but has {len(spatial)} spatial axes {spatial}"
        )

    return Artifact(
        path=path,
        kind=kind,
        shape=shape,
        spatial_shape=spatial,
        origin=origin,
        background_id=None if background_id is None else int(background_id),
        ignore_id=None if ignore_id is None else int(ignore_id),
        convention=str(attrs.get("convention", "")),
        attrs=attrs,
    )


def write_artifact(
    path: str | Path,
    array: np.ndarray,
    kind: str,
    *,
    origin: tuple[int, ...] | None = None,
    background_id: int | None = None,
    ignore_id: int | None = None,
    convention: str = "",
    chunks: tuple[int, ...] | None = None,
    **provenance: Any,
) -> Path:
    """Write `array` as an artifact of `kind`, with the attrs that make it readable.

    Chiefly for tests and for postprocessors that persist their output; a producer running a model
    writes tile by tile and sets the same attrs itself. Validated by reading it straight back, so a
    producer cannot write something `open_artifact` will later reject.
    """
    path = Path(path)
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; valid kinds are {sorted(KINDS)}")
    spatial = _check_channels(kind, tuple(int(s) for s in array.shape))

    store = zarr.open(
        str(path), mode="w", shape=array.shape, dtype=array.dtype,
        chunks=chunks or tuple(min(256, s) for s in array.shape),
    )
    store[:] = array
    store.attrs.update(
        kind=kind,
        origin=list(origin or (0,) * len(spatial)),
        convention=convention,
        **({"background_id": int(background_id)} if background_id is not None else {}),
        **({"ignore_id": int(ignore_id)} if ignore_id is not None else {}),
        **provenance,
    )
    open_artifact(path)
    return path
