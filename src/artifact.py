"""The prediction artifact: the one interface between whatever produced a prediction and scoring.

This is the whole reason `mia-evals` never imports a model. A producer writes an array plus the
attrs below; everything downstream reads only that. A `mia-train` checkpoint, a collaborator's
segmentation, and a baseline from a paper therefore all enter by the same door, and no code path
here is privileged for "our own" models.

    <artifact>.zarr        (C, *spatial) or (*spatial), any dtype -- a bare array, or a single-level
                           OME-Zarr group whose one dataset (`s0`) is that array (the layout
                           `mia-train`'s `predict.py` writes since 2026-09-15: the group's OME
                           `multiscales` carry the voxel size and physical offset, so a viewer
                           places the prediction on the raw volume; the group's other attrs are
                           the artifact's attrs, repeated on the array)
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

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zarr

# What a producer may declare, and what each kind promises about its array. `channels` is the
# leading axis: an int fixes it, "rank|rank*2" means one or two per spatial axis (the short-range
# affinity offsets alone, or short- then long-range), and None means there is no channel axis at
# all. Which channel counts a post-processor can use is its own business (`check_artifact`): `mws`
# needs all of them, `cc_threshold`, `mws3` and `ws_agglo` read only the short-range half.
#
# `canonical` is what postprocessing this kind must produce, and it is the narrow waist of the
# whole design: metrics attach to the canonical form, never to the kind, so nERL neither knows nor
# cares whether the labelling came from affinities, a watershed, or a file someone sent us.
KINDS: dict[str, dict[str, Any]] = {
    "affinity":     {"channels": "rank|rank*2", "canonical": "instances", "floating": True},
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
    #: The array inside a single-level OME-Zarr group, or None when `path` is the array itself.
    array_path: Path | None = None
    #: Spatial axis names in storage order, e.g. "zyx", from the group's OME `multiscales`; None for
    #: a bare array, which declares none. A skeleton states the order its node positions count in,
    #: and the two are compared before any node is looked up: transposed, every lookup still lands
    #: on a real voxel and the score is plausible nonsense.
    axes: str | None = None

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
        return zarr.open(str(self.array_path or self.path), mode="r")

    def read(self, origin: tuple[int, ...] | None = None,
             size: tuple[int, ...] | None = None,
             channels: int | None = None) -> np.ndarray:
        """A block, in coordinates *absolute to the source volume* rather than to this array.

        Absolute, because a caller holding a bounding box from a data config has it in volume
        coordinates and has no reason to know that this artifact covers a sub-region. Subtracting
        `origin` at each callsite instead is the kind of arithmetic that is wrong once and then
        wrong quietly -- an off-by-`origin` read returns real data from the wrong place.
        """
        store = self._store()
        limit = None if channels is None or self.channels is None else min(channels, self.channels)
        if origin is None and size is None:
            return np.asarray(store[:] if limit is None else store[:limit])
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
        if self.channels is None:
            return np.asarray(store[window])
        # Reading only the channels the consumer declared. The alternative -- read all of them and
        # slice afterwards -- allocates the full array first, which for a 7-gigavoxel six-channel
        # affinity artifact is 85 GB rather than 42 GB.
        leading = slice(None) if limit is None else slice(0, limit)
        return np.asarray(store[(leading, *window)])

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
    if expected == "rank|rank*2" and channels not in (len(spatial), 2 * len(spatial)):
        raise ValueError(
            f"kind='affinity' over {len(spatial)} spatial axes needs {2 * len(spatial)} channels "
            f"(short-range then long-range, one per axis) or {len(spatial)} (the short-range ones "
            f"only), got {channels}"
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
    array_path: Path | None = None
    attrs: dict[str, Any]
    axes: str | None = None
    if hasattr(store, "shape"):
        attrs = dict(store.attrs)
    else:
        axes = spatial_axes(dict(store.attrs))
        level = single_level(store)
        if level is None:
            # A zarr *group* with several (or no) levels, most likely a multiscale OME-Zarr
            # pyramid. Worth its own message rather than the `AttributeError: 'Group' object has
            # no attribute 'shape'` this used to raise: the source volumes read through `miao`
            # *are* multiscale OME-Zarr, so handing one to the scorer is the natural mistake, and
            # the fix is to name a level.
            levels = sorted(str(k) for k in store.keys()) if hasattr(store, "keys") else []
            hint = f" Name one level, for example {path.name}/{levels[0]}." if levels else ""
            raise ValueError(
                f"{path} is a zarr group, not an array. A prediction artifact is a "
                f"single-resolution array -- bare, or the one dataset of a single-level OME-Zarr "
                f"group -- because scoring compares one voxel lattice against the ground truth "
                f"and a multiscale pyramid does not say which level that is.{hint}"
            )
        array_path = path / level
        group_attrs = {k: v for k, v in dict(store.attrs).items() if k != "ome"}
        store = store[level]
        attrs = {**group_attrs, **dict(store.attrs)}

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

    # `floating` was declared per kind but never checked, so a float array could be declared as a
    # labelling and only fail later, inside a postprocessor. Checking it here matters more than the
    # late error suggests: float32 holds integers exactly only up to 2**24, so a labelling stored
    # as float32 silently merges distinct ids -- and the array still reads back as a plausible
    # labelling with a plausible object count.
    if not KINDS[kind]["floating"] and np.issubdtype(np.dtype(store.dtype), np.floating):
        raise ValueError(
            f"{path} is kind={kind!r}, which is a labelling, but its dtype is {store.dtype}. Ids "
            "must be an integer type: float32 represents integers exactly only below 2**24, so "
            "larger ids collapse into one another and the result still looks like a valid "
            "labelling. Write it as an integer dtype instead of casting at read time."
        )

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
        array_path=array_path,
        axes=axes,
    )


def spatial_axes(group_attrs: dict[str, Any]) -> str | None:
    """"zyx"-style names of an OME group's spatial axes, in storage order; None if it names none."""
    ome = group_attrs.get("ome") if isinstance(group_attrs.get("ome"), dict) else group_attrs
    scales = ome.get("multiscales") if isinstance(ome, dict) else None
    if not isinstance(scales, list) or not scales or not isinstance(scales[0], dict):
        return None
    names = [
        str(axis.get("name")) for axis in scales[0].get("axes") or []
        if isinstance(axis, dict) and axis.get("type") != "channel"
    ]
    return "".join(names) if names and all(len(n) == 1 for n in names) else None


class LazyLabelling:
    """A stored labelling over one region, read only where a metric looks at it.

    `score_once` hands this, instead of an array, to metrics that need the labelling at a few
    points -- a skeleton's nodes -- when the post-processor leaves the stored values untouched
    (`identity`). The zebrafinch benchmark region is 478 gigavoxels, 3.8 TB as uint64, while its
    431,659 skeleton nodes touch a few thousand chunks. `lookup` reads each of those chunks once;
    anything that would need the whole array (`np.asarray`) is refused, so pairing this with a voxel
    metric fails at once instead of trying to allocate terabytes.
    """

    def __init__(self, artifact: Artifact, origin: tuple[int, ...], shape: tuple[int, ...],
                 workers: int = 16) -> None:
        if not artifact.is_labelling:
            raise ValueError(f"{artifact.path} is kind={artifact.kind!r}, not a labelling")
        self.artifact = artifact
        self.origin = tuple(int(o) for o in origin)
        self.shape = tuple(int(s) for s in shape)
        self.ndim = len(self.shape)
        self.workers = int(workers)
        #: Region-local voxel + this = artifact-local voxel.
        self._offset = np.asarray(self.origin, dtype=np.int64) - np.asarray(artifact.origin)
        high = self._offset + np.asarray(self.shape)
        if np.any(self._offset < 0) or np.any(high > np.asarray(artifact.spatial_shape)):
            raise ValueError(
                f"region at {self.origin} of shape {self.shape} is not inside {artifact.path}, "
                f"which covers origin {artifact.origin}, shape {artifact.spatial_shape}"
            )
        self._local = threading.local()

    def _store(self) -> Any:
        store = getattr(self._local, "store", None)
        if store is None:
            store = self._local.store = self.artifact._store()
        return store

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._store().dtype)

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise TypeError(
            f"refusing to materialise the lazily scored labelling {self.artifact.path} "
            f"({int(np.prod(self.shape)):,} voxels): only point lookups are supported"
        )

    def lookup(self, points: np.ndarray) -> np.ndarray:
        """The label at each region-local integer point, `(N, ndim)`; each chunk is read once."""
        points = np.asarray(points, dtype=np.int64).reshape(-1, self.ndim)
        if points.shape[0] == 0:
            return np.zeros(0, dtype=self.dtype)
        if np.any(points < 0) or np.any(points >= np.asarray(self.shape)):
            raise IndexError(f"points fall outside the region of shape {self.shape}")
        local = points + self._offset
        store = self._store()
        chunks = np.asarray(store.chunks, dtype=np.int64)
        extent = np.asarray(store.shape, dtype=np.int64)
        keys = local // chunks
        order = np.lexsort(keys.T[::-1])
        ordered = keys[order]
        starts = np.flatnonzero(np.r_[True, np.any(np.diff(ordered, axis=0) != 0, axis=1)])
        ends = np.r_[starts[1:], len(order)]
        out = np.empty(points.shape[0], dtype=store.dtype)

        def read(run: int) -> None:
            low = ordered[starts[run]] * chunks
            high = np.minimum(low + chunks, extent)
            window = tuple(slice(a, b) for a, b in zip(low, high, strict=True))
            block = np.asarray(self._store()[window])
            members = order[starts[run]:ends[run]]
            out[members] = block[tuple((local[members] - low).T)]

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(read, range(len(starts))))
        return out


def write_scored(
    path: str | Path,
    labels: np.ndarray,
    like: Artifact,
    origin: tuple[int, ...],
    **attrs: Any,
) -> Path:
    """Persist a post-processed labelling of `like` -- the exact voxels a row was scored on.

    The scorer applies the postprocessor in memory and throws the result away, so what is on disk
    is the producer's output before the size filter, and a viewer shows something other than
    what the number was computed on. This writes the labelling as a single-level OME-Zarr group
    with `like`'s geometry (the same voxel size; the translation shifted when the scored region
    starts inside the artifact), as the narrowest unsigned type that holds its ids.
    """
    path = Path(path)
    if labels.ndim != len(like.spatial_shape):
        raise ValueError(f"a scored labelling must be spatial, got shape {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"a labelling must be integer, got {labels.dtype}")
    if labels.size and int(labels.min()) < 0:
        raise ValueError("a labelling with negative ids cannot be written as unsigned")
    largest = int(labels.max()) if labels.size else 0
    labels = labels.astype(np.uint32 if largest < 2**32 else np.uint64, copy=False)

    payload: dict[str, Any] = {
        "kind": "instances", "background_id": 0, "origin": list(origin),
        "source_artifact": str(like.path), **attrs,
    }
    chunks = tuple(min(256, s) for s in labels.shape)
    ome = _shifted_ome(like, origin) if like.array_path is not None else None
    if ome is None:
        # The source carries no geometry to inherit: a bare array, like the source itself.
        store = zarr.open(str(path), mode="w", shape=labels.shape, dtype=labels.dtype,
                          chunks=chunks)
        store[:] = labels
        store.attrs.update(**payload)
    else:
        group = zarr.open_group(str(path), mode="w", zarr_format=3)
        level = group.create_array(name="s0", shape=labels.shape, dtype=labels.dtype,
                                   chunks=chunks)
        level[:] = labels
        group.attrs.update(ome=ome, **payload)
        level.attrs.update(**payload)
    open_artifact(path)
    return path


def _shifted_ome(like: Artifact, origin: tuple[int, ...]) -> dict[str, Any] | None:
    """`like`'s OME multiscales with the translation moved to the scored region's first voxel."""
    import copy

    root = zarr.open(str(like.path), mode="r")
    ome = copy.deepcopy(dict(root.attrs).get("ome"))
    if not isinstance(ome, dict) or len(ome.get("multiscales") or []) != 1:
        return None
    scales = ome["multiscales"][0]
    axes = [a for a in scales.get("axes", []) if a.get("type") != "channel"]
    (dataset,) = scales["datasets"]
    scale = shift = None
    for t in dataset.get("coordinateTransformations", []):
        if t["type"] == "scale":
            scale = t
        elif t["type"] == "translation":
            shift = t
    if scale is None:
        return None
    if shift is None:
        shift = {"type": "translation", "translation": [0.0] * len(scale["scale"])}
        dataset["coordinateTransformations"].append(shift)
    # Drop a channel axis: the scored labelling has none.
    if len(scale["scale"]) == len(axes) + 1:
        scales["axes"] = axes
        scale["scale"] = scale["scale"][1:]
        shift["translation"] = shift["translation"][1:]
    offset = [o - a for o, a in zip(origin, like.origin, strict=True)]
    shift["translation"] = [
        float(t) + float(o) * float(v)
        for t, o, v in zip(shift["translation"], offset, scale["scale"], strict=True)
    ]
    return ome


def single_level(group: Any) -> str | None:
    """The one dataset path of a single-level OME-Zarr group, else None.

    Only OME `multiscales` metadata with exactly one dataset qualifies: it names the lattice
    unambiguously. A group without that metadata, or with a pyramid, is not an artifact.
    """
    attrs = dict(group.attrs)
    ome = attrs.get("ome") if isinstance(attrs.get("ome"), dict) else attrs
    scales = ome.get("multiscales") if isinstance(ome, dict) else None
    if not isinstance(scales, list) or len(scales) != 1:
        return None
    datasets = scales[0].get("datasets") if isinstance(scales[0], dict) else None
    if not isinstance(datasets, list) or len(datasets) != 1:
        return None
    level = datasets[0].get("path")
    if not isinstance(level, str) or level not in group:
        return None
    return level


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
