"""What a task owns: reading ground truth, and the context a metric may not assume.

Postprocessors interpret predictions and metrics compare labellings. Neither knows where the truth
comes from, and that is deliberate -- a task is the only thing that touches the data. It exists
because ground truth for one canonical form still arrives in incompatible shapes: NISB's instance
truth is a traced skeleton pickle, the corpus' instance truth is a dense label array in an
OME-NGFF store, and semantic truth is a label array with a class vocabulary.

`context` carries what a metric needs but must not fetch for itself -- `ignore_id`, `background_id`,
the region's `origin`, whether the region is the whole annotated extent. Passed in rather than
looked up so no metric ever reaches back into an artifact or a config, which is what keeps a metric
testable with two arrays and nothing else.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from artifact import CANONICAL_FORMS, Artifact


@dataclass(frozen=True)
class Volume:
    """One scoreable region: where the image is, where its truth is, and which part is annotated.

    Mirrors a `miao` volume entry, because that is where these come from. `bounding_box` is
    `[[lo, hi], ...]` in level-0 voxels in the data config's spatial order, and it is **not** a
    speed crop: several volumes in the corpus annotate an offset sub-box, and LICONN dentate gyrus
    and hippocampus have 0% ground-truth foreground in their own central 256^3 block. Scoring
    without the box counts unannotated space as background and returns a plausible number that
    means nothing.

    `fixed_axes` pins stored axes to one index each, `{"t": 10}` for one frame of a time series
    (miao's setting of the same name). The volume is then that frame, spatial like any other.
    """

    name: str
    path: Path
    label_key: str | None = None
    image_key: str = "raw"
    bounding_box: tuple[tuple[int, int], ...] | None = None
    zarr_version: str = "zarr3"
    fixed_axes: dict[str, int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def origin(self) -> tuple[int, ...] | None:
        return None if self.bounding_box is None else tuple(low for low, _ in self.bounding_box)

    @property
    def shape(self) -> tuple[int, ...] | None:
        return (
            None if self.bounding_box is None
            else tuple(high - low for low, high in self.bounding_box)
        )


class BaseTask(abc.ABC):
    """A benchmark task: the canonical form it scores, and how to read its ground truth."""

    #: "instances" or "classes". A postprocessor producing the other form is a config error.
    canonical: str = ""

    def __init__(self, **settings: Any) -> None:
        self.settings = settings
        if self.canonical not in CANONICAL_FORMS:
            raise ValueError(
                f"{type(self).__name__}.canonical must be one of {CANONICAL_FORMS}, "
                f"got {self.canonical!r}"
            )

    @abc.abstractmethod
    def ground_truth(self, volume: Volume, artifact: Artifact) -> Any:
        """This volume's truth over the region the artifact covers.

        Whatever the task's metrics expect: an array for a voxel metric, a path for a skeleton one.
        """

    def in_volume_frame(self) -> bool:
        """Whether this task's truth and boxes count the source volume's own voxels.

        True unless a task scores against truth written on the prediction's own lattice instead
        (`instance_seg` with `truth_kind = "instances_resampled"`), where the region is simply the
        artifact's extent and no position in the volume is needed.
        """
        return True

    def place(self, artifact: Artifact) -> Artifact:
        """The artifact at its position in the volume's own voxels, where boxes and truth count.

        An artifact says where its first voxel sits with `origin`. mia-train's `predict.py` writes
        `origin = [0, 0, 0]` and records the region it covered separately, as `native_box`, in the
        store's level-0 voxels. On the store's own lattice (`scale` 1 on every axis) that box's low
        corner *is* the first voxel's position, so the artifact is moved there: taken at its word
        it would be scored as though it started at voxel 0, every lookup landing on a real voxel --
        the wrong one -- and nothing raising. A prediction resampled to another lattice has no such
        position, since its voxels are not the store's, and is refused. An artifact without
        `native_box` (FFN's, a submitted segmentation) is taken at its word.
        """
        box = artifact.attrs.get("native_box")
        if box is None or not self.in_volume_frame():
            return artifact
        scale = [float(s) for s in artifact.attrs.get("scale") or ()]
        if len(scale) != len(box) or any(s != 1.0 for s in scale):
            raise ValueError(
                f"{artifact.path} was predicted on a lattice resampled from the store's (scale "
                f"{scale}), but {type(self).__name__} counts the volume's own voxels: its boxes "
                "and skeleton nodes have no position on that lattice. Predict at the store's "
                "resolution."
            )
        extent = [int(high) - int(low) for low, high in box]
        if extent != list(artifact.spatial_shape):
            raise ValueError(
                f"{artifact.path}: native_box {box} spans {extent} voxels, the array "
                f"{list(artifact.spatial_shape)}"
            )
        return replace(artifact, origin=tuple(int(low) for low, _ in box))

    def region(self, volume: Volume, artifact: Artifact) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The (origin, shape) actually scored: the artifact's extent, clipped to the annotation.

        The intersection rather than either one alone. The artifact may cover a block of a larger
        volume, and the annotation may cover a sub-box of it; scoring their union would include
        voxels with no truth, and scoring only the box would fail when the artifact is smaller.
        """
        art_low = np.asarray(artifact.origin)
        art_high = art_low + np.asarray(artifact.spatial_shape)
        if volume.bounding_box is None:
            return tuple(int(v) for v in art_low), tuple(int(v) for v in art_high - art_low)

        box_low = np.asarray(volume.origin)
        box_high = box_low + np.asarray(volume.shape)
        low = np.maximum(art_low, box_low)
        high = np.minimum(art_high, box_high)
        if np.any(high <= low):
            raise ValueError(
                f"volume {volume.name!r}: the artifact covers "
                f"[{art_low.tolist()}, {art_high.tolist()}) but its annotation covers "
                f"[{box_low.tolist()}, {box_high.tolist()}); they do not overlap, so there is "
                "nothing to score. Predict over the annotated region."
            )
        return tuple(int(v) for v in low), tuple(int(v) for v in (high - low))

    def context(self, volume: Volume, artifact: Artifact) -> dict[str, Any]:
        """Everything a metric may need about this region, resolved by the task.

        `whole_region` is never claimed without evidence. It previously read

            whole = volume.bounding_box is None or (origin == ... and shape == ...)

        which treats "this data config declares no bounding box" as "the artifact covers the whole
        volume". Those are different claims: the absence of a box says the volume is *annotated*
        throughout, and says nothing about how much of it the *artifact* covers. NISB cubes are
        fully annotated and so declare no box, so a 512^3 prediction over one cube reported
        `whole_region = True` -- which made `skeleton_erl` hand `funlib` an uncropped 784,783-node
        skeleton in absolute coordinates and raise `IndexError: index 865 is out of bounds for axis
        2 with size 512`.

        The crash was the mild half. `whole_region` is also what the leaderboard groups on, and nERL
        is not comparable across extents (0.3045 over a whole cube against 0.4192 on a 512^3 block
        of it), so the same bug could publish a sub-region score labelled as a whole-cube one and
        rank it against genuine whole-cube numbers.

        The artifact's own extent cannot in general be checked against the source store, because a
        data config's `bounding_box` counts native voxels while a resampled prediction lives on a
        different lattice -- see `region()`. So the producer declares it, and absent a declaration
        the answer is no: under-claiming crops a skeleton that did not need cropping (a no-op, since
        every node is inside) and labels a row as a sub-region, while over-claiming corrupts a
        published number. A prediction `place` has positioned on the store's own lattice is the
        exception: its extent is in the box's units, so the region is compared with the box, and
        the producer's `covers_full_box` -- which describes the box it was *predicted* over, often
        the scored one plus a margin -- is not consulted.
        """
        origin, shape = self.region(volume, artifact)
        placed = self.in_volume_frame() and "native_box" in artifact.attrs
        declared = None if placed else artifact.attrs.get("covers_full_box")
        if declared is not None:
            whole = bool(declared)
        elif volume.bounding_box is not None:
            whole = (origin == tuple(volume.origin or ())
                     and shape == tuple(volume.shape or ()))
        else:
            whole = False
        return {
            "origin": origin,
            "shape": shape,
            "whole_region": whole,
            "ignore_id": artifact.ignore_id,
            "background_id": 0 if artifact.background_id is None else artifact.background_id,
            "volume": volume.name,
            "axes": artifact.axes,
        }

    def truth_digest(self, volume: Volume) -> str | None:
        """A content hash of this volume's ground truth, when it is one file; None otherwise.

        Recorded per volume and compared as part of the task's identity, so ground truth that was
        rebuilt differently cannot join a table scored against the old one. A label array is not
        hashed (terabytes, and its store path and key already name it); a skeleton file is.
        """
        return None
