"""The two tasks: instance segmentation and semantic segmentation.

They differ in what their ground truth is, how it is read, and on which grid it is compared --
which is why they are two classes rather than two branches of one. Everything else about scoring
them is shared, and lives in the postprocess and metric registries.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from artifact import Artifact, Geometry

from .base import BaseTask, Volume
from .registry import TaskRegistry

#: The level a score must read. `s0` is native; a coarser rung changes what a voxel means and so
#: silently rescales every metric.
NATIVE_LEVEL = "s0"


def read_labels(
    volume: Volume, origin: tuple[int, ...], shape: tuple[int, ...], level: str = NATIVE_LEVEL
) -> np.ndarray:
    """A block of a volume's label array, at native resolution, as int64.

    Read straight from the store rather than through `miao`, deliberately: miao resamples labels
    through float32, which collapses large integer ids -- a MICrONS volume returns a median of one
    instance per patch where the store holds tens. That is tolerable for training targets and fatal
    for scoring, where the ids *are* the answer. When miao gains a nearest-neighbour label path this
    becomes a call into it.
    """
    if volume.label_key is None:
        raise ValueError(
            f"volume {volume.name!r} has no label_key, so there is no voxel ground truth to score "
            "against. Give it one, or use a task whose truth lives elsewhere (e.g. a skeleton)."
        )
    store = zarr.open(str(volume.path), mode="r")
    array = store[f"{volume.label_key}/{level}"]
    window: tuple[Any, ...] = tuple(slice(o, o + s) for o, s in zip(origin, shape, strict=True))
    if volume.fixed_axes:
        window = pinned_window(volume, level, window)
    return np.asarray(array[window]).astype(np.int64)


def pinned_window(volume: Volume, level: str, window: tuple[Any, ...]) -> tuple[Any, ...]:
    """`window` (spatial, storage order) as an index into a label array with pinned axes.

    A time series scored one frame at a time keeps its labels as t, c, z, y, x, so the spatial
    window alone would land on t, c and z. Where each pinned axis sits and which index it takes at
    this level come from miao's `fix_axes`, the resolution its sampler uses; a channel axis left
    over must hold one channel and is taken too, so the read is spatial like any other.
    """
    from miao.zarr_meta import fix_axes, read_ome_metadata

    assert volume.label_key is not None and volume.fixed_axes is not None  # checked by the caller
    meta = fix_axes(
        read_ome_metadata(volume.path, volume.label_key, volume.zarr_version),
        volume.fixed_axes, f"volume {volume.name!r}: ",
    )
    scale = next(s for s in meta.scales.values() if s.path == level)
    spatial = [name for name in meta.axis_names if name != "c"]
    if len(spatial) != len(window):
        raise ValueError(
            f"volume {volume.name!r}: pinning {volume.fixed_axes} leaves {volume.label_key} with "
            f"axes {''.join(spatial)!r}, but the region scored is {len(window)}-D"
        )
    remaining = iter(zip(meta.axis_names, scale.shape, strict=True))
    region = iter(window)
    index: list[Any] = []
    for dim in range(len(scale.fixed_index) + len(meta.axis_names)):
        if dim in scale.fixed_index:
            index.append(scale.fixed_index[dim])
            continue
        name, size = next(remaining)
        if name != "c":
            index.append(next(region))
        elif size == 1:
            index.append(0)
        else:
            raise ValueError(
                f"volume {volume.name!r}: {volume.label_key} holds {size} label channels; a score "
                "reads one labelling, so pin the channel with fixed_axes"
            )
    return tuple(index)


@TaskRegistry.register("instance_seg")
class InstanceSegmentation(BaseTask):
    """Score an instance labelling.

    `truth` is either a traced skeleton or a dense label array, chosen by `truth_kind`, because the
    two are not interchangeable and the metrics that consume them are different: a skeleton gives
    nERL and node-level VOI, a label array gives panoptic quality and voxel-level VOI. A task may
    not silently substitute one for the other -- their numbers are not comparable.
    """

    canonical = "instances"

    TRUTH_KINDS = ("skeleton", "instances", "instances_resampled")

    def __init__(self, truth_kind: str = "skeleton", skeleton_name: str = "skeleton.pkl",
                 **settings: Any) -> None:
        super().__init__(truth_kind=truth_kind, skeleton_name=skeleton_name, **settings)
        if truth_kind not in self.TRUTH_KINDS:
            raise ValueError(
                f"truth_kind must be one of {self.TRUTH_KINDS}, got {truth_kind!r}. A skeleton "
                "scores run length over traced paths; a label array scores voxel overlap. They "
                "measure different things and their numbers do not belong in one column."
            )
        self.truth_kind = truth_kind
        self.skeleton_name = skeleton_name

    def in_volume_frame(self) -> bool:
        return self.truth_kind != "instances_resampled"

    def region(
        self, volume: Volume, artifact: Artifact
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """For a resampled artifact, the region is the artifact's own extent.

        `bounding_box` in a data config counts *native* voxels of the source store, while a
        prediction resampled to the training resolution lives on a different lattice entirely --
        512 output voxels over 412 native ones, for `liconn_mouse_hippocampus`. Intersecting the two
        would compare coordinates in different units, which is arithmetic that succeeds and means
        nothing. The producer already clipped to the box; `covers_full_box` records whether the
        lattice reached its far edge.
        """
        if self.truth_kind == "instances_resampled":
            return artifact.origin, artifact.spatial_shape
        return super().region(volume, artifact)

    def truth_artifact_path(self, volume: Volume, artifact: Artifact) -> Path:
        """`<volume>.gt.zarr` beside the prediction."""
        return artifact.path.parent / f"{volume.name}.gt.zarr"

    def skeleton_path(self, volume: Volume) -> Path:
        """The volume's skeleton, relative to its store: `skeleton_name` with `{volume}` filled in.

        One file per store was enough for NISB (`skeleton.pkl` in each cube). A store scored as
        several volumes -- the zebrafinch regions, each with test and validation skeletons -- holds
        one file per volume, e.g. `skeletons/{volume}.pkl`.
        """
        return volume.path / self.skeleton_name.format(volume=volume.name)

    def truth_digest(self, volume: Volume) -> str | None:
        if self.truth_kind != "skeleton":
            return None
        path = self.skeleton_path(volume)
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def ground_truth(self, volume: Volume, artifact: Artifact) -> Any:
        if self.truth_kind == "instances_resampled":
            from artifact import open_artifact

            path = self.truth_artifact_path(volume, artifact)
            if not path.exists():
                raise FileNotFoundError(
                    f"no ground-truth artifact at {path}. `truth_kind = \"instances_resampled\"` "
                    "scores against the labelling the producer wrote on the prediction's own "
                    "grid -- mia-train's src/predict.py emits it beside each prediction, "
                    "and it cannot be reconstructed here without redoing that resampling."
                )
            truth = open_artifact(path)
            if truth.spatial_shape != artifact.spatial_shape:
                raise ValueError(
                    f"{path.name} is {truth.spatial_shape} but the prediction is "
                    f"{artifact.spatial_shape}; they were not written on the same grid"
                )
            for key in ("native_box", "read_shape", "image_level", "label_level"):
                if truth.attrs.get(key) != artifact.attrs.get(key):
                    raise ValueError(
                        f"{path.name} and {artifact.path.name} disagree about {key!r}: "
                        f"{truth.attrs.get(key)} vs {artifact.attrs.get(key)}. They describe "
                        "different regions, so one of them is stale -- re-run prediction."
                    )
            return truth.load()
        if self.truth_kind == "skeleton":
            path = self.skeleton_path(volume)
            if not path.is_file():
                raise FileNotFoundError(
                    f"volume {volume.name!r} has no skeleton at {path} (skeleton_name = "
                    f"{self.skeleton_name!r}, relative to the volume's store). Volumes with dense "
                    'voxel truth instead need truth_kind = "instances".'
                )
            return path
        origin, shape = self.region(volume, artifact)
        return read_labels(volume, origin, shape)


#: Slack for float round-off when a voxel centre is compared with a boundary, in voxels.
_EPS = 1e-6


def _ome_level0(volume: Volume, key: str) -> tuple[Geometry, tuple[int, ...], str]:
    """(geometry, shape, dataset path) of level 0 of a volume's OME group, spatial axes only.

    Read through miao, which composes the multiscale-level transform with the dataset's the way it
    does when it samples the same stores for training; a pinned axis (`fixed_axes`) is removed
    first, and so is a channel axis.
    """
    from miao.zarr_meta import read_ome_metadata

    meta = read_ome_metadata(volume.path, key, volume.zarr_version, [0])
    if volume.fixed_axes:
        # Imported only here: a miao older than `fixed_axes` has no `fix_axes`, and cannot load a
        # data config that pins an axis in the first place.
        from miao.zarr_meta import fix_axes

        meta = fix_axes(meta, volume.fixed_axes, f"volume {volume.name!r}: ")
    types = {str(axis.get("name")): axis.get("type") for axis in meta.axes}
    units = {str(axis.get("name")): axis.get("unit") for axis in meta.axes}
    level = meta.scales[0]
    spatial = [
        i for i, name in enumerate(meta.axis_names)
        if name != "c" and types.get(name) not in ("channel", "time")
    ]
    names = [meta.axis_names[i] for i in spatial]
    shift = level.translation_or_zeros()
    found = {units.get(name) for name in names}
    return (
        Geometry(
            axes="".join(names),
            voxel_size=tuple(float(level.scale_factors[i]) for i in spatial),
            translation=tuple(float(shift[i]) for i in spatial),
            unit=str(found.pop()) if len(found) == 1 and None not in found else None,
        ),
        tuple(int(level.shape[i]) for i in spatial),
        str(level.path),
    )


def _centres_inside(first: float, step: float, count: int, low: float, high: float
                    ) -> tuple[int, int]:
    """[i0, i1): the voxels of one axis whose centres, at `first + i * step` for i in [0, count),
    lie in [low, high)."""
    start = math.ceil((low - first) / step - _EPS)
    stop = math.ceil((high - first) / step - _EPS)
    return max(0, start), min(count, stop)


@dataclass(frozen=True)
class LabelGrid:
    """Where one artifact's voxels land on one volume's label array, per spatial axis.

    Every range is `[lo, hi)`. `annotated` and `region` count label voxels from the label array's
    first voxel; `window` counts the artifact's own voxels; `index[a][k]` is the window voxel whose
    extent holds the centre of region voxel `k` along axis `a`.
    """

    #: The label array, clipped to the volume's bounding box when it declares one.
    annotated: tuple[tuple[int, int], ...]
    #: `annotated`, clipped to the label voxels whose centres the artifact covers.
    region: tuple[tuple[int, int], ...]
    window: tuple[tuple[int, int], ...]
    index: tuple[np.ndarray, ...]


@TaskRegistry.register("semantic_seg")
class SemanticSegmentation(BaseTask):
    """Score a class labelling against a dense label array, on the label array's own grid.

    No `truth_kind`: a semantic score is per-voxel class agreement, and there is no skeleton
    analogue of it.

    **The truth's grid is the one scored on.** Semantic ground truth is usually painted on a grid
    of its own: a CellMap crop is a small array placed in its volume by an OME translation, and its
    voxels are half the size of the raw's. A prediction lives on whatever grid its producer chose --
    the raw's, or the resolution a model was trained at. So both are placed by their OME geometry
    (`artifact.Geometry`), and every label voxel takes the class of the prediction voxel that holds
    its centre: the prediction is upsampled onto the truth, never the truth downsampled onto the
    prediction. Truth painted at 2 nm is then scored at 2 nm whatever resolution a model ran at,
    and two rows that predicted on different grids score the same voxels, which is what lets them
    share a table. A label voxel centre that falls exactly on a boundary between two prediction
    voxels takes the upper one.

    The scored region is counted in label voxels from the label array's first voxel: the label
    array, clipped to the volume's `bounding_box` (image level-0 voxels, as everywhere) when one is
    declared, and to the label voxels the artifact covers. `whole_region` is whether the artifact
    covered all of it. An artifact must therefore declare OME geometry -- a single-level OME group,
    as mia-train's predict.py writes -- with the label's spatial axes, in the same order.
    """

    canonical = "classes"

    def __init__(self, **settings: Any) -> None:
        super().__init__(**settings)
        self._grids: dict[tuple[str, str, str], LabelGrid] = {}

    def in_volume_frame(self) -> bool:
        return False

    def place(self, artifact: Artifact) -> Artifact:
        """Nothing to move: an artifact is placed by its OME geometry, which it must declare."""
        if artifact.geometry is None:
            raise ValueError(
                f"{artifact.path} declares no OME geometry (a single-level OME-Zarr group whose "
                "`multiscales` give its voxel size and translation). A semantic task scores on the "
                "label array's own grid and places the prediction on it by that geometry; without "
                "it there is no saying which label voxels a prediction voxel covers."
            )
        return artifact

    def grid(self, volume: Volume, artifact: Artifact) -> LabelGrid:
        """How `artifact` lands on `volume`'s label array; computed once per pair."""
        key = (volume.name, str(volume.path), str(artifact.path))
        if key not in self._grids:
            self._grids[key] = self._resolve(volume, artifact)
        return self._grids[key]

    def _resolve(self, volume: Volume, artifact: Artifact) -> LabelGrid:
        if volume.label_key is None:
            raise ValueError(f"volume {volume.name!r} has no label_key: no truth to score against")
        predicted = self.place(artifact).geometry
        assert predicted is not None                           # place() refuses a missing one
        label, label_shape, level = _ome_level0(volume, volume.label_key)
        if level != NATIVE_LEVEL:
            raise ValueError(
                f"volume {volume.name!r}: {volume.label_key}'s first level is {level!r}, but the "
                f"truth is read from {NATIVE_LEVEL!r}"
            )
        if predicted.axes != label.axes:
            raise ValueError(
                f"{artifact.path} stores axes {predicted.axes!r} but volume {volume.name!r}'s "
                f"labels are {label.axes!r}. Scoring across that permutation is not implemented, "
                "and guessing it would compare every voxel with the wrong one."
            )
        if predicted.unit and label.unit and predicted.unit != label.unit:
            raise ValueError(
                f"{artifact.path} is placed in {predicted.unit} but volume {volume.name!r}'s "
                f"labels in {label.unit}"
            )
        annotated = [(0, n) for n in label_shape]
        if volume.bounding_box is not None:
            image, _, _ = _ome_level0(volume, volume.image_key)
            if image.axes != label.axes:
                raise ValueError(
                    f"volume {volume.name!r}: image axes {image.axes!r}, label axes {label.axes!r}"
                )
            annotated = [
                _centres_inside(t, s, n, ti + (lo - 0.5) * si, ti + (hi - 0.5) * si)
                for t, s, n, ti, si, (lo, hi) in zip(
                    label.translation, label.voxel_size, label_shape, image.translation,
                    image.voxel_size, volume.bounding_box, strict=True,
                )
            ]
        if any(hi <= lo for lo, hi in annotated):
            raise ValueError(
                f"volume {volume.name!r}: its label array {volume.label_key} and its bounding_box "
                f"{volume.bounding_box} share no voxel"
            )
        region, window, index = [], [], []
        for axis, (t, s, n) in enumerate(zip(label.translation, label.voxel_size, label_shape,
                                             strict=True)):
            tp, sp = predicted.translation[axis], predicted.voxel_size[axis]
            count = artifact.spatial_shape[axis]
            covered = _centres_inside(t, s, n, tp - 0.5 * sp, tp + (count - 0.5) * sp)
            lo, hi = max(annotated[axis][0], covered[0]), min(annotated[axis][1], covered[1])
            if hi <= lo:
                raise ValueError(
                    f"volume {volume.name!r}: {artifact.path} covers no annotated label voxel on "
                    f"axis {label.axes[axis]!r} (prediction voxels centred from {tp} every {sp}, "
                    f"{count} of them; annotated label voxels {annotated[axis]} centred from {t} "
                    f"every {s}). Predict over the annotated region."
                )
            centres = t + np.arange(lo, hi, dtype=np.float64) * s
            held = np.floor((centres - tp) / sp + 0.5 + _EPS).astype(np.int64)
            held = np.clip(held, 0, count - 1)
            region.append((lo, hi))
            window.append((int(held[0]), int(held[-1]) + 1))
            index.append(held - held[0])
        return LabelGrid(tuple(annotated), tuple(region), tuple(window), tuple(index))

    def region(self, volume: Volume, artifact: Artifact) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The label voxels scored, from the label array's first voxel; see the class docstring."""
        found = self.grid(volume, artifact).region
        return tuple(lo for lo, _ in found), tuple(hi - lo for lo, hi in found)

    def read_window(
        self, volume: Volume, artifact: Artifact
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The prediction voxels holding the region's label-voxel centres, for `Artifact.read`."""
        found = self.grid(volume, artifact).window
        return (
            tuple(o + lo for o, (lo, _) in zip(artifact.origin, found, strict=True)),
            tuple(hi - lo for lo, hi in found),
        )

    def align(self, prediction: np.ndarray, volume: Volume, artifact: Artifact) -> np.ndarray:
        """Each region label voxel takes the class of the prediction voxel holding its centre."""
        found = self.grid(volume, artifact)
        expected = tuple(hi - lo for lo, hi in found.window)
        if tuple(prediction.shape) != expected:
            raise ValueError(
                f"post-processing {artifact.path} returned shape {prediction.shape}, but the "
                f"window read was {expected}"
            )
        return np.asarray(prediction)[np.ix_(*found.index)]

    def context(self, volume: Volume, artifact: Artifact) -> dict[str, Any]:
        """As `BaseTask.context`; whole means the artifact covered every annotated label voxel."""
        found = self.grid(volume, artifact)
        origin, shape = self.region(volume, artifact)
        return {
            "origin": origin,
            "shape": shape,
            "whole_region": found.region == found.annotated,
            "ignore_id": artifact.ignore_id,
            "background_id": 0 if artifact.background_id is None else artifact.background_id,
            "volume": volume.name,
            "axes": artifact.axes,
        }

    def ground_truth(self, volume: Volume, artifact: Artifact) -> Any:
        origin, shape = self.region(volume, artifact)
        return read_labels(volume, origin, shape)
