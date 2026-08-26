"""The two tasks: instance segmentation and semantic segmentation.

They differ in exactly one thing -- what their ground truth is and how it is read -- which is why
they are two small classes rather than two branches of one. Everything else about scoring them is
shared, and lives in the postprocess and metric registries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import zarr

from artifact import Artifact

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
    window = tuple(slice(o, o + s) for o, s in zip(origin, shape, strict=True))
    return np.asarray(array[window]).astype(np.int64)


@TaskRegistry.register("instance_seg")
class InstanceSegmentation(BaseTask):
    """Score an instance labelling.

    `truth` is either a traced skeleton or a dense label array, chosen by `truth_kind`, because the
    two are not interchangeable and the metrics that consume them are different: a skeleton gives
    nERL and node-level VOI, a label array gives panoptic quality and voxel-level VOI. A task may
    not silently substitute one for the other -- their numbers are not comparable.
    """

    canonical = "instances"

    def __init__(self, truth_kind: str = "skeleton", skeleton_name: str = "skeleton.pkl",
                 **settings: Any) -> None:
        super().__init__(truth_kind=truth_kind, skeleton_name=skeleton_name, **settings)
        if truth_kind not in ("skeleton", "labels"):
            raise ValueError(
                f"truth_kind must be 'skeleton' or 'labels', got {truth_kind!r}. A skeleton scores "
                "run length over traced paths; a label array scores voxel overlap. They measure "
                "different things and their numbers do not belong in one column."
            )
        self.truth_kind = truth_kind
        self.skeleton_name = skeleton_name

    def ground_truth(self, volume: Volume, artifact: Artifact) -> Any:
        if self.truth_kind == "skeleton":
            path = volume.path / self.skeleton_name
            if not path.is_file():
                raise FileNotFoundError(
                    f"volume {volume.name!r} has no {self.skeleton_name} at {path}. NISB cubes "
                    "carry one inside the zarr group; volumes with dense voxel truth instead need "
                    'truth_kind = "labels".'
                )
            return path
        origin, shape = self.region(volume, artifact)
        return read_labels(volume, origin, shape)


@TaskRegistry.register("semantic_seg")
class SemanticSegmentation(BaseTask):
    """Score a class labelling against a dense label array.

    No `truth_kind`: a semantic score is per-voxel class agreement, and there is no skeleton
    analogue of it.
    """

    canonical = "classes"

    def ground_truth(self, volume: Volume, artifact: Artifact) -> Any:
        origin, shape = self.region(volume, artifact)
        return read_labels(volume, origin, shape)
