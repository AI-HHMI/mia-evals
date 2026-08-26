"""The postprocessors that need no post-processing, and the one that only needs an argmax.

Both exist so that a model which already emits a finished answer is an ordinary submission rather
than a special case. `identity` is what makes a segmentation someone sends us scoreable with no
model, no checkpoint and no validation split: its search space is a single empty candidate, so
there is nothing to fit and nothing to fit it on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BasePostprocess
from .registry import PostprocessRegistry


@PostprocessRegistry.register("identity")
class Identity(BasePostprocess):
    """A labelling, already. Passed through as int64 and otherwise untouched.

    Accepts both labelling kinds and yields whichever form it was given, so one entry serves an
    instance submission and a semantic one.
    """

    accepts = ("instances", "class_labels")
    produces = "same"

    def __call__(self, array: np.ndarray, **params: Any) -> np.ndarray:
        if array.dtype.kind == "f":
            raise ValueError(
                f"identity was handed a floating-point array (dtype {array.dtype}); a labelling "
                "must be integral, or the ids are not ids. Check the artifact's `kind`."
            )
        return array.astype(np.int64, copy=False)


@PostprocessRegistry.register("argmax")
class Argmax(BasePostprocess):
    """(K, *spatial) class scores -> (*spatial) class labelling.

    Correct only where the classes are mutually exclusive per voxel, which is what a single label
    array on disk means. Overlapping or hierarchical classes -- an organelle membrane inside its own
    lumen -- need independent per-class thresholds and a per-class metric instead, because an argmax
    forces one winner and a single confusion matrix cannot represent a voxel belonging to two
    classes. That is a different postprocessor, not a flag on this one.
    """

    accepts = ("class_scores",)
    produces = "classes"

    def __call__(self, array: np.ndarray, **params: Any) -> np.ndarray:
        if array.ndim < 2:
            raise ValueError(f"argmax needs a leading class axis, got shape {array.shape}")
        return np.asarray(array.argmax(axis=0)).astype(np.int64)


@PostprocessRegistry.register("per_class_threshold")
class PerClassThreshold(BasePostprocess):
    """(K, *spatial) independent per-class scores -> a labelling, highest scoring class over its own
    threshold.

    For scores trained with a per-class sigmoid rather than a softmax, where "no class is confident
    here" is a possible answer and an argmax cannot express it. One threshold is fitted for every
    class at once, from `thresholds`; a genuinely per-class fit is K independent sweeps and is not
    what this does, which is why the candidates are named `threshold` and not `thresholds`.
    """

    accepts = ("class_scores",)
    produces = "classes"

    def __init__(self, thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
                 background: int = 0, **settings: Any) -> None:
        super().__init__(thresholds=thresholds, background=background, **settings)
        self.thresholds = tuple(float(t) for t in thresholds)
        self.background = int(background)

    def search_space(self) -> list[dict[str, Any]]:
        return [{"threshold": t} for t in self.thresholds]

    def __call__(self, array: np.ndarray, **params: Any) -> np.ndarray:
        threshold = float(params["threshold"])
        best = np.asarray(array.argmax(axis=0)).astype(np.int64)
        confident = np.asarray(array.max(axis=0)) >= threshold
        return np.where(confident, best, self.background)
