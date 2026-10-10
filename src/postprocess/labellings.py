"""The postprocessors that need no post-processing, and the one that only needs an argmax.

Both exist so that a model which already emits a finished answer is an ordinary submission rather
than a special case. `identity` is what makes a segmentation someone sends us scoreable with no
model, no checkpoint and no validation split: by default its search space is a single empty
candidate, so there is nothing to fit and nothing to fit it on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BasePostprocess
from .registry import PostprocessRegistry
from .size_filter import (
    FillBases,
    drop_small_components,
    fingerprint,
    parse_fill_distances,
    parse_min_sizes,
    with_fill,
)


@PostprocessRegistry.register("identity")
class Identity(BasePostprocess):
    """A labelling, already: scored as it is, or through the add-ons every instance route takes.

    Accepts both labelling kinds and yields whichever form it was given, so one entry serves an
    instance submission and a semantic one. `min_sizes` and `fill_distances` are the add-ons
    `cc_threshold` and `mws` take after producing their labelling -- drop components below a voxel
    count, then grow the survivors into the background within a distance -- swept jointly, fill
    innermost, and fitted on [data.fit]. With either set it takes instance labellings only: a class
    region is not a component, and neither add-on has a reading there. The defaults, `[0]`, add
    nothing, so a plain identity route has nothing to fit and its records read as they always have.
    """

    accepts: tuple[str, ...] = ("instances", "class_labels")
    produces = "same"

    def __init__(
        self,
        min_sizes: tuple[int, ...] | list[int] = (0,),
        fill_distances: tuple[int | str, ...] | list[int | str] = (0,),
        **settings: Any,
    ) -> None:
        super().__init__(min_sizes=min_sizes, fill_distances=fill_distances, **settings)
        self.min_sizes = parse_min_sizes(min_sizes, "identity")
        self.fill_distances = parse_fill_distances(fill_distances, "identity")
        if self.min_sizes != (0,) or self.fill_distances != (0,):
            self.accepts = ("instances",)
        self._fill_bases = FillBases()

    def search_space(self) -> list[dict[str, Any]]:
        if self.min_sizes == (0,) and self.fill_distances == (0,):
            return [{}]
        return with_fill([{"min_size": m} for m in self.min_sizes], self.fill_distances)

    def __call__(self, array: np.ndarray, **params: Any) -> np.ndarray:
        if array.dtype.kind == "f":
            raise ValueError(
                f"identity was handed a floating-point array (dtype {array.dtype}); a labelling "
                "must be integral, or the ids are not ids. Check the artifact's `kind`."
            )
        min_size = int(params.get("min_size", 0))
        fill = params.get("fill_distance", 0)
        if fill:
            return self._fill_bases.fill(
                (fingerprint(array), min_size), array.size,
                lambda: drop_small_components(array, min_size), fill,
            )
        # `copy=False` and no widening: a labelling read straight from an artifact is already the
        # right thing, and at these volumes an unnecessary cast is tens of gigabytes. A size filter
        # relabels it in place, which is safe because the runner reads the artifact afresh for
        # every candidate; `min_size` 0 returns it untouched.
        return drop_small_components(array, min_size)

    def lazy(self, artifact: Any, origin: tuple[int, ...], shape: tuple[int, ...],
             **params: Any) -> Any:
        """The stored labelling itself, read only where a point-lookup metric looks -- unless an
        add-on is to change it, which needs every voxel."""
        if params.get("min_size") or params.get("fill_distance"):
            return None
        from artifact import LazyLabelling

        return LazyLabelling(artifact, origin, shape)

    def describe(self, params: dict[str, Any]) -> str:
        if not params:
            return super().describe(params)            # "Identity", as plain identity rows read
        parts = [f"{key}={params[key]}" for key in ("min_size", "fill_distance") if params.get(key)]
        return f"identity({', '.join(parts)})" if parts else "identity(no filter)"


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
