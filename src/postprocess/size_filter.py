"""Dropping instance components below a voxel count, shared by every postprocessor that needs it.

Its own module because it is not specific to how the labelling was produced. `cc_threshold` and
`mws` both emit enormous numbers of tiny fragments on real affinity maps -- 3,583,131 predicted
objects against 3,620 true ones for thresholded components, 57,542 against 273 for mutex watershed
on one volume -- and PQ's recognition term counts objects unweighted by size, so in both cases
specks dominate the score independently of the segmentation's actual quality.

Kept as a plain function rather than folded into the base class: it is a filter over a labelling
with no knowledge of a sweep, a config or a kind, and the two callers differ in nothing but which
parameter name they read it from.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BasePostprocess
from .registry import PostprocessRegistry


def drop_small_components(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Relabel every component smaller than `min_size` voxels to background, in place.

    Both passes work in slabs, for the same reason and against two different blowups:

    **Counting.** `np.bincount` converts its argument to `intp`, so calling it on a 7-gigavoxel
    `uint32` labelling silently allocates a 56 GB int64 copy of the whole thing. Accumulating
    per-slab counts into one int64 histogram converts a thirty-second of it at a time.

    **Applying.** `remap[labels]` allocates a second 28 GB labelling beside the first, and
    `np.isin` builds a 7 GB boolean mask. Indexing a slab and writing back in place holds one slab
    and keeps the peak at the single labelling the caller already has.

    In place is deliberate -- the caller is `__call__`, whose labelling is freshly computed and has
    no other reader -- but it does mean this is not safe to hand an array you still need unfiltered.

    `min_size <= 0` returns the input untouched, so the default reproduces earlier records exactly.
    """
    if min_size <= 0:
        return labels
    slab = max(1, labels.shape[0] // 32)
    bounds = range(0, labels.shape[0], slab)

    sizes = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    for start in bounds:
        # A leading-axis slice of a C-contiguous array is contiguous, so `.ravel()` is a view and
        # only the intp conversion of this slab is materialised.
        sizes += np.bincount(labels[start : start + slab].ravel(), minlength=sizes.size)

    # Background needs no special case: it maps to itself either way, since a kept id maps to
    # `arange[id]` and a dropped one to 0, and those coincide at id 0. (An explicit guard here was
    # removed after a mutation test showed it could not change any output.)
    remap = np.where(sizes >= min_size, np.arange(sizes.size), 0).astype(labels.dtype, copy=False)
    for start in bounds:
        chunk = labels[start : start + slab]
        chunk[...] = remap[chunk]
    return labels


@PostprocessRegistry.register("size_filter")
class SizeFilter(BasePostprocess):
    """Drop components below a voxel count from a labelling that already exists.

    `cc_threshold` and `mws` apply the same filter as part of producing a labelling. This entry
    exists for the case where the labelling is the *input*: a stored `instances` artifact, whether
    written by an expensive postprocessor whose output was persisted or handed over by someone else.
    Mutex watershed on a 7-gigavoxel volume takes eleven hours, so re-running it once per candidate
    to sweep a filter is not an option -- the labelling is computed once and the filter swept over
    it, which is only possible if the filter is a postprocessor in its own right.

    Accepts `instances` only, unlike `identity`. A size filter on `class_labels` would be
    meaningless: a class region is not a component, and "drop small ones" has no reading there.
    """

    accepts = ("instances",)
    produces = "instances"

    def __init__(
        self, min_sizes: tuple[int, ...] | list[int] = (0,), **settings: Any
    ) -> None:
        super().__init__(min_sizes=min_sizes, **settings)
        if not min_sizes:
            raise ValueError(
                "size_filter with an empty `min_sizes` has nothing to sweep. Use `[0]` for no "
                "filter, which is the default."
            )
        if any(int(v) < 0 for v in min_sizes):
            raise ValueError(f"min_sizes must be non-negative voxel counts, got {list(min_sizes)}")
        self.min_sizes = tuple(sorted({int(v) for v in min_sizes}))

    def search_space(self) -> list[dict[str, Any]]:
        return [{"min_size": min_size} for min_size in self.min_sizes]

    def __call__(self, array: np.ndarray, **params: Any) -> np.ndarray:
        if array.dtype.kind == "f":
            raise ValueError(
                f"size_filter was handed a floating-point array (dtype {array.dtype}); a labelling "
                "must be integral, or the ids are not ids. Check the artifact's `kind`."
            )
        # `drop_small_components` works in place. Safe here because the runner reads the artifact
        # afresh for every candidate, so each sweep step owns its array -- but not safe in general,
        # which is why the helper says so.
        return drop_small_components(array, int(params.get("min_size", 0)))

    def describe(self, params: dict[str, Any]) -> str:
        min_size = int(params.get("min_size", 0))
        return f"size_filter(min_size={min_size})" if min_size else "size_filter(no filter)"
