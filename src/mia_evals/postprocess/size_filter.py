"""The two add-ons every instance post-processor takes: dropping components below a voxel count
(`min_sizes`), and growing the surviving segments into the background around them
(`fill_distances`).

Its own module because neither is specific to how the labelling was produced. `cc_threshold` and
`mws` both emit enormous numbers of tiny fragments on real affinity maps -- 3,583,131 predicted
objects against 3,620 true ones for thresholded components, 57,542 against 273 for mutex watershed
on one volume -- and PQ's recognition term counts objects unweighted by size, so in both cases
specks dominate the score independently of the segmentation's actual quality. A finished labelling
can carry specks too, and gaps: `identity` takes both settings, so a stored labelling (a persisted
mutex watershed, or one handed over by another lab) is filtered and filled exactly as a computed
one is.

Kept as plain functions rather than folded into the base class: a filter over a labelling with no
knowledge of a sweep, a config or a kind; the three callers differ only in how they produce the
labelling first. Each sweeps the add-ons innermost, fill inside size filter, so `FillBases` can
serve every fill distance of one size filter from one distance transform.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import ndimage


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


def nearest_segment(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(distance, index) from every background voxel to its nearest segment voxel, in voxels.

    Exact Euclidean (`scipy.ndimage.distance_transform_edt`). Split from `fill_holes` so that one
    labelling can be filled to several distances from a single transform, which on a 1000^3 block
    is a float64 distance and an int32 index per voxel per axis -- about 20 GB.
    """
    return ndimage.distance_transform_edt(labels == 0, return_indices=True)


def fill_holes(
    labels: np.ndarray,
    max_distance: float,
    nearest: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Grow the segments into the background within `max_distance` voxels of them.

    Every background voxel at most `max_distance` from a segment takes the label of its nearest
    segment voxel, so segments meet where their gaps close instead of each dilating by a fixed
    amount. Meant for the holes a size filter leaves and for the gaps of a model trained on eroded
    labels; `math.inf` fills every background voxel, which on this task also floods the true
    background between neurons. `nearest` is `nearest_segment(labels)`, when the caller has it.
    Returns a new array, or `labels` itself when there is nothing to do.
    """
    if max_distance <= 0 or not labels.any():
        return labels
    distance, index = nearest if nearest is not None else nearest_segment(labels)
    fill = (labels == 0) & (distance <= max_distance)
    filled = labels.copy()
    filled[fill] = labels[tuple(axis[fill] for axis in index)]
    return filled


def parse_min_sizes(values: Any, owner: str) -> tuple[int, ...]:
    """A `min_sizes` setting, checked and sorted; `[0]`, no filter, is every caller's default."""
    if not values:
        raise ValueError(
            f"{owner} with an empty `min_sizes` has nothing to sweep. Use `[0]` for no size "
            "filter, which is the default."
        )
    if any(int(v) < 0 for v in values):
        raise ValueError(f"min_sizes must be non-negative voxel counts, got {list(values)}")
    return tuple(sorted({int(v) for v in values}))


def parse_fill_distances(values: Any, owner: str) -> tuple[int | str, ...]:
    """A `fill_distances` setting, checked and sorted with "all" last, as `mws` reads its own."""
    if not values:
        raise ValueError(
            f"{owner} with an empty `fill_distances` has nothing to sweep. Use `[0]` for no "
            "filling, which is the default."
        )
    bad = [v for v in values
           if v != "all" and (isinstance(v, bool) or not isinstance(v, int) or v < 0)]
    if bad:
        raise ValueError(f'fill_distances must be non-negative voxel counts or "all", got {bad}')
    return tuple(sorted({int(v) for v in values if v != "all"})) + (
        ("all",) if "all" in values else ()
    )


def with_fill(
    space: list[dict[str, Any]], fill_distances: tuple[int | str, ...]
) -> list[dict[str, Any]]:
    """`space` with every fill distance swept innermost, or `space` itself for the default `[0]`,
    so a config without `fill_distances` fits and records exactly as before the setting existed."""
    if fill_distances == (0,):
        return space
    return [{**point, "fill_distance": fill} for point in space for fill in fill_distances]


def fill_limit(fill: int | str) -> float:
    """A `fill_distance` as `fill_holes` takes it: "all" is no limit."""
    return math.inf if fill == "all" else int(fill)


def fingerprint(array: np.ndarray) -> str:
    """Which array this is, cheaply: shape, dtype and a hash of a one-in-N sample, as `mws` keys
    its caches. Distinguishes the volumes of one fit, which is all a cache key here needs."""
    flat = array.reshape(-1)
    step = max(1, flat.size // 1_000_000)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(flat[::step]).tobytes())
    digest.update(np.ascontiguousarray(flat[-1024:]).tobytes())
    return f"{array.shape}:{array.dtype}:{digest.hexdigest()}"


#: What `FillBases` may hold: an lm_zebrafish frame's basis (113 M voxels) is ~3 GB, so every fit
#: frame of a sweep fits; a gigavoxel block's (~28 GB) never does, and then only the latest is kept,
#: as `mws` keeps one.
FILL_CACHE_BYTES = 16 * 2**30
#: A basis's size per voxel at most: an int64 labelling, a float64 distance, three int32 indices.
BASIS_BYTES_PER_VOXEL = 8 + 8 + 3 * 4


class FillBases:
    """Size-filtered labellings and their distance transforms, kept for the fill candidates that
    follow them.

    The fill candidates of one size filter differ only in distance, so they share its filtered
    labelling and that labelling's transform (`nearest_segment`). The runner scores a candidate on
    every fit volume before the next, so a basis is reused only if every volume's survives in
    between: they are kept, least recently used first out, while they fit `budget` bytes. Room is
    made before a basis is built, so a block too large to keep two of frees the old one first.
    """

    def __init__(self, budget: int = FILL_CACHE_BYTES) -> None:
        self.budget = budget
        self._entries: OrderedDict[Any, tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]] = (
            OrderedDict()
        )

    def fill(self, key: Any, voxels: int, filtered: Callable[[], np.ndarray],
             fill: int | str) -> np.ndarray:
        """`filtered()`'s labelling filled to `fill`, its basis computed once per `key`.

        A fresh array every time, never the cached one, since that serves the next distance.
        """
        entry = self._entries.get(key)
        if entry is None:
            need = voxels * BASIS_BYTES_PER_VOXEL
            while self._entries and self._held() + need > self.budget:
                self._entries.popitem(last=False)
            labels = filtered()
            entry = (labels, nearest_segment(labels))
            self._entries[key] = entry
        else:
            self._entries.move_to_end(key)
        labels, nearest = entry
        out = fill_holes(labels, fill_limit(fill), nearest)
        return out.copy() if out is labels else out

    def _held(self) -> int:
        return sum(labels.nbytes + distance.nbytes + index.nbytes
                   for labels, (distance, index) in self._entries.values())
