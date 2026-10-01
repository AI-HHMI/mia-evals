"""Three-channel affinities -> instances: a mutex watershed on the signed nearest-neighbour graph.

For affinity maps that carry only the three short-range channels -- one voxel along each axis, what
most U-Net-style affinity models predict. `mws` cannot use them: its repulsive edges are the
long-range channels, and a mutex watershed over attractive edges alone merges everything.

**Each edge is attractive or repulsive by its own value.** An affinity above 0.5 says the two
voxels belong together, so the edge is attractive with priority `a`; one at or below 0.5 says they
do not, so it is repulsive with priority `1 - a`. Sorted by priority, highest first, the most
confident statement of either kind comes first, exactly as in `mws`, and the same compiled kernel
(`mws_kernel.py`) runs on the result. This is the mutex watershed on a signed graph, the form in
which GASP (Bailoni et al., TPAMI 2022) treats it as one linkage rule among several.

Two properties follow from the construction, and the tests check both:

  * every merge follows an edge above 0.5, so each segment lies inside one connected component of
    the affinities thresholded at 0.5 -- never coarser than thresholded components there;
  * a merge is refused when a more confident repulsive edge between the two clusters came first, so
    a weak leak through an otherwise clearly predicted membrane does not join the two objects, as
    it would under a threshold.

It needs only that a cut sit below 0.5, in whichever channel it shows. Measured on a 256^3 crop
of the fit block (pq at min_size 500): affinities of the un-eroded truth 0.87; of the truth eroded
by one voxel 0.61 with the fill (0.51 without: the membrane's voxels are left as specks); blurred
so that 17% of cuts rose above 0.5, 0.01 -- everything merged.

No threshold is fitted; the size filter (`min_sizes`) and optionally `fill_distances` are, exactly
as for `mws`. Every edge is held in memory -- three per voxel, about 2.2 G at 896^3 -- and there is
no streaming path: the route is meant for blocks of that size.

A six-channel artifact is accepted as well and only its short-range half is read, so that one of
our models can be put through the same decoder as a three-channel one.

Branch `three-channel-mws` only (2026-10-01), for scoring a colleague's three-channel model; see
docs/scoring_three_channel_affinities.md.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import Any

import numpy as np

from .base import BasePostprocess
from .mws import MAX_IN_MEMORY_EDGES, SHORT_OFFSETS, MutexWatershed, check_offsets
from .mws_kernel import compact_roots, finalize, initial_capacities, make_state, run_edges
from .registry import PostprocessRegistry
from .size_filter import drop_small_components, fill_holes, nearest_segment

#: An affinity above this is attractive; at or below it, repulsive.
NEUTRAL = 0.5


def check_short_range(artifact: Any, route: str) -> None:
    """Refuse affinities whose first three channels are not the forward nearest-neighbour edges.

    `check_offsets` does the comparison. This only adds the case worth its own message: the same
    three edges pointing backwards, which is how gunpowder-trained models (LSD, MALA) store them,
    and which read as forward edges would put every boundary one voxel off.
    """
    declared = artifact.attrs.get("offsets")
    if declared is not None:
        first = {tuple(int(x) for x in offset) for offset in declared[: len(SHORT_OFFSETS)]}
        if first == {tuple(-x for x in offset) for offset in SHORT_OFFSETS}:
            raise ValueError(
                f"{artifact.path} declares its nearest-neighbour edges pointing backwards "
                f"({list(declared[: len(SHORT_OFFSETS)])}): channel i at voxel p is the edge to "
                f"p - e_i, the gunpowder convention. {route} reads it as the edge to p + e_i. "
                "Roll each channel by -1 along its own axis before writing the artifact and "
                "declare offsets [[1, 0, 0], [0, 1, 0], [0, 0, 1]] "
                "(docs/scoring_three_channel_affinities.md)."
            )
    check_offsets(artifact, SHORT_OFFSETS)


def sweep_settings(
    route: str,
    min_sizes: tuple[int, ...] | list[int],
    fill_distances: tuple[int | str, ...] | list[int | str],
) -> tuple[tuple[int, ...], tuple[int | str, ...]]:
    """`min_sizes` and `fill_distances` checked and normalised, as `mws` does them."""
    if not min_sizes:
        raise ValueError(
            f"{route} with an empty `min_sizes` has nothing to sweep. Use `[0]` for no size "
            "filter, which is the default."
        )
    if any(int(v) < 0 for v in min_sizes):
        raise ValueError(f"min_sizes must be non-negative voxel counts, got {list(min_sizes)}")
    if not fill_distances:
        raise ValueError(
            f"{route} with an empty `fill_distances` has nothing to sweep. Use `[0]` for no "
            "filling, which is the default."
        )
    bad = [v for v in fill_distances
           if v != "all" and (isinstance(v, bool) or not isinstance(v, int) or v < 0)]
    if bad:
        raise ValueError(f'fill_distances must be non-negative voxel counts or "all", got {bad}')
    sizes = tuple(sorted({int(v) for v in min_sizes}))
    fills: tuple[int | str, ...] = tuple(sorted({int(v) for v in fill_distances if v != "all"}))
    return sizes, fills + (("all",) if "all" in fill_distances else ())


class SizeFilterAndFill:
    """The size filter, then optionally the survivors grown back into the holes it leaves.

    `mws`'s `_filled`, lifted out so both three-channel routes share it. The filtered labelling and
    its distance transform are kept for the last (labelling, min_size): the fill distances of one
    size are adjacent in `search_space()`, and each such basis is ~28 GB on a 1000^3 block.
    """

    def __init__(self) -> None:
        self._basis: tuple[tuple, np.ndarray, tuple[np.ndarray, np.ndarray]] | None = None

    def __call__(self, labels: np.ndarray, key: Any, min_size: int, fill: int | str) -> np.ndarray:
        if not fill:
            return drop_small_components(labels.copy(), min_size)
        basis_key = (key, min_size)
        if self._basis is None or self._basis[0] != basis_key:
            self._basis = None                           # free the previous basis first
            filtered = drop_small_components(labels.copy(), min_size)
            self._basis = (basis_key, filtered, nearest_segment(filtered))
        _, filtered, nearest = self._basis
        filled = fill_holes(filtered, math.inf if fill == "all" else int(fill), nearest)
        # Always a fresh array, as the unfilled path returns: the basis serves the next distance.
        return filled.copy() if filled is filtered else filled


def describe_finish(params: dict[str, Any]) -> list[str]:
    """The size filter and fill of a candidate, as they read in a leaderboard row."""
    min_size = int(params.get("min_size", 0))
    fill = params.get("fill_distance", 0)
    return ([f"min_size={min_size}"] if min_size else []) + ([f"fill={fill}"] if fill else [])


def count_signed_edges(shape: tuple[int, ...]) -> int:
    """How many edges `build_signed_edges` emits for this shape: one per voxel pair per axis."""
    return sum(
        int(np.prod([max(int(s) - o, 0) for s, o in zip(shape, offset, strict=True)]))
        for offset in SHORT_OFFSETS
    )


def build_signed_edges(
    affinities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(3, X, Y, Z) affinities -> flat edge arrays (u, v, priority, attractive).

    Channel i is the edge from each voxel to the one `SHORT_OFFSETS[i]` away, in the order
    `mws.build_edges` emits its attractive edges. Written into preallocated arrays rather than
    concatenated, which at 896^3 saves a transient second copy of every edge (about 45 GB).
    """
    shape = affinities.shape[1:]
    total = count_signed_edges(shape)
    index = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    u = np.empty(total, dtype=np.int64)
    v = np.empty(total, dtype=np.int64)
    priority = np.empty(total, dtype=np.float32)
    attractive = np.empty(total, dtype=bool)
    start = 0
    for channel, offset in enumerate(SHORT_OFFSETS):
        overlap = tuple(slice(0, max(s - o, 0)) for s, o in zip(shape, offset, strict=True))
        shifted = tuple(slice(o, o + max(s - o, 0)) for s, o in zip(shape, offset, strict=True))
        a = affinities[channel][overlap]
        stop = start + a.size
        u[start:stop].reshape(a.shape)[...] = index[overlap]
        v[start:stop].reshape(a.shape)[...] = index[shifted]
        pull = a > NEUTRAL
        attractive[start:stop].reshape(a.shape)[...] = pull
        priority[start:stop].reshape(a.shape)[...] = np.where(pull, a, 1.0 - a)
        start = stop
    return u, v, priority, attractive


def segment_signed(affinities: np.ndarray, *, info: dict[str, Any] | None = None) -> np.ndarray:
    """(3, X, Y, Z) affinities -> (X, Y, Z) labels 1..k, by the compiled mutex watershed kernel.

    `mws._in_memory` with the signed edges. Every voxel is in a cluster, so there is no 0.
    """
    shape = tuple(int(s) for s in affinities.shape[1:])
    n_nodes = int(np.prod(shape))
    n_edges = count_signed_edges(shape)
    if n_edges > MAX_IN_MEMORY_EDGES:
        raise ValueError(
            f"this block has {n_edges:,} short-range edges, more than {MAX_IN_MEMORY_EDGES:,}: "
            "mws3 holds every edge in memory and has no streaming path"
        )
    start = time.perf_counter()
    u, v, priority, attractive = build_signed_edges(affinities)
    # As in `mws`: a stable sort, so the many exact ties of float16 affinities are broken by
    # position in `build_signed_edges`' output, the order the reference sees them in.
    order = np.argsort(-priority, kind="stable")
    del priority
    # One gather at a time, so only one extra edge array is alive beside the order.
    u = u[order]
    v = v[order]
    attractive = attractive[order]
    del order
    pairs, pool = initial_capacities(n_nodes)
    state = make_state(np.int64(n_nodes), np.int64(pairs), np.int64(pool))
    state, growths = run_edges(state, u, v, attractive)
    del u, v, attractive
    parent, insertions = state[0], int(state[-1][3])
    del state
    labels = compact_roots(finalize(parent).reshape(shape))
    if info is not None:
        info.update(implementation="compiled kernel, signed short-range edges in memory",
                    seconds=round(time.perf_counter() - start, 1), edges=n_edges,
                    pair_insertions=insertions, growths=growths)
    return labels


@PostprocessRegistry.register("mws3")
class ThreeChannelMutexWatershed(BasePostprocess):
    """Short-range affinities -> instances by the signed mutex watershed; see the module docstring.

    `min_sizes` and `fill_distances` are swept as in `mws` (fill innermost, so one distance
    transform serves every distance of a size). The watershed itself has no parameter, so it runs
    once per volume and the sweep is applied on top of it.
    """

    accepts = ("affinity",)
    produces = "instances"

    #: Labellings kept: the fit block's and the test block's, ~6 GB each at 896^3 as int64.
    CACHE_ENTRIES = 2

    def __init__(
        self,
        min_sizes: tuple[int, ...] | list[int] = (0,),
        fill_distances: tuple[int | str, ...] | list[int | str] = (0,),
        **settings: Any,
    ) -> None:
        super().__init__(min_sizes=min_sizes, fill_distances=fill_distances, **settings)
        self.min_sizes, self.fill_distances = sweep_settings("mws3", min_sizes, fill_distances)
        self._labellings: OrderedDict[str, tuple[np.ndarray, dict[str, Any]]] = OrderedDict()
        self._finish = SizeFilterAndFill()
        self._last_run: dict[str, Any] | None = None

    def reads_channels(self) -> int:
        return len(SHORT_OFFSETS)

    def check_artifact(self, artifact: Any) -> None:
        check_short_range(artifact, "mws3")

    def run_info(self) -> dict[str, Any] | None:
        return None if self._last_run is None else dict(self._last_run)

    def _labelling(self, affinities: np.ndarray, key: str) -> np.ndarray:
        cached = self._labellings.get(key)
        if cached is None:
            info: dict[str, Any] = {}
            labels = segment_signed(affinities, info=info).astype(np.int64)
            print(f"  mws3 watershed: {info['edges']:,} edges, {info['seconds']:.0f} s, "
                  f"{info['pair_insertions']:,} pair insertions, {info['growths']} growth(s)",
                  flush=True)
            cached = (labels, info)
            self._labellings[key] = cached
            while len(self._labellings) > self.CACHE_ENTRIES:
                self._labellings.popitem(last=False)
        else:
            self._labellings.move_to_end(key)
        self._last_run = cached[1]
        return cached[0]

    def search_space(self) -> list[dict[str, Any]]:
        space = [{"min_size": min_size} for min_size in self.min_sizes]
        if self.fill_distances == (0,):
            return space
        return [{**point, "fill_distance": fill} for point in space for fill in self.fill_distances]

    def __call__(self, array: np.ndarray, **params: Any) -> np.ndarray:
        affinities = np.asarray(array[: len(SHORT_OFFSETS)], dtype=np.float32)
        key = MutexWatershed._fingerprint(affinities)
        labels = self._labelling(affinities, key)
        return self._finish(labels, key, int(params.get("min_size", 0)),
                            params.get("fill_distance", 0))

    def describe(self, params: dict[str, Any]) -> str:
        finish = describe_finish(params)
        return f"mws3({', '.join(finish)})" if finish else "mws3"
