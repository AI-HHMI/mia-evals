"""Mutex watershed: affinities -> instances with no threshold at all.

Moved here from the repository root, where it was a standalone script. The algorithm is ours,
written from the paper -- Wolf et al., *The Mutex Watershed* (ECCV 2018) -- because the reference
implementation `affogato` is CMake-only and does not pip-install, and the PyPI package of that name
is an unrelated project. `tests/unit/test_mws.py` checks it against cases whose answers follow from
the definition rather than from a reference.

Why it exists beside `cc_threshold`: the model is trained on six channels and measurement showed
the three that thresholded components discards are its *better* predictions -- the long-range
channels separate same-object from different-object more confidently (+0.40 to +0.46) than the
short-range ones (+0.35), because a 10-voxel relationship is a contextual question a ViT answers
well while a 1-voxel one needs localisation a 16x-upsampled head cannot express. Mutex watershed
uses both: short-range as *attractive* edges, long-range as *repulsive* ones, and no threshold
anywhere -- repulsion does the separating.

**Measured worse on this task, so far.** On a 512^3 block of seed100 with the best available
checkpoint it scored 0.024 nERL against thresholded components' 0.584, with 242 mergers against 4.
Most of that is `repulsive_stride = 4` discarding 64x of the repulsive edges (at stride 1 it
recovers to 0.380), but components still win by 2x. It is kept because the reasoning above is
sound and the failure is a tuning story, not because it is currently the better choice.

**One production path, one oracle.** `mutex_watershed_reference` below is the algorithm written as
plainly as possible, in Python: the definition, and the oracle the compiled kernel is tested
against. Nothing scores with it. `segment()` is what the post-processor calls. It runs the compiled
kernel (`mws_kernel.py`) with every edge in memory, or, above `MAX_IN_MEMORY_EDGES`, streams the
edges through disk in priority bands (`mws_stream.py`) into the same kernel. Both reproduce the
reference's partition exactly -- the unit tests require it -- so which one runs decides the time
and memory of a scoring job, never its numbers. Until 2026-09-23 `segment()` ran the Python
reference: about 2.5 hours per 896^3 block, against about 20 minutes compiled on a node of its own
(measured that day on all seven gary_comparison mws records; a second memory-heavy job on the same
node made it two to five times slower).

The labels come out numbered differently from the reference's, a permutation of the same clusters.
No metric depends on numbering, but VOI sums its terms in label order, so it can move in the 15th
digit: re-scoring the seven records above reproduced every pq, choice and partition bit for bit,
and VOI to within 6e-15.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .base import BasePostprocess
from .registry import PostprocessRegistry
from .mws_kernel import compact_roots, finalize, initial_capacities, make_state, run_edges
from .size_filter import drop_small_components

SHORT_OFFSETS = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
LONG = 10
LONG_OFFSETS = ((LONG, 0, 0), (0, LONG, 0), (0, 0, LONG))

#: Above this many edges, `segment()` streams them through disk instead of holding them. Held in
#: memory the edge list peaks at about 37 bytes per edge, while it is being sorted: source and
#: target (16), priority and its negation (8), the attractive flag (1), the sort order (8) and the
#: sort's buffer (4). 8 G edges is therefore about 300 GB; an 896^3 block at stride 1 has 4.3 G.
#: Streaming holds one priority band at a time instead, at the price of writing every edge to disk
#: and reading it back. Both feed the same kernel in the same order, so the limit changes time and
#: memory only; a scoring config can move it with `max_in_memory_edges` under `[postprocess]`.
MAX_IN_MEMORY_EDGES = 8_000_000_000
#: Streaming block edge, in voxels. Rounded down to a multiple of the repulsive stride, so that the
#: sources each block keeps are exactly the ones `build_edges` keeps for the whole volume.
STREAM_BLOCK = 256


def build_edges(
    affinities: np.ndarray, repulsive_stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(6, X, Y, Z) affinities -> flat edge arrays (u, v, priority, attractive).

    Priority is what the algorithm sorts on, and it differs by edge type. For an attractive edge
    the affinity *is* the merge evidence, so priority = a. For a repulsive edge the evidence is
    that the two voxels are *different*, which is strong when the affinity is low, so
    priority = 1 - a. Sorting both by priority descending puts the most confident assertion of
    either kind first, which is exactly what mutex watershed requires.

    `repulsive_stride` subsamples the long-range edges. Every voxel contributing three repulsive
    edges is affordable at small volumes and not at large ones, and the repulsive edges exist to
    place constraints rather than to cover every pair -- taking every k-th voxel keeps the
    constraint field while cutting the edge count by k^3.
    """
    shape = affinities.shape[1:]
    index = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)

    us, vs, priorities, attractive = [], [], [], []
    for channel, offset in enumerate(SHORT_OFFSETS + LONG_OFFSETS):
        is_attractive = channel < len(SHORT_OFFSETS)
        # max(s - o, 0): a negative stop would wrap and silently produce a `u` and `v` of
        # different lengths, which is reachable whenever an axis is shorter than the long-range
        # offset -- true of any block under 10 voxels deep.
        overlap = tuple(slice(0, max(s - o, 0)) for s, o in zip(shape, offset, strict=True))
        shifted = tuple(slice(o, o + max(s - o, 0)) for s, o in zip(shape, offset, strict=True))
        a = affinities[channel][overlap]
        u, v = index[overlap], index[shifted]

        if not is_attractive and repulsive_stride > 1:
            keep = tuple(slice(None, None, repulsive_stride) for _ in shape)
            a, u, v = a[keep], u[keep], v[keep]

        us.append(u.ravel())
        vs.append(v.ravel())
        priorities.append((a if is_attractive else 1.0 - a).ravel())
        attractive.append(np.full(u.size, is_attractive, dtype=bool))

    return (
        np.concatenate(us),
        np.concatenate(vs),
        np.concatenate(priorities).astype(np.float32),
        np.concatenate(attractive),
    )


def count_edges(shape: tuple[int, ...], repulsive_stride: int) -> int:
    """How many edges `build_edges` would emit for this shape and stride, without building them."""
    total = 0
    for channel, offset in enumerate(SHORT_OFFSETS + LONG_OFFSETS):
        extents = [max(int(s) - o, 0) for s, o in zip(shape, offset, strict=True)]
        if channel >= len(SHORT_OFFSETS) and repulsive_stride > 1:
            extents = [-(-e // repulsive_stride) for e in extents]
        total += int(np.prod(extents))
    return total


def mutex_watershed_reference(
    u: np.ndarray, v: np.ndarray, priority: np.ndarray, attractive: np.ndarray, n_nodes: int
) -> np.ndarray:
    """Mutex watershed: edges in descending priority -> a label per node. The reference.

    The algorithm as plainly as it can be written, and the oracle `mws_kernel` is tested against;
    nothing scores with it. It walks about 0.3 M edges per second, against 2.3 M compiled.

    Union-find, plus a set of forbidden partners per cluster. Walking edges from most to least
    confident:

      * an **attractive** edge merges its two clusters, unless a mutex forbids it;
      * a **repulsive** edge records a mutex between them, so no later (weaker) attractive edge can
        join them.

    No threshold appears anywhere. Every attractive edge would eventually merge if left alone, so
    it is the repulsive constraints that carve the partition -- which is why the long-range
    channels matter and why discarding them forces a threshold to do their job.
    """
    parent = np.arange(n_nodes, dtype=np.int64)
    forbidden: dict[int, set[int]] = {}

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:      # path compression
            parent[x], x = root, parent[x]
        return int(root)

    order = np.argsort(-priority, kind="stable")
    for e in order:
        ru, rv = find(int(u[e])), find(int(v[e]))
        if ru == rv:
            continue
        if attractive[e]:
            if rv in forbidden.get(ru, ()):
                continue
            # Union by set size, so the mutex sets merge in the cheaper direction.
            big, small = (
                (ru, rv) if len(forbidden.get(ru, ())) >= len(forbidden.get(rv, ()))
                else (rv, ru)
            )
            parent[small] = big
            moved = forbidden.pop(small, set())
            if moved:
                target = forbidden.setdefault(big, set())
                for other in moved:
                    target.add(other)
                    partners = forbidden.get(other)
                    if partners is not None:
                        partners.discard(small)
                        partners.add(big)
        else:
            forbidden.setdefault(ru, set()).add(rv)
            forbidden.setdefault(rv, set()).add(ru)

    roots = np.array([find(i) for i in range(n_nodes)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return (labels + 1).astype(np.uint32)


def _in_memory(affinities: np.ndarray, repulsive_stride: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Every edge built, sorted and fed to the compiled kernel at once."""
    shape = tuple(int(s) for s in affinities.shape[1:])
    n_nodes = int(np.prod(shape))
    u, v, priority, attractive = build_edges(affinities, repulsive_stride)
    n_edges = int(u.size)
    # The reference's ordering, expression for expression. Affinities are float16 on disk, so equal
    # priorities are abundant, and the partition depends on how those ties are broken: here, as
    # there, by position in `build_edges`' output.
    order = np.argsort(-priority, kind="stable")
    del priority
    u, v, attractive = u[order], v[order], attractive[order]
    del order
    pairs, pool = initial_capacities(n_nodes)
    state = make_state(np.int64(n_nodes), np.int64(pairs), np.int64(pool))
    state, growths = run_edges(state, u, v, attractive)
    del u, v, attractive
    parent, insertions = state[0], int(state[-1][3])
    del state
    labels = compact_roots(finalize(parent).reshape(shape))
    return labels, {"edges": n_edges, "pair_insertions": insertions, "growths": growths}


def _streamed(
    affinities: np.ndarray, repulsive_stride: int, scratch: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    """The edges written to disk in priority bands and fed to the same kernel band by band."""
    from .mws_stream import segment_streaming      # imports this module's offsets

    shape = tuple(int(s) for s in affinities.shape[1:])
    block = max(repulsive_stride, (STREAM_BLOCK // repulsive_stride) * repulsive_stride)

    def read_block(origin: tuple[int, ...], size: tuple[int, ...]) -> np.ndarray:
        window = tuple(slice(o, o + n) for o, n in zip(origin, size, strict=True))
        return affinities[(slice(None), *window)]

    scratch.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="mws_stream_", dir=scratch))
    try:
        labels, stats = segment_streaming(
            read_block, shape, work, block=block, repulsive_stride=repulsive_stride
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return labels, {k: stats[k] for k in ("edges", "pair_insertions", "growths")}


def segment(
    affinities: np.ndarray,
    repulsive_stride: int,
    *,
    max_in_memory_edges: int = MAX_IN_MEMORY_EDGES,
    scratch: str | Path | None = None,
    info: dict[str, Any] | None = None,
) -> np.ndarray:
    """(6, X, Y, Z) affinities -> (X, Y, Z) instance labels 1..k, by the compiled kernel.

    In memory when the block has at most `max_in_memory_edges` edges, streamed through `scratch`
    otherwise. The partition is the reference's either way; the labels are numbered in the order of
    the clusters' root ids, so they are a permutation of the reference's numbering, which no metric
    sees. `info`, when given, receives how the labelling was computed.
    """
    shape = tuple(int(s) for s in affinities.shape[1:])
    n_edges = count_edges(shape, repulsive_stride)
    start = time.perf_counter()
    if n_edges <= max_in_memory_edges:
        labels, stats = _in_memory(affinities, repulsive_stride)
        how = "compiled kernel, edges in memory"
    else:
        if scratch is None:
            raise ValueError(
                f"this block has {n_edges:,} edges, more than max_in_memory_edges="
                f"{max_in_memory_edges:,}, so its edges must be streamed through disk -- but no "
                "scratch directory was given. `mia-evals score` passes its --scratch; a direct "
                "caller passes `scratch=` (or calls `use_scratch` on the post-processor)."
            )
        labels, stats = _streamed(affinities, repulsive_stride, Path(scratch))
        how = "compiled kernel, edges streamed through disk"
    if info is not None:
        info.update(implementation=how, seconds=round(time.perf_counter() - start, 1), **stats)
    return labels


@PostprocessRegistry.register("mws")
class MutexWatershed(BasePostprocess):
    """Six-channel affinities -> instances, with the repulsive edges doing the separating.

    `repulsive_strides` and `min_sizes` are the sweep, and `search_space()` is their cross-product.
    Subsampling the long-range edges is what makes this affordable on a large block -- every voxel
    contributing three repulsive edges is fine at 128^3 and not at 2000^3 -- but it is also the
    setting that decides the result, so it is fitted on validation rather than defaulted and
    forgotten. Stride 1 keeps every repulsive edge.

    **`min_sizes` matters more here than the name suggests.** Measured on `kasthuri15_ac4`, mutex
    watershed at stride 1 recovered 192 of 273 true objects at SQ 0.733 and cut `voi_merge` from
    6.753 to 0.550 against thresholded components -- it very nearly removes the merge errors. Its
    PQ was still 0.0049, because it also returned 57,350 single-figure fragments and PQ counts each
    as a false positive regardless of size. Without a size filter the metric hides the improvement
    almost completely.
    """

    accepts = ("affinity",)
    produces = "instances"

    def __init__(
        self,
        repulsive_strides: tuple[int, ...] | list[int] = (1, 2, 4),
        min_sizes: tuple[int, ...] | list[int] = (0,),
        max_in_memory_edges: int = MAX_IN_MEMORY_EDGES,
        **settings: Any,
    ) -> None:
        super().__init__(
            repulsive_strides=repulsive_strides, min_sizes=min_sizes,
            max_in_memory_edges=max_in_memory_edges, **settings
        )
        if int(max_in_memory_edges) < 0:
            raise ValueError(f"max_in_memory_edges must be >= 0, got {max_in_memory_edges}")
        self.max_in_memory_edges = int(max_in_memory_edges)
        self._scratch: Path | None = None
        self._last_run: dict[str, Any] | None = None
        if not repulsive_strides:
            raise ValueError("mws with an empty `repulsive_strides` has nothing to sweep")
        bad = sorted(s for s in repulsive_strides if int(s) < 1)
        if bad:
            raise ValueError(f"repulsive_strides must all be >= 1, got {bad}; 1 keeps every edge")
        self.repulsive_strides = tuple(int(s) for s in repulsive_strides)
        if not min_sizes:
            raise ValueError(
                "mws with an empty `min_sizes` has nothing to sweep. Use `[0]` for no size filter, "
                "which is the default."
            )
        if any(int(v) < 0 for v in min_sizes):
            raise ValueError(f"min_sizes must be non-negative voxel counts, got {list(min_sizes)}")
        self.min_sizes = tuple(sorted({int(v) for v in min_sizes}))
        # The watershed depends on the affinities and the stride only; the size filter is applied on
        # top. Sweeping `min_sizes` therefore needs ONE watershed per (volume, stride), not one per
        # candidate -- at 896^3 a watershed is 4 G edges and about two hours, so four candidates
        # would have cost eight. Keyed by a content fingerprint rather than object identity because
        # the scorer re-reads the artifact for every candidate.
        self._labellings: OrderedDict[tuple, tuple[np.ndarray, dict[str, Any]]] = OrderedDict()

    CACHE_ENTRIES = 4                      # int64 labellings; 4 x 896^3 is ~23 GB

    def use_scratch(self, directory: str | Path) -> None:
        """Where to stream edges for a block above `max_in_memory_edges`."""
        self._scratch = Path(directory) / "mws"

    def run_info(self) -> dict[str, Any] | None:
        """How the labelling behind the last call was computed: implementation, edges, seconds."""
        return None if self._last_run is None else dict(self._last_run)

    @staticmethod
    def _fingerprint(affinities: np.ndarray) -> str:
        flat = affinities.reshape(-1)
        step = max(1, flat.size // 1_000_000)
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.ascontiguousarray(flat[::step]).tobytes())
        digest.update(np.ascontiguousarray(flat[-1024:]).tobytes())
        return f"{affinities.shape}:{affinities.dtype}:{digest.hexdigest()}"

    def _labelling(self, affinities: np.ndarray, stride: int) -> np.ndarray:
        key = (stride, self._fingerprint(affinities))
        cached = self._labellings.get(key)
        if cached is None:
            info: dict[str, Any] = {}
            labels = segment(
                affinities, stride, max_in_memory_edges=self.max_in_memory_edges,
                scratch=self._scratch, info=info,
            ).astype(np.int64)
            print(f"  mws watershed: {info['edges']:,} edges, {info['implementation']}, "
                  f"{info['seconds']:.0f} s, {info['pair_insertions']:,} pair insertions, "
                  f"{info['growths']} growth(s)", flush=True)
            cached = (labels, info)
            self._labellings[key] = cached
            while len(self._labellings) > self.CACHE_ENTRIES:
                self._labellings.popitem(last=False)
        else:
            self._labellings.move_to_end(key)
        self._last_run = cached[1]
        return cached[0]

    def search_space(self) -> list[dict[str, Any]]:
        return [
            {"repulsive_stride": stride, "min_size": min_size}
            for stride in self.repulsive_strides
            for min_size in self.min_sizes
        ]

    def __call__(self, array: np.ndarray, **params: Any) -> np.ndarray:
        if array.shape[0] < 6:
            raise ValueError(
                f"mutex watershed needs all six affinity channels -- three attractive and three "
                f"repulsive -- but this artifact has {array.shape[0]}. With only the short-range "
                "half there are no repulsive edges, so it would degenerate to connected components "
                "over the attractive graph; use cc_threshold for that."
            )
        labels = self._labelling(np.asarray(array, dtype=np.float32), int(params["repulsive_stride"]))
        return drop_small_components(labels.copy(), int(params.get("min_size", 0)))

    def describe(self, params: dict[str, Any]) -> str:
        text = f"mws(repulsive_stride={params['repulsive_stride']}"
        min_size = int(params.get("min_size", 0))
        return text + (f", min_size={min_size}" if min_size else "") + ")"
