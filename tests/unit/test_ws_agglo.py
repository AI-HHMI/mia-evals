"""LSD-style watershed + agglomeration for three-channel affinities (branch `three-channel-mws`).

No reference implementation is installable (waterz is C++ built at import), so each step is checked
against what follows from its definition: fragments against objects they must not straddle, the
region graph against a brute-force count, and every agglomeration step against the merge function
recomputed from scratch over the segments as they stood.
"""

from __future__ import annotations

import numpy as np
import pytest

from postprocess import ws_agglo
from postprocess.ws_agglo import (
    BINS,
    WatershedAgglomeration,
    agglomerate,
    fill_outside,
    fragments,
    parse_merge_function,
    region_graph,
    roots_at,
    seeded_watershed,
)

pytestmark = pytest.mark.unit


def _affinities_from_labels(labels: np.ndarray) -> np.ndarray:
    """(3, *shape) forward nearest-neighbour affinities: 1 where p and p + e_i share an object."""
    aff = np.zeros((3, *labels.shape), dtype=np.float32)
    for axis in range(3):
        lo = tuple(slice(0, -1) if a == axis else slice(None) for a in range(3))
        hi = tuple(slice(1, None) if a == axis else slice(None) for a in range(3))
        aff[axis][lo] = (labels[lo] == labels[hi]) & (labels[lo] > 0)
    return aff


def _boxes() -> np.ndarray:
    """A 3 x 3 grid of 7 x 7 x 20 objects, separated by one-voxel background planes."""
    labels = np.zeros((24, 24, 20), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            labels[1 + 8 * i: 8 + 8 * i, 1 + 8 * j: 8 + 8 * j] = 1 + 3 * i + j
    return labels


def _score(hist: np.ndarray, total: float, count: int, quantile: float | None) -> float:
    """The merge function from scratch: mean, or waterz's pivot rank into the cumulative bins."""
    if quantile is None:
        return total / count
    pivot = min(count, max(1, round(count * quantile)))
    return float(np.searchsorted(np.cumsum(hist), pivot)) / (BINS - 1)


def _minimax_cost(height: np.ndarray, start: tuple[int, ...]) -> np.ndarray:
    """Lowest maximum height over 6-connected paths from `start` to every voxel, by Dijkstra."""
    import heapq

    cost = np.full(height.shape, np.inf)
    cost[start] = float(height[start])
    heap = [(cost[start], start)]
    while heap:
        c, p = heapq.heappop(heap)
        if c > cost[p]:
            continue
        for axis in range(3):
            for step in (-1, 1):
                q = list(p)
                q[axis] += step
                if not 0 <= q[axis] < height.shape[axis]:
                    continue
                q = tuple(q)
                reach = max(c, float(height[q]))
                if reach < cost[q]:
                    cost[q] = reach
                    heapq.heappush(heap, (reach, q))
    return cost


@pytest.mark.parametrize("seed", range(5))
def test_seeded_watershed_gives_every_voxel_its_cheapest_seed(seed):
    """The definition, ties included: each voxel's seed is one whose path to it has the lowest
    maximum height. (It replaces `ndimage.watershed_ift`, which corrupts memory at 896^3.)"""
    rng = np.random.default_rng(seed)
    shape = (5, 6, 7)
    height = rng.integers(0, 40, size=shape).astype(np.uint16)       # few levels: ties everywhere
    starts = [np.unravel_index(i, shape) for i in rng.choice(height.size, size=4, replace=False)]
    markers = np.zeros(shape, np.int32)
    for label, start in enumerate(starts, 1):
        markers[start] = label
    labels = seeded_watershed(height, markers)
    assert all(labels[start] == label for label, start in enumerate(starts, 1))
    costs = np.stack([_minimax_cost(height, start) for start in starts])
    chosen = np.take_along_axis(costs, (labels.astype(np.int64) - 1)[None], axis=0)[0]
    assert np.array_equal(chosen, costs.min(axis=0))


def test_seeded_watershed_without_seeds_labels_nothing():
    height = np.zeros((3, 4, 5), np.uint16)
    assert not seeded_watershed(height, np.zeros((3, 4, 5), np.int32)).any()


def test_merge_function_names():
    assert parse_merge_function("mean") is None
    assert parse_merge_function("hist_quant_75") == 0.75
    assert parse_merge_function("hist_quant_100") == 1.0
    for bad in ("median", "hist_quant_0", "hist_quant_101", "hist_quant_x", "hist_quant_"):
        with pytest.raises(ValueError, match="merge function"):
            parse_merge_function(bad)


def test_fragments_label_the_inside_and_never_straddle_two_objects():
    """Inside voxels carry ids 1..n and outside ones 0, as LSD's; filling the outside from the
    nearest inside voxel then covers every voxel without carrying a fragment across a membrane."""
    labels = _boxes()
    frags, inside, n = fragments(_affinities_from_labels(labels).mean(axis=0), 0.5, 3, "distance")
    assert (frags[inside] > 0).all() and not frags[~inside].any()
    assert np.unique(frags[inside]).size == n and (~inside).any()
    full = fill_outside(frags, inside)
    assert full.min() == 1 and np.array_equal(full[inside], frags[inside])
    covered = set()
    for f in range(1, n + 1):
        objects = np.unique(labels[(full == f) & (labels > 0)])
        assert objects.size == 1, f"fragment {f} straddles objects {objects}"
        covered.add(int(objects[0]))
    assert covered == set(range(1, 10))


def test_a_piece_without_a_maximum_still_gets_its_own_fragment():
    """A thin object beside a thick one has no maximum over the 10-voxel window, so LSD's seeding
    leaves it to be flooded from its neighbour's seed, across the membrane. Here it gets its own."""
    labels = np.zeros((30, 20, 20), dtype=np.int64)
    labels[1:15] = 1                                     # thick
    labels[17:19] = 2                                    # two voxels thin, two voxels away
    frags, inside, n = fragments(_affinities_from_labels(labels).mean(axis=0), 0.5, 10, "distance")
    assert ((labels == 2) & inside).any()
    for f in range(1, n + 1):
        assert np.unique(labels[frags == f]).size == 1, f"fragment {f} straddles the membrane"


def test_no_interior_means_no_fragments():
    frags, inside, n = fragments(np.full((6, 6, 6), 0.2, np.float32), 0.5, 3, "distance")
    assert n == 0 and not frags.any() and not inside.any()
    assert fill_outside(frags, inside) is frags


def test_all_interior_is_one_fragment():
    """No outside voxel to measure a distance from: a single basin, not a transform of nothing."""
    frags, inside, n = fragments(np.full((6, 6, 6), 0.9, np.float32), 0.5, 3, "distance")
    assert n == 1 and (frags == 1).all() and inside.all()


@pytest.mark.parametrize("seed", range(3))
def test_region_graph_matches_a_brute_force_count(seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 6, size=(5, 6, 7)).astype(np.int32)     # 0: no fragment
    short = rng.random((3, 5, 6, 7), dtype=np.float32)
    lo, hi, hist, total = region_graph(labels, 5, short)
    expected: dict[tuple[int, int], list[float]] = {}
    for axis in range(3):
        for p in np.ndindex(labels.shape):
            q = list(p)
            q[axis] += 1
            if q[axis] >= labels.shape[axis] or labels[p] == labels[tuple(q)]:
                continue
            if labels[p] == 0 or labels[tuple(q)] == 0:
                continue
            pair = tuple(sorted((int(labels[p]), int(labels[tuple(q)]))))
            expected.setdefault(pair, []).append(float(short[(axis, *p)]))
    assert sorted(zip(lo.tolist(), hi.tolist(), strict=True)) == sorted(expected)
    for k, pair in enumerate(zip(lo.tolist(), hi.tolist(), strict=True)):
        values = np.array(expected[pair])
        assert total[k] == pytest.approx(values.sum())
        bins = np.minimum((values * BINS).astype(int), BINS - 1)
        assert np.array_equal(hist[k], np.bincount(bins, minlength=BINS))


@pytest.mark.parametrize("function", ["mean", "hist_quant_50", "hist_quant_75"])
@pytest.mark.parametrize("seed", range(3))
def test_every_merge_is_the_best_pair_scored_from_scratch(function, seed):
    """Replay the merges: each must be the best-scoring pair of segments at that moment, its score
    the merge function over all original faces between them -- exactness and greediness at once."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(1, 31, size=(9, 9, 9)).astype(np.int32)
    short = rng.random((3, 9, 9, 9), dtype=np.float32)
    lo, hi, hist, total = region_graph(labels, 30, short)
    merges = agglomerate(lo, hi, hist.copy(), total, 31, function, 0.0)
    quantile = parse_merge_function(function)
    segment = {node: {node} for node in range(1, 31)}
    owner = {node: node for node in range(1, 31)}
    for keep, gone, recorded in merges.tolist():
        keep, gone = int(keep), int(gone)
        pooled: dict[tuple[int, int], list] = {}
        for k in range(lo.size):
            a, b = owner[int(lo[k])], owner[int(hi[k])]
            if a == b:
                continue
            stats = pooled.setdefault(tuple(sorted((a, b))), [np.zeros(BINS, np.int64), 0.0, 0])
            stats[0] += hist[k]
            stats[1] += float(total[k])
            stats[2] += int(hist[k].sum())
        scores = {pair: _score(*stats, quantile) for pair, stats in pooled.items()}
        assert recorded == pytest.approx(scores[tuple(sorted((keep, gone)))])
        assert recorded == pytest.approx(max(scores.values())), "not the best pair"
        for node in segment.pop(gone):
            owner[node] = keep
            segment[keep].add(node)
    assert len(segment) == 1, "down to 0.0 every connected segment must have merged"


def test_a_threshold_keeps_the_merges_before_the_first_one_below_it():
    """Scores need not fall monotonically; a run stopping at t ends at the first merge below t."""
    merges = np.array([[1, 2, 0.9], [3, 4, 0.4], [1, 3, 0.8]])
    assert roots_at(merges, 5, 0.5).tolist() == [0, 1, 1, 3, 4]
    assert roots_at(merges, 5, 0.3).tolist() == [0, 1, 1, 1, 1]
    assert roots_at(merges, 5, 0.95).tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("function", ["hist_quant_50", "hist_quant_75", "mean"])
def test_perfect_affinities_give_back_the_objects(function):
    labels = _boxes()
    processor = WatershedAgglomeration(seed_distance=3, merge_functions=[function],
                                       thresholds=[0.5])
    out = processor(_affinities_from_labels(labels), **processor.search_space()[0])
    segments = {obj: np.unique(out[labels == obj]) for obj in range(1, 10)}
    assert all(s.size == 1 for s in segments.values()), segments
    assert len({int(s[0]) for s in segments.values()}) == 9


def test_one_fragmentation_and_one_agglomeration_serve_the_whole_sweep(monkeypatch):
    calls = {"fragments": 0, "agglomerate": 0}
    real_fragments, real_agglomerate = ws_agglo.fragments, ws_agglo.agglomerate

    def counted_fragments(*args, **kwargs):
        calls["fragments"] += 1
        return real_fragments(*args, **kwargs)

    def counted_agglomerate(*args, **kwargs):
        calls["agglomerate"] += 1
        return real_agglomerate(*args, **kwargs)

    monkeypatch.setattr(ws_agglo, "fragments", counted_fragments)
    monkeypatch.setattr(ws_agglo, "agglomerate", counted_agglomerate)
    aff = _affinities_from_labels(_boxes())
    processor = WatershedAgglomeration(seed_distance=3, thresholds=[0.3, 0.6],
                                       min_sizes=[0, 100])
    space = processor.search_space()
    assert len(space) == 2 * 2 * 2                       # functions x thresholds x sizes
    for params in space:
        processor(aff.copy(), **params)                  # a fresh read per candidate, as the runner
    assert calls == {"fragments": 1, "agglomerate": 2}
    assert processor.run_info()["fragments"] > 0


def test_the_route_reads_three_channels_and_describes_its_candidates():
    processor = WatershedAgglomeration(min_sizes=[0, 500])
    assert processor.reads_channels() == 3
    point = {"interior": 0.5, "elevation": "distance", "merge_function": "hist_quant_75",
             "threshold": 0.4, "min_size": 500}
    assert processor.describe(point) == "ws_agglo(hist_quant_75, threshold=0.4, min_size=500)"
    assert processor.describe({**point, "elevation": "affinity", "min_size": 0}) == \
        "ws_agglo(hist_quant_75, threshold=0.4, elevation=affinity)"
    assert len(processor.search_space()) == 2 * 5 * 2
    for bad in ({"merge_functions": ["median"]}, {"thresholds": [1.5]}, {"interiors": [1.0]},
                {"elevations": ["height"]}, {"seed_distance": 0}):
        with pytest.raises(ValueError):
            WatershedAgglomeration(**bad)
