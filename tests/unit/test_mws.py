"""Mutex watershed, checked against cases whose answers follow from the definition.

Ported from `mia_score_mws.py --self-test`, which was a function inside a script and therefore ran
only when someone remembered to pass the flag. The cases are unchanged; what changed is that they
now run in CI.

Written from the paper rather than checked against a reference implementation, because the
reference (`affogato`) is CMake-only and does not pip-install -- so these cases are the only thing
standing between the implementation and a plausible-looking partition.
"""

from __future__ import annotations

import numpy as np
import pytest

from postprocess.mws import (
    LONG, build_edges, count_edges, mutex_watershed_reference, segment,
)

pytestmark = pytest.mark.unit


def test_definitional_cases():
    # 1. Two nodes, one attractive edge -> one cluster.
    out = mutex_watershed_reference(np.array([0]), np.array([1]), np.array([0.9]), np.array([True]), 2)
    assert len(set(out.tolist())) == 1, out

    # 2. The same pair, but a stronger repulsive edge first -> two clusters. This is the whole
    #    point: repulsion seen earlier blocks a later merge.
    out = mutex_watershed_reference(np.array([0, 0]), np.array([1, 1]), np.array([0.9, 0.5]),
                          np.array([False, True]), 2)
    assert len(set(out.tolist())) == 2, out

    # 3. Weaker repulsion, stronger attraction -> merged, because the merge is processed first and
    #    the mutex arrives too late to undo it.
    out = mutex_watershed_reference(np.array([0, 0]), np.array([1, 1]), np.array([0.9, 0.2]),
                          np.array([True, False]), 2)
    assert len(set(out.tolist())) == 1, out

    # 4. Transitivity of the constraint: a-b merge, b-c mutex, then a-c attractive must be blocked.
    out = mutex_watershed_reference(
        np.array([0, 1, 0]), np.array([1, 2, 2]), np.array([0.9, 0.8, 0.7]),
        np.array([True, False, True]), 3)
    assert out[0] == out[1] != out[2], out

    # NOTE both remaining cases use a volume LARGER than the long-range offset. At 8^3 with
    # LONG=10 there are no long-range edges at all, so a test there would pass or fail for
    # reasons that have nothing to do with repulsion.
    side = 3 * LONG

    # 5. Repulsive channels at affinity 1 mean repulsion 0, i.e. no constraints, so every
    #    connected component collapses to one label -- MWS degenerates to connected components
    #    over the attractive graph, as it must.
    rng = np.random.default_rng(0)
    aff = np.zeros((6, side, side, side), dtype=np.float32)
    aff[:3] = rng.random((3, side, side, side))    # arbitrary attractive weights
    aff[3:] = 1.0
    labels = segment(aff, 1)
    assert len(np.unique(labels)) == 1, np.unique(labels)

    # 6. A plane of repulsion across x splits the volume. The short-range x edge at the plane is
    #    cut, and every long-range x pair straddling it repels.
    cut = side // 2
    aff = np.zeros((6, side, side, side), dtype=np.float32)
    aff[:3] = 0.9
    aff[3:] = 1.0
    aff[0, cut] = 0.0                              # attractive x edge across the plane: no pull
    aff[3, max(cut - LONG + 1, 0) : cut + 1] = 0.0  # long-range x pairs straddling it: full push
    labels = segment(aff, 1)
    assert len(np.unique(labels)) >= 2, f"expected a split, got {np.unique(labels)}"


def test_mws_search_space_is_the_cross_product_and_names_min_size():
    """The size filter matters more for mws than for cc_threshold; see the class docstring.

    On kasthuri15_ac4 mws recovered 192/273 objects and cut voi_merge 6.753 -> 0.550, yet scored
    PQ 0.0049 because it also returned 57,350 tiny fragments. The parameter must therefore be
    swept and must appear in the description, or a leaderboard row cannot be read.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from postprocess.mws import MutexWatershed

    processor = MutexWatershed(repulsive_strides=[1, 2], min_sizes=[500, 0])
    assert processor.search_space() == [
        {"repulsive_stride": 1, "min_size": 0},
        {"repulsive_stride": 1, "min_size": 500},
        {"repulsive_stride": 2, "min_size": 0},
        {"repulsive_stride": 2, "min_size": 500},
    ]
    assert "min_size" not in processor.describe({"repulsive_stride": 1, "min_size": 0})
    assert "min_size=500" in processor.describe({"repulsive_stride": 1, "min_size": 500})


def test_mws_default_applies_no_size_filter():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from postprocess.mws import MutexWatershed

    assert MutexWatershed(repulsive_strides=[1]).search_space() == [
        {"repulsive_stride": 1, "min_size": 0}
    ]


# --- the production path must reproduce the reference, partition for partition ------------------
#
# `segment()` scores every mws record. It runs the compiled kernel, in memory or with the edges
# streamed through disk, and both must give exactly the partition the Python reference gives: the
# labels may be numbered differently, which no metric sees, but no voxel may change cluster.


def _canonical(labels) -> np.ndarray:
    """Relabel by order of first appearance, so only the partition matters."""
    labels = np.asarray(labels).ravel()
    _, first_index, inverse = np.unique(labels, return_index=True, return_inverse=True)
    remap = np.empty(first_index.size, dtype=np.int64)
    remap[np.argsort(first_index)] = np.arange(first_index.size)
    return remap[inverse]


def _float16_affinities(shape, seed):
    """float16-quantised like the artifacts on disk, which makes exact ties abundant."""
    rng = np.random.default_rng(seed)
    return rng.random((6, *shape), dtype=np.float32).astype(np.float16).astype(np.float32)


def _reference_labels(aff, stride):
    u, v, priority, attractive = build_edges(aff, stride)
    return mutex_watershed_reference(u, v, priority, attractive, int(np.prod(aff.shape[1:])))


@pytest.mark.parametrize("shape", [(12, 12, 12), (5, 17, 9), (20, 8, 14), (3, 3, 3)])
@pytest.mark.parametrize("stride", [1, 2, 3])
def test_count_edges_matches_build_edges(shape, stride):
    aff = np.zeros((6, *shape), dtype=np.float32)
    assert count_edges(shape, stride) == build_edges(aff, stride)[0].size


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("shape", [(14, 14, 14), (6, 23, 11), (18, 9, 21)])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("path", ["memory", "stream"])
def test_segment_reproduces_the_reference_partition(monkeypatch, tmp_path, seed, shape, stride,
                                                   path):
    aff = _float16_affinities(shape, seed)
    info = {}
    if path == "memory":
        labels = segment(aff, stride, info=info)
        assert info["implementation"] == "compiled kernel, edges in memory"
    else:
        # Small blocks, so every volume here is several blocks per axis and the per-block stride
        # subsampling has to line up with the whole-volume one (the block is a multiple of it).
        from postprocess import mws as module
        monkeypatch.setattr(module, "STREAM_BLOCK", 8)
        labels = segment(aff, stride, max_in_memory_edges=0, scratch=tmp_path, info=info)
        assert info["implementation"] == "compiled kernel, edges streamed through disk"
        assert not any(tmp_path.rglob("bucket_*")), "the streamed edges must be cleaned up"
    assert info["edges"] == count_edges(shape, stride)
    assert labels.shape == shape and labels.min() == 1
    assert np.array_equal(_canonical(_reference_labels(aff, stride)), _canonical(labels))


def test_segment_matches_on_smooth_affinities_with_large_clusters():
    """Random affinities give small clusters; a smooth field gives few large ones, the regime where
    merges move long partner chains and the tables grow."""
    rng = np.random.default_rng(3)
    z, y, x = np.meshgrid(*[np.linspace(0, 3 * np.pi, 24)] * 3, indexing="ij")
    field = (np.sin(z) * np.cos(y) + np.sin(x)) / 2
    aff = np.clip(0.5 + 0.45 * field[None] + 0.05 * rng.standard_normal((6, 24, 24, 24)), 0, 1)
    aff = aff.astype(np.float16).astype(np.float32)
    for stride in (1, 2):
        expected = _canonical(_reference_labels(aff, stride))
        assert np.array_equal(expected, _canonical(segment(aff, stride)))


def test_streaming_without_a_scratch_directory_is_refused():
    aff = _float16_affinities((6, 6, 6), 0)
    with pytest.raises(ValueError, match="scratch"):
        segment(aff, 1, max_in_memory_edges=0)


def test_the_postprocessor_streams_through_its_scratch_and_says_so(tmp_path):
    from postprocess.mws import MutexWatershed

    aff = _float16_affinities((10, 10, 10), 1)
    in_memory = MutexWatershed(repulsive_strides=[1])
    held = in_memory(aff.copy(), repulsive_stride=1, min_size=0)
    assert in_memory.run_info()["implementation"] == "compiled kernel, edges in memory"

    streaming = MutexWatershed(repulsive_strides=[1], max_in_memory_edges=0)
    streaming.use_scratch(tmp_path)
    streamed = streaming(aff.copy(), repulsive_stride=1, min_size=0)
    assert streaming.run_info()["implementation"] == "compiled kernel, edges streamed through disk"
    assert np.array_equal(_canonical(held), _canonical(streamed))
