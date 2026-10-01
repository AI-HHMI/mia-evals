"""The signed mutex watershed for three-channel affinities (branch `three-channel-mws`).

The kernel is `mws`'s and already tested against the reference there; what is new is the edge set,
so the oracle is again `mutex_watershed_reference`, run on the signed edges.
"""

from __future__ import annotations

import numpy as np
import pytest

from postprocess.mws import mutex_watershed_reference
from postprocess.mws3 import (
    ThreeChannelMutexWatershed,
    build_signed_edges,
    count_signed_edges,
    segment_signed,
)

pytestmark = pytest.mark.unit

FORWARD = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def _canonical(labels) -> np.ndarray:
    """Relabel by order of first appearance, so only the partition matters."""
    labels = np.asarray(labels).ravel()
    _, first_index, inverse = np.unique(labels, return_index=True, return_inverse=True)
    remap = np.empty(first_index.size, dtype=np.int64)
    remap[np.argsort(first_index)] = np.arange(first_index.size)
    return remap[inverse]


def _affinities_from_labels(labels: np.ndarray) -> np.ndarray:
    """(3, *shape) forward nearest-neighbour affinities: 1 where p and p + e_i share an object."""
    aff = np.zeros((3, *labels.shape), dtype=np.float32)
    for axis in range(3):
        lo = tuple(slice(0, -1) if a == axis else slice(None) for a in range(3))
        hi = tuple(slice(1, None) if a == axis else slice(None) for a in range(3))
        aff[axis][lo] = (labels[lo] == labels[hi]) & (labels[lo] > 0)
    return aff


def _thresholded_components(aff: np.ndarray) -> np.ndarray:
    """Connected components of the graph of edges above 0.5, one id per voxel, flat."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    u, v, _, attractive = build_signed_edges(aff)
    n = int(np.prod(aff.shape[1:]))
    graph = coo_matrix((np.ones(int(attractive.sum())), (u[attractive], v[attractive])),
                       shape=(n, n))
    return connected_components(graph, directed=False)[1]


@pytest.mark.parametrize("shape", [(12, 12, 12), (5, 17, 9), (3, 3, 3), (1, 4, 6)])
def test_count_signed_edges_matches_build(shape):
    aff = np.zeros((3, *shape), dtype=np.float32)
    assert count_signed_edges(shape) == build_signed_edges(aff)[0].size


@pytest.mark.parametrize("value, attractive, priority",
                         [(0.9, True, 0.9), (0.5, False, 0.5), (0.1, False, 0.9)])
def test_an_edge_takes_its_sign_from_its_value(value, attractive, priority):
    aff = np.zeros((3, 2, 1, 1), dtype=np.float32)       # one edge: voxel 0 -> voxel 1 along x
    aff[0, 0, 0, 0] = value
    u, v, p, a = build_signed_edges(aff)
    assert u.tolist() == [0] and v.tolist() == [1]
    assert a.tolist() == [attractive] and p[0] == pytest.approx(priority)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("shape", [(14, 14, 14), (6, 23, 11), (18, 9, 21)])
def test_segment_reproduces_the_reference_on_the_signed_edges(seed, shape):
    rng = np.random.default_rng(seed)
    # float16-quantised like the artifacts on disk, which makes exact ties abundant.
    aff = rng.random((3, *shape), dtype=np.float32).astype(np.float16).astype(np.float32)
    expected = mutex_watershed_reference(*build_signed_edges(aff), int(np.prod(shape)))
    info: dict = {}
    labels = segment_signed(aff, info=info)
    assert info["edges"] == count_signed_edges(shape)
    assert labels.shape == shape and labels.min() == 1
    assert np.array_equal(_canonical(expected), _canonical(labels))


def test_segment_matches_on_smooth_affinities_with_large_clusters():
    """Random affinities give small clusters; a smooth field gives few large ones, the regime where
    merges move long partner chains and the kernel's tables grow."""
    rng = np.random.default_rng(3)
    z, y, x = np.meshgrid(*[np.linspace(0, 3 * np.pi, 24)] * 3, indexing="ij")
    field = (np.sin(z) * np.cos(y) + np.sin(x)) / 2
    aff = np.clip(0.5 + 0.45 * field[None] + 0.05 * rng.standard_normal((3, 24, 24, 24)), 0, 1)
    aff = aff.astype(np.float16).astype(np.float32)
    expected = mutex_watershed_reference(*build_signed_edges(aff), 24 ** 3)
    assert np.array_equal(_canonical(expected), _canonical(segment_signed(aff)))


@pytest.mark.parametrize("seed", range(3))
def test_never_coarser_than_thresholded_components_at_one_half(seed):
    """Every merge follows an edge above 0.5, so each segment lies inside one component."""
    rng = np.random.default_rng(seed)
    aff = (0.25 + 0.5 * rng.random((3, 12, 12, 12), dtype=np.float32)).astype(np.float32)
    labels = segment_signed(aff).ravel()
    components = _thresholded_components(aff)
    pairs = np.unique(np.stack([labels, components]), axis=1)
    assert np.unique(pairs[0]).size == pairs.shape[1], "a segment spans two components"
    assert np.unique(labels).size > np.unique(components).size, "expected strictly finer here"


def test_a_weak_leak_through_a_clear_membrane_does_not_merge():
    """Two objects separated by a plane of confident cuts and joined by one weak attractive edge: a
    threshold at 0.5 merges them through the leak, the signed watershed keeps them apart."""
    side, cut = 8, 3
    aff = np.full((3, side, side, side), 0.95, dtype=np.float32)
    aff[0, cut] = 0.02                                   # x edges from plane 3 to plane 4: cut...
    aff[0, cut, 3, 3] = 0.6                              # ...except one weak attractive leak
    assert np.unique(_thresholded_components(aff)).size == 1, "the threshold merges through it"
    labels = segment_signed(aff)
    assert np.unique(labels[: cut + 1]).size == 1 and np.unique(labels[cut + 1:]).size == 1
    assert labels[0, 0, 0] != labels[-1, 0, 0], "the leak merged the two objects"


def test_perfect_affinities_reproduce_touching_objects():
    labels = np.zeros((16, 16, 16), dtype=np.int64)
    labels[:8] = 1
    labels[8:, :8] = 2
    labels[8:, 8:] = 3
    labels[2:5, 2:5, 2:5] = 0                            # a background pocket inside object 1
    out = segment_signed(_affinities_from_labels(labels))
    segments = {obj: np.unique(out[labels == obj]) for obj in (1, 2, 3)}
    assert all(s.size == 1 for s in segments.values()), segments
    assert len({int(s[0]) for s in segments.values()}) == 3


def test_the_route_reads_three_channels_and_sweeps_the_filter_then_the_fill():
    processor = ThreeChannelMutexWatershed(min_sizes=[500, 0], fill_distances=["all", 0])
    assert processor.reads_channels() == 3
    assert processor.search_space() == [
        {"min_size": 0, "fill_distance": 0}, {"min_size": 0, "fill_distance": "all"},
        {"min_size": 500, "fill_distance": 0}, {"min_size": 500, "fill_distance": "all"},
    ]
    assert processor.describe({"min_size": 500, "fill_distance": "all"}) == \
        "mws3(min_size=500, fill=all)"
    assert processor.describe({"min_size": 0}) == "mws3"
    assert ThreeChannelMutexWatershed().search_space() == [{"min_size": 0}]
    with pytest.raises(ValueError, match="unknown key"):
        ThreeChannelMutexWatershed(repulsive_strides=[1])


def test_the_route_on_six_channels_uses_their_short_range_half_and_filters():
    rng = np.random.default_rng(5)
    aff = rng.random((6, 10, 10, 10), dtype=np.float32)
    # Two processors, so the second result is computed rather than found in the first's cache.
    assert np.array_equal(_canonical(ThreeChannelMutexWatershed()(aff, min_size=0)),
                          _canonical(ThreeChannelMutexWatershed()(aff[:3].copy(), min_size=0)))
    filtered = ThreeChannelMutexWatershed()(aff, min_size=4)
    sizes = np.bincount(filtered.ravel())
    assert (sizes[1:][sizes[1:] > 0] >= 4).all() and sizes[0] > 0


def test_offsets_are_checked_and_backward_edges_get_their_own_message(tmp_path):
    from artifact import open_artifact, write_artifact
    from postprocess.ws_agglo import WatershedAgglomeration

    def stored(name, offsets, channels=3):
        attrs = {} if offsets is None else {"offsets": offsets}
        return open_artifact(write_artifact(
            tmp_path / f"{name}.zarr", np.zeros((channels, 4, 4, 4), np.float16), "affinity",
            **attrs))

    six = FORWARD + [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
    for processor in (ThreeChannelMutexWatershed(), WatershedAgglomeration()):
        name = type(processor).__name__
        processor.check_artifact(stored(f"forward_{name}", FORWARD))
        processor.check_artifact(stored(f"undeclared_{name}", None))
        processor.check_artifact(stored(f"six_{name}", six, channels=6))
        with pytest.raises(ValueError, match="pointing backwards"):
            processor.check_artifact(stored(f"backward_{name}", [[-1, 0, 0], [0, -1, 0],
                                                                 [0, 0, -1]]))
        with pytest.raises(ValueError, match="reads them as"):
            processor.check_artifact(stored(f"swapped_{name}", FORWARD[::-1]))


def test_the_six_channel_routes_refuse_a_short_range_only_artifact(tmp_path):
    from artifact import open_artifact, write_artifact
    from postprocess.mws import MutexWatershed
    from postprocess.mws_blockwise import BlockwiseMutexWatershed

    artifact = open_artifact(write_artifact(tmp_path / "three.zarr",
                                            np.zeros((3, 4, 4, 4), np.float16), "affinity"))
    for processor in (MutexWatershed(repulsive_strides=[1]), BlockwiseMutexWatershed()):
        with pytest.raises(ValueError, match="mws3 or ws_agglo"):
            processor.check_artifact(artifact)
