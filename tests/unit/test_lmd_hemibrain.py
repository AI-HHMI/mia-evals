"""truth.lmd_hemibrain's steps against what they stand in for: LSD's relabelling and erosion."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

pytest.importorskip("cc3d")
pytest.importorskip("fastremap")


def lsd_erode(labels: np.ndarray) -> np.ndarray:
    """LSD's erode_in_block (erode_blockwise_3d.py) on one block: each label, whole array."""
    foreground = np.zeros(labels.shape, dtype=bool)
    for label in np.unique(labels):
        if label == 0:
            continue
        foreground |= ndimage.binary_erosion(labels == label, iterations=2, border_value=1)
    out = labels.copy()
    out[~foreground] = 0
    return out


@pytest.mark.unit
def test_erosion_in_grown_bounding_boxes_is_lsds_whole_array_loop():
    """Pieces touching each other, the array's faces and background, thick and one voxel thin."""
    from truth.lmd_hemibrain import erode, pieces_of

    rng = np.random.default_rng(0)
    labels = np.kron(rng.integers(1, 5, size=(6, 6, 6)), np.ones((5, 5, 5), dtype=np.int64))
    labels = labels.astype(np.uint64)
    labels[rng.random(labels.shape) < 0.03] = 0
    labels[:, 12, 3] = 9                                   # a 1-voxel thread across the array
    pieces, _ = pieces_of(labels)
    expected = lsd_erode(pieces)
    removed = erode(pieces)
    assert np.array_equal(pieces, expected)
    assert removed > 0 and removed == int((expected == 0).sum() - (labels == 0).sum())


@pytest.mark.unit
def test_pieces_are_each_bodys_face_connected_components():
    """Corner contact does not join; another body cutting through splits."""
    from truth.lmd_hemibrain import pieces_of

    labels = np.zeros((8, 8, 8), dtype=np.uint64)
    labels[1:3, 1:3, 1:3] = 7_000_000_007                   # one body, two cubes at a corner
    labels[3:5, 3:5, 3:5] = 7_000_000_007
    labels[:, 6, 6] = 9                                    # a rod cut by another body
    labels[4, 6, 6] = 8
    pieces, body_of = pieces_of(labels)
    for body in (7_000_000_007, 8, 9):
        mask = labels == body
        reference, count = ndimage.label(mask)
        pairs = set(zip(pieces[mask].tolist(), reference[mask].tolist(), strict=True))
        assert len(pairs) == count == len(set(pieces[mask].tolist()))
        assert all(body_of[piece] == body for piece, _ in pairs)
    assert len(body_of) == 5


@pytest.mark.unit
def test_largest_piece_per_neuron_keeps_one_skeleton_each_unchanged():
    import networkx as nx

    from truth.lmd_hemibrain import largest_pieces

    graph = nx.Graph()
    graph.graph.update(source="kimimaro", region_box=[[0, 9]] * 3)
    for n, (piece, body) in enumerate([(1, 5), (1, 5), (2, 5), (2, 5), (3, 6)]):
        graph.add_node(10 + n, id=piece, neuron_id=piece, body_id=body, index_position=(n, 0, 0),
                       nm_position=(8.0 * n, 0.0, 0.0))
    graph.add_edges_from([(10, 11), (12, 13)])
    largest = largest_pieces(graph, {1: 100, 2: 50, 3: 7})
    assert sorted(largest.nodes) == [0, 1, 2]
    assert [largest.nodes[n]["neuron_id"] for n in range(3)] == [1, 1, 3]
    positions = [largest.nodes[n]["index_position"] for n in range(3)]
    assert positions == [(0, 0, 0), (1, 0, 0), (4, 0, 0)]
    assert list(largest.edges) == [(0, 1)]
    assert largest.graph["selection"] == "largest piece per neuron"
    tie = largest_pieces(graph, {1: 50, 2: 50, 3: 7})
    assert tie.nodes[0]["neuron_id"] == 1                  # a tie goes to the lower id
