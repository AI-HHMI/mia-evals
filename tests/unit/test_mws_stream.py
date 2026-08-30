"""Streaming mutex watershed must equal the one-shot kernel exactly, at every block size.

The point of streaming over blockwise-with-stitching is that it is *exact*: bucketing by priority
reproduces the global edge ordering that the algorithm is defined on. That is only checkable at a
scale where the one-shot path also runs, so it is checked on small synthetic volumes -- and across
several block sizes, because a single block exercises none of the boundary handling.

Uses synthetic affinities rather than corpus data so it runs anywhere. The corpus-scale version
(64^3 and 128^3 of real affinities, three block sizes each) lives in
/nrs/scicompsoft/orhane/mia-train-scratch/lmd1_arm2_eval/test_mws_stream.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from postprocess.mws import build_edges, mutex_watershed  # noqa: E402
from postprocess.mws_kernel import mutex_watershed_kernel  # noqa: E402
from postprocess.mws_stream import (  # noqa: E402
    EDGE,
    block_records,
    offset_steps,
    segment_streaming,
)

pytestmark = pytest.mark.unit


def canonical(labels) -> np.ndarray:
    labels = np.asarray(labels).ravel()
    _, first_index, inverse = np.unique(labels, return_index=True, return_inverse=True)
    by_appearance = np.argsort(first_index)
    remap = np.empty_like(by_appearance)
    remap[by_appearance] = np.arange(by_appearance.size)
    return remap[inverse]


def affinities(shape, seed=0):
    """float16-quantised, because that is what makes exact ties abundant on real data -- and ties
    are what a decomposition-dependent tie-break gets wrong."""
    rng = np.random.default_rng(seed)
    return rng.random((6, *shape), dtype=np.float32).astype(np.float16).astype(np.float32)


def one_shot(aff):
    n = int(np.prod(aff.shape[1:]))
    u, v, priority, attractive = build_edges(aff, 1)
    order = np.argsort(-priority, kind="stable").astype(np.int64)
    want = max(1024, 16 * n)
    capacity = 1
    while capacity < want:
        capacity <<= 1
    parent, _, _ = mutex_watershed_kernel(
        u.astype(np.int64), v.astype(np.int64), order, attractive.astype(np.bool_),
        np.int64(n), np.int64(capacity), np.int64(want))
    return parent


@pytest.mark.parametrize("size,block", [(16, 16), (16, 8), (16, 4), (24, 12), (24, 8)])
def test_streaming_equals_one_shot(tmp_path, size, block):
    aff = affinities((size, size, size))
    shape = tuple(aff.shape[1:])

    def read_block(origin, sz):
        return aff[(slice(None),) + tuple(slice(o, o + s) for o, s in zip(origin, sz, strict=True))]

    streamed, stats = segment_streaming(read_block, shape, tmp_path, block=block, n_buckets=32)
    assert np.array_equal(canonical(one_shot(aff)), canonical(streamed)), (
        f"streaming with block={block} disagrees with the one-shot kernel"
    )
    # The high-water marks are what a caller sizes the next, larger run from, so they must be
    # reported and must be under the capacity that was actually allocated.
    assert 0 < stats["pair_insertions"] <= stats["pair_capacity"] // 2
    assert 0 < stats["pool_used"] <= stats["pool_capacity"]
    assert stats["edges"] > 0


def test_streaming_also_equals_the_python_reference(tmp_path):
    """Transitively implied, but stated directly: the reference is the specification."""
    aff = affinities((16, 16, 16), seed=3)
    shape = tuple(aff.shape[1:])
    u, v, priority, attractive = build_edges(aff, 1)
    reference = mutex_watershed(u, v, priority, attractive, int(np.prod(shape)))

    def read_block(origin, sz):
        return aff[(slice(None),) + tuple(slice(o, o + s) for o, s in zip(origin, sz, strict=True))]

    streamed, _ = segment_streaming(read_block, shape, tmp_path, block=8, n_buckets=32)
    assert np.array_equal(canonical(reference), canonical(streamed))


def test_block_decomposition_emits_every_edge_exactly_once():
    """The edge multiset must not depend on the block size.

    Checked separately from the partition because the two failed independently: when streaming
    disagreed at smaller blocks, the edge multiset was provably identical (162,816 edges, none
    missing, none duplicated) and the fault was tie-breaking. Keeping them apart means a future
    regression says which of the two broke.
    """
    aff = affinities((12, 12, 12), seed=1)
    shape = tuple(aff.shape[1:])
    steps = offset_steps(shape)
    u, v, _, _ = build_edges(aff, 1)
    expected = {(int(a), int(b)) for a, b in zip(u, v, strict=True)}
    assert len(expected) == u.size, "build_edges emitted a duplicate edge"

    for block in (12, 6, 4, 3):
        seen: set[tuple[int, int]] = set()
        count = 0
        for z in range(0, shape[0], block):
            for y in range(0, shape[1], block):
                for x in range(0, shape[2], block):
                    origin = (z, y, x)
                    size = tuple(min(block, s - o) for s, o in zip(shape, origin, strict=True))
                    sl = tuple(slice(o, o + s) for o, s in zip(origin, size, strict=True))
                    records = block_records(aff[(slice(None),) + sl], origin, shape, 1)
                    count += records.size
                    seen.update(
                        zip(records["source"].tolist(),
                            (records["source"] + steps[records["offset"]]).tolist(),
                            strict=True)
                    )
        assert count == u.size, f"block={block} emitted {count} edges, expected {u.size}"
        assert seen == expected, f"block={block} emitted a different edge set"


def test_edge_record_is_thirteen_bytes_with_a_float32_priority():
    """float16 priority was tried and reordered the edge list, changing the partition. The dtype is
    load-bearing, so a change to it should fail here rather than silently alter results."""
    assert EDGE.itemsize == 13
    assert EDGE["priority"] == np.dtype("<f4")
