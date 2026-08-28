"""The compiled mutex-watershed kernel must produce the same partition as the reference.

There is no external oracle for this algorithm. BANIS stubs it out (`raise NotImplementedError`),
PyPI's `affogato` is an unrelated package, and the real C++ affogato is CMake-only. So the readable
Python implementation in `postprocess.mws` *is* the specification, and these tests are the entire
correctness argument for the compiled version.

Partitions are compared invariant to representative choice. The union-by-size heuristic is free to
keep either member of a merged pair as the root, so two correct implementations can label the same
clusters differently. An earlier version of this comparison sorted by root id and reported 11
failures against a kernel whose partitions were in fact identical -- hence `canonical` below, and
hence a test for `canonical` itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from postprocess.mws import mutex_watershed  # noqa: E402
from postprocess.mws_kernel import mutex_watershed_kernel  # noqa: E402

pytestmark = pytest.mark.unit


def canonical(labels) -> np.ndarray:
    """Relabel by order of first appearance, so only the partition matters."""
    labels = np.asarray(labels)
    _, first_index, inverse = np.unique(labels, return_index=True, return_inverse=True)
    by_appearance = np.argsort(first_index)
    remap = np.empty_like(by_appearance)
    remap[by_appearance] = np.arange(by_appearance.size)
    return remap[inverse]


def run_kernel(u, v, priority, attractive, n_nodes):
    order = np.argsort(-priority, kind="stable").astype(np.int64)
    want = max(64, 8 * int(u.size))
    capacity = 1
    while capacity < want:
        capacity <<= 1
    parent, _, _ = mutex_watershed_kernel(
        u.astype(np.int64), v.astype(np.int64), order, attractive.astype(np.bool_),
        np.int64(n_nodes), np.int64(capacity), np.int64(want))
    return parent


def test_canonical_ignores_which_member_became_the_representative():
    """Guards the comparison itself, using roots from a real disagreement that turned out not to
    be one."""
    reference_roots = [0, 1, 2, 3, 4, 5, 4, 7, 8, 5, 10]
    kernel_roots = [0, 1, 2, 3, 6, 5, 6, 7, 8, 5, 10]   # same clusters, different reps
    assert np.array_equal(canonical(reference_roots), canonical(kernel_roots))


def test_canonical_still_detects_a_different_partition():
    assert not np.array_equal(canonical([0, 0, 1, 1]), canonical([0, 1, 0, 1]))


@pytest.mark.parametrize("seed", range(12))
def test_kernel_matches_reference_on_random_graphs(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 40))
    m = int(rng.integers(n, 6 * n))
    u = rng.integers(0, n, size=m).astype(np.int64)
    v = rng.integers(0, n, size=m).astype(np.int64)
    keep = u != v
    u, v = u[keep], v[keep]
    if u.size == 0:
        pytest.skip("degenerate draw: no edges after removing self-loops")
    priority = rng.random(u.size).astype(np.float32)
    attractive = rng.random(u.size) < 0.6
    expected = canonical(mutex_watershed(u, v, priority, attractive, n))
    assert np.array_equal(expected, canonical(run_kernel(u, v, priority, attractive, n)))


def test_a_repulsive_edge_blocks_a_later_attractive_one():
    """The algorithm's defining behaviour, stated directly rather than via a random draw."""
    u = np.array([0, 0], dtype=np.int64)
    v = np.array([1, 1], dtype=np.int64)
    priority = np.array([0.9, 0.5], dtype=np.float32)   # repulsive first, so it wins
    attractive = np.array([False, True])
    parent = run_kernel(u, v, priority, attractive, 2)
    assert canonical(parent).tolist() == [0, 1], "the mutex should have prevented the merge"


def test_an_attractive_edge_merges_when_nothing_forbids_it():
    u = np.array([0], dtype=np.int64)
    v = np.array([1], dtype=np.int64)
    parent = run_kernel(u, v, np.array([0.9], dtype=np.float32), np.array([True]), 2)
    assert canonical(parent).tolist() == [0, 0]


def test_too_small_a_pair_table_raises_rather_than_corrupting():
    """Capacity is checked, not assumed: a silently overflowing table would corrupt the partition
    in a way only a comparison against the reference could catch -- and at whole-volume scale there
    is no reference to compare against."""
    rng = np.random.default_rng(0)
    n, m = 200, 2000
    u = rng.integers(0, n, size=m).astype(np.int64)
    v = rng.integers(0, n, size=m).astype(np.int64)
    keep = u != v
    u, v = u[keep], v[keep]
    priority = rng.random(u.size).astype(np.float32)
    attractive = np.zeros(u.size, dtype=bool)       # all repulsive: maximum pair insertions
    order = np.argsort(-priority, kind="stable").astype(np.int64)
    with pytest.raises(RuntimeError, match="too small"):
        mutex_watershed_kernel(u, v, order, attractive, np.int64(n), np.int64(64), np.int64(64))
