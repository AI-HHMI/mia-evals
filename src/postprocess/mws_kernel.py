"""The mutex watershed inner loop, compiled.

Split out from `mws.py` because it is the only part that needed rewriting and the only part that
must not drift from the reference: `mutex_watershed_reference` in `mws.py` stays as the readable
definition of the algorithm, and `tests/unit/test_mws.py` requires the two produce *identical*
labellings. There is no other oracle available -- BANIS stubs mutex watershed out with
`NotImplementedError`, PyPI's `affogato` is an unrelated package, and the real C++ affogato is
CMake-only -- so agreement with the reference implementation is the whole of the correctness story.

Why compile it at all, measured on a 256^3 block of liconn_mouse_hippocampus (98.5 M edges):

    build_edges     4.99 s    1.3%
    argsort        10.72 s    2.7%
    union-find    377.07 s   96.0%      0.26 M edges/s
    total         392.8 s = 23.4 us/voxel,  peak RSS 9.4 GB

At that rate the zebrafish doublecube (7.078 G voxels, 41.6 G edges) is 46 hours. And of the 9.4 GB
peak only 3.3 GB is arrays: the other 6.1 GB is the Python `dict[int, set[int]]` holding mutex
partners, which at full volume scale would be some 2.6 TB. So compiling the loop is not only a
speed fix, it removes the dominant memory term as well.

**The edge arrays are the term it does not fix**: 21 B/edge over 41.6 G edges is 0.87 TB, which
still cannot be held. That needs priority-bucketed streaming (or blockwise processing, which
changes the answer because global edge order is what the algorithm is defined on). This module is
step one of two, and on its own it makes whole volumes faster, not possible.

Mutex storage: partners live in one flat pool with a per-root singly-linked chain, and membership is
tested against a global open-addressed hash set of packed root pairs. The two structures answer the
two different questions the loop asks -- "iterate this root's partners" when merging, and "are these
two roots forbidden" on every attractive edge -- and neither is fast at the other. Measured decision
mix on that block: 57% same-root, 21% mutex added, 17% merged, 4% blocked, so the membership test
runs about 21 M times and must be O(1).
"""

from __future__ import annotations

import numba
import numpy as np
from numba import njit

#: Empty slot marker for the pair hash. Node ids are non-negative, so -1 cannot collide.
EMPTY = np.int64(-1)


@njit(cache=True, nogil=True)
def _find(parent: np.ndarray, x: np.int64) -> np.int64:
    """Union-find root with full path compression."""
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        nxt = parent[x]
        parent[x] = root
        x = nxt
    return root


@njit(inline="always")
def _pack(a: np.int64, b: np.int64, n_nodes: np.int64) -> np.int64:
    """Order-independent key for a root pair."""
    if a < b:
        return a * n_nodes + b
    return b * n_nodes + a


@njit(inline="always")
def _hash_slot(key: np.int64, mask: np.int64) -> np.int64:
    # Fibonacci hashing: key * 2^64/phi, taking the high bits via the mask after a shift. Cheap and
    # adequate here -- the keys are structured (a * n + b), so the low bits alone collide heavily.
    h = np.uint64(key) * np.uint64(11400714819323198485)
    return np.int64((h >> np.uint64(29)) & np.uint64(mask))


@njit(cache=True, nogil=True)
def _pair_find(keys: np.ndarray, key: np.int64, mask: np.int64) -> np.int64:
    """Slot holding `key`, or the first empty slot where it would be inserted."""
    slot = _hash_slot(key, mask)
    while True:
        current = keys[slot]
        if current == EMPTY or current == key:
            return slot
        slot = (slot + 1) & mask


@njit(cache=True, nogil=True)
def mutex_watershed_kernel(
    u: np.ndarray,
    v: np.ndarray,
    order: np.ndarray,
    attractive: np.ndarray,
    n_nodes: np.int64,
    pair_capacity: np.int64,
    pool_capacity: np.int64,
) -> tuple[np.ndarray, np.int64, np.int64]:
    """Edges in `order` -> a root per node. Returns (parent, pairs_used, pool_used).

    `pair_capacity` must be a power of two and comfortably larger than the number of distinct
    forbidden root pairs that coexist; `pool_capacity` bounds total partner-chain entries. Both are
    returned as high-water marks so the caller can detect a too-small guess rather than silently
    corrupting the result -- exceeding either raises.
    """
    parent = np.arange(n_nodes, dtype=np.int64)

    # Global set of forbidden root pairs, for O(1) membership.
    pair_keys = np.full(pair_capacity, EMPTY, dtype=np.int64)
    mask = pair_capacity - 1
    pairs_used = np.int64(0)

    # Per-root partner chains: head[root] -> pool index, pool_next linking, pool_val the partner.
    head = np.full(n_nodes, -1, dtype=np.int64)
    chain_len = np.zeros(n_nodes, dtype=np.int64)
    pool_val = np.empty(pool_capacity, dtype=np.int64)
    pool_next = np.empty(pool_capacity, dtype=np.int64)
    pool_used = np.int64(0)

    for idx in range(order.shape[0]):
        e = order[idx]
        ru = _find(parent, u[e])
        rv = _find(parent, v[e])
        if ru == rv:
            continue

        key = _pack(ru, rv, n_nodes)
        slot = _pair_find(pair_keys, key, mask)
        forbidden = pair_keys[slot] == key

        if attractive[e]:
            if forbidden:
                continue
            # Union by chain length, so the shorter partner list is the one walked.
            if chain_len[ru] >= chain_len[rv]:
                big = ru
                small = rv
            else:
                big = rv
                small = ru
            parent[small] = big

            # Re-key every constraint of `small` onto `big`.
            node = head[small]
            while node != -1:
                other = _find(parent, pool_val[node])
                nxt = pool_next[node]
                if other != big:
                    old = _pack(small, pool_val[node], n_nodes)
                    old_slot = _pair_find(pair_keys, old, mask)
                    if pair_keys[old_slot] == old:
                        # Tombstone-free deletion is not possible in open addressing without
                        # rehashing, so the stale key is left in place. It can only ever be looked
                        # up for a root that no longer exists, since `small` is no longer a root.
                        pass
                    new_key = _pack(big, other, n_nodes)
                    new_slot = _pair_find(pair_keys, new_key, mask)
                    if pair_keys[new_slot] != new_key:
                        if pairs_used + 1 > pair_capacity // 2:
                            raise RuntimeError("mws: pair table too small")
                        pair_keys[new_slot] = new_key
                        pairs_used += 1
                        if pool_used + 2 > pool_capacity:
                            raise RuntimeError("mws: partner pool too small")
                        pool_val[pool_used] = other
                        pool_next[pool_used] = head[big]
                        head[big] = pool_used
                        chain_len[big] += 1
                        pool_used += 1
                        pool_val[pool_used] = big
                        pool_next[pool_used] = head[other]
                        head[other] = pool_used
                        chain_len[other] += 1
                        pool_used += 1
                node = nxt
            head[small] = -1
            chain_len[small] = 0
        else:
            if not forbidden:
                if pairs_used + 1 > pair_capacity // 2:
                    raise RuntimeError("mws: pair table too small")
                pair_keys[slot] = key
                pairs_used += 1
                if pool_used + 2 > pool_capacity:
                    raise RuntimeError("mws: partner pool too small")
                pool_val[pool_used] = rv
                pool_next[pool_used] = head[ru]
                head[ru] = pool_used
                chain_len[ru] += 1
                pool_used += 1
                pool_val[pool_used] = ru
                pool_next[pool_used] = head[rv]
                head[rv] = pool_used
                chain_len[rv] += 1
                pool_used += 1

    for i in range(n_nodes):
        parent[i] = _find(parent, i)
    return parent, pairs_used, pool_used


def _numba_version() -> str:
    return numba.__version__
