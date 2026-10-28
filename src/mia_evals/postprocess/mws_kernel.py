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
changes the answer because global edge order is what the algorithm is defined on), which
`mws_stream.py` provides on top of this same loop. `mws.segment()` is the one entry point the scorer
uses: it runs this kernel with every edge in memory, or streams the edges through disk when there
are too many to hold. The Python reference is never used for scoring.

**Capacity grows instead of being guessed.** Measured pair insertions run from 0.43 per voxel on
the hemibrain crop to 3.5 on the zebrafish cubes, an 8x spread that no size chosen up front fits:
sized for zebrafish, a hemibrain block reserves ten times what it uses; sized lower, zebrafish
raised "pair table too small" hours into a run. So `process_edges` stops *before* the first edge it
lacks headroom for -- having changed nothing for that edge except path compression, which never
changes the partition -- and `run_edges` grows the state (`grow_state`) and resumes at that very
edge. Every decision is taken in the same order and against the same state as in an uninterrupted
run, so the partition is identical. Growing the pair table rehashes it and drops stale pairs, those
naming a cluster that has since been merged into another: lookups only ever ask about current
roots, and a cluster that stops being a root never becomes one again, so a stale pair can never be
found.

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
def _ordered(a: np.int64, b: np.int64) -> tuple[np.int64, np.int64]:
    """The pair with the smaller root first, so lookups are order-independent."""
    if a < b:
        return a, b
    return b, a


@njit(inline="always")
def _hash_pair(a: np.int64, b: np.int64, capacity: np.int64) -> np.int64:
    """Fibonacci-mixed hash of an ordered pair, reduced into [0, capacity).

    The pair is hashed rather than packed into a single key. `a * n_nodes + b` overflows int64 once
    n_nodes exceeds about 3 G -- at 7.08 G voxels the largest key would be ~5e19 against int64's
    9.2e18 -- so a packed key silently aliases distinct pairs at exactly the scale this is for.
    Mixing both components separately has no such ceiling.

    Reduced with `%` rather than `& (capacity - 1)` so the table need not be a power of two. That
    constraint was expensive at scale rather than cosmetic: the zebrafish doublecube needs 49.5 G
    slots, which a power of two rounds to 68.72 G, taking the pair table from 792 GB to 1,100 GB and
    the whole run from 1,754 GB to 2,062 GB against a 1.945 TB node. One division per lookup buys
    that back -- about 9 G divisions over the run, roughly 90 seconds against twelve hours.

    Callers should pass an odd capacity. Masking kept only the low bits, so it depended entirely
    on the hash being well mixed; modulo instead depends on the capacity sharing no factor with
    residual structure in the hash, and odd suffices given two odd multipliers and the xor-fold.
    """
    h = np.uint64(a) * np.uint64(11400714819323198485)
    h ^= np.uint64(b) * np.uint64(14029467366897019727)
    h ^= h >> np.uint64(29)
    return np.int64(h % np.uint64(capacity))


@njit(cache=True, nogil=True)
def _pair_find(
    key_a: np.ndarray, key_b: np.ndarray, a: np.int64, b: np.int64, capacity: np.int64
) -> np.int64:
    """Slot holding the ordered pair (a, b), or the first empty slot for inserting it.

    Both components are compared, so a hash collision costs a probe rather than a wrong answer. A
    64-bit hash alone would alias about three pairs in a 10^10-pair run, and each alias would be a
    spurious mutex silently blocking a legitimate merge.
    """
    slot = _hash_pair(a, b, capacity)
    while True:
        if key_a[slot] == EMPTY:
            return slot
        if key_a[slot] == a and key_b[slot] == b:
            return slot
        # Linear probe with an explicit wrap. Only the initial hash pays a division; the step is a
        # compare, as cheap as the mask it replaces.
        slot += 1
        if slot == capacity:
            slot = 0


@njit(cache=True, nogil=True)
def make_state(
    n_nodes: np.int64, pair_capacity: np.int64, pool_capacity: np.int64
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray]:
    """Allocate the union-find and mutex structures.

    Returned as a tuple of arrays rather than a class so the state can be threaded through repeated
    `process_edges` calls: whole-volume runs feed edges in priority-ordered batches and must carry
    the partition across them, since the algorithm is defined on one global ordering.

    `counters` holds [slots_used, pool_used, headroom_needed, pair_insertions]. `slots_used` is what
    the half-full ceiling is checked against and falls back to the live pair count whenever
    `grow_state` rehashes; `pair_insertions` only ever counts up, for reporting.
    """
    parent = np.arange(n_nodes, dtype=np.int64)
    key_a = np.full(pair_capacity, EMPTY, dtype=np.int64)
    key_b = np.full(pair_capacity, EMPTY, dtype=np.int64)
    head = np.full(n_nodes, -1, dtype=np.int64)
    chain_len = np.zeros(n_nodes, dtype=np.int64)
    pool_val = np.empty(pool_capacity, dtype=np.int64)
    pool_next = np.empty(pool_capacity, dtype=np.int64)
    counters = np.zeros(4, dtype=np.int64)
    return parent, key_a, key_b, head, chain_len, pool_val, pool_next, counters


@njit(cache=True, nogil=True)
def process_edges(
    u: np.ndarray,
    v: np.ndarray,
    attractive: np.ndarray,
    parent: np.ndarray,
    key_a: np.ndarray,
    key_b: np.ndarray,
    head: np.ndarray,
    chain_len: np.ndarray,
    pool_val: np.ndarray,
    pool_next: np.ndarray,
    counters: np.ndarray,
) -> np.int64:
    """Consume edges, already in descending priority order, updating the state; how many it took.

    Edges must arrive in globally descending priority across all calls. The algorithm's result is
    defined by that order, so a batch containing an edge weaker than one in a later batch changes
    the answer -- which is why the streaming driver buckets by priority rather than by position.

    Returns early, at the first edge whose worst case would not fit -- one pair for a new mutex,
    one per entry of the smaller cluster's partner chain for a merge -- with that headroom in
    `counters[2]` and nothing of that edge applied. `run_edges` grows the state and calls again
    from that edge. A return equal to `u.shape[0]` means the whole batch was consumed.
    """
    capacity = np.int64(key_a.shape[0])
    pair_room = capacity // 2
    pool_room = np.int64(pool_val.shape[0])
    for i in range(u.shape[0]):
        ru = _find(parent, u[i])
        rv = _find(parent, v[i])
        if ru == rv:
            continue

        lo, hi = _ordered(ru, rv)
        slot = _pair_find(key_a, key_b, lo, hi, capacity)
        forbidden = key_a[slot] != EMPTY

        if attractive[i]:
            if forbidden:
                continue
            if chain_len[ru] >= chain_len[rv]:
                big = ru
                small = rv
            else:
                big = rv
                small = ru
            need = chain_len[small]
            if counters[0] + need > pair_room or counters[1] + 2 * need > pool_room:
                counters[2] = need
                return np.int64(i)
            parent[small] = big

            node = head[small]
            while node != -1:
                other = _find(parent, pool_val[node])
                nxt = pool_next[node]
                if other != big:
                    nlo, nhi = _ordered(big, other)
                    nslot = _pair_find(key_a, key_b, nlo, nhi, capacity)
                    if key_a[nslot] == EMPTY:
                        if counters[0] + 1 > key_a.shape[0] // 2:
                            raise RuntimeError("mws: pair table too small")
                        key_a[nslot] = nlo
                        key_b[nslot] = nhi
                        counters[0] += 1
                        counters[3] += 1
                        if counters[1] + 2 > pool_val.shape[0]:
                            raise RuntimeError("mws: partner pool too small")
                        pool_val[counters[1]] = other
                        pool_next[counters[1]] = head[big]
                        head[big] = counters[1]
                        chain_len[big] += 1
                        counters[1] += 1
                        pool_val[counters[1]] = big
                        pool_next[counters[1]] = head[other]
                        head[other] = counters[1]
                        chain_len[other] += 1
                        counters[1] += 1
                node = nxt
            head[small] = -1
            chain_len[small] = 0
        else:
            if not forbidden:
                if counters[0] + 1 > pair_room or counters[1] + 2 > pool_room:
                    counters[2] = 1
                    return np.int64(i)
                key_a[slot] = lo
                key_b[slot] = hi
                counters[0] += 1
                counters[3] += 1
                if counters[1] + 2 > pool_val.shape[0]:
                    raise RuntimeError("mws: partner pool too small")
                pool_val[counters[1]] = rv
                pool_next[counters[1]] = head[ru]
                head[ru] = counters[1]
                chain_len[ru] += 1
                counters[1] += 1
                pool_val[counters[1]] = ru
                pool_next[counters[1]] = head[rv]
                head[rv] = counters[1]
                chain_len[rv] += 1
                counters[1] += 1
    counters[2] = 0
    return np.int64(u.shape[0])


@njit(cache=True, nogil=True)
def _count_live(key_a: np.ndarray, key_b: np.ndarray, parent: np.ndarray) -> np.int64:
    """Pairs whose two clusters are both still roots -- the only pairs a lookup can ever find."""
    live = np.int64(0)
    for s in range(key_a.shape[0]):
        a = key_a[s]
        if a != EMPTY and parent[a] == a and parent[key_b[s]] == key_b[s]:
            live += 1
    return live


@njit(cache=True, nogil=True)
def _rehash(
    key_a: np.ndarray, key_b: np.ndarray, parent: np.ndarray, capacity: np.int64
) -> tuple[np.ndarray, np.ndarray]:
    """The live pairs of a table, reinserted into a fresh one of `capacity` slots (odd)."""
    new_a = np.full(capacity, EMPTY, dtype=np.int64)
    new_b = np.full(capacity, EMPTY, dtype=np.int64)
    for s in range(key_a.shape[0]):
        a = key_a[s]
        if a == EMPTY:
            continue
        b = key_b[s]
        if parent[a] != a or parent[b] != b:
            continue                          # stale: one side has merged away
        slot = _pair_find(new_a, new_b, a, b, capacity)
        new_a[slot] = a
        new_b[slot] = b
    return new_a, new_b


def grow_state(state: tuple, pool_growth: float = 1.5) -> tuple:
    """Make room for the headroom `process_edges` asked for in `counters[2]`; the new state.

    The pair table is rehashed whenever it is short, which also drops stale pairs, and is only
    enlarged if the live pairs plus the request would not sit at a quarter full or less: after a
    rehash the table has room to double before the next one. The partner pool is copied into a
    larger one when it is short; its entries are chain links addressed by index, so they are
    copied, never rehashed.
    """
    parent, key_a, key_b, head, chain_len, pool_val, pool_next, counters = state
    need = int(counters[2])
    capacity = int(key_a.shape[0])
    if int(counters[0]) + need > capacity // 2:
        live = int(_count_live(key_a, key_b, parent))
        target = 4 * (live + need)
        new_capacity = (capacity if target <= capacity else max(2 * capacity, target)) | 1
        key_a, key_b = _rehash(key_a, key_b, parent, np.int64(new_capacity))
        counters[0] = live
    used = int(counters[1])
    if used + 2 * need > pool_val.shape[0]:
        size = max(int(pool_val.shape[0] * pool_growth), used + 2 * need)
        grown_val = np.empty(size, dtype=np.int64)
        grown_next = np.empty(size, dtype=np.int64)
        grown_val[:used] = pool_val[:used]
        grown_next[:used] = pool_next[:used]
        pool_val, pool_next = grown_val, grown_next
    counters[2] = 0
    return parent, key_a, key_b, head, chain_len, pool_val, pool_next, counters


def run_edges(
    state: tuple, u: np.ndarray, v: np.ndarray, attractive: np.ndarray
) -> tuple[tuple, int]:
    """Feed edges, in descending priority, through the kernel; the final state and its growths.

    The state is returned because growing it replaces arrays: callers must use the returned tuple.
    """
    total, done, growths = int(u.shape[0]), 0, 0
    while True:
        done += int(process_edges(u[done:], v[done:], attractive[done:], *state))
        if done >= total:
            return state, growths
        state = grow_state(state)
        growths += 1


def initial_capacities(n_nodes: int) -> tuple[int, int]:
    """Starting pair-table and pool sizes: room for one pair insertion per voxel.

    That covers the hemibrain crop (0.43 per voxel) with no growth at all, and reaches the zebrafish
    cubes' 3.5 in two or three growths. Odd, because the table reduces by modulo.
    """
    return max(1025, 2 * n_nodes) | 1, max(1024, 2 * n_nodes)


def compact_roots(roots: np.ndarray) -> np.ndarray:
    """Root ids -> labels 1..k, uint32 while k allows, in the order of the roots' ids.

    Numbered from 1, never 0: mutex watershed assigns every voxel to a cluster, so there is no
    background, and a metric reading `background_id = 0` would silently drop a real cluster.

    In slabs rather than `np.unique(..., return_inverse=True)` over the whole array, which on a
    7.08-gigavoxel volume is a 114 GB transient; two slab passes need one slab plus the id map.
    Compacting also keeps later consumers on the metrics' counting path, which needs the id span
    under 2**31 -- raw root ids reach the voxel count.
    """
    slab = max(1, roots.shape[0] // 64)
    bounds = range(0, roots.shape[0], slab)
    distinct = np.unique(np.concatenate([np.unique(roots[lo: lo + slab]) for lo in bounds]))
    narrow = np.uint32 if distinct.size < np.iinfo(np.uint32).max else np.int64
    labels = np.empty(roots.shape, dtype=narrow)
    for lo in bounds:
        labels[lo: lo + slab] = np.searchsorted(distinct, roots[lo: lo + slab]) + 1
    return labels


@njit(cache=True, nogil=True)
def finalize(parent: np.ndarray) -> np.ndarray:
    """Resolve every node to its root, so the caller sees a flat labelling."""
    for i in range(parent.shape[0]):
        parent[i] = _find(parent, i)
    return parent


def mutex_watershed_kernel(
    u: np.ndarray,
    v: np.ndarray,
    order: np.ndarray,
    attractive: np.ndarray,
    n_nodes: np.int64,
    pair_capacity: np.int64,
    pool_capacity: np.int64,
) -> tuple[np.ndarray, np.int64, np.int64]:
    """One-shot form: all edges at once, `order` giving descending priority.

    Returns (roots per node, pair insertions, pool entries used). The capacities are starting
    sizes: the state grows when they run short. What the tests compare against the Python
    reference; `mws.segment()` does the same with the edge arrays gathered in place, to hold one
    copy of them rather than two.

    The edge arrays are gathered into priority order rather than indirected through `order` inside
    the loop. Indirection costs three random reads per edge into arrays of 10^8 elements; the
    gather is one sequential pass and the loop then reads sequentially.
    """
    state = make_state(n_nodes, pair_capacity, pool_capacity)
    state, _ = run_edges(
        state,
        np.ascontiguousarray(u[order]),
        np.ascontiguousarray(v[order]),
        np.ascontiguousarray(attractive[order]),
    )
    counters = state[-1]
    return finalize(state[0]), counters[3], counters[1]


def _numba_version() -> str:
    return numba.__version__
