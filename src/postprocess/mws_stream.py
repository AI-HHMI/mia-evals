"""Mutex watershed over a volume too large to hold its own edge list.

A 7.08-gigavoxel volume has 41.6 G edges; at the 21 bytes/edge the in-memory form uses that is
870 GB, so the one-shot path cannot run however fast the loop gets. The edges are never needed
*simultaneously* though -- the algorithm only wants them in descending priority order -- so they can
be streamed.

Two passes:

  1. Walk the volume in blocks, derive each block's edges, append each to one of `n_buckets` files
     chosen by priority band. Only a block and its own edges are ever resident.
  2. Read buckets from the highest band down, sort each internally by exact priority, and feed them
     to `process_edges`, which carries the union-find and mutex state across calls.

**Bucketing by priority is what makes this exact.** Blockwise mutex watershed -- solving each block
independently and stitching -- returns a different partition, because the result is defined by one
global edge ordering and a block boundary breaks it. Buckets instead partition the priority range,
so sorting within a bucket and visiting buckets in order reproduces the global sort exactly.

**Ties must be broken canonically, not by arrival order.** Affinities are float16 on disk, so there
are at most 65,536 distinct values and they pile up near 0 and 1: exact ties are abundant, and the
algorithm's result depends on their order. A stable sort preserves arrival order, which differs
between one block (channel-major, then spatial) and many blocks (block-major, then channel, then
spatial) -- measured, that alone moved a 64^3 partition from 163 segments to 161 and 166 while the
edge multiset was identical at every block size (162,816 edges, none missing, none duplicated). So
each bucket is sorted by (-priority, offset, source), which depends on nothing but the edge itself.
That order also *is* the reference's: `build_edges` emits channel-major with source ascending inside
a channel, so (offset, source) reproduces its stable tie-breaking.

An edge is 13 bytes on disk: source voxel (int64), offset index (uint8), priority (float32). The
target voxel and whether the edge is attractive both follow from the offset index, so storing them
would store the same fact twice -- 542 GB rather than 870 GB at full volume.

**The priority is float32, not float16, and that is not a size oversight.** float16 was tried: on
a 64^3 block it changed 25.0% of the priorities and reordered the edge list. Mutex watershed is
defined by that ordering, so the streamed result would not have been the one-shot result. It would
have been a plausible segmentation that was not the algorithm's answer, and undetectable at
whole-volume scale where no reference exists. 84 GB is cheap for the exactness this design is for.

No halo is read. An affinity stored at voxel p for offset o describes the edge p -> p+o, so
emitting it needs the value at p only; the target's own value is irrelevant. Keying edges on their
source is also what makes each edge emitted exactly once, by the block that owns its source.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .mws import LONG_OFFSETS, SHORT_OFFSETS
from .mws_kernel import finalize, make_state, process_edges

OFFSETS = SHORT_OFFSETS + LONG_OFFSETS
N_SHORT = len(SHORT_OFFSETS)
DEFAULT_BUCKETS = 256

#: One record per edge. A structured dtype rather than three concatenated arrays per chunk: the
#: latter cannot be read back, since nothing in the file says where one chunk's arrays end.
EDGE = np.dtype([("source", "<i8"), ("offset", "u1"), ("priority", "<f4")])


def _strides(shape: tuple[int, ...]) -> np.ndarray:
    return np.array([int(np.prod(shape[i + 1:])) for i in range(len(shape))], dtype=np.int64)


def offset_steps(shape: tuple[int, ...]) -> np.ndarray:
    """Linear-index delta per offset, so the target voxel need not be stored."""
    strides = _strides(shape)
    return np.array([int(np.dot(strides, off)) for off in OFFSETS], dtype=np.int64)


def block_records(
    affinities: np.ndarray,
    origin: tuple[int, int, int],
    shape: tuple[int, int, int],
    repulsive_stride: int = 1,
) -> np.ndarray:
    """Edge records for every edge whose source voxel lies in this block.

    Priority follows the reference: the affinity itself for an attractive edge, 1 - affinity for a
    repulsive one, so descending priority means "most confident assertion first" for both kinds.
    """
    core = tuple(affinities.shape[1:])
    volume_strides = _strides(shape)
    origin_arr = np.asarray(origin, dtype=np.int64)
    chunks = []

    for index, offset in enumerate(OFFSETS):
        attractive = index < N_SHORT
        # Keep only sources whose target is still inside the volume.
        limits = [
            min(c, s - o - g)
            for c, s, o, g in zip(core, shape, offset, origin, strict=True)
        ]
        if any(limit <= 0 for limit in limits):
            continue
        window = tuple(slice(0, limit) for limit in limits)
        values = affinities[index][window]

        grids = np.meshgrid(
            *[np.arange(limit, dtype=np.int64) for limit in limits], indexing="ij"
        )
        if not attractive and repulsive_stride > 1:
            keep = tuple(slice(None, None, repulsive_stride) for _ in core)
            values = values[keep]
            grids = [g[keep] for g in grids]

        source = sum(
            (grids[axis].ravel() + origin_arr[axis]) * volume_strides[axis]
            for axis in range(len(core))
        )
        flat = values.ravel().astype(np.float32)
        priority = flat if attractive else (np.float32(1.0) - flat)

        record = np.empty(source.size, dtype=EDGE)
        record["source"] = source
        record["offset"] = index
        record["priority"] = priority
        chunks.append(record)

    if not chunks:
        return np.empty(0, dtype=EDGE)
    return np.concatenate(chunks)


def band_of(priority: np.ndarray, n_buckets: int) -> np.ndarray:
    """Bucket index, 0 = highest priority, so visiting buckets in order is descending priority."""
    scaled = np.clip(priority.astype(np.float32), 0.0, 1.0)
    return np.clip(((1.0 - scaled) * n_buckets).astype(np.int64), 0, n_buckets - 1)


def write_buckets(
    read_block: Callable[[tuple[int, int, int], tuple[int, int, int]], np.ndarray],
    shape: tuple[int, int, int],
    scratch: Path,
    block: int = 256,
    n_buckets: int = DEFAULT_BUCKETS,
    repulsive_stride: int = 1,
) -> list[int]:
    """Pass one. `read_block(origin, size)` gives that region's affinities; returns edge counts."""
    scratch.mkdir(parents=True, exist_ok=True)
    handles = [open(scratch / f"bucket_{i:04d}.bin", "wb") for i in range(n_buckets)]
    counts = [0] * n_buckets
    try:
        for z in range(0, shape[0], block):
            for y in range(0, shape[1], block):
                for x in range(0, shape[2], block):
                    origin = (z, y, x)
                    size = tuple(
                        min(block, s - o) for s, o in zip(shape, origin, strict=True)
                    )
                    records = block_records(
                        read_block(origin, size), origin, shape, repulsive_stride
                    )
                    if records.size == 0:
                        continue
                    bands = band_of(records["priority"], n_buckets)
                    order = np.argsort(bands, kind="stable")
                    records, bands = records[order], bands[order]
                    edges = np.searchsorted(bands, np.arange(n_buckets + 1))
                    for bucket in range(n_buckets):
                        lo, hi = int(edges[bucket]), int(edges[bucket + 1])
                        if hi > lo:
                            handles[bucket].write(records[lo:hi].tobytes())
                            counts[bucket] += hi - lo
    finally:
        for handle in handles:
            handle.close()
    return counts


def segment_streaming(
    read_block: Callable[[tuple[int, int, int], tuple[int, int, int]], np.ndarray],
    shape: tuple[int, int, int],
    scratch: Path,
    block: int = 256,
    n_buckets: int = DEFAULT_BUCKETS,
    repulsive_stride: int = 1,
    pair_capacity: int | None = None,
    pool_capacity: int | None = None,
    keep_buckets: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    """Exact mutex watershed over the whole volume, without holding the edge list.

    Labels are union-find root ids, not compacted to 1..k, and background is not special: every
    voxel belongs to some cluster. Compaction is skipped on purpose; see the comment at the return.

    Returns the labelling and a stats dict. The high-water marks are part of the return rather
    than printed and forgotten: capacity has to be sized per volume, and the only honest basis for
    that is what a comparable volume actually used. Sizing a ten-hour run from a safety multiplier
    instead costs real headroom -- at 8x the voxel count the zebrafish doublecube projects to
    1,869 GB against a 1.9 TB node, where its measured need is far lower.
    """
    n_nodes = int(np.prod(shape))
    counts = write_buckets(
        read_block, shape, scratch, block=block, n_buckets=n_buckets,
        repulsive_stride=repulsive_stride,
    )

    # Pair insertions per voxel, with nothing reclaimed: 2.45 measured at 256^3 but 4.05 at 64^3,
    # since a small volume is proportionally more surface and merges less. Sizing from the 256^3
    # figure put the ceiling at 4.00 and a 64^3 run raised "pair table too small" -- so size from
    # the small-volume rate with headroom. The table must also stay under half full, hence 16x.
    # Odd, and not rounded up to a power of two: the table reduces by modulo, so any size works.
    # Rounding cost 1.4x at the scale that matters -- the zebrafish doublecube needs 49.5 G slots,
    # which a power of two takes to 68.72 G, i.e. 792 GB of pair table becoming 1,100 GB.
    capacity = (pair_capacity or max(1025, 16 * n_nodes)) | 1
    state = make_state(
        np.int64(n_nodes), np.int64(capacity), np.int64(pool_capacity or max(1024, 16 * n_nodes))
    )
    parent, key_a, key_b, head, chain_len, pool_val, pool_next, counters = state
    steps = offset_steps(shape)

    for bucket in range(n_buckets):
        if counts[bucket] == 0:
            continue
        path = scratch / f"bucket_{bucket:04d}.bin"
        records = np.fromfile(path, dtype=EDGE)
        if records.size != counts[bucket]:
            raise ValueError(
                f"{path}: {records.size} records, expected {counts[bucket]}"
            )
        # Canonical total order: priority first, then offset, then source. A stable sort on
        # priority alone would inherit the order edges happened to be written in, which depends on
        # the block decomposition -- see the module docstring.
        order = np.lexsort(
            (records["source"], records["offset"], -records["priority"])
        )
        source = np.ascontiguousarray(records["source"][order])
        offset_index = records["offset"][order]
        process_edges(
            source,
            np.ascontiguousarray(source + steps[offset_index]),
            np.ascontiguousarray(offset_index < np.uint8(N_SHORT)),
            parent, key_a, key_b, head, chain_len, pool_val, pool_next, counters,
        )
        if not keep_buckets:
            os.unlink(path)

    roots = finalize(parent).reshape(shape)
    # Root ids are returned as labels rather than compacted to 1..k. `np.unique(...,
    # return_inverse=True)` would sort a copy of the whole array and build an inverse -- about
    # 114 GB of transient on top of everything else at 7.08 G voxels, which is what would have
    # killed the largest run at its final step after ten hours. The metrics factorise ids rather
    # than assuming a width (see cc_threshold), so compaction buys nothing they need.
    #
    # Narrowed only when it is provably safe: a 7.08-gigavoxel volume has root ids beyond uint32.
    if int(roots.max()) <= np.iinfo(np.uint32).max:
        roots = roots.astype(np.uint32)
    stats = {
        "edges": int(sum(counts)),
        "pair_insertions": int(counters[0]),
        "pool_used": int(counters[1]),
        "pair_capacity": int(capacity),
        "pool_capacity": int(pool_val.shape[0]),
        "n_nodes": n_nodes,
        "segments": int(np.unique(roots).size),
    }
    return roots, stats
