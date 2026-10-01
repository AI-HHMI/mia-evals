"""Mutex watershed over a region too large for one process: blocks, then one stitch across faces.

    python -m postprocess.mws_blockwise configs/<task>/<route>.toml --test <dir> --scratch <dir> \\
        [--worker K --workers N] [--processes P]

Exact mutex watershed is one global pass over every edge, and its union-find holds every voxel --
about 90 bytes a voxel before a single mutex is stored, 43 TB over LSD's zebrafinch benchmark region
(478 gigavoxels) -- which no node holds, whether or not the edges are streamed (`mws_stream`). So
the region is cut into blocks, and the algorithm changes in exactly one respect: **every edge inside
a block is decided before any edge that crosses a block face.** Otherwise it is the same algorithm,
run by the same compiled kernel (`mws_kernel`), over the same edges, in the same canonical order
(descending priority, then offset, then source voxel).

  1. *Segment.* Each block on its own: the kernel over the edges with both ends in the block, as
     `mws.segment` runs it over a whole array. The block's clusters are written as local labels;
     kept beside them are the mutexes between clusters that touch a face, and the face planes --
     labels and affinities -- that the crossing edges are built from.
  2. *Stitch.* The kernel once more, over a graph whose nodes are those boundary clusters: first
     the carried mutexes, then the crossing edges in canonical order. Between any two clusters
     only the first crossing edge in that order can act -- every later one finds the two merged,
     or forbidden, or already holding a mutex, and changes nothing -- so each face contributes one
     edge per pair of clusters instead of one per voxel, and the reduction is exact.
  3. *Relabel.* Each block's local labels, through the stitched lookup table, into the artifact.

So the result is exactly mutex watershed under that one change of order: the tests require it to
equal the Python reference run on the modified order, partition for partition, and to equal
`mws.segment` label for label when one block covers the region. How far the change of order moves
the partition from exact mutex watershed is a measurement (docs/neurite_tracing.md), not a claim.

**Masking happens inside the watershed.** With a mask, an edge with either end outside it is
dropped and those voxels are labelled 0 (background), so a cluster can neither form nor be split
through masked tissue. Clusters grow only along 1-voxel attractive edges, inside the region, so
each is connected within region and mask: the result already is the masked, connected-component
labelling LSD's protocol scores, with nothing joined through the margin outside the region.

**Parallel and resumable.** Stage 1 is the bulk of the work and has no dependencies: blocks are
dealt round-robin to `--workers` processes (an LSF job array), and a finished block leaves a
marker that a rerun skips. Stages 2 and 3 belong to a single finisher -- `--workers 1`, or the
scorer itself, which runs whatever is left -- with a pool of `--processes`. The labelling appears
under its final name only after the last block is relabelled: work happens in `<name>.partial`,
renamed into place at the end.

Memory: a block peaks at about `PEAK_BYTES_PER_VOXEL` bytes a voxel in stage 1, 20 GB for
256 x 512 x 512, and each pool process holds one block.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import socket
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.errors import ContainsArrayError

from .base import BasePostprocess
from .mws import LONG, LONG_OFFSETS, SHORT_OFFSETS, build_edges, check_offsets
from .mws_kernel import EMPTY, finalize, initial_capacities, make_state, run_edges
from .registry import PostprocessRegistry

OFFSETS = SHORT_OFFSETS + LONG_OFFSETS
N_SHORT = len(SHORT_OFFSETS)
#: Part of every plan's fingerprint: bump it when a change could move a partition, so a half-done
#: run of the old code cannot be finished by the new one.
ALGORITHM = "mws_blockwise v1"
#: Stage-1 peak per block voxel, edges in memory (`mws.MAX_IN_MEMORY_EDGES`' 37 B/edge, 6 edges a
#: voxel) plus the kernel's state.
PEAK_BYTES_PER_VOXEL = 300
CHUNK = 256                                  # the labelling's chunk edge, as `write_scored` uses

Box = tuple[tuple[int, ...], tuple[int, ...]]


# ------------------------------------------------------------------------------- where data is


@dataclass(frozen=True)
class Source:
    """A region of an affinity artifact, and its mask: the picklable thing pool processes reopen.

    `origin` and `shape` are the region in the artifact's own absolute coordinates (those of
    `Artifact.read`); every block is addressed relative to `origin`. `mask` is an OME array (a
    single-level group, or the path of one level of a pyramid) mapped onto the artifact's lattice
    through both sides' OME scale and translation.
    """

    affinities: str
    origin: tuple[int, ...]
    shape: tuple[int, ...]
    mask: str | None = None
    #: Where the artifact's first voxel sits when not where its own `origin` says (`BaseTask.place`
    #: moved it): pool processes reopen the artifact by path and must place it the same way.
    placed_at: tuple[int, ...] | None = None

    def _artifact(self) -> Any:
        from dataclasses import replace

        from artifact import open_artifact

        artifact = open_artifact(self.affinities)
        if self.placed_at is None:
            return artifact
        return replace(artifact, origin=tuple(int(o) for o in self.placed_at))

    def read(self, low: Sequence[int], size: Sequence[int]) -> np.ndarray:
        return self._artifact().read(
            tuple(o + lo for o, lo in zip(self.origin, low, strict=True)),
            tuple(int(s) for s in size))

    def read_mask(self, low: Sequence[int], size: Sequence[int]) -> np.ndarray | None:
        if self.mask is None:
            return None
        absolute = [o + lo for o, lo in zip(self.origin, low, strict=True)]
        return lattice_mask(self._artifact(), Path(self.mask), absolute, size)

    def describe(self) -> dict[str, Any]:
        artifact = self._artifact()
        attrs = json.dumps(artifact.attrs, sort_keys=True, default=str).encode()
        return {"affinities": str(Path(self.affinities).resolve()),
                "attrs_sha256": hashlib.sha256(attrs).hexdigest(),
                "shape": list(artifact.shape), "region": [list(self.origin), list(self.shape)],
                "mask": None if self.mask is None else str(Path(self.mask).resolve())}


@dataclass(frozen=True)
class ArraySource:
    """The same interface over arrays already in memory, for one process (`__call__`, tests)."""

    affinities: np.ndarray
    mask_array: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.affinities.shape[1:])

    def _window(self, low: Sequence[int], size: Sequence[int]) -> tuple[slice, ...]:
        return tuple(slice(lo, lo + s) for lo, s in zip(low, size, strict=True))

    def read(self, low: Sequence[int], size: Sequence[int]) -> np.ndarray:
        return self.affinities[(slice(None), *self._window(low, size))]

    def read_mask(self, low: Sequence[int], size: Sequence[int]) -> np.ndarray | None:
        if self.mask_array is None:
            return None
        return np.asarray(self.mask_array[self._window(low, size)], dtype=bool)

    def describe(self) -> dict[str, Any]:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.ascontiguousarray(self.affinities).tobytes())
        if self.mask_array is not None:
            digest.update(np.ascontiguousarray(self.mask_array).tobytes())
        return {"array": f"{self.affinities.shape}:{self.affinities.dtype}:{digest.hexdigest()}"}


def _ome_level(path: Path) -> tuple[Any, list[str], list[float], list[float]]:
    """(array, spatial axis names, scale, translation) of an OME array's level, storage order.

    `path` is a single-level group, a pyramid group (its first, finest level is used), or the path
    of one level inside a pyramid (`.../labels/neuropil_mask/s0`). Both OME 0.4 (`multiscales` at
    the top of a zarr v2 group's attrs) and 0.5 (under `ome`) are read.
    """
    node = zarr.open(str(path), mode="r")
    level: str | None = None
    group_path = path
    if hasattr(node, "shape"):                         # one level of a pyramid, named directly
        group_path, level = path.parent, path.name
    group = zarr.open_group(str(group_path), mode="r")
    attrs = dict(group.attrs)
    ome = attrs.get("ome") if isinstance(attrs.get("ome"), dict) else attrs
    scales = ome.get("multiscales") if isinstance(ome, dict) else None
    if not isinstance(scales, list) or not scales:
        raise ValueError(f"{group_path} carries no OME multiscales, so it cannot be placed on a "
                         "lattice: a mask needs its own scale and translation")
    multiscale = scales[0]
    if multiscale.get("coordinateTransformations"):
        raise ValueError(f"{group_path}: multiscale-level coordinateTransformations are not "
                         "supported; put the transform on the dataset")
    datasets = multiscale["datasets"]
    dataset = datasets[0] if level is None else next(
        (d for d in datasets if d.get("path") == level), None)
    if dataset is None:
        raise ValueError(f"{path}: no dataset {level!r} in its group's multiscales")
    axes = multiscale.get("axes") or []
    spatial = [i for i, a in enumerate(axes) if a.get("type") != "channel"]
    if len(spatial) != len(axes):
        raise ValueError(f"{group_path}: a mask cannot have a channel axis")
    scale = translation = None
    for transform in dataset.get("coordinateTransformations", []):
        if transform["type"] == "scale":
            scale = [float(v) for v in transform["scale"]]
        elif transform["type"] == "translation":
            translation = [float(v) for v in transform["translation"]]
    if scale is None:
        raise ValueError(f"{group_path}/{dataset['path']} has no scale transform")
    translation = translation or [0.0] * len(scale)
    return group[dataset["path"]], [str(a["name"]) for a in axes], scale, translation


def _artifact_geometry(artifact: Any) -> tuple[list[str], list[float], list[float]]:
    """(spatial axis names, scale, translation) of an artifact's lattice, storage order."""
    group = zarr.open_group(str(artifact.path), mode="r")
    attrs = dict(group.attrs)
    ome = attrs.get("ome") if isinstance(attrs.get("ome"), dict) else attrs
    (multiscale,) = ome["multiscales"]
    (dataset,) = multiscale["datasets"]
    axes = multiscale["axes"]
    spatial = [i for i, a in enumerate(axes) if a.get("type") != "channel"]
    scale = translation = None
    for transform in dataset.get("coordinateTransformations", []):
        if transform["type"] == "scale":
            scale = [float(transform["scale"][i]) for i in spatial]
        elif transform["type"] == "translation":
            translation = [float(transform["translation"][i]) for i in spatial]
    if scale is None:
        raise ValueError(f"{artifact.path} has no scale transform")
    return ([str(axes[i]["name"]) for i in spatial], scale,
            translation or [0.0] * len(scale))


def lattice_mask(artifact: Any, mask_path: Path, low: Sequence[int],
                 size: Sequence[int]) -> np.ndarray:
    """The mask on the artifact's lattice over [low, low + size) (absolute), as booleans.

    Each lattice voxel takes the mask voxel its centre falls in: centres from both sides' OME
    scale and translation (the centre of the first voxel, in this corpus' convention), so a mask
    at a coarser resolution, an offset origin or another axis order is placed correctly. A voxel
    whose centre falls outside the mask array is refused rather than assumed outside: a mask that
    does not cover the region is a wrong mask, and silently masking out the difference would
    delete real neurites from the score.
    """
    if artifact.array_path is None:
        raise ValueError(f"{artifact.path} is a bare array with no OME geometry to place a mask by")
    names, scale, translation = _artifact_geometry(artifact)
    array, mask_names, mask_scale, mask_translation = _ome_level(mask_path)
    if sorted(names) != sorted(mask_names):
        raise ValueError(f"{mask_path} has axes {mask_names}, the artifact {names}")
    indices: dict[str, np.ndarray] = {}
    for axis, name in enumerate(names):
        local = np.arange(int(size[axis]), dtype=np.int64) + int(low[axis]) - artifact.origin[axis]
        centre = translation[axis] + local * scale[axis]
        m = mask_names.index(name)
        j = np.floor((centre - mask_translation[m]) / mask_scale[m] + 0.5).astype(np.int64)
        if j.size and (j.min() < 0 or j.max() >= array.shape[m]):
            raise ValueError(
                f"the region's {name} voxels map to mask voxels [{j.min()}, {j.max()}], but "
                f"{mask_path} has {array.shape[m]} along {name}: the mask does not cover the region"
            )
        indices[name] = j
    window = tuple(slice(int(indices[n].min()), int(indices[n].max()) + 1) for n in mask_names)
    block = np.asarray(array[window]) != 0
    picked = block[np.ix_(*[indices[n] - indices[n].min() for n in mask_names])]
    return np.transpose(picked, [mask_names.index(n) for n in names])


# ------------------------------------------------------------------------------------ the plan


def block_boxes(shape: Sequence[int], block: Sequence[int], stride: int,
                chunk: int = CHUNK) -> list[Box]:
    """[low, high) boxes of `block` voxels tiling `shape`, C order; the last per axis cut to fit.

    Every block that does not end the region spans whole `chunk`s, so no two writers share a chunk;
    starts on a multiple of the repulsive stride, so the long-range sources it keeps are the ones
    `build_edges` keeps over the whole region; and is at least `LONG` deep, so a crossing edge
    always lands in the next block.
    """
    if len(block) != len(shape):
        raise ValueError(f"block {list(block)} has {len(block)} axes, the region {len(shape)}")
    for axis, (b, s) in enumerate(zip(block, shape, strict=True)):
        if b >= s:
            continue
        if b % chunk or b % stride or b < LONG:
            raise ValueError(
                f"block {list(block)}: axis {axis} is {b} voxels; a block shorter than its axis "
                f"must be a multiple of {chunk} (the labelling's chunk) and of the repulsive "
                f"stride {stride}, and at least {LONG}"
            )
    ranges = [[(lo, min(lo + b, s)) for lo in range(0, s, b)]
              for b, s in zip(block, shape, strict=True)]
    return [(tuple(r[0] for r in combo), tuple(r[1] for r in combo))
            for combo in itertools.product(*ranges)]


def _neighbours(boxes: list[Box], block: Sequence[int], shape: Sequence[int]) -> dict:
    """index -> {axis: index of the next block along it}, for the blocks that have one."""
    counts = [len(range(0, s, b)) for b, s in zip(block, shape, strict=True)]
    out: dict[int, dict[int, int]] = {}
    for index in range(len(boxes)):
        position = np.unravel_index(index, counts)
        nxt = {}
        for axis in range(len(shape)):
            if position[axis] + 1 < counts[axis]:
                step = list(position)
                step[axis] += 1
                nxt[axis] = int(np.ravel_multi_index(step, counts))
        out[index] = nxt
    return out


# ----------------------------------------------------------------------------- stage 1: blocks


def segment_block(source: Any, low: tuple[int, ...], high: tuple[int, ...],
                  region: tuple[int, ...], stride: int) -> dict[str, Any]:
    """One block's own mutex watershed -> its local labels and what the stitch needs from it.

    Local labels are 1..k in the order of the clusters' root ids, 0 outside the mask: for an
    unmasked block that is `mws.segment`'s own numbering.
    """
    size = tuple(h - lo for lo, h in zip(low, high, strict=True))
    raw = np.asarray(source.read(low, size))
    mask = source.read_mask(low, size)
    affinities = np.asarray(raw, dtype=np.float32)        # as `MutexWatershed.__call__` reads them
    u, v, priority, attractive = build_edges(affinities, stride)
    del affinities
    if mask is not None:
        inside = mask.reshape(-1)
        keep = inside[u] & inside[v]
        u, v, priority, attractive = u[keep], v[keep], priority[keep], attractive[keep]
        del keep
    edges = int(u.size)
    order = np.argsort(-priority, kind="stable")          # `mws._in_memory`'s order, exactly
    del priority
    u, v, attractive = u[order], v[order], attractive[order]
    del order
    voxels = int(np.prod(size))
    pairs, pool = initial_capacities(voxels)
    state = make_state(np.int64(voxels), np.int64(pairs), np.int64(pool))
    state, growths = run_edges(state, u, v, attractive)
    del u, v, attractive
    roots, key_a, key_b = finalize(state[0]), state[1], state[2]

    ids = roots if mask is None else np.where(mask.reshape(-1), roots, -1)
    distinct = np.unique(ids)
    masked = bool(distinct.size and distinct[0] < 0)
    local = (np.searchsorted(distinct, ids) + (0 if masked else 1)).astype(np.uint32)
    clusters = int(distinct.size - (1 if masked else 0))
    labels = local.reshape(size)

    # The live mutexes -- both ends still roots -- between clusters touching a face that has a
    # neighbour: the only clusters a crossing edge can reach.
    occupied = key_a != EMPTY
    a, b = key_a[occupied], key_b[occupied]
    live = (roots[a] == a) & (roots[b] == b)
    la, lb = local[a[live]], local[b[live]]
    boundary = np.zeros(clusters + 1, dtype=bool)
    data: dict[str, Any] = {}
    for axis in range(len(size)):
        depth = min(LONG, size[axis])
        if low[axis] > 0:
            plane = np.moveaxis(labels, axis, 0)[:depth]
            boundary[plane.ravel()] = True
            data[f"target_labels_{axis}"] = np.ascontiguousarray(plane)
        if high[axis] < region[axis]:
            plane = np.moveaxis(labels, axis, 0)[size[axis] - depth:]
            boundary[plane.ravel()] = True
            data[f"source_labels_{axis}"] = np.ascontiguousarray(plane)
            data[f"short_{axis}"] = np.ascontiguousarray(
                np.moveaxis(raw[axis], axis, 0)[size[axis] - 1:])
            data[f"long_{axis}"] = np.ascontiguousarray(
                np.moveaxis(raw[N_SHORT + axis], axis, 0)[size[axis] - depth:])
    boundary[0] = False
    carried = boundary[la] & boundary[lb]
    data.update(
        labels=labels, clusters=np.int64(clusters),
        mutex=np.stack([la[carried], lb[carried]], axis=1).astype(np.uint32),
        stats=np.array([edges, growths, int(live.sum())], dtype=np.int64),
    )
    return data


# ----------------------------------------------------------------------------- stage 2: stitch


def face_pairs(source_face: dict[str, np.ndarray], target_face: dict[str, np.ndarray], axis: int,
               low: tuple[int, ...], high: tuple[int, ...], region: tuple[int, ...],
               base_source: int, base_target: int, stride: int) -> tuple[np.ndarray, ...]:
    """The crossing edges of one face, one per pair of clusters: the first in canonical order.

    Returns (lo, hi, priority, offset, source) arrays, lo < hi being global cluster ids. `low` and
    `high` are the source block's box; the target block is the next one along `axis`.
    """
    labels = source_face["labels"]                # (depth, *across): the last planes along axis
    targets = target_face["labels"]               # (depth', *across): the next block's first ones
    depth = labels.shape[0]
    across = [a for a in range(len(region)) if a != axis]
    strides = np.array([int(np.prod(region[a + 1:])) for a in range(len(region))], dtype=np.int64)
    grids = np.meshgrid(*[np.arange(low[a], high[a], dtype=np.int64) for a in across],
                        indexing="ij")
    across_index = sum(g * strides[a] for g, a in zip(grids, across, strict=True))
    if stride > 1:
        across_keep = np.ones(across_index.shape, dtype=bool)
        for g in grids:
            across_keep &= g % stride == 0

    parts = []
    for channel, shift in ((axis, 1), (N_SHORT + axis, LONG)):
        values = source_face["short" if channel < N_SHORT else "long"]
        for j in range(values.shape[0]):
            plane = high[axis] - values.shape[0] + j     # source coordinate along `axis`
            target = plane + shift - high[axis]          # its target's plane in the next block
            if target >= targets.shape[0]:
                continue                                 # past the region's end
            lu = labels[depth - values.shape[0] + j]
            lv = targets[target]
            keep = (lu > 0) & (lv > 0)
            if channel >= N_SHORT and stride > 1:
                if plane % stride:
                    continue
                keep &= across_keep
            a = values[j].astype(np.float32)
            priority = a if channel < N_SHORT else np.float32(1.0) - a
            gu = lu[keep].astype(np.int64) + base_source
            gv = lv[keep].astype(np.int64) + base_target
            parts.append((np.minimum(gu, gv), np.maximum(gu, gv), priority[keep],
                          np.full(gu.size, channel, dtype=np.uint8),
                          across_index[keep] + plane * strides[axis]))
    empty = np.empty(0, dtype=np.int64)
    if not parts:
        return empty, empty, np.empty(0, np.float32), np.empty(0, np.uint8), empty
    lo, hi, priority, offset, source = (np.concatenate(p) for p in zip(*parts, strict=True))
    if lo.size == 0:                                     # every crossing edge masked out
        return lo, hi, priority, offset, source
    order = np.lexsort((source, offset, -priority, hi, lo))
    lo, hi = lo[order], hi[order]
    first = np.r_[True, (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])]
    return lo[first], hi[first], priority[order][first], offset[order][first], source[order][first]


def stitch(faces: list[tuple[np.ndarray, ...]], mutexes: list[np.ndarray],
           clusters: int) -> tuple[np.ndarray, dict[str, int]]:
    """Global cluster id -> final label: the kernel over the boundary clusters' graph.

    The carried mutexes go first -- they were decided inside the blocks, before any crossing edge
    -- then the crossing edges in canonical order. Final labels are 1..m in the order of each final
    cluster's smallest global id, so with nothing to stitch the table is the identity.
    """
    lo, hi, priority, offset, source = (np.concatenate(p) for p in zip(*faces, strict=True)) \
        if faces else (np.empty(0, np.int64),) * 5
    mutex = np.concatenate(mutexes) if mutexes else np.empty((0, 2), dtype=np.int64)
    nodes = np.unique(np.concatenate([lo, hi, mutex.ravel()]))
    table = np.arange(clusters + 1, dtype=np.int64)
    stats = {"boundary_clusters": int(nodes.size), "crossing_pairs": int(lo.size),
             "carried_mutexes": int(mutex.shape[0]), "growths": 0}
    if nodes.size:
        order = np.lexsort((source, offset, -priority))
        state = make_state(np.int64(nodes.size), *map(np.int64, initial_capacities(nodes.size)))
        state, grown = run_edges(
            state, np.searchsorted(nodes, mutex[:, 0]), np.searchsorted(nodes, mutex[:, 1]),
            np.zeros(mutex.shape[0], dtype=bool))
        stats["growths"] += grown
        state, grown = run_edges(
            state, np.searchsorted(nodes, lo[order]), np.searchsorted(nodes, hi[order]),
            offset[order] < N_SHORT)
        stats["growths"] += grown
        table[nodes] = nodes[finalize(state[0])]
    final, inverse = np.unique(table[1:], return_inverse=True)
    table[1:] = inverse + 1
    stats["segments"] = int(final.size)
    return table.astype(np.uint32 if final.size < 2**32 - 1 else np.uint64), stats


# ---------------------------------------------------------------------------------- the driver


def _write_npz(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.stem}.{socket.gethostname()}.{os.getpid()}.npz")
    np.savez(temporary, **data)
    os.replace(temporary, path)


def _write_json(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}")
    temporary.write_text(json.dumps(record, indent=1))
    os.replace(temporary, path)


def _face(data: Any, kind: str, axis: int) -> dict[str, np.ndarray]:
    if kind == "source":
        return {"labels": data[f"source_labels_{axis}"], "short": data[f"short_{axis}"],
                "long": data[f"long_{axis}"]}
    return {"labels": data[f"target_labels_{axis}"]}


class BlockwiseRun:
    """One region's blockwise watershed on disk: the plan, the markers, the three stages.

    Layout, for an output `<dir>/<name>.zarr`: the labelling is built in `<name>.zarr.partial`
    (its `fragments` array holds stage 1's local labels, its `s0` the result) and renamed into
    place when complete; `<name>.zarr.blocks` holds the plan, one `seg_<i>.npz` per segmented
    block, the stitched table and one marker per relabelled block, and is deleted at the end.
    """

    def __init__(self, source: Any, path: Path, block: Sequence[int], stride: int,
                 like: Any = None, attrs: dict[str, Any] | None = None,
                 chunk: int = CHUNK) -> None:
        self.source, self.path, self.stride = source, Path(path), int(stride)
        self.region = tuple(int(s) for s in source.shape)
        self.block = tuple(min(int(b), s) for b, s in zip(block, self.region, strict=True))
        self.chunk = int(chunk)
        self.boxes = block_boxes(self.region, self.block, self.stride, self.chunk)
        self.next = _neighbours(self.boxes, self.block, self.region)
        self.partial = self.path.with_name(self.path.name + ".partial")
        self.work = self.path.with_name(self.path.name + ".blocks")
        self.like, self.attrs = like, dict(attrs or {})
        self.plan = {"algorithm": ALGORITHM, "source": source.describe(),
                     "block": list(self.block), "repulsive_stride": self.stride,
                     "blocks": len(self.boxes)}
        self.stats: dict[str, Any] = {}

    def _start(self) -> None:
        self.work.mkdir(parents=True, exist_ok=True)
        plan_path = self.work / "plan.json"
        if not plan_path.exists():
            _write_json(plan_path, self.plan)
        found = json.loads(plan_path.read_text())
        if found != json.loads(json.dumps(self.plan)):
            raise SystemExit(
                f"{self.partial} is being built by a different watershed (its plan is {found}, "
                f"this one's {self.plan}). Rerun with the settings that made it, or delete "
                f"{self.partial} and {self.work} to start over."
            )
        group = zarr.open_group(str(self.partial), mode="a", zarr_format=3)
        try:
            group.create_array(name="fragments", shape=self.region, dtype=np.uint32,
                               chunks=self.block)
            stale = sorted(self.work.glob("seg_*.npz"))
            if stale:
                shutil.rmtree(self.partial)
                raise SystemExit(
                    f"{self.work} holds {len(stale)} segmented block(s) but {self.partial} had "
                    f"no fragments; delete {self.work} too, then rerun."
                )
        except ContainsArrayError:
            pass

    def segmented(self, index: int) -> Path:
        return self.work / f"seg_{index}.npz"

    def run_block(self, index: int) -> None:
        low, high = self.boxes[index]
        data = segment_block(self.source, low, high, self.region, self.stride)
        fragments = zarr.open_array(str(self.partial / "fragments"), mode="r+")
        fragments[tuple(slice(lo, h) for lo, h in zip(low, high, strict=True))] = data.pop("labels")
        _write_npz(self.segmented(index), data)

    def run(self, worker: int = 0, workers: int = 1, processes: int = 1) -> Path | None:
        """This worker's share of stage 1; with `workers == 1`, everything, then the result."""
        if not 0 <= worker < workers:
            raise SystemExit(f"--worker {worker} is not one of --workers {workers} (0-based)")
        if self.path.exists():
            return self.path
        self._start()
        todo = [i for i in range(worker, len(self.boxes), workers)
                if not self.segmented(i).exists()]
        print(f"mws_blockwise {self.path.name}: {len(self.boxes)} blocks of {list(self.block)} "
              f"over {list(self.region)}; worker {worker} of {workers} segments {len(todo)}",
              flush=True)
        started = time.perf_counter()
        _pool_map(_segment_one, [(self, i) for i in todo], processes)
        self.stats["segment_seconds"] = round(time.perf_counter() - started, 1)
        if workers > 1:
            return None
        return self._finish(processes)

    def _finish(self, processes: int) -> Path:
        counts = np.zeros(len(self.boxes), dtype=np.int64)
        totals = np.zeros(2, dtype=np.int64)                 # block edges, block growths
        mutexes = []
        table_path = self.work / "table.npy"
        stitched = table_path.exists()
        for i in range(len(self.boxes)):
            with np.load(self.segmented(i)) as record:       # one open file at a time
                counts[i] = int(record["clusters"])
                totals += record["stats"][:2]
                if not stitched:
                    mutexes.append(record["mutex"].astype(np.int64))
        bases = np.concatenate([[0], np.cumsum(counts)[:-1]])
        if not stitched:
            started = time.perf_counter()
            jobs = [(self, i, axis, int(bases[i]), int(bases[j]))
                    for i in range(len(self.boxes)) for axis, j in self.next[i].items()]
            faces = _pool_map(_face_one, jobs, processes)
            table, stats = stitch(faces, [m + b for m, b in zip(mutexes, bases, strict=True)],
                                  int(counts.sum()))
            del faces, mutexes
            np.save(table_path.with_name(".table.npy"), table)
            os.replace(table_path.with_name(".table.npy"), table_path)
            # The dtype is recorded so nothing needs to open the table but the relabelling: a file
            # still open when `_complete` deletes the work directory stays behind on NFS as a
            # hidden `.nfs*` entry, and the directory cannot be removed.
            stats.update(clusters=int(counts.sum()), block_edges=int(totals[0]),
                         block_growths=int(totals[1]), dtype=str(table.dtype),
                         stitch_seconds=round(time.perf_counter() - started, 1))
            _write_json(self.work / "stitch.json", stats)
        self.stats.update(json.loads((self.work / "stitch.json").read_text()))
        group = zarr.open_group(str(self.partial), mode="a", zarr_format=3)
        if "s0" not in group:
            try:
                group.create_array(name="s0", shape=self.region, dtype=self.stats["dtype"],
                                   chunks=tuple(min(self.chunk, s) for s in self.region))
            except ContainsArrayError:
                pass
        started = time.perf_counter()
        jobs = [(self, i, int(bases[i]), int(counts[i])) for i in range(len(self.boxes))
                if not (self.work / f"relabelled_{i}").exists()]
        _pool_map(_relabel_one, jobs, processes)
        self.stats["relabel_seconds"] = round(time.perf_counter() - started, 1)
        return self._complete(group)

    def _complete(self, group: Any) -> Path:
        from artifact import _shifted_ome

        provenance = {"kind": "instances", "background_id": 0, **self.attrs,
                      "mws_blockwise": {**self.plan, **{k: v for k, v in self.stats.items()}}}
        if self.like is not None:
            provenance.setdefault("origin", list(self.source.origin))
            provenance.setdefault("source_artifact", str(self.like.path))
            ome = _shifted_ome(self.like, tuple(self.source.origin))
            if ome is not None:
                group.attrs.update(ome=ome)
        group.attrs.update(**provenance)
        group["s0"].attrs.update(**provenance)
        shutil.rmtree(self.partial / "fragments")
        os.rename(self.partial, self.path)
        shutil.rmtree(self.work)
        print(f"mws_blockwise completed {self.path}: {self.stats.get('segments')} segments from "
              f"{self.stats.get('clusters')} block clusters", flush=True)
        return self.path

    def read(self) -> np.ndarray:
        return np.asarray(zarr.open_array(str(self.path / "s0"), mode="r")[:])


def _segment_one(job: tuple[BlockwiseRun, int]) -> None:
    run, index = job
    run.run_block(index)


def _face_one(job: tuple[BlockwiseRun, int, int, int, int]) -> tuple[np.ndarray, ...]:
    run, index, axis, base_source, base_target = job
    low, high = run.boxes[index]
    target = run.next[index][axis]
    with np.load(run.segmented(index)) as source, np.load(run.segmented(target)) as other:
        return face_pairs(
            _face(source, "source", axis), _face(other, "target", axis),
            axis, low, high, run.region, base_source, base_target, run.stride,
        )


def _relabel_one(job: tuple[BlockwiseRun, int, int, int]) -> None:
    run, index, base, count = job
    low, high = run.boxes[index]
    window = tuple(slice(lo, h) for lo, h in zip(low, high, strict=True))
    table = np.load(run.work / "table.npy", mmap_mode="r")
    lookup = np.concatenate([[0], table[base + 1: base + count + 1]]).astype(table.dtype)
    del table                                    # the mapping closed before the work dir goes
    local = np.asarray(zarr.open_array(str(run.partial / "fragments"), mode="r")[window])
    zarr.open_array(str(run.partial / "s0"), mode="r+")[window] = lookup[local]
    (run.work / f"relabelled_{index}").touch()


def _pool_map(function: Any, jobs: list[Any], processes: int) -> list[Any]:
    """`function` over `jobs`, in a process pool when `processes > 1`; results in job order."""
    if processes <= 1 or len(jobs) <= 1:
        return [function(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=min(processes, len(jobs))) as pool:
        return list(pool.map(function, jobs))


# --------------------------------------------------------------------------- the post-processor


@PostprocessRegistry.register("mws_blockwise")
class BlockwiseMutexWatershed(BasePostprocess):
    """Six-channel affinities -> instances by blockwise mutex watershed, for regions of any size.

    The route for regions whose exact watershed does not fit (`mws`, up to a few gigavoxels). No
    size filter: on a region this size it would need global segment sizes, and the neurite tracing
    tasks it is for score every segment. `mask` is a key in the volume's store (the affinity
    artifact's `source_path`), or an absolute path, of an OME array whose nonzero voxels are kept;
    see the module docstring for why it is applied inside the watershed.

    For a skeleton metric the scorer asks for the labelling lazily: it is built (or finished) in
    `<--scratch>/mws_blockwise/repulsive_stride<s>/<volume>.zarr` with a pool of `processes`, and
    only the chunks under the skeleton's nodes are read back. The same directory is where
    `python -m postprocess.mws_blockwise` workers build it ahead of the scorer.
    """

    accepts = ("affinity",)
    produces = "instances"

    def __init__(
        self,
        block: Sequence[int] = (256, 512, 512),
        repulsive_strides: Sequence[int] = (1,),
        mask: str | None = None,
        processes: int = 1,
        chunk: int = CHUNK,
    ) -> None:
        super().__init__(block=block, repulsive_strides=repulsive_strides, mask=mask,
                         processes=processes, chunk=chunk)
        self.block = tuple(int(b) for b in block)
        self.repulsive_strides = tuple(int(s) for s in repulsive_strides)
        if not self.repulsive_strides or min(self.repulsive_strides) < 1:
            raise ValueError(f"repulsive_strides must be >= 1, got {list(repulsive_strides)}")
        self.mask = mask
        self.processes = int(processes)
        self.chunk = int(chunk)
        self._scratch: Path | None = None
        self._last_run: dict[str, Any] | None = None

    def check_artifact(self, artifact: Any) -> None:
        check_offsets(artifact, OFFSETS)

    def use_scratch(self, directory: str | Path) -> None:
        self._scratch = Path(directory)

    def run_info(self) -> dict[str, Any] | None:
        return None if self._last_run is None else dict(self._last_run)

    def search_space(self) -> list[dict[str, Any]]:
        return [{"repulsive_stride": s} for s in self.repulsive_strides]

    def _mask_path(self, artifact: Any) -> str | None:
        if self.mask is None:
            return None
        if Path(self.mask).is_absolute():
            return self.mask
        store = artifact.attrs.get("source_path")
        if not store:
            raise ValueError(f"mask {self.mask!r} is relative to the volume's store, but "
                             f"{artifact.path} records no source_path")
        return str(Path(store) / self.mask)

    def blockwise(self, artifact: Any, origin: tuple[int, ...], shape: tuple[int, ...],
                  stride: int) -> BlockwiseRun:
        """The on-disk run for this artifact's region, as the scorer and the workers address it."""
        if self._scratch is None:
            raise ValueError("mws_blockwise needs a scratch directory (the scorer's --scratch)")
        name = artifact.path.name
        name = name if name.endswith(".zarr") else f"{name}.zarr"
        source = Source(str(artifact.path), tuple(origin), tuple(shape), self._mask_path(artifact),
                        placed_at=tuple(artifact.origin))
        path = self._scratch / "mws_blockwise" / f"repulsive_stride{stride}" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return BlockwiseRun(source, path, self.block, stride, like=artifact,
                            attrs={"convention": self.describe({"repulsive_stride": stride})},
                            chunk=self.chunk)

    def lazy(self, artifact: Any, origin: tuple[int, ...], shape: tuple[int, ...],
             **params: Any) -> Any:
        from artifact import LazyLabelling, open_artifact

        run = self.blockwise(artifact, origin, shape, int(params["repulsive_stride"]))
        path = run.run(processes=self.processes)
        assert path is not None                                  # one worker finishes the run
        # The group when it carries OME geometry (inherited from the affinities), else its array.
        has_ome = "ome" in dict(zarr.open_group(str(path), mode="r").attrs)
        labelling = open_artifact(path if has_ome else path / "s0")
        self._last_run = labelling.attrs.get("mws_blockwise")
        return LazyLabelling(labelling, tuple(origin), tuple(shape))

    def __call__(self, array: np.ndarray, **params: Any) -> np.ndarray:
        """In memory, one process: for a region small enough to hold, e.g. a voxel metric's."""
        import tempfile

        if array.shape[0] < 6:
            raise ValueError(f"mws_blockwise needs all six affinity channels, got {array.shape[0]}")
        if self.mask is not None:
            raise ValueError("mws_blockwise reads its mask from the artifact's store, which an "
                             "in-memory array does not have; score it lazily")
        if self._scratch is None:
            raise ValueError("mws_blockwise needs a scratch directory (the scorer's --scratch)")
        self._scratch.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="mws_blockwise_", dir=self._scratch))
        try:
            run = BlockwiseRun(ArraySource(np.asarray(array)), work / "labels.zarr", self.block,
                               int(params["repulsive_stride"]), chunk=self.chunk)
            run.run()
            self._last_run = dict(run.stats)
            return run.read()
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def describe(self, params: dict[str, Any]) -> str:
        return (f"mws_blockwise(repulsive_stride={params['repulsive_stride']}, "
                f"block={list(self.block)})")


# --------------------------------------------------------------------------------------- the CLI


def main() -> None:
    """Build a scoring config's blockwise labellings ahead of `mia-evals score`, in parallel."""
    import components  # noqa: F401  (populates the registries)
    from config import load_scoring_config
    from evaluate import build, resolve_artifacts

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("config", type=Path, help="the scoring config the labellings are for")
    parser.add_argument("--test", type=Path, required=True, help="as for mia-evals score")
    parser.add_argument("--val", type=Path, default=None, help="as for mia-evals score")
    parser.add_argument("--scratch", type=Path, required=True,
                        help="the --scratch the scorer will be given; the labellings go there")
    parser.add_argument("--worker", type=int, default=0, help="this worker's index, 0-based")
    parser.add_argument("--workers", type=int, default=1,
                        help="how many workers share stage 1; with 1, the run is finished too")
    parser.add_argument("--processes", type=int, default=None,
                        help="pool size; default the config's `processes`")
    args = parser.parse_args()

    config = load_scoring_config(args.config)
    task, processor, _ = build(config)
    if not isinstance(processor, BlockwiseMutexWatershed):
        raise SystemExit(f"{args.config} post-processes with {config.postprocess.name!r}, "
                         "not mws_blockwise")
    processor.use_scratch(args.scratch)
    splits = [(args.test, config.volumes)]
    if args.val is not None and config.fit_volumes is not None:
        splits.append((args.val, config.fit_volumes))
    for spec, volumes in splits:
        artifacts = resolve_artifacts(spec, volumes)
        for volume in volumes:
            artifact = task.place(artifacts[volume.name])        # as the scorer places it
            processor.check_artifact(artifact)
            origin, shape = task.region(volume, artifact)
            for params in processor.search_space():
                processor.blockwise(artifact, origin, shape, params["repulsive_stride"]).run(
                    args.worker, args.workers, args.processes or processor.processes)


if __name__ == "__main__":
    main()
