"""Three-channel affinities -> instances by LSD's decoding: watershed fragments, then agglomeration.

What Sheridan et al. (2023, local shape descriptors) and Funke et al. (2019, MALA) do with their
networks' nearest-neighbour affinities -- the decoder a model trained like theirs is tuned for, and
so a fairer route for one than `mws3`. Three steps, after the LSD code (`lsd.post.fragments`,
waterz) but written here on scipy and numba, which mia-evals already needs:

  1. **Fragments.** The three short-range channels are averaged, and voxels above `interior` (0.5)
     count as inside an object. Seeds are the local maxima of the distance to the nearest outside
     voxel over a `seed_distance`-wide window (10), and a seeded watershed floods the inverted
     distance (`elevation = "distance"`, LSD's choice) or one minus the mean affinity
     (`"affinity"`). It over-segments on purpose. Unlike LSD, every connected inside piece gets a
     seed (`_seed_every_piece`), so the flood never has to cross a membrane to reach one. As LSD
     did for its isotropic FIB-SEM volumes, fragments are made in 3D, with no epsilon pre-merge
     and no mean-affinity filter. As LSD's fragments are, the outside voxels belong to no fragment
     while the region graph is built, so
     two fragments touch only where their inside voxels do, and fragments separated by a predicted
     membrane are never compared. For the labelling, which this task scores voxel by voxel (where
     an unlabelled voxel is a hole), each outside voxel then takes the fragment of its nearest
     inside voxel.
  2. **Agglomeration.** Neighbouring fragments are joined in order of the affinities on the faces
     they share, highest first. A pair's score is a quantile of those affinities, read from a
     256-bin histogram as in waterz (`hist_quant_50`, `hist_quant_75`: the two merge functions LSD
     used), or their mean (`mean`). A merge pools the faces it joins, so every score is exact for
     the faces between two current segments. Merging runs down to the lowest threshold swept, and
     the order of the merges is kept.
  3. **Threshold.** The segmentation at threshold `t` is every merge made before the first one that
     scored below `t` -- where an agglomeration stopping at `t` would end -- so one agglomeration
     serves every threshold. Thresholds are in affinity units, so a higher `t` merges less;
     waterz's are one minus these.

Fitted on the fit block: the merge function, the threshold and the size filter, plus the fragment
settings (`interiors`, `elevations`) when more than one is given. Not bit-identical to waterz: the
quantile is read from the same 256 bins, but tie-breaking and the watershed's plateaus are this
code's own.

**It needs boundaries that are low in all three channels.** Inside is the mean of the three above
0.5, so a cut that shows in one channel only -- the affinities of un-eroded labels, where touching
neurons meet in a one-voxel plane and only the edge across it is 0 -- leaves that mean at 2/3:
the insides of two touching neurons join, and fragments run across them. LSD and MALA train on
labels eroded by GrowBoundary, whose two-voxel membranes zero every edge that touches them.
Measured on a 256^3 crop of the fit block (pq at min_size 500; mia-evals/
gary_comparison_neuron_instance/smoke/three_channel_mws/crop_check.py on /nrs): affinities of the
un-eroded truth 0.04 here against 0.87 for `mws3`; of the truth eroded by one voxel 0.56 against
0.61 (`mws3` with fill).

**The watershed is compiled here (`seeded_watershed`), not scipy's.** `ndimage.watershed_ift`
corrupts memory at this size: on the 896^3 fit block (scipy 1.18.1) it returned fragment ids above
the seed count in one run and segfaulted in the next, on the same input, with correct seeds.

A six-channel artifact is accepted as well and only its short-range half is read.

Branch `three-channel-mws` only (2026-10-01); see docs/scoring_three_channel_affinities.md. Adapted
from mia-train-experiments/gary_comparison/probes/watershed_agglomeration/ws_agglo.py (2026-09-23),
which ran the same steps, with mean scoring only, on one of our models' short-range channels -- but
built its region graph from every voxel's basin. Outside voxels all sit at the watershed's top
level, where a flood deals them out in tie order, so a stray one can land in the next object's
basin and give two fragments across a membrane a contact of high affinity; on synthetic boxes that
merged objects that never touch (tests/unit/test_ws_agglo.py).
"""

from __future__ import annotations

import heapq
import time
from collections import OrderedDict
from typing import Any

import numpy as np
from numba import njit
from scipy import ndimage

from .base import BasePostprocess
from .mws import SHORT_OFFSETS, MutexWatershed
from .mws3 import SizeFilterAndFill, check_short_range, describe_finish, sweep_settings
from .registry import PostprocessRegistry

#: Histogram resolution of the quantile merge functions: waterz's.
BINS = 256
ELEVATIONS = ("distance", "affinity")
#: Highest height the watershed floods: heights are 16-bit, one bucket per level.
LEVELS = 65535
#: LSD's fragment settings, left out of a row's description when used.
DEFAULT_INTERIOR, DEFAULT_ELEVATION = 0.5, "distance"


def parse_merge_function(name: str) -> float | None:
    """`"mean"` -> None; `"hist_quant_<q>"` -> q / 100. Anything else is refused."""
    if name == "mean":
        return None
    prefix = "hist_quant_"
    q = name[len(prefix):] if name.startswith(prefix) else ""
    if q.isdigit() and 0 < int(q) <= 100:
        return int(q) / 100
    raise ValueError(
        f'merge function must be "mean" or "hist_quant_<q>" with 0 < q <= 100, got {name!r}'
    )


@njit(cache=True, nogil=True)
def _flood(height: np.ndarray, labels: np.ndarray, shape: tuple[int, int, int],
           after: np.ndarray) -> None:
    """In place on flat C-order `labels`: every 0 takes a seed's label, cheapest path first.

    A bucket queue over the 65,536 heights, one FIFO chain per level threaded through `after`. A
    voxel is labelled when first reached and queued at the cost of its path, the highest height on
    it, so no level is ever revisited and each voxel is handled once.
    """
    sx, sy, sz = shape
    plane = sy * sz
    head = np.full(LEVELS + 1, -1, dtype=np.int64)
    tail = np.full(LEVELS + 1, -1, dtype=np.int64)
    for p in range(labels.size):
        if labels[p] != 0:
            level = np.int64(height[p])
            after[p] = -1
            if tail[level] == -1:
                head[level] = p
            else:
                after[tail[level]] = p
            tail[level] = p
    for current in range(LEVELS + 1):
        while head[current] != -1:
            p = head[current]
            head[current] = after[p]
            if head[current] == -1:
                tail[current] = -1
            x = p // plane
            y = (p - x * plane) // sz
            z = p - x * plane - y * sz
            for k in range(6):
                if k == 0:
                    if x == 0:
                        continue
                    q = p - plane
                elif k == 1:
                    if x == sx - 1:
                        continue
                    q = p + plane
                elif k == 2:
                    if y == 0:
                        continue
                    q = p - sz
                elif k == 3:
                    if y == sy - 1:
                        continue
                    q = p + sz
                elif k == 4:
                    if z == 0:
                        continue
                    q = p - 1
                else:
                    if z == sz - 1:
                        continue
                    q = p + 1
                if labels[q] == 0:
                    labels[q] = labels[p]
                    level = max(np.int64(height[q]), np.int64(current))
                    after[q] = -1
                    if tail[level] == -1:
                        head[level] = q
                    else:
                        after[tail[level]] = q
                    tail[level] = q


def seeded_watershed(height: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Seeded watershed over 16-bit heights, 6-connected -> int32 labels.

    Every voxel that is 0 in `markers` takes the label of the seed whose path to it has the lowest
    maximum height, ties first come, first served -- the image foresting transform `watershed_ift`
    computes, which this replaces (module docstring). Voxels no seed can reach stay 0.
    """
    if height.ndim != 3 or height.shape != markers.shape:
        raise ValueError(f"need two equal 3D shapes, got {height.shape} and {markers.shape}")
    labels = np.array(markers, dtype=np.int32, order="C")
    _flood(np.ascontiguousarray(height, dtype=np.uint16).reshape(-1), labels.reshape(-1),
           tuple(int(s) for s in labels.shape), np.empty(labels.size, dtype=np.int64))
    return labels


def fragments(
    mean_affinity: np.ndarray, interior: float, seed_distance: int, elevation: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Seeded watershed over the mean short-range affinity -> (fragments, inside, n).

    `fragments` holds int32 ids 1..n on the inside voxels (`mean_affinity > interior`) and 0 on the
    others, as LSD's `watershed_from_affinities` returns them; `inside` is that mask.
    """
    inside = mean_affinity > interior
    if not inside.any():
        return np.zeros(mean_affinity.shape, dtype=np.int32), inside, 0
    if inside.all():                     # no outside voxel to measure a distance from: one basin
        return np.ones(mean_affinity.shape, dtype=np.int32), inside, 1
    distance = ndimage.distance_transform_edt(inside).astype(np.float32)
    maxima = ndimage.maximum_filter(distance, size=seed_distance) == distance
    maxima &= inside
    markers, n = ndimage.label(maxima)
    del maxima
    n = _seed_every_piece(inside, distance, markers, int(n))
    if elevation == "distance":
        height = np.round((1.0 - distance / float(distance.max())) * LEVELS)
    else:
        height = np.round((1.0 - np.clip(mean_affinity, 0.0, 1.0)) * LEVELS)
    del distance
    labels = seeded_watershed(height.astype(np.uint16), markers)
    del markers, height
    labels[~inside] = 0
    return labels, inside, int(n)


def _seed_every_piece(inside: np.ndarray, distance: np.ndarray, markers: np.ndarray, n: int) -> int:
    """Give each connected piece of `inside` without a seed one at its deepest voxel; the new n.

    LSD keeps only the maxima over the window, which leaves a small piece beside a larger one --
    a thin process, or a part cut off by a narrow predicted membrane -- with no seed of its own.
    The flood then reaches it across the membrane from the neighbour's seed, and the two are one
    fragment before agglomeration can keep them apart: on affinities of eroded labels that merged
    nearly everything (VOI merge 3.7 on a 256^3 crop of the fit block). In place on `markers`.
    """
    pieces, n_pieces = ndimage.label(inside)
    seeded = np.zeros(n_pieces + 1, dtype=bool)
    seeded[0] = True                                     # 0 is outside, not a piece
    seeded[pieces[markers > 0]] = True
    if seeded.all():
        return n
    voxels = np.flatnonzero(~seeded[pieces])
    piece = pieces.reshape(-1)[voxels]
    del pieces
    order = np.lexsort((-distance.reshape(-1)[voxels], piece))   # by piece, deepest first
    first = order[np.r_[True, piece[order][1:] != piece[order][:-1]]]
    markers.reshape(-1)[voxels[first]] = np.arange(n + 1, n + 1 + first.size, dtype=markers.dtype)
    return n + int(first.size)


def fill_outside(labels: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Every outside voxel -> the fragment of its nearest inside voxel (Euclidean), so none is 0."""
    if inside.all() or not inside.any():
        return labels
    index = ndimage.distance_transform_edt(~inside, return_distances=False, return_indices=True)
    return labels[tuple(index)]


def region_graph(
    labels: np.ndarray, n: int, short: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(lo, hi, histogram, affinity sum) over every pair of fragments that share a face.

    The face between voxel p and p + e_axis carries channel `axis` at p, the edge `mws3` reads.
    Label 0 is no fragment and never part of a face. `histogram` is (pairs, BINS) of int32 face
    counts; the sum is for `mean`.
    """
    base = np.int64(n + 1)
    keys, codes, values = [], [], []
    for axis in range(len(SHORT_OFFSETS)):
        lo = tuple(slice(0, -1) if a == axis else slice(None) for a in range(labels.ndim))
        hi = tuple(slice(1, None) if a == axis else slice(None) for a in range(labels.ndim))
        a, b = labels[lo], labels[hi]
        face = (a != b) & (a > 0) & (b > 0)
        av, bv = a[face].astype(np.int64), b[face].astype(np.int64)
        keys.append(np.minimum(av, bv) * base + np.maximum(av, bv))
        del av, bv
        value = np.clip(short[axis][lo][face], 0.0, 1.0)
        # bin = floor(a * BINS), the top bin closed at 1.0.
        codes.append(np.minimum(value * BINS, BINS - 1).astype(np.uint8))
        values.append(value)
        del face
    key = np.concatenate(keys)
    del keys
    pair, inverse = np.unique(key, return_inverse=True)
    del key
    code = np.concatenate(codes).astype(np.int64)
    del codes
    histogram = np.bincount(inverse * BINS + code, minlength=pair.size * BINS)
    histogram = histogram.reshape(pair.size, BINS).astype(np.int32)
    total = np.bincount(inverse, weights=np.concatenate(values), minlength=pair.size)
    return pair // base, pair % base, histogram, total


def agglomerate(
    lo: np.ndarray,
    hi: np.ndarray,
    histogram: np.ndarray,
    total: np.ndarray,
    n_nodes: int,
    merge_function: str,
    t_min: float,
) -> np.ndarray:
    """Best-scoring pair first, down to `t_min` -> (kept, absorbed, score) per merge, in order.

    Exact: a merge pools the faces of its two segments with every common neighbour (histograms and
    sums add), so each score is the merge function over all faces between two current segments.
    `histogram` is modified in place; pass a copy to agglomerate one graph twice.
    """
    quantile = parse_merge_function(merge_function)
    counts = histogram.sum(axis=1)

    def score(record: list) -> float:
        if quantile is None:
            return record[1] / record[2]
        # waterz's pivot: the quantile's 1-based rank among the faces, at least 1, at most all.
        pivot = min(record[2], max(1, round(record[2] * quantile)))
        return float(np.searchsorted(np.cumsum(record[0]), pivot)) / (BINS - 1)

    # record = [histogram row, affinity sum, face count, version]; one list shared by both ends.
    adjacency: list[dict[int, list] | None] = [{} for _ in range(n_nodes)]
    heap = []
    for e, (u, v) in enumerate(zip(lo.tolist(), hi.tolist(), strict=True)):
        record = [histogram[e], float(total[e]), int(counts[e]), 0]
        adjacency[u][v] = record
        adjacency[v][u] = record
        heap.append((-score(record), u, v, 0))
    heapq.heapify(heap)

    merges = []
    while heap:
        negative, u, v, version = heapq.heappop(heap)
        nu, nv = adjacency[u], adjacency[v]
        if nu is None or nv is None:
            continue
        record = nu.get(v)
        if record is None or record[3] != version:
            continue                                     # stale: the pair changed since
        if -negative < t_min:
            break
        keep, gone = (u, v) if len(nu) >= len(nv) else (v, u)
        merges.append((keep, gone, -negative))
        kept, absorbed = adjacency[keep], adjacency[gone]
        assert kept is not None and absorbed is not None
        del kept[gone]
        for w, moved in absorbed.items():
            if w == keep:
                continue
            neighbours = adjacency[w]
            assert neighbours is not None
            del neighbours[gone]
            existing = kept.get(w)
            if existing is None:
                kept[w] = neighbours[keep] = target = moved
            else:
                if quantile is not None:
                    existing[0] += moved[0]
                existing[1] += moved[1]
                existing[2] += moved[2]
                target = existing
            target[3] += 1
            heapq.heappush(heap, (-score(target), keep, w, target[3]))
        adjacency[gone] = None
    return np.array(merges, dtype=np.float64).reshape(-1, 3)


def roots_at(merges: np.ndarray, n_nodes: int, threshold: float) -> np.ndarray:
    """Each node's segment after every merge made before the first that scored below `threshold`."""
    below = np.flatnonzero(merges[:, 2] < threshold)
    k = int(below[0]) if below.size else len(merges)
    parent = np.arange(n_nodes, dtype=np.int64)
    if k:
        parent[merges[:k, 1].astype(np.int64)] = merges[:k, 0].astype(np.int64)
    while True:                                          # pointer jumping to the roots
        jumped = parent[parent]
        if np.array_equal(jumped, parent):
            return parent
        parent = jumped


@PostprocessRegistry.register("ws_agglo")
class WatershedAgglomeration(BasePostprocess):
    """Short-range affinities -> instances by LSD's fragments + agglomeration; see the module.

    The sweep is fragments (outermost) x merge function x threshold x size filter (x fill, when
    given), so each fragmentation and each agglomeration is computed once per volume and every
    threshold and size is applied on top.
    """

    accepts = ("affinity",)
    produces = "instances"

    #: Fragmentations kept, with their region graphs and agglomerations: the fit block's and the
    #: test block's, ~6 GB each at 896^3.
    CACHE_ENTRIES = 2

    def __init__(
        self,
        interiors: tuple[float, ...] | list[float] = (DEFAULT_INTERIOR,),
        elevations: tuple[str, ...] | list[str] = (DEFAULT_ELEVATION,),
        seed_distance: int = 10,
        merge_functions: tuple[str, ...] | list[str] = ("hist_quant_50", "hist_quant_75"),
        thresholds: tuple[float, ...] | list[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
        min_sizes: tuple[int, ...] | list[int] = (0,),
        fill_distances: tuple[int | str, ...] | list[int | str] = (0,),
        **settings: Any,
    ) -> None:
        super().__init__(
            interiors=interiors, elevations=elevations, seed_distance=seed_distance,
            merge_functions=merge_functions, thresholds=thresholds, min_sizes=min_sizes,
            fill_distances=fill_distances, **settings,
        )
        if not interiors or any(not 0.0 < float(i) < 1.0 for i in interiors):
            raise ValueError(
                f"interiors must be affinities strictly inside (0, 1), got {interiors}"
            )
        self.interiors = tuple(sorted({float(i) for i in interiors}))
        if not elevations or set(elevations) - set(ELEVATIONS):
            raise ValueError(f"elevations must be drawn from {list(ELEVATIONS)}, got {elevations}")
        self.elevations = tuple(e for e in ELEVATIONS if e in elevations)
        if (isinstance(seed_distance, bool) or not isinstance(seed_distance, int)
                or seed_distance < 1):
            raise ValueError(f"seed_distance must be a positive voxel count, got {seed_distance!r}")
        self.seed_distance = seed_distance
        if not merge_functions:
            raise ValueError("ws_agglo with an empty `merge_functions` has nothing to sweep")
        for name in merge_functions:
            parse_merge_function(name)
        self.merge_functions = tuple(dict.fromkeys(merge_functions))
        if not thresholds or any(not 0.0 <= float(t) <= 1.0 for t in thresholds):
            raise ValueError(f"thresholds must be affinities in [0, 1], got {thresholds}")
        self.thresholds = tuple(sorted({float(t) for t in thresholds}))
        self.min_sizes, self.fill_distances = sweep_settings("ws_agglo", min_sizes, fill_distances)
        self._graphs: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
        #: The last thresholded labelling, before the size filter: one serves every size.
        self._cut: tuple[tuple, np.ndarray] | None = None
        self._finish = SizeFilterAndFill()
        self._last_run: dict[str, Any] | None = None

    def reads_channels(self) -> int:
        return len(SHORT_OFFSETS)

    def check_artifact(self, artifact: Any) -> None:
        check_short_range(artifact, "ws_agglo")

    def run_info(self) -> dict[str, Any] | None:
        return None if self._last_run is None else dict(self._last_run)

    def search_space(self) -> list[dict[str, Any]]:
        space = [
            {"interior": interior, "elevation": elevation, "merge_function": function,
             "threshold": threshold, "min_size": min_size}
            for interior in self.interiors
            for elevation in self.elevations
            for function in self.merge_functions
            for threshold in self.thresholds
            for min_size in self.min_sizes
        ]
        if self.fill_distances == (0,):
            return space
        return [{**point, "fill_distance": fill} for point in space for fill in self.fill_distances]

    def _graph(self, short: np.ndarray, key: tuple) -> dict[str, Any]:
        graph = self._graphs.get(key)
        if graph is not None:
            self._graphs.move_to_end(key)
            return graph
        _, interior, elevation = key
        start = time.perf_counter()
        labels, inside, n = fragments(short.mean(axis=0), interior, self.seed_distance, elevation)
        inside_fraction = float(inside.mean())
        made = time.perf_counter()
        lo, hi, histogram, total = region_graph(labels, n, short)
        labels = fill_outside(labels, inside)
        del inside
        done = time.perf_counter()
        print(f"  ws_agglo fragments: {n:,} (interior {interior}, {elevation} elevation; "
              f"{inside_fraction:.1%} of voxels inside) in {made - start:.0f} s; "
              f"{lo.size:,} neighbouring pairs in {done - made:.0f} s", flush=True)
        graph = {
            "labels": labels, "n": n, "lo": lo, "hi": hi, "histogram": histogram, "total": total,
            "merges": {},
            "info": {"implementation": "compiled seeded watershed + exact agglomeration",
                     "fragments": n, "inside_fraction": round(inside_fraction, 4),
                     "neighbouring_pairs": int(lo.size),
                     "seconds_fragments": round(made - start, 1),
                     "seconds_region_graph": round(done - made, 1)},
        }
        self._graphs[key] = graph
        while len(self._graphs) > self.CACHE_ENTRIES:
            self._graphs.popitem(last=False)
        return graph

    def _merges(self, graph: dict[str, Any], function: str) -> np.ndarray:
        merges = graph["merges"].get(function)
        if merges is None:
            start = time.perf_counter()
            merges = agglomerate(graph["lo"], graph["hi"], graph["histogram"].copy(),
                                 graph["total"], graph["n"] + 1, function, self.thresholds[0])
            seconds = time.perf_counter() - start
            print(f"  ws_agglo {function}: {len(merges):,} merges down to {self.thresholds[0]} "
                  f"in {seconds:.0f} s", flush=True)
            graph["merges"][function] = merges
            graph["info"][f"seconds_{function}"] = round(seconds, 1)
        return merges

    def __call__(self, array: np.ndarray, **params: Any) -> np.ndarray:
        short = np.asarray(array[: len(SHORT_OFFSETS)], dtype=np.float32)
        key = (MutexWatershed._fingerprint(short), float(params["interior"]),
               str(params["elevation"]))
        graph = self._graph(short, key)
        function, threshold = str(params["merge_function"]), float(params["threshold"])
        cut_key = (key, function, threshold)
        if self._cut is None or self._cut[0] != cut_key:
            self._cut = None                             # free the previous labelling first
            roots = roots_at(self._merges(graph, function), graph["n"] + 1, threshold)
            self._cut = (cut_key, roots.astype(np.uint32)[graph["labels"]])
        self._last_run = {**graph["info"], "merge_function": function, "threshold": threshold}
        return self._finish(self._cut[1], cut_key, int(params.get("min_size", 0)),
                            params.get("fill_distance", 0))

    def describe(self, params: dict[str, Any]) -> str:
        parts = [str(params["merge_function"]), f"threshold={params['threshold']}"]
        if float(params["interior"]) != DEFAULT_INTERIOR:
            parts.append(f"interior={params['interior']}")
        if params["elevation"] != DEFAULT_ELEVATION:
            parts.append(f"elevation={params['elevation']}")
        return f"ws_agglo({', '.join(parts + describe_finish(params))})"
