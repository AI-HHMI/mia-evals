"""Skeleton instance metrics: nERL and VOI, as the NISB benchmark and the LSD paper define them.

The computation is upstream's (`utils.instance_metrics`, recycled from BANIS verbatim, over
`funlib.evaluate` -- the same `expected_run_length` and `rand_voi` calls the LSD authors' own
evaluation makes); this is the adapter that makes it a registry entry. What it adds:

**Node positions are absolute, and the skeleton is always cut to the scored region.**
`expected_run_length` walks a skeleton and asks which segment each node landed in, so the nodes
are cropped to the region and shifted to its origin before anything is looked up. That makes a
sub-region score *pessimistic* rather than wrong -- a branch leaving the region ends there, and
the truncation counts as a split -- and the resulting numbers are comparable between models scored
over the same region but are **not** the benchmark's numbers. Recorded as `whole_region` so a
leaderboard can refuse to mix the two. This used to hand a whole-region skeleton to upstream
untouched, which is right only when the region starts at voxel 0, as a NISB cube does; the
zebrafinch regions do not.

**The same axes on both sides.** A skeleton declares the order its positions count in (`axes` on
the graph, e.g. "zyx") and an OME artifact declares its own; they are compared first. Transposed,
every lookup still lands on a real voxel and the score is plausible nonsense.

**The labelling is read only at the nodes.** Upstream reads it one node at a time. Handed an
`artifact.LazyLabelling` -- which the runner passes when the post-processor leaves a stored
labelling untouched -- the nodes are looked up in one chunk-grouped pass, and upstream receives
the values indexed exactly as the array would be. That is the only way a 478-gigavoxel labelling
is scoreable.

**Not silently dropping nERL.** `funlib.evaluate` returns `np.float32`, which is *not* a subclass
of `float` (`np.float64` is), so the obvious `isinstance(v, float)` filter keeps VOI and discards
nERL -- which looks exactly like the metric failing rather than like a type check misfiring.

**Units.** Node positions are in nanometres, so `erl` and `max_erl` are too; `erl_um` and
`max_erl_um` are the same numbers in micrometres, the unit the LSD paper reports.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseMetric, KeyInfo
from .registry import MetricRegistry

NUMERIC = (int, float, np.integer, np.floating)


def load_skeleton(path: str | Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def crop_skeleton(skeleton: Any, origin: tuple[int, ...], shape: tuple[int, ...]) -> Any:
    """Keep only nodes inside the region, re-indexed to region-local coordinates.

    Edges survive only if both endpoints do; the rest disappear with their nodes.
    """
    import networkx as nx

    low = np.asarray(origin)
    high = low + np.asarray(shape)
    keep = [
        node for node in skeleton.nodes
        if np.all(np.asarray(skeleton.nodes[node]["index_position"]) >= low)
        and np.all(np.asarray(skeleton.nodes[node]["index_position"]) < high)
    ]
    cropped = skeleton.subgraph(keep).copy()
    for node in cropped.nodes:
        cropped.nodes[node]["index_position"] = (
            np.asarray(cropped.nodes[node]["index_position"]) - low
        )
    assert isinstance(cropped, nx.Graph)
    return cropped


class NodeLabels:
    """A labelling known only at a skeleton's nodes, indexable there exactly as the array would be.

    Upstream's `compute_metrics` reads `pred_seg[x, y, z]` once per node. Handing it this instead
    of a zarr array keeps that verbatim code untouched and replaces hundreds of thousands of
    single-voxel reads with the one chunk-grouped pass that produced `values`.
    """

    def __init__(self, positions: np.ndarray, values: np.ndarray) -> None:
        self._values = {
            tuple(position): value
            for position, value in zip(positions.tolist(), values.tolist(), strict=True)
        }

    def __getitem__(self, key: Any) -> Any:
        return self._values[tuple(int(k) for k in key)]


@MetricRegistry.register("skeleton_erl")
class SkeletonExpectedRunLength(BaseMetric):
    """nERL, ERL, VOI and merge/split counts against a traced skeleton.

    `truth` is the path to a skeleton pickle: a networkx graph whose nodes carry `id` (the unit run
    length is computed over -- a neuron, or for the LSD regions a connected piece of one),
    `index_position` (absolute voxel, in `axes` order) and `nm_position`. Ranked on `nerl`, expected
    run length normalised by what a perfect segmentation scores on the same nodes, so it lands in
    [0, 1] and is comparable between models over one region.

    **nERL is not comparable across regions.** The same model measured 0.3045 over a whole NISB cube
    and 0.4192 on a 512^3 block of that same cube, because a shorter region truncates more branches.
    `region` is returned so a leaderboard can group by it rather than rank across it.
    """

    canonical = "instances"
    consumes = "labels"
    higher_is_better = True
    primary = "nerl"
    report_keys = ("erl_um", "voi_sum", "voi_split", "voi_merge", "n_non0_mergers", "n_splits")
    key_info = {
        "nerl": KeyInfo(True, "Normalised expected run length: ERL along the traced skeletons "
                              "divided by the maximum possible."),
        "erl_um": KeyInfo(True, "Expected run length in micrometres."),
        "voi_sum": KeyInfo(False, "voi_split + voi_merge."),
        "voi_split": KeyInfo(False, "Variation of information on the skeleton nodes, split term "
                                    "H(prediction | truth): over-segmentation."),
        "voi_merge": KeyInfo(False, "Variation of information on the skeleton nodes, merge term "
                                    "H(truth | prediction): under-segmentation."),
        "n_non0_mergers": KeyInfo(False, "Number of (merging segment, skeleton) pairs where one "
                                         "segment spans several skeletons."),
        "n_splits": KeyInfo(False, "Number of skeleton edges the segmentation cuts."),
    }
    point_lookups = True

    def __call__(self, prediction: Any, truth: Any, **context: Any) -> dict[str, float]:
        from utils.instance_metrics import compute_metrics

        shape = tuple(int(s) for s in prediction.shape)
        origin = tuple(int(o) for o in context.get("origin", (0,) * len(shape)))
        skeleton = truth if not isinstance(truth, (str, Path)) else load_skeleton(truth)
        full_nodes = skeleton.number_of_nodes()
        declared, axes = skeleton.graph.get("axes"), context.get("axes")
        if declared and axes and declared != axes:
            raise ValueError(
                f"the skeleton's positions count in {declared!r} order but the artifact's axes "
                f"are {axes!r}. Every lookup would land on a real voxel of the wrong neurite; "
                "predict in the skeleton's axis order (the data config's output_axes)."
            )

        cropped = crop_skeleton(skeleton, origin, shape)
        kept = cropped.number_of_nodes()
        if kept == 0:
            raise ValueError(
                f"no skeleton nodes inside the region at origin {origin} of shape {shape}; "
                "the region and the skeleton do not overlap"
            )
        labelling: Any = prediction
        if hasattr(prediction, "lookup"):
            positions = np.array(
                [cropped.nodes[n]["index_position"] for n in cropped.nodes], dtype=np.int64
            ).reshape(-1, len(shape))
            labelling = NodeLabels(positions, prediction.lookup(positions))

        # `compute_metrics` opens a path, so the cropped skeleton is written where it can.
        scratch = Path(context["scratch_dir"]) / "cropped_skeleton.pkl"
        scratch.parent.mkdir(parents=True, exist_ok=True)
        with open(scratch, "wb") as handle:
            pickle.dump(cropped, handle)

        raw = compute_metrics(labelling, str(scratch))
        scores = {k: float(v) for k, v in raw.items() if isinstance(v, NUMERIC)}
        scores["erl_um"] = scores["erl"] / 1000.0
        scores["max_erl_um"] = scores["max_erl"] / 1000.0
        scores["skeleton_nodes"] = float(kept)
        scores["skeleton_nodes_full"] = float(full_nodes)
        scores["whole_region"] = float(bool(context.get("whole_region", False)))
        return scores
