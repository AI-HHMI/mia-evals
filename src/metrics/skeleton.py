"""Skeleton-based instance metrics: nERL and VOI, as the NISB benchmark defines them.

The computation is upstream's (`utils.instance_metrics`, recycled from BANIS verbatim); this is the
adapter that makes it a registry entry. Two things it adds, both of which were previously spread
across the scoring scripts:

**Cropping the skeleton to the scored region.** `expected_run_length` walks a skeleton and asks
which segment each node landed in, so scoring a sub-region means dropping the nodes outside it.
That makes a sub-region score *pessimistic* rather than wrong -- a branch leaving the region ends
there, and the truncation counts as a split -- and the resulting numbers are comparable between
models scored over the same region but are **not** the benchmark's numbers. Recorded as
`whole_region` so a leaderboard can refuse to mix the two.

**Not silently dropping nERL.** `funlib.evaluate` returns `np.float32`, which is *not* a subclass
of `float` (`np.float64` is), so the obvious `isinstance(v, float)` filter keeps VOI and discards
nERL -- which looks exactly like the metric failing rather than like a type check misfiring.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseMetric
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


@MetricRegistry.register("skeleton_erl")
class SkeletonExpectedRunLength(BaseMetric):
    """nERL, VOI and merge/split counts against a traced skeleton.

    `truth` is a path to the skeleton pickle. Ranked on `nerl`, which is expected run length
    normalised by the maximum the skeleton allows, so it lands in [0, 1] and is comparable between
    models over one region.

    **nERL is not comparable across regions.** The same model measured 0.3045 over a whole NISB cube
    and 0.4192 on a 512^3 block of that same cube, because a shorter region truncates more branches.
    `region` is returned so a leaderboard can group by it rather than rank across it.
    """

    canonical = "instances"
    consumes = "labels"
    higher_is_better = True
    primary = "nerl"
    report_keys = ("voi_sum", "voi_split", "voi_merge", "n_non0_mergers", "n_splits")

    def __call__(self, prediction: np.ndarray, truth: Any, **context: Any) -> dict[str, float]:
        from utils.instance_metrics import compute_metrics

        origin = tuple(int(o) for o in context.get("origin", (0,) * prediction.ndim))
        shape = tuple(int(s) for s in prediction.shape)
        skeleton = truth if not isinstance(truth, (str, Path)) else load_skeleton(truth)
        full_nodes = skeleton.number_of_nodes()

        whole_region = bool(context.get("whole_region", False))
        if whole_region:
            # `compute_metrics` opens a path, and the benchmark's own path hands it the file
            # untouched -- so for a whole region nothing about the skeleton is allowed to differ.
            skeleton_path = str(truth)
            kept = full_nodes
        else:
            cropped = crop_skeleton(skeleton, origin, shape)
            kept = cropped.number_of_nodes()
            if kept == 0:
                raise ValueError(
                    f"no skeleton nodes inside the region at origin {origin} of shape {shape}; "
                    "the region and the skeleton do not overlap"
                )
            scratch = Path(context["scratch_dir"]) / "cropped_skeleton.pkl"
            scratch.parent.mkdir(parents=True, exist_ok=True)
            with open(scratch, "wb") as handle:
                pickle.dump(cropped, handle)
            skeleton_path = str(scratch)

        raw = compute_metrics(prediction, skeleton_path)
        scores = {k: float(v) for k, v in raw.items() if isinstance(v, NUMERIC)}
        scores["skeleton_nodes"] = float(kept)
        scores["skeleton_nodes_full"] = float(full_nodes)
        scores["whole_region"] = float(whole_region)
        return scores
