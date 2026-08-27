"""Instance metrics against dense voxel ground truth: VOI, adapted Rand, panoptic quality.

These exist because the skeleton metrics cannot be used on most of the corpus. NISB ships a traced
skeleton; the instance volumes in `lmd-v0.0.1` -- Kasthuri AC3/AC4, the zebrafish cubes, hemibrain,
the LICONN blocks -- ship dense voxel labels and no skeleton at all. Scoring them needs metrics
that compare two labellings directly.

All three come from one contingency table, computed once, because they are all functions of the
same joint distribution of (true id, predicted id). Computing them separately would mean three
passes over a volume whose labelling can be 49 GB.

**These are not the skeleton metrics under another name.** Skeleton VOI is computed over *nodes* --
one sample per traced point -- while this is over *voxels*, so a thick process contributes in
proportion to its volume rather than its length. Both are legitimate; they answer different
questions and their numbers must never be put in one column.

**Panoptic quality here, not mAP.** mAP needs a confidence per predicted instance, and thresholded
connected components produces none -- there is no score attached to a component. Reporting mAP over
unranked instances silently reduces it to a single-threshold F-measure, which is what PQ already is
and says so. A postprocessor that does emit per-instance confidence can add a ranking metric
beside this one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseMetric
from .registry import MetricRegistry

#: Largest joint code table counted densely, in cells. Above this the joint is sorted instead.
#: 128 M cells is 1 GB as int64. The bound exists because the dense table's size is the *product*
#: of the two code spaces: 11,690 true objects against a few million spurious components -- which is
#: what an over-fragmented prediction looks like -- would be 280 GB of mostly-zero counters.
DENSE_TABLE_LIMIT = 1 << 27


def _factorize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct values, and a small integer code per element. Linear time where it can be.

    `np.unique(..., return_inverse=True)` sorts, which is O(n log n) and the dominant cost at these
    sizes. A label array almost always holds few distinct ids packed into a modest numeric range, so
    the codes can be built with counting instead: one `bincount` over the shifted values, then a
    lookup table. Falls back to the sort when the range is too wide for a table -- an id space like
    MICrONS' 64-bit segment ids, where a dense table would be terabytes.

    Codes are int32: there cannot be more distinct ids than voxels, and a volume with 2**31 distinct
    ids is not a segmentation. Halving the width matters at 7 gigavoxels, where an int64 code array
    alone is 57 GB.
    """
    low = int(values.min())
    span = int(values.max()) - low + 1
    # A table is worth it while it stays comparable to the data itself; 2**22 is a floor so small
    # arrays with a wide-ish range still take the fast path.
    if 0 < span <= max(4 * values.size, 1 << 22) and span < (1 << 31):
        shifted = (values - low).astype(np.int64, copy=False)
        occupied = np.bincount(shifted, minlength=span)
        present = np.flatnonzero(occupied)
        table = np.zeros(span, dtype=np.int32)
        table[present] = np.arange(present.size, dtype=np.int32)
        return present.astype(np.int64) + low, table[shifted]
    distinct, codes = np.unique(values, return_inverse=True)
    return distinct, codes.astype(np.int32, copy=False)


def contingency(
    truth: np.ndarray, prediction: np.ndarray, ignore_id: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Joint voxel counts of (true id, predicted id), as sparse triples.

    Returns `(true_ids, pred_ids, counts, total)` where the three arrays are parallel: entry `i`
    says `counts[i]` voxels have true id `true_ids[i]` and predicted id `pred_ids[i]`.

    Sparse rather than a dense matrix because instance ids are unbounded -- a zebrafish cube holds
    11,690 objects and a fragmented prediction can hold far more, so a dense table would be
    hundreds of millions of mostly-zero cells.

    Built by factorising each side and counting the combined code, **not** by
    `np.unique(pairs, axis=0)`. That reads better and is unusable: measured at 1.3 s per million
    voxels, it costs ~3 minutes on a 134-megavoxel volume and ~2.5 hours on a 7-gigavoxel one, per
    call, with a 113 GB temporary -- against five threshold candidates on each of two splits.
    """
    truth = np.asarray(truth).ravel()
    prediction = np.asarray(prediction).ravel()
    if truth.shape != prediction.shape:
        raise ValueError(
            f"truth has {truth.size} voxels and the prediction {prediction.size}; they must cover "
            "the same region"
        )
    if ignore_id is not None:
        keep = truth != ignore_id
        truth, prediction = truth[keep], prediction[keep]
    if truth.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, 0

    true_ids, true_codes = _factorize(truth)
    pred_ids, pred_codes = _factorize(prediction)
    # One key per voxel. int64 because the product of the two code spaces exceeds 2**31 long before
    # either side does.
    key = true_codes.astype(np.int64) * pred_ids.size + pred_codes

    # Dense counting when the table fits (see DENSE_TABLE_LIMIT), sorting when it does not. Both
    # branches must agree; `tests/unit/test_voxel_instance.py` lowers the limit to force the second.
    table_size = true_ids.size * pred_ids.size
    if table_size <= DENSE_TABLE_LIMIT:
        counts = np.bincount(key, minlength=table_size)
        occupied = np.flatnonzero(counts)
        return (
            true_ids[occupied // pred_ids.size],
            pred_ids[occupied % pred_ids.size],
            counts[occupied],
            int(truth.size),
        )
    keys, counts = np.unique(key, return_counts=True)
    return (
        true_ids[keys // pred_ids.size],
        pred_ids[keys % pred_ids.size],
        counts,
        int(truth.size),
    )


def _group_sums(ids: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Total count per distinct id -- one marginal of the sparse table."""
    if ids.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    ids, counts = ids[order], counts[order]
    boundaries = np.flatnonzero(np.diff(ids)) + 1
    return np.add.reduceat(counts, np.concatenate([[0], boundaries]))


#: The sparse joint table: (true_ids, pred_ids, counts, total). Passed between metrics so a
#: caller computing several of them factorises the two labellings once rather than once each.
Table = tuple[np.ndarray, np.ndarray, np.ndarray, int]


def _marginal(ids: np.ndarray, counts: np.ndarray, total: int) -> np.ndarray:
    """Per-group probabilities of one marginal."""
    return _group_sums(ids, counts) / total


def variation_of_information(
    truth: np.ndarray, prediction: np.ndarray,
    ignore_id: int | None = None, background_id: int | None = 0,
) -> dict[str, float]:
    """VOI split and merge, in **bits**. Lower is better; 0 means the labellings agree exactly.

    Three conventions here are not free choices -- they are what `funlib.evaluate.rand_voi` does,
    and this repository reports skeleton VOI through that function. Two VOI columns on one
    leaderboard that disagreed about any of them would be a silent unit error:

      * **`voi_split` is over-segmentation.** It is `H(prediction | truth)`: one true object broken
        into several predicted pieces means knowing the truth still does not tell you the
        prediction. The other direction, `H(truth | prediction)`, is `voi_merge`. Getting these the
        wrong way round is invisible in the numbers and inverts every diagnosis, so it is pinned by
        `tests/unit/test_voxel_instance.py` against funlib's own output.
      * **Bits, not nats.** funlib uses log base 2.
      * **True background excluded**, hence `background_id = 0` as the default rather than None.
        Background is usually the single largest "object", and including it lets a model that
        predicts nothing but background score a respectable VOI.

    Verified equal to `funlib.evaluate.rand_voi` on constructed over- and under-segmentations.
    """
    return voi_from_table(contingency(truth, prediction, ignore_id), background_id)


def voi_from_table(table: Table, background_id: int | None = 0) -> dict[str, float]:
    """VOI from an already-computed contingency table. See `variation_of_information`."""
    ids_t, ids_p, counts, total = table
    if background_id is not None:
        keep = ids_t != background_id
        ids_t, ids_p, counts = ids_t[keep], ids_p[keep], counts[keep]
        total = int(counts.sum())
    if total == 0:
        return {"voi_split": 0.0, "voi_merge": 0.0, "voi_sum": 0.0}

    joint = counts / total
    p_true = _marginal(ids_t, counts, total)
    p_pred = _marginal(ids_p, counts, total)

    h_joint = -float((joint * np.log2(joint)).sum())
    h_true = -float((p_true * np.log2(p_true)).sum())
    h_pred = -float((p_pred * np.log2(p_pred)).sum())
    mutual = h_true + h_pred - h_joint
    # Clamped at zero: both differences are non-negative in exact arithmetic, and a -1e-16 from
    # floating point reads as a nonsensical negative VOI in a results table.
    split = max(0.0, h_pred - mutual)  # H(prediction | truth): over-segmentation
    merge = max(0.0, h_true - mutual)  # H(truth | prediction): under-segmentation
    return {"voi_split": split, "voi_merge": merge, "voi_sum": split + merge}


def adapted_rand_error(
    truth: np.ndarray, prediction: np.ndarray,
    ignore_id: int | None = None, background_id: int | None = 0,
) -> float:
    """1 - the F-score of the pairwise "same object" relation. Lower is better.

    The SNEMI3D definition: over pairs of voxels, does the prediction agree with the truth about
    whether they belong to one object?
    """
    return rand_error_from_table(contingency(truth, prediction, ignore_id), background_id)


def rand_error_from_table(table: Table, background_id: int | None = 0) -> float:
    """Adapted Rand error from an already-computed table. See `adapted_rand_error`."""
    ids_t, ids_p, counts, _ = table
    if background_id is not None:
        keep = ids_t != background_id
        ids_t, ids_p, counts = ids_t[keep], ids_p[keep], counts[keep]
    if counts.size == 0:
        return 1.0
    # Precision = sum n_ij^2 / sum n_.j^2, recall = the same over the other marginal: the fraction
    # of co-labelled voxel pairs the two labellings agree on, in each direction.
    sum_joint = float((counts.astype(np.float64) ** 2).sum())
    squared_pred = float((_group_sums(ids_p, counts).astype(np.float64) ** 2).sum())
    squared_true = float((_group_sums(ids_t, counts).astype(np.float64) ** 2).sum())
    if squared_pred == 0.0 or squared_true == 0.0:
        return 1.0
    precision = sum_joint / squared_pred
    recall = sum_joint / squared_true
    if precision + recall == 0.0:
        return 1.0
    return 1.0 - 2.0 * precision * recall / (precision + recall)


def panoptic_quality(
    truth: np.ndarray, prediction: np.ndarray, iou_threshold: float = 0.5,
    ignore_id: int | None = None, background_id: int | None = 0,
) -> dict[str, float]:
    """Matched-instance quality: PQ = SQ x RQ at a fixed IoU threshold.

    At a threshold above 0.5 the matching is unambiguous -- an object can overlap at most one other
    by more than half its union -- which is why 0.5 is the standard choice and why this needs no
    greedy assignment.
    """
    return pq_from_table(
        contingency(truth, prediction, ignore_id), iou_threshold, background_id
    )


def pq_from_table(
    table: Table, iou_threshold: float = 0.5, background_id: int | None = 0
) -> dict[str, float]:
    """Panoptic quality from an already-computed table. See `panoptic_quality`."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
    ids_t, ids_p, counts, _ = table

    def sizes(ids: np.ndarray) -> dict[int, int]:
        """Voxels per id in this region, summed over the joint table's rows."""
        distinct = np.unique(ids)
        return dict(zip(
            (int(i) for i in distinct),
            (int(c) for c in _group_sums(ids, counts)),
            strict=True,
        ))

    size_t, size_p = sizes(ids_t), sizes(ids_p)
    real_t = {i for i in size_t if background_id is None or i != background_id}
    real_p = {i for i in size_p if background_id is None or i != background_id}

    matched_iou, matched_t, matched_p = [], set(), set()
    for t, p, overlap in zip(ids_t, ids_p, counts, strict=True):
        t, p, overlap = int(t), int(p), int(overlap)
        if t not in real_t or p not in real_p:
            continue
        iou = overlap / (size_t[t] + size_p[p] - overlap)
        if iou > iou_threshold:
            matched_iou.append(iou)
            matched_t.add(t)
            matched_p.add(p)

    true_positive = len(matched_iou)
    false_negative = len(real_t) - len(matched_t)
    false_positive = len(real_p) - len(matched_p)
    denominator = true_positive + 0.5 * false_positive + 0.5 * false_negative
    segmentation_quality = float(np.mean(matched_iou)) if matched_iou else 0.0
    recognition_quality = true_positive / denominator if denominator else 0.0
    return {
        "pq": segmentation_quality * recognition_quality,
        "sq": segmentation_quality,
        "rq": recognition_quality,
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "instances_truth": float(len(real_t)),
        "instances_predicted": float(len(real_p)),
    }


@MetricRegistry.register("voxel_instance")
class VoxelInstance(BaseMetric):
    """VOI, adapted Rand and panoptic quality against a dense voxel labelling.

    `truth` is an integer array the same shape as the prediction. Ranked on `pq`, which is bounded
    in [0, 1] and, unlike VOI, does not change scale with the number of objects in the region.
    """

    canonical = "instances"
    consumes = "labels"
    higher_is_better = True
    primary = "pq"
    # voi_merge and voi_split first: they are what distinguishes one merged blob from a shattered
    # one, and both failures give PQ near zero. `instances_predicted` is deliberately absent -- it
    # is dominated by dust (median component size measured at 2 voxels on a real prediction) and
    # reads as over-segmentation when the actual failure was the opposite.
    report_keys = ("voi_merge", "voi_split", "sq", "rq", "adapted_rand_error")

    def __init__(self, iou_threshold: float = 0.5, **settings: Any) -> None:
        super().__init__(iou_threshold=iou_threshold, **settings)
        self.iou_threshold = float(iou_threshold)

    def __call__(self, prediction: np.ndarray, truth: Any, **context: Any) -> dict[str, float]:
        ignore_id = context.get("ignore_id")
        background_id = context.get("background_id", 0)
        # ONE table for all three. Each metric used to call `contingency` itself, so a
        # 7-gigavoxel volume factorised both labellings three times and the scoring job was
        # killed by the memory limit -- each pass allocates two int32 code arrays, 56 GB at that
        # size. The module docstring claimed "computed once" long before the code did it.
        table = contingency(np.asarray(truth), prediction, ignore_id)
        scores = pq_from_table(table, self.iou_threshold, background_id)
        scores.update(voi_from_table(table, background_id))
        scores["adapted_rand_error"] = rand_error_from_table(table, background_id)
        scores["iou_threshold"] = self.iou_threshold
        return scores
