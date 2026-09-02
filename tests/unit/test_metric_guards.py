"""Panoptic quality must not be satisfiable by segment statistics alone.

This exists because of a measured control rather than a hypothetical. `docs/controls.md` runs mutex
watershed on the model's own affinity field translated bodily by a third of the volume: the result
has the model's exact per-channel affinity distribution and 58,473 segments against the model's
57,542 -- within 1.6% on segment count, and so indistinguishable from it on any size statistic --
and it scores `pq = 0.0000` with zero matched objects.

That is the property worth pinning. The leaderboard's fitted `size_filter(min_size=50000)` deletes
roughly 43% of the objects mutex watershed recovers, which makes the ranking metric look vulnerable
to a prediction that merely emits plausibly-sized blobs. It is not, and these tests fail if that ever
changes -- for instance if someone relaxed `iou_threshold` past 0.5, below which a translated
prediction starts matching. A regression here would silently promote whichever entry produced the
most convincingly-sized debris.

The truth here is synthetic and the assertions follow from the definition of IoU, so these are not
tied to any dataset or checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from metrics.voxel_instance import panoptic_quality

pytestmark = pytest.mark.unit

SIDE = 6        # cube edge, in voxels
PERIOD = 9      # lattice period, so cubes are separated by 3 voxels of background
REPEATS = 5     # cubes per axis -> 125 objects


def _lattice() -> np.ndarray:
    """A cubic lattice of `SIDE`-voxel cubes with distinct ids, separated by background.

    Deliberately regular: a lattice is the *most* favourable possible case for a prediction that
    gets segment sizes right and locations wrong, because every object is interchangeable with every
    other. If translation can be made to score on anything, it scores here.
    """
    extent = PERIOD * REPEATS
    truth = np.zeros((extent,) * 3, dtype=np.int64)
    label = 1
    for i in range(REPEATS):
        for j in range(REPEATS):
            for k in range(REPEATS):
                box = (
                    slice(i * PERIOD, i * PERIOD + SIDE),
                    slice(j * PERIOD, j * PERIOD + SIDE),
                    slice(k * PERIOD, k * PERIOD + SIDE),
                )
                truth[box] = label
                label += 1
    return truth


def test_identical_labelling_scores_one() -> None:
    """The upper anchor: without it, "translated scores 0" could just mean the metric is broken."""
    truth = _lattice()
    scores = panoptic_quality(truth, truth.copy())
    assert scores["pq"] == pytest.approx(1.0)
    assert scores["true_positive"] == REPEATS**3


@pytest.mark.parametrize("shift", [2, 3, 4, PERIOD // 2, PERIOD * REPEATS // 3])
def test_translation_scores_zero_with_statistics_preserved(shift: int) -> None:
    """Translate the truth and every size statistic survives, but nothing matches.

    Two cubes of edge `s` offset by `d` along one axis have IoU `(s - d) / (s + d)`, which exceeds
    0.5 only while `d < s / 3` -- here `d < 2`. So every shift of 2 or more must score zero, and the
    smallest parametrised case sits exactly on the boundary (IoU = 0.5, and the threshold is a strict
    inequality). The lattice period is coprime enough to the shifts that a translated cube never
    lands on a *different* cube's position either.
    """
    truth = _lattice()
    prediction = np.roll(truth, shift, axis=0)

    # The prediction is the truth, moved. Identical object count and identical size distribution.
    truth_ids, truth_counts = np.unique(truth[truth > 0], return_counts=True)
    pred_ids, pred_counts = np.unique(prediction[prediction > 0], return_counts=True)
    assert truth_ids.size == pred_ids.size
    np.testing.assert_array_equal(np.sort(truth_counts), np.sort(pred_counts))

    scores = panoptic_quality(truth, prediction)
    assert scores["pq"] == 0.0
    assert scores["true_positive"] == 0.0
    # Every real object unmatched on both sides, rather than the prediction being empty.
    assert scores["false_negative"] == REPEATS**3
    assert scores["false_positive"] == REPEATS**3


def test_sub_threshold_translation_still_matches() -> None:
    """The complement: a shift of 1 gives IoU 5/7 > 0.5, so it MUST match.

    Without this the zero above would be consistent with a metric that never matches anything
    imperfect, which would be a different bug with the same symptom.
    """
    truth = _lattice()
    scores = panoptic_quality(truth, np.roll(truth, 1, axis=0))
    assert scores["true_positive"] == REPEATS**3
    assert scores["sq"] == pytest.approx(5 / 7, rel=1e-6)


def test_plausibly_sized_debris_scores_near_zero() -> None:
    """Right number of segments, right order of size, no spatial agreement -> near zero.

    The `random` arm of `docs/controls.md` in miniature. Distinct from the translation case: this
    prediction is not a rearrangement of the truth, it is unrelated structure with comparable
    statistics, which is what an over-fragmenting model plus a size filter actually produces.

    **Near zero, not zero, and the difference is instructive.** 125 cubes dropped at random into a
    45^3 lattice occasionally land within one voxel of a true cube, which really does clear
    IoU > 0.5 -- so a couple of chance matches are correct behaviour, not a metric defect. This
    synthetic is far denser than any real volume: the measured `random` arm scored exactly 0.0000
    on 105 Mvox with irregular objects, where chance alignment is negligible. The assertion is
    therefore "does not reward it", which is the property that matters, rather than a zero the
    geometry cannot honestly deliver.
    """
    truth = _lattice()
    rng = np.random.default_rng(0)
    # Same object count, same cube size, positions drawn at random.
    prediction = np.zeros_like(truth)
    extent = truth.shape[0]
    for label in range(1, REPEATS**3 + 1):
        corner = rng.integers(0, extent - SIDE, size=3)
        prediction[tuple(slice(c, c + SIDE) for c in corner)] = label

    assert np.unique(prediction[prediction > 0]).size > 0.8 * REPEATS**3
    scores = panoptic_quality(truth, prediction)
    assert scores["pq"] < 0.05
    # Chance alignment only: a handful out of 125, against 125 for the identical labelling.
    assert scores["true_positive"] < 0.05 * REPEATS**3


def test_single_merged_blob_scores_zero() -> None:
    """The opposite failure: one label over everything. Bounds the metric from the other side."""
    truth = _lattice()
    prediction = np.ones_like(truth)
    scores = panoptic_quality(truth, prediction)
    assert scores["pq"] == 0.0
    assert scores["true_positive"] == 0.0
