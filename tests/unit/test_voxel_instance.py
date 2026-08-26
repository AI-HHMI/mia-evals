"""Voxel instance metrics, including the convention that is invisible when wrong.

The split/merge orientation is the reason this file exists. `voi_split` and `voi_merge` are both
plausible numbers whichever way round they are assigned, so getting them backwards inverts every
diagnosis without changing anything that looks wrong -- and this repository reports skeleton VOI
through `funlib.evaluate`, so a disagreement would put two differently-defined columns on one
leaderboard. The parity test pins ours to funlib's, units included.
"""

from __future__ import annotations

import numpy as np
import pytest

from metrics.voxel_instance import (
    adapted_rand_error,
    panoptic_quality,
    variation_of_information,
)

pytestmark = pytest.mark.unit

# Two true objects (1 and 2) in a field of background.
TRUTH = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 0]], dtype=np.int64)


def _oversegmented() -> np.ndarray:
    """One true object broken into two predicted pieces."""
    out = TRUTH.copy()
    out[0, 2] = 5
    return out


def _undersegmented() -> np.ndarray:
    """Two true objects fused into one prediction."""
    out = TRUTH.copy()
    out[out == 2] = 1
    return out


def test_perfect_prediction_scores_perfectly():
    voi = variation_of_information(TRUTH, TRUTH)
    assert voi == {"voi_split": 0.0, "voi_merge": 0.0, "voi_sum": 0.0}
    assert adapted_rand_error(TRUTH, TRUTH) == pytest.approx(0.0)
    assert panoptic_quality(TRUTH, TRUTH)["pq"] == pytest.approx(1.0)


def test_split_and_merge_are_not_interchangeable():
    """Over-segmentation must move `voi_split` alone, and under-segmentation `voi_merge` alone."""
    over = variation_of_information(TRUTH, _oversegmented())
    assert over["voi_split"] > 0.0
    assert over["voi_merge"] == pytest.approx(0.0)

    under = variation_of_information(TRUTH, _undersegmented())
    assert under["voi_merge"] > 0.0
    assert under["voi_split"] == pytest.approx(0.0)


def test_panoptic_quality_penalises_both_failure_modes():
    assert panoptic_quality(TRUTH, _oversegmented())["pq"] < 1.0
    assert panoptic_quality(TRUTH, _undersegmented())["pq"] < 1.0

    counts = panoptic_quality(TRUTH, _undersegmented())
    # Object 2 disappeared into object 1: one true object goes unmatched.
    assert counts["false_negative"] >= 1.0


def test_background_is_excluded_so_it_cannot_dilute_a_real_error():
    """Counting background hides foreground mistakes in proportion to how much background there is.

    The prediction here gets every background voxel right and merges the two real objects. With
    background counted, that merge is averaged over the whole volume and reads as a small error;
    with background excluded it is the whole answer. Since background is the majority class in
    every volume in the corpus -- and overwhelmingly so in the large ones -- including it would let
    the same mistake score better the emptier the crop.
    """
    merged = _undersegmented()
    excluded = variation_of_information(TRUTH, merged, background_id=0)
    included = variation_of_information(TRUTH, merged, background_id=None)
    assert excluded["voi_sum"] > included["voi_sum"]

    # And the dilution grows with the amount of background, which is the part that makes it unsafe.
    padded_truth = np.pad(TRUTH, ((0, 8), (0, 0)))
    padded_merged = np.pad(merged, ((0, 8), (0, 0)))
    more_background = variation_of_information(padded_truth, padded_merged, background_id=None)
    assert more_background["voi_sum"] < included["voi_sum"]
    # Excluding it makes the score indifferent to how much empty space was scored, as it must be.
    assert variation_of_information(padded_truth, padded_merged, background_id=0)[
        "voi_sum"
    ] == pytest.approx(excluded["voi_sum"])


def test_ignore_id_removes_voxels_from_every_metric():
    truth = TRUTH.copy()
    truth[2, 3] = -1                     # mark one voxel unannotated
    wrong_there = TRUTH.copy()
    wrong_there[2, 3] = 99               # and predict nonsense at it
    assert panoptic_quality(truth, wrong_there, ignore_id=-1)["pq"] == pytest.approx(
        panoptic_quality(TRUTH, TRUTH)["pq"]
    )


@pytest.mark.parity
def test_matches_funlib_exactly():
    """Same convention *and* same units as the skeleton path's `rand_voi`.

    Skipped where `funlib.evaluate` is absent -- it is git-install-only, behind the `instance`
    extra -- but this is the test that makes the two VOI columns commensurable, so it must run
    wherever that extra is installed.
    """
    funlib = pytest.importorskip("funlib.evaluate")

    for prediction in (TRUTH, _oversegmented(), _undersegmented()):
        reference = funlib.rand_voi(
            TRUTH.astype(np.uint64), prediction.astype(np.uint64), return_cluster_scores=False
        )
        mine = variation_of_information(TRUTH, prediction)
        assert mine["voi_split"] == pytest.approx(reference["voi_split"], abs=1e-9)
        assert mine["voi_merge"] == pytest.approx(reference["voi_merge"], abs=1e-9)


# --------------------------------------------------- the two code paths must not diverge


def test_factorize_table_and_sort_paths_agree():
    """`_factorize` counts into a table when it can and sorts when it cannot.

    Two implementations of the same function is a standing invitation to divergence, so both are
    run on the same input here. The sort path is reached by an id space too wide for a table --
    genuine in this corpus, where MICrONS segment ids run past 2**40.
    """
    from metrics.voxel_instance import _factorize

    packed = np.array([5, 5, 9, 1, 9, 1, 5], dtype=np.int64)
    wide = np.array([2**41, 2**41, 2**45, 7, 2**45, 7, 2**41], dtype=np.int64)

    ids_packed, codes_packed = _factorize(packed)
    ids_wide, codes_wide = _factorize(wide)

    # Same structure either way: sorted distinct values, and codes indexing into them.
    assert ids_packed.tolist() == [1, 5, 9]
    assert ids_wide.tolist() == [7, 2**41, 2**45]
    assert codes_packed.tolist() == codes_wide.tolist()
    assert ids_packed[codes_packed].tolist() == packed.tolist()
    assert ids_wide[codes_wide].tolist() == wide.tolist()


def test_contingency_dense_and_sparse_paths_agree(monkeypatch):
    """The joint table is counted densely when it fits and sorted when it does not.

    Both branches must return the same triples. Forced by lowering `DENSE_TABLE_LIMIT` rather than
    by constructing a 128-million-cell case, so the *real* fallback runs -- a test that
    reimplemented the fallback would only prove the reimplementation right.

    The fallback is not hypothetical: it is the branch an over-fragmented prediction takes, which is
    exactly what a badly thresholded model produces.
    """
    from metrics import voxel_instance

    rng = np.random.default_rng(7)
    truth = rng.integers(0, 12, size=4000, dtype=np.int64)
    prediction = rng.integers(0, 40, size=4000, dtype=np.int64)

    dense = voxel_instance.contingency(truth, prediction)
    monkeypatch.setattr(voxel_instance, "DENSE_TABLE_LIMIT", 1)
    sparse = voxel_instance.contingency(truth, prediction)

    for left, right in zip(dense, sparse, strict=True):
        assert np.array_equal(np.asarray(left), np.asarray(right))
    # And the result is a real table, not two empties that trivially match.
    assert dense[2].sum() == truth.size
    assert dense[0].size > 100
