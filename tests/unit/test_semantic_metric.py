"""The semantic metric: classes as sets of label ids, truth-side ignore, per-volume vs pooled.

Every expected value is counted independently, voxel by voxel, rather than through a confusion
matrix -- the matrix is what is under test.
"""

from __future__ import annotations

import numpy as np
import pytest

import components  # noqa: F401  (populates the registries)
from metrics.semantic import Semantic

pytestmark = pytest.mark.unit


def brute_iou(truth, prediction, ids, ignore=()):
    keep = ~np.isin(truth, list(ignore))
    t, p = np.isin(truth[keep], ids), np.isin(prediction[keep], ids)
    return np.count_nonzero(t & p) / np.count_nonzero(t | p)


def noisy(seed: int, shape=(10, 10, 10), ids: int = 6):
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, ids, shape)
    prediction = truth.copy()
    flip = rng.random(shape) < 0.3
    prediction[flip] = rng.integers(0, ids, int(flip.sum()))
    return truth, prediction


def test_every_class_matches_a_voxel_count_including_overlapping_composites():
    truth, prediction = noisy(0)
    classes = {"a": [1], "b": [2, 3], "c": [3, 4, 5], "all": [1, 2, 3, 4, 5]}
    result = Semantic(num_classes=8, classes=classes, ignore_truth=[0])(prediction, truth)
    expected = {name: brute_iou(truth, prediction, ids, ignore=[0])
                for name, ids in classes.items()}
    for name, value in expected.items():
        assert result[f"iou/{name}"] == pytest.approx(value)
    assert result["mean_iou"] == pytest.approx(np.mean(list(expected.values())))
    dice = [2 * v / (1 + v) for v in expected.values()]
    assert result["mean_dice"] == pytest.approx(np.mean(dice))
    keep = truth != 0
    assert result["pixel_accuracy"] == pytest.approx(np.mean(truth[keep] == prediction[keep]))


def test_a_composite_id_in_the_truth_counts_for_the_composite_and_against_its_leaves():
    """CellMap paints id 37 (`nuc`) where only the whole nucleus was annotated."""
    truth = np.array([1, 2, 9])
    prediction = np.array([1, 2, 1])
    classes = {"leaf1": [1], "leaf2": [2], "both": [1, 2, 9]}
    result = Semantic(num_classes=10, classes=classes)(prediction, truth)
    assert result["iou/leaf1"] == pytest.approx(0.5)     # the composite voxel is not leaf 1
    assert result["iou/leaf2"] == pytest.approx(1.0)
    assert result["iou/both"] == pytest.approx(1.0)


def test_ignored_truth_and_the_artifacts_ignore_id_leave_voxels_out_entirely():
    truth = np.array([0, 0, 7, 1, 1, 2])
    prediction = np.array([2, 2, 2, 1, 1, 2])
    metric = Semantic(num_classes=8, classes={"one": [1], "two": [2]}, ignore_truth=[0])
    result = metric(prediction, truth, ignore_id=7)
    assert result["iou/one"] == 1.0 and result["iou/two"] == 1.0
    assert metric.details()["confusion"]["ignored"] == 3


def test_each_call_returns_its_own_volume_and_result_pools_every_voxel():
    first, second = noisy(1), noisy(2, shape=(4, 5, 6))
    classes = {"a": [1, 2], "b": [3], "c": [4, 5]}
    metric = Semantic(num_classes=8, classes=classes, ignore_truth=[0])
    own = [metric(p, t) for t, p in (first, second)]
    for (truth, prediction), result in zip((first, second), own, strict=True):
        alone = Semantic(num_classes=8, classes=classes, ignore_truth=[0])(prediction, truth)
        assert result == alone
    pooled_truth = np.concatenate([first[0].ravel(), second[0].ravel()])
    pooled_prediction = np.concatenate([first[1].ravel(), second[1].ravel()])
    for name, ids in classes.items():
        assert metric.result()[f"iou/{name}"] == pytest.approx(
            brute_iou(pooled_truth, pooled_prediction, ids, ignore=[0])
        )
    metric.reset()
    assert metric.result()["classes_present"] == 0.0


def test_a_class_absent_from_the_truth_is_left_out_of_the_mean():
    truth = np.array([1, 1, 1, 1])
    prediction = np.array([1, 1, 2, 2])
    result = Semantic(num_classes=4, classes={"a": [1], "b": [2]})(prediction, truth)
    assert result["classes_present"] == 1.0
    assert result["mean_iou"] == pytest.approx(0.5)
    assert "iou/b" not in result


def test_a_predicted_id_outside_the_vocabulary_belongs_to_no_class():
    """It used to be clipped into the last id, crediting it to whatever class that was."""
    truth = np.array([3, 3, 3, 3])
    prediction = np.array([3, 3, 999, -5])
    metric = Semantic(num_classes=4, classes={"last": [3]})
    assert metric(prediction, truth)["iou/last"] == pytest.approx(0.5)
    assert metric.details()["confusion"]["predicted_ids"] == [3, -1]


def test_a_truth_id_outside_the_vocabulary_is_an_error():
    with pytest.raises(ValueError, match="raise it to cover the label vocabulary"):
        Semantic(num_classes=4)(np.array([1]), np.array([4]))


def test_details_hold_the_pooled_matrix_over_the_ids_that_occur():
    metric = Semantic(num_classes=8, classes={"a": [1]}, ignore_truth=[0])
    metric(np.array([1, 2, 2]), np.array([1, 1, 0]))
    metric(np.array([5]), np.array([1]))
    confusion = metric.details()["confusion"]
    assert confusion["truth_ids"] == [1]
    assert confusion["predicted_ids"] == [1, 2, 5]
    assert confusion["counts"] == [[1, 1, 1]]
    assert confusion["ignored"] == 1


def test_without_a_table_every_id_is_its_own_class():
    truth = np.array([0, 1, 1, 2])
    prediction = np.array([0, 1, 2, 2])
    result = Semantic(num_classes=3, class_names={"1": "mito"})(prediction, truth)
    assert result["iou/class_0"] == 1.0 and result["iou/mito"] == 0.5
    assert result["iou/class_2"] == 0.5 and result["classes_present"] == 3.0


@pytest.mark.parametrize("settings, message", [
    ({"classes": {"empty": []}}, "lists no ids"),
    ({"classes": {"far": [9]}}, "outside"),
    ({"classes": {"a": [1]}, "class_names": {"1": "a"}}, "not both"),
])
def test_a_class_table_is_checked_when_built(settings, message):
    with pytest.raises(ValueError, match=message):
        Semantic(num_classes=4, **settings)
