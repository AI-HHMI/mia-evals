"""Semantic segmentation metrics: IoU and Dice from one confusion matrix over the whole set.

Absorbed from `mia-train/src/evals/semantic_seg.py`, which registered this computation but was
wired to nothing -- no config section built it and no engine code called it -- so it moved here
without changing any behaviour in that repo.

**One confusion matrix over everything, not an average of per-crop scores.** IoU is a ratio of
sums. Averaging per-volume IoUs would weight a crop containing three voxels of a class the same as
one containing three million, and would have to invent a value for the volumes where a class is
absent. Accumulating counts and dividing once has neither problem.

**Mean IoU over classes actually present in the ground truth.** These volumes are dominated by
background, so pixel accuracy alone is close to meaningless -- it is reported, but a model that
predicts background everywhere scores well on it and 1/K on mean IoU. A class absent from the
region contributes nothing rather than a fabricated zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseMetric
from .registry import MetricRegistry


class ConfusionMatrix:
    """Counts of (true class, predicted class), accumulated across every region scored."""

    def __init__(self, num_classes: int) -> None:
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        self.num_classes = num_classes
        self.counts = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(
        self, predicted: np.ndarray, truth: np.ndarray, ignore_id: int | None = None
    ) -> None:
        predicted = np.asarray(predicted).ravel()
        truth = np.asarray(truth).ravel()
        if ignore_id is not None:
            valid = truth != ignore_id
            predicted, truth = predicted[valid], truth[valid]
        out_of_range = (truth >= self.num_classes) | (truth < 0)
        if out_of_range.any():
            raise ValueError(
                f"ground truth holds class id {int(truth[out_of_range].max())} but num_classes="
                f"{self.num_classes}; raise it to cover the label vocabulary, or set ignore_id if "
                "that value means 'unannotated'"
            )
        indices = truth * self.num_classes + np.clip(predicted, 0, self.num_classes - 1)
        self.counts += np.bincount(
            indices, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def metrics(self, class_names: dict[int, str] | None = None) -> dict[str, float]:
        counts = self.counts.astype(np.float64)
        true_positive = np.diag(counts)
        actual, predicted = counts.sum(1), counts.sum(0)
        union = actual + predicted - true_positive
        present = actual > 0

        iou = np.where(union > 0, true_positive / np.maximum(union, 1), 0.0)
        dice = np.where(
            (actual + predicted) > 0,
            2 * true_positive / np.maximum(actual + predicted, 1),
            0.0,
        )
        result = {
            "pixel_accuracy": float(true_positive.sum() / max(counts.sum(), 1)),
            "mean_iou": float(iou[present].mean()) if present.any() else 0.0,
            "mean_dice": float(dice[present].mean()) if present.any() else 0.0,
            "classes_present": float(present.sum()),
        }
        for klass in np.flatnonzero(present).tolist():
            # Named where the task supplies a vocabulary. OME-NGFF `image-label` metadata carries
            # `colors` but no `properties`, so nothing on disk names these ids -- an unnamed class
            # reads as `iou/class_37`, which is honest but unusable in a leaderboard column.
            name = (class_names or {}).get(klass, f"class_{klass}")
            result[f"iou/{name}"] = float(iou[klass])
        return result


@MetricRegistry.register("semantic")
class Semantic(BaseMetric):
    """IoU and Dice against a dense class labelling.

    Stateful across calls on purpose: the runner scores each region in turn and this accumulates
    into one matrix, which is what makes the reported mean an over-the-set number rather than an
    average of per-region ones. `reset()` between splits.
    """

    canonical = "classes"
    consumes = "labels"
    higher_is_better = True
    accumulates = True          # one confusion matrix over every volume; see BaseMetric
    primary = "mean_iou"
    report_keys = ("mean_dice", "pixel_accuracy", "classes_present")

    def __init__(
        self,
        num_classes: int = 64,
        class_names: dict[int, str] | None = None,
        **settings: Any,
    ) -> None:
        super().__init__(num_classes=num_classes, class_names=class_names, **settings)
        self.num_classes = int(num_classes)
        # TOML has no integer keys, so a config's `class_names` arrives keyed by string.
        self.class_names = {int(k): str(v) for k, v in (class_names or {}).items()}
        self.confusion = ConfusionMatrix(self.num_classes)

    def reset(self) -> None:
        self.confusion = ConfusionMatrix(self.num_classes)

    def __call__(self, prediction: np.ndarray, truth: Any, **context: Any) -> dict[str, float]:
        self.confusion.update(prediction, np.asarray(truth), context.get("ignore_id"))
        return self.confusion.metrics(self.class_names)
