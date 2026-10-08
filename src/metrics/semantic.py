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

**A class is a set of label ids.** The matrix counts raw ids (truth row, predicted column), and a
reported class sums a block of it: a leaf class is one id, a composite one the union of its
members, so CellMap's `mito` is `[3, 4, 5, 50]` -- membrane, lumen, ribosomes, and the id a crop
uses where only the whole organelle was painted. Hierarchical classes then come out of a single
mutually-exclusive labelling exactly, with no second pass over the data, and a voxel whose truth is
a composite id counts for the composite and against each of its leaves -- which is how CellMap's
own per-class arrays count it. Without `classes`, every id is its own class.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseMetric, KeyInfo
from .registry import MetricRegistry


class ConfusionMatrix:
    """Counts of (true id, predicted id), accumulated across every region scored.

    Truth ids live in `[0, num_classes)`; anything outside is an error, because it means the
    vocabulary was configured too small or an unannotated value was not declared ignorable. A
    *predicted* id outside that range is a class no truth can have, and is counted in one extra
    column that belongs to no class -- it used to be clipped into the last id, which silently
    credited it to whichever class that was.
    """

    def __init__(self, num_classes: int) -> None:
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        self.num_classes = num_classes
        self.counts = np.zeros((num_classes, num_classes + 1), dtype=np.int64)
        self.ignored = 0

    def update(
        self, predicted: np.ndarray, truth: np.ndarray, ignore: tuple[int, ...] = ()
    ) -> None:
        predicted = np.asarray(predicted).ravel()
        truth = np.asarray(truth).ravel()
        if predicted.shape != truth.shape:
            raise ValueError(
                f"prediction has {predicted.size} voxels but the truth {truth.size}; they were not "
                "put on one grid"
            )
        if ignore:
            valid = ~np.isin(truth, np.asarray(ignore))
            self.ignored += int(truth.size - np.count_nonzero(valid))
            predicted, truth = predicted[valid], truth[valid]
        out_of_range = (truth >= self.num_classes) | (truth < 0)
        if out_of_range.any():
            raise ValueError(
                f"ground truth holds class id {int(truth[out_of_range].max())} but num_classes="
                f"{self.num_classes}; raise it to cover the label vocabulary, or list that value "
                "in ignore_truth if it means 'unannotated'"
            )
        width = self.num_classes + 1
        columns = np.where(
            (predicted >= 0) & (predicted < self.num_classes), predicted, self.num_classes
        ).astype(np.int64)
        self.counts += np.bincount(
            truth.astype(np.int64) * width + columns, minlength=self.num_classes * width
        ).reshape(self.num_classes, width)

    def merge(self, other: ConfusionMatrix) -> None:
        self.counts += other.counts
        self.ignored += other.ignored


@MetricRegistry.register("semantic")
class Semantic(BaseMetric):
    """IoU and Dice against a dense class labelling.

    Stateful across calls on purpose: the runner scores each region in turn and this accumulates
    into one matrix, which is what makes the reported mean an over-the-set number rather than an
    average of per-region ones. Each call returns that region's own numbers; `result()` returns
    the pooled ones, and `reset()` starts over between splits.

    Settings: `num_classes`, the size of the id space (256 for a uint8 labelling); `classes`, an
    optional `{name: [ids]}` table of the classes reported, each the union of its ids;
    `class_names`, `{id: name}` for the default one-class-per-id table; `ignore_truth`, truth values
    left out entirely (CellMap's 0, unannotated). Truth equal to the artifact's own `ignore_id` is
    left out as well.
    """

    canonical = "classes"
    consumes = "labels"
    higher_is_better = True
    accumulates = True          # one confusion matrix over every volume; see BaseMetric
    primary = "mean_iou"
    report_keys = ("mean_dice", "pixel_accuracy", "classes_present")
    key_info = {
        "mean_iou": KeyInfo(True, "Mean intersection-over-union over the classes present."),
        "mean_dice": KeyInfo(True, "Mean Dice coefficient over the classes present."),
        "pixel_accuracy": KeyInfo(True, "Fraction of voxels given the correct class."),
        "classes_present": KeyInfo(None, "Number of classes present in the ground truth."),
    }

    def __init__(
        self,
        num_classes: int = 64,
        class_names: dict[int, str] | None = None,
        classes: dict[str, list[int]] | None = None,
        ignore_truth: list[int] | tuple[int, ...] = (),
        **settings: Any,
    ) -> None:
        super().__init__(num_classes=num_classes, class_names=class_names, classes=classes,
                         ignore_truth=list(ignore_truth), **settings)
        self.num_classes = int(num_classes)
        # TOML has no integer keys, so a config's `class_names` arrives keyed by string.
        self.class_names = {int(k): str(v) for k, v in (class_names or {}).items()}
        self.ignore_truth = tuple(int(v) for v in ignore_truth)
        #: Reported class name -> the ids it is the union of.
        self.groups: dict[str, tuple[int, ...]]
        if classes is None:
            self.groups = {
                self.class_names.get(i, f"class_{i}"): (i,) for i in range(self.num_classes)
            }
        else:
            if self.class_names:
                raise ValueError("give either `classes` or `class_names`, not both")
            self.groups = {}
            for name, ids in classes.items():
                members = tuple(sorted({int(i) for i in ids}))
                if not members:
                    raise ValueError(f"class {name!r} lists no ids")
                bad = [i for i in members if not 0 <= i < self.num_classes]
                if bad:
                    raise ValueError(
                        f"class {name!r} lists id(s) {bad} outside "
                        f"[0, num_classes={self.num_classes})"
                    )
                self.groups[str(name)] = members
        width = self.num_classes + 1
        self._names = list(self.groups)
        self._truth_members = np.zeros((len(self._names), self.num_classes), dtype=np.float64)
        self._predicted_members = np.zeros((len(self._names), width), dtype=np.float64)
        for row, name in enumerate(self._names):
            self._truth_members[row, list(self.groups[name])] = 1.0
            self._predicted_members[row, list(self.groups[name])] = 1.0
        self.confusion = ConfusionMatrix(self.num_classes)

    def reset(self) -> None:
        self.confusion = ConfusionMatrix(self.num_classes)

    def _metrics(self, confusion: ConfusionMatrix) -> dict[str, float]:
        counts = confusion.counts.astype(np.float64)
        hit = np.einsum("gi,ij,gj->g", self._truth_members, counts, self._predicted_members)
        actual = self._truth_members @ counts.sum(axis=1)
        predicted = self._predicted_members @ counts.sum(axis=0)
        union = actual + predicted - hit
        present = actual > 0
        iou = np.where(union > 0, hit / np.maximum(union, 1), 0.0)
        dice = np.where(actual + predicted > 0, 2 * hit / np.maximum(actual + predicted, 1), 0.0)
        result = {
            "pixel_accuracy": float(np.trace(counts[:, :self.num_classes]) / max(counts.sum(), 1)),
            "mean_iou": float(iou[present].mean()) if present.any() else 0.0,
            "mean_dice": float(dice[present].mean()) if present.any() else 0.0,
            "classes_present": float(present.sum()),
        }
        # Per class, where present. Dice is not repeated per class: for one class it is
        # 2 * IoU / (1 + IoU), so it would add nothing but bulk to every record.
        for row in np.flatnonzero(present).tolist():
            result[f"iou/{self._names[row]}"] = float(iou[row])
        return result

    def __call__(self, prediction: np.ndarray, truth: Any, **context: Any) -> dict[str, float]:
        ignore = self.ignore_truth
        if context.get("ignore_id") is not None:
            ignore = (*ignore, int(context["ignore_id"]))
        this = ConfusionMatrix(self.num_classes)
        this.update(prediction, np.asarray(truth), ignore)
        self.confusion.merge(this)
        return self._metrics(this)

    def result(self) -> dict[str, float]:
        return self._metrics(self.confusion)

    def details(self) -> dict[str, Any]:
        """The pooled confusion matrix, restricted to the ids that occur, plus the ignored count.

        `counts[r][c]` is the number of voxels whose truth is `truth_ids[r]` and whose prediction is
        `predicted_ids[c]`; a predicted id of -1 stands for every id outside the vocabulary. Any
        other grouping of classes can be scored from this without re-reading a voxel.
        """
        counts = self.confusion.counts
        rows = np.flatnonzero(counts.sum(axis=1))
        columns = np.flatnonzero(counts.sum(axis=0))
        return {"confusion": {
            "truth_ids": rows.tolist(),
            "predicted_ids": [-1 if c == self.num_classes else int(c) for c in columns],
            "counts": counts[np.ix_(rows, columns)].tolist(),
            "ignored": int(self.confusion.ignored),
        }}
