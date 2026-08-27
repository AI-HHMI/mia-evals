"""Affinities -> instances by thresholding the short-range channels and taking components.

This is the published NISB baseline's post-processing, and the components pass itself is upstream's
own code (`utils.connected_components`), untouched. What this file adds is only the wrapper: the
threshold parameterisation, and the sweep that the runner fits on validation.

**The threshold is `sigmoid(0.2 * L)` for integer logits `L`.** That is BANIS' `eval_ranges`, and
the reason to keep it is that it is the grid the published numbers were selected on -- a different
grid would score a differently-tuned model. Logits rather than probabilities in the config for the
same reason: `[3, 4, 5, 6, 7]` is what the baseline sweeps, and the probabilities it corresponds to
(0.646 ... 0.802) are unmemorable and invite being rounded into a different sweep.

**Only the short-range channels are read.** The baseline keeps three at inference
(`prediction_channels = 3`) and this matches it, so the numbers stay comparable. The long-range
channels are still trained -- they are half the loss -- and `mws` is the postprocessor that uses
them.

**`min_size` drops components below a voxel count, and it is fitted, not fixed.** Thresholded
components on a real affinity map produce enormous numbers of tiny fragments: one scored checkpoint
returned 3,583,131 predicted objects against 3,620 true ones, i.e. 3,583,013 false positives.
PQ's recognition term is `TP / (TP + 0.5*FP + 0.5*FN)` and counts objects *unweighted by size*, so
a two-voxel speck costs exactly as much as a missed neuron and RQ is driven to ~0 arithmetically,
independent of how well the tissue was actually segmented. On a 256^3 block at the fitted logit,
dropping components under 500 voxels moved FP 299 -> 13 and PQ 0.065 -> 0.352 with the true-positive
count unchanged.

The filter only ever deletes a whole component. It cannot move a boundary, split a merged object or
join a split one -- so it is honest about *dust* and says nothing about *topology*. VOI is
size-weighted and barely responds to it, which is exactly why both are reported: `min_size` cleans
up what PQ over-counts and leaves what VOI measures alone.

Being a swept parameter, it is chosen on validation like the logit. A size filter tuned on the
reported split would be selecting on the reported number, and "PQ after discarding everything
small" is only a meaningful claim when the threshold was not chosen to maximise it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import BasePostprocess
from .registry import PostprocessRegistry
from .size_filter import drop_small_components

#: BANIS' `scale_sigmoid` factor. Both codebases train with plain
#: `binary_cross_entropy_with_logits`, so a logit is directly comparable; this is only how the
#: baseline stores and thresholds them.
LOGIT_SCALE = 0.2


def threshold_of(logit: float) -> float:
    """`sigmoid(0.2 * logit)`, the probability a swept logit corresponds to."""
    return 1.0 / (1.0 + math.exp(-LOGIT_SCALE * logit))


@PostprocessRegistry.register("cc_threshold")
class ConnectedComponentThreshold(BasePostprocess):
    """Threshold the short-range affinities, then label 6-connected components of what survives.

    `logits` and `min_sizes` are the sweep, and `search_space()` is their cross-product. The runner
    scores each candidate on validation and applies the winner to test; sweeping on the reported
    split would be selecting on the reported number.
    """

    accepts = ("affinity",)
    produces = "instances"

    def __init__(
        self,
        logits: tuple[float, ...] | list[float] = (3, 4, 5, 6, 7),
        min_sizes: tuple[int, ...] | list[int] = (0,),
        short_range_channels: int = 3,
        **settings: Any,
    ) -> None:
        super().__init__(
            logits=logits,
            min_sizes=min_sizes,
            short_range_channels=short_range_channels,
            **settings,
        )
        if not logits:
            raise ValueError(
                "cc_threshold with an empty `logits` has nothing to sweep and would produce no "
                "segmentation at all. Give at least one logit."
            )
        if not min_sizes:
            raise ValueError(
                "cc_threshold with an empty `min_sizes` has nothing to sweep. Use `[0]` for no "
                "size filter, which is the default."
            )
        if any(int(v) < 0 for v in min_sizes):
            raise ValueError(f"min_sizes must be non-negative voxel counts, got {list(min_sizes)}")
        self.logits = tuple(float(v) for v in logits)
        # Sorted so a sweep is reported in a readable order regardless of how the config lists it.
        self.min_sizes = tuple(sorted({int(v) for v in min_sizes}))
        self.short_range_channels = int(short_range_channels)

    def reads_channels(self) -> int | None:
        return self.short_range_channels

    def search_space(self) -> list[dict[str, Any]]:
        """Every (logit, min_size) pair, logit-major.

        A full cross-product rather than a two-stage fit, because the two are not independent in
        principle: a higher threshold fragments more, so it invites a larger size filter. The cost
        is real -- a 9-logit sweep over four LICONN/zebrafish volumes took 4.1 hours -- so a config
        that has already established its logit can pin `logits` to that one value and sweep only
        `min_sizes`, which is a reduction the config makes explicit rather than one hidden here.
        """
        return [
            {"logit": logit, "min_size": min_size}
            for logit in self.logits
            for min_size in self.min_sizes
        ]

    def __call__(self, array: np.ndarray, **params: Any) -> np.ndarray:
        # Imported here, not at module scope: numba compiles the pass on first call, which neither
        # `--help` nor a test that only checks the search space should pay for.
        from utils.connected_components import compute_connected_component_segmentation

        logit = float(params["logit"])
        if array.shape[0] < self.short_range_channels:
            raise ValueError(
                f"cc_threshold reads the first {self.short_range_channels} channels, but this "
                f"array has {array.shape[0]}"
            )
        # Compared at the stored dtype. Promoting a whole cube of float16 to float32 costs 146 GiB
        # and holds 219 GiB during the copy, and float16's ~1e-3 resolution near 0.65-0.80 is far
        # finer than the gap between successive sweep thresholds, so the comparison is unaffected.
        hard = np.ascontiguousarray(
            array[: self.short_range_channels] > threshold_of(logit)
        )
        # Returned in the components pass's own dtype (uint32) rather than widened to int64. The
        # metrics factorise ids rather than assuming a width, and at 7 gigavoxels the cast is not
        # free: it holds the uint32 result and the int64 copy at once, 85 GB where 28 GB will do.
        labels = np.asarray(compute_connected_component_segmentation(hard))
        return drop_small_components(labels, int(params.get("min_size", 0)))

    def describe(self, params: dict[str, Any]) -> str:
        logit = float(params["logit"])
        text = f"cc_threshold(logit={logit:+g}, thr={threshold_of(logit):.4f}"
        # Named in the description, not just stored, because it changes the score by more than most
        # model differences do: a leaderboard row without it invites comparing a filtered PQ against
        # an unfiltered one.
        min_size = int(params.get("min_size", 0))
        return text + (f", min_size={min_size}" if min_size else "") + ")"
