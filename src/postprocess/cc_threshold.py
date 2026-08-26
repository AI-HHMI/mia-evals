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
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import BasePostprocess
from .registry import PostprocessRegistry

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

    `logits` is the sweep. The runner scores each on validation and applies the winner to test;
    sweeping on the reported split would be selecting on the reported number.
    """

    accepts = ("affinity",)
    produces = "instances"

    def __init__(
        self,
        logits: tuple[float, ...] | list[float] = (3, 4, 5, 6, 7),
        short_range_channels: int = 3,
        **settings: Any,
    ) -> None:
        super().__init__(logits=logits, short_range_channels=short_range_channels, **settings)
        if not logits:
            raise ValueError(
                "cc_threshold with an empty `logits` has nothing to sweep and would produce no "
                "segmentation at all. Give at least one logit."
            )
        self.logits = tuple(float(v) for v in logits)
        self.short_range_channels = int(short_range_channels)

    def reads_channels(self) -> int | None:
        return self.short_range_channels

    def search_space(self) -> list[dict[str, Any]]:
        return [{"logit": logit} for logit in self.logits]

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
        return np.asarray(compute_connected_component_segmentation(hard))

    def describe(self, params: dict[str, Any]) -> str:
        logit = float(params["logit"])
        return f"cc_threshold(logit={logit:+g}, thr={threshold_of(logit):.4f})"
