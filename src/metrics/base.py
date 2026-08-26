"""What a metric is, and the two declarations that keep a task honest about it.

A metric consumes a *canonical form* -- an instance labelling or a class labelling -- and never an
artifact. That is the whole point of the narrow waist: nERL cannot tell, and must not care, whether
the labelling it is handed came from thresholded affinities, a mutex watershed, or a file a
collaborator sent us.

Two declarations, both load-bearing:

**`consumes`.** `"labels"` for a metric computed from a hard labelling; `"scores"` for one that
needs the un-argmaxed prediction. Average precision is the case that forces the distinction: it is
a ranking metric and cannot be computed from an argmax at all. Because a whole-cube affinity
artifact is ~51 GB and gets deleted once a score exists, whether the artifact must be *retained* is
decided by this field rather than by a fixed policy -- which is the difference between being able to
add AP to a task later and having to re-predict everything.

**`higher_is_better`.** Stated by the metric, not by the task, so a task cannot rank on VOI as
though more were better. Every leaderboard ordering derives from this.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np

CONSUMES = ("labels", "scores")


class BaseMetric(abc.ABC):
    """One number (or a family of them) from a prediction and its ground truth."""

    #: "instances" or "classes" -- the canonical form this scores.
    canonical: str = ""
    #: "labels" or "scores"; see the module docstring.
    consumes: str = "labels"
    #: Whether a larger value is a better model.
    higher_is_better: bool = True
    #: The key in this metric's result dict that a task may rank on. Metrics returning several
    #: numbers (nERL also yields VOI, merge and split counts) name the headline one here.
    primary: str = ""

    def __init__(self, **settings: Any) -> None:
        self.settings = settings
        if self.consumes not in CONSUMES:
            raise ValueError(
                f"{type(self).__name__}.consumes must be one of {CONSUMES}, got {self.consumes!r}"
            )
        if not self.primary:
            raise ValueError(f"{type(self).__name__} must name a `primary` key")

    @abc.abstractmethod
    def __call__(self, prediction: np.ndarray, truth: Any, **context: Any) -> dict[str, float]:
        """Score one region. `truth` is whatever this metric's canonical form implies.

        `context` carries what a metric may need but must not assume: `origin` and `shape` of the
        region, `ignore_id`, `background_id`. Passed rather than fetched so a metric never reaches
        back into an artifact or a config.
        """
