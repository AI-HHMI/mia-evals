"""Turning a prediction into something scoreable, and the hyperparameter that choice needs.

A postprocessor is the *only* place a prediction's format is interpreted. It declares which
artifact kinds it accepts and which canonical form it produces, and the runner checks both, so a
task cannot accidentally threshold class scores as if they were affinities.

**The fit belongs here, not in the runner.** `cc_threshold` fits one scalar, `per_class_threshold`
fits K of them, `mws` fits a stride, `identity` fits nothing. A generic sweep loop in the runner
would have to know which is which; instead each postprocessor states its own candidate set through
`search_space()`, and the runner's job reduces to "score every candidate on val, keep the best,
apply it to test". `identity` returning a single empty candidate is what lets an externally
produced segmentation be scored with no validation split at all -- there is nothing to choose.

That split matters beyond tidiness: a threshold chosen on the split being reported is selecting on
the number being reported. The runner enforces fit-on-val because no postprocessor can see which
split it is being handed.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np

from artifact import CANONICAL_FORMS, KINDS


class BasePostprocess(abc.ABC):
    """Prediction array -> integer labelling, under a set of fitted parameters."""

    #: Artifact kinds this accepts. Checked against the artifact's declaration by the runner.
    accepts: tuple[str, ...] = ()
    #: Which scoreable form this produces: "instances", "classes", or "same" for a postprocessor
    #: that preserves whatever form its input already was -- which only `identity` is, and only
    #: because splitting it into two classes differing by one string would make a config author
    #: choose between them for no reason.
    produces: str = ""

    def __init__(self, **settings: Any) -> None:
        self.settings = settings
        unknown = sorted(set(self.accepts) - set(KINDS))
        if unknown:
            raise ValueError(f"{type(self).__name__} accepts unknown kind(s) {unknown}")
        if self.produces not in (*CANONICAL_FORMS, "same"):
            raise ValueError(
                f"{type(self).__name__}.produces must be one of {(*CANONICAL_FORMS, 'same')}, "
                f"got {self.produces!r}"
            )

    def produces_for(self, artifact_canonical: str) -> str:
        """The canonical form this yields given an artifact whose form is `artifact_canonical`."""
        return artifact_canonical if self.produces == "same" else self.produces

    def search_space(self) -> list[dict[str, Any]]:
        """Candidate parameter sets to choose between on the validation split.

        One empty dict -- the default -- means this postprocessor has nothing to fit, so the
        winning candidate is trivially the only one and no validation data is consulted.
        """
        return [{}]

    @abc.abstractmethod
    def __call__(self, array: np.ndarray, **params: Any) -> np.ndarray:
        """Prediction -> (*spatial) integer labelling. `params` comes from `search_space()`."""

    def describe(self, params: dict[str, Any]) -> str:
        """How a fitted parameter set should read in a leaderboard row."""
        if not params:
            return type(self).__name__
        inner = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{type(self).__name__}({inner})"
