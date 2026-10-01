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
import inspect
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
        # The settings a postprocessor takes are its own __init__'s named parameters. Anything else
        # would be kept and never read: the config would run without it while its record still
        # listed it (`fill_distances` on cc_threshold, 2026-09-29).
        variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        takes = sorted(
            name for name, parameter in inspect.signature(type(self).__init__).parameters.items()
            if name != "self" and parameter.kind not in variadic
        )
        unknown = sorted(set(settings) - set(takes))
        if unknown:
            raise ValueError(
                f"[postprocess] {type(self).__name__} has unknown key(s) {unknown}; "
                f"it takes {takes or 'none'}"
            )
        self.settings = settings
        unknown = sorted(set(self.accepts) - set(KINDS))
        if unknown:
            raise ValueError(f"{type(self).__name__} accepts unknown kind(s) {unknown}")
        if self.produces not in (*CANONICAL_FORMS, "same"):
            raise ValueError(
                f"{type(self).__name__}.produces must be one of {(*CANONICAL_FORMS, 'same')}, "
                f"got {self.produces!r}"
            )

    def reads_channels(self) -> int | None:
        """How many leading channels this consumes, or None for all of them.

        Declared so the runner can read only what will be used. Not a micro-optimisation at these
        sizes: the zebrafish doublecube's six affinity channels are 85 GB as float16 while
        `cc_threshold` reads three, and the 43 GB saved is the difference between fitting in a
        300 GB reservation and being killed part-way through a ten-hour job.
        """
        return None

    def produces_for(self, artifact_canonical: str) -> str:
        """The canonical form this yields given an artifact whose form is `artifact_canonical`."""
        return artifact_canonical if self.produces == "same" else self.produces

    def search_space(self) -> list[dict[str, Any]]:
        """Candidate parameter sets to choose between on the validation split.

        One empty dict -- the default -- means this postprocessor has nothing to fit, so the
        winning candidate is trivially the only one and no validation data is consulted.
        """
        return [{}]

    def check_artifact(self, artifact: Any) -> None:
        """Refuse an artifact whose declared layout differs from what this post-processor reads.

        Nothing to check by default. The affinity routes compare the artifact's declared `offsets`
        with the ones they assume, rather than trusting channel order.
        """

    def use_scratch(self, directory: Any) -> None:
        """Where this postprocessor may write large intermediates; the runner passes its --scratch.

        Ignored by default. `mws` streams a block's edges there when they are too many to hold.
        """

    def run_info(self) -> dict[str, Any] | None:
        """How the most recent call computed its result, for the record; None if nothing to say."""
        return None

    def lazy(self, artifact: Any, origin: tuple[int, ...], shape: tuple[int, ...],
             **params: Any) -> Any:
        """This region's labelling without reading it, or None when that is impossible.

        Only a post-processor that leaves the stored values as they are can do this -- `identity` --
        because anything computed from the values has to read them. The runner asks only when every
        metric declares `point_lookups`, and falls back to `__call__` on None.
        """
        return None

    @abc.abstractmethod
    def __call__(self, array: np.ndarray, **params: Any) -> np.ndarray:
        """Prediction -> (*spatial) integer labelling. `params` comes from `search_space()`."""

    def describe(self, params: dict[str, Any]) -> str:
        """How a fitted parameter set should read in a leaderboard row."""
        if not params:
            return type(self).__name__
        inner = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{type(self).__name__}({inner})"
