"""Mutex watershed, checked against cases whose answers follow from the definition.

Ported from `mia_score_mws.py --self-test`, which was a function inside a script and therefore ran
only when someone remembered to pass the flag. The cases are unchanged; what changed is that they
now run in CI.

Written from the paper rather than checked against a reference implementation, because the
reference (`affogato`) is CMake-only and does not pip-install -- so these cases are the only thing
standing between the implementation and a plausible-looking partition.
"""

from __future__ import annotations

import numpy as np
import pytest

from postprocess.mws import LONG, mutex_watershed, segment

pytestmark = pytest.mark.unit


def test_definitional_cases():
    # 1. Two nodes, one attractive edge -> one cluster.
    out = mutex_watershed(np.array([0]), np.array([1]), np.array([0.9]), np.array([True]), 2)
    assert len(set(out.tolist())) == 1, out

    # 2. The same pair, but a stronger repulsive edge first -> two clusters. This is the whole
    #    point: repulsion seen earlier blocks a later merge.
    out = mutex_watershed(np.array([0, 0]), np.array([1, 1]), np.array([0.9, 0.5]),
                          np.array([False, True]), 2)
    assert len(set(out.tolist())) == 2, out

    # 3. Weaker repulsion, stronger attraction -> merged, because the merge is processed first and
    #    the mutex arrives too late to undo it.
    out = mutex_watershed(np.array([0, 0]), np.array([1, 1]), np.array([0.9, 0.2]),
                          np.array([True, False]), 2)
    assert len(set(out.tolist())) == 1, out

    # 4. Transitivity of the constraint: a-b merge, b-c mutex, then a-c attractive must be blocked.
    out = mutex_watershed(
        np.array([0, 1, 0]), np.array([1, 2, 2]), np.array([0.9, 0.8, 0.7]),
        np.array([True, False, True]), 3)
    assert out[0] == out[1] != out[2], out

    # NOTE both remaining cases use a volume LARGER than the long-range offset. At 8^3 with
    # LONG=10 there are no long-range edges at all, so a test there would pass or fail for
    # reasons that have nothing to do with repulsion.
    side = 3 * LONG

    # 5. Repulsive channels at affinity 1 mean repulsion 0, i.e. no constraints, so every
    #    connected component collapses to one label -- MWS degenerates to connected components
    #    over the attractive graph, as it must.
    rng = np.random.default_rng(0)
    aff = np.zeros((6, side, side, side), dtype=np.float32)
    aff[:3] = rng.random((3, side, side, side))    # arbitrary attractive weights
    aff[3:] = 1.0
    labels = segment(aff, 1)
    assert len(np.unique(labels)) == 1, np.unique(labels)

    # 6. A plane of repulsion across x splits the volume. The short-range x edge at the plane is
    #    cut, and every long-range x pair straddling it repels.
    cut = side // 2
    aff = np.zeros((6, side, side, side), dtype=np.float32)
    aff[:3] = 0.9
    aff[3:] = 1.0
    aff[0, cut] = 0.0                              # attractive x edge across the plane: no pull
    aff[3, max(cut - LONG + 1, 0) : cut + 1] = 0.0  # long-range x pairs straddling it: full push
    labels = segment(aff, 1)
    assert len(np.unique(labels)) >= 2, f"expected a split, got {np.unique(labels)}"
