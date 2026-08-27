"""Tests for the size filter, and for the sweep it adds.

The filter's job is narrow -- delete whole components below a voxel count -- and the tests are
aimed at the ways an implementation can look right while being wrong at scale: slab boundaries,
background, and dtype.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from postprocess.cc_threshold import (  # noqa: E402
    ConnectedComponentThreshold,
    drop_small_components,
)


def test_min_size_zero_is_the_identity():
    """The default must reproduce records written before the filter existed."""
    labels = np.array([[[0, 1, 2, 2]]], dtype=np.uint32)
    assert np.array_equal(drop_small_components(labels.copy(), 0), labels)
    assert np.array_equal(drop_small_components(labels.copy(), -5), labels)


def test_drops_only_components_below_the_threshold():
    labels = np.zeros((1, 1, 10), dtype=np.uint32)
    labels[0, 0, 0:1] = 1        # 1 voxel
    labels[0, 0, 1:4] = 2        # 3 voxels
    labels[0, 0, 4:10] = 3       # 6 voxels
    out = drop_small_components(labels.copy(), 3)
    assert set(np.unique(out)) == {0, 2, 3}, "the 1-voxel component should be gone, 3 and 6 kept"
    assert (out == 2).sum() == 3 and (out == 3).sum() == 6, "kept components must be untouched"


def test_background_survives_being_the_smallest_thing_present():
    """Background stays background even when its voxel count is below the threshold.

    Kept as a guarantee, not as a discriminating test: id 0 maps to 0 whether the filter considers
    it droppable or not, so this passes under either implementation. Noted explicitly because an
    earlier version of the filter carried a special-case guard here that a mutation test proved
    was unreachable.
    """
    labels = np.full((1, 1, 9), 7, dtype=np.uint32)
    labels[0, 0, 4] = 0                       # a single background voxel among 8 foreground
    out = drop_small_components(labels.copy(), 5)
    assert out[0, 0, 4] == 0
    assert (out == 7).sum() == 8, "the large component must survive"


def test_a_component_spanning_slab_boundaries_is_counted_whole():
    """The blowup fix counts in slabs; a component split across them must not be under-counted.

    Built so the trap bites: the component has 40 voxels spread one-per-slice over 40 slices, so a
    per-slab implementation that forgot to accumulate would see 1 voxel per slab and delete it.
    A shape[0] of 64 guarantees more than one slab at `shape[0] // 32`.
    """
    labels = np.zeros((64, 4, 4), dtype=np.uint32)
    labels[:40, 0, 0] = 1                     # 40 voxels, one on each of 40 different slices
    labels[0, 1, 1] = 2                       # 1 voxel, genuinely small
    out = drop_small_components(labels.copy(), 10)
    assert (out == 1).sum() == 40, "component 1 spans slabs and is over threshold; must survive"
    assert (out == 2).sum() == 0, "component 2 is 1 voxel; must be dropped"


def test_preserves_dtype_so_the_metrics_do_not_pay_for_a_widening_cast():
    labels = np.array([[[1, 1, 2]]], dtype=np.uint32)
    assert drop_small_components(labels.copy(), 2).dtype == np.uint32


def test_ids_are_not_renumbered():
    """Surviving ids keep their values, so a record stays traceable back to the labelling."""
    labels = np.zeros((1, 1, 8), dtype=np.uint32)
    labels[0, 0, 0:4] = 5
    labels[0, 0, 4:5] = 6
    labels[0, 0, 5:8] = 9
    out = drop_small_components(labels.copy(), 3)
    assert set(np.unique(out)) == {0, 5, 9}


def test_search_space_is_the_cross_product_logit_major():
    processor = ConnectedComponentThreshold(logits=[0, 3], min_sizes=[500, 0])
    assert processor.search_space() == [
        {"logit": 0.0, "min_size": 0},
        {"logit": 0.0, "min_size": 500},
        {"logit": 3.0, "min_size": 0},
        {"logit": 3.0, "min_size": 500},
    ], "min_sizes are sorted, and logit varies slowest"


def test_describe_names_min_size_when_it_is_active():
    processor = ConnectedComponentThreshold(logits=[0], min_sizes=[0, 500])
    assert "min_size" not in processor.describe({"logit": 0.0, "min_size": 0})
    assert "min_size=500" in processor.describe({"logit": 0.0, "min_size": 500})


def test_default_sweep_applies_no_size_filter():
    """Adding the parameter must not silently change what an existing config does."""
    assert ConnectedComponentThreshold(logits=[0]).search_space() == [{"logit": 0.0, "min_size": 0}]


@pytest.mark.parametrize("bad", [[], [-1, 5]])
def test_rejects_an_unusable_min_sizes(bad):
    with pytest.raises(ValueError, match="min_sizes"):
        ConnectedComponentThreshold(logits=[0], min_sizes=bad)
