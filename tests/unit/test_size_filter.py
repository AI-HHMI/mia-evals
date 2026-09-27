"""Tests for the `size_filter` postprocessor.

`drop_small_components` itself is covered in test_cc_threshold.py, including a mutation-verified
slab-boundary case. These cover the registry entry wrapped around it, which exists so a filter can
be swept over a labelling that already exists rather than recomputed with it: mutex watershed on a
7-gigavoxel volume takes eleven hours, so sweeping a filter by re-running it is not an option.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import components  # noqa: F401,E402  (populates the registries)
from postprocess.registry import PostprocessRegistry  # noqa: E402
from postprocess.size_filter import SizeFilter  # noqa: E402

pytestmark = pytest.mark.unit


def test_it_is_registered_so_a_config_can_name_it():
    assert "size_filter" in PostprocessRegistry._registry
    built = PostprocessRegistry.build("size_filter", min_sizes=[0, 500])
    assert isinstance(built, SizeFilter)


def test_it_accepts_instances_and_not_class_labels():
    """A size filter on class labels is meaningless: a class region is not a component."""
    assert SizeFilter.accepts == ("instances",)
    assert SizeFilter.produces == "instances"


def test_search_space_is_the_sorted_deduplicated_sweep():
    assert PostprocessRegistry.build(
        "size_filter", min_sizes=[50000, 0, 500, 500]
    ).search_space() == [{"min_size": 0}, {"min_size": 500}, {"min_size": 50000}]


def test_default_is_no_filter_at_all():
    assert SizeFilter().search_space() == [{"min_size": 0}]


def test_describe_distinguishes_no_filter_from_a_filter():
    processor = SizeFilter(min_sizes=[0, 5000])
    assert "no filter" in processor.describe({"min_size": 0})
    assert "min_size=5000" in processor.describe({"min_size": 5000})


def test_it_drops_small_components_and_keeps_large_ones():
    labels = np.zeros((1, 1, 10), dtype=np.uint32)
    labels[0, 0, 0:1] = 1        # 1 voxel
    labels[0, 0, 1:4] = 2        # 3 voxels
    labels[0, 0, 4:10] = 3       # 6 voxels
    out = SizeFilter(min_sizes=[3])(labels.copy(), min_size=3)
    assert set(np.unique(out)) == {0, 2, 3}
    assert (out == 2).sum() == 3 and (out == 3).sum() == 6


def test_min_size_zero_passes_the_labelling_through_unchanged():
    labels = np.array([[[0, 1, 2, 2, 5]]], dtype=np.uint32)
    assert np.array_equal(SizeFilter()(labels.copy(), min_size=0), labels)


def test_a_float_array_is_refused_rather_than_silently_labelled():
    """The same guard `identity` has: a labelling must be integral or the ids are not ids."""
    with pytest.raises(ValueError, match="floating-point"):
        SizeFilter()(np.zeros((2, 2, 2), dtype=np.float32), min_size=0)


@pytest.mark.parametrize("bad", [[], [-1, 5]])
def test_rejects_an_unusable_min_sizes(bad):
    with pytest.raises(ValueError, match="min_sizes"):
        SizeFilter(min_sizes=bad)


# --- filling the holes the filter leaves ---------------------------------------------------------

from postprocess.size_filter import fill_holes, nearest_segment  # noqa: E402


def _row(values: list[int]) -> np.ndarray:
    """One X row as a (X, 1, 1) labelling, so distances are plain index differences."""
    return np.array(values, dtype=np.int64).reshape(-1, 1, 1)


def test_fill_closes_a_gap_from_both_sides():
    """Each hole voxel takes its nearest segment: the gap is shared, not taken by one side."""
    filled = fill_holes(_row([1, 1, 0, 0, 2, 2]), 1)
    assert filled.ravel().tolist() == [1, 1, 1, 2, 2, 2]


def test_fill_stops_at_its_distance_and_inf_fills_everything():
    labels = _row([1, 0, 0, 0, 0, 2])
    assert fill_holes(labels, 1).ravel().tolist() == [1, 1, 0, 0, 2, 2]
    assert fill_holes(labels, float("inf")).ravel().tolist() == [1, 1, 1, 2, 2, 2]
    assert labels.ravel().tolist() == [1, 0, 0, 0, 0, 2], "the input must not be modified"


def test_fill_with_nothing_to_do_returns_its_input():
    labels = _row([1, 0, 2])
    assert fill_holes(labels, 0) is labels
    empty = _row([0, 0, 0])
    assert fill_holes(empty, float("inf")) is empty


def test_a_precomputed_transform_gives_the_same_fill():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, (6, 7, 5)) * (rng.random((6, 7, 5)) < 0.3)
    nearest = nearest_segment(labels)
    for distance in (1, 2, float("inf")):
        assert np.array_equal(fill_holes(labels, distance, nearest), fill_holes(labels, distance))
