"""Tests for the two add-ons every instance post-processor takes -- `min_sizes` and
`fill_distances` -- on `identity`, where they are the whole route for a finished labelling.

`drop_small_components` itself is covered in test_cc_threshold.py, including a mutation-verified
slab-boundary case. These cover identity's use of it, which replaced the old `size_filter`
post-processor with the same candidates, the same params and the same labellings, and the fill
that follows it: a filter is swept over a labelling that already exists rather than recomputed
with it, because mutex watershed on a 7-gigavoxel volume takes eleven hours.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import components  # noqa: F401,E402  (populates the registries)
from postprocess.labellings import Identity  # noqa: E402
from postprocess.registry import PostprocessRegistry  # noqa: E402
from postprocess.size_filter import (  # noqa: E402
    FillBases,
    drop_small_components,
    fill_holes,
    nearest_segment,
)

pytestmark = pytest.mark.unit


def test_size_filter_is_no_longer_a_post_processor_of_its_own():
    assert "size_filter" not in PostprocessRegistry.available()
    assert isinstance(PostprocessRegistry.build("identity", min_sizes=[0, 500]), Identity)


def test_plain_identity_is_unchanged(tmp_path):
    """Nothing to fit, empty params, both labelling kinds, the unread path for huge labellings."""
    from artifact import open_artifact, write_artifact

    plain = Identity()
    assert plain.search_space() == [{}]
    assert plain.accepts == ("instances", "class_labels")
    assert plain.describe({}) == "Identity"
    labels = np.array([[[0, 1, 2, 2, 5]]], dtype=np.uint32)
    assert plain(labels) is labels
    path = write_artifact(tmp_path / "a.zarr", labels, "instances", background_id=0)
    artifact = open_artifact(path)
    assert plain.lazy(artifact, (0, 0, 0), labels.shape) is not None


def test_an_add_on_makes_identity_take_instances_only():
    """A size filter on class labels is meaningless: a class region is not a component."""
    assert Identity(min_sizes=[0, 5]).accepts == ("instances",)
    assert Identity(fill_distances=[0, 1]).accepts == ("instances",)


def test_search_space_is_the_sorted_deduplicated_sweep():
    assert PostprocessRegistry.build(
        "identity", min_sizes=[50000, 0, 500, 500]
    ).search_space() == [{"min_size": 0}, {"min_size": 500}, {"min_size": 50000}]


def test_fill_distances_are_swept_inside_each_size():
    assert Identity(min_sizes=[0, 5], fill_distances=["all", 2, 0]).search_space() == [
        {"min_size": 0, "fill_distance": 0}, {"min_size": 0, "fill_distance": 2},
        {"min_size": 0, "fill_distance": "all"}, {"min_size": 5, "fill_distance": 0},
        {"min_size": 5, "fill_distance": 2}, {"min_size": 5, "fill_distance": "all"},
    ]


def test_describe_names_what_was_applied():
    processor = Identity(min_sizes=[0, 5000], fill_distances=[0, 3])
    assert processor.describe({"min_size": 0}) == "identity(no filter)"
    assert processor.describe({"min_size": 5000}) == "identity(min_size=5000)"
    assert processor.describe({"min_size": 5000, "fill_distance": 3}) == \
        "identity(min_size=5000, fill_distance=3)"
    assert processor.describe({"min_size": 0, "fill_distance": "all"}) == \
        "identity(fill_distance=all)"


def test_it_drops_small_components_and_keeps_large_ones():
    labels = np.zeros((1, 1, 10), dtype=np.uint32)
    labels[0, 0, 0:1] = 1        # 1 voxel
    labels[0, 0, 1:4] = 2        # 3 voxels
    labels[0, 0, 4:10] = 3       # 6 voxels
    out = Identity(min_sizes=[3])(labels.copy(), min_size=3)
    assert set(np.unique(out)) == {0, 2, 3}
    assert (out == 2).sum() == 3 and (out == 3).sum() == 6


def test_min_size_zero_passes_the_labelling_through_unchanged():
    labels = np.array([[[0, 1, 2, 2, 5]]], dtype=np.uint32)
    assert np.array_equal(Identity(min_sizes=[0, 3])(labels.copy(), min_size=0), labels)


def test_a_filtered_labelling_cannot_be_scored_unread():
    processor = Identity(min_sizes=[0, 3])
    assert processor.lazy(None, (0,), (1,), min_size=3) is None
    assert processor.lazy(None, (0,), (1,), min_size=0, fill_distance=1) is None


def test_the_fill_follows_the_filter():
    labels = np.zeros((1, 3, 12), dtype=np.uint32)
    labels[0, :, 0:4] = 1
    labels[0, :, 5:7] = 2
    labels[0, 1, 9] = 3                                            # a 1-voxel speck
    processor = Identity(min_sizes=[0, 2], fill_distances=[0, 1, "all"])
    for candidate in processor.search_space():
        expected = fill_holes(drop_small_components(labels.copy(), candidate["min_size"]),
                              {0: 0, 1: 1, "all": np.inf}[candidate["fill_distance"]])
        assert np.array_equal(processor(labels.copy(), **candidate), expected), candidate


def test_a_float_array_is_refused_rather_than_silently_labelled():
    with pytest.raises(ValueError, match="floating-point"):
        Identity()(np.zeros((2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="floating-point"):
        Identity(fill_distances=[1])(np.zeros((2, 2, 2), dtype=np.float32), fill_distance=1)


@pytest.mark.parametrize("settings, match", [
    ({"min_sizes": []}, "min_sizes"), ({"min_sizes": [-1, 5]}, "min_sizes"),
    ({"fill_distances": []}, "fill_distances"), ({"fill_distances": [-1]}, "fill_distances"),
    ({"fill_distances": [1.5]}, "fill_distances"), ({"fill_distances": [True]}, "fill_distances"),
    ({"fill_distances": ["most"]}, "fill_distances"),
])
def test_rejects_an_unusable_setting(settings, match):
    with pytest.raises(ValueError, match=match):
        Identity(**settings)


# ---------------------------------------------------------------- one basis per size, per volume


def test_the_fill_bases_serve_every_distance_of_a_size_and_every_volume_of_a_fit():
    """The runner scores a candidate on each fit volume before the next candidate: with both
    volumes' bases kept, each is built once per size rather than once per candidate."""
    rng = np.random.default_rng(1)
    volumes = [rng.integers(0, 4, (5, 6, 7)) * (rng.random((5, 6, 7)) < 0.4) for _ in range(2)]
    bases, built = FillBases(), []

    def filtered(labels):
        built.append(1)
        return labels.copy()

    for fill in (1, 2, "all"):
        for index, labels in enumerate(volumes):
            out = bases.fill(("volume", index), labels.size, lambda v=labels: filtered(v), fill)
            limit = np.inf if fill == "all" else fill
            assert np.array_equal(out, fill_holes(labels, limit))
    assert len(built) == 2


def test_a_basis_too_large_to_keep_two_of_frees_the_old_one_first():
    labels = np.zeros((4, 4, 4), dtype=np.int64)
    labels[0] = 1
    bases = FillBases(budget=1)                                    # nothing fits but the latest
    bases.fill("a", labels.size, lambda: labels.copy(), 1)
    bases.fill("b", labels.size, lambda: labels.copy(), 1)
    assert list(bases._entries) == ["b"]


def test_a_filled_result_is_never_the_cached_labelling():
    labels = np.ones((2, 2, 2), dtype=np.int64)                    # nothing to fill
    bases = FillBases()
    first = bases.fill("k", labels.size, lambda: labels.copy(), 1)
    first[...] = 7
    assert (bases.fill("k", labels.size, lambda: labels.copy(), 1) == 1).all()


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
