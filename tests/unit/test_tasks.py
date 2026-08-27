"""Tests for what a task tells a metric about the region it is scoring.

Concentrated on `whole_region`, because it is a claim about provenance rather than a number: it
decides whether a skeleton is cropped, and it decides which leaderboard rows may be ranked against
one another. Getting it wrong does not look like an error, it looks like a comparable result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import components  # noqa: F401,E402  (populates the registries)
from artifact import write_artifact  # noqa: E402
from tasks.base import Volume  # noqa: E402
from tasks.registry import TaskRegistry  # noqa: E402

CUBE = "/groups/miaai/miaai/lmd-v0.0.1/dev/nisb/base/val/seed100.zarr"


def make_artifact(tmp_path: Path, origin=(1024, 1024, 384), shape=(64, 64, 64), **attrs):
    return write_artifact(
        tmp_path / "pred.zarr",
        np.zeros((6, *shape), dtype=np.float16),
        "affinity",
        origin=origin,
        convention="sigmoid(0.2 * logit)",
        **attrs,
    )


def context_for(tmp_path, volume, **attrs):
    from artifact import open_artifact

    path = make_artifact(tmp_path, **attrs)
    task = TaskRegistry.build("instance_seg", truth_kind="skeleton")
    return task.context(volume, open_artifact(path))


def test_no_bounding_box_does_not_mean_whole_region(tmp_path):
    """The regression. A fully-annotated volume declares no box; the artifact still covers 64^3.

    Previously `bounding_box is None` short-circuited to whole_region=True, which made
    `skeleton_erl` pass an uncropped skeleton in absolute coordinates and raise
    `IndexError: index 865 is out of bounds for axis 2 with size 512`.
    """
    volume = Volume(name="nisb_base_val_seed100", path=Path(CUBE), bounding_box=None)
    assert context_for(tmp_path, volume)["whole_region"] is False


def test_producer_may_declare_full_coverage(tmp_path):
    """`covers_full_box` is the evidence, because only the producer knows the source extent."""
    volume = Volume(name="v", path=Path(CUBE), bounding_box=None)
    assert context_for(tmp_path, volume, covers_full_box=True)["whole_region"] is True
    assert context_for(tmp_path, volume, covers_full_box=False)["whole_region"] is False


def test_a_declaration_outranks_a_matching_bounding_box(tmp_path):
    """An explicit False is honoured even when the extents happen to line up.

    The producer knows about resampling and clipping that the box comparison cannot see, so its
    statement wins rather than being second-guessed.
    """
    volume = Volume(
        name="v", path=Path(CUBE),
        bounding_box=((1024, 1088), (1024, 1088), (384, 448)),
    )
    ctx = context_for(tmp_path, volume, covers_full_box=False)
    assert ctx["origin"] == (1024, 1024, 384) and ctx["shape"] == (64, 64, 64)
    assert ctx["whole_region"] is False


@pytest.mark.parametrize(
    "box, expected",
    [
        (((1024, 1088), (1024, 1088), (384, 448)), True),    # exactly the artifact's extent
        (((0, 3000), (0, 3000), (0, 1350)), False),          # artifact is a sub-box of it
    ],
)
def test_bounding_box_is_compared_when_nothing_is_declared(tmp_path, box, expected):
    volume = Volume(name="v", path=Path(CUBE), bounding_box=box)
    assert context_for(tmp_path, volume)["whole_region"] is expected
