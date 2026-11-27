"""Tests for what a task tells a metric about the region it is scoring.

Concentrated on `whole_region`, because it is a claim about provenance rather than a number: it
decides whether a skeleton is cropped, and it decides which leaderboard rows may be ranked against
one another. Getting it wrong does not look like an error, it looks like a comparable result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import components  # noqa: F401,E402  (populates the registries)
from artifact import write_artifact  # noqa: E402
from tasks.base import Volume  # noqa: E402
from tasks.registry import TaskRegistry  # noqa: E402
from tasks.segmentation import read_labels  # noqa: E402

pytestmark = pytest.mark.unit

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


def test_read_labels_ignores_unknown_keys_in_the_stores_root_metadata(tmp_path):
    store = tmp_path / "vol.zarr"
    labels = np.arange(64, dtype=np.uint64).reshape(4, 4, 4)
    zarr.open_group(str(store), mode="w", zarr_format=3)
    zarr.create_array(str(store / "labels/inst/s0"), data=labels, zarr_format=3)
    root = json.loads((store / "zarr.json").read_text())
    root["_source"] = {"dataset": "hemibrain"}  # the hemibrain crops' root zarr.json has this
    (store / "zarr.json").write_text(json.dumps(root))
    volume = Volume(name="v", path=store, label_key="labels/inst")
    assert np.array_equal(read_labels(volume, (1, 0, 0), (2, 4, 4)), labels[1:3])


def _store_naming(tmp_path: Path, axes: str, shape: tuple[int, ...]) -> Path:
    from conftest import name_axes

    store = tmp_path / "vol.zarr"
    name_axes(store, axes, shape)
    return store


def test_a_box_is_converted_to_the_stores_axis_order(tmp_path):
    """A box is written in the data config's `output_axes` order; a store may save another.

    Applied as written, a z, y, x box clipped a whole-frame labelling of a store saved x, y, z
    (mia-evals#5): here 12 x 6 x 4 would become 4 x 6 x 4.
    """
    from artifact import open_artifact
    from tasks.base import store_box

    store = _store_naming(tmp_path, "xyz", (12, 6, 4))
    volume = Volume(name="tile", path=store, bounding_box=((0, 4), (0, 6), (0, 12)), box_axes="zyx")
    assert store_box(volume) == ((0, 12), (0, 6), (0, 4))

    path = write_artifact(tmp_path / "pred.zarr", np.zeros((12, 6, 4), dtype=np.uint32),
                          "instances", background_id=0)
    task = TaskRegistry.build("instance_seg", truth_kind="instances")
    assert task.region(volume, open_artifact(path)) == ((0, 0, 0), (12, 6, 4))
    assert task.context(volume, open_artifact(path))["whole_region"] is True


def test_a_pinned_frame_leaves_the_stores_spatial_axes(tmp_path):
    """A time series stored t, c, x, y, z with one frame pinned is a volume stored x, y, z."""
    from tasks.base import store_axes, store_box

    store = _store_naming(tmp_path, "tcxyz", (3, 1, 12, 6, 4))
    volume = Volume(name="frame", path=store, fixed_axes={"t": 1},
                    bounding_box=((0, 4), (0, 6), (0, 12)), box_axes="zyx")
    assert store_axes(volume) == "xyz"
    assert store_box(volume) == ((0, 12), (0, 6), (0, 4))


def test_a_box_for_axes_the_store_does_not_have_is_refused(tmp_path):
    from tasks.base import store_box

    store = _store_naming(tmp_path, "zyx", (4, 4, 4))
    volume = Volume(name="v", path=store, bounding_box=((0, 4), (0, 4)), box_axes="yx")
    with pytest.raises(ValueError, match="must name the same axes"):
        store_box(volume)


def test_a_store_that_names_no_axes_is_refused_when_its_volume_declares_a_box(tmp_path):
    """Without the store's axis order, a box written in the data config's order cannot be placed."""
    from tasks.base import store_box

    store = tmp_path / "bare.zarr"
    zarr.open_group(str(store), mode="w", zarr_format=3)
    box = ((0, 4), (0, 4), (0, 4))
    with pytest.raises(ValueError, match="could not be read"):
        store_box(Volume(name="v", path=store, bounding_box=box, box_axes="zyx"))
    # No box: nothing to place. A volume built in code states its box in the store's order.
    assert store_box(Volume(name="v", path=store, box_axes="zyx")) is None
    assert store_box(Volume(name="v", path=store, bounding_box=box)) == box
