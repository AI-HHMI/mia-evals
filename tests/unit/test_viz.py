"""The figure commands read artifacts only; here they are run end to end on tiny fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from artifact import write_artifact

pytestmark = pytest.mark.unit


def _nisb_like_cube(root, shape=(20, 20, 20)):
    """A cube in NISB's own layout: `raw/s0` as (c, x, y, z) uint8, labels as (x, y, z) uint16."""
    path = root / "cube.zarr"
    group = zarr.open(str(path), mode="w")
    raw = group.create_array("raw/s0", shape=(1, *shape), dtype="u1", chunks=(1, *shape))
    raw[:] = np.random.default_rng(0).integers(0, 255, size=(1, *shape), dtype=np.uint8)
    labels = np.zeros(shape, dtype=np.uint16)
    labels[:10] = 1
    labels[10:] = 2
    seg = group.create_array("labels/public_gt-cell-nisb/s0", shape=shape, dtype="u2", chunks=shape)
    seg[:] = labels
    return path


def test_viz_affinities_draws_a_six_channel_block_artifact(tmp_path, monkeypatch):
    """`predict.py` writes six channels at the block's own origin; the figure wants the first three.

    It used to read every channel and index them with a three-channel ground truth, which fails
    only once a six-channel artifact is passed -- i.e. on every artifact the current producer
    writes. The origin offset is exercised too: the region shown is inside a block that does not
    start at the cube's corner.
    """
    from viz import affinities

    cube = _nisb_like_cube(tmp_path)
    block = np.random.default_rng(1).random((6, 12, 12, 12), dtype=np.float32).astype(np.float16)
    artifact = write_artifact(
        tmp_path / "aff.zarr", block, "affinity", origin=(4, 4, 4), run="unit_run", step=7,
    )
    monkeypatch.setattr("sys.argv", [
        "mia-evals-viz-affinities", "--affinities", str(artifact), "--cube", str(cube),
        "--origin", "6", "6", "6", "--size", "8", "--slices", "2",
    ])
    affinities.main()
    figures = list(tmp_path.glob("affinities_unit_run_step7_8_6-6-6.png"))
    assert len(figures) == 1
