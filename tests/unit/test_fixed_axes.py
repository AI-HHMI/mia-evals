"""Volumes pinned with miao's `fixed_axes`: one frame of a time series, scored as a 3D volume.

The frame has to reach four places: the data config's entry, the label read, the check that a
prediction belongs to the volume it is scored as, and the task identity. The first two go through a
miao that knows the setting, and skip without one.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import components  # noqa: F401,E402  (populates the registries)
from artifact import open_artifact, write_artifact  # noqa: E402
from config import _volume_record, load_data_config  # noqa: E402
from evaluate import check_same_volume  # noqa: E402
from report.record import identity_differences, task_identity  # noqa: E402
from tasks.base import Volume  # noqa: E402
from tasks.segmentation import read_labels  # noqa: E402


def needs_fixed_axes() -> None:
    miao_config = pytest.importorskip("miao.config")
    if "fixed_axes" not in miao_config.VolumeConfig.model_fields:
        pytest.skip("the installed miao predates fixed_axes")


def timeseries(path: Path, frames: int = 4, channels: int = 1) -> Path:
    """A t, c, z, y, x OME-Zarr label group, `labels/seg`; frame t holds the value t + 1."""
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    group = root.create_group("labels").create_group("seg")
    axes = [{"name": "t", "type": "time"}, {"name": "c", "type": "channel"}] + [
        {"name": axis, "type": "space", "unit": "nanometer"} for axis in "zyx"
    ]
    group.attrs["ome"] = {"version": "0.5", "multiscales": [{"axes": axes, "datasets": [
        {"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 2, 1, 1]}]},
    ]}]}
    values = np.arange(1, frames + 1, dtype=np.uint32)[:, None, None, None, None]
    group.create_array("s0", data=np.broadcast_to(values, (frames, channels, 8, 8, 8)).copy(),
                       chunks=(1, 1, 8, 8, 8))
    return path


def data_config(tmp_path: Path, fixed_axes: str) -> Path:
    path = tmp_path / "data.yaml"
    path.write_text(textwrap.dedent(f"""\
        resolutions: [[2, 1, 1]]
        patch_size: [4, 4, 4]
        output_axes: lczyx
        volumes:
        - name: frame
          path: {tmp_path / "ts.zarr"}
          image_key: raw
          label_key: labels/seg
          zarr_version: zarr3
          fixed_axes: {fixed_axes}
          bounding_box: [[0, 8], [0, 8], [0, 8]]
        """))
    return path


def test_a_data_config_entry_carries_its_frame(tmp_path):
    """miao's own forms are accepted and normalised: a one-index range is that index."""
    needs_fixed_axes()
    (volume,) = load_data_config(data_config(tmp_path, '{t: "2:3"}'))
    assert volume.fixed_axes == {"t": 2}
    assert "fixed_axes" not in volume.extra


def test_an_entry_pinning_several_frames_is_refused(tmp_path):
    """miao would expand it into one volume per frame; a scored volume is one labelling."""
    needs_fixed_axes()
    with pytest.raises(ValueError, match="give each its own entry"):
        load_data_config(data_config(tmp_path, "{t: [1, 2]}"))


def test_labels_are_read_at_the_pinned_frame(tmp_path):
    """The spatial window lands on z, y, x of frame 2, with the single channel taken."""
    needs_fixed_axes()
    volume = Volume(name="frame", path=timeseries(tmp_path / "ts.zarr"), label_key="labels/seg",
                    fixed_axes={"t": 2})
    block = read_labels(volume, origin=(1, 2, 3), shape=(4, 4, 4))
    assert block.shape == (4, 4, 4)
    assert np.all(block == 3)


def test_several_label_channels_are_refused(tmp_path):
    needs_fixed_axes()
    volume = Volume(name="frame", path=timeseries(tmp_path / "ts.zarr", channels=2),
                    label_key="labels/seg", fixed_axes={"t": 2})
    with pytest.raises(ValueError, match="2 label channels"):
        read_labels(volume, origin=(0, 0, 0), shape=(4, 4, 4))


@pytest.mark.parametrize(("declared", "pinned", "refused"), [
    ({"t": 2}, {"t": 2}, False),
    ({"t": 25}, {"t": 2}, True),     # another frame's prediction, filed under this volume's name
    (None, {"t": 2}, True),           # a prediction that read no particular frame
    ({"t": 2}, None, True),
    (None, None, False),              # an ordinary volume, as before
])
def test_a_prediction_must_come_from_the_frame_it_is_scored_as(tmp_path, declared, pinned, refused):
    store = tmp_path / "ts.zarr"
    attrs = {"source_path": str(store), **({"source_fixed_axes": declared} if declared else {})}
    path = write_artifact(tmp_path / "pred.zarr", np.zeros((4, 4, 4), dtype=np.uint32),
                          "instances", origin=(0, 0, 0), background_id=0, **attrs)
    volume = Volume(name="frame", path=store, label_key="labels/seg", fixed_axes=pinned)
    if refused:
        with pytest.raises(SystemExit, match="fixed_axes"):
            check_same_volume(volume, open_artifact(path), "test")
    else:
        check_same_volume(volume, open_artifact(path), "test")


def test_the_frame_is_part_of_what_the_task_is():
    """Two frames of one store are two tasks; a record from before fixed_axes is an unpinned one."""
    base = {"name": "frame", "path": "/data/ts.zarr", "label_key": "labels/seg",
            "bounding_box": None}
    before = task_identity([base], "voxel_instance", "pq")
    t10 = task_identity([{**base, "fixed_axes": {"t": 10}}], "voxel_instance", "pq")
    t25 = task_identity([{**base, "fixed_axes": {"t": 25}}], "voxel_instance", "pq")
    assert task_identity([{**base, "fixed_axes": None}], "voxel_instance", "pq") == before
    assert t10 != before
    assert identity_differences(t10, t25) == ["frame: fixed_axes {'t': 25} vs {'t': 10}"]


def test_a_record_names_the_frame_only_when_there_is_one():
    assert "fixed_axes" not in _volume_record(Volume(name="v", path=Path("/x.zarr")))
    pinned = Volume(name="v", path=Path("/x.zarr"), fixed_axes={"t": 10})
    assert _volume_record(pinned)["fixed_axes"] == {"t": 10}
