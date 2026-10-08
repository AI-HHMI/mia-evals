"""`semantic_seg` scores on the label array's own grid, placing the prediction by its OME geometry.

The stores here are built the way lmd holds a CellMap crop: an 8 nm raw at the origin, and a 4 nm
label array a few raw voxels in, its first voxel centred half a label voxel inside a raw voxel's
corner -- so every raw voxel holds exactly 2 x 2 x 2 label voxels. Expected placements are written
out by hand or counted voxel by voxel, never with the code under test.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
import zarr

import components  # noqa: F401  (populates the registries)
from artifact import open_artifact, write_artifact
from tasks.base import Volume
from tasks.registry import TaskRegistry

pytestmark = pytest.mark.unit

RAW = 8.0
LABEL = 4.0
#: The crop starts at raw voxel 5 (centre 40 nm, extent [36, 44)): its first 4 nm voxel is centred
#: at 38 nm, half a label voxel in from 36.
CROP_FIRST_RAW = 5
CROP_TRANSLATION = CROP_FIRST_RAW * RAW - LABEL / 2


def _ome(group, axes: str, scale, translation, channel: bool = False) -> None:
    listed = ([{"name": "c", "type": "channel"}] if channel else []) + [
        {"name": a, "type": "space", "unit": "nanometer"} for a in axes
    ]
    group.attrs.update(ome={"version": "0.5", "multiscales": [{
        "axes": listed,
        "datasets": [{"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": [float(s) for s in scale]},
            {"type": "translation", "translation": [float(t) for t in translation]},
        ]}],
    }]})


def make_store(root: Path, labels: np.ndarray, name: str = "cell") -> Path:
    """An lmd-like store: `raw` (8 nm, at the origin) and `labels/crop` (4 nm, translated)."""
    path = root / f"{name}.zarr"
    store = zarr.open_group(str(path), mode="w", zarr_format=3)
    raw = store.create_group("raw")
    raw.create_array("s0", shape=(16, 16, 16), dtype="u1", chunks=(16, 16, 16))[:] = 0
    _ome(raw, "zyx", (RAW,) * 3, (0.0,) * 3)
    crop = store.create_group("labels").create_group("crop")
    crop.create_array("s0", shape=labels.shape, dtype="u1", chunks=labels.shape)[:] = labels
    _ome(crop, "zyx", (LABEL,) * 3, (CROP_TRANSLATION,) * 3)
    return path


def ome_artifact(path: Path, array: np.ndarray, voxel: float, first: float,
                 kind: str = "class_labels", axes: str = "zyx") -> Path:
    """A single-level OME group like mia-train's predict.py writes, voxel size and first centre."""
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.create_array("s0", shape=array.shape, dtype=array.dtype, chunks=array.shape)[:] = array
    channel = kind == "class_scores"
    _ome(group, axes, ((1.0,) if channel else ()) + (voxel,) * 3,
         ((0.0,) if channel else ()) + (first,) * 3, channel=channel)
    group.attrs.update(kind=kind, run="unit_run", step=7,
                       **({"background_id": 0} if kind == "class_labels" else {}))
    return path


@pytest.fixture
def crop(tmp_path):
    rng = np.random.default_rng(0)
    labels = rng.integers(1, 6, (6, 6, 6)).astype(np.uint8)
    store = make_store(tmp_path, labels)
    box = ((CROP_FIRST_RAW, CROP_FIRST_RAW + 3),) * 3          # 6 label voxels = 3 raw voxels
    volume = Volume(name="cell", path=store, label_key="labels/crop", bounding_box=box)
    return volume, labels


def scored(volume, artifact):
    task = TaskRegistry.build("semantic_seg")
    placed = task.place(artifact)
    origin, shape = task.region(volume, placed)
    window_origin, window_shape = task.read_window(volume, placed)
    block = placed.read(window_origin, window_shape)
    return task, origin, shape, task.align(block, volume, placed), task.context(volume, placed)


def test_a_raw_resolution_prediction_is_upsampled_two_to_one(crop, tmp_path):
    volume, labels = crop
    # Predicted on the raw's own grid over raw voxels [3, 10): the crop plus two voxels of context.
    prediction = np.random.default_rng(1).integers(0, 9, (7, 7, 7)).astype(np.uint8)
    path = ome_artifact(tmp_path / "pred.zarr", prediction, RAW, 3 * RAW)
    task, origin, shape, aligned, context = scored(volume, open_artifact(path))
    assert origin == (0, 0, 0) and shape == (6, 6, 6) and context["whole_region"]
    held = np.array([2, 2, 3, 3, 4, 4])            # label voxel i lies in raw voxel 5 + i // 2
    np.testing.assert_array_equal(aligned, prediction[np.ix_(held, held, held)])


def test_a_prediction_on_the_label_grid_is_taken_voxel_for_voxel(crop, tmp_path):
    volume, labels = crop
    prediction = np.random.default_rng(2).integers(0, 9, (10, 10, 10)).astype(np.uint8)
    path = ome_artifact(tmp_path / "pred.zarr", prediction, LABEL, CROP_TRANSLATION - 2 * LABEL)
    _, origin, shape, aligned, _ = scored(volume, open_artifact(path))
    np.testing.assert_array_equal(aligned, prediction[2:8, 2:8, 2:8])


def test_a_prediction_on_an_unrelated_grid_takes_the_voxel_holding_each_centre(crop, tmp_path):
    volume, _ = crop
    first, voxel = 33.7, 5.0
    prediction = np.random.default_rng(3).integers(0, 9, (8, 8, 8)).astype(np.uint8)
    path = ome_artifact(tmp_path / "pred.zarr", prediction, voxel, first)
    _, origin, shape, aligned, _ = scored(volume, open_artifact(path))
    expected = np.empty(shape, dtype=np.uint8)
    for k in np.ndindex(*shape):
        centre = [CROP_TRANSLATION + (o + i) * LABEL for o, i in zip(origin, k, strict=True)]
        holder = tuple(int(np.floor((c - (first - voxel / 2)) / voxel)) for c in centre)
        expected[k] = prediction[holder]
    np.testing.assert_array_equal(aligned, expected)


def test_a_prediction_covering_part_of_the_crop_scores_that_part_and_says_so(crop, tmp_path):
    volume, _ = crop
    # Raw voxels [6, 10) hold label voxels 2..5 of the crop (centres 46 .. 58 nm).
    path = ome_artifact(tmp_path / "pred.zarr", np.ones((4, 4, 4), np.uint8), RAW, 6 * RAW)
    _, origin, shape, aligned, context = scored(volume, open_artifact(path))
    assert origin == (2, 2, 2) and shape == (4, 4, 4)
    assert context["whole_region"] is False and aligned.shape == (4, 4, 4)


def test_a_bounding_box_inside_the_label_array_clips_the_annotated_region(crop, tmp_path):
    volume, _ = crop
    clipped = Volume(name=volume.name, path=volume.path, label_key=volume.label_key,
                     bounding_box=((5, 7), (5, 8), (5, 8)))
    path = ome_artifact(tmp_path / "pred.zarr", np.ones((7, 7, 7), np.uint8), RAW, 3 * RAW)
    _, origin, shape, _, context = scored(clipped, open_artifact(path))
    assert origin == (0, 0, 0) and shape == (4, 6, 6) and context["whole_region"]


def test_a_prediction_beside_the_crop_is_refused(crop, tmp_path):
    volume, _ = crop
    path = ome_artifact(tmp_path / "pred.zarr", np.ones((3, 3, 3), np.uint8), RAW, 12 * RAW)
    with pytest.raises(ValueError, match="covers no annotated label voxel"):
        scored(volume, open_artifact(path))


def test_an_artifact_without_geometry_is_refused(crop, tmp_path):
    volume, _ = crop
    path = write_artifact(tmp_path / "bare.zarr", np.ones((6, 6, 6), np.uint8), "class_labels",
                          background_id=0)
    with pytest.raises(ValueError, match="declares no OME geometry"):
        scored(volume, open_artifact(path))


def test_transposed_axes_are_refused(crop, tmp_path):
    volume, _ = crop
    path = ome_artifact(tmp_path / "pred.zarr", np.ones((7, 7, 7), np.uint8), RAW, 3 * RAW,
                        axes="xyz")
    with pytest.raises(ValueError, match="permutation"):
        scored(volume, open_artifact(path))


# ------------------------------------------------------------------------------------ end to end


def _write_configs(root: Path, store: Path, route: str, postprocess: str) -> Path:
    data = root / "data.yaml"
    data.write_text(textwrap.dedent(f"""\
        resolutions:
        - - 8.0
          - 8.0
          - 8.0
        patch_size:
        - 2
        - 2
        - 2
        output_axes: lczyx
        volumes:
        - name: cell
          path: {store}
          image_key: raw
          label_key: labels/crop
          zarr_version: zarr3
          bounding_box:
          - - 5
            - 8
          - - 5
            - 8
          - - 5
            - 8
        """))
    config = root / f"{route}.toml"
    config.write_text(textwrap.dedent(f"""\
        task_name = "unit_semantic"
        route = "{route}"

        [data]
        config_path = "{data}"

        [task]
        name = "semantic_seg"

        [postprocess]
        name = "{postprocess}"

        [metric]
        names = ["semantic"]
        rank_by = "semantic"

        [metric.semantic]
        num_classes = 8
        ignore_truth = [0]

        [metric.semantic.classes]
        one = [1]
        two = [2]
        low = [1, 2, 3]
        """))
    return config


def _score(config: Path, artifacts: Path, root: Path) -> dict:
    import evaluate

    args = type("Args", (), {
        "config": config, "test": artifacts, "val": None, "leaderboard": root / "board",
        "run_dir": None, "scored_out": root / f"kept_{config.stem}", "no_scored": False,
        "scratch": root / "scratch",
    })()
    evaluate.cmd_score(args)
    [record] = (root / "board" / "unit_semantic" / "records").glob(f"*.{config.stem}.json")
    return json.loads(record.read_text())


def test_scores_a_raw_grid_prediction_end_to_end_through_either_route(crop, tmp_path):
    volume, labels = crop
    # The truth downsampled to the raw grid by taking each raw voxel's first label voxel, with two
    # raw voxels of context: upsampled back it reproduces exactly those label voxels.
    raw_grid = np.zeros((7, 7, 7), np.uint8)
    raw_grid[2:5, 2:5, 2:5] = labels[::2, ::2, ::2]
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    ome_artifact(artifacts / "cell.zarr", raw_grid, RAW, 3 * RAW)
    upsampled = np.repeat(np.repeat(np.repeat(labels[::2, ::2, ::2], 2, 0), 2, 1), 2, 2)

    record = _score(_write_configs(tmp_path, volume.path, "identity", "identity"), artifacts,
                    tmp_path)
    assert record["schema_version"] == 2
    expected = {
        name: np.count_nonzero(np.isin(labels, ids) & np.isin(upsampled, ids))
        / np.count_nonzero(np.isin(labels, ids) | np.isin(upsampled, ids))
        for name, ids in {"one": [1], "two": [2], "low": [1, 2, 3]}.items()
    }
    mean = np.mean(list(expected.values()))
    assert record["scores"]["semantic"]["mean_iou"] == pytest.approx(mean)
    assert record["region"]["volumes"]["cell"]["shape"] == [6, 6, 6]
    assert record["region"]["volumes"]["cell"]["whole_region"] is True
    assert record["config"]["metric_kwargs"]["semantic"]["classes"]["low"] == [1, 2, 3]
    confusion = record["details"]["semantic"]["confusion"]
    assert sum(map(sum, confusion["counts"])) == labels.size
    kept = open_artifact(record["region"]["volumes"]["cell"]["scored_artifact"])
    assert kept.kind == "class_labels" and kept.geometry.translation == (5 * RAW,) * 3

    # The same prediction as one-hot class scores through argmax scores identically.
    scores = np.zeros((8, 7, 7, 7), np.float32)
    np.put_along_axis(scores, raw_grid[None].astype(np.int64), 1.0, axis=0)
    ome_artifact(artifacts / "cell.zarr", scores, RAW, 3 * RAW, kind="class_scores")
    argmax = _score(_write_configs(tmp_path, volume.path, "argmax", "argmax"), artifacts, tmp_path)
    assert argmax["scores"] == record["scores"]
