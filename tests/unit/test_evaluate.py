"""End to end: a task config, an artifact, a record, a leaderboard.

Built from synthetic volumes so it needs no cluster storage, no checkpoint, and neither numba nor
funlib. What it exercises is the wiring the design rests on: that a config resolves through the
registries, that an incompatible pairing is refused rather than attempted, that a sweep cannot be
fitted on the reported split, and that the rendered table is reproducible from the records.
"""

from __future__ import annotations

import json
import textwrap

import numpy as np
import pytest
import zarr

from artifact import write_artifact

pytestmark = pytest.mark.unit


def _volume(root, name: str, labels: np.ndarray) -> str:
    """A minimal OME-ish store: just the label pyramid rung a score reads."""
    path = root / f"{name}.zarr"
    group = zarr.open(str(path), mode="w")
    array = group.create_array(
        "labels/gt/s0", shape=labels.shape, dtype="i8", chunks=labels.shape
    )
    array[:] = labels
    return str(path)


def _data_config(root, entries: list[tuple[str, str, tuple[int, ...]]]) -> str:
    """A miao YAML naming those volumes, with a bounding box each."""
    volumes = "\n".join(
        textwrap.dedent(f"""\
            - name: {name}
              path: {path}
              image_key: raw
              label_key: labels/gt
              zarr_version: zarr3
              bounding_box:
              - - 0
                - {shape[0]}
              - - 0
                - {shape[1]}
              - - 0
                - {shape[2]}
        """)
        for name, path, shape in entries
    )
    path = root / "data.yaml"
    path.write_text(
        "resolutions:\n- - 8.0\n  - 8.0\n  - 8.0\npatch_size:\n- 4\n- 4\n- 4\n"
        "output_axes: lcxyz\nvolumes:\n" + volumes
    )
    return str(path)


def _task_config(root, data_path: str, body: str) -> str:
    path = root / "task.toml"
    path.write_text(f'task_name = "unit_task"\n\n[data]\nconfig_path = "{data_path}"\n\n{body}')
    return str(path)


def _truth_and_split(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    truth = np.zeros((4, 4, 4), dtype=np.int64)
    truth[0:2] = 1
    truth[3] = 2
    prediction = truth.copy()
    prediction[1] = 7 + seed                              # object 1 broken in two
    return truth, prediction


@pytest.fixture
def instances(tmp_path):
    """One volume, and a prediction that splits one of its two objects."""
    truth, prediction = _truth_and_split()
    volume = _volume(tmp_path, "cube", truth)
    data = _data_config(tmp_path, [("cube", volume, truth.shape)])
    artifact = write_artifact(
        tmp_path / "pred.zarr", prediction, "instances", background_id=0, run="unit_run", step=42
    )
    return tmp_path, data, artifact


def test_scores_an_instance_submission_and_writes_a_record(instances, monkeypatch):
    tmp_path, data, artifact = instances
    config = _task_config(tmp_path, data, textwrap.dedent("""\
        [task]
        name = "instance_seg"
        truth_kind = "labels"

        [postprocess]
        name = "identity"

        [metric]
        names = ["voxel_instance"]
        rank_by = "voxel_instance"
        """))

    import evaluate

    records = tmp_path / "records"
    args = type("Args", (), {
        "config": config, "test": artifact, "val": None, "record": records,
        "run_dir": None, "label": "", "scratch": tmp_path / "scratch",
    })()
    evaluate.cmd_score(args)

    written = list(records.rglob("*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text())
    assert payload["task_name"] == "unit_task"
    assert payload["ranking"]["metric"] == "voxel_instance"
    assert payload["ranking"]["higher_is_better"] is True
    # The split is real, so PQ must be below 1 -- a record that reported a perfect score here
    # would mean the prediction never reached the metric.
    assert 0.0 < payload["ranking"]["value"] < 1.0
    assert payload["region"]["volumes"]["cube"]["shape"] == [4, 4, 4]
    assert payload["postprocess"]["fitted_on"] is None


def test_a_sweep_cannot_be_fitted_on_the_reported_split(instances, tmp_path):
    """The rule the whole runner exists to enforce."""
    _, data, _ = instances
    affinities = write_artifact(
        tmp_path / "aff.zarr", np.full((6, 4, 4, 4), 0.9, np.float16), "affinity"
    )
    config = _task_config(tmp_path, data, textwrap.dedent("""\
        [task]
        name = "instance_seg"
        truth_kind = "labels"

        [postprocess]
        name = "cc_threshold"
        logits = [3, 5, 7]

        [metric]
        names = ["voxel_instance"]
        """))

    import evaluate

    args = type("Args", (), {
        "config": config, "test": affinities, "val": None, "record": tmp_path / "r",
        "run_dir": None, "label": "", "scratch": tmp_path / "s",
    })()
    with pytest.raises(SystemExit, match="selecting on the number being reported"):
        evaluate.cmd_score(args)


def test_incompatible_kind_and_postprocessor_are_refused(instances, tmp_path):
    """Thresholding class scores as affinities produces a segmentation, not an error."""
    _, data, _ = instances
    scores = write_artifact(
        tmp_path / "scores.zarr", np.zeros((5, 4, 4, 4), np.float32), "class_scores"
    )
    config = _task_config(tmp_path, data, textwrap.dedent("""\
        [task]
        name = "instance_seg"
        truth_kind = "labels"

        [postprocess]
        name = "cc_threshold"
        logits = [5]

        [metric]
        names = ["voxel_instance"]
        """))

    import evaluate

    args = type("Args", (), {
        "config": config, "test": scores, "val": None, "record": tmp_path / "r",
        "run_dir": None, "label": "", "scratch": tmp_path / "s",
    })()
    with pytest.raises(ValueError, match="accepts artifacts of kind"):
        evaluate.cmd_score(args)


def test_task_and_metric_canonical_forms_must_agree(instances, tmp_path):
    _, data, _ = instances
    config = _task_config(tmp_path, data, textwrap.dedent("""\
        [task]
        name = "semantic_seg"

        [postprocess]
        name = "identity"

        [metric]
        names = ["voxel_instance"]
        """))

    from config import load_task_config
    from evaluate import build

    with pytest.raises(ValueError, match="consume something else"):
        build(load_task_config(config))


def test_leaderboard_renders_and_detects_drift(instances, tmp_path):
    tmp_path_, data, artifact = instances
    config = _task_config(tmp_path_, data, textwrap.dedent("""\
        [task]
        name = "instance_seg"
        truth_kind = "labels"

        [postprocess]
        name = "identity"

        [metric]
        names = ["voxel_instance"]
        """))

    import evaluate
    from report import leaderboard

    records = tmp_path_ / "records2"
    evaluate.cmd_score(type("Args", (), {
        "config": config, "test": artifact, "val": None, "record": records,
        "run_dir": None, "label": "arm_a", "scratch": tmp_path_ / "s2",
    })())

    output = tmp_path_ / "LEADERBOARD.md"
    leaderboard.write(records, output)
    text = output.read_text()
    assert "GENERATED FILE" in text
    assert "unit_task" in text and "arm_a" in text
    # The postprocessor is a column, so a reader cannot mistake a post-processing difference for a
    # model difference.
    assert "postprocess" in text
    assert leaderboard.check(records, output)

    output.write_text(text + "\nhand edit\n")
    assert not leaderboard.check(records, output)


# --------------------------------------------------------------- several volumes at once


@pytest.fixture
def two_volumes(tmp_path):
    """Two volumes, each with its own artifact, and one of them predicted perfectly.

    The case that exposed two bugs a single-volume test could not: the runner used to read one
    artifact for every volume, and to overwrite each volume's scores with the next one's. Both are
    silent -- the reported number is real, just not of what it claims.
    """
    root = tmp_path / "multi"
    root.mkdir()
    artifacts = root / "artifacts"
    artifacts.mkdir()

    entries = []
    for index, name in enumerate(("alpha", "beta")):
        truth, split = _truth_and_split(index)
        prediction = truth if name == "beta" else split      # beta is perfect, alpha is not
        entries.append((name, _volume(root, name, truth), truth.shape))
        write_artifact(
            artifacts / f"{name}.zarr", prediction, "instances",
            background_id=0, run="unit_run", step=7,
        )
    data = _data_config(root, entries)
    config = _task_config(root, data, textwrap.dedent("""\
        [task]
        name = "instance_seg"
        truth_kind = "labels"

        [postprocess]
        name = "identity"

        [metric]
        names = ["voxel_instance"]
        """))
    return root, config, artifacts


def test_each_volume_is_scored_against_its_own_artifact(two_volumes):
    root, config, artifacts = two_volumes
    import evaluate

    records = root / "records"
    evaluate.cmd_score(type("Args", (), {
        "config": config, "test": artifacts, "val": None, "record": records,
        "run_dir": None, "label": "multi", "scratch": root / "s",
    })())
    payload = json.loads((records / "unit_task" / "multi.json").read_text())

    per_volume = payload["per_volume"]
    assert set(per_volume) == {"alpha", "beta"}
    # beta was predicted exactly, alpha was over-segmented. If both volumes were read from one
    # artifact these would be equal, and if scores were overwritten only one would survive.
    assert per_volume["beta"]["voxel_instance"]["pq"] == pytest.approx(1.0)
    assert per_volume["alpha"]["voxel_instance"]["pq"] < 1.0

    # The aggregate is the unweighted mean over volumes, not the last volume's score.
    aggregate = payload["scores"]["voxel_instance"]["pq"]
    expected = (per_volume["alpha"]["voxel_instance"]["pq"]
                + per_volume["beta"]["voxel_instance"]["pq"]) / 2
    assert aggregate == pytest.approx(expected)
    assert payload["scores"]["voxel_instance"]["volumes_scored"] == 2.0
    assert payload["ranking"]["value"] == pytest.approx(expected)


def test_a_missing_per_volume_artifact_is_refused(two_volumes):
    """Scoring three of four volumes silently changes what the number covers."""
    root, config, artifacts = two_volumes
    import shutil

    import evaluate
    shutil.rmtree(artifacts / "beta.zarr")
    with pytest.raises(SystemExit, match="missing an artifact"):
        evaluate.cmd_score(type("Args", (), {
            "config": config, "test": artifacts, "val": None, "record": root / "r2",
            "run_dir": None, "label": "", "scratch": root / "s2",
        })())


def test_a_single_artifact_is_refused_for_a_multi_volume_task(two_volumes):
    root, config, artifacts = two_volumes
    import evaluate

    with pytest.raises(SystemExit, match="single artifact but this task has 2 volumes"):
        evaluate.cmd_score(type("Args", (), {
            "config": config, "test": artifacts / "alpha.zarr", "val": None,
            "record": root / "r3", "run_dir": None, "label": "", "scratch": root / "s3",
        })())
