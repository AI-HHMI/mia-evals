"""A task is its reported volumes (with their ground truth) and its ranking metric.

Nothing else. The post-processor, the truth route and the fit split may vary between rows and are
shown; the identity may not, and is checked in three places: across the scoring configs in this repo,
against the existing records before a new one is written, and across a directory whenever it is
loaded for rendering or `--check`.
"""

from __future__ import annotations

import glob
import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
import zarr

from artifact import write_artifact
from report import leaderboard, record
from report.record import Submission

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def _config_identity(config):
    import components  # noqa: F401
    from metrics.registry import MetricRegistry

    key = MetricRegistry.get(config.rank_by).primary
    return record.task_identity(config.as_record()["volumes"], config.rank_by, key)


def test_task_files_sharing_a_name_agree_on_what_the_task_is():
    """Two scoring configs may share a task_name only as two routes to the same table."""
    from config import load_scoring_config

    by_name: dict[str, list[tuple[str, dict]]] = {}
    for path in sorted(glob.glob(str(ROOT / "configs" / "scoring" / "*.toml"))):
        config = load_scoring_config(path)
        by_name.setdefault(config.task_name, []).append((path, _config_identity(config)))
    assert by_name, "no scoring configs found"
    for name, entries in by_name.items():
        reference_path, reference = entries[0]
        for path, identity in entries[1:]:
            differences = record.identity_differences(reference, identity)
            assert not differences, (
                f"{path} and {reference_path} both claim task {name!r} but differ: {differences}"
            )


def _submission(volumes, truth_kind="instances", metric="voxel_instance", key="pq",
                label="row", value=0.5) -> Submission:
    return Submission(
        task_name="unit_task",
        producer={"artifacts": {v["name"]: f"/nowhere/{v['name']}.zarr" for v in volumes}},
        scores={metric: {key: value}},
        ranking={"metric": metric, "key": key, "value": value, "higher_is_better": True},
        postprocess={"describe": "identity"},
        region={"volumes": {v["name"]: {"origin": [0, 0, 0], "shape": [4, 4, 4],
                                        "whole_region": False} for v in volumes}},
        config={"task": {"name": "instance_seg", "kwargs": {"truth_kind": truth_kind}},
                "volumes": volumes},
        label=label,
    )


ALPHA = {"name": "alpha", "path": "/store/alpha.zarr", "label_key": "labels/gt",
         "bounding_box": [[0, 4], [0, 4], [0, 4]]}
BETA = {"name": "beta", "path": "/store/beta.zarr", "label_key": "labels/gt",
        "bounding_box": None}


def test_identity_differences_name_what_differs():
    same = record.task_identity([ALPHA, BETA], "voxel_instance", "pq")
    assert record.identity_differences(same, record.task_identity([BETA, ALPHA], "voxel_instance", "pq")) == []

    other_label = dict(ALPHA, label_key="labels/other")
    diffs = record.identity_differences(same, record.task_identity([other_label, BETA], "voxel_instance", "pq"))
    assert diffs == ["alpha: label_key 'labels/other' vs 'labels/gt'"]

    diffs = record.identity_differences(same, record.task_identity([ALPHA], "skeleton_erl", "nerl"))
    assert diffs == ["ranking metric skeleton_erl.nerl vs voxel_instance.pq", "volumes ['alpha'] vs ['alpha', 'beta']"]


def test_a_directory_whose_records_disagree_cannot_be_loaded_or_checked(tmp_path):
    root = tmp_path / "board"
    _submission([ALPHA, BETA], label="a").write(root)
    _submission([ALPHA], label="b").write(root)                 # another test set, same name
    with pytest.raises(ValueError, match="was not scored on the same task"):
        record.load_task(root, "unit_task")
    with pytest.raises(ValueError, match="volumes \\['alpha'\\] vs \\['alpha', 'beta'\\]"):
        leaderboard.check(root, "unit_task")


def test_the_truth_route_becomes_a_column_only_when_it_varies():
    rows = [_submission([ALPHA], truth_kind="instances", label="a"),
            _submission([ALPHA], truth_kind="instances", label="b", value=0.4)]
    assert "| truth |" not in leaderboard.render_task("unit_task", rows)

    # The pre-rename spelling of the same route is not a difference.
    rows[1].config["task"]["kwargs"]["truth_kind"] = "sibling_artifact"
    rows[0].config["task"]["kwargs"]["truth_kind"] = "instances_resampled"
    assert "| truth |" not in leaderboard.render_task("unit_task", rows)

    rows[1].config["task"]["kwargs"]["truth_kind"] = "instances"
    page = leaderboard.render_task("unit_task", rows)
    assert "| truth |" in page
    assert "| instances_resampled |" in page and "| instances |" in page


def _volume(root, name, labels):
    path = root / f"{name}.zarr"
    group = zarr.open(str(path), mode="w")
    array = group.create_array("labels/gt/s0", shape=labels.shape, dtype="i8", chunks=labels.shape)
    array[:] = labels
    return str(path)


def _data_config(root, name, entries):
    volumes = "\n".join(
        textwrap.dedent(f"""\
            - name: {vname}
              path: {vpath}
              image_key: raw
              label_key: labels/gt
              zarr_version: zarr3
              bounding_box: [[0, {shape[0]}], [0, {shape[1]}], [0, {shape[2]}]]
        """)
        for vname, vpath, shape in entries
    )
    path = root / f"{name}.yaml"
    path.write_text(
        "resolutions:\n- - 8.0\n  - 8.0\n  - 8.0\npatch_size:\n- 4\n- 4\n- 4\n"
        "output_axes: lcxyz\nvolumes:\n" + volumes
    )
    return str(path)


def test_scoring_refuses_a_record_that_would_redefine_the_task(tmp_path):
    """The check that matters: before a record joins a table, not after."""
    truth = np.zeros((4, 4, 4), dtype=np.int64)
    truth[:2] = 1
    truth[3] = 2
    alpha = _volume(tmp_path, "alpha", truth)
    beta = _volume(tmp_path, "beta", truth)
    body = textwrap.dedent("""\
        [task]
        name = "instance_seg"
        truth_kind = "instances"

        [postprocess]
        name = "identity"

        [metric]
        names = ["voxel_instance"]
        """)

    def task_file(name, data):
        path = tmp_path / f"{name}.toml"
        path.write_text(f'task_name = "unit_task"\n\n[data]\nconfig_path = "{data}"\n\n{body}')
        return str(path)

    one = task_file("one", _data_config(tmp_path, "one", [("alpha", alpha, truth.shape)]))
    two = task_file("two", _data_config(tmp_path, "two", [("alpha", alpha, truth.shape),
                                                          ("beta", beta, truth.shape)]))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for name in ("alpha", "beta"):
        write_artifact(artifacts / f"{name}.zarr", truth.copy(), "instances", background_id=0)

    import evaluate

    def args(config, label):
        return type("Args", (), {
            "config": config, "test": artifacts, "val": None, "leaderboard": tmp_path / "board",
            "run_dir": None, "scored_out": None, "no_scored": False, "label": label,
            "scratch": tmp_path / "scratch",
        })()

    evaluate.cmd_score(args(one, "first"))
    with pytest.raises(SystemExit, match="volumes \\['alpha', 'beta'\\] vs \\['alpha'\\]"):
        evaluate.cmd_score(args(two, "second"))
    # The refused record was never written, so the table is still consistent.
    assert [p.name for p in (tmp_path / "board" / "unit_task" / "records").glob("*.json")] == ["first.json"]
    assert leaderboard.check(tmp_path / "board") == []
    # Re-scoring the same task is, of course, fine.
    evaluate.cmd_score(args(one, "third"))
    written = json.loads((tmp_path / "board" / "unit_task" / "records" / "third.json").read_text())
    assert written["task_name"] == "unit_task"
