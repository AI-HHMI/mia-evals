"""End to end: a task config, an artifact, a record, a leaderboard.

Built from synthetic volumes so it needs no cluster storage, no checkpoint, and neither numba nor
funlib. What it exercises is the wiring the design rests on: that a config resolves through the
registries, that an incompatible pairing is refused rather than attempted, that a sweep cannot be
fitted on the reported split, and that the rendered table is reproducible from the records.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

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
        truth_kind = "instances"

        [postprocess]
        name = "identity"

        [metric]
        names = ["voxel_instance"]
        rank_by = "voxel_instance"
        """))

    import evaluate

    records = tmp_path / "records"
    args = type("Args", (), {
        "config": config, "test": artifact, "val": None, "leaderboard": records,
        "run_dir": None,
        "scored_out": None, "no_scored": False, "label": "", "scratch": tmp_path / "scratch",
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
        truth_kind = "instances"

        [postprocess]
        name = "cc_threshold"
        logits = [3, 5, 7]

        [metric]
        names = ["voxel_instance"]
        """))

    import evaluate

    args = type("Args", (), {
        "config": config, "test": affinities, "val": None, "leaderboard": tmp_path / "r",
        "run_dir": None,
        "scored_out": None, "no_scored": False, "label": "", "scratch": tmp_path / "s",
    })()
    with pytest.raises(SystemExit, match="selecting on the number being reported"):
        evaluate.cmd_score(args)


def test_an_artifact_predicted_over_another_volume_is_refused(instances, tmp_path):
    """A prediction of one volume filed under another volume's name.

    Artifacts are matched to volumes by file name, so a misfiled prediction would be scored
    against the wrong ground truth without an error. The producer records the store it read
    (`source_path`), and that is what is checked; an artifact without it is trusted as before.
    """
    root, data, _ = instances
    _, prediction = _truth_and_split(seed=1)
    config = _task_config(root, data, textwrap.dedent("""\
        [task]
        name = "instance_seg"
        truth_kind = "instances"

        [postprocess]
        name = "identity"

        [metric]
        names = ["voxel_instance"]
        rank_by = "voxel_instance"
        """))

    import evaluate

    def args(test):
        return type("Args", (), {
            "config": config, "test": test, "val": None, "leaderboard": tmp_path / "records",
            "run_dir": None,
        "scored_out": None, "no_scored": False, "label": "", "scratch": tmp_path / "scratch",
        })()

    stray = write_artifact(
        tmp_path / "stray.zarr", prediction, "instances", background_id=0,
        source_path=str(tmp_path / "some_other_cube.zarr"),
    )
    with pytest.raises(SystemExit, match="predicted over"):
        evaluate.cmd_score(args(stray))

    matching = write_artifact(
        tmp_path / "matching.zarr", prediction, "instances", background_id=0,
        source_path=str(root / "cube.zarr"),
    )
    evaluate.cmd_score(args(matching))
    assert len(list((tmp_path / "records").rglob("*.json"))) == 1


def test_incompatible_kind_and_postprocessor_are_refused(instances, tmp_path):
    """Thresholding class scores as affinities produces a segmentation, not an error."""
    _, data, _ = instances
    scores = write_artifact(
        tmp_path / "scores.zarr", np.zeros((5, 4, 4, 4), np.float32), "class_scores"
    )
    config = _task_config(tmp_path, data, textwrap.dedent("""\
        [task]
        name = "instance_seg"
        truth_kind = "instances"

        [postprocess]
        name = "cc_threshold"
        logits = [5]

        [metric]
        names = ["voxel_instance"]
        """))

    import evaluate

    args = type("Args", (), {
        "config": config, "test": scores, "val": None, "leaderboard": tmp_path / "r",
        "run_dir": None,
        "scored_out": None, "no_scored": False, "label": "", "scratch": tmp_path / "s",
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
        truth_kind = "instances"

        [postprocess]
        name = "identity"

        [metric]
        names = ["voxel_instance"]
        """))

    import evaluate
    from report import leaderboard

    root = tmp_path_ / "leaderboard2"
    evaluate.cmd_score(type("Args", (), {
        "config": config, "test": artifact, "val": None, "leaderboard": root,
        "run_dir": None,
        "scored_out": None, "no_scored": False, "label": "arm_a", "scratch": tmp_path_ / "s2",
    })())

    # Scoring writes the record *and* renders that task's table, so the two cannot drift apart by
    # a forgotten second command. Both land under the task's own directory.
    assert (root / "unit_task" / "records" / "arm_a.json").is_file()
    # ... and the post-processed labelling that was scored is kept and named, so a viewer can
    # show exactly the voxels behind the number.
    import json as _json
    record_ = _json.loads((root / "unit_task" / "records" / "arm_a.json").read_text())
    scored = record_["postprocess"]["scored_artifacts"]
    assert scored and all(Path(p).is_dir() for p in scored.values())
    from artifact import open_artifact as _open
    kept = _open(next(iter(scored.values())))
    assert kept.kind == "instances"      # a bare array here, since the fixture's source is one
    output = root / "unit_task" / "README.md"
    assert output.is_file()

    text = output.read_text()
    assert "GENERATED FILE" in text
    assert "unit_task" in text and "arm_a" in text
    # The postprocessor is a column, so a reader cannot mistake a post-processing difference for a
    # model difference.
    assert "postprocess" in text
    assert leaderboard.check(root) == []

    output.write_text(text + "\nhand edit\n")
    assert leaderboard.check(root) == [output]
    assert leaderboard.check(root, "unit_task") == [output]

    # Rebuilding one task repairs it, and the index is refreshed alongside.
    leaderboard.write(root, "unit_task")
    assert leaderboard.check(root) == []
    index = (root / "README.md").read_text()
    assert "unit_task" in index
    # The index lists tasks; it must not rank anything, or the split back into per-task pages
    # would have bought nothing.
    assert "arm_a" not in index


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
        truth_kind = "instances"

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
        "config": config, "test": artifacts, "val": None, "leaderboard": records,
        "run_dir": None,
        "scored_out": None, "no_scored": False, "label": "multi", "scratch": root / "s",
    })())
    payload = json.loads((records / "unit_task" / "records" / "multi.json").read_text())

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
            "config": config, "test": artifacts, "val": None, "leaderboard": root / "r2",
            "run_dir": None,
        "scored_out": None, "no_scored": False, "label": "", "scratch": root / "s2",
        })())


def test_a_single_artifact_is_refused_for_a_multi_volume_task(two_volumes):
    root, config, artifacts = two_volumes
    import evaluate

    with pytest.raises(SystemExit, match="single artifact but this task has 2 volumes"):
        evaluate.cmd_score(type("Args", (), {
            "config": config, "test": artifacts / "alpha.zarr", "val": None,
            "leaderboard": root / "r3", "run_dir": None,
        "scored_out": None, "no_scored": False,
            "label": "", "scratch": root / "s3",
        })())


def test_leaderboard_root_prefers_a_checkout(tmp_path, monkeypatch):
    """A source checkout beside the module wins, so an editable install works from anywhere."""
    import evaluate

    monkeypatch.chdir(tmp_path)
    root = evaluate._leaderboard_root()
    # This test suite runs against the checkout, where `<repo>/leaderboard/` exists.
    assert root == Path(evaluate.__file__).resolve().parents[1] / "leaderboard"
    assert root.is_dir()
    assert root != tmp_path / "leaderboard"


def test_leaderboard_root_falls_back_to_cwd_when_installed(tmp_path, monkeypatch):
    """Without a checkout beside the module, default to the working directory.

    Regression test for a real fault found by a from-scratch install check: with the path derived
    only from `__file__`, a non-editable install resolved the default to
    `<venv>/lib/pythonX.Y/leaderboard`, so `score` would write git-tracked records into
    site-packages and `leaderboard --check` compared a table that did not exist.
    """
    import evaluate

    fake_site_packages = tmp_path / "venv" / "lib" / "python3.14" / "site-packages"
    fake_site_packages.mkdir(parents=True)
    monkeypatch.setattr(evaluate, "__file__", str(fake_site_packages / "evaluate.py"))
    monkeypatch.chdir(tmp_path)

    assert evaluate._leaderboard_root() == tmp_path / "leaderboard"


# ------------------------------------------------------------ the fit split lives in the task


def _split_task_config(root, test_data: str, fit_data: str | None, body: str,
                       test_volumes=None, fit_volumes=None) -> str:
    """A task file declaring `[data.test]` and, unless `fit_data` is None, `[data.fit]`."""
    def table(name, data_path, volumes):
        lines = [f"[data.{name}]", f'config_path = "{data_path}"']
        if volumes is not None:
            lines.append("volumes = [" + ", ".join(f'"{v}"' for v in volumes) + "]")
        return "\n".join(lines) + "\n"

    text = 'task_name = "unit_task"\n\n' + table("test", test_data, test_volumes)
    if fit_data is not None:
        text += "\n" + table("fit", fit_data, fit_volumes)
    path = root / "split_task.toml"
    path.write_text(text + "\n" + body)
    return str(path)


SIZE_FILTER_BODY = textwrap.dedent("""\
    [task]
    name = "instance_seg"
    truth_kind = "instances"

    [postprocess]
    name = "size_filter"
    min_sizes = [0, 3]

    [metric]
    names = ["voxel_instance"]
    """)


def test_the_fit_split_is_declared_in_the_task_and_may_not_overlap_the_reported_one(tmp_path):
    from config import load_task_config

    truth, _ = _truth_and_split()
    a = _volume(tmp_path, "alpha", truth)
    b = _volume(tmp_path, "beta", truth)
    both = _data_config(tmp_path, [("alpha", a, truth.shape), ("beta", b, truth.shape)])

    config = load_task_config(_split_task_config(
        tmp_path, both, both, SIZE_FILTER_BODY, test_volumes=["alpha"], fit_volumes=["beta"],
    ))
    assert [v.name for v in config.volumes] == ["alpha"]
    assert [v.name for v in config.fit_volumes] == ["beta"]
    assert config.fit_data_config_path.resolve() == Path(both).resolve()
    record = config.as_record()
    assert [v["name"] for v in record["fit_volumes"]] == ["beta"]

    with pytest.raises(ValueError, match="share volume"):
        load_task_config(_split_task_config(
            tmp_path, both, both, SIZE_FILTER_BODY, test_volumes=["alpha"],
        ))                                       # fit = both volumes, so alpha is on both sides

    # A plain [data] is still the reported split alone, with no fit split.
    plain = load_task_config(_task_config(tmp_path, both, SIZE_FILTER_BODY))
    assert plain.fit_volumes is None and plain.fit_data_config_path is None

    # [data.fit] without [data.test] is a shape error, not a silently missing test split.
    lonely = tmp_path / "lonely.toml"
    lonely.write_text(
        'task_name = "unit_task"\n\n[data.fit]\nconfig_path = "' + both + '"\n\n' + SIZE_FILTER_BODY
    )
    with pytest.raises(ValueError, match=r"\[data.fit\] needs a \[data.test\]"):
        load_task_config(lonely)


def test_a_sweep_is_fitted_on_the_declared_fit_split_and_applied_to_the_test_split(tmp_path):
    """`--val` covers `[data.fit]`'s volumes, `--test` covers `[data.test]`'s; no other flag."""
    truth, _ = _truth_and_split()
    a = _volume(tmp_path, "alpha", truth)
    b = _volume(tmp_path, "beta", truth)
    both = _data_config(tmp_path, [("alpha", a, truth.shape), ("beta", b, truth.shape)])
    config = _split_task_config(
        tmp_path, both, both, SIZE_FILTER_BODY, test_volumes=["alpha"], fit_volumes=["beta"],
    )

    # On the fit volume a 2-voxel speck of a third id is a false positive that min_size=3 removes,
    # so the fit must choose 3; the test volume is predicted perfectly either way.
    fit_prediction = truth.copy()
    fit_prediction[3, 0, :2] = 9
    val_dir = tmp_path / "val"; val_dir.mkdir()
    test_dir = tmp_path / "test"; test_dir.mkdir()
    write_artifact(val_dir / "beta.zarr", fit_prediction, "instances", background_id=0)
    write_artifact(test_dir / "alpha.zarr", truth.copy(), "instances", background_id=0)

    import evaluate

    records = tmp_path / "records"
    args = type("Args", (), {
        "config": config, "test": test_dir, "val": val_dir, "leaderboard": records,
        "run_dir": None, "scored_out": None, "no_scored": False, "label": "split",
        "scratch": tmp_path / "scratch",
    })()
    evaluate.cmd_score(args)
    payload = json.loads((records / "unit_task" / "records" / "split.json").read_text())
    assert payload["postprocess"]["params"] == {"min_size": 3}
    assert Path(payload["postprocess"]["fitted_on_data_config"]).resolve() == Path(both).resolve()
    assert list(payload["region"]["volumes"]) == ["alpha"]

    # A task with no [data.fit] cannot be handed a --val: there is nothing declared to fit on.
    plain = _task_config(tmp_path, both, SIZE_FILTER_BODY)
    args.config = plain
    with pytest.raises(SystemExit, match="declares no fit split"):
        evaluate.cmd_score(args)
