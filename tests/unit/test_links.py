"""Fileglancer links on the leaderboard: the path mapping and the existence-checked cell."""
from __future__ import annotations

from pathlib import Path

import pytest

from report.links import (
    FALLBACK_RULES,
    FILEGLANCER,
    NEUROGLANCER,
    artifact_directory,
    checkpoint_directory,
    fetch_shares,
    fileglancer_url,
    link_cell,
    load_share_keys,
    neuroglancer_state,
    ome_transform,
    share_url,
    view_entries,
)

SHARES = {
    "/nrs/scicompsoft": "nrs_scicompsoft",
    "/groups/scicompsoft/home": "groups_scicompsoft_home",
    "/groups/scicompsoft/scicompsoft": "groups_scicompsoft_scicompsoft",
    "/groups/miaai/miaai": "groups_miaai_miaai",
}


@pytest.mark.unit
def test_paths_map_to_the_share_with_the_longest_matching_mount():
    assert fileglancer_url("/nrs/scicompsoft/orhane/x/y.zarr", SHARES) == \
        f"{FILEGLANCER}/browse/nrs_scicompsoft/orhane/x/y.zarr"
    assert fileglancer_url("/groups/scicompsoft/home/orhane", SHARES) == \
        f"{FILEGLANCER}/browse/groups_scicompsoft_home/orhane"
    assert fileglancer_url("/nrs/scicompsoft", SHARES) == f"{FILEGLANCER}/browse/nrs_scicompsoft"
    # A prefix match on characters alone is not a mount match: /nrs/scicompsoft2 is not under
    # /nrs/scicompsoft, so it falls through to the naming rule and gets its own share name.
    assert fileglancer_url("/nrs/scicompsoft2/x", SHARES) == \
        f"{FILEGLANCER}/browse/nrs_scicompsoft2/x"


@pytest.mark.unit
def test_the_fallback_rule_reproduces_the_published_names():
    """Off the network the rule must give the same URL the list would, or `--check` flaps."""
    for mount, name in SHARES.items():
        assert fileglancer_url(f"{mount}/a/b", None) == f"{FILEGLANCER}/browse/{name}/a/b"
    assert fileglancer_url("/scratch/orhane/x", None) is None
    assert fileglancer_url("/tmp/x", None) is None
    assert len(FALLBACK_RULES) == 4


@pytest.mark.unit
def test_the_live_share_list_if_reachable_agrees_with_the_rule():
    shares = fetch_shares(timeout=2.0)
    if shares is None:
        pytest.skip("fileglancer not reachable from here")
    for mount, name in SHARES.items():
        if mount in shares:
            assert shares[mount] == name
            assert fileglancer_url(f"{mount}/a", shares) == fileglancer_url(f"{mount}/a", None)


@pytest.mark.unit
def test_artifact_and_checkpoint_directories_come_from_the_record():
    producer = {"artifacts": {"a": "/nrs/x/test/a.zarr", "b": "/nrs/x/test/b.zarr"}, "step": 50000}
    assert artifact_directory(producer) == Path("/nrs/x/test")
    assert artifact_directory({"artifacts": {"only": "/nrs/x/test/a.zarr"}}) == Path("/nrs/x/test")
    assert artifact_directory({"artifacts": {}}) is None
    assert checkpoint_directory(producer, {"run_dir": "/nrs/runs/r"}) == \
        Path("/nrs/runs/r/checkpoints/step_50000")
    assert checkpoint_directory(producer, {"run_dir_missing": "/nrs/runs/gone"}) == \
        Path("/nrs/runs/gone/checkpoints/step_50000")
    assert checkpoint_directory({"checkpoint": "/elsewhere/ckpt", "step": 1}, {}) == \
        Path("/elsewhere/ckpt")
    assert checkpoint_directory({"step": 5}, {}) is None
    assert checkpoint_directory({}, {"run_dir": "/nrs/runs/r"}) is None


@pytest.mark.unit
def test_the_cell_links_what_exists_marks_what_is_gone_and_omits_what_was_never_named():
    present = Path("/nrs/scicompsoft/orhane/present")
    gone = Path("/nrs/scicompsoft/orhane/gone")
    cell = link_cell(
        [("artifacts", present), ("checkpoint", gone), ("other", None)],
        SHARES, missing=lambda p: p == gone,
    )
    assert cell == (f"[artifacts]({FILEGLANCER}/browse/nrs_scicompsoft/orhane/present)"
                    " · checkpoint (missing)")
    assert link_cell([("artifacts", None)], SHARES) == "—"
    # A path on a mount fileglancer does not publish is shown as a path, not dropped.
    assert link_cell([("artifacts", Path("/scratch/x"))], SHARES, missing=lambda p: False) == \
        "artifacts: `/scratch/x`"


@pytest.mark.unit
def test_absence_counts_as_missing_only_where_the_storage_is_mounted(tmp_path):
    from report.links import known_missing, mount_root

    assert mount_root("/nrs/scicompsoft/orhane/x/y.zarr") == Path("/nrs/scicompsoft")
    assert mount_root("/groups/scicompsoft/home/orhane/p") == Path("/groups/scicompsoft/home")
    assert mount_root("/elsewhere/p") == Path("/")
    # A mount this machine does not have: the file is not "missing", it is out of sight.
    assert known_missing("/nrs/no_such_group_here/orhane/x.zarr") is False
    # The root is always mounted, so an absent unpatterned path is missing for real.
    assert known_missing(tmp_path / "gone") is True
    assert known_missing(tmp_path) is False


# ---------------------------------------------------------------------------- viewer links

KEYS = {"/nrs/scicompsoft/orhane/mia-train-scratch": "ARTKEY",
        "/groups/miaai/miaai/lmd-v0.0.1/data": "RAWKEY"}


def _ome_artifact(path: Path, shape=(4, 6, 8), scale=(8.0, 8.0, 8.0),
                  shift=(100.0, 200.0, 300.0)):
    import numpy as np
    import zarr

    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    level = group.create_array(name="s0", shape=shape, dtype="u4")
    level[:] = np.ones(shape, dtype="u4")
    group.attrs.update(kind="instances", background_id=0, ome={"version": "0.5", "multiscales": [{
        "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
        "datasets": [{"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": list(scale)},
            {"type": "translation", "translation": list(shift)},
        ]}],
    }]})
    return path


@pytest.mark.unit
def test_share_urls_use_the_key_of_the_covering_directory_and_the_real_path(tmp_path):
    url = share_url("/nrs/scicompsoft/orhane/mia-train-scratch/x/test/v.zarr", KEYS)
    assert url == (f"{FILEGLANCER}/files/ARTKEY/nrs/scicompsoft/orhane/mia-train-scratch"
                   "/x/test/v.zarr")
    assert share_url("/groups/miaai/miaai/lmd-v0.0.1/data/s.zarr/raw", KEYS) == \
        f"{FILEGLANCER}/files/RAWKEY/groups/miaai/miaai/lmd-v0.0.1/data/s.zarr/raw"
    assert share_url("/nrs/scicompsoft/orhane/elsewhere/v.zarr", KEYS) is None
    # A symlink is served at its target's path.
    target = tmp_path / "real.zarr"
    target.mkdir()
    link = tmp_path / "link.zarr"
    link.symlink_to(target)
    assert share_url(link, {str(tmp_path): "K"}) == f"{FILEGLANCER}/files/K{target}"


@pytest.mark.unit
def test_the_state_places_three_layers_and_centres_on_the_prediction(tmp_path):
    artifact = _ome_artifact(tmp_path / "v.zarr")
    transform = ome_transform(artifact)
    assert transform == (["z", "y", "x"], [8.0, 8.0, 8.0], [100.0, 200.0, 300.0], [4, 6, 8])
    url = neuroglancer_state("R", "P", "T", transform, "arm", window=(301.0, 1130.0))
    assert url.startswith(NEUROGLANCER)
    import json
    import urllib.parse
    state = json.loads(urllib.parse.unquote(url[len(NEUROGLANCER):]))
    assert [(layer["type"], layer["source"]) for layer in state["layers"]] == [
        ("image", "R/|zarr3:"), ("segmentation", "P/|zarr3:"), ("segmentation", "T/|zarr3:")]
    assert state["layers"][2]["visible"] is False and state["layout"] == "4panel-alt"
    assert state["layers"][0]["shaderControls"] == {"normalized": {"range": [301.0, 1130.0]}}
    assert state["dimensions"] == {"z": [8e-9, "m"], "y": [8e-9, "m"], "x": [8e-9, "m"]}
    # Raised memory budgets travel with every view: at the viewer's 1 GB / 2 GB defaults our
    # single-level 256^3 segmentation chunks do not all fit and show as holes.
    from report.links import GPU_MEMORY_LIMIT, SYSTEM_MEMORY_LIMIT
    assert (state["gpuMemoryLimit"], state["systemMemoryLimit"]) == (
        GPU_MEMORY_LIMIT, SYSTEM_MEMORY_LIMIT) == (4_000_000_000, 8_000_000_000)
    # centre = translation / voxel + shape / 2, in the state's own (voxel) units
    assert state["position"] == pytest.approx([100 / 8 + 2, 200 / 8 + 3, 300 / 8 + 4])
    assert ome_transform(tmp_path / "absent.zarr") is None
    # Side by side: two synchronised viewers, truth visible on the left, labelling on the right.
    side = json.loads(urllib.parse.unquote(
        neuroglancer_state("R", "P", "T", transform, "arm", "side_by_side")[len(NEUROGLANCER):]))
    assert side["layout"] == {"type": "row", "children": [
        {"type": "viewer", "layers": ["raw", "truth"], "layout": "xy"},
        {"type": "viewer", "layers": ["raw", "arm"], "layout": "xy"}]}
    assert side["layers"][2]["visible"] is True
    # Without readable metadata the state has no position, and works all the same.
    plain = json.loads(urllib.parse.unquote(
        neuroglancer_state("R", "P", None, None, "arm")[len(NEUROGLANCER):]))
    assert "position" not in plain and "shaderControls" not in plain["layers"][0]


@pytest.mark.unit
def test_view_entries_need_a_share_over_both_the_raw_store_and_the_artifact(tmp_path):
    artifacts = {"vol": "/nrs/scicompsoft/orhane/mia-train-scratch/x/test/vol.zarr"}
    config = {"volumes": [{"name": "vol", "path": "/groups/miaai/miaai/lmd-v0.0.1/data/s.zarr"}]}
    producer = {"artifacts": artifacts, "kind": "instances"}
    entries = view_entries(producer, {}, config, KEYS, "arm", exists=lambda p: True)
    assert [(e[0], e[3]) for e in entries] == [("vol", False)]
    overlay, side = entries[0][1], entries[0][2]
    assert overlay.startswith(NEUROGLANCER) and side.startswith(NEUROGLANCER)
    assert "RAWKEY" in overlay and "ARTKEY" in overlay and "vol.gt.zarr" in overlay
    assert overlay != side and "unfiltered" not in overlay
    # Missing artifact, uncovered raw store, or no keys at all: no entry, no error.
    assert view_entries(producer, {}, config, KEYS, "arm", exists=lambda p: False) == []
    assert view_entries(producer, {}, config, {}, "arm", exists=lambda p: True) == []
    assert view_entries(producer, {}, {"volumes": []}, KEYS, "arm") == []


@pytest.mark.unit
def test_views_show_the_scored_labelling_when_the_record_names_one():
    import json
    import urllib.parse

    artifacts = {"vol": "/nrs/scicompsoft/orhane/mia-train-scratch/x/test/vol.zarr"}
    scored = {"vol": "/nrs/scicompsoft/orhane/mia-train-scratch/x/scored/vol.zarr"}
    config = {"volumes": [{"name": "vol", "path": "/groups/miaai/miaai/lmd-v0.0.1/data/s.zarr"}]}
    producer = {"artifacts": artifacts, "kind": "instances"}
    (entry,) = view_entries(producer, {"scored_artifacts": scored}, config, KEYS, "arm",
                            exists=lambda p: True)
    assert entry[3] is True
    state = json.loads(urllib.parse.unquote(entry[1][len(NEUROGLANCER):]))
    by_name = {layer["name"]: layer for layer in state["layers"]}
    assert "x/scored/vol.zarr" in by_name["arm"]["source"]
    assert "x/test/vol.zarr" in by_name["unfiltered"]["source"]
    assert by_name["unfiltered"]["visible"] is False
    # Affinities are not a segmentation layer, so no hidden unfiltered layer for those rows.
    (entry,) = view_entries({**producer, "kind": "affinity"}, {"scored_artifacts": scored},
                            config, KEYS, "arm", exists=lambda p: True)
    assert "unfiltered" not in entry[1]
    # A scored copy that was deleted falls back to the producer's output.
    (entry,) = view_entries(producer, {"scored_artifacts": scored}, config, KEYS, "arm",
                            exists=lambda p: "scored" not in str(p))
    assert entry[3] is False


@pytest.mark.unit
def test_share_keys_come_from_the_untracked_file(tmp_path):
    assert load_share_keys(tmp_path / "none.json") == {}
    (tmp_path / "k.json").write_text('{"shares": {"/a/b/": "K1"}}')
    assert load_share_keys(tmp_path / "k.json") == {"/a/b": "K1"}


@pytest.mark.unit
def test_html_views_page_has_clickable_links_in_leaderboard_order(tmp_path):
    from report.leaderboard import render_views
    from report.record import Submission

    def submission(label, value, artifact):
        return Submission(
            task_name="t", producer={"artifacts": {"vol": artifact}}, scores={},
            ranking={"metric": "m", "key": "k", "value": value, "higher_is_better": True},
            postprocess={}, region={},
            config={"volumes": [
                {"name": "vol", "path": "/groups/miaai/miaai/lmd-v0.0.1/data/s.zarr"}]},
            label=label,
        )

    from report.links import record_views

    covered = _ome_artifact(tmp_path / "vol.zarr")
    keys = {**KEYS, str(tmp_path): "TMPKEY"}
    rows = [submission("worse", 0.1, str(covered)), submission("better", 0.9, str(covered))]
    for row in rows:                       # what `mia-evals score` stores, given these keys
        row.views = record_views(row.producer, row.postprocess, row.config, keys, row.label)
    page = render_views("t", rows)
    assert page is not None and page.startswith("<!doctype html>")
    assert page.index("<h2>1. better</h2>") < page.index("<h2>2. worse</h2>")
    assert page.count('<a href="https://fileglancer.int.janelia.org/neuroglancer/#!') == 4
    assert "missing" not in page.split("<h2>")[1]
    assert "&amp;" not in page.split("<h2>")[0]          # header text is plain
    assert render_views("t", []) is None

    # A record scored without keys carries no links, and the page says so instead of guessing.
    bare = submission("bare", 0.5, str(covered))
    page = render_views("t", [bare])
    assert "<i>missing</i>" in page and "neuroglancer/#!" not in page


@pytest.mark.unit
def test_views_dir_task_placeholder_puts_each_page_in_its_task_directory(tmp_path):
    """`{task}` in `views_dir` expands per task: the /nrs layout keeps every task's files under
    mia-evals/<task>/, its HTML views page included."""
    import json

    from report.leaderboard import write_views
    from report.record import Submission

    covered = _ome_artifact(tmp_path / "vol.zarr")
    (tmp_path / "fileglancer_shares.json").write_text(json.dumps({
        "shares": {**KEYS, str(tmp_path): "TMPKEY"},
        "views_dir": str(tmp_path / "mia-evals" / "{task}" / "views"),
    }))
    row = Submission(
        task_name="t1", producer={"artifacts": {"vol": str(covered)}}, scores={},
        ranking={"metric": "m", "key": "k", "value": 1.0, "higher_is_better": True},
        postprocess={}, region={},
        config={"volumes": [{"name": "vol", "path": "/groups/miaai/miaai/lmd-v0.0.1/data/s.zarr"}]},
        label="row",
    )
    url = write_views(tmp_path, "t1", [row])
    page = tmp_path / "mia-evals" / "t1" / "views" / "t1.html"
    assert page.is_file()
    assert url is not None and url.endswith("/mia-evals/t1/views/t1.html")


def test_the_task_page_links_its_views_page_and_check_agrees(tmp_path):
    """The table links the views page through the data link that serves it, once it exists, so
    `--check` renders the same text `leaderboard` wrote; without a page there is no link."""
    import json

    from report import leaderboard
    from report.record import Submission

    covered = _ome_artifact(tmp_path / "vol.zarr")
    (tmp_path / "fileglancer_shares.json").write_text(json.dumps({
        "shares": {**KEYS, str(tmp_path): "TMPKEY"},
        "views_dir": str(tmp_path / "mia-evals" / "{task}" / "views"),
    }))
    Submission(
        task_name="t1", producer={"artifacts": {"vol": str(covered)}},
        scores={"m": {"k": 1.0}},
        ranking={"metric": "m", "key": "k", "value": 1.0, "higher_is_better": True},
        postprocess={"describe": "identity"}, region={},
        config={"volumes": [{"name": "vol", "path": "/groups/miaai/miaai/lmd-v0.0.1/data/s.zarr"}]},
        label="row",
    ).write(tmp_path)

    output, _index = leaderboard.write(tmp_path, "t1")       # the task page and the index
    text = output.read_text()
    assert (tmp_path / "mia-evals" / "t1" / "views" / "t1.html").is_file()
    assert "[neuroglancer views for every row below](https://fileglancer.int.janelia.org/files/TMPKEY/" in text
    assert "/mia-evals/t1/views/t1.html)" in text
    assert leaderboard.check(tmp_path, "t1") == []

    # Without the key file (anyone else's machine) the page cannot be rewritten, but the link the
    # committed table already carries is kept, so rendering and --check agree everywhere.
    (tmp_path / "fileglancer_shares.json").unlink()
    leaderboard.write(tmp_path, "t1")
    assert "[neuroglancer views for every row below](https://fileglancer.int.janelia.org/files/TMPKEY/" in output.read_text()
    assert leaderboard.check(tmp_path, "t1") == []


@pytest.mark.unit
def test_refresh_views_fills_records_from_the_local_keys(tmp_path):
    """Records scored without keys (or before links were stored) gain links where this machine's
    keys cover their artifacts; nothing is removed where they do not."""
    import json

    from report import leaderboard
    from report.record import Submission, load_record

    covered = _ome_artifact(tmp_path / "vol.zarr")
    row = Submission(
        task_name="t1", producer={"artifacts": {"vol": str(covered)}}, scores={"m": {"k": 1.0}},
        ranking={"metric": "m", "key": "k", "value": 1.0, "higher_is_better": True},
        postprocess={"describe": "identity"}, region={},
        config={"volumes": [{"name": "vol", "path": "/groups/miaai/miaai/lmd-v0.0.1/data/s.zarr"}]},
        label="row",
    )
    path = row.write(tmp_path)
    assert load_record(path).views == {}
    assert leaderboard.refresh_views(tmp_path) == []          # no key file: nothing to do

    (tmp_path / "fileglancer_shares.json").write_text(json.dumps({
        "shares": {**KEYS, str(tmp_path): "TMPKEY"},
        "views_dir": str(tmp_path / "views"),
    }))
    assert leaderboard.refresh_views(tmp_path) == [path]
    views = load_record(path).views
    assert set(views) == {"vol"} and views["vol"]["shows"] == "before size filter"
    assert views["vol"]["overlay"].startswith("https://fileglancer.int.janelia.org/neuroglancer/#!")
    assert leaderboard.refresh_views(tmp_path) == []          # already current: not rewritten
