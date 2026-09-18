"""configs/ is one directory per task: `configs/<task_name>/<route>.toml` and `configs/<task_name>/data/*.yaml`."""
from __future__ import annotations

import glob
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCORING = sorted(glob.glob(str(ROOT / "configs" / "*" / "*.toml")))


@pytest.mark.unit
@pytest.mark.parametrize("path", SCORING, ids=lambda p: str(Path(p).relative_to(ROOT / "configs")))
def test_scoring_config_lives_in_its_task_directory_under_its_route_name(path):
    import components  # noqa: F401
    from config import load_scoring_config

    config = load_scoring_config(Path(path))
    assert Path(path).parent.name == config.task_name, "directory name must be the task name"
    assert Path(path).stem == config.route, "file stem must be the route (the record name's last part)"
    for data in (config.data_config_path, config.fit_data_config_path):
        if data is not None:
            assert Path(data).resolve().parent == (Path(path).parent / "data").resolve(), \
                f"{data} is not in this task's data/ directory"


@pytest.mark.unit
def test_configs_hold_nothing_but_task_directories():
    stray = [p.name for p in (ROOT / "configs").iterdir() if not p.is_dir() or not glob.glob(str(p / "*.toml"))]
    assert not stray, f"configs/ may only hold task directories with scoring configs: {stray}"
    assert SCORING, "no scoring configs found"


@pytest.mark.unit
def test_a_volume_shared_by_two_tasks_is_the_same_entry():
    """The zebrafish task re-declares two of the eight-volume task's volumes; a drift between the two
    copies would silently score one task on a different region than the other reports."""
    entries: dict[str, dict] = {}
    for data in sorted(glob.glob(str(ROOT / "configs" / "*" / "data" / "*.yaml"))):
        for volume in yaml.safe_load(Path(data).read_text()).get("volumes") or []:
            seen = entries.setdefault(volume["name"], volume)
            assert seen == volume, f"{volume['name']} differs between task data configs ({data})"
    assert entries
