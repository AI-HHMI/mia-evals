"""Every task config shown in the README must actually load.

The README's only configuration example had drifted badly: it omitted the required `[task]`
section, carried a `[predict]` section from before prediction moved to mia-train, and used
`fit_on` / `sweep_logits` / `higher_is_better` keys that do not exist alongside metric names
(`nerl`, `voi`) that are not registered. It would not load, and nothing noticed -- documentation
is the one part of a repository with no failing build to catch it.

So the example is executed rather than trusted. Any fenced `toml` block in the README that looks
like a task config -- it has a `task_name` -- is written to a temp file and passed through the real
loader. A block that cannot load is a bug in the README.

Deliberately checks loading only, not scoring: `load_task_config` parses the `miao` YAML without
opening any store, so this needs no data on disk and runs anywhere. The example points at an
in-repo data config for that reason.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

README = ROOT / "README.md"
FENCE = re.compile(r"```toml\n(.*?)```", re.DOTALL)


def all_blocks() -> list[tuple[int, str]]:
    """Every fenced toml block, paired with its line number so a failure can be located."""
    text = README.read_text()
    return [
        (text[: match.start()].count("\n") + 1, match.group(1))
        for match in FENCE.finditer(text)
    ]


def task_config_blocks() -> list[tuple[int, str]]:
    """Just the blocks that look like a task config, i.e. that declare a `task_name`."""
    return [(line, body) for line, body in all_blocks() if "task_name" in body]


@pytest.mark.unit
@pytest.mark.parametrize(
    "line, body", task_config_blocks(), ids=lambda v: f"L{v}" if isinstance(v, int) else ""
)
def test_readme_task_config_loads(line, body, tmp_path):
    import components  # noqa: F401  (populates the registries)
    from config import load_task_config

    # `config_path` in the README is relative to configs/tasks/, so the example is written there.
    target = ROOT / "configs" / "tasks" / "_readme_example.toml"
    target.write_text(body)
    try:
        config = load_task_config(target)
    except Exception as exc:                      # noqa: BLE001 -- reported, not handled
        pytest.fail(
            f"the toml block at README.md:{line} does not load: {type(exc).__name__}: {exc}\n"
            "A configuration example that cannot be loaded is worse than none: it is copied."
        )
    finally:
        target.unlink(missing_ok=True)

    assert config.task.name, "the example must name a task"
    assert config.metrics, "the example must name at least one metric"


@pytest.mark.unit
def test_the_readme_still_shows_a_task_config():
    """Guards the vacuous pass: no blocks collected means the test above checks nothing."""
    assert task_config_blocks(), (
        "no fenced toml block in README.md declares a task_name. Either the example was removed, "
        "or the fence label changed and this test silently stopped checking anything."
    )


@pytest.mark.unit
def test_readme_examples_are_valid_toml_at_all():
    """Every toml block, not just task configs -- a syntax error in any of them is a typo."""
    for line, body in all_blocks():
        try:
            tomllib.loads(body)
        except tomllib.TOMLDecodeError as exc:
            pytest.fail(f"README.md:{line} is not valid toml: {exc}")
