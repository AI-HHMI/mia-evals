"""Helpers shared by the markdown tables (`leaderboard.py`) and the HTML page (`page.py`)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .links import SHARE_KEYS, load_share_keys, share_url, views_directory
from .record import Submission, readme_path

if TYPE_CHECKING:
    from metrics.base import KeyInfo


def report_keys(metric_name: str) -> tuple[str, ...]:
    """The secondary columns a metric asks for, or none if it is not registered here.

    Looked up rather than stored in the record so that adding a column to a metric changes every
    table on the next render, without rewriting records that were correct when written.
    """
    try:
        import components  # noqa: F401  (populates the registry)
        from metrics.registry import MetricRegistry

        return tuple(MetricRegistry.get(metric_name).report_keys)
    except (ImportError, KeyError):
        return ()


def key_info(metric_name: str) -> dict[str, KeyInfo]:
    """Direction and description of each of a metric's leaderboard keys, as the metric declares
    them (`BaseMetric.key_info`), or nothing if it is not registered here."""
    try:
        import components  # noqa: F401  (populates the registry)
        from metrics.registry import MetricRegistry

        return dict(MetricRegistry.get(metric_name).key_info)
    except (ImportError, KeyError):
        return {}


def region_key(submission: Submission) -> str:
    """A short label for the extent scored, used to group rows that may be compared."""
    volumes = submission.region.get("volumes") or {}
    if not volumes:
        return "unspecified"
    parts = []
    for name in sorted(volumes):
        entry = volumes[name]
        shape = entry.get("shape")
        whole = entry.get("whole_region")
        extent = "x".join(str(int(s)) for s in shape) if shape else "?"
        parts.append(f"{name} {extent}{'' if whole else ' (sub-region)'}")
    return "; ".join(parts)


#: The truth kinds as they were named before the 2026-09-14 rename, so records written under the
#: old names read as the same route and do not force a column that shows a spelling difference.
LEGACY_TRUTH_KINDS = {"labels": "instances", "sibling_artifact": "instances_resampled"}


def truth_kind(submission: Submission) -> str:
    """How the record's task read its ground truth; `—` for a task with no such setting."""
    kwargs = (submission.config.get("task") or {}).get("kwargs") or {}
    kind = str(kwargs.get("truth_kind", "—"))
    return LEGACY_TRUTH_KINDS.get(kind, kind)


def cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "—" if value is None else str(value)


_VIEWS_LINE_URL = re.compile(r"^\*\*Views:\*\* \[[^\]]*\]\((\S+)\)", re.MULTILINE)


def views_link(root: Path, task_name: str) -> str | None:
    """The data-link URL of this task's HTML views page, or None if there is none.

    The URL carries the share's key, and the table it goes into is committed. That is accepted:
    the links only resolve for people with Janelia access, and a table that names its views is
    worth more than one that hides them. The key *file* stays untracked all the same -- so on a
    machine without it, the link already in the committed table is kept rather than dropped,
    which is what lets anyone re-render or `--check` the table and get the same text.
    """
    target = views_directory(root / SHARE_KEYS)
    if target is not None:
        page = Path(str(target).replace("{task}", task_name)) / f"{task_name}.html"
        if page.is_file():
            return share_url(page, load_share_keys(root / SHARE_KEYS))
    existing = readme_path(root, task_name)
    if existing.is_file():
        match = _VIEWS_LINE_URL.search(existing.read_text())
        if match:
            return match.group(1)
    return None
