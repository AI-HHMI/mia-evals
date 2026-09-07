"""Rendering the leaderboard from records, and checking the rendered file has not drifted.

Two rules are enforced here rather than left to whoever writes the table, because both are ways to
publish a ranking that is wrong while looking right:

**The postprocessor is a column.** A model emitting affinities plus thresholded components against
one emitting instance masks directly is a fair end-to-end comparison -- that is what a leaderboard
should measure. But "A beats B" can be a post-processing difference, and a table that hides which
postprocessor produced each row invites the reader to attribute it to the model.

**Regions are never mixed in one table.** Several metrics are not comparable across extents: the
same model measured 0.3045 nERL over a whole NISB cube and 0.4192 on a 512^3 block of that same
cube, because a shorter region truncates more branches. Rows are grouped by the region they were
scored over, and a group with more than one region is split rather than sorted together.

`--check` renders into memory and compares, writing nothing. Same idea as the data-config
generators' drift check: a stale committed table is caught by CI instead of by a reader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .record import (
    Submission,
    load_records,
    load_task,
    readme_path,
    task_names,
)

# `..` from `leaderboard/<task>/README.md` is `leaderboard/`, so docs are two levels up -- one more
# than the old single table at `leaderboard/README.md` needed.
TASK_HEADER = """<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with `mia-evals leaderboard --task {task}`, or verify with `--check`.
     Rows come from ./records/; edit a record, not this table. -->

# {task}

A score here is only interpretable against a floor. See
[docs/controls.md](../../docs/controls.md) for what these tasks measure with no model at all.
"""

INDEX_HEADER = """<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with `mia-evals leaderboard`, or verify with `--check`. -->

# Leaderboards

One directory per task, each holding that task's records and its own table. **Tasks are not
comparable with one another** -- they differ in volumes, scored extent, resolution and metric -- so
each has its own page rather than a section of a shared one.

A score is only interpretable against a floor. See [docs/controls.md](../docs/controls.md) for what
these tasks measure with no model at all.
"""


def _report_keys(metric_name: str) -> tuple[str, ...]:
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


def _region_key(submission: Submission) -> str:
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


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "—" if value is None else str(value)


def render_task(task_name: str, submissions: list[Submission]) -> str:
    """One task's page: a table per region it was scored over."""
    header = TASK_HEADER.format(task=task_name)
    if not submissions:
        return header + "\nNo records yet.\n"

    out = [header]
    by_region: dict[str, list[Submission]] = {}
    for submission in submissions:
        by_region.setdefault(_region_key(submission), []).append(submission)

    if len(by_region) > 1:
        out.append(
            "> Scored over different regions, so the groups below are **not** comparable with "
            "one another. Several of these metrics change with extent.\n"
        )

    for region in sorted(by_region):
        group = by_region[region]
        ranking = group[0].ranking
        metric, key = ranking.get("metric", "?"), ranking.get("key", "?")
        higher = bool(ranking.get("higher_is_better", True))
        group.sort(key=lambda s: s.ranking.get("value", 0.0), reverse=higher)

        # Columns in the order the *metrics* declare, not discovery order. A union over
        # sorted keys gave the alphabetically-first six, which for an instance task meant a
        # constant setting and a misleading count while the diagnostic split/merge terms were
        # dropped. `report_keys` lives on the metric because it knows which of its outputs are
        # diagnostic; anything a metric does not name stays out of the table and remains in
        # the record.
        extra: list[str] = []
        for submission in group:
            for name in sorted(submission.scores):
                for inner in _report_keys(name):
                    column = f"{name}.{inner}"
                    if (column != f"{metric}.{key}"
                            and column not in extra
                            and inner in submission.scores[name]):
                        extra.append(column)

        out.append(f"\n**Region:** {region}\n")
        direction = "higher is better" if higher else "lower is better"
        columns = ["#", "model", f"{metric}.{key} ({direction})", "postprocess", *extra[:6]]
        out.append("| " + " | ".join(columns) + " |")
        out.append("|" + "|".join(["---"] * len(columns)) + "|")
        for position, submission in enumerate(group, start=1):
            cells = [
                str(position),
                submission.identifier(),
                _cell(submission.ranking.get("value")),
                str(submission.postprocess.get("describe", "—")),
            ]
            for column in extra[:6]:
                name, _, inner = column.partition(".")
                cells.append(_cell(submission.scores.get(name, {}).get(inner)))
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_index(root: Path) -> str:
    """The directory listing at `leaderboard/README.md`: which tasks exist, and how big.

    Deliberately holds no scores. A shared page that ranked anything would recreate the thing the
    split was for -- two tasks under one heading, read as one comparison.
    """
    out = [INDEX_HEADER, "\n| task | entries | table |", "| --- | --- | --- |"]
    names = task_names(root)
    for name in names:
        count = len(load_task(root, name))
        out.append(f"| `{name}` | {count} | [{name}/README.md]({name}/README.md) |")
    if not names:
        out.append("| _none yet_ |  |  |")
    return "\n".join(out).rstrip() + "\n"


def write_task(root: Path, task_name: str) -> Path:
    """Render one task's table in place, leaving every other task's file untouched."""
    output = readme_path(root, task_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_task(task_name, load_task(root, task_name)))
    return output


def write_index(root: Path) -> Path:
    output = root / "README.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_index(root))
    return output


def write(root: Path, task_name: str | None = None) -> list[Path]:
    """Rebuild one task's table, or every task's plus the index.

    The index is rewritten even for a single task, because a new task's first record creates a
    directory the index does not yet list, and a stale index is the one drift `--check` on a single
    task cannot see.
    """
    written = [write_task(root, task_name)] if task_name else [
        write_task(root, name) for name in task_names(root)
    ]
    return [*written, write_index(root)]


def check(root: Path, task_name: str | None = None) -> list[Path]:
    """Files whose rendered content differs from what is on disk. Empty means up to date."""
    names = [task_name] if task_name else task_names(root)
    stale = []
    for name in names:
        output = readme_path(root, name)
        expected = render_task(name, load_task(root, name))
        if not output.is_file() or output.read_text() != expected:
            stale.append(output)
    index = root / "README.md"
    if not index.is_file() or index.read_text() != render_index(root):
        stale.append(index)
    return stale
