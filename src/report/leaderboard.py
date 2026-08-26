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

from .record import Submission, load_records

HEADER = """<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with `python src/evaluate.py leaderboard`, or verify with `--check`.
     Rows come from leaderboard/records/; edit a record, not this table. -->

# Leaderboard
"""


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


def render(records: dict[str, list[Submission]]) -> str:
    """Markdown for every task, one table per (task, region) group."""
    if not records:
        return HEADER + "\nNo records yet.\n"

    out = [HEADER]
    for task_name in sorted(records):
        submissions = records[task_name]
        out.append(f"\n## {task_name}\n")

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

            # Union across rows: two submissions of one task may report different metric families
            # (a skeleton one and a voxel one), and dropping a column because one row lacks it
            # would hide a number that was measured.
            extra: list[str] = []
            for submission in group:
                for name, values in sorted(submission.scores.items()):
                    for inner in sorted(values):
                        column = f"{name}.{inner}"
                        if column != f"{metric}.{key}" and column not in extra:
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


def write(records_root: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(load_records(records_root)))
    return output


def check(records_root: Path, output: Path) -> bool:
    """True when the committed table matches what the records render to."""
    if not output.is_file():
        return False
    return output.read_text() == render(load_records(records_root))
