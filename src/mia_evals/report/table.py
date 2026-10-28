"""What one task's leaderboard shows, decided once for both of its renderings.

`leaderboard.render_task` writes it as the task's README.md and `page.section` as the task's part
of leaderboard/index.html. Both format a `TaskTable` from `build` and neither reads records itself,
so the two cannot drift apart: the same rows in the same order, the same columns under the same
headers, the same cell text, the same links and the same preamble. The page adds presentation only
-- sortable columns, each column's best value highlighted, task cards, a glossary of the metric
keys -- and takes that from here as well; the directions and descriptions are the metrics' own
(`BaseMetric.key_info`). tests/unit/test_leaderboard_sync.py reads both outputs back and holds them
to the same content.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import cell, key_info, region_key, report_keys, truth_kind
from .links import Link, artifact_directory, checkpoint_directory, known_missing, link_entries
from .record import Submission

#: Preamble text, shown word for word by both renderings.
LINKS_NOTE = ("The artifact/checkpoint/view links below point to locations on the Janelia cluster "
              "and will only work on the Janelia network.")
VIEWS_TEXT = "neuroglancer views for every row below"
MIXED_REGIONS = ("Scored over different regions, so the groups below are **not** comparable with "
                 "one another. Several of these metrics change with extent.")
REGION_LABEL = "Region:"
NO_RECORDS = "No records yet."

#: Secondary metric columns per table, beyond the ranking one. A metric names its own, in order
#: (`report_keys`); whatever falls past this stays in the record.
MAX_EXTRA_COLUMNS = 6


@dataclass(frozen=True)
class Column:
    """One column. `kind` is "rank", "model", "links", "metric", "postprocess" or "truth"."""

    kind: str
    header: str
    #: For a metric column: the metric and its key, which direction is better (None when the
    #: metric does not say) and the metric's description of the key.
    metric: str = ""
    key: str = ""
    higher_is_better: bool | None = None
    description: str = ""


@dataclass(frozen=True)
class Cell:
    """One cell's text. A rank or metric cell also carries the number it sorts on and whether it is
    its column's best; a links cell carries its entries, its text being their labels."""

    text: str
    value: float | None = None
    best: bool = False
    links: tuple[Link, ...] = ()


@dataclass(frozen=True)
class Row:
    identifier: str
    cells: tuple[Cell, ...]


@dataclass(frozen=True)
class Group:
    """The rows scored over one region, in ranking order, under their columns."""

    region: str
    columns: tuple[Column, ...]
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class TaskTable:
    name: str
    #: The task's views page, or None when it has none.
    views: str | None
    ranking_key: str
    entries: int
    groups: tuple[Group, ...]

    @property
    def mixed_regions(self) -> bool:
        return len(self.groups) > 1


def build(task_name: str, submissions: list[Submission], views: str | None,
          shares: dict[str, str] | None,
          missing: Callable[[Path], bool] = known_missing) -> TaskTable:
    """Everything a task's leaderboard shows, from its records: a group per region scored over."""
    show_truth = len({truth_kind(s) for s in submissions}) > 1
    by_region: dict[str, list[Submission]] = {}
    for submission in submissions:
        by_region.setdefault(region_key(submission), []).append(submission)
    groups = tuple(_group(region, by_region[region], show_truth, shares, missing)
                   for region in sorted(by_region))
    ranking_key = str(submissions[0].ranking.get("key", "?")) if submissions else "—"
    return TaskTable(task_name, views, ranking_key, len(submissions), groups)


def _number(value: Any) -> float | None:
    """What a cell sorts on, or None for a value that is not a plain number."""
    is_number = isinstance(value, int | float) and not isinstance(value, bool)
    return float(value) if is_number else None


def _group(region: str, group: list[Submission], show_truth: bool,
           shares: dict[str, str] | None, missing: Callable[[Path], bool]) -> Group:
    ranking = group[0].ranking
    metric, key = str(ranking.get("metric", "?")), str(ranking.get("key", "?"))
    higher = bool(ranking.get("higher_is_better", True))
    group = sorted(group, key=lambda s: s.ranking.get("value", 0.0), reverse=higher)

    # Columns in the order the *metrics* declare, not discovery order. A union over sorted keys
    # gave the alphabetically-first six, which for an instance task meant a constant setting and
    # a misleading count while the diagnostic split/merge terms were dropped. `report_keys` lives
    # on the metric because it knows which of its outputs are diagnostic; anything a metric does
    # not name stays out of the table and remains in the record.
    extra: list[tuple[str, str]] = []
    for submission in group:
        for name in sorted(submission.scores):
            for inner in report_keys(name):
                if ((name, inner) != (metric, key) and (name, inner) not in extra
                        and inner in submission.scores[name]):
                    extra.append((name, inner))
    extra = extra[:MAX_EXTRA_COLUMNS]

    direction = "higher is better" if higher else "lower is better"
    ranked = key_info(metric).get(key)
    metric_columns = [Column("metric", f"{metric}.{key} ({direction})", metric, key, higher,
                             ranked.description if ranked else "")]
    for name, inner in extra:
        info = key_info(name).get(inner)
        metric_columns.append(Column("metric", f"{name}.{inner}", name, inner,
                                     info.higher_is_better if info else None,
                                     info.description if info else ""))
    columns = (Column("rank", "#"), Column("model", "model"), Column("links", "links"),
               metric_columns[0], Column("postprocess", "postprocess"),
               *([Column("truth", "truth")] if show_truth else []), *metric_columns[1:])

    values = [[s.ranking.get("value"), *(s.scores.get(n, {}).get(i) for n, i in extra)]
              for s in group]
    best: list[float | None] = []
    for j, column in enumerate(metric_columns):
        found = [n for n in (_number(row[j]) for row in values) if n is not None]
        if not found or column.higher_is_better is None:
            best.append(None)
        else:
            best.append(max(found) if column.higher_is_better else min(found))

    rows = []
    for position, (submission, numbers) in enumerate(zip(group, values, strict=True), start=1):
        links = tuple(link_entries(
            [("artifacts", artifact_directory(submission.producer)),
             ("checkpoint", checkpoint_directory(submission.producer, submission.provenance))],
            shares, missing,
        ))
        scores = []
        for value, top in zip(numbers, best, strict=True):
            number = _number(value)
            scores.append(Cell(cell(value), number, number is not None and number == top))
        cells = (
            Cell(str(position), float(position)),
            Cell(submission.identifier()),
            Cell(" · ".join(link.label for link in links) or "—", links=links),
            scores[0],
            Cell(str(submission.postprocess.get("describe", "—"))),
            *([Cell(truth_kind(submission))] if show_truth else []),
            *scores[1:],
        )
        rows.append(Row(submission.identifier(), cells))
    return Group(region, columns, tuple(rows))
