"""The page's list of benchmarks: every task once, under what it ranks on; a header of fixed size.

The list is the page when no task is open and the menu above an open one, so it is the only way
from one task to another: a task missing from it cannot be reached, and a header that held
something per task would grow with every task added.
"""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from report import page
from report.record import Submission, load_task, task_names
from report.table import build

ROOT = Path(__file__).resolve().parents[2] / "leaderboard"


class IndexReader(HTMLParser):
    """The groups of the page's `<nav>`: each one's title, what it ranks by, and its tasks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.groups: list[dict[str, Any]] = []
        self.inside = False
        self.path: list[str] = []          # open tags inside the nav, innermost last

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "nav":
            self.inside = True
        elif self.inside and tag != "wbr":      # a line-break hint inside a name: no end tag
            self.path.append(tag)
            if tag == "h2":
                self.groups.append({"title": "", "ranked": "", "tasks": []})
            elif tag == "a":
                self.groups[-1]["tasks"].append({"href": a["href"], "name": "", "entries": ""})

    def handle_data(self, data: str) -> None:
        if not self.inside or not self.path:
            return
        if "a" in self.path and self.path[-1] in ("b", "small"):
            field = "name" if self.path[-1] == "b" else "entries"
            self.groups[-1]["tasks"][-1][field] += data
        elif "h2" in self.path:
            self.groups[-1]["ranked" if self.path[-1] == "small" else "title"] += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "nav":
            self.inside = False
        elif self.inside:
            self.path.pop()


def read_index(html: str) -> list[dict[str, Any]]:
    reader = IndexReader()
    reader.feed(html)
    reader.close()
    return reader.groups


def _header(html: str) -> str:
    return html[html.index("<header>"):html.index("</header>")]


def _row(metric: str, key: str) -> Submission:
    return Submission(
        task_name="t", producer={"step": 5}, scores={metric: {key: 0.5}},
        ranking={"metric": metric, "key": key, "value": 0.5, "higher_is_better": True},
        postprocess={"describe": "identity"},
        region={"volumes": {"v": {"origin": [0, 0, 0], "shape": [4, 4, 4], "whole_region": True}}},
        config={}, label="a",
    )


@pytest.mark.unit
def test_every_task_is_listed_once_under_what_it_ranks_on():
    names = task_names(ROOT)
    groups = read_index(page.render_page(ROOT))
    assert names and sorted(t["name"] for g in groups for t in g["tasks"]) == names
    for group in groups:
        for task in group["tasks"]:
            submissions = load_task(ROOT, task["name"])
            ranking = submissions[0].ranking
            assert task["href"] == f"#{task['name']}"
            assert task["entries"] == \
                f"{len(submissions)} {'entry' if len(submissions) == 1 else 'entries'}"
            metric = ranking["metric"]
            assert group["title"] == page.KINDS.get(metric, metric), task["name"]
            assert group["ranked"] == f"ranked by {ranking['key']}", task["name"]


@pytest.mark.unit
def test_groups_are_the_kinds_then_other_metrics_then_tasks_without_records():
    tables = [
        build("empty", [], None, None),
        build("new", [_row("some_new_metric", "score")], None, None),
        build("classes", [_row("semantic", "mean_iou")], None, None),
        build("cells_b", [_row("voxel_instance", "pq")], None, None),
        build("cells_a", [_row("voxel_instance", "pq"), _row("voxel_instance", "pq")], None, None),
    ]
    assert read_index(f"<nav>{page.index(tables)}</nav>") == [
        {"title": "Instance segmentation", "ranked": "ranked by pq", "tasks": [
            {"href": "#cells_b", "name": "cells_b", "entries": "1 entry"},
            {"href": "#cells_a", "name": "cells_a", "entries": "2 entries"}]},
        {"title": "Semantic segmentation", "ranked": "ranked by mean_iou", "tasks": [
            {"href": "#classes", "name": "classes", "entries": "1 entry"}]},
        {"title": "some_new_metric", "ranked": "ranked by score", "tasks": [
            {"href": "#new", "name": "new", "entries": "1 entry"}]},
        {"title": "No records yet", "ranked": "", "tasks": [
            {"href": "#empty", "name": "empty", "entries": "0 entries"}]},
    ]


@pytest.mark.unit
def test_the_header_is_the_same_for_any_number_of_tasks(tmp_path):
    assert task_names(ROOT) and not task_names(tmp_path)
    full, empty = page.render_page(ROOT), page.render_page(tmp_path)
    assert _header(full) == _header(empty)
    assert not read_index(empty) and "<section" not in empty
