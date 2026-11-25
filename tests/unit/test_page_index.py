"""The page's list of benchmarks: every task once, under what it ranks on, with a link to each of
its versions; a header of fixed size.

The list is the page when no task is open and the menu above an open one, so it is the only way
from one task to another: a version missing from it cannot be reached, and a header that held
something per task would grow with every task added.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from report import page
from report.record import Submission, load_task, task_names, task_version
from report.table import build

ROOT = Path(__file__).resolve().parents[2] / "leaderboard"


class IndexReader(HTMLParser):
    """The groups of the page's `<nav>`: each one's title, what it ranks by, and its rows.

    A row is a task: its name, where the name leads, and its `versions` (where each leads, and
    its label). A name without a version has no `versions`.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.groups: list[dict[str, Any]] = []
        self.inside = False
        self.path: list[str] = []          # open tags inside the nav, innermost last
        self.version: dict[str, str] | None = None      # the version link being read

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "nav":
            self.inside = True
        elif self.inside and tag != "wbr":      # a line-break hint inside a name: no end tag
            self.path.append(tag)
            if tag == "h2":
                self.groups.append({"title": "", "ranked": "", "tasks": []})
            elif tag == "li":
                self.groups[-1]["tasks"].append({"href": "", "name": ""})
            elif tag == "a" and "ver" in (a.get("class") or "").split():
                self.version = {"href": str(a["href"]), "label": ""}
                self.groups[-1]["tasks"][-1].setdefault("versions", []).append(self.version)
            elif tag == "a":
                self.groups[-1]["tasks"][-1]["href"] = a["href"]

    def handle_data(self, data: str) -> None:
        if not self.inside or not self.path:
            return
        if self.version is not None:
            self.version["label"] += data
        elif "a" in self.path and self.path[-1] == "b":
            self.groups[-1]["tasks"][-1]["name"] += data
        elif "h2" in self.path:
            self.groups[-1]["ranked" if self.path[-1] == "small" else "title"] += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "nav":
            self.inside = False
        elif self.inside:
            self.path.pop()
            if tag == "a":
                self.version = None


def read_index(html: str) -> list[dict[str, Any]]:
    reader = IndexReader()
    reader.feed(html)
    reader.close()
    return reader.groups


def _header(html: str) -> str:
    return html[html.index("<header>"):html.index("</header>")]


def _title_links(section: str) -> tuple[list[str], list[str]]:
    """The version links by a section's title: where each leads, and which are marked as open."""
    head = section[:section.index('<p class="note">')]
    return (re.findall(r'href="#([^"]+)"', head),
            re.findall(r'class="ver on" href="#([^"]+)"', head))


def _title(section: str) -> str:
    """A section's title as shown."""
    return re.findall(r'<h2 class="mono">([^<]*)</h2>', section)[0]


def _row(metric: str, key: str) -> Submission:
    return Submission(
        task_name="t", producer={"step": 5}, scores={metric: {key: 0.5}},
        ranking={"metric": metric, "key": key, "value": 0.5, "higher_is_better": True},
        postprocess={"describe": "identity"},
        region={"volumes": {"v": {"origin": [0, 0, 0], "shape": [4, 4, 4], "whole_region": True}}},
        config={}, label="a",
    )


@pytest.mark.unit
def test_every_version_is_listed_once_in_its_tasks_row_under_what_it_ranks_on():
    names = task_names(ROOT)
    groups = read_index(page.render_page(ROOT))
    listed = [v["href"] for g in groups for task in g["tasks"] for v in task["versions"]]
    assert names and sorted(listed) == sorted(f"#{name}" for name in names)
    for group in groups:
        for task in group["tasks"]:
            numbers = [int(version["label"][1:]) for version in task["versions"]]
            assert numbers == sorted(numbers, reverse=True), task["name"]      # newest first
            assert task["href"] == task["versions"][0]["href"], task["name"]   # the name opens it
            for version in task["versions"]:
                name = version["href"][1:]
                assert name == f"{task['name']}_{version['label']}"
                ranking = load_task(ROOT, name)[0].ranking
                assert group["title"] == page.KINDS.get(ranking["metric"], ranking["metric"]), name
                assert group["ranked"] == f"ranked by {ranking['key']}", name


@pytest.mark.unit
def test_every_title_is_the_tasks_name_then_a_link_to_each_version_the_open_one_marked():
    by_task: dict[str, list[str]] = {}
    for name in task_names(ROOT):
        by_task.setdefault(task_version(name)[0], []).append(name)
    assert any(len(names) > 1 for names in by_task.values()), "no task has two versions"
    html = page.render_page(ROOT)
    for task, names in by_task.items():
        newest_first = sorted(names, key=lambda name: task_version(name)[1] or 0,
                              reverse=True)
        for name in names:
            section = html[html.index(f'<section id="{name}">'):]
            assert _title(section) == task, name          # the marked link says which version
            assert _title_links(section) == (newest_first, [name]), name

    # A task with one version shows that one, marked. A name without a version shows none, and
    # a section given no versions keeps the version in its title.
    one = build("cells_v1", [_row("voxel_instance", "pq")], None, None)
    plain = build("cells", [_row("voxel_instance", "pq")], None, None)
    assert _title_links(page.section(one, [one])) == (["cells_v1"], ["cells_v1"])
    assert _title(page.section(one, [one])) == "cells" and _title(page.section(one)) == "cells_v1"
    assert _title(page.section(plain, [plain])) == "cells"
    assert 'class="ver' not in page.section(one) + page.section(plain, [plain])


@pytest.mark.unit
def test_a_task_has_one_row_with_a_link_to_each_version_newest_first():
    tables = [
        build("cells_v1", [_row("voxel_instance", "pq"), _row("voxel_instance", "pq")], None, None),
        build("cells_v2", [_row("voxel_instance", "pq")], None, None),
        build("cells_crop_v2", [_row("voxel_instance", "pq")], None, None),
        build("cells", [_row("voxel_instance", "pq")], None, None),      # a name without a version
    ]
    assert read_index(f"<nav>{page.index(tables)}</nav>") == [
        {"title": "Instance segmentation", "ranked": "ranked by pq", "tasks": [
            {"href": "#cells_v2", "name": "cells", "versions": [
                {"href": "#cells_v2", "label": "v2"}, {"href": "#cells_v1", "label": "v1"}]},
            {"href": "#cells_crop_v2", "name": "cells_crop", "versions": [
                {"href": "#cells_crop_v2", "label": "v2"}]},
            {"href": "#cells", "name": "cells"}]},
    ]
    assert task_version("cells_v12") == ("cells", 12)
    assert [task_version(name)[1] for name in ("cells", "cells_v0", "cells_v01")] == [None] * 3


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
            {"href": "#cells_b", "name": "cells_b"}, {"href": "#cells_a", "name": "cells_a"}]},
        {"title": "Semantic segmentation", "ranked": "ranked by mean_iou", "tasks": [
            {"href": "#classes", "name": "classes"}]},
        {"title": "some_new_metric", "ranked": "ranked by score", "tasks": [
            {"href": "#new", "name": "new"}]},
        {"title": "No records yet", "ranked": "", "tasks": [
            {"href": "#empty", "name": "empty"}]},
    ]


@pytest.mark.unit
def test_the_header_is_the_same_for_any_number_of_tasks(tmp_path):
    assert task_names(ROOT) and not task_names(tmp_path)
    full, empty = page.render_page(ROOT), page.render_page(tmp_path)
    assert _header(full) == _header(empty)
    assert not read_index(empty) and "<section" not in empty
