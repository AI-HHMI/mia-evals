"""The README table and the HTML page show one leaderboard: `table.build`, written twice.

Both renderings are read back -- the markdown by its lines, the page by an HTML parser -- into the
same plain content (preamble, regions, headers, cell text, link entries) and compared, so a
renderer that shows something the other does not fails here, whoever adds it. Highlighting,
sorting, the task cards and the glossary are the page's presentation and are not compared.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from report import leaderboard, page
from report.common import views_link
from report.record import Submission, load_task, task_names
from report.table import LINKS_NOTE, MIXED_REGIONS, NO_RECORDS, REGION_LABEL, build

ROOT = Path(__file__).resolve().parents[2] / "leaderboard"
LINK = re.compile(r"\[(?P<name>[^\]]+)\]\((?P<url>[^)]+)\)")
PLAIN_MIXED = MIXED_REGIONS.replace("**", "")


def _content() -> dict[str, Any]:
    return {"note": False, "views": None, "mixed": False, "empty": False, "groups": []}


def _markdown_links(text: str) -> tuple[tuple[str, str | None], ...]:
    if text == "—":
        return ()
    entries = []
    for part in text.split(" · "):
        match = LINK.fullmatch(part)
        entries.append((match["name"], match["url"]) if match else (part.replace("`", ""), None))
    return tuple(entries)


def read_markdown(text: str) -> dict[str, Any]:
    """A task's README.md as plain content."""
    out = _content()
    for line in text.splitlines():
        if line == LINKS_NOTE:
            out["note"] = True
        elif line == NO_RECORDS:
            out["empty"] = True
        elif line.startswith("**Views:** "):
            out["views"] = LINK.search(line)["url"]
        elif line.startswith("> "):
            out["mixed"] = line[2:].replace("**", "") == PLAIN_MIXED
        elif line.startswith(f"**{REGION_LABEL}** "):
            out["groups"].append({"region": line.split("** ", 1)[1], "headers": [], "rows": []})
        elif line.startswith("| "):
            group = out["groups"][-1]
            cells = line[2:-2].split(" | ")
            if not group["headers"]:
                group["headers"] = cells
            else:
                at = group["headers"].index("links")
                group["rows"].append([*cells[:at], _markdown_links(cells[at]), *cells[at + 1:]])
    return out


class PageReader(HTMLParser):
    """Each `<section>` of the page as the plain content `read_markdown` gives a README."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: dict[str, dict[str, Any]] = {}
        self.section: dict[str, Any] | None = None
        self.text: list[str] | None = None        # the element being read, and what it is
        self.reading = ""
        self.closes = ""
        self.cell: list[str] | None = None
        self.entries: list[tuple[str, str | None]] = []
        self.entry: tuple[list[str], str | None] | None = None
        self.entry_closes = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        kind = (a.get("class") or "").split()[:1]
        name = kind[0] if kind else ""
        if tag == "section":
            self.section = self.sections[str(a["id"])] = _content()
        elif self.section is None:
            return
        elif tag == "p" and name in ("note", "views", "mixed", "empty"):
            self.text, self.reading, self.closes = [], name, "p"
        elif tag == "a" and self.reading == "views":
            self.section["views"] = a["href"]
        elif tag == "div" and name == "region":
            self.section["groups"].append({"region": [], "headers": [], "rows": []})
        elif tag == "li":
            self.text, self.reading, self.closes = [], "region", "li"
        elif tag == "th":
            self.text, self.reading, self.closes = [], "header", "th"
        elif tag == "tr" and "data-id" in a:
            self.section["groups"][-1]["rows"].append([])
        elif tag == "td":
            self.cell, self.entries = [], []
        elif self.cell is not None and name in ("chip", "missing", "nolink"):
            self.entry, self.entry_closes = ([], a.get("href")), tag

    def handle_data(self, data: str) -> None:
        for sink in (self.text, self.cell, self.entry[0] if self.entry else None):
            if sink is not None:
                sink.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.section is None:
            return
        if self.entry is not None and tag == self.entry_closes:
            self.entries.append(("".join(self.entry[0]), self.entry[1]))
            self.entry = None
        elif self.cell is not None and tag == "td":
            group = self.section["groups"][-1]
            row = group["rows"][-1]
            links = group["headers"][len(row)] == "links"
            row.append(tuple(self.entries) if links else "".join(self.cell))
            self.cell = None
        elif self.text is not None and tag == self.closes:
            text, reading = "".join(self.text), self.reading
            self.text, self.reading = None, ""
            if reading == "note":
                self.section["note"] = text == LINKS_NOTE
            elif reading == "empty":
                self.section["empty"] = text == NO_RECORDS
            elif reading == "mixed":
                self.section["mixed"] = text == PLAIN_MIXED
            elif reading == "region":
                self.section["groups"][-1]["region"].append(text)
            elif reading == "header":
                self.section["groups"][-1]["headers"].append(text)
        elif tag == "section":
            for group in self.section["groups"]:
                group["region"] = "; ".join(group["region"])
            self.section = None


def read_page(html: str) -> dict[str, dict[str, Any]]:
    """Every task section of a rendered page, by task name, as plain content."""
    reader = PageReader()
    reader.feed(html)
    reader.close()
    return reader.sections


@pytest.mark.unit
def test_every_task_reads_the_same_in_its_readme_and_on_the_page():
    names = task_names(ROOT)
    sections = read_page(page.render_page(ROOT))
    assert names and sorted(sections) == names
    for name in names:
        readme = leaderboard.render_task(name, load_task(ROOT, name), views_link(ROOT, name))
        content = read_markdown(readme)
        assert content["note"] and content["groups"], name
        assert content == sections[name], name


def _row(label: str, value: float, *, shape: int = 4, truth: str = "instances",
         artifact: str | None = None, run_dir: str | None = None,
         voi_split: float | None = 0.5) -> Submission:
    return Submission(
        task_name="t", producer={"artifacts": {"v": artifact} if artifact else {}, "step": 5},
        scores={"voxel_instance": {"pq": value, "voi_merge": 0.1, "voi_split": voi_split}},
        ranking={"metric": "voxel_instance", "key": "pq", "value": value, "higher_is_better": True},
        postprocess={"describe": "mws(min_size=10)"},
        region={"volumes": {"v": {"origin": [0, 0, 0], "shape": [shape] * 3,
                                  "whole_region": False}}},
        config={"task": {"kwargs": {"truth_kind": truth}}},
        provenance={"run_dir": run_dir} if run_dir else {},
        label=label,
    )


@pytest.mark.unit
def test_links_truth_regions_and_missing_values_read_the_same_in_both():
    """A served link, a path nothing serves, a deleted target, a missing value, the truth column
    and a second region, each as the README shows it and as the page does."""
    rows = [
        _row("a", 0.5, artifact="/nrs/scicompsoft/x/a/v.zarr", run_dir="/nrs/scicompsoft/x/run"),
        _row("b", 0.7, artifact="/scratch/b/v.zarr", truth="instances_resampled", voi_split=None),
        _row("c", 0.6, shape=8, artifact="/nrs/scicompsoft/x/c/v.zarr"),
    ]
    # The links name the artifacts' directory; c's is the deleted one.
    table = build("t", rows, "https://example.org/t.html", None,
                  missing=lambda path: str(path).endswith("/c"))
    content = read_markdown(leaderboard.markdown(table))
    assert content == read_page(page.section(table))["t"]

    assert content["mixed"] and content["views"] == "https://example.org/t.html"
    first, second = content["groups"]
    assert first["headers"] == ["#", "model", "links", "voxel_instance.pq (higher is better)",
                                "postprocess", "truth", "voxel_instance.voi_merge",
                                "voxel_instance.voi_split"]
    b, a = first["rows"]
    assert b[2] == (("artifacts: /scratch/b", None),) and b[-1] == "—"
    assert [name for name, _ in a[2]] == ["artifacts", "checkpoint"]
    assert all(url and url.startswith("https://") for _, url in a[2])
    assert second["rows"][0][2] == (("artifacts (missing)", None),)

    # The page's highlight follows each metric's own direction: pq up, voi_split down.
    group = table.groups[0]
    at = {column.header: j for j, column in enumerate(group.columns)}
    assert [r.cells[at["voxel_instance.pq (higher is better)"]].best for r in group.rows] == \
        [True, False]
    assert [r.cells[at["voxel_instance.voi_split"]].best for r in group.rows] == [False, True]


@pytest.mark.unit
def test_an_empty_task_reads_the_same_in_both():
    table = build("t", [], "https://example.org/t.html", None)
    content = read_markdown(leaderboard.markdown(table))
    assert content == read_page(page.section(table))["t"]
    assert content["note"] and content["empty"] and content["views"] is None


@pytest.mark.unit
def test_every_metric_declares_the_direction_and_meaning_of_its_table_keys():
    """The page's arrows, highlight and glossary come from `key_info`, so every key a metric can
    put in a table needs one, and its ranking key must agree with `higher_is_better`."""
    import components  # noqa: F401  (populates the registry)
    from metrics.registry import MetricRegistry

    for name in MetricRegistry.available():
        metric = MetricRegistry.get(name)
        keys = (metric.primary, *metric.report_keys)
        assert not [k for k in keys if k not in metric.key_info], name
        assert metric.key_info[metric.primary].higher_is_better is metric.higher_is_better, name
        assert all(info.description for info in metric.key_info.values()), name
