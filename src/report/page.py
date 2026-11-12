"""One self-contained, sortable HTML page of every task's leaderboard: `leaderboard/index.html`.

Each task's section shows exactly what the task's README.md shows -- both are `table.build`'s
content, written once as markdown (`leaderboard.markdown`) and once here -- with presentation on
top: sortable columns, each column's best value highlighted in the metric's own direction, a
region list that folds away, and a glossary of the metric keys.

One task is shown at a time, the one the URL's `#task` names. Without one the page shows the list of
benchmarks (`index`), and above a task that same list is the menu its header's one button opens,
so the header is one row however many tasks there are.
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from .common import views_link
from .links import Link, default_shares
from .record import load_task, task_names
from .table import (
    LINKS_NOTE,
    MIXED_REGIONS,
    NO_RECORDS,
    REGION_LABEL,
    VIEWS_TEXT,
    Cell,
    Column,
    Group,
    TaskTable,
    build,
)

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport"
 content="width=device-width, initial-scale=1">
<!-- GENERATED FILE -- do not edit by hand. Regenerate with `mia-evals leaderboard`. -->
<title>mia-evals leaderboard</title>
<style>
 :root { --bg: #f6f7f9; --card: #fff; --ink: #14181f; --muted: #5d6675; --line: #e3e6eb;
  --head: #f1f3f6; --hover: #f5f8ff; --accent: #2557d6; --accent-soft: #e8eefc;
  --best: #0a7d4f; --best-soft: #e3f4ec; --shadow: 0 1px 2px rgba(16,24,40,.06), 0 1px 3px
   rgba(16,24,40,.08); --pop: 0 12px 32px rgba(16,24,40,.18); }
 @media (prefers-color-scheme: dark) { :root { --bg: #0e1116; --card: #161b22; --ink: #e6e9ee;
  --muted: #9aa4b2; --line: #262d38; --head: #1b212b; --hover: #1a2230; --accent: #7aa2ff;
  --accent-soft: #1b2842; --best: #4cc38a; --best-soft: #143324; --shadow: none;
  --pop: 0 12px 32px rgba(0,0,0,.6); } }
 * { box-sizing: border-box; }
 body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.5 -apple-system,
  BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif; -webkit-font-smoothing: antialiased; }
 a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; }
 .mono, .run, .sfx, code, dl.legend dt, .group b { font-family: ui-monospace, SFMono-Regular,
  Menlo, Consolas, monospace; }
 header { background: var(--card); border-bottom: 1px solid var(--line); }
 .wrap { max-width: 1400px; margin: 0 auto; padding: 0 24px; }
 header .wrap { padding-top: 28px; padding-bottom: 24px; }
 .bar { display: flex; align-items: center; gap: 16px; }
 h1 { flex: 1; margin: 0; font-size: 26px; font-weight: 700; letter-spacing: -.02em; }
 h1 a { color: inherit; } h1 a:hover { text-decoration: none; }
 .lede { margin: 6px 0 0; color: var(--muted); max-width: 70ch; }
 #menu { display: none; align-items: center; gap: 6px; padding: 6px 12px; border: 1px solid
  var(--line); border-radius: 8px; background: var(--card); color: var(--ink); font: inherit;
  font-size: 13px; font-weight: 600; cursor: pointer; }
 #menu::after { content: "\\25BE"; color: var(--muted); }
 #menu:hover, #menu[aria-expanded="true"] { border-color: var(--accent); color: var(--accent); }
 nav { max-width: 1400px; margin: 0 auto; padding: 28px 24px 0; }
 .groups { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(340px, 100%), 1fr));
  gap: 16px; align-items: start; }
 .group { overflow: hidden; }
 .group h2 { display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
  margin: 0; padding: 12px 14px; font-size: 14px; letter-spacing: 0; }
 .group small { color: var(--muted); font-size: 12px; font-weight: 400; white-space: nowrap; }
 .group ul { margin: 0; padding: 0; list-style: none; }
 .group a { display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
  padding: 7px 14px; border-top: 1px solid var(--line); color: var(--ink); }
 .group b { font-size: 12.5px; font-weight: 500; overflow-wrap: anywhere; }
 .group a:hover { background: var(--hover); text-decoration: none; }
 .group a:hover b, .group a.on b { color: var(--accent); }
 .group a.on { background: var(--accent-soft); }
 /* a benchmark is open: the header is one row, and the list is the menu its button opens */
 .board header .wrap { padding-top: 12px; padding-bottom: 12px; }
 .board h1 { font-size: 17px; letter-spacing: -.01em; }
 .board h1 a:hover { color: var(--accent); }
 .board .lede { display: none; }
 .board #menu { display: inline-flex; }
 .board nav { display: none; position: absolute; left: 0; right: 0; z-index: 10; padding-top: 8px; }
 .board nav.open { display: block; }
 .board .groups { gap: 6px 16px; padding: 6px; max-height: calc(100vh - 80px); overflow: auto;
  background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  box-shadow: var(--pop); }
 .board .group { border: 0; box-shadow: none; background: none; }
 main.wrap { padding-bottom: 56px; }
 section { padding-top: 32px; scroll-margin-top: 120px; }
 h2 { margin: 0 0 4px; font-size: 20px; font-weight: 650; letter-spacing: -.01em; }
 .note, .empty { margin: 0 0 16px; color: var(--muted); }
 .views { margin: 0 0 20px; }
 .mixed { margin: 0 0 20px; padding: 10px 14px; border-left: 3px solid var(--line);
  color: var(--muted); }
 .region { margin: 28px 0 10px; color: var(--muted); font-size: 13px; }
 .region summary { width: fit-content; cursor: pointer; user-select: none; }
 .region summary:hover { color: var(--ink); }
 .region ul { margin: 4px 0 0; padding-left: 20px; color: var(--ink); }
 .region ul.cols { columns: 300px; column-gap: 32px; } .region ul.cols li { break-inside: avoid; }
 .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  box-shadow: var(--shadow); }
 .scroll { overflow: auto; max-height: 75vh; border-radius: 12px; }
 table { border-collapse: separate; border-spacing: 0; width: 100%; }
 th, td { padding: 10px 14px; text-align: left; white-space: nowrap; border-bottom: 1px solid
  var(--line); }
 th { position: sticky; top: 0; z-index: 1; background: var(--head); color: var(--muted);
  font-size: 12px;
  font-weight: 600; letter-spacing: .02em; cursor: pointer; user-select: none; }
 th:hover { color: var(--ink); }
 th .pre { display: block; font-weight: 400; }
 th.asc::after { content: " \\25B2"; font-size: 9px; } th.desc::after { content: " \\25BC";
  font-size: 9px; }
 tbody tr:last-child td { border-bottom: 0; } tbody tr:hover td { background: var(--hover); }
 td.n { text-align: right; font-variant-numeric: tabular-nums; }
 th.n { text-align: right; }
 td.best { color: var(--best); font-weight: 650; }
 .run { font-size: 12.5px; } .sfx { font-size: 12px; color: var(--muted); }
 .tag { display: inline-block; padding: 1px 8px; border-radius: 6px; background: var(--head);
  font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
 .chip { display: inline-block; margin-right: 4px; padding: 2px 9px; border: 1px solid var(--line);
  border-radius: 999px; font-size: 12px; color: var(--accent); }
 .chip:hover { background: var(--accent-soft); text-decoration: none; }
 .missing, .nolink { margin-right: 8px; font-size: 12px; color: var(--muted); }
 .legend-gap { height: 24px; }
 dl.legend { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 14px
  32px;
  margin: 0; padding: 20px; }
 dl.legend dt { margin-bottom: 2px; font-size: 12.5px; font-weight: 600; overflow-wrap: anywhere; }
  dl.legend dd { margin: 0; color: var(--muted); }
 .arrow { color: var(--accent); }
 .foot { margin-top: 32px; color: var(--muted); font-size: 12.5px; }
 @media (max-width: 640px) { .wrap { padding: 0 14px; } h1 { font-size: 22px; }
  nav { padding-left: 14px; padding-right: 14px; } dl.legend { grid-template-columns: 1fr; } }
</style></head><body>
<header><div class="wrap">
 <div class="bar"><h1><a href="#">mia-evals leaderboard</a></h1>
 <button id="menu" type="button" aria-expanded="false" aria-controls="benchmarks">All benchmarks
 </button></div>
 <p class="lede">Model predictions scored on volumetric segmentation benchmarks.</p>
</div></header>
<nav id="benchmarks" aria-label="Benchmarks"><div class="groups">
__NAV__
</div></nav>
<main class="wrap">
__BODY__
<p class="foot">Generated from the records under <span class="mono">leaderboard/</span> by
<span class="mono">mia-evals leaderboard</span>, from the same tables as each task's README.md.</p>
</main>
<script>
const sections = [...document.querySelectorAll("section")];
const nav = document.querySelector("nav"), menu = document.getElementById("menu");
const drop = on => { nav.classList.toggle("open", on); menu.setAttribute("aria-expanded", on); };
// the one benchmark the #hash names; with an unknown or empty hash, the list of benchmarks
function show() {
  const id = decodeURIComponent(location.hash.slice(1));
  const board = sections.some(s => s.id === id);
  for (const s of sections) s.hidden = s.id !== id;
  for (const a of nav.querySelectorAll("a"))
   a.classList.toggle("on", board && a.hash.slice(1) === id);
  document.body.classList.toggle("board", board);
  drop(false);
}
addEventListener("hashchange", show); show();
menu.onclick = () => drop(!nav.classList.contains("open"));
nav.onclick = e => { if (e.target.closest("a")) { drop(false); scrollTo(0, 0); } };
addEventListener("click", e => { if (!e.target.closest(".groups, #menu")) drop(false); });
addEventListener("keydown", e => {
  if (e.key === "Escape" && nav.classList.contains("open")) { drop(false); menu.focus(); }
});
for (const th of document.querySelectorAll("th")) th.title = "Click to sort";
for (const th of document.querySelectorAll("th")) th.onclick = () => {
  const table = th.closest("table"), col = th.cellIndex;
  const dir = th.classList.contains("desc") ? 1 : -1;      // first click is descending
  for (const h of table.querySelectorAll("th")) h.classList.remove("asc", "desc");
  th.classList.add(dir > 0 ? "asc" : "desc");
  const key = row => { const v = row.cells[col].dataset.v; return v === undefined ?
   row.cells[col].textContent : +v; };
  const rows = [...table.tBodies[0].rows];
  rows.sort((a, b) => { const x = key(a), y = key(b); return dir * (x < y ? -1 : x > y ? 1 : 0); });
  rows.forEach(r => table.tBodies[0].append(r));
};
</script>
</body></html>
"""

ARROWS = {True: "↑", False: "↓", None: ""}

#: A table's region list is one volume per row up to this many volumes, and set in columns above
#: it: the CellMap task scores 43 crops.
REGION_ROWS = 10

#: What the list of benchmarks calls the tasks ranked on each metric, in the order it shows them.
#: Tasks ranked on any other metric follow, under the metric's own name.
KINDS = {
    "voxel_instance": "Instance segmentation",
    "skeleton_erl": "Neurite tracing",
    "semantic": "Semantic segmentation",
}


def _bold(text: str) -> str:
    """Escaped `text` with markdown's `**strong**` as HTML."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(text))


def _header(column: Column) -> str:
    """The README's header text; a metric column's `<metric>.` prefix sits on its own line."""
    numeric = ' class="n"' if column.kind in ("rank", "metric") else ""
    prefix, dot, rest = column.header.partition(".")
    if column.kind == "metric" and dot:
        return f'<th{numeric}><span class="pre">{escape(prefix + dot)}</span>{escape(rest)}</th>'
    return f"<th{numeric}>{escape(column.header)}</th>"


def _links(links: tuple[Link, ...]) -> str:
    """The README's links entries: a fileglancer link, `name (missing)`, or the bare path."""
    if not links:
        return "—"
    out = []
    for link in links:
        if link.missing:
            out.append(f'<span class="missing">{escape(link.label)}</span>')
        elif link.url is None:
            out.append(f'<span class="nolink">{escape(link.name)}: '
                       f"<code>{escape(link.path)}</code></span>")
        else:
            out.append(f'<a class="chip" href="{escape(link.url)}">{escape(link.name)}</a>')
    return "".join(out)


def _cell(column: Column, cell: Cell) -> str:
    if column.kind == "model":
        run, sep, rest = cell.text.partition(".step")
        return (f'<td><span class="run">{escape(run)}</span>'
                f'<span class="sfx">{escape(sep + rest)}</span></td>')
    if column.kind == "links":
        return f"<td>{_links(cell.links)}</td>"
    if column.kind == "postprocess":
        return f'<td><span class="tag">{escape(cell.text)}</span></td>'
    if column.kind in ("rank", "metric"):
        data = "" if cell.value is None else f' data-v="{cell.value}"'
        best = " best" if cell.best else ""
        return f'<td class="n{best}"{data}>{escape(cell.text)}</td>'
    return f"<td>{escape(cell.text)}</td>"


def _legend(columns: tuple[Column, ...]) -> str:
    """The metric keys of one table, each with its direction and the metric's description."""
    items, seen = [], set()
    for column in columns:
        if column.kind != "metric" or not column.description:
            continue
        if (column.metric, column.key) in seen:
            continue
        seen.add((column.metric, column.key))
        items.append(f'<div><dt><span class="arrow">{ARROWS[column.higher_is_better]}</span> '
                     f"{escape(column.key)}</dt><dd>{escape(column.description)}</dd></div>")
    if not items:
        return ""
    return f'<div class="legend-gap"></div><dl class="legend card">{"".join(items)}</dl>'


def _group(group: Group) -> str:
    volumes = group.region.split("; ")
    items = "".join(f"<li>{escape(volume)}</li>" for volume in volumes)
    columns = ' class="cols"' if len(volumes) > REGION_ROWS else ""
    header = "".join(_header(column) for column in group.columns)
    rows = "".join(
        f'<tr data-id="{escape(row.identifier)}">'
        + "".join(_cell(column, c) for column, c in zip(group.columns, row.cells, strict=True))
        + "</tr>"
        for row in group.rows
    )
    # <details open>: the list can be folded away under its label, and starts unfolded.
    return (f'<details class="region" open><summary>{escape(REGION_LABEL)}</summary>'
            f"<ul{columns}>{items}</ul></details>"
            f'<div class="card"><div class="scroll"><table><thead><tr>{header}</tr></thead>'
            f"<tbody>{rows}</tbody></table></div></div>" + _legend(group.columns))


def section(table: TaskTable) -> str:
    """`table` as the task's section of the page; `leaderboard.markdown` writes it as README.md."""
    name = escape(table.name)
    parts = [f'<section id="{name}"><h2 class="mono">{name}</h2>',
             f'<p class="note">{escape(LINKS_NOTE)}</p>']
    if not table.groups:
        parts.append(f'<p class="empty">{escape(NO_RECORDS)}</p>')
    else:
        if table.views:
            parts.append(f'<p class="views"><b>Views:</b> <a href="{escape(table.views)}">'
                         f"{escape(VIEWS_TEXT)}</a></p>")
        if table.mixed_regions:
            parts.append(f'<p class="mixed">{_bold(MIXED_REGIONS)}</p>')
        parts.extend(_group(group) for group in table.groups)
    parts.append("</section>")
    return "\n".join(parts)


def _metric(table: TaskTable) -> str:
    """The metric a task ranks on, its tables' first metric column; empty if it has no records."""
    for group in table.groups:
        for column in group.columns:
            if column.kind == "metric":
                return column.metric
    return ""


def index(tables: list[TaskTable]) -> str:
    """The list of benchmarks: every task once, with its number of entries, under what it ranks on.

    Shown as the page itself when no task is open, and as the menu above an open one. The groups
    are the `KINDS`, then any other metric, then the tasks without records.
    """
    by_ranking: dict[tuple[str, str], list[TaskTable]] = {}
    for table in tables:
        by_ranking.setdefault((_metric(table), table.ranking_key), []).append(table)
    kinds = list(KINDS)

    def position(ranking: tuple[str, str]) -> tuple[bool, int, tuple[str, str]]:
        metric = ranking[0]
        return not metric, kinds.index(metric) if metric in kinds else len(kinds), ranking

    groups = []
    for metric, key in sorted(by_ranking, key=position):
        title = KINDS.get(metric, metric) if metric else NO_RECORDS.rstrip(".")
        ranked = f"<small>ranked by {escape(key)}</small>" if metric else ""
        # <wbr>: a name too long for its column wraps after an underscore, not inside a word.
        items = "".join(
            f'<li><a href="#{escape(table.name)}"><b>{escape(table.name).replace("_", "_<wbr>")}'
            f"</b><small>{table.entries} {'entry' if table.entries == 1 else 'entries'}</small>"
            "</a></li>"
            for table in by_ranking[metric, key]
        )
        groups.append(f'<div class="group card"><h2>{escape(title)}{ranked}</h2>'
                      f"<ul>{items}</ul></div>")
    return "\n".join(groups)


def render_page(root: Path) -> str:
    """The list of benchmarks and every task's section, each section from the same `table.build`
    as that task's README.md."""
    shares = default_shares()
    tables = [build(task, load_task(root, task), views_link(root, task), shares)
              for task in task_names(root)]
    return (PAGE.replace("__NAV__", index(tables))
            .replace("__BODY__", "\n".join(section(table) for table in tables)))
