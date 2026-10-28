"""One self-contained, sortable HTML page of every task's leaderboard: `leaderboard/index.html`.

Each task's section shows exactly what the task's README.md shows -- both are `table.build`'s
content, written once as markdown (`leaderboard.markdown`) and once here -- with presentation on
top: the task cards to switch between tasks, sortable columns, each column's best value highlighted
in the metric's own direction, and a glossary of the metric keys.
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
   rgba(16,24,40,.08); }
 @media (prefers-color-scheme: dark) { :root { --bg: #0e1116; --card: #161b22; --ink: #e6e9ee;
  --muted: #9aa4b2; --line: #262d38; --head: #1b212b; --hover: #1a2230; --accent: #7aa2ff;
  --accent-soft: #1b2842; --best: #4cc38a; --best-soft: #143324; --shadow: none; } }
 * { box-sizing: border-box; }
 body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.5 -apple-system,
  BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif; -webkit-font-smoothing: antialiased; }
 a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; }
 .mono, .run, .sfx, code, dl.legend dt { font-family: ui-monospace, SFMono-Regular, Menlo,
  Consolas, monospace; }
 header { background: var(--card); border-bottom: 1px solid var(--line); }
 .wrap { max-width: 1400px; margin: 0 auto; padding: 0 24px; }
 header .wrap { padding-top: 28px; }
 h1 { margin: 0; font-size: 26px; font-weight: 700; letter-spacing: -.02em; }
 .lede { margin: 6px 0 36px; color: var(--muted); max-width: 70ch; }
 nav { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px;
  padding-bottom: 20px; }
 nav a { display: flex; flex-direction: column; gap: 2px; padding: 10px 14px; border: 1px solid
  var(--line);
  border-radius: 10px; background: var(--card); color: var(--ink); }
 nav a:hover { border-color: var(--accent); text-decoration: none; }
 nav a.on { border-color: var(--accent); background: var(--accent-soft); box-shadow: 0 0 0 1px
  var(--accent); }
 nav b { font: 600 12.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-wrap:
  anywhere; }
 nav small { color: var(--muted); font-size: 12px; }
 nav a.on b { color: var(--accent); }
 main.wrap { padding-bottom: 56px; }
 section { padding-top: 32px; }
 h2 { margin: 0 0 4px; font-size: 20px; font-weight: 650; letter-spacing: -.01em; }
 .note, .empty { margin: 0 0 16px; color: var(--muted); }
 .views { margin: 0 0 20px; }
 .mixed { margin: 0 0 20px; padding: 10px 14px; border-left: 3px solid var(--line);
  color: var(--muted); }
 .region { margin: 28px 0 10px; color: var(--muted); font-size: 13px; }
 .region ul { margin: 4px 0 0; padding-left: 20px; color: var(--ink); }
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
  dl.legend { grid-template-columns: 1fr; } }
</style></head><body>
<header><div class="wrap">
 <h1>mia-evals leaderboard</h1>
 <p class="lede">Model predictions scored on volumetric segmentation benchmarks. Select a benchmark
 below; click any column header to sort.</p>
 <nav>__NAV__</nav>
</div></header>
<main class="wrap">
__BODY__
<p class="foot">Generated from the records under <span class="mono">leaderboard/</span> by
<span class="mono">mia-evals leaderboard</span>, from the same tables as each task's README.md.</p>
</main>
<script>
const sections = [...document.querySelectorAll("section")];
// one benchmark at a time, chosen by #hash
function show() {
  let id = decodeURIComponent(location.hash.slice(1));
  if (!sections.some(s => s.id === id)) id = sections[0].id;   // unknown or empty hash: first task
  for (const s of sections) s.hidden = s.id !== id;
  for (const a of document.querySelectorAll("nav a"))
   a.classList.toggle("on", a.hash.slice(1) === id);
}
addEventListener("hashchange", show); show();
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
    items = "".join(f"<li>{escape(volume)}</li>" for volume in group.region.split("; "))
    header = "".join(_header(column) for column in group.columns)
    rows = "".join(
        f'<tr data-id="{escape(row.identifier)}">'
        + "".join(_cell(column, c) for column, c in zip(group.columns, row.cells, strict=True))
        + "</tr>"
        for row in group.rows
    )
    return (f'<div class="region">{escape(REGION_LABEL)}<ul>{items}</ul></div>'
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


def render_page(root: Path) -> str:
    """Every task's section, each from the same `table.build` as that task's README.md."""
    shares = default_shares()
    nav, parts = [], []
    for task in task_names(root):
        table = build(task, load_task(root, task), views_link(root, task), shares)
        name = escape(task)
        nav.append(f'<a href="#{name}"><b>{name}</b><small>{table.entries} submissions &middot; '
                   f"ranked by {escape(table.ranking_key)}</small></a>")
        parts.append(section(table))
    return PAGE.replace("__NAV__", "".join(nav)).replace("__BODY__", "\n".join(parts))
