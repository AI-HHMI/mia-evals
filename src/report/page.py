"""One self-contained, sortable HTML page holding every task's leaderboard: `leaderboard/index.html`.

Rendered from records alone, like the markdown tables, and with the same grouping rule: rows scored
over different regions are never in one table. Every metric column sorts on click; the ranking
metric is the initial order.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from .leaderboard import _cell, _region_key, _report_keys, _truth_kind
from .links import artifact_directory, checkpoint_directory, default_shares, fileglancer_url
from .record import Submission, load_task, task_names

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<!-- GENERATED FILE -- do not edit by hand. Regenerate with `mia-evals leaderboard`. -->
<title>mia-evals leaderboard</title>
<style>
 body {{ font: 14px/1.4 system-ui, sans-serif; margin: 1.5em; color: #111; background: #fff; }}
 h2 {{ font-family: ui-monospace, monospace; margin-top: 2em; }}
 table {{ border-collapse: collapse; }} td, th {{ padding: .25em .8em; border-bottom: 1px solid #ddd;
 text-align: left; white-space: nowrap; }} td.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
 th {{ cursor: pointer; user-select: none; background: #f4f4f4; }} th.asc::after {{ content: " ▲"; }}
 th.desc::after {{ content: " ▼"; }} a {{ color: #1a5fb4; }} .note {{ color: #555; }}
 @media (prefers-color-scheme: dark) {{
  body {{ color: #eee; background: #181818; }} th {{ background: #262626; }}
  td, th {{ border-color: #333; }} a {{ color: #7ab0ff; }} .note {{ color: #aaa; }}
 }}
</style></head><body>
<h1>mia-evals leaderboard</h1>
<p class="note">Click a column header to sort. Links point at the Janelia cluster and only work on
the Janelia network. Tables are per scored region and are not comparable with one another.</p>
{body}
<script>
for (const th of document.querySelectorAll("th")) th.onclick = () => {{
  const table = th.closest("table"), col = th.cellIndex;
  const dir = th.classList.contains("desc") ? 1 : -1;      // numbers: first click is descending
  for (const h of table.querySelectorAll("th")) h.classList.remove("asc", "desc");
  th.classList.add(dir > 0 ? "asc" : "desc");
  const key = row => {{ const v = row.cells[col].dataset.v; return v === undefined ? row.cells[col].textContent : +v; }};
  const rows = [...table.tBodies[0].rows];
  rows.sort((a, b) => {{ const x = key(a), y = key(b); return dir * (x < y ? -1 : x > y ? 1 : 0); }});
  rows.forEach(r => table.tBodies[0].append(r));
}};
</script>
</body></html>
"""


def _link(name: str, path: Path | None, shares: dict[str, str] | None) -> str | None:
    url = None if path is None else fileglancer_url(path, shares)
    return None if url is None else f'<a href="{escape(url)}">{name}</a>'


def _links(submission: Submission, shares: dict[str, str] | None) -> str:
    """Fileglancer links to the row's artifacts and checkpoint, then its neuroglancer views."""
    links = [
        _link("artifacts", artifact_directory(submission.producer), shares),
        _link("checkpoint", checkpoint_directory(submission.producer, submission.provenance), shares),
    ]
    for volume, view in sorted(submission.views.items()):
        links.append(f'<a href="{escape(str(view["overlay"]))}">{escape(volume)}</a>')
    return " · ".join(link for link in links if link) or "—"


def _group_table(group: list[Submission], shares: dict[str, str] | None, show_truth: bool) -> str:
    ranking = group[0].ranking
    metric, key = ranking.get("metric", "?"), ranking.get("key", "?")
    higher = bool(ranking.get("higher_is_better", True))
    group.sort(key=lambda s: s.ranking.get("value", 0.0), reverse=higher)

    # Every key the metrics report, not only the first six the markdown table keeps.
    ranked = f"{metric}.{key}"
    columns = [ranked]
    for submission in group:
        for name in sorted(submission.scores):
            for inner in _report_keys(name):
                column = f"{name}.{inner}"
                if column not in columns and inner in submission.scores[name]:
                    columns.append(column)

    head = ["model", "links", "postprocess", *(["truth"] if show_truth else []),
            f"{ranked} ({'higher' if higher else 'lower'} is better)", *columns[1:]]
    rows = []
    for submission in group:
        cells = [
            f"<td>{escape(submission.identifier())}</td>",
            f"<td>{_links(submission, shares)}</td>",
            f"<td>{escape(str(submission.postprocess.get('describe', '—')))}</td>",
            *([f"<td>{escape(_truth_kind(submission))}</td>"] if show_truth else []),
        ]
        for column in columns:
            name, _, inner = column.partition(".")
            value = submission.scores.get(name, {}).get(inner)
            data = "" if value is None else f' data-v="{value}"'
            cells.append(f'<td class="n"{data}>{_cell(value)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "".join(f"<th>{escape(h)}</th>" for h in head)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def render_page(root: Path) -> str:
    shares = default_shares()
    parts = []
    for task in task_names(root):
        submissions = load_task(root, task)
        parts.append(f"<h2>{escape(task)}</h2>")
        by_region: dict[str, list[Submission]] = {}
        for submission in submissions:
            by_region.setdefault(_region_key(submission), []).append(submission)
        show_truth = len({_truth_kind(s) for s in submissions}) > 1
        for region in sorted(by_region):
            parts.append(f'<p class="note">Region: {escape(region)}</p>')
            parts.append(_group_table(by_region[region], shares, show_truth))
        if not submissions:
            parts.append('<p class="note">No records yet.</p>')
    return PAGE.format(body="\n".join(parts))
