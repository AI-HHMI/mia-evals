"""One self-contained, sortable HTML page holding every task's leaderboard: `leaderboard/index.html`.

Rendered from records alone, like the markdown tables, and with the same grouping rule: rows scored
over different regions are never in one table. Every metric column sorts on click; the ranking
metric is the initial order.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

from .leaderboard import _cell, _region_key, _report_keys, _truth_kind
from .links import artifact_directory, checkpoint_directory, default_shares, fileglancer_url
from .record import Submission, load_task, task_names

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<!-- GENERATED FILE -- do not edit by hand. Regenerate with `mia-evals leaderboard`. -->
<title>mia-evals leaderboard</title>
<style>
 :root { --bg: #f6f7f9; --card: #fff; --ink: #14181f; --muted: #5d6675; --line: #e3e6eb;
  --head: #f1f3f6; --hover: #f5f8ff; --accent: #2557d6; --accent-soft: #e8eefc;
  --best: #0a7d4f; --best-soft: #e3f4ec; --shadow: 0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.08); }
 @media (prefers-color-scheme: dark) { :root { --bg: #0e1116; --card: #161b22; --ink: #e6e9ee;
  --muted: #9aa4b2; --line: #262d38; --head: #1b212b; --hover: #1a2230; --accent: #7aa2ff;
  --accent-soft: #1b2842; --best: #4cc38a; --best-soft: #143324; --shadow: none; } }
 * { box-sizing: border-box; }
 body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.5 -apple-system,
  BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif; -webkit-font-smoothing: antialiased; }
 a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; }
 .mono, .run, .sfx, dl.legend dt { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
 header { background: var(--card); border-bottom: 1px solid var(--line); }
 .wrap { max-width: 1400px; margin: 0 auto; padding: 0 24px; }
 header .wrap { padding-top: 28px; }
 h1 { margin: 0; font-size: 26px; font-weight: 700; letter-spacing: -.02em; }
 .lede { margin: 6px 0 20px; color: var(--muted); max-width: 70ch; }
 nav { display: flex; gap: 4px; overflow-x: auto; margin-bottom: -1px; }
 nav a { padding: 10px 14px; border-bottom: 2px solid transparent; color: var(--muted);
  font-weight: 500; white-space: nowrap; }
 nav a:hover { color: var(--ink); text-decoration: none; }
 nav a.on { color: var(--accent); border-color: var(--accent); }
 nav a small { margin-left: 6px; padding: 1px 7px; border-radius: 999px; background: var(--head);
  color: var(--muted); font-size: 11px; font-weight: 600; }
 nav a.on small { background: var(--accent-soft); color: var(--accent); }
 main { padding-top: 28px; padding-bottom: 56px; }
 h2 { margin: 0 0 4px; font-size: 20px; font-weight: 650; letter-spacing: -.01em; }
 .meta { margin: 0 0 20px; color: var(--muted); }
 .region { margin: 28px 0 10px; color: var(--muted); font-size: 13px; }
 .region b { color: var(--ink); font-weight: 600; }
 .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  box-shadow: var(--shadow); }
 .scroll { overflow: auto; max-height: 75vh; border-radius: 12px; }
 table { border-collapse: separate; border-spacing: 0; width: 100%; }
 th, td { padding: 10px 14px; text-align: left; white-space: nowrap; border-bottom: 1px solid var(--line); }
 th { position: sticky; top: 0; z-index: 1; background: var(--head); color: var(--muted); font-size: 12px;
  font-weight: 600; letter-spacing: .02em; cursor: pointer; user-select: none; }
 th:hover { color: var(--ink); }
 th.asc::after { content: " \25B2"; font-size: 9px; } th.desc::after { content: " \25BC"; font-size: 9px; }
 tbody tr:last-child td { border-bottom: 0; } tbody tr:hover td { background: var(--hover); }
 td.n { text-align: right; font-variant-numeric: tabular-nums; }
 th.n { text-align: right; }
 td.best { color: var(--best); font-weight: 650; }
 td.date { color: var(--muted); font-variant-numeric: tabular-nums; }
 .run { font-size: 12.5px; } .sfx { font-size: 12px; color: var(--muted); }
 .tag { display: inline-block; padding: 1px 8px; border-radius: 6px; background: var(--head);
  font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
 .chip { display: inline-block; margin-right: 4px; padding: 2px 9px; border: 1px solid var(--line);
  border-radius: 999px; font-size: 12px; color: var(--accent); }
 .chip:hover { background: var(--accent-soft); text-decoration: none; }
 .chip.view { color: var(--muted); } .chip.view:hover { color: var(--accent); }
 figure { margin: 24px 0 0; padding: 18px 20px 12px; } figure svg { display: block; width: 100%; height: auto; }
 figcaption { font-weight: 600; margin-bottom: 4px; } figcaption span { color: var(--muted); font-weight: 400; }
 svg text { fill: var(--muted); font-size: 11px; } svg .grid { stroke: var(--line); }
 svg .frontier { fill: none; stroke: var(--accent); stroke-width: 2; }
 svg .dot { fill: var(--card); stroke: var(--muted); stroke-width: 1.5; opacity: .85; }
 svg .dot.front { fill: var(--accent); stroke: var(--card); stroke-width: 2; opacity: 1; }
 dl.legend { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 14px 32px;
  margin: 0; padding: 20px; }
 dl.legend div { display: grid; grid-template-columns: 7.5em 1fr; gap: 0 12px; }
 dl.legend dt { font-size: 12.5px; font-weight: 600; } dl.legend dd { margin: 0; color: var(--muted); }
 .arrow { color: var(--accent); }
 .foot { margin-top: 32px; color: var(--muted); font-size: 12.5px; }
 @media (max-width: 640px) { .wrap { padding: 0 14px; } h1 { font-size: 22px; }
  dl.legend { grid-template-columns: 1fr; } }
</style></head><body>
<header><div class="wrap">
 <h1>mia-evals leaderboard</h1>
 <p class="lede">Model predictions scored on volumetric segmentation benchmarks. Select a benchmark below;
 click any column header to sort.</p>
 <nav>__NAV__</nav>
</div></header>
<main class="wrap">
__BODY__
<p class="foot">Generated from the records under <span class="mono">leaderboard/</span> by
<span class="mono">mia-evals leaderboard</span>. Tables are per scored region and are not comparable
with one another. Fileglancer and neuroglancer links only work on the Janelia network.</p>
</main>
<script>
const sections = [...document.querySelectorAll("section")];
function show() {                                          // one benchmark at a time, chosen by #hash
  const id = decodeURIComponent(location.hash.slice(1)) || sections[0].id;
  for (const s of sections) s.hidden = s.id !== id;
  for (const a of document.querySelectorAll("nav a")) a.classList.toggle("on", a.hash.slice(1) === id);
}
addEventListener("hashchange", show); show();
for (const th of document.querySelectorAll("th")) th.onclick = () => {
  const table = th.closest("table"), col = th.cellIndex;
  const dir = th.classList.contains("desc") ? 1 : -1;      // first click is descending
  for (const h of table.querySelectorAll("th")) h.classList.remove("asc", "desc");
  th.classList.add(dir > 0 ? "asc" : "desc");
  const key = row => { const v = row.cells[col].dataset.v; return v === undefined ? row.cells[col].textContent : +v; };
  const rows = [...table.tBodies[0].rows];
  rows.sort((a, b) => { const x = key(a), y = key(b); return dir * (x < y ? -1 : x > y ? 1 : 0); });
  rows.forEach(r => table.tBodies[0].append(r));
};
</script>
</body></html>
"""


#: Per metric column, keyed by name without its metric prefix: (↑ higher / ↓ lower is better, short
#: description).
METRICS = {
    "pq": ("↑", "Panoptic quality, sq * rq: how well predicted objects match true ones at IoU > 0.5."),
    "sq": ("↑", "Segmentation quality: mean IoU of the matched object pairs."),
    "rq": ("↑", "Recognition quality: F1 of object matching, balancing missed and spurious objects."),
    "voi_split": ("↓", "Variation of information, split term H(prediction | truth): over-segmentation."),
    "voi_merge": ("↓", "Variation of information, merge term H(truth | prediction): under-segmentation."),
    "voi_sum": ("↓", "voi_split + voi_merge."),
    "adapted_rand_error": ("↓", "Adapted Rand error, 1 - Rand F-score of the voxel pairing."),
    "nerl": ("↑", "Normalised expected run length: ERL along the traced skeletons divided by the maximum possible."),
    "erl_um": ("↑", "Expected run length in micrometres."),
    "n_non0_mergers": ("↓", "Number of nodes where the segmentation merges distinct skeletons."),
    "n_splits": ("↓", "Number of skeleton edges the segmentation cuts."),
    "mean_iou": ("↑", "Mean intersection-over-union over the classes present."),
    "mean_dice": ("↑", "Mean Dice coefficient over the classes present."),
    "pixel_accuracy": ("↑", "Fraction of voxels given the correct class."),
    "classes_present": ("", "Number of classes present in the ground truth."),
}


def _scored_time(submission: Submission) -> datetime | None:
    """When the record was scored, or None if unknown."""
    return datetime.fromisoformat(submission.scored_at) if submission.scored_at else None


def _plot(group: list[Submission], label: str, higher: bool) -> str:
    """Ranking score against scoring date, with the best-so-far frontier, as inline SVG."""
    points = sorted((t, s.ranking["value"], s.identifier()) for s in group if (t := _scored_time(s)))
    if len(points) < 2:
        return ""
    width, height, left, right, top, bottom = 900, 320, 56, 16, 14, 32
    t0, t1 = points[0][0], points[-1][0]
    values = [p[1] for p in points]
    pad = (max(values) - min(values)) * 0.08 or 0.01
    v0, v1 = min(values) - pad, max(values) + pad
    x = lambda t: left + (width - left - right) * (((t - t0) / (t1 - t0)) if t1 != t0 else 0.5)
    y = lambda v: height - bottom - (height - top - bottom) * (v - v0) / (v1 - v0)

    grid = "".join(
        f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
        f'<text x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>'
        for v in (v0 + (v1 - v0) * i / 4 for i in range(5))
    )
    ticks = "".join(
        f'<text x="{x(t):.1f}" y="{height - 10}" text-anchor="{anchor}">{t:%b %d}</text>'
        for t, anchor in ((t0 + (t1 - t0) * i / 3, a) for i, a in
                          ((0, "start"), (1, "middle"), (2, "middle"), (3, "end")))
    )
    best, frontier, path = None, set(), []
    for i, (t, v, _) in enumerate(points):
        if best is None or (v > best if higher else v < best):
            best = v
            frontier.add(i)
        path.append(f"{x(t):.1f},{y(best):.1f}")
    path.append(f"{x(t1):.1f},{y(best):.1f}")
    dots = "".join(
        f'<circle class="dot{" front" if i in frontier else ""}" cx="{x(t):.1f}" cy="{y(v):.1f}" '
        f'r="{5 if i in frontier else 4}"><title>{escape(name)}: {v:.4f} ({t:%Y-%m-%d})</title></circle>'
        for i, (t, v, name) in enumerate(points)
    )
    return (
        f'<figure class="card"><figcaption>{escape(label)} over time '
        f'<span>&mdash; line: best so far</span></figcaption>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(label)} by scoring date">'
        f'{grid}{ticks}<polyline class="frontier" points="{" ".join(path)}"/>{dots}</svg></figure>'
    )


def _legend(labels: list[str]) -> str:
    items = [f'<div><dt><span class="arrow">{METRICS[k][0]}</span> {escape(k)}</dt>'
             f"<dd>{escape(METRICS[k][1])}</dd></div>"
             for k in dict.fromkeys(label.rpartition(".")[2] for label in labels) if k in METRICS]
    return f'<dl class="legend card">{"".join(items)}</dl>' if items else ""


def _link(name: str, path: Path | None, shares: dict[str, str] | None) -> str | None:
    url = None if path is None else fileglancer_url(path, shares)
    return None if url is None else f'<a class="chip" href="{escape(url)}">{name}</a>'


def _links(submission: Submission, shares: dict[str, str] | None) -> str:
    """Fileglancer links to the row's artifacts and checkpoint, then its neuroglancer views."""
    links = [
        _link("artifacts", artifact_directory(submission.producer), shares),
        _link("checkpoint", checkpoint_directory(submission.producer, submission.provenance), shares),
    ]
    for volume, view in sorted(submission.views.items()):
        links.append(f'<a class="chip view" title="neuroglancer view" '
                     f'href="{escape(str(view["overlay"]))}">{escape(volume)}</a>')
    return "".join(link for link in links if link) or "—"


def _date_cell(submission: Submission) -> str:
    time = _scored_time(submission)
    if time is None:
        return '<td class="date">—</td>'
    return f'<td class="date" data-v="{time.timestamp()}">{time:%Y-%m-%d}</td>'


def _model_cell(submission: Submission) -> str:
    run, sep, rest = submission.identifier().partition(".step")
    return f'<td><span class="run">{escape(run)}</span><span class="sfx">{sep}{escape(rest)}</span></td>'


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

    bare = [c.partition(".")[2] for c in columns]
    labels = [b if bare.count(b) == 1 else c for b, c in zip(bare, columns, strict=True)]
    arrows = [METRICS.get(label.rpartition(".")[2], ("",))[0] for label in labels]

    # The best value in each metric column is highlighted, in that metric's own direction.
    best = {}
    for column, arrow in zip(columns, arrows, strict=True):
        name, _, inner = column.partition(".")
        found = [s.scores[name][inner] for s in group if inner in s.scores.get(name, {})]
        best[column] = (max(found) if arrow == "↑" else min(found)) if found and arrow else None

    cols = [("date", "scored")]
    cols += [("n", f"{a} {label}".strip()) for a, label in zip(arrows, labels, strict=True)]
    cols += [("", "model"), ("", "postprocess"), *([("", "truth")] if show_truth else []),
             ("", "links")]
    rows = []
    for submission in group:
        cells = [_date_cell(submission)]
        for column in columns:
            name, _, inner = column.partition(".")
            value = submission.scores.get(name, {}).get(inner)
            data = "" if value is None else f' data-v="{value}"'
            top = " best" if value is not None and value == best[column] else ""
            cells.append(f'<td class="n{top}"{data}>{_cell(value)}</td>')
        cells += [
            _model_cell(submission),
            f'<td><span class="tag">{escape(str(submission.postprocess.get("describe", "—")))}</span></td>',
            *([f"<td>{escape(_truth_kind(submission))}</td>"] if show_truth else []),
            f"<td>{_links(submission, shares)}</td>",
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "".join(f'<th class="{c}">{escape(h)}</th>' for c, h in cols)
    table = (f'<div class="card"><div class="scroll"><table><thead><tr>{header}</tr></thead>'
             f"<tbody>{''.join(rows)}</tbody></table></div></div>")
    plot = _plot(group, labels[0], higher) if len(group) > 10 else ""
    return table + plot + '<div style="height:24px"></div>' + _legend(labels)


def render_page(root: Path) -> str:
    shares = default_shares()
    parts, nav = [], []
    for task in task_names(root):
        submissions = load_task(root, task)
        name = escape(task)
        nav.append(f'<a href="#{name}">{name}<small>{len(submissions)}</small></a>')
        parts.append(f'<section id="{name}"><h2 class="mono">{name}</h2>')
        if not submissions:
            parts.append('<p class="meta">No records yet.</p></section>')
            continue
        ranking = submissions[0].ranking
        parts.append(f'<p class="meta">{len(submissions)} submissions &middot; ranked by '
                     f'<b>{escape(str(ranking.get("key", "?")))}</b> '
                     f'({escape(str(ranking.get("metric", "?")))})</p>')
        by_region: dict[str, list[Submission]] = {}
        for submission in submissions:
            by_region.setdefault(_region_key(submission), []).append(submission)
        show_truth = len({_truth_kind(s) for s in submissions}) > 1
        for region in sorted(by_region):
            parts.append(f'<p class="region">Scored region: <b>{escape(region)}</b></p>')
            parts.append(_group_table(by_region[region], shares, show_truth))
        parts.append("</section>")
    return PAGE.replace("__NAV__", "".join(nav)).replace("__BODY__", "\n".join(parts))
