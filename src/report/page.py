"""One self-contained, sortable HTML page of every task's leaderboard: `leaderboard/index.html`.

Rendered from records alone, like the markdown tables, and with the same grouping rule: rows scored
over different regions are never in one table. Every metric column sorts on click; the ranking
metric is the initial order.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from .common import cell, region_key, report_keys, truth_kind, views_link
from .links import (
    MISSING,
    artifact_directory,
    checkpoint_directory,
    default_shares,
    fileglancer_url,
    known_missing,
)
from .record import Submission, load_task, task_names

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
 .mono, .run, .sfx, dl.legend dt { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas,
  monospace; }
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
 .views { margin: 0 0 20px; }
 .meta { margin: 0 0 20px; color: var(--muted); }
 .about { margin: 0 0 10px; max-width: 90ch; color: var(--muted); }
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
 th.asc::after { content: " \25B2"; font-size: 9px; } th.desc::after { content: " \25BC";
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
  below;
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


#: Per metric column, keyed by name without its metric prefix: (↑ higher / ↓ lower is better, short
#: description).
METRICS = {
    "pq": ("↑",
        "Panoptic quality, sq * rq: how well predicted"
        " objects match true ones at IoU > 0.5."),
    "sq": ("↑", "Segmentation quality: mean IoU of the matched object pairs."),
    "rq": ("↑",
        "Recognition quality: F1 of object matching,"
        " balancing missed and spurious objects."),
    "voi_split": ("↓",
        "Variation of information, split term"
        " H(prediction | truth): over-segmentation."),
    "voi_merge": ("↓",
        "Variation of information, merge term H(truth |"
        " prediction): under-segmentation."),
    "voi_sum": ("↓", "voi_split + voi_merge."),
    "adapted_rand_error": ("↓", "Adapted Rand error, 1 - Rand F-score of the voxel pairing."),
    "nerl": ("↑",
        "Normalised expected run length: ERL along the traced"
        " skeletons divided by the maximum possible."),
    "erl_um": ("↑", "Expected run length in micrometres."),
    "n_non0_mergers": ("↓",
        "Number of (merging segment, skeleton) pairs where"
        " one segment spans several skeletons."),
    "n_splits": ("↓", "Number of skeleton edges the segmentation cuts."),
    "mean_iou": ("↑", "Mean intersection-over-union over the classes present."),
    "mean_dice": ("↑", "Mean Dice coefficient over the classes present."),
    "pixel_accuracy": ("↑", "Fraction of voxels given the correct class."),
    "classes_present": ("", "Number of classes present in the ground truth."),
}


def _legend(labels: list[str]) -> str:
    items = [f'<div><dt><span class="arrow">{METRICS[k][0]}</span> {escape(k)}</dt>'
             f"<dd>{escape(METRICS[k][1])}</dd></div>"
             for k in dict.fromkeys(label.rpartition(".")[2] for label in labels) if k in METRICS]
    return f'<dl class="legend card">{"".join(items)}</dl>' if items else ""


def _link(name: str, path: Path | None, shares: dict[str, str] | None) -> str | None:
    if path is None:
        return None
    if known_missing(path):
        return f'<span class="sfx">{name} ({MISSING})</span>'
    url = fileglancer_url(path, shares)
    return None if url is None else f'<a class="chip" href="{escape(url)}">{name}</a>'


def _links(submission: Submission, shares: dict[str, str] | None) -> str:
    """Fileglancer links to the row's artifacts and checkpoint; deleted ones are marked missing."""
    producer = submission.producer
    links = [
        _link("artifacts", artifact_directory(producer), shares),
        _link("checkpoint", checkpoint_directory(producer, submission.provenance), shares),
    ]
    return "".join(link for link in links if link) or "—"


def _model_cell(submission: Submission) -> str:
    run, sep, rest = submission.identifier().partition(".step")
    return (f'<td><span class="run">{escape(run)}</span>'
            f'<span class="sfx">{sep}{escape(rest)}</span></td>')


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
            for inner in report_keys(name):
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

    # Same order as the README tables: rank, model, links, ranking metric, postprocess, truth, rest.
    cols = [("n", "#"), ("", "model"), ("", "links"),
            ("n", f"{arrows[0]} {labels[0]}".strip()), ("", "postprocess"),
            *([("", "truth")] if show_truth else []),
            *[("n", f"{a} {lab}".strip()) for a, lab in zip(arrows[1:], labels[1:], strict=True)]]
    rows = []
    for position, submission in enumerate(group, start=1):
        metric_cells = []
        for column in columns:
            name, _, inner = column.partition(".")
            value = submission.scores.get(name, {}).get(inner)
            data = "" if value is None else f' data-v="{value}"'
            top = " best" if value is not None and value == best[column] else ""
            metric_cells.append(f'<td class="n{top}"{data}>{cell(value)}</td>')
        describe = escape(str(submission.postprocess.get("describe", "—")))
        cells = [
            f'<td class="n" data-v="{position}">{position}</td>',
            _model_cell(submission),
            f"<td>{_links(submission, shares)}</td>",
            metric_cells[0],
            f'<td><span class="tag">{describe}</span></td>',
            *([f"<td>{escape(truth_kind(submission))}</td>"] if show_truth else []),
            *metric_cells[1:],
        ]
        rows.append(f'<tr data-id="{escape(submission.identifier())}">' + "".join(cells) + "</tr>")
    header = "".join(f'<th class="{c}">{escape(h)}</th>' for c, h in cols)
    table = (f'<div class="card"><div class="scroll"><table><thead><tr>{header}</tr></thead>'
             f"<tbody>{''.join(rows)}</tbody></table></div></div>")
    return table + '<div style="height:24px"></div>' + _legend(labels)


def render_page(root: Path) -> str:
    shares = default_shares()
    parts, nav = [], []
    for task in task_names(root):
        submissions = load_task(root, task)
        name = escape(task)
        key = escape(str(submissions[0].ranking.get("key", "?"))) if submissions else "—"
        nav.append(f'<a href="#{name}"><b>{name}</b>'
                   f'<small>{len(submissions)} submissions &middot; ranked by {key}</small></a>')
        parts.append(f'<section id="{name}"><h2 class="mono">{name}</h2>')
        if not submissions:
            parts.append('<p class="meta">No records yet.</p></section>')
            continue
        ranking = submissions[0].ranking
        notes = str(submissions[0].config.get("notes") or "").strip()
        paragraphs = [" ".join(p.split()) for p in notes.split("\n\n") if p.strip()]
        parts.extend(f'<p class="about">{escape(p)}</p>' for p in paragraphs)
        parts.append(f'<p class="meta">{len(submissions)} submissions &middot; ranked by '
                     f'<b>{escape(str(ranking.get("key", "?")))}</b> '
                     f'({escape(str(ranking.get("metric", "?")))})</p>')
        views = views_link(root, task)
        if views:
            parts.append(f'<p class="views"><b>Views:</b> <a href="{escape(views)}">'
                         "neuroglancer views for every row below</a></p>")
        by_region: dict[str, list[Submission]] = {}
        for submission in submissions:
            by_region.setdefault(region_key(submission), []).append(submission)
        show_truth = len({truth_kind(s) for s in submissions}) > 1
        for region in sorted(by_region):
            items = "".join(f"<li>{escape(v)}</li>" for v in region.split("; "))
            parts.append(f'<div class="region">Scored region:<ul>{items}</ul></div>')
            parts.append(_group_table(by_region[region], shares, show_truth))
        parts.append("</section>")
    return PAGE.replace("__NAV__", "".join(nav)).replace("__BODY__", "\n".join(parts))
