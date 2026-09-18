"""Single entrypoint: score a prediction artifact against a task, and render the leaderboard.

    mia-evals score  configs/tasks/<task>.toml --test <artifact.zarr> \\
        [--val <artifact.zarr>] [--leaderboard leaderboard/]
    mia-evals leaderboard [--task <task_name>] [--check]

Scoring writes the record *and* re-renders that task's table, so the two never drift apart by a
forgotten second command. `mia-evals leaderboard` rebuilds every task; `--task` rebuilds one.

**Fit on validation, apply to test, and no way around it.** A postprocessor with more than one
candidate in its search space and no `--val` artifact is a hard error, not a default. The
alternative -- sweeping on the artifact being reported and keeping the best -- selects on the number
being published, which is how a threshold sweep turns into an inflated result. `identity` and any
other single-candidate postprocessor need no `--val`, which is what lets a finished segmentation be
scored with no validation data at all.

Nothing here imports torch. Prediction happens in whatever repository owns the model -- for our own
runs, `mia-train/src/predict.py` -- and this reads the artifact it wrote. That is what makes an
external submission an ordinary input rather than a special case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from artifact import Artifact, open_artifact, write_scored
from config import TaskConfig, load_task_config
from metrics.base import BaseMetric
from postprocess.base import BasePostprocess
from report import leaderboard, record
from report.record import Submission, git_commit
from tasks.base import BaseTask, Volume

#: Where this module was installed from. Used only to look up `mia-evals`' own git commit for a
#: record's provenance, which is a property of the source and not of the working directory. There
#: is no `.git` under a non-editable install, and `git_commit` reports "unavailable" for that.
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _leaderboard_root() -> Path:
    """The `leaderboard/` directory the CLI defaults to, for records and the rendered table.

    This used to be `Path(__file__).parents[1] / "leaderboard"` unconditionally, which is correct
    for a source checkout or an editable install but wrong for a plain `pip install .`: there
    `__file__` is `<venv>/lib/pythonX.Y/site-packages/evaluate.py`, so the default became
    `<venv>/lib/pythonX.Y/leaderboard`. `mia-evals leaderboard` then compared a table that did not
    exist, and `mia-evals score` would have written git-tracked records into site-packages.

    Records and the table are version-controlled artifacts of a clone, so the working directory is
    the right fallback. The checkout still wins when this module is running from one, which keeps
    an editable install usable from any directory.
    """
    beside_source = Path(__file__).resolve().parents[1] / "leaderboard"
    return beside_source if beside_source.is_dir() else Path.cwd() / "leaderboard"


DEFAULT_LEADERBOARD = _leaderboard_root()


def build(config: TaskConfig) -> tuple[BaseTask, BasePostprocess, dict[str, BaseMetric]]:
    """Instantiate the task, postprocessor and metrics the config names, and check they compose."""
    import components  # noqa: F401  (populates the registries)
    from metrics.registry import MetricRegistry
    from postprocess.registry import PostprocessRegistry
    from tasks.registry import TaskRegistry

    task = TaskRegistry.build(config.task.name, **config.task.kwargs)
    processor = PostprocessRegistry.build(config.postprocess.name, **config.postprocess.kwargs)
    metric_objects = {
        name: MetricRegistry.build(name, **config.metric_kwargs.get(name, {}))
        for name in config.metrics
    }

    mismatched = {
        name: metric.canonical
        for name, metric in metric_objects.items()
        if metric.canonical != task.canonical
    }
    if mismatched:
        raise ValueError(
            f"task {config.task.name!r} scores a {task.canonical!r} labelling, but metric(s) "
            f"{mismatched} consume something else. A metric cannot be applied to a form it was "
            "not written for."
        )
    needs_scores = sorted(n for n, m in metric_objects.items() if m.consumes == "scores")
    if needs_scores:
        raise NotImplementedError(
            f"metric(s) {needs_scores} consume prediction scores rather than a labelling, which "
            "this runner does not yet pass through. They are declared so that artifact retention "
            "can already be decided correctly; wiring them is the next step."
        )
    return task, processor, metric_objects


def check_compatible(artifact: Artifact, processor: BasePostprocess, task: BaseTask) -> None:
    """Refuse a pairing that would produce a number meaning something other than it claims."""
    if artifact.kind not in processor.accepts:
        raise ValueError(
            f"{type(processor).__name__} accepts artifacts of kind {list(processor.accepts)}, but "
            f"{artifact.path.name} declares kind={artifact.kind!r}. Thresholding class scores as "
            "though they were affinities produces a segmentation rather than an error, which is "
            "why this is checked rather than attempted."
        )
    produced = processor.produces_for(artifact.canonical)
    if produced != task.canonical:
        raise ValueError(
            f"{type(processor).__name__} produces a {produced!r} labelling but the task scores "
            f"{task.canonical!r}"
        )


def check_same_volume(volume: Volume, artifact: Artifact, role: str) -> None:
    """Refuse an artifact predicted over a different store than the volume it is scored as.

    A directory of artifacts is matched to a task's volumes by file name, so a prediction of one
    volume filed under another's name would be scored against the wrong ground truth -- real data
    in the wrong place, plausible numbers, and nothing raises. The producer records where it read
    from (`source_path`; the pre-refactor scripts wrote `cube`), which is enough to catch it by
    name. An artifact from elsewhere, carrying neither attribute, is trusted as before: the
    attribute is provenance, not a requirement.
    """
    declared = artifact.attrs.get("source_path") or artifact.attrs.get("cube")
    if not declared:
        return
    if Path(str(declared)).resolve() != Path(volume.path).resolve():
        raise SystemExit(
            f"{role} artifact {artifact.path} was predicted over\n    {declared}\n"
            f"but is being scored as volume {volume.name!r}, whose data is\n    {volume.path}\n"
            "That would score real data against the wrong ground truth without an error. Check "
            "which split the artifact belongs to: the task's [data.test] volumes go to --test and "
            "its [data.fit] volumes to --val."
        )


def resolve_artifacts(spec: Path, volumes: tuple[Volume, ...]) -> dict[str, Artifact]:
    """Map each volume to its own artifact.

    One prediction per volume, never one artifact for all of them: a prediction over Kasthuri AC4
    says nothing about a zebrafish cube, and reading one where the other was meant would score real
    data from the wrong specimen. `spec` is therefore either a directory holding `<volume>.zarr` per
    volume, or -- only when the task has exactly one volume -- that volume's artifact directly.
    """
    if spec.is_dir() and not (spec / "zarr.json").exists() and not (spec / ".zarray").exists():
        found, missing = {}, []
        for volume in volumes:
            candidate = spec / f"{volume.name}.zarr"
            if candidate.exists():
                found[volume.name] = open_artifact(candidate)
            else:
                missing.append(candidate.name)
        if missing:
            raise SystemExit(
                f"{spec} is missing an artifact for {len(missing)} of {len(volumes)} volume(s): "
                f"{missing}. Every volume in the task's data config needs its own prediction; "
                "scoring a subset silently changes what the reported number covers."
            )
        return found

    if len(volumes) != 1:
        raise SystemExit(
            f"{spec} is a single artifact but this task has {len(volumes)} volumes "
            f"({[v.name for v in volumes]}). Pass a directory containing <volume>.zarr for each."
        )
    return {volumes[0].name: open_artifact(spec)}


def _aggregate(
    per_volume: dict[str, dict[str, dict[str, float]]],
    metric_objects: dict[str, BaseMetric],
) -> dict[str, dict[str, float]]:
    """Per-volume results -> one number per metric key, by each metric's own rule.

    A metric that `accumulates` has already combined the volumes internally (one confusion matrix
    over all of them), so its final return *is* the answer and averaging it again would be wrong.
    Everything else is combined here as an **unweighted mean over volumes**, which is the only
    aggregation consistent with an eval set that weights its volumes equally: size-weighted, the
    8.4-gigavoxel zebrafish cube would be ~80% of the score and the two LICONN blocks ~1%.
    """
    combined: dict[str, dict[str, float]] = {}
    for name, metric in metric_objects.items():
        volumes = [per_volume[v][name] for v in per_volume if name in per_volume[v]]
        if not volumes:
            continue
        if metric.accumulates:
            combined[name] = dict(volumes[-1])
            continue
        keys = sorted({k for entry in volumes for k in entry})
        combined[name] = {
            key: float(np.mean([entry[key] for entry in volumes if key in entry]))
            for key in keys
        }
        combined[name]["volumes_scored"] = float(len(volumes))
    return combined


def score_once(
    artifacts: dict[str, Artifact],
    volumes: tuple[Volume, ...],
    task: BaseTask,
    processor: BasePostprocess,
    metric_objects: dict[str, BaseMetric],
    params: dict[str, Any],
    scratch: Path,
    keep: Path | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, Any], dict[str, Any]]:
    """Postprocess and score every volume under one parameter set.

    Returns (aggregate, per-volume, regions). Both halves are kept: the aggregate is what ranks,
    and the per-volume numbers are what make a bad aggregate diagnosable -- on this eval set one
    modality failing completely and three working looks identical, in the mean, to all four being
    mediocre.

    With `keep`, each volume's post-processed labelling -- the voxels actually scored -- is
    written to `keep/<volume>.zarr` (`artifact.write_scored`) and named in the regions, so a
    viewer can show what the number was computed on rather than the producer's raw output.
    """
    for metric in metric_objects.values():
        # A metric that accumulates must start clean for each candidate, or the second candidate
        # scores against the first one's counts as well as its own.
        if metric.accumulates and hasattr(metric, "reset"):
            metric.reset()

    per_volume: dict[str, dict[str, dict[str, float]]] = {}
    regions: dict[str, Any] = {}
    for volume in volumes:
        artifact = artifacts[volume.name]
        origin, shape = task.region(volume, artifact)
        context = task.context(volume, artifact)
        context["scratch_dir"] = scratch / volume.name
        prediction = processor(
            artifact.read(origin, shape, processor.reads_channels()), **params
        )
        truth = task.ground_truth(volume, artifact)
        regions[volume.name] = {
            "origin": list(origin), "shape": list(shape),
            "whole_region": context["whole_region"],
            "artifact": str(artifact.path),
        }
        if keep is not None and task.canonical == "instances":
            keep.mkdir(parents=True, exist_ok=True)
            regions[volume.name]["scored_artifact"] = str(write_scored(
                keep / f"{volume.name}.zarr", prediction, artifact, origin,
                convention=processor.describe(params),
                postprocess={"name": type(processor).__name__, "params": params},
            ))
        per_volume[volume.name] = {
            name: metric(prediction, truth, **context)
            for name, metric in metric_objects.items()
        }
    return _aggregate(per_volume, metric_objects), per_volume, {"volumes": regions}


def fit(
    artifacts: dict[str, Artifact],
    volumes: tuple[Volume, ...],
    task: BaseTask,
    processor: BasePostprocess,
    metric_objects: dict[str, BaseMetric],
    config: TaskConfig,
    scratch: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """The candidate that scores best on this (validation) artifact, and what it scored."""
    ranking_metric = metric_objects[config.rank_by]
    best: tuple[float, dict[str, Any], dict[str, dict[str, float]]] | None = None
    for candidate in processor.search_space():
        scores, _, _ = score_once(
            artifacts, volumes, task, processor, metric_objects, candidate, scratch
        )
        value = scores[config.rank_by][ranking_metric.primary]
        signed = value if ranking_metric.higher_is_better else -value
        print(f"  {processor.describe(candidate)}: "
              f"{config.rank_by}.{ranking_metric.primary} = {value:.4f}", flush=True)
        if best is None or signed > best[0]:
            best = (signed, candidate, scores)
    assert best is not None                      # search_space() is non-empty by construction
    return best[1], best[2]


def cmd_score(args: argparse.Namespace) -> None:
    config = load_task_config(args.config)
    if args.val is not None and config.fit_volumes is None:
        # A config problem, reported before any artifact is opened.
        raise SystemExit(
            f"{args.config} declares no fit split, so --val has nothing to fit on. Add a "
            "[data.fit] table naming the data config (and optionally the volumes) the validation "
            "artifacts cover; it may not share a volume with [data.test]."
        )
    task, processor, metric_objects = build(config)
    ranking_metric = metric_objects[config.rank_by]

    test_artifacts = resolve_artifacts(Path(args.test), config.volumes)
    for volume in config.volumes:
        check_compatible(test_artifacts[volume.name], processor, task)
        check_same_volume(volume, test_artifacts[volume.name], "--test")
    representative = next(iter(test_artifacts.values()))
    candidates = processor.search_space()
    scratch = Path(args.scratch or (Path(args.test).parent / ".mia_evals_scratch"))
    scratch.mkdir(parents=True, exist_ok=True)

    if len(candidates) > 1 and args.val is None:
        raise SystemExit(
            f"{config.postprocess.name} has {len(candidates)} candidate settings "
            f"({', '.join(processor.describe(c) for c in candidates[:3])}...), so one must be "
            "chosen on a validation artifact and then applied here. Pass --val with artifacts of "
            "the task's [data.fit] volumes, or configure a single candidate.\n\nSweeping on --test "
            "and reporting the best would be selecting on the number being reported."
        )

    if args.val is not None:
        # The fit split is the task's own declaration, `[data.fit]`: a different *set of volumes*
        # from the reported one (lmd_ssl_v1 finetunes on four of eight eval volumes and reports on
        # the other four), checked disjoint at load. Postprocessor, metrics and ranking are the
        # reported task's, so the two halves measure the same thing.
        assert config.fit_volumes is not None            # refused above, before any artifact
        print(f"fit volumes: {[v.name for v in config.fit_volumes]}", flush=True)
        val_artifacts = resolve_artifacts(Path(args.val), config.fit_volumes)
        for volume in config.fit_volumes:
            check_compatible(val_artifacts[volume.name], processor, task)
            check_same_volume(volume, val_artifacts[volume.name], "--val")
        print(f"fitting {config.postprocess.name} on {Path(args.val).name}", flush=True)
        params, val_scores = fit(
            val_artifacts, config.fit_volumes, task, processor, metric_objects, config, scratch
        )
        print(f"chose {processor.describe(params)}", flush=True)
    else:
        params, val_scores = candidates[0], {}

    print(f"scoring {Path(args.test).name}", flush=True)
    keep = None if args.no_scored else Path(args.scored_out or scratch / "scored")
    scores, per_volume, region = score_once(
        test_artifacts, config.volumes, task, processor, metric_objects, params, scratch,
        keep=keep,
    )
    scored = {name: r["scored_artifact"] for name, r in region["volumes"].items()
              if "scored_artifact" in r}
    if scored:
        print(f"scored labellings kept under {keep}", flush=True)
    value = scores[config.rank_by][ranking_metric.primary]
    for name in sorted(per_volume):
        each = per_volume[name][config.rank_by].get(ranking_metric.primary)
        print(f"    {name:32s} {ranking_metric.primary} = {each:.4f}", flush=True)
    print(f"  {config.rank_by}.{ranking_metric.primary} = {value:.4f} "
          f"(unweighted mean over {len(per_volume)} volumes)", flush=True)

    submission = Submission(
        task_name=config.task_name,
        producer={
            "artifacts": {n: str(a.path) for n, a in sorted(test_artifacts.items())},
            "run": representative.attrs.get("run"),
            "step": representative.attrs.get("step"),
            "kind": representative.kind,
            "convention": representative.convention,
            "artifact_attrs": {
                k: v for k, v in representative.attrs.items() if k != "resolved_config"
            },
        },
        scores=scores,
        per_volume=per_volume,
        ranking={
            "metric": config.rank_by,
            "key": ranking_metric.primary,
            "value": value,
            "higher_is_better": ranking_metric.higher_is_better,
        },
        postprocess={
            "name": config.postprocess.name,
            "params": params,
            "describe": processor.describe(params),
            "fitted_on": None if args.val is None else str(Path(args.val).resolve()),
            "fitted_on_data_config": (
                None if args.val is None else str(config.fit_data_config_path)
            ),
            "validation_scores": val_scores,
            "scored_artifacts": scored,
        },
        region=region,
        config=config.as_record(),
        provenance=_provenance(args, representative),
        label=args.label,
    )
    root = Path(args.leaderboard)
    # A task is its test set and its ranking metric. Before this record joins a table, it must be
    # scored on the same thing as the records already there -- otherwise it is a different task
    # wearing the same name, and the table would rank things that are not comparable.
    try:
        existing = record.load_task(root, config.task_name)     # also checks they agree
        if existing:
            reference = existing[0]
            record.assert_same_task(
                config.task_name, reference, submission,
                record.records_dir(root, config.task_name) / f"{reference.identifier()}.json",
                args.config,
            )
    except ValueError as error:
        raise SystemExit(str(error)) from None
    path = submission.write(root)
    print(f"record: {path}", flush=True)
    # Rendered here rather than left to a follow-up `mia-evals leaderboard`: a record that is not
    # in the table is invisible, and the failure mode of "score, then forget to regenerate" is a
    # committed table that silently omits a submission. Only this task's page is touched.
    for rendered in leaderboard.write(root, config.task_name):
        print(f"updated: {rendered}", flush=True)


def _provenance(args: argparse.Namespace, test: Artifact) -> dict[str, Any]:
    """The producing run's own record of itself, copied in.

    Copied rather than referenced because a run directory on `/nrs` is not permanent, and a record
    holding only a path stops being checkable the moment it is cleaned up.
    """
    provenance: dict[str, Any] = {"mia_evals_commit": git_commit(_REPO_ROOT)}
    run_dir = args.run_dir or test.attrs.get("run_dir")
    if run_dir and Path(run_dir).is_dir():
        run_path = Path(run_dir)
        resolved = run_path / "resolved_config.json"
        commit = run_path / "git_commit.txt"
        provenance["run_dir"] = str(run_path)
        if resolved.is_file():
            provenance["resolved_config"] = json.loads(resolved.read_text())
        if commit.is_file():
            provenance["producer_commit"] = commit.read_text().strip()
    elif run_dir:
        provenance["run_dir_missing"] = str(run_dir)
    return provenance


def cmd_leaderboard(args: argparse.Namespace) -> None:
    root, task = Path(args.leaderboard), args.task
    known = record.task_names(root)
    if task and task not in known:
        raise SystemExit(
            f"no task {task!r} under {root}. It holds: {known or '(nothing yet)'}"
        )
    try:
        _render_or_check(args, root, task)
    except ValueError as error:                 # records that do not belong together
        raise SystemExit(str(error)) from None


def _render_or_check(args: argparse.Namespace, root: Path, task: str | None) -> None:
    if args.check:
        stale = leaderboard.check(root, task)
        if not stale:
            print(f"{root} is up to date")
            return
        listing = "\n".join(f"    {path}" for path in stale)
        raise SystemExit(
            f"{len(stale)} file(s) do not match their records:\n{listing}\n"
            f"Regenerate with\n    mia-evals leaderboard"
            + (f" --task {task}" if task else "")
        )
    for path in leaderboard.write(root, task):
        print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="score an artifact against a task config")
    score.add_argument("config", type=Path, help="a task .toml from configs/tasks/")
    score.add_argument("--test", type=Path, required=True,
                       help="directory of <volume>.zarr artifacts to report on (or a single "
                            "artifact, if the task has one volume)")
    score.add_argument("--scored-out", type=Path, default=None,
                       help="where to keep the post-processed test labellings that were scored "
                            "(default: <scratch>/scored); they are what the views show")
    score.add_argument("--no-scored", action="store_true",
                       help="do not keep the post-processed labellings")
    score.add_argument("--val", type=Path, default=None,
                       help="artifacts to fit the postprocessor's hyperparameter on, same form as "
                            "--test, covering the task's [data.fit] volumes; required whenever "
                            "there is more than one candidate")
    score.add_argument("--leaderboard", type=Path, default=DEFAULT_LEADERBOARD,
                       help="leaderboard root; the record lands in <root>/<task_name>/records/ and "
                            f"that task's table is re-rendered (default {DEFAULT_LEADERBOARD})")
    score.add_argument("--run-dir", type=Path, default=None,
                       help="the producing run directory, whose resolved config and commit are "
                            "copied into the record")
    score.add_argument("--label", type=str, default="",
                       help="record filename stem; defaults to the run name and step")
    score.add_argument("--scratch", type=Path, default=None,
                       help="scratch directory for intermediates (e.g. a cropped skeleton)")
    score.set_defaults(func=cmd_score)

    board = sub.add_parser("leaderboard", help="render or verify the leaderboard tables")
    board.add_argument("--leaderboard", type=Path, default=DEFAULT_LEADERBOARD,
                       help=f"leaderboard root (default {DEFAULT_LEADERBOARD})")
    board.add_argument("--task", type=str, default=None,
                       help="rebuild only this task's table; default is every task")
    board.add_argument("--check", action="store_true",
                       help="verify the committed tables match the records; write nothing")
    board.set_defaults(func=cmd_leaderboard)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
