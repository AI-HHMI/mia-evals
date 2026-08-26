"""Single entrypoint: score a prediction artifact against a task, and render the leaderboard.

    python src/evaluate.py score  configs/tasks/<task>.toml --test <artifact.zarr> \\
        [--val <artifact.zarr>] [--record leaderboard/records]
    python src/evaluate.py leaderboard [--check]

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

from artifact import Artifact, open_artifact
from config import TaskConfig, load_task_config
from metrics.base import BaseMetric
from postprocess.base import BasePostprocess
from report import leaderboard
from report.record import Submission, git_commit
from tasks.base import BaseTask, Volume

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = _REPO_ROOT / "leaderboard" / "records"
DEFAULT_LEADERBOARD = _REPO_ROOT / "leaderboard" / "README.md"


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
) -> tuple[dict[str, dict[str, float]], dict[str, Any], dict[str, Any]]:
    """Postprocess and score every volume under one parameter set.

    Returns (aggregate, per-volume, regions). Both halves are kept: the aggregate is what ranks,
    and the per-volume numbers are what make a bad aggregate diagnosable -- on this eval set one
    modality failing completely and three working looks identical, in the mean, to all four being
    mediocre.
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
    task, processor, metric_objects = build(config)
    ranking_metric = metric_objects[config.rank_by]

    test_artifacts = resolve_artifacts(Path(args.test), config.volumes)
    for artifact in test_artifacts.values():
        check_compatible(artifact, processor, task)
    representative = next(iter(test_artifacts.values()))
    candidates = processor.search_space()
    scratch = Path(args.scratch or (Path(args.test).parent / ".mia_evals_scratch"))
    scratch.mkdir(parents=True, exist_ok=True)

    if len(candidates) > 1 and args.val is None:
        raise SystemExit(
            f"{config.postprocess.name} has {len(candidates)} candidate settings "
            f"({', '.join(processor.describe(c) for c in candidates[:3])}...), so one must be "
            "chosen on a validation artifact and then applied here. Pass --val, or configure a "
            "single candidate.\n\nSweeping on --test and reporting the best would be selecting on "
            "the number being reported."
        )

    if args.val is not None:
        # The fit split may be a different *set of volumes*, not merely different data over the
        # same ones: lmd_ssl_v1 finetunes on four of eight eval volumes and reports on the other
        # four, so the threshold is chosen on volumes the model trained on. `--val-config` names
        # that half. Only its volumes are taken from it -- postprocessor, metrics and ranking stay
        # the reported task's, or the two halves would not be measuring the same thing.
        fit_config = (
            config if args.val_config is None else load_task_config(args.val_config)
        )
        if fit_config is not config:
            if fit_config.postprocess.name != config.postprocess.name:
                raise SystemExit(
                    f"--val-config uses postprocess {fit_config.postprocess.name!r} but the "
                    f"reported task uses {config.postprocess.name!r}. The fitted parameter would "
                    "not apply to the postprocessor it is handed to."
                )
            overlap = {v.name for v in config.volumes} & {v.name for v in fit_config.volumes}
            if overlap:
                raise SystemExit(
                    f"--val-config shares volume(s) {sorted(overlap)} with the reported task. "
                    "Fitting a threshold on a volume that is then reported is selecting on the "
                    "number being published, which is the one thing this split exists to prevent."
                )
            print(f"fit volumes: {[v.name for v in fit_config.volumes]}", flush=True)
        val_artifacts = resolve_artifacts(Path(args.val), fit_config.volumes)
        for artifact in val_artifacts.values():
            check_compatible(artifact, processor, task)
        print(f"fitting {config.postprocess.name} on {Path(args.val).name}", flush=True)
        params, val_scores = fit(
            val_artifacts, fit_config.volumes, task, processor, metric_objects, config, scratch
        )
        print(f"chose {processor.describe(params)}", flush=True)
    else:
        params, val_scores = candidates[0], {}

    print(f"scoring {Path(args.test).name}", flush=True)
    scores, per_volume, region = score_once(
        test_artifacts, config.volumes, task, processor, metric_objects, params, scratch
    )
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
            "fitted_on_config": None if args.val_config is None else str(args.val_config),
            "validation_scores": val_scores,
        },
        region=region,
        config=config.as_record(),
        provenance=_provenance(args, representative),
        label=args.label,
    )
    path = submission.write(Path(args.record))
    print(f"record: {path}", flush=True)


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
    records, output = Path(args.record), Path(args.output)
    if args.check:
        if leaderboard.check(records, output):
            print(f"{output} is up to date")
            return
        raise SystemExit(
            f"{output} does not match {records}. Regenerate it with\n"
            "    python src/evaluate.py leaderboard"
        )
    print(f"wrote {leaderboard.write(records, output)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="score an artifact against a task config")
    score.add_argument("config", type=Path, help="a task .toml from configs/tasks/")
    score.add_argument("--test", type=Path, required=True,
                       help="directory of <volume>.zarr artifacts to report on (or a single "
                            "artifact, if the task has one volume)")
    score.add_argument("--val", type=Path, default=None,
                       help="artifacts to fit the postprocessor's hyperparameter on, same form as "
                            "--test; required whenever there is more than one candidate")
    score.add_argument("--val-config", type=Path, default=None,
                       help="task .toml whose volumes form the fit split, when it is a different "
                            "set of volumes than the reported one (as in lmd_ssl_v1, which "
                            "finetunes on half the eval set and reports on the other half)")
    score.add_argument("--record", type=Path, default=DEFAULT_RECORDS,
                       help=f"where the submission record is written (default {DEFAULT_RECORDS})")
    score.add_argument("--run-dir", type=Path, default=None,
                       help="the producing run directory, whose resolved config and commit are "
                            "copied into the record")
    score.add_argument("--label", type=str, default="",
                       help="record filename stem; defaults to the run name and step")
    score.add_argument("--scratch", type=Path, default=None,
                       help="scratch directory for intermediates (e.g. a cropped skeleton)")
    score.set_defaults(func=cmd_score)

    board = sub.add_parser("leaderboard", help="render or verify the leaderboard")
    board.add_argument("--record", type=Path, default=DEFAULT_RECORDS)
    board.add_argument("--output", type=Path, default=DEFAULT_LEADERBOARD)
    board.add_argument("--check", action="store_true",
                       help="verify the committed table matches the records; write nothing")
    board.set_defaults(func=cmd_leaderboard)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
