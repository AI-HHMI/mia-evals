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


def score_once(
    artifact: Artifact,
    volumes: tuple[Volume, ...],
    task: BaseTask,
    processor: BasePostprocess,
    metric_objects: dict[str, BaseMetric],
    params: dict[str, Any],
    scratch: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Postprocess and score one artifact under one parameter set, over every volume."""
    for metric in metric_objects.values():
        # Stateful metrics accumulate across volumes on purpose (one confusion matrix over the
        # whole set, not an average of per-volume means), so they must start clean per candidate.
        if hasattr(metric, "reset"):
            metric.reset()

    scores: dict[str, dict[str, float]] = {}
    regions: dict[str, Any] = {}
    for volume in volumes:
        origin, shape = task.region(volume, artifact)
        context = task.context(volume, artifact)
        context["scratch_dir"] = scratch
        prediction = processor(artifact.read(origin, shape), **params)
        truth = task.ground_truth(volume, artifact)
        regions[volume.name] = {
            "origin": list(origin), "shape": list(shape),
            "whole_region": context["whole_region"],
        }
        for name, metric in metric_objects.items():
            scores[name] = metric(prediction, truth, **context)
    return scores, {"volumes": regions}


def fit(
    artifact: Artifact,
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
        scores, _ = score_once(
            artifact, volumes, task, processor, metric_objects, candidate, scratch
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

    test = open_artifact(args.test)
    check_compatible(test, processor, task)
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
        validation = open_artifact(args.val)
        check_compatible(validation, processor, task)
        print(f"fitting {config.postprocess.name} on {Path(args.val).name}", flush=True)
        params, val_scores = fit(
            validation, config.volumes, task, processor, metric_objects, config, scratch
        )
        print(f"chose {processor.describe(params)}", flush=True)
    else:
        params, val_scores = candidates[0], {}

    print(f"scoring {Path(args.test).name}", flush=True)
    scores, region = score_once(
        test, config.volumes, task, processor, metric_objects, params, scratch
    )
    value = scores[config.rank_by][ranking_metric.primary]
    print(f"  {config.rank_by}.{ranking_metric.primary} = {value:.4f}", flush=True)

    submission = Submission(
        task_name=config.task_name,
        producer={
            "artifact": str(Path(args.test).resolve()),
            "run": test.attrs.get("run"),
            "step": test.attrs.get("step"),
            "kind": test.kind,
            "convention": test.convention,
            "artifact_attrs": {k: v for k, v in test.attrs.items() if k != "resolved_config"},
        },
        scores=scores,
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
            "validation_scores": val_scores,
        },
        region=region,
        config=config.as_record(),
        provenance=_provenance(args, test),
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
                       help="the artifact to report on")
    score.add_argument("--val", type=Path, default=None,
                       help="artifact to fit the postprocessor's hyperparameter on; required "
                            "whenever there is more than one candidate")
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
