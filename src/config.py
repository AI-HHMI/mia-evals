"""Parsing a scoring config -- `configs/scoring/*.toml` -- and resolving the `miao` data configs it points at.

A scoring config says three things: which task is scored (`task_name`, the reported volumes and
the ranking metric), where a post-processing sweep is fitted, and which post-processing route turns
the artifact into a labelling. Several scoring configs may score one task through different routes;
what makes them one task is checked in `report.record`.

Two files, on purpose. The data lives in a `miao` YAML, generated with a provenance header and a
drift check, and `miao` sets `extra="forbid"` -- so a data config *cannot* carry `task`, `metric` or
`split` keys, and the drafts in the corpus that tried raise on load. The task lives here instead,
and references the data by path.

That split also settles split membership. `miao` rejects a per-volume `split:` key, so a split is
stated in the scoring config: `[data.test]` names the data config (and optionally a `volumes` filter) of
the reported volumes, and `[data.fit]` the one the post-processing sweep is fitted on. A task
without a sweep needs no fit split and may write a plain `[data]` instead. The two splits may not
share a volume, which is checked here, at load, rather than after an hour of scoring.

The YAML is read through `miao` itself rather than parsed here. miao owns that format and
validates it, so a malformed or task-key-carrying config is reported against the file by its own
schema instead of being silently half-understood by a parser in this repo.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tasks.base import Volume

#: Fields of a miao volume entry that a score cares about. Anything else in the entry -- weights,
#: resolutions, normalisation windows -- describes how to *sample* the volume for training and has
#: no bearing on scoring a prediction over it, so it is carried in `extra` rather than dropped.
VOLUME_FIELDS = ("name", "path", "image_key", "label_key", "bounding_box", "zarr_version")


@dataclass(frozen=True)
class Section:
    """A registry lookup name plus the kwargs to construct it with."""

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoringConfig:
    """One fully resolved evaluation, assembled from a `.toml`."""

    task_name: str
    task: Section
    postprocess: Section
    metrics: tuple[str, ...]
    rank_by: str
    data_config_path: Path
    volumes: tuple[Volume, ...]
    metric_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: str = ""
    #: The fit split, from `[data.fit]`: where a swept post-processing parameter is chosen. None
    #: for a task that declares none, which is fine for a single-candidate post-processor.
    fit_data_config_path: Path | None = None
    fit_volumes: tuple[Volume, ...] | None = None

    def as_record(self) -> dict[str, Any]:
        """The settings, flattened for a submission record. Paths as strings, no objects."""
        return {
            "task_name": self.task_name,
            "task": {"name": self.task.name, "kwargs": self.task.kwargs},
            "postprocess": {"name": self.postprocess.name, "kwargs": self.postprocess.kwargs},
            "metrics": list(self.metrics),
            "rank_by": self.rank_by,
            "data_config_path": str(self.data_config_path),
            "volumes": [_volume_record(v) for v in self.volumes],
            "fit_data_config_path": (
                None if self.fit_data_config_path is None else str(self.fit_data_config_path)
            ),
            "fit_volumes": (
                None if self.fit_volumes is None else [_volume_record(v) for v in self.fit_volumes]
            ),
            "notes": self.notes,
        }


def _volume_record(v: Volume) -> dict[str, Any]:
    return {
        "name": v.name,
        "path": str(v.path),
        "label_key": v.label_key,
        "bounding_box": None if v.bounding_box is None
        else [list(pair) for pair in v.bounding_box],
    }


def load_data_config(path: str | Path) -> tuple[Volume, ...]:
    """The volumes a miao data config describes, as `Volume`s.

    Read *through miao*, not by parsing the YAML here. miao owns the format, validates it, and is
    already a core dependency -- and it sets `extra="forbid"`, so a data config carrying `task:`,
    `metric:` or `split:` keys is rejected rather than silently half-honoured. That rejection is
    the whole reason a task lives in its own file: the hand-written eval drafts in the corpus carry
    exactly those keys and fail to load with six validation errors.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"[data].config_path does not exist: {path}")
    try:
        from miao.config import load_config
    except ImportError as error:
        raise ImportError(
            "reading a data config needs `miao`, a core dependency of mia-evals: "
            "pip install -e ."
        ) from error

    try:
        config = load_config(path)
    except Exception as error:
        raise ValueError(
            f"{path} is not a valid miao config:\n{error}\n\n"
            "miao sets extra=\"forbid\", so `task:`, `metric:`, `label_class:` and per-volume "
            "`split:` keys are rejected. Those belong in the scoring config that references this file, "
            "not in the data config -- move them there and regenerate."
        ) from error

    volumes = []
    for entry in config.volumes:
        box = entry.bounding_box
        volumes.append(Volume(
            name=str(entry.name),
            path=Path(str(entry.path)),
            image_key=str(entry.image_key),
            label_key=entry.label_key,
            bounding_box=None if box is None else tuple(
                (int(pair[0]), int(pair[1])) for pair in box
            ),
            zarr_version=str(entry.zarr_version),
            # What remains describes how to *sample* the volume for training -- weights,
            # resolutions, normalisation windows -- and has no bearing on scoring a prediction over
            # it. Carried rather than dropped so a record can show the data config in full.
            extra={
                k: v for k, v in entry.model_dump(mode="json").items()
                if k not in VOLUME_FIELDS
            },
        ))
    if not volumes:
        raise ValueError(f"{path} describes no volumes")
    return tuple(volumes)


def _section(raw: dict[str, Any], name: str, required: bool = True) -> Section:
    if name not in raw:
        if required:
            raise ValueError(f"config is missing the [{name}] section")
        return Section(name="")
    body = dict(raw[name])
    if "name" not in body:
        raise ValueError(f"[{name}] must set 'name' to select a registered implementation")
    return Section(name=str(body.pop("name")), kwargs=body)


def _resolve_split(entry: dict[str, Any], task_path: Path, label: str) -> tuple[Path, tuple[Volume, ...]]:
    """One split's `config_path` (relative to the scoring config) and optional `volumes` filter."""
    entry = dict(entry)
    if "config_path" not in entry:
        raise ValueError(
            f"{label} must set config_path, pointing at the miao YAML that defines the volumes. "
            "Data is referenced rather than restated so that one generated, provenance-stamped "
            "config serves every task over the same dataset."
        )
    config_path = Path(str(entry.pop("config_path")))
    if not config_path.is_absolute():
        config_path = (task_path.parent / config_path).resolve()
    available = load_data_config(config_path)

    wanted = entry.pop("volumes", None)
    if wanted is None:
        volumes = available
    else:
        by_name = {v.name: v for v in available}
        unknown = [n for n in wanted if n not in by_name]
        if unknown:
            raise ValueError(
                f"{label}.volumes names {unknown}, which {config_path.name} does not contain. "
                f"It holds: {sorted(by_name)}"
            )
        volumes = tuple(by_name[n] for n in wanted)
    if entry:
        raise ValueError(f"{label} has unknown key(s) {sorted(entry)}")
    return config_path, volumes


def load_scoring_config(path: str | Path) -> ScoringConfig:
    """Parse a scoring config, resolving its data config(s) and filtering to the volumes it names.

    `[data]` takes one of two shapes. `[data.test]` plus an optional `[data.fit]`, each with a
    `config_path` and an optional `volumes` filter, states the reported split and the split a
    post-processing sweep is fitted on. A plain `[data]` with the same keys is the reported split
    alone, for a task whose post-processor has a single candidate and so needs nothing fitted.
    """
    path = Path(path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    if "task_name" not in raw:
        raise ValueError("config must set a top-level 'task_name'")
    for section in ("task", "postprocess", "metric", "data"):
        if section not in raw:
            raise ValueError(f"config is missing the [{section}] section")

    data = dict(raw["data"])
    test_entry = data.pop("test", None)
    fit_entry = data.pop("fit", None)
    fit_path: Path | None = None
    fit_volumes: tuple[Volume, ...] | None = None
    if test_entry is None:
        if fit_entry is not None:
            raise ValueError(
                "[data.fit] needs a [data.test] beside it: once a fit split is declared, the "
                "reported split is stated as [data.test] rather than as top-level [data] keys"
            )
        config_path, volumes = _resolve_split(data, path, "[data]")
    else:
        if data:
            raise ValueError(
                f"[data] mixes a [data.test] table with top-level key(s) {sorted(data)}; "
                "put them under [data.test]"
            )
        config_path, volumes = _resolve_split(test_entry, path, "[data.test]")
        if fit_entry is not None:
            fit_path, fit_volumes = _resolve_split(fit_entry, path, "[data.fit]")
            overlap = {v.name for v in volumes} & {v.name for v in fit_volumes}
            if overlap:
                raise ValueError(
                    f"[data.fit] and [data.test] share volume(s) {sorted(overlap)}. Fitting a "
                    "threshold on a volume that is then reported is selecting on the number being "
                    "published, which is the one thing the split exists to prevent."
                )

    metric_section = dict(raw["metric"])
    names = metric_section.pop("names", None)
    if not names:
        raise ValueError("[metric] must set `names` to a non-empty list of registered metrics")
    rank_by = metric_section.pop("rank_by", None) or names[0]
    if rank_by not in names:
        raise ValueError(
            f"[metric].rank_by = {rank_by!r} is not among names {list(names)}; a task can only "
            "rank on a metric it computes"
        )
    # Whether larger is better is the *metric's* declaration, never the task's, so it is not
    # settable here -- a task that could assert the direction could rank on VOI as though more
    # were better.
    for reserved in ("higher_is_better",):
        if reserved in metric_section:
            raise ValueError(
                f"[metric].{reserved} is not configurable: it is declared by the metric itself, "
                "so that no task can rank in the wrong direction."
            )

    return ScoringConfig(
        task_name=str(raw["task_name"]),
        task=_section(raw, "task"),
        postprocess=_section(raw, "postprocess"),
        metrics=tuple(str(n) for n in names),
        rank_by=str(rank_by),
        data_config_path=config_path,
        volumes=volumes,
        metric_kwargs={str(k): dict(v) for k, v in metric_section.items() if isinstance(v, dict)},
        notes=str(raw.get("notes", "")),
        fit_data_config_path=fit_path,
        fit_volumes=fit_volumes,
    )
