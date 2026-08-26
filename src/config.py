"""Parsing a task `.toml`, and resolving the `miao` data config it points at.

Two files, on purpose. The data lives in a `miao` YAML, generated with a provenance header and a
drift check, and `miao` sets `extra="forbid"` -- so a data config *cannot* carry `task`, `metric` or
`split` keys, and the drafts in the corpus that tried raise on load. The task lives here instead,
and references the data by path.

That split also settles split membership. `miao` rejects a per-volume `split:` key, so which volumes
belong to which split is a name filter in `[data].volumes`. One data config per dataset; one task
file per (dataset x split x task).

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
class TaskConfig:
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

    def as_record(self) -> dict[str, Any]:
        """The settings, flattened for a submission record. Paths as strings, no objects."""
        return {
            "task_name": self.task_name,
            "task": {"name": self.task.name, "kwargs": self.task.kwargs},
            "postprocess": {"name": self.postprocess.name, "kwargs": self.postprocess.kwargs},
            "metrics": list(self.metrics),
            "rank_by": self.rank_by,
            "data_config_path": str(self.data_config_path),
            "volumes": [
                {
                    "name": v.name,
                    "path": str(v.path),
                    "label_key": v.label_key,
                    "bounding_box": None if v.bounding_box is None
                    else [list(pair) for pair in v.bounding_box],
                }
                for v in self.volumes
            ],
            "notes": self.notes,
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
            "`split:` keys are rejected. Those belong in the task .toml that references this file, "
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


def load_task_config(path: str | Path) -> TaskConfig:
    """Parse a task `.toml`, resolving its data config and filtering to this task's volumes."""
    path = Path(path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    if "task_name" not in raw:
        raise ValueError("config must set a top-level 'task_name'")
    for section in ("task", "postprocess", "metric", "data"):
        if section not in raw:
            raise ValueError(f"config is missing the [{section}] section")

    data = dict(raw["data"])
    if "config_path" not in data:
        raise ValueError(
            "[data] must set config_path, pointing at the miao YAML that defines the volumes. "
            "Data is referenced rather than restated so that one generated, provenance-stamped "
            "config serves every task over the same dataset."
        )
    config_path = Path(str(data.pop("config_path")))
    if not config_path.is_absolute():
        config_path = (path.parent / config_path).resolve()
    available = load_data_config(config_path)

    wanted = data.pop("volumes", None)
    if wanted is None:
        volumes = available
    else:
        by_name = {v.name: v for v in available}
        unknown = [n for n in wanted if n not in by_name]
        if unknown:
            raise ValueError(
                f"[data].volumes names {unknown}, which {config_path.name} does not contain. "
                f"It holds: {sorted(by_name)}"
            )
        volumes = tuple(by_name[n] for n in wanted)
    if data:
        raise ValueError(f"[data] has unknown key(s) {sorted(data)}")

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

    return TaskConfig(
        task_name=str(raw["task_name"]),
        task=_section(raw, "task"),
        postprocess=_section(raw, "postprocess"),
        metrics=tuple(str(n) for n in names),
        rank_by=str(rank_by),
        data_config_path=config_path,
        volumes=volumes,
        metric_kwargs={str(k): dict(v) for k, v in metric_section.items() if isinstance(v, dict)},
        notes=str(raw.get("notes", "")),
    )
