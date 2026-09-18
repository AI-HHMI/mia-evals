"""The submission record: one JSON file per evaluation, and the only input to the leaderboard.

The leaderboard is *derived*, never hand-edited, for the same reason the data configs are
generated: a table someone can edit is a table that can disagree with the runs behind it. Records
are small and git-tracked, so an entry arrives as a reviewable diff and the whole table can be
rebuilt from the repository alone.

**Provenance is copied, not referenced.** A record carries the producing run's `resolved_config`
and git commit *inline*, because a run directory on `/nrs` is not permanent -- the scoring scripts
already delete their 51 GB affinity artifacts, and run directories get cleaned. A record holding
only a path becomes an unfalsifiable claim the moment that path disappears; one holding the
resolved config stays checkable forever.

**The fitted hyperparameter is part of the record.** A number produced with the threshold that won
on validation is a different claim from one produced with a default, and a leaderboard that does
not say which invites the reader to assume the wrong one.
"""

from __future__ import annotations

import importlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def git_commit(repo: Path) -> str:
    """`<sha>` or `<sha> (dirty)` for a repository, or "unavailable".

    Not an error when it fails: a released tarball has no `.git`, and a record from one is still
    worth having -- it just cannot claim a commit.
    """
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        return f"{sha} (dirty)" if dirty else sha
    except (subprocess.SubprocessError, OSError):
        return "unavailable"


def _component_versions() -> dict[str, str]:
    """Versions of everything whose change could move a number."""
    versions = {"python": platform.python_version()}
    # `cc3d` used to be here, and was dropped along with the dependency: nothing imports it since
    # connected components became the numba implementation in `utils/`, so its version could not
    # move a number. `funlib.evaluate` records as "absent" unless the skeleton metric is installed.
    for module in ("numpy", "zarr", "miao", "numba", "networkx", "funlib.evaluate"):
        try:
            # `importlib.import_module`, not `__import__`: the latter returns the top-level
            # package, so a dotted name would report `funlib`'s version and not the
            # submodule's.
            found = importlib.import_module(module)
            versions[module] = getattr(found, "__version__", "present")
        except ImportError:
            versions[module] = "absent"
    return versions


@dataclass
class Submission:
    """One scored evaluation of one model on one task.

    `region` is recorded because several metrics are not comparable across extents -- the same
    model measured 0.3045 nERL over a whole NISB cube and 0.4192 on a 512^3 block of it. A
    leaderboard that ranked across regions would be ordering models by how much of a volume each
    happened to be scored on.
    """

    task_name: str
    #: What produced the prediction: run directory, step, and the artifact's own attrs.
    producer: dict[str, Any]
    #: Metric name -> its result dict, aggregated over the task's volumes.
    scores: dict[str, dict[str, float]]
    #: `rank_by` metric, its primary key, the value, and the direction. Denormalised so a renderer
    #: never has to import a metric class to sort a table.
    ranking: dict[str, Any]
    #: The postprocessor and the parameters that won on validation, plus what they scored there.
    postprocess: dict[str, Any]
    #: (origin, shape) actually scored, per volume.
    region: dict[str, Any]
    #: The scoring config, expanded.
    config: dict[str, Any]
    #: Volume name -> metric name -> result dict, before aggregation. Kept because an aggregate
    #: alone cannot distinguish "one modality failed outright" from "all four were mediocre", and
    #: on an 8-volume eval set spanning three modalities that is the difference that matters.
    per_volume: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    #: Copied from the producing run so the entry survives its run directory being deleted.
    provenance: dict[str, Any] = field(default_factory=dict)
    #: Volume name -> {"overlay": url, "side_by_side": url, "shows": "scored" | "before size filter"}:
    #: the neuroglancer views of this row, computed when it was scored from the scorer's own
    #: fileglancer key file. Stored in the record so the task's views page can be rendered from
    #: records alone, by whoever hosts it, without holding the keys that made these links; a
    #: volume absent here is shown as missing on that page.
    views: dict[str, dict[str, Any]] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=_component_versions)
    schema_version: int = SCHEMA_VERSION
    #: Legacy free-text name. Empty on every record written since 2026-09-18; the identifier
    #: is derived instead, so a row's name always says which checkpoint it scored.
    label: str = ""
    #: The scoring route (`ScoringConfig.route`), third part of the identifier.
    route: str = ""

    def identity(self) -> dict[str, Any]:
        """What this record claims to be a row of; see `task_identity`."""
        return task_identity(
            self.config.get("volumes") or [],
            str(self.ranking.get("metric")),
            str(self.ranking.get("key")),
        )

    def identifier(self) -> str:
        """`<run>.step<N>.<route>`: exactly which checkpoint was scored, and through which route.

        `run` is the producing run directory's name (`gary__1a_dinov3_axial_subpixel_20260916_215544`),
        which already carries the experiment, the arm and the launch time; `step` is the checkpoint;
        `route` is the scoring config's `route`. Two records may share a run and step only through
        different routes, and `mia-evals score` refuses to overwrite an existing identifier. The
        convention replaced hand-written labels on 2026-09-18, after those had made the tables
        unreadable (`2c_step50000`, `2c_step50000_sizefilter`, `sam1_arm4_8nm_gb16_r0_step200000`
        said neither which run nor, for two of them, which route).
        """
        if self.label:                       # legacy records only; nothing writes labels any more
            return self.label
        artifacts = self.producer.get("artifacts") or {}
        run = str(
            self.producer.get("run")
            or next(iter(artifacts.values()), None)
            or "submission"
        )
        stem = Path(run).name.replace("/", "_")
        step = self.producer.get("step")
        route = self.route or str(self.postprocess.get("name") or "unknown")
        return f"{stem}.step{step}.{route}" if step is not None else f"{stem}.{route}"

    def write(self, root: Path) -> Path:
        """Write to `<root>/<task_name>/records/<identifier>.json`, creating directories."""
        directory = records_dir(root, self.task_name)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.identifier()}.json"
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=False) + "\n")
        return path


# ------------------------------------------------------------------------------ task identity
#
# A task is the thing rows compete on: the reported volumes *with their ground truth* (store path,
# label key, bounding box) and the metric they are ranked by. Nothing else. The post-processor,
# the route the truth was read by (`truth_kind`), the split a sweep was fitted on and the producer
# are all properties of a submission, recorded and, where they vary within a table, shown as
# columns -- a reader may attribute a difference to them, but must be able to see them.
#
# `task_name` alone used to decide which directory a record landed in, and nothing compared a
# scoring config with the records already there: a second file reusing the name with another test set
# would have joined the table, separated only by the region grouping if the extents happened to
# differ. The identity is therefore checked twice -- against the existing records before a new one
# is written, and across a directory whenever it is loaded -- so a table cannot come to hold rows
# that were scored on different things.


def task_identity(volumes: list[dict[str, Any]], metric: str, key: str) -> dict[str, Any]:
    """The comparable core of a task, in a canonical form that survives a JSON round trip."""
    return {
        "volumes": sorted(
            (
                {
                    "name": str(v.get("name")),
                    "path": None if v.get("path") is None else str(v.get("path")),
                    "label_key": v.get("label_key"),
                    "bounding_box": (
                        None if v.get("bounding_box") is None
                        else [[int(a), int(b)] for a, b in v["bounding_box"]]
                    ),
                }
                for v in volumes
            ),
            key=lambda v: v["name"],
        ),
        "metric": metric,
        "key": key,
    }


def identity_differences(reference: dict[str, Any], other: dict[str, Any]) -> list[str]:
    """Human-readable differences between two identities; empty when they agree."""
    out: list[str] = []
    if (reference["metric"], reference["key"]) != (other["metric"], other["key"]):
        out.append(
            f"ranking metric {other['metric']}.{other['key']} vs "
            f"{reference['metric']}.{reference['key']}"
        )
    mine = {v["name"]: v for v in other["volumes"]}
    theirs = {v["name"]: v for v in reference["volumes"]}
    if set(mine) != set(theirs):
        out.append(f"volumes {sorted(mine)} vs {sorted(theirs)}")
    for name in sorted(set(mine) & set(theirs)):
        for field_name in ("path", "label_key", "bounding_box"):
            if mine[name][field_name] != theirs[name][field_name]:
                out.append(
                    f"{name}: {field_name} {mine[name][field_name]!r} vs "
                    f"{theirs[name][field_name]!r}"
                )
    return out


def assert_same_task(task_name: str, reference: Submission, other: Submission,
                     reference_path: Path | str, other_path: Path | str) -> None:
    """Refuse two records under one task name that were scored on different things."""
    differences = identity_differences(reference.identity(), other.identity())
    if differences:
        listing = "\n".join(f"    {d}" for d in differences)
        raise ValueError(
            f"task {task_name!r}: {other_path} was not scored on the same task as "
            f"{reference_path}:\n{listing}\nA task is its reported volumes (with their ground "
            "truth) and its ranking metric; anything scored on a different set or ranked "
            "differently needs its own task_name. Post-processor, truth route and fit split may "
            "differ and are shown in the table."
        )


# --------------------------------------------------------------------------- the on-disk layout
#
# One directory per task, holding that task's records and its own rendered table:
#
#     leaderboard/<task_name>/records/<identifier>.json
#     leaderboard/<task_name>/README.md
#
# Task-major rather than kind-major (`records/<task>/` beside one shared `README.md`) because a
# task is the unit everything else operates on. A record belongs to exactly one task, a table
# ranks within exactly one task, and two tasks are never comparable -- `lmd_ssl_v1_neuron_instance`
# averages four volumes at 8 nm while `lmd_ssl_v1_zebrafish_instance` scores one, and putting them
# under one heading in one file invited exactly the comparison both refuse. Splitting the directory
# makes adding a task a new directory instead of an edit to a shared file, and makes a task's
# records and its table move, diff and review together.


def task_dir(root: Path, task_name: str) -> Path:
    """`<root>/<task_name>` -- everything belonging to one task."""
    return root / task_name


def records_dir(root: Path, task_name: str) -> Path:
    return task_dir(root, task_name) / "records"


def readme_path(root: Path, task_name: str) -> Path:
    return task_dir(root, task_name) / "README.md"


def task_names(root: Path) -> list[str]:
    """Every task with a records directory under `root`, sorted.

    Read from the directory tree rather than from the records themselves so a task that has a
    directory but no submissions yet still renders an (empty) table instead of vanishing.
    """
    if not root.is_dir():
        return []
    return sorted(
        entry.name for entry in root.iterdir()
        if entry.is_dir() and (entry / "records").is_dir()
    )


def _load(path: Path) -> Submission:
    payload = json.loads(path.read_text())
    version = int(payload.get("schema_version", 0))
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"{path} was written by a newer mia-evals (schema {version} > {SCHEMA_VERSION}); "
            "update this checkout rather than rendering an incomplete table"
        )
    return Submission(**payload)


def load_record(path: Path) -> Submission:
    """One record file, validated for schema version."""
    return _load(path)


def load_task(root: Path, task_name: str) -> list[Submission]:
    """One task's submissions.

    The task name comes from the directory, and a record claiming a different one is an error
    rather than a silent regroup: it means a record was written or moved into the wrong task, and
    rendering it under the directory's name would publish it in a table it was not scored for.
    Likewise a record whose test volumes or ranking metric differ from the others' (see
    `task_identity`): it belongs to a different task, whatever its file says.
    """
    directory = records_dir(root, task_name)
    if not directory.is_dir():
        return []
    submissions: list[Submission] = []
    first: tuple[Path, Submission] | None = None
    for path in sorted(directory.glob("*.json")):
        submission = _load(path)
        if submission.task_name != task_name:
            raise ValueError(
                f"{path} is under {task_name!r} but its record says task_name="
                f"{submission.task_name!r}. Move it to the directory it belongs to."
            )
        # Every record of a task must have been scored on the same thing; the first one on disk
        # is the reference only because *some* record has to be, and any disagreement is named.
        if first is None:
            first = (path, submission)
        else:
            assert_same_task(task_name, first[1], submission, first[0], path)
        submissions.append(submission)
    return submissions


def load_records(root: Path) -> dict[str, list[Submission]]:
    """Every record under `root`, grouped by task name.

    A record whose schema version is newer than this code's is an error rather than a skip: quietly
    omitting a submission would make the leaderboard silently incomplete, which is worse than
    failing to render it.
    """
    return {name: load_task(root, name) for name in task_names(root)}
