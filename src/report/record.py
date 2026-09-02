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
    #: The task config, expanded.
    config: dict[str, Any]
    #: Volume name -> metric name -> result dict, before aggregation. Kept because an aggregate
    #: alone cannot distinguish "one modality failed outright" from "all four were mediocre", and
    #: on an 8-volume eval set spanning three modalities that is the difference that matters.
    per_volume: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    #: Copied from the producing run so the entry survives its run directory being deleted.
    provenance: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=_component_versions)
    schema_version: int = SCHEMA_VERSION
    label: str = ""

    def identifier(self) -> str:
        """A filesystem-safe name for this submission, stable across re-runs of the same eval."""
        if self.label:
            return self.label
        artifacts = self.producer.get("artifacts") or {}
        run = str(
            self.producer.get("run")
            or next(iter(artifacts.values()), None)
            or "submission"
        )
        step = self.producer.get("step")
        stem = Path(run).name.replace("/", "_")
        return f"{stem}_step{step}" if step is not None else stem

    def write(self, root: Path) -> Path:
        """Write to `<root>/<task_name>/<identifier>.json`, creating directories as needed."""
        directory = root / self.task_name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.identifier()}.json"
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=False) + "\n")
        return path


def load_records(root: Path) -> dict[str, list[Submission]]:
    """Every record under `root`, grouped by task name.

    A record whose schema version is newer than this code's is an error rather than a skip: quietly
    omitting a submission would make the leaderboard silently incomplete, which is worse than
    failing to render it.
    """
    grouped: dict[str, list[Submission]] = {}
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text())
        version = int(payload.get("schema_version", 0))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"{path} was written by a newer mia-evals (schema {version} > {SCHEMA_VERSION}); "
                "update this checkout rather than rendering an incomplete table"
            )
        submission = Submission(**payload)
        grouped.setdefault(submission.task_name, []).append(submission)
    return grouped
