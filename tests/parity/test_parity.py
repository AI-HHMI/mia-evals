"""The parity gate: `src/` must reproduce the pre-refactor scorer's numbers, exactly.

A refactor can silently change results. This scores one recorded affinity artifact through the
current `src/` runner and requires every value to equal what `mia_score.py` produced on the same
artifact with the same sweep, as recorded in `expected/`.

**What this does and does not prove.** Both paths call the same `utils.connected_components` and
`utils.instance_metrics` -- byte-identical copies of upstream BANIS -- so this cannot catch an error
inside the metric mathematics. It catches errors in everything around it: which region is read, how
the threshold is applied, whether the skeleton is cropped to the scored region, which convention
split and merge are reported under. That is where this repository's real bugs have been. Its first
run found `whole_region` being inferred from the absence of a bounding box, which fed `funlib` an
uncropped 784,783-node skeleton where 9,057 were in scope.

**Why a recorded file rather than running both paths.** Once the two agreed, the recorded numbers
became the cheaper oracle: keeping the old driver alive forever costs ~250 lines of duplicate
scoring logic that must be maintained but is never used in anger. The file records the artifact's
SHA-256, so a fixture can never be silently compared against a different array.

Marked `parity`: needs the `instance` extra and a real artifact on /nrs, and takes minutes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

EXPECTED = Path(__file__).parent / "expected"
#: Keys that legitimately differ or carry no comparable value.
IGNORE = {"logit", "threshold", "whole_region", "skeleton_nodes_full"}

pytestmark = pytest.mark.parity


def artifact_sha256(path: Path, store) -> str:
    """SHA-256 over the raw array bytes, channel by channel.

    Replaces an earlier XOR digest that was close to decorative: XOR of `float16` values widened to
    `uint64` carries at most 16 bits, is order-insensitive, and cancels any value occurring an even
    number of times -- and affinity maps are full of saturated runs. Chunked by channel so a
    7-gigavoxel artifact is never materialised whole just to be hashed.
    """
    digest = hashlib.sha256()
    for channel in range(store.shape[0]):
        digest.update(np.ascontiguousarray(store[channel]).tobytes())
    return digest.hexdigest()


def fixtures() -> list[Path]:
    return sorted(EXPECTED.glob("*.json"))


@pytest.mark.parametrize("fixture_path", fixtures(), ids=lambda p: p.stem)
def test_src_reproduces_the_recorded_numbers(fixture_path, tmp_path):
    zarr = pytest.importorskip("zarr")
    pytest.importorskip("funlib.evaluate")

    expected = json.loads(fixture_path.read_text())
    artifact_path = Path(expected["artifact"])
    if not artifact_path.exists():
        pytest.skip(f"{artifact_path} is not on this filesystem")

    store = zarr.open(str(artifact_path), mode="r")
    if "artifact_sha256" in expected:
        actual = artifact_sha256(artifact_path, store)
        assert actual == expected["artifact_sha256"], (
            f"{artifact_path} does not match the array these numbers were recorded from "
            f"(sha256 {actual[:16]}... vs {expected['artifact_sha256'][:16]}...). The fixture is "
            "pinned by content on purpose: scoring a different array and comparing to these "
            "numbers would report a refactor bug that is really a changed input."
        )

    import components  # noqa: F401
    from config import load_task_config
    from evaluate import build, resolve_artifacts, score_once

    task_config = Path(expected["task_config"])
    if not task_config.is_absolute():
        task_config = Path(__file__).resolve().parents[2] / task_config
    config = load_task_config(task_config)
    task, processor, metrics = build(config)
    volume_name = expected["volume"]
    volumes = tuple(v for v in config.volumes if v.name == volume_name)
    assert volumes, f"{volume_name} absent from {expected['task_config']}"

    # The runner resolves `<volume>.zarr` inside a directory; link rather than copy.
    (tmp_path / f"{volume_name}.zarr").symlink_to(artifact_path)
    artifacts = resolve_artifacts(tmp_path, volumes)

    recorded = {float(r["logit"]): r for r in expected["sweep"]}
    mismatches, compared = [], 0
    for candidate in processor.search_space():
        logit = float(candidate["logit"])
        assert logit in recorded, f"logit {logit} is swept but not recorded; sweeps must match"
        _, per_volume, _ = score_once(
            artifacts, volumes, task, processor, metrics, candidate, tmp_path / "scratch")
        got = per_volume[volume_name][config.rank_by]
        for key in sorted((set(got) & set(recorded[logit])) - IGNORE):
            compared += 1
            if float(got[key]) != float(recorded[logit][key]):
                mismatches.append(
                    f"logit {logit:+g} {key}: recorded {recorded[logit][key]!r}, got {got[key]!r}")

    assert compared > 0, "compared nothing; the fixture and the sweep share no keys"
    assert not mismatches, (
        f"{len(mismatches)} of {compared} values differ from the recorded pre-refactor run:\n  "
        + "\n  ".join(mismatches)
    )


def test_at_least_one_fixture_exists():
    """Guards against the suite passing because it silently collected nothing."""
    assert fixtures(), f"no parity fixtures in {EXPECTED}; the gate would pass vacuously"
