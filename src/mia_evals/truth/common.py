"""What every benchmark-data builder shares: OME wrappers over read-only arrays, skeletons, hashes.

**Wrappers, not copies.** A release array is zarr v2 with funlib-style `offset` / `resolution`
attributes on each level and no OME-NGFF `multiscales`, which is what `miao` reads (and so what
mia-train's `predict.py` reads through it) and what the scorer takes an artifact's geometry from. A
wrapper is a zarr v2 group of our own holding that metadata, whose levels are symlinks into the
release. The group directories are real -- the metadata has to live somewhere we can write -- and
only the level arrays are links.

**Translations are voxel centres, relative to the reference level's.** A funlib array records the
world position of its first voxel's *corner* (`offset`), so its first voxel's *centre* sits half a
voxel in. Declaring each array's translation as `offset + (scale - reference_scale) / 2` puts the
reference level's centres at `offset + i * scale` and every coarser array where mean-downsampling
put it -- the same convention lmd stores and mia-train's artifacts use, so a neuroglancer view lines
the wrapper up with a prediction made on it.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

AXES_ZYX = [{"name": axis, "type": "space", "unit": "nanometer"} for axis in "zyx"]
OME_VERSION = "0.4"                   # the zarr v2 flavour of OME-NGFF; 0.5 requires zarr v3 arrays
SKELETON_FORMAT = "mia-evals skeleton v1"


def sha256(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Write `payload`, refusing to change a file that already holds something else."""
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() == text:
            return
        raise FileExistsError(f"{path} exists with different content; remove it to rebuild")
    path.write_text(text)


def link(target: Path, at: Path) -> None:
    """Symlink `at` -> `target`; an existing identical link is fine, anything else is refused."""
    if not target.exists():
        raise FileNotFoundError(f"link target {target} does not exist")
    if at.is_symlink():
        if Path(os.readlink(at)) == target:
            return
        raise FileExistsError(f"{at} already links to {os.readlink(at)}, not {target}")
    if at.exists():
        raise FileExistsError(f"{at} exists and is not a link")
    at.symlink_to(target, target_is_directory=True)


def funlib_level(array: Path) -> dict[str, Any]:
    """Shape, chunks, dtype, `offset` and `resolution` of one funlib-style zarr v2 array."""
    meta = json.loads((array / ".zarray").read_text())
    attrs = json.loads((array / ".zattrs").read_text())
    return {
        "shape": [int(s) for s in meta["shape"]],
        "chunks": [int(c) for c in meta["chunks"]],
        "dtype": str(meta["dtype"]),
        "offset": [float(o) for o in attrs.get("offset", [0.0] * len(meta["shape"]))],
        "resolution": [float(r) for r in attrs["resolution"]],
    }


def multiscales(name: str, datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"version": OME_VERSION, "name": name, "axes": AXES_ZYX, "datasets": datasets}]


def dataset(path: str, scale: Sequence[float], translation: Sequence[float]) -> dict[str, Any]:
    return {
        "path": path,
        "coordinateTransformations": [
            {"type": "scale", "scale": [float(s) for s in scale]},
            {"type": "translation", "translation": [float(t) for t in translation]},
        ],
    }


def make_group(path: Path, attrs: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / ".zgroup", {"zarr_format": 2})
    write_json(path / ".zattrs", attrs)


def wrap_pyramid(
    group: Path,
    name: str,
    levels: Sequence[tuple[str, Path]],
    reference_resolution: Sequence[float],
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """An OME multiscale group at `group` whose datasets `(name, source)` are linked funlib arrays.

    Each array's translation is `offset + (resolution - reference_resolution) / 2` (see the module
    docstring). Returns what was linked, for the manifest.
    """
    datasets, linked = [], []
    for level, source in levels:
        meta = funlib_level(source)
        translation = [
            o + (r - base) / 2.0
            for o, r, base in zip(meta["offset"], meta["resolution"], reference_resolution,
                                  strict=True)
        ]
        datasets.append(dataset(level, meta["resolution"], translation))
        linked.append({"level": level, "target": str(source), **meta})
    make_group(group, {"multiscales": multiscales(name, datasets), **(extra or {})})
    for level, source in levels:
        link(source, group / level)
    return linked


def write_skeleton(path: Path, graph: Any) -> dict[str, Any]:
    """Pickle a skeleton graph, or check an existing file is byte-identical; its manifest entry."""
    payload = pickle.dumps(graph, protocol=5)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"{path} exists with different content; remove it to rebuild")
    else:
        path.write_bytes(payload)
    return {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
