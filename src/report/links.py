"""Browser links for a leaderboard row: the scored artifacts and the checkpoint, via fileglancer.

Fileglancer (https://fileglancer.int.janelia.org, Janelia login) shows a cluster path at
`/browse/<share>/<path below the share's mount>`, where the shares are the file-share paths it
publishes at `/api/file-share-paths` (`/nrs/scicompsoft` -> `nrs_scicompsoft`,
`/groups/scicompsoft/home` -> `groups_scicompsoft_home`, ...). The mapping is by longest matching
mount path, so it is taken from that list when the API answers and from the fixed naming rule the
list follows when it does not (a render off the network still produces the same links).

Both links are optional: a record from a producer whose files are not on this cluster simply has
no paths, and a row whose files were deleted -- scratch is reclaimed once a number is recorded --
says so instead of linking to nothing. Existence is checked when the table is rendered, so a
committed table goes stale when files disappear and `--check` reports it, which is the point.
The check is made only where it can be made: on a machine that does not mount the file's storage
at all (a collaborator's laptop, CI), the link is kept as it was, so a render or `--check` away
from the cluster neither declares everything missing nor rewrites the tables.

**Viewer links.** Fileglancer serves a directory's files to its hosted neuroglancer through a
*share* it creates for that directory: `https://fileglancer.int.janelia.org/files/<key>/<absolute
path without the leading slash>`, the key covering everything below the shared directory (checked
2026-09-15: no login is needed, only reachability of the internal host). `leaderboard/
fileglancer_shares.json` maps shared directories to their keys; with a share covering the raw
store and one covering the artifacts, `view_links` emits one neuroglancer state per volume that
opens the raw image, the prediction and the ground truth together. Alignment is the OME-Zarr
metadata's job (predict.py writes it; the raw stores have it), so the state carries no transform.
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

FILEGLANCER = "https://fileglancer.int.janelia.org"
SHARES_API = f"{FILEGLANCER}/api/file-share-paths"
NEUROGLANCER = f"{FILEGLANCER}/neuroglancer/#!"
SHARE_KEYS = "fileglancer_shares.json"     # untracked, beside the leaderboard's task directories
MISSING = "missing"

#: `(regex over the absolute path, share-name template)` -- the naming rule the published list
#: follows, for the mounts this cluster's work lands on. Checked against the live list in tests.
FALLBACK_RULES: tuple[tuple[str, str], ...] = (
    (r"^/nrs/([^/]+)", r"nrs_\1"),
    (r"^/nearline/([^/]+)", r"nearline_\1"),
    (r"^/groups/([^/]+)/home", r"groups_\1_home"),
    (r"^/groups/([^/]+)/([^/]+)", r"groups_\1_\2"),
)


def fetch_shares(timeout: float = 3.0) -> dict[str, str] | None:
    """`mount path -> share name` from fileglancer, or None when it cannot be reached."""
    try:
        with urllib.request.urlopen(SHARES_API, timeout=timeout) as response:
            payload = json.load(response)
    except Exception:  # noqa: BLE001 -- any failure means "use the rule", nothing to recover
        return None
    items = payload if isinstance(payload, list) else next(
        (v for v in payload.values() if isinstance(v, list)), []
    )
    shares = {}
    for item in items:
        mount, name = item.get("mount_path"), item.get("name")
        if mount and name:
            shares[str(mount).rstrip("/")] = str(name)
    return shares or None


@lru_cache(maxsize=1)
def default_shares() -> dict[str, str] | None:
    return fetch_shares()


def fileglancer_url(
    path: str | os.PathLike[str], shares: dict[str, str] | None = None
) -> str | None:
    """The fileglancer page for an absolute cluster path, or None if no share covers it."""
    text = str(Path(path))
    if shares:
        covering = (m for m in shares if text == m or text.startswith(m + "/"))
        best = max(covering, key=len, default=None)
        if best is not None:
            rest = text[len(best):].lstrip("/")
            return f"{FILEGLANCER}/browse/{shares[best]}" + (f"/{rest}" if rest else "")
    for pattern, template in FALLBACK_RULES:
        match = re.match(pattern, text)
        if match:
            name = match.expand(template)
            rest = text[match.end():].lstrip("/")
            return f"{FILEGLANCER}/browse/{name}" + (f"/{rest}" if rest else "")
    return None


def artifact_directory(producer: dict[str, Any]) -> Path | None:
    """The one directory holding the scored artifacts, or None if the record names none."""
    artifacts = producer.get("artifacts") or {}
    paths = [str(p) for p in artifacts.values() if p]
    if not paths:
        return None
    parents = {str(Path(p).parent) for p in paths}
    return Path(os.path.commonpath(sorted(parents))) if len(parents) > 1 else Path(parents.pop())


def checkpoint_directory(producer: dict[str, Any], provenance: dict[str, Any]) -> Path | None:
    """`<run_dir>/checkpoints/step_<step>`, or an explicit `producer.checkpoint`, or None.

    `run_dir_missing` is the scorer's own note that the run directory was already gone when the
    row was scored; the link is still built so the table can say "missing" rather than nothing.
    """
    explicit = producer.get("checkpoint")
    if explicit:
        return Path(str(explicit))
    run_dir = provenance.get("run_dir") or provenance.get("run_dir_missing")
    step = producer.get("step")
    if not run_dir or step is None:
        return None
    return Path(str(run_dir)) / "checkpoints" / f"step_{int(step)}"


def mount_root(path: str | os.PathLike[str]) -> Path:
    """The storage mount a cluster path lives on: `/nrs/<group>`, `/groups/<group>/<sub>`, ...,
    or the filesystem root when the path follows no known pattern."""
    text = str(Path(path))
    for pattern, _ in FALLBACK_RULES:
        match = re.match(pattern, text)
        if match:
            return Path(match.group(0))
    return Path("/")


def known_missing(path: str | os.PathLike[str]) -> bool:
    """True only when the file is absent AND its storage is mounted here, so absence means
    deletion rather than "this machine cannot see the cluster"."""
    return mount_root(path).is_dir() and not os.path.exists(path)


def link_cell(entries: list[tuple[str, Path | None]], shares: dict[str, str] | None,
              missing: Any = known_missing) -> str:
    """One table cell: `[artifacts](url) · [checkpoint](url)`, `name (missing)` for a deleted
    target, nothing for a record that never named one."""
    parts = []
    for name, path in entries:
        if path is None:
            continue
        url = fileglancer_url(path, shares)
        if missing(path):
            parts.append(f"{name} ({MISSING})")
        elif url is None:
            parts.append(f"{name}: `{path}`")
        else:
            parts.append(f"[{name}]({url})")
    return " · ".join(parts) if parts else "—"


# ------------------------------------------------------------------------------ viewer links


def load_share_keys(path: Path) -> dict[str, str]:
    """`shared directory -> key` from the (git-ignored) key file, or nothing if there is none."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    return {str(Path(k)): str(v) for k, v in (payload.get("shares") or {}).items()}


def views_directory(path: Path) -> Path | None:
    """Where the HTML view pages are written so a data link serves them: the key file's
    `views_dir`, which must lie under one of its shares, or None. May contain `{task}`, which
    `write_views` expands to the task name so every task's page lives in its own directory."""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    target = payload.get("views_dir")
    return Path(str(target)) if target else None


def share_url(path: str | os.PathLike[str], keys: dict[str, str]) -> str | None:
    """The fileglancer data URL of a file under a shared directory, or None if none covers it.

    Symlinks are resolved first: the ground-truth files beside the watershed labellings link to
    another directory, and the server serves the real path.
    """
    real = os.path.realpath(path)
    covering = [d for d in keys if real == d or real.startswith(d.rstrip("/") + "/")]
    if not covering:
        return None
    key = keys[max(covering, key=len)]
    return f"{FILEGLANCER}/files/{key}/{real.lstrip('/')}"


def ome_transform(artifact: Path) -> tuple[list[str], list[float], list[float], list[int]] | None:
    """(axis names, scale in nm, translation in nm, shape) of a single-level OME artifact."""
    meta = artifact / "zarr.json"
    if not meta.is_file():
        return None
    root = json.loads(meta.read_text())
    ome = (root.get("attributes") or {}).get("ome") or {}
    scales = ome.get("multiscales") or []
    if len(scales) != 1 or len(scales[0].get("datasets") or []) != 1:
        return None
    dataset = scales[0]["datasets"][0]
    level = artifact / dataset["path"] / "zarr.json"
    if not level.is_file():
        return None
    axes = [a["name"] for a in scales[0]["axes"] if a.get("type") != "channel"]
    scale = shift = None
    for t in dataset.get("coordinateTransformations", []):
        if t.get("type") == "scale":
            scale = [float(v) for v in t["scale"]][-len(axes):]
        elif t.get("type") == "translation":
            shift = [float(v) for v in t["translation"]][-len(axes):]
    shape = [int(v) for v in json.loads(level.read_text())["shape"]][-len(axes):]
    if scale is None:
        return None
    return axes, scale, shift or [0.0] * len(axes), shape


def intensity_window(config: dict[str, Any], volume: str) -> tuple[float, float] | None:
    """The data config's `normalize_min/max` for a volume, when the config is at hand.

    uint16 stores (the liconn volumes) fill a tiny part of their range, so neuroglancer's default
    window shows them black; the training window is the sensible one to open them with.
    """
    path = config.get("data_config_path")
    if not path or not Path(str(path)).is_file():
        return None
    try:
        import yaml

        entries = yaml.safe_load(Path(str(path)).read_text()).get("volumes") or []
    except Exception:  # noqa: BLE001 -- a window is a convenience, never a failure
        return None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("name") == volume:
            low, high = entry.get("normalize_min"), entry.get("normalize_max")
            if low is not None and high is not None:
                return float(low), float(high)
    return None


def neuroglancer_state(raw: str, prediction: str, truth: str | None,
                       transform: tuple[list[str], list[float], list[float], list[int]] | None,
                       name: str, mode: str = "overlay",
                       window: tuple[float, float] | None = None,
                       unfiltered: str | None = None) -> str:
    """A viewer URL over three layers: the raw image, the prediction, the ground truth.

    `overlay`: one viewer, the prediction drawn over the raw, the truth present but hidden until
    its tab is clicked. `side_by_side`: two viewers sharing position and zoom, truth over raw on
    the left and prediction over raw on the right. Centred on the prediction when its OME
    metadata is readable. Sources use fileglancer's `<url>/|zarr3:` form. `unfiltered`, when
    given, is the producer's output before post-processing, added as a hidden layer.
    """
    image: dict[str, Any] = {"type": "image", "source": f"{raw}/|zarr3:", "name": "raw"}
    if window is not None:
        image["shaderControls"] = {"normalized": {"range": [window[0], window[1]]}}
    layers: list[dict[str, Any]] = [
        image,
        {"type": "segmentation", "source": f"{prediction}/|zarr3:", "name": name},
    ]
    if truth:
        layers.append({"type": "segmentation", "source": f"{truth}/|zarr3:", "name": "truth",
                       "visible": mode == "side_by_side"})
    if unfiltered:
        layers.append({"type": "segmentation", "source": f"{unfiltered}/|zarr3:",
                       "name": "unfiltered", "visible": False})
    state: dict[str, Any] = {"layers": layers, "selectedLayer": {"visible": True, "layer": name}}
    if mode == "side_by_side" and truth:
        state["layout"] = {"type": "row", "children": [
            {"type": "viewer", "layers": ["raw", "truth"], "layout": "xy"},
            {"type": "viewer", "layers": ["raw", name], "layout": "xy"},
        ]}
    else:
        state["layout"] = "4panel-alt"
    if transform is not None:
        axes, scale, shift, shape = transform
        state["dimensions"] = {a: [v * 1e-9, "m"] for a, v in zip(axes, scale, strict=True)}
        state["position"] = [t / v + n / 2 for t, v, n in zip(shift, scale, shape, strict=True)]
    return NEUROGLANCER + urllib.parse.quote(json.dumps(state, separators=(",", ":")))


def view_entries(producer: dict[str, Any], postprocess: dict[str, Any], config: dict[str, Any],
                 keys: dict[str, str], label: str, exists: Any = os.path.exists
                 ) -> list[tuple[str, str, str, bool]]:
    """Per volume: (volume, overlay URL, side-by-side URL, scored?), for the volumes the shares
    cover. The labelling shown is the post-processed one the row was scored on when the record
    names it and it still exists (`scored` True); otherwise the producer's output as written,
    which for a labelling is the one before the size filter and for affinities nothing a
    segmentation layer can show."""
    volumes = {v["name"]: v for v in (config.get("volumes") or []) if isinstance(v, dict)}
    image_key = (producer.get("artifact_attrs") or {}).get("source_image_key") or "raw"
    scored_paths = postprocess.get("scored_artifacts") or {}
    out = []
    for volume, artifact in sorted((producer.get("artifacts") or {}).items()):
        store = (volumes.get(volume) or {}).get("path")
        if not store or not exists(artifact):
            continue
        scored = scored_paths.get(volume)
        shown = scored if scored and exists(scored) else artifact
        raw = share_url(Path(store) / image_key, keys)
        prediction = share_url(shown, keys)
        if raw is None or prediction is None:
            continue
        unfiltered = None
        if shown != artifact and producer.get("kind") != "affinity":
            unfiltered = share_url(artifact, keys)
        truth_path = Path(artifact).with_name(f"{volume}.gt.zarr")
        truth = share_url(truth_path, keys) if exists(truth_path) else None
        transform = ome_transform(Path(shown))
        window = intensity_window(config, volume)
        out.append((
            volume,
            neuroglancer_state(raw, prediction, truth, transform, label, "overlay", window,
                               unfiltered),
            neuroglancer_state(raw, prediction, truth, transform, label, "side_by_side", window,
                               unfiltered),
            shown != artifact,
        ))
    return out


def record_views(producer: dict[str, Any], postprocess: dict[str, Any], config: dict[str, Any],
                 keys: dict[str, str], label: str) -> dict[str, dict[str, Any]]:
    """`view_entries` in the form a record stores: volume -> overlay, side-by-side, what is shown.

    Computed at scoring time from the *scorer's* key file, because that is the machine whose data
    links cover the artifacts being scored. Empty when no key covers them, which the views page
    reports as missing rather than inventing a link that would not resolve.
    """
    return {
        volume: {
            "overlay": overlay,
            "side_by_side": side,
            "shows": "scored" if is_scored else "before size filter",
        }
        for volume, overlay, side, is_scored in view_entries(
            producer, postprocess, config, keys, label
        )
    }
