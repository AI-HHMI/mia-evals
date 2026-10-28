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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

FILEGLANCER = "https://fileglancer.int.janelia.org"
SHARES_API = f"{FILEGLANCER}/api/file-share-paths"
NEUROGLANCER = f"{FILEGLANCER}/neuroglancer/#!"

#: Neuroglancer memory budgets written into every view, in bytes. The viewer's defaults are 1 GB of
#: GPU memory and 2 GB of system memory, far too little for our segmentation layers: they have one
#: resolution level and 256^3 chunks, 128 MiB decoded for a uint64 truth array and 64 MiB for a
#: uint32 labelling. With a whole 896^3 block in view, the three-panel layout crosses 37 chunks per
#: segmentation layer, 2.3 GiB for the labelling alone, and a chunk the viewer cannot keep is simply
#: not drawn: it shows as a chunk-shaped hole, and which chunks lose out changes from load to load.
#: These are budgets, not reservations, so a zoomed-in view uses far less; someone on a smaller
#: machine can lower them in the viewer's settings panel for their session. Even these do not cover
#: the heaviest view (truth switched on, whole block, ~7 GiB): smaller chunks or a coarser level in
#: the arrays themselves are the fix that removes the holes for everyone.
GPU_MEMORY_LIMIT = 4_000_000_000
SYSTEM_MEMORY_LIMIT = 8_000_000_000
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


@dataclass(frozen=True)
class Link:
    """One entry of a row's links cell, as both the README table and the HTML page show it."""

    name: str
    path: str
    #: The fileglancer page, or None when no share covers the path.
    url: str | None
    #: Deleted from storage this machine mounts; see `known_missing`.
    missing: bool

    @property
    def label(self) -> str:
        """The entry as text: its name, `name (missing)`, or `name: path` when nothing serves it."""
        if self.missing:
            return f"{self.name} ({MISSING})"
        if self.url is None:
            return f"{self.name}: {self.path}"
        return self.name


def link_entries(entries: list[tuple[str, Path | None]], shares: dict[str, str] | None,
                 missing: Any = known_missing) -> list[Link]:
    """The links cell's entries, in order; a target the record never named gets none."""
    return [Link(name, str(path), fileglancer_url(path, shares), bool(missing(path)))
            for name, path in entries if path is not None]


def markdown_links(links: list[Link]) -> str:
    """The links cell as markdown: `[artifacts](url) · [checkpoint](url)`, `—` when empty."""
    parts = []
    for link in links:
        if link.missing:
            parts.append(link.label)
        elif link.url is None:
            parts.append(f"{link.name}: `{link.path}`")
        else:
            parts.append(f"[{link.name}]({link.url})")
    return " · ".join(parts) if parts else "—"


def link_cell(entries: list[tuple[str, Path | None]], shares: dict[str, str] | None,
              missing: Any = known_missing) -> str:
    """One table cell: `[artifacts](url) · [checkpoint](url)`, `name (missing)` for a deleted
    target, nothing for a record that never named one."""
    return markdown_links(link_entries(entries, shares, missing))


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


#: OME-NGFF units as (factor, SI unit), the form neuroglancer converts a store's units to; a
#: dimension the view declares must match the one neuroglancer derives from the source.
SI_UNITS = {
    "nanometer": (1e-9, "m"), "micrometer": (1e-6, "m"),
    "nanosecond": (1e-9, "s"), "microsecond": (1e-6, "s"), "millisecond": (1e-3, "s"),
    "second": (1.0, "s"), "minute": (60.0, "s"), "hour": (3600.0, "s"), "day": (86400.0, "s"),
}


def pinned_axes(image: Path, fixed: dict[str, int] | None) -> dict[str, tuple[list[Any], float]]:
    """Neuroglancer dimension and position of each axis a volume pins, from its image group.

    A volume pinned with `fixed_axes` is one frame of a time series, but the raw layer shows the
    whole series, and a view placed only in z, y and x opens on whatever frame neuroglancer picks.
    Placing it on the frame needs that axis declared as the store declares it. The position is the
    frame's index (plus any translation), which lies inside the frame whether a viewer takes
    integer coordinates as voxel corners or centres. Empty when nothing is pinned, or when the
    image's metadata does not say what a pinned axis is.
    """
    if not fixed:
        return {}
    if (image / "zarr.json").is_file():
        attrs = json.loads((image / "zarr.json").read_text()).get("attributes") or {}
        attrs = attrs.get("ome") or attrs
    elif (image / ".zattrs").is_file():
        attrs = json.loads((image / ".zattrs").read_text())
    else:
        return {}
    scales = attrs.get("multiscales") or []
    if not scales or not scales[0].get("datasets"):
        return {}
    axes = [a if isinstance(a, dict) else {"name": a} for a in scales[0].get("axes") or []]
    names = [a.get("name") for a in axes]
    scale, shift = [1.0] * len(axes), [0.0] * len(axes)
    for t in scales[0]["datasets"][0].get("coordinateTransformations") or []:
        if t.get("type") == "scale":
            scale = [float(v) for v in t["scale"]]
        elif t.get("type") == "translation":
            shift = [float(v) for v in t["translation"]]
    out: dict[str, tuple[list[Any], float]] = {}
    for axis, index in fixed.items():
        if axis not in names:
            return {}
        i = names.index(axis)
        unit = axes[i].get("unit")
        if unit and unit not in SI_UNITS:
            return {}
        factor, si = SI_UNITS[unit] if unit else (1.0, "")
        out[axis] = ([scale[i] * factor, si], shift[i] / scale[i] + float(index))
    return out


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
                       unfiltered: str | None = None,
                       pinned: dict[str, tuple[list[Any], float]] | None = None) -> str:
    """A viewer URL over three layers: the raw image, the prediction, the ground truth.

    `overlay`: one viewer, the prediction drawn over the raw, the truth present but hidden until
    its tab is clicked. `side_by_side`: two viewers sharing position and zoom, truth over raw on
    the left and prediction over raw on the right. Centred on the prediction when its OME
    metadata is readable. Sources use fileglancer's `<url>/|zarr3:` form. `unfiltered`, when
    given, is the producer's output before post-processing, added as a hidden layer. The viewer's
    memory budgets are raised to `GPU_MEMORY_LIMIT` / `SYSTEM_MEMORY_LIMIT`.
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
    state: dict[str, Any] = {"layers": layers, "selectedLayer": {"visible": True, "layer": name},
                             "gpuMemoryLimit": GPU_MEMORY_LIMIT,
                             "systemMemoryLimit": SYSTEM_MEMORY_LIMIT}
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
        # Axes the raw image has and the prediction lacks (the frame of a time series), placed
        # where the prediction was made; see `pinned_axes`.
        for axis, (dimension, position) in (pinned or {}).items():
            state["dimensions"][axis] = dimension
            state["position"].append(position)
    return NEUROGLANCER + urllib.parse.quote(json.dumps(state, separators=(",", ":")))


def view_entries(producer: dict[str, Any], postprocess: dict[str, Any], config: dict[str, Any],
                 keys: dict[str, str], label: str, exists: Any = os.path.exists
                 ) -> list[tuple[str, str, str, bool]]:
    """Per volume: (volume, overlay URL, side-by-side URL, scored?), for the volumes the shares
    cover. The labelling shown is the post-processed one the row was scored on when the record
    names it and it still exists (`scored` True); otherwise the producer's output as written,
    which for a labelling is the one before the size filter and for affinities nothing a
    segmentation layer can show. A task that scored against the store's label array shows that
    array as the truth: a semantic task, which scored on the array's own grid, and an instance
    task that reads its truth from the store (`truth_kind = "instances"`). A `.gt.zarr` a producer
    left beside its artifact is not what such a task scored, and may carry no geometry to place
    it by. Any other task shows the `<volume>.gt.zarr` beside the producer's artifact, the
    resampled truth an `instances_resampled` task scores against."""
    volumes = {v["name"]: v for v in (config.get("volumes") or []) if isinstance(v, dict)}
    image_key = (producer.get("artifact_attrs") or {}).get("source_image_key") or "raw"
    scored_paths = postprocess.get("scored_artifacts") or {}
    truth_kind = ((config.get("task") or {}).get("kwargs") or {}).get("truth_kind")
    semantic = (config.get("task") or {}).get("name") == "semantic_seg"
    out = []
    for volume, artifact in sorted((producer.get("artifacts") or {}).items()):
        entry = volumes.get(volume) or {}
        store = entry.get("path")
        if not store or not exists(artifact):
            continue
        scored = scored_paths.get(volume)
        shown = scored if scored and exists(scored) else artifact
        raw = share_url(Path(store) / image_key, keys)
        prediction = share_url(shown, keys)
        if raw is None or prediction is None:
            continue
        unfiltered = None
        # The producer's labelling before post-processing, when that changed it: never for
        # affinities or class scores (not a segmentation layer), nor for `identity` with no add-on
        # applied (every fitted value 0 or none fitted), whose copy is the same labels.
        unchanged = postprocess.get("name") == "identity" and not any(
            (postprocess.get("params") or {}).values()
        )
        if (shown != artifact and producer.get("kind") not in ("affinity", "class_scores")
                and not unchanged):
            unfiltered = share_url(artifact, keys)
        truth_path = Path(artifact).with_name(f"{volume}.gt.zarr")
        if (semantic or truth_kind == "instances") and entry.get("label_key"):
            truth = share_url(Path(store) / str(entry["label_key"]), keys)
        elif exists(truth_path):
            truth = share_url(truth_path, keys)
        else:
            truth = None
        transform = ome_transform(Path(shown))
        window = intensity_window(config, volume)
        pinned = pinned_axes(Path(store) / image_key, entry.get("fixed_axes"))
        out.append((
            volume,
            neuroglancer_state(raw, prediction, truth, transform, label, "overlay", window,
                               unfiltered, pinned),
            neuroglancer_state(raw, prediction, truth, transform, label, "side_by_side", window,
                               unfiltered, pinned),
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
