"""The LSD paper's zebra finch (j0126) neurite tracing benchmark, as mia-evals benchmark data.

    python -m truth.lsd_zebrafinch --out /groups/miaai/miaai/mia-evals-data/zebrafinch_j0126

Sheridan et al. 2023 (Nat Methods 20:295) scored every method on the j0126 zebra finch volume with
hand-traced skeletons: 50 for testing, 12 for choosing thresholds. Their release sits on the Janelia
cluster, read-only; this builds, beside it:

    zebrafinch_j0126.zarr/             OME-Zarr 0.4 wrapper over the authors' realigned volume
        raw/s0..s6                     -> zebrafinch_realigned.zarr/volumes/raw/s*        20x9x9 nm
        labels/neuropil_mask/s0..s6    -> .../volumes/ffn_neuropil_mask/s*               20x18x18 nm
        skeletons/<volume>.pkl         the ground truth, one per (region, split)
    ffn_januszewski2018/<volume>.zarr  FFN's segmentation as a reference artifact, one per region
    manifest.json, README.md           what was built from what, with hashes and this command

**The ground truth is the node sets LSD scored, not the traced NML files.** For each region the
release's database dump holds, per skeleton node, a `masked` flag (inside the FFN neuropil mask)
and a `component_id`: the skeleton cut to the region and the mask, its pieces relabelled as
connected components. LSD's evaluation computes expected run length over those components, so a
correct segmentation that keeps two pieces of one neurite (split by a masked cell body) in one
segment is scored as a merge -- unless the segmentation is masked and relabelled the same way,
which is what the released FFN volumes are. Measured on 2026-09-30 and asserted again here: every
node position is a whole number of 20x9x9 nm voxels, the kept nodes are exactly the masked nodes
inside the region, the component ids are exactly the connected components of the kept graph, and
no component is numbered 0 (funlib's background). On the benchmark region a perfect segmentation
scores an expected run length of 240.66 um, which puts FFN's published 16.747 um at nERL 0.0696 --
LSD Supplementary Table 2 prints 0.070. The NML node ids are not the database's (they collide by
accident), so provenance runs through the dump.

**Axes are the volume's own, z, y, x.** `index_position` is the absolute voxel of the wrapper's
`s0`, exactly `nm / (20, 9, 9)`; the data configs set `output_axes: lczyx` so that a bounding box,
an artifact's origin and a skeleton node all count in the same order.

Two regions, because two tasks use them: the benchmark region (478 gigavoxels) and the 11 um cube
at its centre. The rest of LSD's ladder is one line each in `ROIS`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .common import (
    SKELETON_FORMAT,
    dataset,
    funlib_level,
    link,
    make_group,
    multiscales,
    sha256,
    wrap_pyramid,
    write_skeleton,
)

RELEASE = Path("/nrs/funke/sheridana/lsd_data_upload/zebrafinch/testing")
REALIGNED = Path("/nrs/funke/sheridana/zebrafinch/zebrafinch_realigned.zarr/volumes")
RESOLUTION = (20.0, 9.0, 9.0)                  # z, y, x nm: the realigned volume's s0
LEVELS = tuple(f"s{k}" for k in range(7))
WRAPPER = "zebrafinch_j0126.zarr"
FFN_RUN = "ffn_januszewski2018"

#: LSD's regions, daisy world nanometres (z, y, x): (release name, offset, shape). Copied from the
#: authors' lsd_experiments/scripts/configs/zebrafinch/rois/zebrafinch_<release name>.json.
ROIS: dict[str, tuple[str, tuple[int, int, int], tuple[int, int, int]]] = {
    "benchmark": ("benchmark_roi", (4000, 7200, 4500), (106000, 83700, 87300)),
    "11um": ("11_micron_roi", (50800, 43200, 44100), (10800, 10800, 10800)),
}
SPLITS = {"test": "testing", "val": "validation"}

PUBLIC_SOURCES = {
    "skeletons, masks, FFN and LSD segmentations": "s3://open-neurodata/funke/zebrafinch/ "
    "(the 'Data download' notebook of https://github.com/funkelab/lsd)",
    "raw (rawdata_realigned)":
        "gs://j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/",
}


def volume_name(roi: str, split: str) -> str:
    return f"zebrafinch_{roi}_{split}"


def roi_box(roi: str) -> list[list[int]]:
    """[[lo, hi], ...] in s0 voxels, z, y, x."""
    _, offset, shape = ROIS[roi]
    box = []
    for o, s, r in zip(offset, shape, RESOLUTION, strict=True):
        if o % r or s % r:
            raise ValueError(f"region {roi} is not a whole number of voxels: {offset}, {shape}")
        box.append([int(o // r), int((o + s) // r)])
    return box


def read_bson(path: Path) -> list[dict[str, Any]]:
    import bson  # pymongo's; only this builder needs it

    with open(path, "rb") as handle:
        return list(bson.decode_file_iter(handle))


def database(split: str) -> Path:
    """The split's dump directory; beside it sits the authors' `db_dump.py`, not a database."""
    name = SPLITS[split]
    dump = f"zebrafinch_gt_skeletons_new_gt_9_9_20_{name}"
    return RELEASE / "ground_truth" / name / "consolidated" / dump


def region_skeleton(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]], split: str, roi: str
) -> tuple[nx.Graph, dict[str, Any]]:
    """The nodes LSD scored in this region and split, as a graph whose `id` is the component."""
    base = database(split)
    release_name = ROIS[roi][0]
    files = {
        "components": base / f"nodes_zebrafinch_components_{release_name}_masked.bson",
        "mask": base / f"nodes_zebrafinch_mask_{release_name}_masked.bson",
    }
    component = {int(d["id"]): int(d["component_id"]) for d in read_bson(files["components"])}
    masked = {int(d["id"]): bool(d["masked"]) for d in read_bson(files["mask"])}
    box = roi_box(roi)
    voxel_nm = np.asarray(RESOLUTION, dtype=np.int64)

    graph = nx.Graph()
    for node in nodes:
        nid = int(node["id"])
        if component.get(nid, -1) < 0 or not masked.get(nid, False):
            continue
        nm = np.array([node["z"], node["y"], node["x"]], dtype=np.int64)
        if np.any(nm % voxel_nm):
            raise ValueError(f"node {nid} at {nm.tolist()} nm is not on the 20x9x9 nm voxel grid")
        voxel = (nm // voxel_nm).tolist()
        if not all(lo <= v < hi for v, (lo, hi) in zip(voxel, box, strict=True)):
            raise ValueError(f"node {nid} at voxel {voxel} is scored in {roi} but lies outside")
        graph.add_node(
            nid,
            id=component[nid],
            index_position=tuple(int(v) for v in voxel),
            nm_position=tuple(float(v) for v in nm),
            neuron_id=int(node["neuron_id"]),
        )
    for edge in edges:
        u, v = int(edge["source"]), int(edge["target"])
        if u in graph and v in graph:
            graph.add_edge(u, v)

    # The component ids must be exactly the connected components of the kept graph: that is how LSD
    # made them, and the metric counts a segment touching two of them as a merge.
    pieces = [
        frozenset(graph.nodes[n]["id"] for n in piece) for piece in nx.connected_components(graph)
    ]
    ids = {graph.nodes[n]["id"] for n in graph.nodes}
    if any(len(p) != 1 for p in pieces) or len(pieces) != len(ids):
        raise ValueError(
            f"{roi}/{split}: component ids are not the skeleton's connected components"
        )
    if 0 in ids:
        raise ValueError(f"{roi}/{split}: a component is numbered 0, funlib's background")

    lengths: dict[int, float] = {}
    for u, v in graph.edges:
        a, b = graph.nodes[u]["nm_position"], graph.nodes[v]["nm_position"]
        lengths[graph.nodes[u]["id"]] = (
            lengths.get(graph.nodes[u]["id"], 0.0) + float(np.linalg.norm(np.subtract(a, b)))
        )
    cable = sum(lengths.values())
    perfect = sum(v * v for v in lengths.values()) / cable if cable else 0.0
    stats = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": len(ids),
        "neurons": len({graph.nodes[n]["neuron_id"] for n in graph.nodes}),
        "cable_mm": round(cable / 1e6, 6),
        "perfect_erl_um": round(perfect / 1e3, 4),
        "sources": {k: {"path": str(p), "sha256": sha256(p)} for k, p in files.items()},
    }
    graph.graph.update(
        format=SKELETON_FORMAT,
        axes="zyx",
        voxel_size_nm=list(RESOLUTION),
        volume=volume_name(roi, split),
        region=release_name,
        split=split,
        region_box=box,
        source=f"LSD release, {base}",
    )
    return graph, stats


def ffn_artifact(out: Path, roi: str, wrapper: Path) -> dict[str, Any]:
    """FFN's released segmentation of one region, wrapped as a mia-evals `instances` artifact."""
    release_name = ROIS[roi][0]
    source = RELEASE / "segmentations" / "data.zarr" / "volumes" / "FFN" / release_name
    meta = funlib_level(source)
    box = roi_box(roi)
    origin = [lo for lo, _ in box]
    expected = {
        "offset": [float(v) for v in ROIS[roi][1]],
        "resolution": list(RESOLUTION),
        "shape": [hi - lo for lo, hi in box],
    }
    for key, value in expected.items():
        if meta[key] != value:
            raise ValueError(f"{source} has {key} {meta[key]}, expected {value}")
    if meta["dtype"] not in ("<u8", "<u4"):
        raise ValueError(f"{source} is {meta['dtype']}, not an unsigned labelling")

    path = out / f"{volume_name(roi, 'test')}.zarr"
    translation = [o * r for o, r in zip(origin, RESOLUTION, strict=True)]
    make_group(path, {
        "multiscales": multiscales(
            f"{FFN_RUN} {release_name}", [dataset("s0", RESOLUTION, translation)]
        ),
        "kind": "instances",
        "background_id": 0,
        "origin": origin,
        "covers_full_box": True,
        "source_path": str(wrapper),
        "source_image_key": "raw",
        "run": FFN_RUN,
        "mask": "labels/neuropil_mask",
        "convention": (
            "FFN segmentation of j0126 (Januszewski et al. 2018, Nat Methods 15:605), cropped "
            "to the region, restricted to the FFN neuropil mask and relabelled into connected "
            "components by Sheridan et al. 2023; byte-identical to zebrafinch_realigned.zarr/"
            f"volumes/ffn_{release_name}_masked_ffn_relabelled"
        ),
        "source_segmentation": str(source),
    })
    link(source, path / "s0")
    return {"path": str(path), "target": str(source), "origin": origin, **meta}


def verify(wrapper: Path, ffn: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Read everything back the way its consumers will: miao for the wrapper, mia-evals for FFN."""
    from miao.zarr_meta import read_ome_metadata

    from artifact import open_artifact

    report: dict[str, Any] = {}
    for key in ("raw", "labels/neuropil_mask"):
        meta = read_ome_metadata(wrapper, key, "zarr2")
        report[key] = {
            level: {"scale": s.scale_factors, "translation": s.translation, "shape": s.shape}
            for level, s in sorted(meta.scales.items())
        }
    for name, entry in ffn.items():
        artifact = open_artifact(Path(entry["path"]))
        if list(artifact.origin) != entry["origin"] or artifact.kind != "instances":
            raise ValueError(f"{name}: read back as {artifact.kind} at {artifact.origin}")
        report[name] = {"kind": artifact.kind, "shape": list(artifact.spatial_shape),
                        "origin": list(artifact.origin), "axes": artifact.axes,
                        "run": artifact.attrs.get("run")}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="the dataset directory to build")
    parser.add_argument("--rois", nargs="+", default=list(ROIS), choices=list(ROIS))
    args = parser.parse_args()
    out = args.out.resolve()
    wrapper = out / WRAPPER

    make_group(wrapper, {"description": "mia-evals wrapper over the LSD authors' realigned j0126 "
                                        "volume; levels are symlinks, see ../manifest.json"})
    linked = {
        "raw": wrap_pyramid(wrapper / "raw", "raw",
                            [(level, REALIGNED / "raw" / level) for level in LEVELS], RESOLUTION),
    }
    make_group(wrapper / "labels", {"labels": ["neuropil_mask"]})
    linked["labels/neuropil_mask"] = wrap_pyramid(
        wrapper / "labels" / "neuropil_mask", "neuropil_mask",
        [(level, REALIGNED / "ffn_neuropil_mask" / level) for level in LEVELS], RESOLUTION,
        extra={"image-label": {"version": "0.4"}},
    )

    skeletons_dir = wrapper / "skeletons"
    skeletons_dir.mkdir(exist_ok=True)
    skeletons: dict[str, Any] = {}
    for split in SPLITS:
        base = database(split)
        nodes = read_bson(base / "zebrafinch.nodes.bson")
        edges = read_bson(base / "zebrafinch.edges.bson")
        for roi in args.rois:
            graph, stats = region_skeleton(nodes, edges, split, roi)
            name = volume_name(roi, split)
            stats["file"] = write_skeleton(skeletons_dir / f"{name}.pkl", graph)
            stats["region_box_zyx"] = roi_box(roi)
            skeletons[name] = stats
            print(f"{name}: {stats['nodes']} nodes, {stats['components']} components, "
                  f"{stats['cable_mm']:.3f} mm, perfect ERL {stats['perfect_erl_um']:.2f} um",
                  flush=True)

    ffn_dir = out / FFN_RUN
    ffn_dir.mkdir(exist_ok=True)
    ffn = {volume_name(roi, "test"): ffn_artifact(ffn_dir, roi, wrapper) for roi in args.rois}
    readback = verify(wrapper, ffn)

    from report.record import git_commit

    manifest = {
        "built": date.today().isoformat(),
        "command": [Path(sys.executable).name, "-m", "truth.lsd_zebrafinch", *sys.argv[1:]],
        "mia_evals_commit": git_commit(Path(__file__).resolve().parents[2]),
        "release": str(RELEASE),
        "realigned_volume": str(REALIGNED),
        "public_sources": PUBLIC_SOURCES,
        "wrapper": {"path": str(wrapper), "linked": linked},
        "skeletons": skeletons,
        "ffn_reference_artifacts": ffn,
        "readback": readback,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out / "README.md").write_text(README.format(out=out))
    print(f"wrote {out}", flush=True)


README = """# zebrafinch_j0126 -- mia-evals benchmark data

Built by `python -m truth.lsd_zebrafinch --out {out}` from the mia-evals repository;
`manifest.json` records the exact command, commit, sources and hashes. Nothing here copies the
release: the image and mask levels and the FFN arrays are symlinks into /nrs/funke/sheridana
(read-only, not ours). If those disappear, everything is public again from the sources listed in
the manifest.

- `zebrafinch_j0126.zarr/`: OME-Zarr 0.4 wrapper (zarr v2) readable by miao; `raw/` 20x9x9 nm,
  `labels/neuropil_mask/` 20x18x18 nm (the FFN neuropil mask), `skeletons/<volume>.pkl` the ground
  truth (networkx graphs; `id` = LSD's per-region component, `index_position` = absolute z, y, x
  voxel).
- `ffn_januszewski2018/<volume>.zarr`: FFN's released segmentation per region, as an `instances`
  artifact (already masked and relabelled by the LSD authors).

Used by the mia-evals tasks `zebrafinch_neurite_tracing` and `zebrafinch_neurite_tracing_11um`.
"""


if __name__ == "__main__":
    main()
