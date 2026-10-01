"""The LSD paper's hemibrain ellipsoid-body neurite tracing benchmark, as mia-evals benchmark data.

    python -m truth.lsd_hemibrain --out /groups/miaai/miaai/mia-evals-data/hemibrain_eb_lsd

Sheridan et al. 2023 (Nat Methods 20:295) scored hemibrain segmentations on three cubes in the
ellipsoid body -- 12, 22 and 17 um, the release's roi_1, roi_2 and roi_3 -- against voxel ground
truth: the whitelisted proofread neurons, constrained to the ellipsoid body, relabelled into
connected components and slightly eroded (`consolidated_ids`). The release has no skeletons, and
the paper reports VOI only. This builds, beside the release (read-only, nothing copied):

    roi_<k>.zarr/                      OME-Zarr 0.4 wrapper over the region's arrays, z, y, x, 8 nm
        raw/s0                         -> roi_<k>/raw: the core and a context margin around it
        labels/consolidated_ids/s0     -> roi_<k>/consolidated_ids: the core only, so not on raw's
                                          voxel grid -- for viewing and the one-time VOI check,
                                          never read by a task as a label array
        labels/eb_mask/s0              -> volumes/mask: the ellipsoid body at 256 nm
        skeletons/<volume>.pkl         the ground truth, derived here
    ffn_januszewski2018/<volume>.zarr  FFN's segmentation of each region, a reference artifact
    manifest.json, README.md           what was built from what, with hashes and this command

**FFN is the release's `FFN/roi_<k>/consolidated_ids`**: FFN's hemibrain segmentation cut to the
core, restricted to the ellipsoid body and relabelled into connected components, as the LSD
authors scored it. Measured on 2026-09-30: its voxel VOI against `consolidated_ids` reproduces
their Supplementary Table 3 (roi_1 0.1297 / 0.0461 against 0.129 / 0.046); it covers the whole
ellipsoid body, not only the ground truth's neurons, so merges stay visible; and the uncut
`neuron_ids` score a merge VOI of 0.97 there, which is what the relabelling removes.

**The skeletons are derived from `consolidated_ids`**: kimimaro's TEASAR on every object, each
branch end cut back inside it (below), then thinned to a node every `DOWNSAMPLE` path steps (about
150 nm, the spacing of the zebrafinch tracings; endpoints and branch points are kept, and every
kept node is an original skeleton voxel). Every node is asserted to lie inside its own object, so a
flawless labelling scores nERL 1.
kimimaro runs block by block, as igneous does at scale: whole-region runs allocate several arrays
of an object's bounding box per worker, and these neurons span the region (roi_1 needed 249 GB).
Blocks overlap by one voxel plane and `fix_borders` puts an object's endpoints on a face where both
neighbours find them, so each object's pieces join on those shared vertices. A neurite running
inside a shared plane is traced twice, so blocks add track. Measured on roi_1, before branch ends
were inset, against a single 1536^3 block (no shared planes, 69 GB, 1.6 h serial): 4.77 mm of
cable unblocked, 5.02 mm in 1024^3 blocks, 5.38 mm in 512^3. The extra track inflates ERL and its
perfect-segmentation ceiling alike, so it barely reaches the ranking number: FFN's roi_1 nERL was
0.8792 on the 1024^3 skeletons and 0.8809 on the unblocked one, with identical merge and split
counts. Hence `BLOCK = 1024`.

**Branch ends are cut back to lie more than `INSET` voxels (16 nm) inside their object.** TEASAR
runs every branch out to its object's surface, and the surfaces are FFN's: the release's neurons
are proofread FFN. A segmentation whose boundary is a voxel off FFN's there puts the end in the
neighbouring segment, and run length counts that as a merge voiding the neighbour's whole run, a
penalty only FFN escapes. Measured on roi_1 (2026-10-01) on uninset skeletons: 64 of gary 5a's 65
merging segments reached a second neuron only through such nodes -- 95% of them tips, 93% within
a voxel of the surface -- and FFN had none. An end moves at most one node spacing along its path,
so a neurite too thin to hold a voxel that deep keeps its end at its most interior voxel there.
Inset, roi_1 loses 1.9% of its cable, 5a keeps 8 such segments and its one real merge, and nERL
goes from 0.417 to 0.828 for 5a and stays 0.879 for FFN.

**The unit of run length is the ground-truth object**, not a connected piece of its skeleton: every
node's `id` is its object. LSD relabelled `consolidated_ids` into connected components before
eroding it, so objects are the counterpart of their zebrafinch components; but the erosion cuts
thin necks -- roi_1's 364 skeletonised objects are 425 pieces at any block size -- and numbering
pieces would make a segmentation that keeps such a neuron whole score as merging them. The
whitelist makes the ground truth sparse (29-48 neurons per region, 30-42% of the voxels), so a
merge with an unlisted neuron is invisible; a merge between two listed ones is not.

**Where the regions are** (measured 2026-09-30 by label-id agreement of 97-100% with lmd's proofread
v1.2 labels): array axes and offsets are z, y, x. The cores, in hemibrain voxels x, y, z, are
roi_1 [24800, 26275) x [25640, 27115) x [17600, 19075), roi_2 [26800, 29525) x [24000, 26725) x
[18800, 21525) and roi_3 [22400, 24500) x [25000, 27100) x [19210, 21310): inside lmd's
ellipsoid-body crop and largely inside gary_comparison's training blocks. The raw is the release's
own: the same image as lmd's copy (correlation 0.98-0.99) at a different contrast.
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
    wrap_pyramid,
    write_skeleton,
)

RELEASE = Path("/nrs/funke/sheridana/lsd_data_upload/hemi/testing")
GROUND_TRUTH = RELEASE / "ground_truth" / "data.zarr" / "volumes"
FFN = RELEASE / "segmentations" / "data.zarr" / "volumes" / "FFN"
FFN_RUN = "ffn_januszewski2018"
RESOLUTION = (8.0, 8.0, 8.0)
#: The release's region names and the paper's edge lengths.
ROIS = {"roi_1": "12 um", "roi_2": "22 um", "roi_3": "17 um"}
#: kimimaro's TEASAR settings, physical units (nm): its documented defaults for connectomics.
TEASAR = {
    "scale": 1.5, "const": 300, "pdrf_scale": 100000, "pdrf_exponent": 4,
    "soma_acceptance_threshold": 3500, "soma_detection_threshold": 750,
    "soma_invalidation_const": 300, "soma_invalidation_scale": 2, "max_paths": None,
}
DUST = 1000              # voxels: objects smaller than this get no skeleton
DOWNSAMPLE = 16          # keep every 16th path voxel: ~150 nm between nodes
INSET = 2                # voxels: branch ends are cut back to lie further than this inside
SLAB = 128               # z slices read at a time
BLOCK = 1024             # kimimaro block edge; neighbours share one voxel plane


def volume_name(roi: str) -> str:
    return f"hemibrain_eb_{roi}"


def core_box(roi: str) -> list[list[int]]:
    """The scored core ([lo, hi) per axis, z, y, x) in the voxels of the region's raw array."""
    raw = funlib_level(GROUND_TRUTH / roi / "raw")
    core = funlib_level(GROUND_TRUTH / roi / "consolidated_ids")
    box = []
    for c, r, n, v in zip(core["offset"], raw["offset"], core["shape"], RESOLUTION, strict=True):
        if (c - r) % v:
            raise ValueError(f"{roi}: the core is not on raw's voxel grid")
        box.append([int((c - r) // v), int((c - r) // v) + n])
    return box


def load_objects(array: Any) -> tuple[np.ndarray, np.ndarray]:
    """The labels renumbered 1..K in the smallest dtype, and the original id of each new one.

    Read in slabs: a uint64 copy of roi_2's core alone would be 162 GB.
    """
    ids: set[int] = set()
    for z in range(0, array.shape[0], SLAB):
        ids.update(np.unique(np.asarray(array[z:z + SLAB])).tolist())
    ids.discard(0)
    original = np.array([0, *sorted(ids)], dtype=np.uint64)
    dtype = np.uint8 if original.size <= 255 else np.uint16 if original.size <= 65535 else np.uint32
    labels = np.empty(array.shape, dtype=dtype)
    for z in range(0, array.shape[0], SLAB):
        labels[z:z + SLAB] = np.searchsorted(original, np.asarray(array[z:z + SLAB]))
    return labels, original


def block_skeletons(args: tuple[np.ndarray, tuple[int, ...]]) -> dict[int, tuple[Any, Any]]:
    """kimimaro on one block, serially: each object's (vertices as core voxels, edges).

    No dust threshold here: an object's corner clipped by a block is small and still has to join
    its neighbours' pieces. Small *objects* are dropped once, by their total size.
    """
    import kimimaro

    crop, origin = args
    skeletons = kimimaro.skeletonize(
        crop, teasar_params=TEASAR, anisotropy=RESOLUTION, dust_threshold=0,
        fix_branching=True, fix_borders=True, progress=False, parallel=1,
    )
    out = {}
    for label, skeleton in skeletons.items():
        voxels = skeleton.vertices / np.asarray(RESOLUTION)
        if not np.allclose(voxels, np.rint(voxels)):
            raise ValueError(f"object {label}'s skeleton has vertices off the voxel grid")
        out[int(label)] = (np.rint(voxels).astype(np.int64) + np.asarray(origin),
                           np.asarray(skeleton.edges, dtype=np.int64))
    return out


def thin(tree: nx.Graph, step: int) -> nx.Graph:
    """The tree's endpoints and branch points, and every `step`-th node on each path between them.

    Kept nodes are joined along their path, so the thinned tree has the original's topology and
    lies on its nodes. Only a forest can be thinned: stitched blocks can close small loops where
    two blocks' paths run inside their shared plane, which the caller breaks first.
    """
    thinned = nx.Graph()
    critical = {n for n in tree if tree.degree(n) != 2}
    thinned.add_nodes_from(critical)
    walked: set[tuple[int, int]] = set()
    for start in critical:
        for first in tree.neighbors(start):
            if (start, first) in walked:
                continue
            path = [start, first]
            while path[-1] not in critical:
                path.append(next(n for n in tree.neighbors(path[-1]) if n != path[-2]))
            walked.add((path[-1], path[-2]))             # the same path, seen from its far end
            kept = path[::step] if path[::step][-1] == path[-1] else [*path[::step], path[-1]]
            thinned.add_edges_from(zip(kept, kept[1:], strict=False))
    return thinned


def inset_tips(tree: nx.Graph, vertices: np.ndarray, labels: np.ndarray, label: int, depth: int,
               limit: int) -> tuple[int, float]:
    """Cut each branch end back to its first voxel more than `depth` voxels inside the object.

    In place, on the full-resolution tree -> (ends moved, path removed in nm). An end moves at most
    `limit - 1` voxels along its path, never onto a branch point or another end, so the topology
    is unchanged; when no voxel in that reach is so deep, it goes to the most interior one (the
    first, on a tie). A voxel outside the array does not count as outside the object: the region's
    faces are not membranes.
    """
    grid = np.stack(np.meshgrid(*[np.arange(-depth, depth + 1)] * 3, indexing="ij"), -1)
    grid = grid.reshape(-1, 3)
    radius = np.sqrt((grid ** 2).sum(1))
    ball, radius = grid[radius <= depth], radius[radius <= depth]
    shape = np.asarray(labels.shape)

    def surface_distance(vertex: int) -> float:
        around = vertices[vertex] + ball
        within = np.all((around >= 0) & (around < shape), axis=1)
        outside = labels[tuple(around[within].T)] != label
        return float(radius[within][outside].min()) if outside.any() else np.inf

    moved, removed = 0, 0.0
    for tip in [n for n in tree if tree.degree(n) == 1]:
        chain, previous = [tip], None
        while len(chain) < limit:
            (onward,) = (n for n in tree.neighbors(chain[-1]) if n != previous)
            if tree.degree(onward) != 2:
                break
            previous = chain[-1]
            chain.append(onward)
        best, deepest = 0, -1.0
        for i, vertex in enumerate(chain):
            distance = surface_distance(vertex)
            if distance > deepest:
                best, deepest = i, distance
            if distance > depth:
                break
        if best:
            moved += 1
            removed += sum(float(np.linalg.norm((vertices[a] - vertices[b]) * RESOLUTION))
                           for a, b in zip(chain[:best], chain[1:best + 1], strict=True))
            tree.remove_nodes_from(chain[:best])
    return moved, removed


def region_skeleton(
    roi: str, workers: int, block: int | None = None
) -> tuple[nx.Graph, dict[str, Any]]:
    """Every whitelisted object of the region's core, skeletonised; `id` is the object."""
    import itertools
    from collections import defaultdict
    from concurrent.futures import ProcessPoolExecutor

    import zarr
    from osteoid import Skeleton

    block = block or BLOCK
    labels, original = load_objects(
        zarr.open_array(str(GROUND_TRUTH / roi / "consolidated_ids"), mode="r")
    )
    sizes = np.zeros(original.size, dtype=np.int64)
    for z in range(0, labels.shape[0], SLAB):
        sizes += np.bincount(labels[z:z + SLAB].ravel(), minlength=original.size)
    box = core_box(roi)
    low = np.array([lo for lo, _ in box])

    blocks = [
        (np.ascontiguousarray(labels[tuple(slice(o, min(o + block + 1, s))
                                           for o, s in zip(origin, labels.shape, strict=True))]),
         origin)
        for origin in itertools.product(*(range(0, max(s - 1, 1), block) for s in labels.shape))
    ]
    pieces: dict[int, list[Any]] = defaultdict(list)

    def collect(results: Any) -> None:
        for result in results:
            for label, (vertices, edges) in result.items():
                pieces[label].append(Skeleton(vertices.astype(np.float32), edges))

    if workers <= 1:
        collect(map(block_skeletons, blocks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            collect(pool.map(block_skeletons, blocks))
    del blocks

    graph = nx.Graph()
    node = 0
    skeletonised = 0
    inset_ends, inset_nm = 0, 0.0
    for label in sorted(pieces):
        if sizes[label] < DUST:
            continue
        skeletonised += 1
        merged = Skeleton.simple_merge(pieces[label]).consolidate()
        whole = nx.Graph()
        whole.add_nodes_from(range(len(merged.vertices)))
        whole.add_weighted_edges_from(
            (int(a), int(b), float(np.linalg.norm(merged.vertices[a] - merged.vertices[b])))
            for a, b in merged.edges
        )
        tree = nx.minimum_spanning_tree(whole)
        ends, length = inset_tips(tree, np.rint(merged.vertices).astype(np.int64), labels, label,
                                  INSET, DOWNSAMPLE)
        inset_ends, inset_nm = inset_ends + ends, inset_nm + length
        thinned = thin(tree, DOWNSAMPLE)
        order = sorted(thinned.nodes)
        voxels = np.rint(merged.vertices[order]).astype(np.int64)
        inside = labels[tuple(voxels.T)] == label
        if not inside.all():
            outside = int((~inside).sum())
            raise ValueError(f"{roi}: {outside} nodes of object {label} lie outside it")
        index = {vertex: node + i for i, vertex in enumerate(order)}
        for i, voxel in enumerate(voxels + low):
            graph.add_node(node + i, id=int(label), index_position=tuple(int(v) for v in voxel),
                           nm_position=tuple(float(v) * r for v, r in zip(voxel, RESOLUTION,
                                                                          strict=True)),
                           neuron_id=int(original[label]))
        graph.add_edges_from((index[a], index[b]) for a, b in thinned.edges)
        node += len(order)

    lengths: dict[int, float] = {}
    for u, v in graph.edges:
        a, b = graph.nodes[u]["nm_position"], graph.nodes[v]["nm_position"]
        lengths[graph.nodes[u]["id"]] = (
            lengths.get(graph.nodes[u]["id"], 0.0) + float(np.linalg.norm(np.subtract(a, b)))
        )
    cable = sum(lengths.values())
    stats = {
        "objects": int(original.size - 1),
        "objects_skeletonised": skeletonised,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": len({graph.nodes[n]["id"] for n in graph.nodes}),
        # Diagnostic only: more pieces than objects means stitching gaps (or erosion-cut necks),
        # which the object ids above bridge.
        "skeleton_pieces": nx.number_connected_components(graph),
        "ends_inset": inset_ends,
        "inset_path_mm": round(inset_nm / 1e6, 6),   # removed from the full-resolution paths
        "cable_mm": round(cable / 1e6, 6),
        "perfect_erl_um": round(sum(v * v for v in lengths.values()) / cable / 1e3, 4)
        if cable else 0.0,
        "region_box_zyx": box,
    }
    graph.graph.update(
        format=SKELETON_FORMAT, axes="zyx", voxel_size_nm=list(RESOLUTION),
        volume=volume_name(roi), region=roi, region_box=box,
        source=f"kimimaro {TEASAR} on {GROUND_TRUTH / roi / 'consolidated_ids'} in {block}^3 "
               f"blocks, objects under {DUST} voxels dropped, branch ends inset {INSET} voxels, "
               f"thinned by {DOWNSAMPLE}",
    )
    return graph, stats


def wrap_region(out: Path, roi: str) -> tuple[Path, dict[str, Any]]:
    wrapper = out / f"{roi}.zarr"
    make_group(wrapper, {"description": f"mia-evals wrapper over the LSD release's hemibrain "
                                        f"{roi} ({ROIS[roi]}); levels are symlinks, see "
                                        "../manifest.json"})
    linked = {"raw": wrap_pyramid(wrapper / "raw", "raw", [("s0", GROUND_TRUTH / roi / "raw")],
                                  RESOLUTION)}
    make_group(wrapper / "labels", {"labels": ["consolidated_ids", "eb_mask"]})
    for name, source in (("consolidated_ids", GROUND_TRUTH / roi / "consolidated_ids"),
                         ("eb_mask", GROUND_TRUTH / "mask")):
        linked[f"labels/{name}"] = wrap_pyramid(
            wrapper / "labels" / name, name, [("s0", source)], RESOLUTION,
            extra={"image-label": {"version": "0.4"}},
        )
    return wrapper, linked


def ffn_artifact(out: Path, roi: str, wrapper: Path) -> dict[str, Any]:
    """FFN's released segmentation of one region's core, wrapped as an `instances` artifact."""
    source = FFN / roi / "consolidated_ids"
    meta = funlib_level(source)
    core = funlib_level(GROUND_TRUTH / roi / "consolidated_ids")
    for key in ("offset", "shape", "resolution"):
        if meta[key] != core[key]:
            raise ValueError(f"{source} has {key} {meta[key]}, the ground truth {core[key]}")
    if meta["dtype"] not in ("<u8", "<u4"):
        raise ValueError(f"{source} is {meta['dtype']}, not an unsigned labelling")
    origin = [lo for lo, _ in core_box(roi)]
    path = out / f"{volume_name(roi)}.zarr"
    make_group(path, {
        "multiscales": multiscales(f"{FFN_RUN} {roi}",
                                   [dataset("s0", RESOLUTION, meta["offset"])]),
        "kind": "instances",
        "background_id": 0,
        "origin": origin,
        "covers_full_box": True,
        "source_path": str(wrapper),
        "source_image_key": "raw",
        "run": FFN_RUN,
        "mask": "labels/eb_mask",
        "convention": (
            "FFN segmentation of the hemibrain (Januszewski et al. 2018; Scheffer et al. 2020), "
            "cropped to the region's core, restricted to the ellipsoid body and relabelled into "
            f"connected components, as released by Sheridan et al. 2023 (FFN/{roi}/"
            "consolidated_ids)"
        ),
        "source_segmentation": str(source),
    })
    link(source, path / "s0")
    return {"path": str(path), "target": str(source), "origin": origin, **meta}


def verify(wrappers: dict[str, Path], ffn: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Read everything back the way its consumers will: miao for the wrappers, mia-evals for FFN."""
    from miao.zarr_meta import read_ome_metadata

    from artifact import open_artifact

    report: dict[str, Any] = {}
    for roi, wrapper in wrappers.items():
        meta = read_ome_metadata(wrapper, "raw", "zarr2")
        report[roi] = {"axes": meta.axis_names, "scale": meta.scales[0].scale_factors,
                       "translation": meta.scales[0].translation, "shape": meta.scales[0].shape}
    for name, entry in ffn.items():
        artifact = open_artifact(Path(entry["path"]))
        if list(artifact.origin) != entry["origin"] or artifact.kind != "instances":
            raise ValueError(f"{name}: read back as {artifact.kind} at {artifact.origin}")
        report[name] = {"kind": artifact.kind, "shape": list(artifact.spatial_shape),
                        "origin": list(artifact.origin), "axes": artifact.axes}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="the dataset directory to build")
    parser.add_argument("--rois", nargs="+", default=list(ROIS), choices=list(ROIS))
    parser.add_argument("--workers", type=int, default=8, help="processes, one block each")
    parser.add_argument("--block", type=int, default=BLOCK, help="kimimaro block edge, voxels")
    parser.add_argument("--skip-skeletons", action="store_true",
                        help="(re)write wrappers and FFN artifacts only; keep existing skeletons")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / FFN_RUN).mkdir(exist_ok=True)

    wrappers, linked, skeletons, ffn = {}, {}, {}, {}
    for roi in args.rois:
        wrappers[roi], linked[roi] = wrap_region(out, roi)
        ffn[volume_name(roi)] = ffn_artifact(out / FFN_RUN, roi, wrappers[roi])
        if args.skip_skeletons:
            continue
        graph, stats = region_skeleton(roi, args.workers, args.block)
        (wrappers[roi] / "skeletons").mkdir(exist_ok=True)
        stats["file"] = write_skeleton(wrappers[roi] / "skeletons" / f"{volume_name(roi)}.pkl",
                                       graph)
        skeletons[volume_name(roi)] = stats
        print(f"{volume_name(roi)}: {stats['objects_skeletonised']}/{stats['objects']} objects, "
              f"{stats['nodes']} nodes, {stats['components']} pieces, {stats['cable_mm']:.3f} mm, "
              f"perfect ERL {stats['perfect_erl_um']:.2f} um", flush=True)

    from report.record import git_commit

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({
        "built": date.today().isoformat(),
        "command": [Path(sys.executable).name, "-m", "truth.lsd_hemibrain", *sys.argv[1:]],
        "mia_evals_commit": git_commit(Path(__file__).resolve().parents[2]),
        "release": str(RELEASE),
        "public_sources": {"everything": "s3://open-neurodata/funke/hemi/ (the 'Data download' "
                                         "notebook of https://github.com/funkelab/lsd)"},
        "teasar": TEASAR, "dust_threshold_voxels": DUST, "downsample": DOWNSAMPLE,
        "inset_voxels": INSET, "block": args.block,
    })
    manifest.setdefault("wrappers", {}).update(
        {roi: {"path": str(w), "linked": linked[roi]} for roi, w in wrappers.items()})
    manifest.setdefault("skeletons", {}).update(skeletons)
    manifest.setdefault("ffn_reference_artifacts", {}).update(ffn)
    manifest.setdefault("readback", {}).update(verify(wrappers, ffn))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out / "README.md").write_text(README.format(out=out))
    print(f"wrote {out}", flush=True)


README = """# hemibrain_eb_lsd -- mia-evals benchmark data

Built by `python -m truth.lsd_hemibrain --out {out}` from the mia-evals repository;
`manifest.json` records the command, commit, sources, kimimaro settings and hashes. Nothing here
copies the release: raw, labels and mask are symlinks into /nrs/funke/sheridana (read-only, not
ours); the public copy is listed in the manifest.

- `roi_<k>.zarr/`: OME-Zarr 0.4 wrappers (zarr v2), z, y, x at 8 nm, readable by miao. `raw/` is
  the region's core plus its context margin; `labels/consolidated_ids/` the whitelisted voxel
  ground truth (core only; for viewing); `labels/eb_mask/` the ellipsoid body at 256 nm;
  `skeletons/<volume>.pkl` the skeleton ground truth derived from `consolidated_ids` (networkx;
  `id` = the ground-truth object, `index_position` = z, y, x voxel of `raw/s0`).
- `ffn_januszewski2018/<volume>.zarr`: FFN's released segmentation of each core (restricted to the
  ellipsoid body and relabelled by the LSD authors), as an `instances` artifact.

Used by the mia-evals task `hemibrain_eb_neurite_tracing`.
"""


if __name__ == "__main__":
    main()
