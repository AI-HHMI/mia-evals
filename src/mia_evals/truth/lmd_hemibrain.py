"""lmd's hemibrain crop in the right alpha lobe, as neurite tracing benchmark data.

    python -m mia_evals.truth.lmd_hemibrain --out /groups/miaai/miaai/mia-evals-data/hemibrain_al_lmd

lmd's `crop-004_aL_R_x16000_y28000_z6000` is a 1000^3 cube at 8 nm (8 um) of the hemibrain's right
alpha lobe, its first voxel at x 16000, y 28000, z 6000 in hemibrain voxels, with the proofread v1.2
segmentation. This builds its skeleton ground truth the way mia_evals.truth.lsd_hemibrain built the
ellipsoid body's, with the same code for the skeletons:

    hemibrain_al_crop004.zarr/         OME-Zarr 0.4 (zarr v2), z, y, x at 8 nm, placed in the brain
        raw/s0                         the hemibrain image over the cube and 200 voxels around it
        labels/consolidated_ids/s0     the ground truth's voxels: the cube's pieces, eroded
        skeletons/<volume>.pkl         every piece's skeleton
        skeletons/<volume>_largest_piece.pkl   each neuron's largest piece: what the task scores
    hemibrain-v1.2-traced-neurons.csv  the body list, downloaded (hash pinned below)
    objects.json                       every piece and neuron: body, type, sizes, cable
    manifest.json, README.md           what was built from what, with hashes and this command

**The raw is a copy, the only one here.** The hemibrain image is public as a sharded neuroglancer
precomputed volume, jpeg in 64^3 chunks (`IMAGE`), which miao cannot read; lmd's crop is a cut of
it without a margin (the store's own `source` attributes name it). This reads the cube and 200
voxels on every side -- the context the EB regions' raw carries -- and checks that the inner 1000^3
is lmd's crop raw byte for byte.

**The neurons are the bodies on hemibrain v1.2's Traced list**, LSD's rule on the current release:
their EB whitelist was exactly the bodies whose neuPrint status was "Traced" in a 2019 export
(measured on their intermediate arrays; see truth.lsd_hemibrain). The v1.2 list is the release's
`traced-neurons.csv`, neuprint-python's non-cropped neurons with status Traced. By v1.2 nearly
everything is traced: the list holds 619 of the cube's 3,278 bodies and 98.2% of its voxels (610
are Kenyon cells), where LSD's 2019 list held 31-39% of the EB's. The other bodies are fragments:
the 2,042 of them with 1,000 voxels or more hold 1.7%. Nothing is masked: the cube is not cut to a
neuropil, and only 0.12% of it is unlabelled.

**Pieces, erosion and skeletons are LSD's and truth.lsd_hemibrain's.** The pieces are each body's
6-connected components inside the cube (LSD's malis relabelling), and they are the unit run length
counts: a neuron that leaves the cube and comes back is several pieces, which a segmentation of the
cube has no way to join. Each piece is eroded as LSD's erode_blockwise_3d.py did (scipy
binary_erosion, 2 iterations, border value 1: 2 voxels off every surface, none off the cube's
faces), here in its bounding box grown by 2 voxels, which holds everything that erosion reads; the
result is their whole-array loop's (tests/unit/test_lmd_hemibrain.py). Then
mia_evals.truth.lsd_hemibrain.skeletonise, unchanged: kimimaro's TEASAR, pieces under 1,000 voxels dropped,
branch ends inset 2 voxels, a node every 16 path voxels. The cube is one 1024^3 block, so nothing
is stitched. Every node also carries `body_id`, its hemibrain body.

**The task scores each neuron's largest piece only**, the one with the most voxels before erosion,
its skeleton node for node the full set's. Measured 2026-10-20: 1,374 pieces, 1,146 left after the
erosion, 1,058 skeletonised (612 neurons, 6.38 mm, perfect ERL 13.8 um); the largest pieces are 612
(4.97 mm, 78% of the cable, perfect ERL 14.4 um). The neurons' other pieces carry the rest: the
large neurons that fill the lobe -- MBON14, PPL106, DPM, APL -- fall into 26 to 83 pieces each.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tarfile
import time
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from . import lsd_hemibrain
from .common import dataset, make_group, multiscales, sha256, write_json, write_skeleton

STORE = Path("/groups/miaai/miaai/lmd-v0.0.1/data/em-drosophila-flyem-hemibrain/"
             "crop-004_aL_R_x16000_y28000_z6000.zarr")
LABELS = STORE / "labels" / "proofread-cell-hemibrain-v1.2" / "s0"
SOURCE_URL = "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg"
IMAGE = "https://storage.googleapis.com/neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg/"
TRACED_URL = "https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz"
TRACED_MEMBER = "exported-traced-adjacencies-v1.2/traced-neurons.csv"
TRACED_SHA256 = "d07c1cac648c12d8135786996ac55de5a2b63d605cd6a24c5ce366224172c825"
TRACED_FILE = "hemibrain-v1.2-traced-neurons.csv"
ORIGIN_XYZ = (16000, 28000, 6000)   # the cube's first voxel in hemibrain voxels (lmd's `source`)
SHAPE = 1000                        # voxels per axis
MARGIN = 200                        # voxels of raw on every side of the cube, as the EB regions'
RESOLUTION = (8.0, 8.0, 8.0)
VOLUME = "hemibrain_al_crop004"
REGION = "crop-004"
WRAPPER = f"{VOLUME}.zarr"
LARGEST = "_largest_piece"          # the second skeleton set's file suffix
ITERATIONS, BORDER_VALUE = 2, 1     # LSD's erode_blockwise_roi*.json
CHUNKS = (256, 256, 256)
SLAB = 100                          # z slices compared at a time


def check_store() -> dict[str, Any]:
    """lmd's record of where the crop was cut; refuses anything but what this module assumes."""
    source = json.loads((STORE / "zarr.json").read_text())["attributes"]["source"]
    got = (source["bbox_origin_voxel"], source["bbox_size_voxel"], source["bbox_coordinate_order"],
           source["voxel_size_nm"], source["source_urls"]["jpeg"])
    if got != (list(ORIGIN_XYZ), [SHAPE] * 3, "XYZ", list(RESOLUTION), SOURCE_URL):
        raise ValueError(f"{STORE}/zarr.json describes {got}, not the crop this module builds")
    for array in (STORE / "raw" / "s0", LABELS):
        axes = json.loads((array / "zarr.json").read_text()).get("dimension_names")
        if axes != ["x", "y", "z"]:
            raise ValueError(f"{array} is {axes}, not x, y, z")
    return source


def traced_list(out: Path) -> dict[int, tuple[str, str]]:
    """hemibrain v1.2's Traced neurons (body -> type, instance), downloaded once into `out`."""
    path = out / TRACED_FILE
    if not path.exists():
        with urllib.request.urlopen(TRACED_URL, timeout=600) as response:
            archive = response.read()
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            member = tar.extractfile(TRACED_MEMBER)
            if member is None:
                raise ValueError(f"{TRACED_URL} has no {TRACED_MEMBER}")
            path.write_bytes(member.read())
    if sha256(path) != TRACED_SHA256:
        raise ValueError(f"{path} is not the list this module was written against (sha256)")
    with open(path) as handle:
        return {int(r["bodyId"]): (r["type"], r["instance"]) for r in csv.DictReader(handle)}


def write_array(path: Path, data: np.ndarray) -> None:
    """A zarr v2 array holding `data`, or a check that the existing one holds exactly it."""
    import zarr

    if path.exists():
        existing = zarr.open_array(str(path), mode="r")
        if existing.shape != data.shape or existing.dtype != data.dtype or \
                not np.array_equal(np.asarray(existing), data):
            raise FileExistsError(f"{path} exists with different content; remove it to rebuild")
        return
    array = zarr.create_array(store=str(path), shape=data.shape, chunks=CHUNKS, dtype=data.dtype,
                              zarr_format=2)
    array[:] = data


def read_raw() -> np.ndarray:
    """The hemibrain image over the cube and MARGIN voxels on every side, z, y, x."""
    import tensorstore as ts

    image = ts.open({"driver": "neuroglancer_precomputed", "scale_index": 0,
                     "kvstore": {"driver": "http", "base_url": IMAGE}}, read=True).result()
    lo, hi = -MARGIN, SHAPE + MARGIN
    x, y, z = ORIGIN_XYZ
    xyz = image[x + lo:x + hi, y + lo:y + hi, z + lo:z + hi, 0].read().result()
    return np.ascontiguousarray(np.asarray(xyz).transpose(2, 1, 0))


def check_raw(raw: np.ndarray) -> None:
    """The inner SHAPE^3 must be lmd's crop raw byte for byte (lmd stores x, y, z)."""
    import zarr

    lmd = zarr.open_array(str(STORE / "raw" / "s0"), mode="r")
    inner = raw[MARGIN:MARGIN + SHAPE, MARGIN:MARGIN + SHAPE, MARGIN:MARGIN + SHAPE]
    for z in range(0, SHAPE, SLAB):
        if not np.array_equal(np.asarray(lmd[:, :, z:z + SLAB]).transpose(2, 1, 0),
                              inner[z:z + SLAB]):
            raise ValueError(f"the image's slices {z}..{z + SLAB} of the cube differ from {STORE}")


def pieces_of(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Each body's 6-connected components (LSD's malis relabelling) and the body of each."""
    import cc3d
    import fastremap

    pieces = cc3d.connected_components(labels, connectivity=6, out_dtype=np.uint32)
    return pieces, {int(p): int(b) for p, b in fastremap.component_map(pieces, labels).items() if p}


def erode(pieces: np.ndarray) -> int:
    """LSD's erode_in_block on one block, in place: each label's binary_erosion (iterations 2,
    border_value 1) -> voxels removed.

    Computed in the label's bounding box grown by ITERATIONS voxels, which holds everything an
    erosion that deep reads; where the box meets the array's faces, border_value applies as it
    did for LSD.
    """
    from scipy import ndimage

    removed = 0
    for label, box in enumerate(ndimage.find_objects(pieces), start=1):
        if box is None:
            continue
        grown = tuple(slice(max(s.start - ITERATIONS, 0), min(s.stop + ITERATIONS, n))
                      for s, n in zip(box, pieces.shape, strict=True))
        crop = pieces[grown]
        mask = crop == label
        lost = mask & ~ndimage.binary_erosion(mask, iterations=ITERATIONS,
                                               border_value=BORDER_VALUE)
        crop[lost] = 0
        removed += int(lost.sum())
    return removed


def largest_pieces(graph: nx.Graph, voxels: dict[int, int]) -> nx.Graph:
    """Each neuron's largest piece (most voxels before erosion; ties to the lower piece id).

    The kept skeletons are the full set's, node for node and edge for edge; nodes are numbered
    0..N-1 in their original order.
    """
    pieces: dict[int, set[int]] = defaultdict(set)
    for _, d in graph.nodes(data=True):
        pieces[d["body_id"]].add(d["neuron_id"])
    keep = {min(ps, key=lambda p: (-voxels[p], p)) for ps in pieces.values()}
    kept = [n for n in graph.nodes if graph.nodes[n]["neuron_id"] in keep]
    index = {n: i for i, n in enumerate(kept)}
    largest = nx.Graph()
    largest.graph.update(graph.graph)
    largest.add_nodes_from((index[n], dict(graph.nodes[n])) for n in kept)
    largest.add_edges_from((index[u], index[v]) for u, v in graph.subgraph(kept).edges)
    largest.graph.update(
        source=graph.graph["source"] + "; then each neuron's largest piece only (most voxels "
                                       "before erosion)",
        selection="largest piece per neuron",
    )
    return largest


def statistics(graph: nx.Graph) -> dict[str, Any]:
    """mia_evals.truth.lsd_hemibrain.skeletonise's statistics of a skeleton set, from the graph alone."""
    lengths: dict[int, float] = defaultdict(float)
    for u, v in graph.edges:
        a, b = graph.nodes[u], graph.nodes[v]
        lengths[a["id"]] += float(np.linalg.norm(np.subtract(a["nm_position"], b["nm_position"])))
    cable = sum(lengths.values())
    return {
        "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
        "components": len({d["id"] for _, d in graph.nodes(data=True)}),
        "skeleton_pieces": nx.number_connected_components(graph),
        "cable_mm": round(cable / 1e6, 6),
        "perfect_erl_um": round(sum(v * v for v in lengths.values()) / cable / 1e3, 4)
        if cable else 0.0,
        "region_box_zyx": graph.graph["region_box"],
    }


def build(out: Path) -> dict[str, Any]:
    import fastremap
    import zarr

    clock, seconds = time.time(), {}

    def lap(step: str) -> None:
        nonlocal clock
        seconds[step] = round(time.time() - clock, 1)
        clock = time.time()
        print(f"{step}: {seconds[step]} s", flush=True)

    source = check_store()
    traced = traced_list(out)
    wrapper = out / WRAPPER
    origin_zyx = [o - MARGIN for o in ORIGIN_XYZ[::-1]]   # the raw's first voxel, hemibrain voxels
    make_group(wrapper, {"description": "mia-evals' copy of the hemibrain around lmd's "
                                        f"{STORE.name}, z, y, x at 8 nm; see ../manifest.json"})

    raw = read_raw()
    check_raw(raw)
    make_group(wrapper / "raw", {"multiscales": multiscales("raw", [
        dataset("s0", RESOLUTION, [o * r for o, r in zip(origin_zyx, RESOLUTION, strict=True)])])})
    write_array(wrapper / "raw" / "s0", raw)
    del raw
    lap("raw")

    labels = np.asarray(zarr.open_array(str(LABELS), mode="r"))
    labels = np.ascontiguousarray(labels.transpose(2, 1, 0))
    ids, counts = fastremap.unique(labels, return_counts=True)
    present = [int(i) for i in ids if i]
    listed = [i for i in present if i in traced]
    listed_voxels = int(counts[np.isin(ids, np.asarray(listed, dtype=ids.dtype))].sum())
    fastremap.mask_except(labels, listed, in_place=True)
    pieces, body_of = pieces_of(labels)
    del labels
    piece_ids, piece_counts = fastremap.unique(pieces, return_counts=True)
    voxels = {int(p): int(c) for p, c in zip(piece_ids, piece_counts, strict=True) if p}
    removed = erode(pieces)
    piece_ids, piece_counts = fastremap.unique(pieces, return_counts=True)
    eroded = {int(p): int(c) for p, c in zip(piece_ids, piece_counts, strict=True) if p}
    cube_nm = [(o + MARGIN) * r for o, r in zip(origin_zyx, RESOLUTION, strict=True)]
    make_group(wrapper / "labels", {"labels": ["consolidated_ids"]})
    make_group(wrapper / "labels" / "consolidated_ids", {
        "multiscales": multiscales("consolidated_ids", [dataset("s0", RESOLUTION, cube_nm)]),
        "image-label": {"version": "0.4"},
    })
    write_array(wrapper / "labels" / "consolidated_ids" / "s0", pieces)
    del pieces
    lap("labels")

    box = [[MARGIN, MARGIN + SHAPE]] * 3
    graph, stats = lsd_hemibrain.skeletonise(
        wrapper / "labels" / "consolidated_ids" / "s0", box, VOLUME, REGION, workers=1)
    for _, d in graph.nodes(data=True):
        d["body_id"] = body_of[d["neuron_id"]]
    graph.graph.update(body_list=f"{TRACED_FILE} (hemibrain v1.2 Traced, {TRACED_URL})")
    largest = largest_pieces(graph, voxels)
    (wrapper / "skeletons").mkdir(exist_ok=True)
    stats["file"] = write_skeleton(wrapper / "skeletons" / f"{VOLUME}.pkl", graph)
    largest_stats = statistics(largest)
    largest_stats["file"] = write_skeleton(wrapper / "skeletons" / f"{VOLUME}{LARGEST}.pkl",
                                           largest)
    lap("skeletons")

    nodes: dict[int, int] = defaultdict(int)
    cable: dict[int, float] = defaultdict(float)
    for _, d in graph.nodes(data=True):
        nodes[d["neuron_id"]] += 1
    for u, v in graph.edges:
        a, b = graph.nodes[u], graph.nodes[v]
        cable[a["neuron_id"]] += float(np.linalg.norm(np.subtract(a["nm_position"],
                                                                  b["nm_position"])))
    chosen = {d["neuron_id"] for _, d in largest.nodes(data=True)}
    objects = [{"piece": p, "body_id": body_of[p], "type": traced[body_of[p]][0],
                "instance": traced[body_of[p]][1], "voxels": voxels[p],
                "voxels_eroded": eroded.get(p, 0), "nodes": nodes[p],
                "cable_um": round(cable[p] / 1e3, 3), "largest_piece": p in chosen}
               for p in sorted(voxels)]
    bodies: dict[str, dict[str, Any]] = {}
    for o in objects:
        b = bodies.setdefault(str(o["body_id"]), {
            "type": o["type"], "instance": o["instance"], "pieces": 0, "pieces_skeletonised": 0,
            "voxels": 0, "cable_um": 0.0})
        b["pieces"] += 1
        b["pieces_skeletonised"] += o["nodes"] > 0
        b["voxels"] += o["voxels"]
        b["cable_um"] = round(b["cable_um"] + o["cable_um"], 3)
    write_json(out / "objects.json", {"objects": objects, "bodies": bodies})

    funnel = {
        "voxels": SHAPE ** 3, "unlabelled_voxels": int(counts[ids == 0].sum()),
        "bodies": len(present), "bodies_listed": len(listed),
        "listed_voxel_fraction": round(listed_voxels / SHAPE ** 3, 4),
        "pieces": len(voxels), "erosion_voxels_removed": removed,
        "erosion_fraction": round(removed / listed_voxels, 4),
        "pieces_left_after_erosion": len(eroded),
        "pieces_skeletonised": stats["objects_skeletonised"],
        "neurons_skeletonised": len({d["body_id"] for _, d in graph.nodes(data=True)}),
        "largest_pieces": largest_stats["components"],
    }
    return {"source": source, "traced": len(traced), "funnel": funnel, "skeletons": stats,
            "largest_piece_skeletons": largest_stats, "seconds": seconds}


def verify(out: Path) -> dict[str, Any]:
    """Read the store back as its consumers will: miao for the arrays, the pickles as pickles."""
    import pickle

    from miao.zarr_meta import read_ome_metadata

    report: dict[str, Any] = {}
    for key in ("raw", "labels/consolidated_ids"):
        meta = read_ome_metadata(out / WRAPPER, key, "zarr2")
        report[key] = {"axes": meta.axis_names, "scale": meta.scales[0].scale_factors,
                       "translation": meta.scales[0].translation, "shape": meta.scales[0].shape}
    for name in (f"{VOLUME}.pkl", f"{VOLUME}{LARGEST}.pkl"):
        graph = pickle.loads((out / WRAPPER / "skeletons" / name).read_bytes())
        positions = np.array([d["index_position"] for _, d in graph.nodes(data=True)])
        if positions.min() < MARGIN or positions.max() >= MARGIN + SHAPE:
            raise ValueError(f"{name} has nodes outside the cube")
        report[name] = {"axes": graph.graph["axes"], "nodes": graph.number_of_nodes(),
                        "index_range": [int(positions.min()), int(positions.max())]}
    return report


def write_readme(out: Path, manifest: dict[str, Any]) -> None:
    (out / "README.md").write_text(README.format(
        out=out, **manifest["funnel"], full=manifest["skeletons"],
        largest=manifest["largest_piece_skeletons"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="the dataset directory to build")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    built = build(out)

    from mia_evals.report.record import git_commit

    manifest = {
        "built": date.today().isoformat(),
        "command": [Path(sys.executable).name, "-m", "mia_evals.truth.lmd_hemibrain", *sys.argv[1:]],
        "mia_evals_commit": git_commit(Path(__file__).resolve().parents[3]),
        "sources": {"lmd_store": str(STORE), "lmd_labels": str(LABELS), "image": IMAGE,
                    "image_as_lmd_names_it": SOURCE_URL, "traced_list": TRACED_URL,
                    "traced_list_sha256": TRACED_SHA256},
        "lmd_source_attributes": built.pop("source"),
        "volume": VOLUME, "region": REGION, "wrapper": str(out / WRAPPER),
        "cube_origin_xyz_hemibrain_voxels": list(ORIGIN_XYZ), "cube_voxels": SHAPE,
        "margin_voxels": MARGIN, "region_box_zyx": [[MARGIN, MARGIN + SHAPE]] * 3,
        "raw_check": "the raw's inner 1000^3 is lmd's crop raw byte for byte",
        "erosion": {"iterations": ITERATIONS, "border_value": BORDER_VALUE,
                    "pieces_connectivity": 6},
        "teasar": lsd_hemibrain.TEASAR, "dust_threshold_voxels": lsd_hemibrain.DUST,
        "downsample": lsd_hemibrain.DOWNSAMPLE, "inset_voxels": lsd_hemibrain.INSET,
        "block": lsd_hemibrain.BLOCK, "traced_neurons_listed": built.pop("traced"), **built,
        "readback": verify(out),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_readme(out, manifest)
    print(f"wrote {out}", flush=True)


README = """# hemibrain_al_lmd -- mia-evals benchmark data

Built by `python -m mia_evals.truth.lmd_hemibrain --out {out}` from the mia-evals
repository; `manifest.json` records the command, commit, sources, settings and hashes, and the
builder's docstring explains each choice. Used by the mia-evals task `hemibrain_al_neurite_tracing`.

**What it is.** Skeleton ground truth for lmd's hemibrain crop
`crop-004_aL_R_x16000_y28000_z6000`: a 1000^3 cube at 8 nm (8 um) in the right alpha lobe of the
mushroom body, its first voxel at x 16000, y 28000, z 6000 in hemibrain voxels. The neurons are the
hemibrain v1.2 proofread segmentation's bodies with status Traced, and the skeletons are made the
way `hemibrain_eb_lsd`'s were, by the same code (`mia_evals.truth.lsd_hemibrain.skeletonise`).

## Files

- `hemibrain_al_crop004.zarr/`: OME-Zarr 0.4 (zarr v2), z, y, x at 8 nm, placed in the brain
  (translations in hemibrain nm), readable by miao.
  - `raw/s0`: the hemibrain image over the cube and 200 voxels of context on every side (1400^3),
    read from the public `clahe_yz/jpeg` volume that lmd cut the crop from. Its inner 1000^3 is
    lmd's crop raw byte for byte (checked at every build). The only copied array here.
  - `labels/consolidated_ids/s0`: the ground truth's voxels, 1000^3 at the cube's place (200
    voxels into `raw`): every piece of every listed neuron, eroded. Ids are pieces, not hemibrain
    bodies.
  - `skeletons/hemibrain_al_crop004.pkl`: every piece's skeleton.
  - `skeletons/hemibrain_al_crop004_largest_piece.pkl`: each neuron's largest piece only. This is
    the set the task scores.

  Both pickles are networkx graphs in mia-evals' skeleton format. Per node: `id` (the ground-truth
  piece, the unit run length is counted over), `index_position` (z, y, x voxel of `raw/s0`),
  `nm_position`, `neuron_id` (the piece's id in `consolidated_ids`) and `body_id` (its hemibrain
  body).
- `hemibrain-v1.2-traced-neurons.csv`: the body list, as downloaded from the hemibrain v1.2
  release.
- `objects.json`: every piece (body, type, instance, voxels before and after erosion, nodes, cable,
  whether it is its neuron's largest) and every neuron.

## How the skeletons were extracted

1. **Neurons.** Keep the bodies on hemibrain v1.2's list of Traced neurons
   (`exported-traced-adjacencies-v1.2/traced-neurons.csv`: non-cropped neurons with status Traced).
   This is the LSD paper's rule for its ellipsoid-body ground truth (they kept exactly the bodies
   whose neuPrint status was "Traced" in 2019) applied to the current release. Of the cube's
   {bodies:,} bodies, {bodies_listed} are on the list, holding {listed_voxel_fraction:.1%} of its
   voxels, so this ground truth is dense, unlike the EB's (31-39% of the voxels). The rest are
   small fragments. Nothing is masked.
2. **Pieces.** Split each body into its 6-connected components inside the cube, as LSD relabelled
   theirs: {pieces:,} pieces. A neuron that leaves the cube and comes back is several pieces, which
   a segmentation of the cube alone has no way to join, so each piece is its own ground-truth
   object.
3. **Erosion.** Erode each piece as LSD did (scipy `binary_erosion`, 2 iterations, border value
   1): 2 voxels come off every surface but none off the cube's faces. {erosion_fraction:.1%} of the
   voxels go, and {pieces_left_after_erosion:,} pieces keep at least one.
4. **Skeletons.** kimimaro's TEASAR on every piece of at least 1,000 voxels (after erosion), each
   branch end cut back to lie more than 2 voxels inside its piece, thinned to a node every 16 path
   voxels (~150 nm); every node is asserted to lie inside its own piece. The cube is one 1024^3
   block, so no blocks are stitched. {pieces_skeletonised:,} pieces of {neurons_skeletonised}
   neurons: {full[nodes]:,} nodes, {full[cable_mm]:.2f} mm of cable, perfect ERL
   {full[perfect_erl_um]:.1f} um.
5. **Largest piece per neuron.** Keep each neuron's piece with the most voxels before erosion and
   drop its other pieces; the kept skeletons are unchanged. {largest_pieces} pieces:
   {largest[nodes]:,} nodes, {largest[cable_mm]:.2f} mm, perfect ERL
   {largest[perfect_erl_um]:.1f} um.

## Caveats

- The ground truth is proofread FFN (Scheffer et al. 2020), so no segmentation is independent of
  it, and there is no reference row: the v1.2 labels are the truth itself.
- The cube's raw is in lmd's SSL corpora (gary_comparison arms 2a/2b, lmd_ssl_v2), and its v1.2
  labels are in lmd's labeled `neuron_instance/all` recipe: a model trained with that recipe has
  seen this ground truth.
"""


if __name__ == "__main__":
    main()
