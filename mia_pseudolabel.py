"""Turn a trained mia-train checkpoint into pseudo-labels over unlabelled NISB cubes.

    # 1. affinities over N blocks of a cube (GPU)
    <mia-train venv>/bin/python mia_pseudolabel.py predict <run_dir> \
        --cube .../train_100/train/seed7.zarr --out-dir aff/seed7 --blocks 4

    # 2. affinities -> instances -> filtered pseudo-labels, into a sidecar container (CPU)
    <mia-train venv>/bin/python mia_pseudolabel.py build --aff-dir aff/seed7 \
        --cube .../train_100/train/seed7.zarr --out /nrs/.../pseudo_r1/seed7.zarr

    # 3. pick the thresholds on a cube whose ground truth we DO have
    <mia-train venv>/bin/python mia_pseudolabel.py calibrate --aff-dir aff/seed0 \
        --cube .../base/val/seed100.zarr

Everything here runs in mia-train's own environment: `cc3d` is the only post-processing
dependency and it ships with mia-train's `affinity` extra. The two phases still split across an
LSF GPU job and a CPU job, because holding a GPU through the segmentation is pure waste and
because every threshold worth tuning lives in phase 2 -- so a sweep costs one prediction pass.

**Instances come from thresholded connected components, not mutex watershed.** MWS is the
theoretically better story -- it uses the long-range channels as repulsive edges and needs no
threshold -- but measured on this task it over-merges catastrophically: on a 512^3 block of
seed100 with the best available checkpoint, MWS scored 0.024 nERL against CC's 0.584, with 242
mergers against 4. Most of that is the default `repulsive_stride=4` discarding 64x of the
repulsive edges (at stride 1 MWS recovers to 0.380), but CC still wins by 2x, so CC it is.

Four design points, each of which is a way to get this silently wrong:

  * **Uncertain must be `-1`, never `0`.** `affinities_from_labels` builds targets as
    `(a == b) & (a > 0)` and masks only `labels != ignore_index`. So `0` is *background*: a
    confident assertion that a voxel belongs to no object, which is trained on. Writing `0` for
    "the teacher was unsure" would teach the student that every ambiguous voxel is definitively
    background -- the exact opposite of abstaining. Hence `IGNORE = -1` and a signed dtype.

  * **A CC label of `0` is not background.** It means "no affinity edge survived the threshold
    here", which happens both at real membrane and wherever the teacher was merely unsure. Those
    two must not collapse: a voxel CC left at 0 becomes background only if its foreground score
    independently agrees, and abstains otherwise. This distinction did not exist under mutex
    watershed, which labels every voxel, and is the single most dangerous part of the switch.

  * **Blocks, not whole cubes.** cc3d segments at ~10-15 M voxels/s, so cost is no longer the
    binding constraint it was under MWS -- but a whole cube is still 48.6 GB of uint32 labels.
    A few blocks per cube buys cube *diversity*, which is what matters when every cube is an
    independent synthetic seed, rather than exhaustive coverage of a few.

  * **The label array is cube-shaped but sparse.** miao resolves the image and label pyramids
    independently, but a label array of a different shape than `raw` would still be a trap for
    every downstream consumer. So the array is the full cube shape with `fill_value=-1`, and
    only the blocks we actually labelled are ever written. Unwritten zarr chunks occupy no disk
    (measured: 8 KB for a 512^3 int32 array with one chunk written), so the whole cube reads as
    "ignore" for free, and `bounding_box` in the miao config restricts sampling to the blocks.
    Each block is its own volume entry with its own `bounding_box`, so no training crop ever
    spans two blocks -- which is what makes independently-numbered blocks safe.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import zlib
from pathlib import Path

import cc3d
import numpy as np
import torch
import zarr

from mia_nisb import NATIVE_LEVEL, open_labels, open_raw, spatial_shape
from mia_predict import load_algorithm, predict
from mia_score_mws import LONG

# The value `affinity_seg` excludes from its loss mask (`ignore_index`, affinity_seg.py:70).
# Signed on purpose: a uint16 label array would store this as 65535 and every "abstain" voxel
# would become a real instance id after `_prepare_labels` casts to int64.
IGNORE = -1
# Six channels are predicted and stored even though CC reads only the three short-range ones:
# the long-range channels cost nothing extra at inference, and keeping them means an existing
# affinity directory stays usable if the post-processing is ever revisited.
AFF_CHANNELS = 6


def scale_threshold(logit: float) -> float:
    """BANIS' `sigmoid(0.2 * logit)`, the scale every threshold in both pipelines is quoted on."""
    return float(1.0 / (1.0 + np.exp(-0.2 * logit)))


# ----------------------------------------------------------------- block geometry


def block_origins(
    shape: tuple[int, int, int], block: int, margin: int, count: int, seed: int
) -> list[tuple[int, int, int]]:
    """Deterministic, non-overlapping core-block origins, inset from the cube faces by `margin`.

    Candidates are a regular grid stepped by `block` so cores never overlap (overlapping cores
    would write conflicting instance ids into the same voxels, since ids are only meaningful
    within one segmentation). The grid is then shuffled by a cube-derived seed and truncated, so
    the choice is reproducible from the config alone and does not drift between rounds.

    The inset is what lets `predict` read a `margin`-wide collar of real image around every core:
    a ViT asked to predict affinities at a block face has no context beyond it, and those
    predictions are its worst exactly where we would otherwise trust them.
    """
    axes = []
    for extent in shape:
        stop = extent - block - margin
        if stop < margin:
            raise SystemExit(
                f"cube axis of {extent} cannot hold a {block}^3 block with a {margin} margin"
            )
        axes.append(list(range(margin, stop + 1, block)))

    grid = [(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]]
    if count > len(grid):
        raise SystemExit(
            f"asked for {count} blocks but only {len(grid)} fit in {shape} "
            f"at block={block} margin={margin}"
        )
    order = np.random.default_rng(seed).permutation(len(grid))
    return [grid[i] for i in sorted(order[:count])]


def tile_grid(
    shape: tuple[int, int, int], target: int
) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Partition a cube into non-overlapping tiles no larger than `target` on each axis.

    For labelling a whole cube rather than sampling a few blocks from it. Each axis is split into
    `ceil(extent / target)` equal parts, so 3000 at target 768 becomes four 750s rather than three
    768s and a 696 remainder -- the tiles come out slightly under target and cover the axis exactly.

    **Non-overlapping is the requirement, not a nicety.** Instance ids are unique only within one
    connected-components call, so each tile is numbered in its own range. Overlapping tiles would
    write two different numberings into the same voxels, and a crop drawn from the overlap would
    read ids from whichever tile was written last while its `bounding_box` claimed the other --
    an affinity target asserting "different object" across a seam that is not there. A partition
    makes that unrepresentable.

    Returns (origin, size) pairs because the tiles are not cubic and the last one on an axis may be
    smaller than the rest.
    """
    axes = []
    for extent in shape:
        count = math.ceil(extent / target)
        size = math.ceil(extent / count)
        axes.append([(index * size, min(size, extent - index * size)) for index in range(count)])
    return [
        ((x, y, z), (sx, sy, sz))
        for x, sx in axes[0]
        for y, sy in axes[1]
        for z, sz in axes[2]
    ]


def cube_seed(cube: Path) -> int:
    """A stable per-cube seed, so seed7's blocks are the same on every rerun and every round.

    `hash()` of a str is salted per process by PYTHONHASHSEED, so it would hand out different
    blocks on every invocation -- and round 2 would then pseudo-label different voxels than
    round 1, quietly destroying the only thing that makes rounds comparable.
    """
    return zlib.crc32(cube.name.encode()) & 0x7FFFFFFF


# ----------------------------------------------------------------- phase 1: predict


def predict_tile(algorithm, image, shape, origin, size, margin, patch, stride, device, label):
    """Blended affinities for one tile's core, predicted with a `margin` collar of real context.

    The collar is read and then cropped away: a ViT asked for affinities at a tile face has no
    context beyond it, and those predictions are its worst exactly where the tiling would otherwise
    trust them. `margin` should be at least `stride`, so every core voxel is central to some patch.
    """
    lo = [min(margin, o) for o in origin]
    hi = [min(margin, extent - (o + b))
          for o, b, extent in zip(origin, size, shape, strict=True)]
    read_origin = [o - l for o, l in zip(origin, lo, strict=True)]
    read_shape = [b + l + h for b, l, h in zip(size, lo, hi, strict=True)]

    print(f"\n[{label}] core {origin} size {size} "
          f"read {tuple(read_shape)} at {tuple(read_origin)}", flush=True)
    affinities = predict(algorithm, image, read_origin, tuple(read_shape), patch, stride,
                         device, AFF_CHANNELS)
    core = affinities[
        :, lo[0]:lo[0] + size[0], lo[1]:lo[1] + size[1], lo[2]:lo[2] + size[2]
    ].copy()
    # Explicit: the read-region array is 51 GB for a 1500^3 tile at margin 128, and it is dead the
    # moment the core is copied out. Without this it stays alive until the next loop iteration
    # rebinds `affinities`, overlapping with the segmentation's own allocations.
    del affinities
    return core


def predict_blocks(
    run_dir: Path, cube: Path, out_dir: Path, tiles, margin: int,
    patch: int, stride: int, step: int | None,
) -> None:
    """Write one 6-channel affinity zarr per tile, each predicted with a margin of context.

    `tiles` is a list of (origin, size) pairs -- sizes rather than one scalar edge, because a
    partition of a cube whose extents are not multiples of the tile size has ragged tiles.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algorithm, loaded = load_algorithm(run_dir, device, step)
    image = open_raw(cube)
    shape = spatial_shape(cube)
    out_dir.mkdir(parents=True, exist_ok=True)

    for index, (origin, size) in enumerate(tiles):
        core = predict_tile(algorithm, image, shape, origin, size, margin, patch, stride,
                            device, f"tile {index + 1}/{len(tiles)}")
        target = out_dir / f"block_{index:04d}.zarr"
        store = zarr.create_array(
            store=str(target), shape=core.shape, dtype="f2",
            chunks=(1, 128, 128, 128), overwrite=True,
        )
        store[:] = core
        store.attrs.update(
            cube=str(cube), run=run_dir.name, step=loaded, origin=list(origin),
            block=list(size), margin=margin, channels=AFF_CHANNELS,
            convention="sigmoid(0.2 * logit), blended in probability space",
        )
        print(f"  wrote {target.name} {core.shape}", flush=True)


# ----------------------------------------------------------------- phase 2: label


def segment_cc(affinities: np.ndarray, threshold: float) -> np.ndarray:
    """(6, X, Y, Z) affinities -> (X, Y, Z) uint32 instances by thresholded connected components.

    Reproduces BANIS' `compute_connected_component_segmentation` -- the post-processing every
    published number on this task was produced with -- but via `cc3d.color_connectivity_graph`
    rather than by importing it. BANIS' version is numba-jitted and its module pulls in torch,
    dask, distributed, filelock and scipy at import time, none of which mia-train's environment
    has; cc3d is already present for the `affinity` extra and is faster besides.

    The affinity convention is **forward**: `affinities[c][v]` is the bond between `v` and
    `v + e_c`, for c in (+x, +y, +z). cc3d's 6-connectivity bitfield wants both directions of
    every edge, so each affinity sets a bit at both endpoints -- bits 1..6 being +x, -x, +y, -y,
    +z, -z. Getting this backwards would shift every boundary by one voxel, so it is pinned by
    hand-checked cases in `self_test_cc`.

    Voxels with no surviving edge come back as 0. That is *not* background -- see the module
    docstring -- and `filter_labels` is what decides which of them are.
    """
    # `np.float32(threshold)`, not the bare Python float. Under NumPy 2's weak promotion a
    # float16 array compared against a Python scalar stays in float16, which rounds the *threshold*
    # to the nearest float16 -- 0.40 becomes 0.400146 -- and flips voxels sitting within 1e-4 of it
    # (measured: 105 of 884,736). Promoting the comparison to float32 makes float16 storage
    # bit-exact against the float32 path, which is what lets the affinities stay float16 in memory.
    x, y, z = affinities.shape[1:]
    # cc3d's connectivity graph is indexed in uint32, so a region of 2^32 voxels or more is
    # rejected outright -- `RuntimeError: maximum length exception`, raised before any work and
    # regardless of shape (measured: 1620^3 = 4.252e9 passes, 1700^3 = 4.913e9 fails; likewise
    # 3000x3000x450 passes and x500 fails). A whole 3000x3000x1350 cube is 2.8x over, which is why
    # tiles exist at all now that memory is no longer the binding constraint.
    if x * y * z >= 2**32:
        raise SystemExit(
            f"tile of {x}x{y}x{z} = {x*y*z:.3e} voxels is at or over cc3d's 2^32 ceiling "
            f"({2**32:.3e}); use a smaller --block"
        )
    hard = affinities[:3] > np.float32(threshold)
    vcg = np.zeros((x, y, z), dtype=np.uint8)
    vcg[:x - 1] |= hard[0, :x - 1].astype(np.uint8) * 1        # +x
    vcg[1:] |= hard[0, :x - 1].astype(np.uint8) * 2            # -x  (same edge, other endpoint)
    vcg[:, :y - 1] |= hard[1, :, :y - 1].astype(np.uint8) * 4  # +y
    vcg[:, 1:] |= hard[1, :, :y - 1].astype(np.uint8) * 8      # -y
    vcg[:, :, :z - 1] |= hard[2, :, :, :z - 1].astype(np.uint8) * 16   # +z
    vcg[:, :, 1:] |= hard[2, :, :, :z - 1].astype(np.uint8) * 32       # -z

    seg = cc3d.color_connectivity_graph(vcg, connectivity=6).astype(np.uint32)
    seg[vcg == 0] = 0

    # `color_connectivity_graph` numbers each component by a seed voxel's index rather than
    # densely -- a 4-voxel line split in two comes back as labels 1 and 3, not 1 and 2 -- so a
    # label VALUE can reach the voxel count. Three things downstream read ids as if they were
    # dense, and all three break quietly without this: the per-block id offset assumes ids fit
    # under `id_stride` (a 384^3 block reached 27,259,103 against a stride of 1e6, for a real
    # component count of 9,976); `np.bincount` in the size filter allocates max_id+1 entries,
    # 27M rather than 10k; and `segments_raw` would report the seed index as a count.
    uniq, inverse = np.unique(seg, return_inverse=True)
    dense = inverse.reshape(seg.shape).astype(np.uint32)
    # np.unique sorts, so uniq[0] is 0 exactly when some voxel is background, and `inverse` then
    # already maps it to 0. With no background at all, every id must shift up to keep 0 free.
    return dense if uniq[0] == 0 else dense + 1


def foreground_score(affinities: np.ndarray) -> np.ndarray:
    """Per-voxel evidence that a voxel is inside *some* object, from its short-range bonds.

    The max over the three short-range channels, not the mean: a voxel on an object's surface is
    strongly bonded along the axes running into the object and weakly bonded across the membrane,
    and averaging would read that as ambiguity when it is in fact a confident surface.

    Note these are `sigmoid(0.2 * logit)` probabilities, so the usable range is compressed --
    a logit of +5 is 0.73, not 0.99. Thresholds must be picked against that scale, which is what
    the `calibrate` subcommand is for.
    """
    return affinities[:3].max(axis=0)


def longrange_disagreement(labels: np.ndarray, affinities: np.ndarray, tau_long: float
                           ) -> np.ndarray:
    """Voxels where CC merged a pair that the long-range channels call *different objects*.

    Connected components reads only the three short-range channels. The three long-range ones --
    the same 10-voxel relationships mutex watershed used as repulsive edges -- are therefore
    completely independent evidence about the very decision CC just made, and by BANIS' own
    measurement they are this model's *better* discriminator (+0.40 to +0.46 separation between
    same-object and different-object, against +0.35 short-range): a 10-voxel question is
    contextual, which is what a ViT answers well.

    So this is mutex watershed's real insight kept, and its role changed. As a *segmenter* the
    repulsive edges over-merged catastrophically (0.024 nERL against CC's 0.584). As a *filter*
    they cost nothing and target the one error that matters: short-range says "connected" while
    long-range says "different" is exactly a merger's signature, and a merger corrupts every
    pairwise target spanning the two fused objects where a split only misstates one seam.

    Both endpoints of a disputed pair abstain, since the evidence does not say which of them is
    on the wrong side of the merge.
    """
    flagged = np.zeros(labels.shape, dtype=bool)
    for axis in range(3):
        extent = labels.shape[axis]
        if extent <= LONG:
            continue
        lo = tuple(slice(0, extent - LONG) if d == axis else slice(None) for d in range(3))
        hi = tuple(slice(LONG, extent) if d == axis else slice(None) for d in range(3))
        # Forward convention, as for the short-range channels: affinities[3 + axis][v] is the
        # bond between v and v + LONG * e_axis.
        merged = (labels[lo] == labels[hi]) & (labels[lo] > 0)
        disputed = merged & (affinities[3 + axis][lo] < np.float32(tau_long))
        flagged[lo] |= disputed
        flagged[hi] |= disputed
    return flagged


def filter_labels(
    instances: np.ndarray, affinities: np.ndarray, tau_bg: float, tau_fg: float, min_size: int,
    tau_long: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """An existing segmentation -> (X, Y, Z) int32 pseudo-labels, with abstentions.

    Split out from `build_labels` because the segmentation does not depend on `tau_*`/`min_size`,
    and `calibrate` sweeps a grid of them.

    Five ways a voxel avoids becoming a positive training target:
      * its foreground evidence is confidently low            -> background (0)
      * its foreground evidence lands in the ambiguous band   -> IGNORE
      * CC left it unlabelled but it is not confidently background -> IGNORE
      * the long-range channels dispute a merge CC made       -> IGNORE
      * it survives into a segment too small to be a real neuron fragment -> IGNORE

    Background is decided from the foreground score rather than from CC's 0, because CC's 0 also
    covers "the threshold cut this voxel loose", which is an absence of evidence rather than
    evidence of absence. Only voxels the score independently calls background become 0.

    Measured caveat on `tau_fg`: it is nearly inert. Any voxel CC labelled already has a
    short-range affinity above CC's own threshold (0.802 at logit +7), which exceeds every useful
    `tau_fg`, so the ambiguous band can only fire on voxels CC already dropped. `tau_bg` is not
    inert -- it owns the background/abstain split among those. `tau_long` exists precisely
    because it reads channels CC does not.
    """
    score = foreground_score(affinities)
    labels = instances.astype(np.int32)

    background = score < np.float32(tau_bg)
    ambiguous = (score >= np.float32(tau_bg)) & (score < np.float32(tau_fg))
    # CC found no surviving edge, yet the voxel does not look like background: abstain rather
    # than assert either way. Under mutex watershed this set was always empty.
    orphan = (instances == 0) & (score >= np.float32(tau_bg))

    labels[ambiguous] = IGNORE
    labels[orphan] = IGNORE
    labels[background] = 0

    # After the label array is settled, so a disputed pair is judged on the ids actually written,
    # and before the size filter, so a segment gutted by disputes is re-measured on what remains.
    disputed = (longrange_disagreement(labels, affinities, tau_long) if tau_long > 0
                else np.zeros(labels.shape, dtype=bool))
    labels[disputed] = IGNORE

    # Size filtering comes last, so it measures what actually survived: a segment mostly carved
    # away by the band or by background may no longer be big enough to be worth training on,
    # and measuring on the raw partition would let a 20-voxel remnant through.
    positive = labels > 0
    small_mask = np.zeros_like(positive)
    if positive.any():
        counts = np.bincount(labels[positive])
        too_small = counts < min_size
        too_small[0] = False   # id 0 is background, never a "small segment"
        # An O(N) lookup rather than np.isin: CC fragments far more than MWS, so the id set can
        # be large enough that np.isin's broadcast is the dominant cost.
        small_mask = too_small[np.maximum(labels, 0)]
        labels[small_mask] = IGNORE

    kept = labels > 0
    stats = {
        "segments_raw": int(instances.max()),
        "segments_kept": int(np.unique(labels[kept]).size),
        "frac_instance": float(kept.mean()),
        "frac_background": float((labels == 0).mean()),
        "frac_ignore": float((labels == IGNORE).mean()),
        "frac_ambiguous": float(ambiguous.mean()),
        "frac_orphan": float(orphan.mean()),
        "frac_small": float(small_mask.mean()),
        "frac_disputed": float(disputed.mean()),
        "voxels": int(labels.size),
    }
    return labels, stats


def build_labels(
    affinities: np.ndarray, tau_bg: float, tau_fg: float, min_size: int, threshold: float,
    tau_long: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """(6, X, Y, Z) affinities -> (X, Y, Z) int32 pseudo-labels. Segment, then filter."""
    return filter_labels(segment_cc(affinities, threshold), affinities, tau_bg, tau_fg, min_size,
                         tau_long)


# ----------------------------------------------------------------- the sidecar container


def create_sidecar(cube: Path, out: Path, label_name: str) -> zarr.Array:
    """An OME-NGFF container that borrows `raw` from the published cube and owns its labels.

    The published cubes are read-only (owned by another user), so pseudo-labels cannot be written
    beside the ground truth under a new key. Instead this builds a container whose `raw` is a
    symlink to the real one -- so the image bytes are never copied for any round -- and whose
    `labels/<name>` is ours. The root metadata is copied verbatim from the source, which is
    exactly right precisely because `raw` is the same tree behind the symlink (verified: the
    published root advertises only `raw/s0..s4`, with no `labels` key to dangle).

    Rounds share a container: `labels/pseudo_r1` and `labels/pseudo_r2` sit side by side, and the
    OME `labels` list accumulates rather than being replaced, so an earlier round stays readable
    after a later one is built.
    """
    shape = spatial_shape(cube)
    out.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(cube / "zarr.json", out / "zarr.json")
    raw_link = out / "raw"
    if raw_link.is_symlink() or raw_link.exists():
        raw_link.unlink()
    # `.resolve()`: symlink_to stores the target verbatim and it would otherwise resolve relative
    # to `out/`, so a relative --cube would leave a dangling `raw` that fails only at read time.
    raw_link.symlink_to((cube / "raw").resolve())

    labels_dir = out / "labels"
    labels_dir.mkdir(exist_ok=True)
    known = []
    if (labels_dir / "zarr.json").exists():
        prior = json.loads((labels_dir / "zarr.json").read_text())
        known = list(prior.get("attributes", {}).get("ome", {}).get("labels", []))
    if label_name not in known:
        known.append(label_name)
    (labels_dir / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "group",
        "attributes": {"ome": {"version": "0.5", "labels": known}},
    }, indent=2))

    group = labels_dir / label_name
    if (group / NATIVE_LEVEL).exists():
        print(f"  note: {label_name} already exists in {out.name}, rebuilding it", flush=True)
    group.mkdir(exist_ok=True)
    # Only `s0` is advertised. miao reads the label pyramid's own multiscales independently of
    # the image's (zarr_meta.py:157, dataset.py:844), so a single-level label pyramid beside a
    # five-level image pyramid is well-formed as long as the requested resolution is the native
    # one -- which for NISB is the 9x9x20 nm that scoring requires anyway.
    (group / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "group",
        "attributes": {"ome": {"version": "0.5", "multiscales": [{
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "xyz"],
            "datasets": [{"path": NATIVE_LEVEL, "coordinateTransformations": [
                {"type": "scale", "scale": [9.0, 9.0, 20.0]}]}],
        }]}},
    }, indent=2))

    return zarr.create_array(
        store=str(group / NATIVE_LEVEL), shape=shape, dtype="i4",
        chunks=(128, 128, 128), fill_value=IGNORE, overwrite=True,
    )


# ----------------------------------------------------------------- subcommands


def load_blocks(aff_dir: Path, cube: Path) -> list[Path]:
    """The block affinity zarrs, checked to actually belong to `cube`."""
    blocks = sorted(aff_dir.glob("block_*.zarr"))
    if not blocks:
        raise SystemExit(f"no block_*.zarr under {aff_dir}")
    for path in blocks:
        stored = zarr.open(str(path), mode="r").attrs.get("cube")
        if stored is not None and Path(stored).resolve() != cube.resolve():
            raise SystemExit(
                f"{path.name} was predicted from {stored}, not {cube}. Scoring or writing labels "
                "against the wrong cube would silently mis-register every voxel."
            )
    return blocks


def block_index(path: Path) -> int:
    """The index encoded in `block_0003.zarr`, so ids depend on identity rather than glob order."""
    return int(path.stem.split("_")[1])


def cmd_predict(args: argparse.Namespace) -> None:
    if args.margin <= LONG:
        raise SystemExit(
            f"--margin {args.margin} must exceed the long-range offset ({LONG}), or the "
            "affinity channels at the core's faces are computed from padding"
        )
    shape = spatial_shape(args.cube)
    if args.full_cube:
        tiles = tile_grid(shape, args.block)
        covered = sum(sx * sy * sz for _, (sx, sy, sz) in tiles)
        print(f"{args.cube.name}: {shape}, {len(tiles)} tiles of {tiles[0][1]} "
              f"covering {100 * covered / (shape[0] * shape[1] * shape[2]):.1f}%", flush=True)
    else:
        origins = block_origins(shape, args.block, args.margin, args.blocks,
                                args.seed if args.seed is not None else cube_seed(args.cube))
        tiles = [(o, (args.block,) * 3) for o in origins]
        print(f"{args.cube.name}: {shape}, {len(tiles)} sampled blocks of {args.block}^3",
              flush=True)
    predict_blocks(args.run_dir, args.cube, args.out_dir, tiles, args.margin,
                   args.patch, args.stride or args.patch // 2, args.step)


def cmd_build(args: argparse.Namespace) -> None:
    blocks = load_blocks(args.aff_dir, args.cube)
    threshold = scale_threshold(args.cc_logit)
    array = create_sidecar(args.cube, args.out, args.label_name)
    bounds, report = [], []
    print(f"cc_logit {args.cc_logit:+.1f} -> threshold {threshold:.4f}", flush=True)

    for path in blocks:
        store = zarr.open(str(path), mode="r")
        origin = [int(v) for v in store.attrs["origin"]]
        # Left float16, as stored: halves the largest allocation in this phase at no
        # numerical cost, since every use below is a comparison or a max.
        affinities = np.asarray(store[:])
        labels, stats = build_labels(affinities, args.tau_bg, args.tau_fg, args.min_size,
                                     threshold, args.tau_long)

        # Instance ids are only unique within one segmentation, so blocks are offset into
        # disjoint ranges. Keyed on the block's own index rather than on enumeration order, so
        # building a subset -- or rebuilding after re-predicting one block -- assigns the same
        # ids to the same voxels as a full build did.
        offset = block_index(path) * args.id_stride
        top = int(labels.max())
        if top >= args.id_stride:
            raise SystemExit(
                f"{path.name} has {top} segments, at or above --id-stride {args.id_stride}; its "
                "ids would collide with the next block's and two objects would read as one"
            )
        labels[labels > 0] += offset

        size = labels.shape
        array[origin[0]:origin[0] + size[0],
              origin[1]:origin[1] + size[1],
              origin[2]:origin[2] + size[2]] = labels
        bounds.append([[o, o + s] for o, s in zip(origin, size, strict=True)])
        report.append({"block": path.name, "origin": origin, **stats})
        print(f"{path.name}: {stats['segments_kept']:>7,} segments  "
              f"inst {stats['frac_instance']:.3f}  bg {stats['frac_background']:.3f}  "
              f"ign {stats['frac_ignore']:.3f}", flush=True)

    first = zarr.open(str(blocks[0]), mode="r").attrs
    meta = {
        "cube": str(args.cube), "label_name": args.label_name,
        "tau_bg": args.tau_bg, "tau_fg": args.tau_fg, "min_size": args.min_size,
        "cc_logit": args.cc_logit, "cc_threshold": threshold, "tau_long": args.tau_long,
        "id_stride": args.id_stride,
        "ignore_index": IGNORE, "bounding_boxes": bounds, "blocks": report,
        "run": first.get("run"), "step": first.get("step"),
    }
    (args.out / "pseudolabel.json").write_text(json.dumps(meta, indent=2))
    mean_ignore = float(np.mean([b["frac_ignore"] for b in report]))
    mean_inst = float(np.mean([b["frac_instance"] for b in report]))
    print(f"\nwrote {args.out}  ({len(report)} blocks, mean instance {mean_inst:.3f}, "
          f"mean ignore {mean_ignore:.3f})", flush=True)


def cmd_label(args: argparse.Namespace) -> None:
    """Predict and build one tile at a time, in one process, without persisting affinities.

    This exists because the two-phase split does not survive large tiles. `predict` writes every
    tile's affinities, `build` reads them all, and only then are they deleted -- which at 384^3
    blocks was 1.4 GB in flight per cube and at 1500^3 tiles is 4 x 36.5 = 146 GB. A hundred
    concurrent cubes then need 14.6 TB of scratch, and on 2026-08-19 that filled a shared 40 TB
    filesystem and killed 67 of 100 jobs with `Disk quota exceeded`.

    Fusing the phases removes the artifact entirely rather than making it smaller: a tile's
    affinities exist only as a numpy array, and the only thing written is the label array itself
    (~2 GB per cube). Peak memory is ~160 GB -- the read-region blend, freed before segmentation --
    against the 240 GB that 16 slots buy.

    What it costs: re-tuning a threshold now means re-predicting, where before the affinities were
    on disk to rebuild from. That was worth paying for while the thresholds were unsettled; it is
    not worth 14.6 TB now that they are.
    """
    if args.margin < args.patch // 2:
        raise SystemExit(
            f"--margin {args.margin} is below the patch stride ({args.patch // 2}), so voxels near "
            "a tile face are never central to any patch and get the model's least reliable "
            "predictions -- which become splits"
        )
    shape = spatial_shape(args.cube)
    if args.full_cube:
        tiles = tile_grid(shape, args.block)
    else:
        origins = block_origins(shape, args.block, args.margin, args.blocks,
                                args.seed if args.seed is not None else cube_seed(args.cube))
        tiles = [(o, (args.block,) * 3) for o in origins]
    covered = sum(math.prod(size) for _, size in tiles)
    threshold = scale_threshold(args.cc_logit)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algorithm, loaded = load_algorithm(args.run_dir, device, args.step)
    image = open_raw(args.cube)
    array = create_sidecar(args.cube, args.out, args.label_name)
    print(f"{args.cube.name}: {shape}, {len(tiles)} tiles of {tiles[0][1]} covering "
          f"{100 * covered / math.prod(shape):.1f}%, cc_logit {args.cc_logit:+.1f} "
          f"-> threshold {threshold:.4f}", flush=True)

    bounds, report = [], []
    for index, (origin, size) in enumerate(tiles):
        core = predict_tile(algorithm, image, shape, origin, size, args.margin, args.patch,
                            args.stride or args.patch // 2, device, f"tile {index + 1}/{len(tiles)}")
        labels, stats = build_labels(core, args.tau_bg, args.tau_fg, args.min_size, threshold,
                                     args.tau_long)
        del core

        # Ids are unique only within one segmentation, so each tile takes a disjoint range keyed on
        # its own index -- not on loop order, so a partial rerun assigns the same ids.
        top = int(labels.max())
        if top >= args.id_stride:
            raise SystemExit(
                f"tile {index} has {top} segments, at or above --id-stride {args.id_stride}; its "
                "ids would collide with the next tile's and two objects would read as one"
            )
        labels[labels > 0] += index * args.id_stride
        array[origin[0]:origin[0] + size[0],
              origin[1]:origin[1] + size[1],
              origin[2]:origin[2] + size[2]] = labels
        bounds.append([[o, o + b] for o, b in zip(origin, size, strict=True)])
        report.append({"block": f"tile_{index:04d}", "origin": list(origin), **stats})
        print(f"  tile {index + 1}: {stats['segments_kept']:>7,} segments  "
              f"inst {stats['frac_instance']:.3f}  bg {stats['frac_background']:.3f}  "
              f"ign {stats['frac_ignore']:.3f}", flush=True)
        del labels

    meta = {
        "cube": str(args.cube), "label_name": args.label_name,
        "tau_bg": args.tau_bg, "tau_fg": args.tau_fg, "tau_long": args.tau_long,
        "min_size": args.min_size, "cc_logit": args.cc_logit, "cc_threshold": threshold,
        "id_stride": args.id_stride, "ignore_index": IGNORE, "margin": args.margin,
        "bounding_boxes": bounds, "blocks": report,
        "run": args.run_dir.name, "step": loaded,
    }
    (args.out / "pseudolabel.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {args.out}  ({len(report)} tiles, mean instance "
          f"{float(np.mean([b['frac_instance'] for b in report])):.3f})", flush=True)


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Sweep the thresholds against real ground truth, on a cube where we have it.

    The point of doing this on a labelled cube: these thresholds decide what the student is
    allowed to learn from, and picking them by eyeballing the pseudo-labels is how confirmation
    bias gets in. Prefer a *held-out* cube (base/val/seed100) -- calibrating on a cube the
    teacher trained on measures memorisation, not generalisation.

    Reported per grid point, over voxel pairs at the short-range offsets (the same relationships
    the affinity loss is built from, restricted to pairs the pseudo-label does not abstain on):

      * `instance`  -- fraction of voxels carrying a positive instance id, i.e. how much
        real supervision survives. Deliberately *not* `1 - frac_ignore`, which counts background
        as supervision and so rewards calling the whole block background.
      * `precision` -- of the pairs the pseudo-label calls same-object, how many really are.
        This is the number that matters: a false "same" is a merger, and mergers corrupt
        quadratically many targets where a split corrupts a seam.
      * `recall`    -- of the truly same-object pairs, how many the pseudo-label asserts.
      * `accuracy`  -- over informative pairs only. Pairs where both sides are background are
        excluded: they are trivially correct and would let a high tau_bg buy a better score.
    """
    blocks = load_blocks(args.aff_dir, args.cube)
    truth = open_labels(args.cube)
    rng = np.random.default_rng(0)
    rows = []

    # Load each block once, with its ground truth and a fixed sampling mask. The mask is drawn
    # once rather than per grid point so the grid points are *paired* -- differences between
    # them are the thresholds, not sampling noise -- and so the sweep does not reallocate a
    # block-sized float array per axis per grid point.
    cached = []
    for path in blocks:
        store = zarr.open(str(path), mode="r")
        origin = [int(v) for v in store.attrs["origin"]]
        # Left float16, as stored: halves the largest allocation in this phase at no
        # numerical cost, since every use below is a comparison or a max.
        affinities = np.asarray(store[:])
        size = affinities.shape[1:]
        gt = np.asarray(truth[origin[0]:origin[0] + size[0],
                              origin[1]:origin[1] + size[1],
                              origin[2]:origin[2] + size[2]]).astype(np.int64)
        picks = [rng.random(tuple(s - (1 if d == axis else 0) for d, s in enumerate(size)))
                 < args.sample for axis in range(3)]
        cached.append((path, affinities, gt, picks))
        print(f"loaded {path.name} {size}", flush=True)

    for logit in args.cc_logit_grid:
        threshold = scale_threshold(logit)
        segmented = [(a, g, p, segment_cc(a, threshold)) for _, a, g, p in cached]
        for tau_bg in args.tau_bg_grid:
            for tau_fg in args.tau_fg_grid:
                if tau_fg < tau_bg:
                    continue
                inst, tp, fp, fn, agree, informative = 0.0, 0, 0, 0, 0, 0
                merges = splits = 0
                for affinities, gt, picks, instances in segmented:
                    labels, stats = filter_labels(instances, affinities, tau_bg, tau_fg,
                                                  args.min_size, args.tau_long)
                    inst += stats["frac_instance"]
                    # Pairwise precision saturates near 1 on this task -- it is dominated by
                    # within-object pairs -- so it cannot rank thresholds on its own. The merge
                    # count is what discriminates, and it is the error that actually costs us.
                    errors = instance_errors(labels, gt, args.min_overlap)
                    merges += errors["merged_pseudo"]
                    splits += errors["split_gt"]
                    size = labels.shape
                    for axis in range(3):
                        lo = tuple(slice(0, s - 1) if d == axis else slice(None)
                                   for d, s in enumerate(size))
                        hi = tuple(slice(1, s) if d == axis else slice(None)
                                   for d, s in enumerate(size))
                        pick = picks[axis] & (labels[lo] != IGNORE) & (labels[hi] != IGNORE)
                        if not pick.any():
                            continue
                        said = (labels[lo][pick] == labels[hi][pick]) & (labels[lo][pick] > 0)
                        real = (gt[lo][pick] == gt[hi][pick]) & (gt[lo][pick] > 0)
                        tp += int((said & real).sum())
                        fp += int((said & ~real).sum())
                        fn += int((~said & real).sum())
                        useful = said | real
                        agree += int((said == real)[useful].sum())
                        informative += int(useful.sum())

                rows.append({
                    "cc_logit": logit, "threshold": threshold,
                    "tau_bg": tau_bg, "tau_fg": tau_fg,
                    "instance": inst / len(cached),
                    "precision": tp / max(tp + fp, 1),
                    "recall": tp / max(tp + fn, 1),
                    "accuracy": agree / max(informative, 1),
                    "merges": merges, "splits": splits,
                })
                r = rows[-1]
                print(f"logit {logit:+.0f} tau_bg={tau_bg:.2f} tau_fg={tau_fg:.2f}  "
                      f"inst={r['instance']:.3f}  prec={r['precision']:.5f}  "
                      f"merges={r['merges']:<4d} splits={r['splits']:<4d}", flush=True)

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}", flush=True)


def gt_boundary(gt: np.ndarray) -> np.ndarray:
    """Voxels with a 6-neighbour carrying a different ground-truth id.

    A cheap stand-in for a distance transform (scipy is not in this environment), used only to ask
    whether abstentions land where the problem is genuinely hard.
    """
    boundary = np.zeros(gt.shape, dtype=bool)
    for axis in range(3):
        lo = tuple(slice(0, -1) if d == axis else slice(None) for d in range(3))
        hi = tuple(slice(1, None) if d == axis else slice(None) for d in range(3))
        differs = gt[lo] != gt[hi]
        boundary[lo] |= differs
        boundary[hi] |= differs
    return boundary


def instance_errors(pseudo: np.ndarray, gt: np.ndarray, min_overlap: int) -> dict:
    """Merge and split counts against ground truth, over voxels both call an object.

    A *merge* is one pseudo-instance overlapping more than one true object; a *split* is one true
    object covered by more than one pseudo-instance. These are the two error modes the threshold
    trades between, and they are not equally costly to us: a merge asserts "same object" across
    every pair spanning two neurons, while a split only misstates a seam.

    `min_overlap` discards incidental contacts, so a handful of voxels bleeding across a membrane
    is not counted as a merge.
    """
    mask = (pseudo > 0) & (gt > 0)
    if not mask.any():
        return {"merged_pseudo": 0, "split_gt": 0, "pseudo_objects": 0, "gt_objects": 0}

    a, b = pseudo[mask].astype(np.int64), gt[mask].astype(np.int64)
    span = int(b.max()) + 1
    pairs, counts = np.unique(a * span + b, return_counts=True)
    pairs, counts = pairs[counts >= min_overlap], counts[counts >= min_overlap]
    pa, pb = pairs // span, pairs % span

    merged = int((np.bincount(np.unique(pa, return_inverse=True)[1]) > 1).sum())
    split = int((np.bincount(np.unique(pb, return_inverse=True)[1]) > 1).sum())
    return {
        "merged_pseudo": merged, "split_gt": split,
        "pseudo_objects": int(np.unique(pa).size), "gt_objects": int(np.unique(pb).size),
    }


def score_against_truth(
    pseudo: np.ndarray, gt: np.ndarray, sample: float, min_overlap: int, seed: int = 0
) -> dict:
    """One block of pseudo-labels versus the ground truth it was never allowed to see."""
    rng = np.random.default_rng(seed)
    trained = pseudo != IGNORE
    abstained = ~trained
    boundary = gt_boundary(gt)

    tp = fp = fn = 0
    for axis in range(3):
        lo = tuple(slice(0, -1) if d == axis else slice(None) for d in range(3))
        hi = tuple(slice(1, None) if d == axis else slice(None) for d in range(3))
        pick = (trained[lo] & trained[hi]) & (rng.random(gt[lo].shape) < sample)
        if not pick.any():
            continue
        said = (pseudo[lo][pick] == pseudo[hi][pick]) & (pseudo[lo][pick] > 0)
        real = (gt[lo][pick] == gt[hi][pick]) & (gt[lo][pick] > 0)
        tp += int((said & real).sum())
        fp += int((said & ~real).sum())
        fn += int((~said & real).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    row = {
        "frac_instance": float((pseudo > 0).mean()),
        "frac_background": float((pseudo == 0).mean()),
        "frac_ignore": float(abstained.mean()),
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": 2 * precision * recall / max(precision + recall, 1e-9),
        # Is abstention aimed at the hard places? The enrichment is P(boundary | abstained) over
        # P(boundary); near 1 means the filter is discarding signal indiscriminately.
        "boundary_rate": float(boundary.mean()),
        "boundary_rate_abstained": float(boundary[abstained].mean()) if abstained.any() else 0.0,
        # How much real object volume the abstention threw away.
        "gt_foreground_abstained": float((gt[abstained] > 0).mean()) if abstained.any() else 0.0,
    }
    row["boundary_enrichment"] = (
        row["boundary_rate_abstained"] / row["boundary_rate"] if row["boundary_rate"] else 0.0
    )
    row.update(instance_errors(pseudo, gt, min_overlap))
    return row


def score_windows(box, edge: int, per_tile: int, seed: int):
    """Deterministic sub-block windows inside one tile's bounding box.

    The oracle scores sub-blocks rather than whole tiles, for two independent reasons.

    *Memory.* A 1500x1500x1350 tile is 3.04e9 voxels, and `score_against_truth` holds the
    pseudo-labels, the ground truth and a random mask over all of them -- ~113 GB, which killed the
    finalise job with TERM_MEMLIMIT. A 384^3 window is ~2 GB.

    *Comparability.* The fragmentation ratio this experiment tracks was baselined at 4.9x on 384^3
    blocks. Scoring a 1500^3 tile instead would move the ratio for reasons unrelated to label
    quality: a larger window holds more complete neurons and proportionally fewer severed by its own
    faces. Holding the scoring window fixed keeps the number meaning the same thing across rounds
    even as the labelling geometry changes.
    """
    lo = [b[0] for b in box]
    hi = [b[1] for b in box]
    size = [h - l for l, h in zip(lo, hi, strict=True)]
    if all(s <= edge for s in size):
        return [tuple(slice(l, h) for l, h in zip(lo, hi, strict=True))]
    rng = np.random.default_rng(seed)
    windows = []
    for _ in range(per_tile):
        origin = [l + int(rng.integers(0, max(s - edge, 1))) for l, s in zip(lo, size, strict=True)]
        windows.append(tuple(slice(o, min(o + edge, h)) for o, h in zip(origin, hi, strict=True)))
    return windows


def cmd_oracle(args: argparse.Namespace) -> None:
    """Score a round's pseudo-labels against the ground truth we pretended not to have.

    `train_100` is synthetic and therefore fully labelled; the experiment ignores that and never
    trains on those labels. This reads them afterwards, purely to measure how good the
    pseudo-labels were -- a measurement a genuinely unlabelled dataset could not provide.

    **Strictly read-only.** The moment these numbers pick a threshold, choose which round to stop
    at, or decide which arm won, the labels have leaked and the experiment stops simulating the
    unlabelled setting. Selection belongs to seed100. This only ever explains, after the fact.

    What to look for across rounds:
      * `pair_precision` up and `frac_instance` up  -> the loop is working
      * `pair_precision` flat while `frac_instance` climbs -> confirmation bias: the teacher is
        asserting more and knowing no more, which is the failure mode iterating invites
      * `merged_pseudo` rising -> the threshold is too permissive for a bootstrap, since a merge
        corrupts far more pairwise targets than a split
      * `boundary_enrichment` near 1 -> abstention is untargeted and is discarding real signal
    """
    manifests = sorted(args.sidecar_root.glob("*.zarr/pseudolabel.json"))
    if not manifests:
        raise SystemExit(f"no */pseudolabel.json under {args.sidecar_root}")

    rows = []
    for manifest in manifests:
        meta = json.loads(manifest.read_text())
        cube = Path(meta["cube"])
        truth = open_labels(cube)
        labels = zarr.open(
            str(manifest.parent / "labels" / meta["label_name"] / NATIVE_LEVEL), mode="r"
        )
        for index, box in enumerate(meta["bounding_boxes"]):
            for window in score_windows(box, args.score_block, args.blocks_per_tile,
                                        zlib.crc32(f"{cube.stem}:{index}".encode())):
                pseudo = np.asarray(labels[window]).astype(np.int64)
                gt = np.asarray(truth[window]).astype(np.int64)
                row = score_against_truth(pseudo, gt, args.sample, args.min_overlap)
                row |= {"cube": cube.stem, "origin": [int(w.start) for w in window]}
                rows.append(row)
                del pseudo, gt
        print(f"{cube.stem}: {len(meta['bounding_boxes'])} tiles -> "
              f"prec={np.mean([r['pair_precision'] for r in rows[-4:]]):.4f} "
              f"inst={np.mean([r['frac_instance'] for r in rows[-4:]]):.3f}", flush=True)

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    summary = {
        "sidecar_root": str(args.sidecar_root),
        "label_name": json.loads(manifests[0].read_text())["label_name"],
        "teacher_run": json.loads(manifests[0].read_text()).get("run"),
        "teacher_step": json.loads(manifests[0].read_text()).get("step"),
        "blocks": len(rows), "cubes": len(manifests),
        "pair_precision": mean("pair_precision"), "pair_recall": mean("pair_recall"),
        "pair_f1": mean("pair_f1"), "frac_instance": mean("frac_instance"),
        "frac_ignore": mean("frac_ignore"), "boundary_enrichment": mean("boundary_enrichment"),
        "gt_foreground_abstained": mean("gt_foreground_abstained"),
        "merged_pseudo": int(sum(r["merged_pseudo"] for r in rows)),
        "split_gt": int(sum(r["split_gt"] for r in rows)),
        "per_block": rows,
    }
    print(f"\n{summary['label_name']}: precision {summary['pair_precision']:.4f}  "
          f"recall {summary['pair_recall']:.4f}  instance {summary['frac_instance']:.3f}  "
          f"merges {summary['merged_pseudo']}  splits {summary['split_gt']}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}", flush=True)


def self_test_cc() -> None:
    """Cases whose answers follow from the affinity convention, not from a reference run."""
    def seg(aff):
        return segment_cc(aff, 0.5)

    a = np.zeros((6, 4, 1, 1), np.float32); a[0, 0, 0, 0] = 1.0
    s = seg(a)[:, 0, 0]
    assert s[0] == s[1] != 0 and s[2] == 0 and s[3] == 0, s

    a = np.zeros((6, 1, 4, 1), np.float32); a[1, 0, :3, 0] = 1.0
    s = seg(a)[0, :, 0]
    assert len(set(s.tolist())) == 1 and s[0] != 0, s

    a = np.zeros((6, 1, 1, 5), np.float32); a[2, 0, 0, 0] = 1.0; a[2, 0, 0, 3] = 1.0
    s = seg(a)[0, 0, :]
    assert s[0] == s[1] != 0 and s[3] == s[4] != 0 and s[0] != s[3] and s[2] == 0, s

    # The one that pins the direction: an edge stored at x=1 joins voxels 1 and 2, not 0 and 1.
    a = np.zeros((6, 3, 1, 1), np.float32); a[0, 1, 0, 0] = 1.0
    s = seg(a)[:, 0, 0]
    assert s[1] == s[2] != 0 and s[0] == 0, s

    # An abstaining voxel must survive filtering as IGNORE, never as background.
    aff = np.full((6, 8, 8, 8), 0.9, np.float32)
    instances = segment_cc(aff, 0.5)
    instances[0, 0, 0] = 0                      # CC left it unlabelled...
    labels, _ = filter_labels(instances, aff, 0.45, 0.55, 1)
    assert labels[0, 0, 0] == IGNORE, labels[0, 0, 0]   # ...but the score says foreground

    print("  self-test: 5/5 cases pass", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="teacher -> 6-channel affinities over N blocks of a cube")
    p.add_argument("run_dir", type=Path, help="the teacher's mia-train run directory")
    p.add_argument("--cube", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--blocks", type=int, default=4,
                   help="how many blocks to sample; ignored with --full-cube")
    p.add_argument("--full-cube", action="store_true",
                   help="label the ENTIRE cube by partitioning it into non-overlapping tiles of "
                        "at most --block on each axis, instead of sampling --blocks of them")
    p.add_argument("--block", type=int, default=384,
                   help="sampled block edge, or the maximum tile edge under --full-cube")
    p.add_argument("--margin", type=int, default=64,
                   help=f"context collar read around each core (must exceed LONG={LONG})")
    p.add_argument("--patch", type=int, default=256,
                   help="inference patch size; must match what the model was trained at, since "
                        "RoPE normalises coordinates by the runtime grid extent")
    p.add_argument("--stride", type=int, default=None, help="default: half the patch")
    p.add_argument("--step", type=int, default=None, help="checkpoint step; default newest")
    p.add_argument("--seed", type=int, default=None, help="default: derived from the cube name")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("build", help="affinities -> filtered pseudo-labels in a sidecar container")
    p.add_argument("--aff-dir", type=Path, required=True)
    p.add_argument("--cube", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="the sidecar .zarr to create")
    p.add_argument("--label-name", default="pseudo_r1")
    p.add_argument("--cc-logit", type=float, default=6.0,
                   help="connected-components threshold, quoted as a logit on BANIS' "
                        "sigmoid(0.2*logit) scale. Higher = fewer mergers, more splits; prefer "
                        "fewer mergers here, since a merger corrupts far more pairwise targets")
    p.add_argument("--tau-bg", type=float, default=0.45,
                   help="below this foreground score a voxel is background (0)")
    p.add_argument("--tau-fg", type=float, default=0.55,
                   help="below this (and above tau-bg) a voxel abstains (-1)")
    p.add_argument("--min-size", type=int, default=200,
                   help="segments smaller than this abstain rather than train")
    p.add_argument("--tau-long", type=float, default=0.0,
                   help="abstain where a long-range affinity below this disputes a merge CC made. "
                        "Reads the channels CC ignores, so it is independent evidence; 0 disables")
    p.add_argument("--id-stride", type=int, default=1_000_000,
                   help="id range reserved per block, so ids never collide between blocks")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("label", help="predict AND build per tile in one pass, no affinity files")
    p.add_argument("run_dir", type=Path, help="the teacher's mia-train run directory")
    p.add_argument("--cube", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="the sidecar .zarr to create")
    p.add_argument("--label-name", default="pseudo_r1")
    p.add_argument("--full-cube", action="store_true")
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--block", type=int, default=1500)
    p.add_argument("--margin", type=int, default=128)
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cc-logit", type=float, default=5.0)
    p.add_argument("--tau-bg", type=float, default=0.40)
    p.add_argument("--tau-fg", type=float, default=0.50)
    p.add_argument("--tau-long", type=float, default=0.30)
    p.add_argument("--min-size", type=int, default=200)
    p.add_argument("--id-stride", type=int, default=1_000_000)
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("calibrate", help="sweep thresholds against ground truth")
    p.add_argument("--aff-dir", type=Path, required=True)
    p.add_argument("--cube", type=Path, required=True, help="a cube WITH ground truth")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cc-logit-grid", type=float, nargs="+", default=[5, 6, 7, 8])
    p.add_argument("--tau-bg-grid", type=float, nargs="+", default=[0.35, 0.45, 0.55])
    p.add_argument("--tau-fg-grid", type=float, nargs="+", default=[0.45, 0.55, 0.65])
    p.add_argument("--min-size", type=int, default=200)
    p.add_argument("--tau-long", type=float, default=0.0)
    p.add_argument("--sample", type=float, default=0.02,
                   help="fraction of candidate voxel pairs to score, per axis")
    p.add_argument("--min-overlap", type=int, default=50,
                   help="voxels of overlap before a pseudo/true pairing counts as a merge")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("oracle", help="score a round's pseudo-labels against withheld ground truth")
    p.add_argument("--sidecar-root", type=Path, required=True,
                   help="a round's directory of <cube>.zarr sidecars written by `build`")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--sample", type=float, default=0.02,
                   help="fraction of candidate voxel pairs to score, per axis")
    p.add_argument("--min-overlap", type=int, default=50,
                   help="voxels of overlap before a pseudo/true pairing counts, so incidental "
                        "bleed across a membrane is not reported as a merge")
    p.add_argument("--score-block", type=int, default=384,
                   help="edge of the window each score is computed over; held fixed across rounds "
                        "so the fragmentation ratio stays comparable as tile geometry changes")
    p.add_argument("--blocks-per-tile", type=int, default=2)
    p.set_defaults(func=cmd_oracle)

    p = sub.add_parser("self-test", help="check the affinity/connectivity conventions")
    p.set_defaults(func=lambda _: self_test_cc())

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
