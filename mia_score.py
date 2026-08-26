"""Stage 2 of scoring a mia-train checkpoint on NISB: affinities -> instances -> nERL / VOI.

Run with `banisvenv`, which carries funlib.evaluate and numba. Stage 1 (`mia_predict.py`) runs in
mia-train's environment and produces the affinity zarr this reads, so mia-train never acquires the
evaluation dependencies.

    <banisvenv>/bin/python mia_score.py aff.zarr --skeleton <cube>/skeleton.pkl --out scores.json

The segmentation and the metrics are BANIS' own functions, imported rather than reimplemented:
`compute_connected_component_segmentation` and `compute_metrics`. Reimplementing them would mean
scoring against a private approximation of the benchmark, which is the one thing that must not
differ.

**Thresholds.** BANIS sweeps `sigmoid(0.2 * L)` for integer logits L, keeps the threshold with the
best nERL on val, and applies that single threshold to test. Sweeping on val is what the benchmark
rules allow; sweeping on test would be selecting on the number being reported.

**Sub-volumes.** `--origin`/`--size` in stage 1 produce a region rather than the whole cube. The
skeleton is then cropped to it, which truncates branches at the faces and so *understates* run
length: every truncated branch reads as a split. Such numbers are comparable between models
scored the same way, but they are not the benchmark's numbers and should not be quoted as such.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import zarr
from mia_nisb import spatial_shape


def load_banis():
    """BANIS' segmentation and metric functions, recycled into `utils`.

    Imported inside the function rather than at module scope because both halves carry heavy
    optional dependencies -- numba compiles the components pass on first call, and
    `funlib.evaluate` is git-install-only -- so `--help` and the tests that only exercise
    `crop_skeleton` do not pay for either.
    """
    from utils.connected_components import compute_connected_component_segmentation
    from utils.instance_metrics import compute_metrics

    return compute_connected_component_segmentation, compute_metrics


def crop_skeleton(skeleton, origin, shape):
    """Keep only nodes inside the region and re-index them to region-local coordinates.

    Edges whose endpoints both survive are kept; the rest disappear with their nodes. That is what
    makes a sub-volume score pessimistic rather than wrong -- a branch leaving the region ends
    there, and `expected_run_length` counts the truncation as a split.
    """
    import networkx as nx

    origin = np.asarray(origin)
    upper = origin + np.asarray(shape)
    keep = []
    for node in skeleton.nodes:
        position = np.asarray(skeleton.nodes[node]["index_position"])
        if np.all(position >= origin) and np.all(position < upper):
            keep.append(node)

    cropped = skeleton.subgraph(keep).copy()
    for node in cropped.nodes:
        cropped.nodes[node]["index_position"] = (
            np.asarray(cropped.nodes[node]["index_position"]) - origin
        )
    assert isinstance(cropped, nx.Graph)
    return cropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("affinities", type=Path, help="affinity zarr from mia_predict.py")
    parser.add_argument("--skeleton", type=Path, required=True, help="the cube's skeleton.pkl")
    parser.add_argument("--out", type=Path, required=True, help="where to write the scores JSON")
    parser.add_argument(
        "--cube",
        type=Path,
        default=None,
        help="the cube these affinities cover; defaults to the path recorded in the zarr. Needed "
        "for affinities predicted before the 2026-08-14 reorganisation, which recorded a path "
        "that no longer exists.",
    )
    parser.add_argument(
        "--logits",
        type=float,
        nargs="+",
        default=list(range(-1, 12)),
        help="thresholds as raw logits; stored as sigmoid(0.2*L), matching BANIS' eval_ranges",
    )
    args = parser.parse_args()

    segment, metrics_of = load_banis()

    store = zarr.open(str(args.affinities), mode="r")
    # Kept as stored (float16). Promoting the whole cube to float32 would need 146 GiB and hold
    # 219 GiB transiently during the copy; thresholding compares fine at float16, whose ~1e-3
    # resolution near 0.5-0.8 is far finer than the gap between successive sweep thresholds.
    affinities = np.asarray(store[:])
    origin = list(store.attrs.get("origin", [0, 0, 0]))
    print(f"affinities {affinities.shape} from run={store.attrs.get('run')} "
          f"step={store.attrs.get('step')}", flush=True)

    with open(args.skeleton, "rb") as handle:
        skeleton = pickle.load(handle)
    full_nodes = skeleton.number_of_nodes()

    # Whole cube: hand BANIS the untouched skeleton, so nothing about the official path differs.
    # The comparison below decides whether the skeleton is cropped, so reading the cube's extent
    # wrongly would silently switch a whole-cube score onto the pessimistic sub-volume path.
    cube = args.cube or Path(store.attrs["cube"])
    if not cube.exists():
        # Affinity zarrs written before the reorganisation record a path under the old tree, which
        # was deleted. Said explicitly because the alternative is a KeyError from deep inside zarr
        # that names neither the cube nor the flag that fixes it.
        raise SystemExit(
            f"the cube recorded in {args.affinities.name} does not exist:\n  {cube}\n"
            "These affinities predate the 2026-08-14 NGFF reorganisation. Pass --cube with the "
            "cube's current location, e.g.\n"
            "  --cube /groups/miaai/miaai/lmd-v0.0.1/dev/nisb/train_100/val/seed100.zarr"
        )
    cube_shape = spatial_shape(cube)
    region = tuple(int(s) for s in affinities.shape[1:])
    whole_cube = tuple(origin) == (0, 0, 0) and region == cube_shape

    if whole_cube:
        skeleton_path = str(args.skeleton)
        kept_nodes = full_nodes
        print(f"whole cube: using {args.skeleton} unmodified ({full_nodes} nodes)", flush=True)
    else:
        # `compute_metrics` opens a path, so a cropped skeleton has to be written back out.
        cropped = crop_skeleton(skeleton, origin, region)
        kept_nodes = cropped.number_of_nodes()
        if kept_nodes == 0:
            raise SystemExit("no skeleton nodes inside this region; check --origin/--size")
        skeleton_path = str(args.out.with_suffix(".skeleton.pkl"))
        with open(skeleton_path, "wb") as handle:
            pickle.dump(cropped, handle)
        print(f"sub-volume: {kept_nodes} of {full_nodes} skeleton nodes inside the region; "
              "scores are comparable between models but are NOT the benchmark's numbers",
              flush=True)

    results = []
    for logit in args.logits:
        threshold = float(1.0 / (1.0 + np.exp(-0.2 * logit)))
        hard = affinities[:3] > threshold
        segmentation = segment(hard)
        scores = metrics_of(segmentation, skeleton_path)
        # numpy scalars, not Python ones: funlib returns float32, and `np.float32` is *not* a
        # subclass of `float` (np.float64 is), so a plain isinstance check silently drops nERL
        # while keeping VOI -- which looks exactly like the metric failing.
        scores = {
            k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else None)
            for k, v in scores.items()
        }
        scores = {k: v for k, v in scores.items() if v is not None}
        scores |= {"logit": logit, "threshold": threshold}
        results.append(scores)
        print(f"  logit {logit:+.0f} (thr {threshold:.4f}): "
              f"nerl={scores.get('nerl', float('nan')):.4f} "
              f"voi_sum={scores.get('voi_sum', float('nan')):.4f} "
              f"mergers={scores.get('n_non0_mergers', -1)}", flush=True)

    best = max(results, key=lambda r: r.get("nerl", -1))
    payload = {
        "run": store.attrs.get("run"),
        "step": store.attrs.get("step"),
        "origin": origin,
        "shape": list(affinities.shape[1:]),
        "skeleton_nodes": kept_nodes,
        "whole_cube": whole_cube,
        "skeleton_nodes_full_cube": full_nodes,
        "sweep": results,
        "best_by_nerl": best,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nbest nERL {best.get('nerl'):.4f} at logit {best['logit']:+.0f} "
          f"(voi_sum {best.get('voi_sum'):.4f}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
