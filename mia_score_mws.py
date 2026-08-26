"""Mutex watershed over 6-channel affinities, scored the same way as the connected-components path.

    <banisvenv>/bin/python mia_score_mws.py aff6.zarr --skeleton <cube>/skeleton.pkl --out s.json

Why this exists: BANIS turns affinities into instances by thresholding the three short-range
channels and running connected components, and `mia_score.py` reproduces that exactly so our
numbers are comparable to the published baselines. But the model is trained on six channels, and
measurement showed the three it discards are its *better* predictions -- the long-range channels
separate same-object from different-object more confidently (+0.40 to +0.46) than the short-range
ones (+0.35), because a 10-voxel relationship is a contextual question a ViT answers well while a
1-voxel one needs localization a 16x-upsampled head cannot express.

Mutex watershed (Wolf et al. 2018) uses both: short-range affinities as *attractive* edges and
long-range as *repulsive* ones. It needs **no threshold** -- repulsion does the separating, which is
the point, since thresholded CC has one global scalar deciding every edge and our failure mode is
over-fragmentation.

`affogato`, the reference implementation, is CMake-only and does not pip-install, and the package of
that name on PyPI is an unrelated project. So the algorithm is implemented here, with
`--self-test` checking it against cases whose answers are known by construction.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import zarr

SHORT_OFFSETS = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
LONG = 10
LONG_OFFSETS = ((LONG, 0, 0), (0, LONG, 0), (0, 0, LONG))


def build_edges(
    affinities: np.ndarray, repulsive_stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(6, X, Y, Z) affinities -> flat edge arrays (u, v, priority, attractive).

    Priority is what the algorithm sorts on, and it differs by edge type. For an attractive edge
    the affinity *is* the merge evidence, so priority = a. For a repulsive edge the evidence is
    that the two voxels are *different*, which is strong when the affinity is low, so
    priority = 1 - a. Sorting both by priority descending puts the most confident assertion of
    either kind first, which is exactly what mutex watershed requires.

    `repulsive_stride` subsamples the long-range edges. Every voxel contributing three repulsive
    edges is affordable at small volumes and not at large ones, and the repulsive edges exist to
    place constraints rather than to cover every pair -- taking every k-th voxel keeps the
    constraint field while cutting the edge count by k^3.
    """
    shape = affinities.shape[1:]
    index = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)

    us, vs, priorities, attractive = [], [], [], []
    for channel, offset in enumerate(SHORT_OFFSETS + LONG_OFFSETS):
        is_attractive = channel < len(SHORT_OFFSETS)
        # max(s - o, 0): a negative stop would wrap and silently produce a `u` and `v` of
        # different lengths, which is reachable whenever an axis is shorter than the long-range
        # offset -- true of any block under 10 voxels deep.
        overlap = tuple(slice(0, max(s - o, 0)) for s, o in zip(shape, offset, strict=True))
        shifted = tuple(slice(o, o + max(s - o, 0)) for s, o in zip(shape, offset, strict=True))
        a = affinities[channel][overlap]
        u, v = index[overlap], index[shifted]

        if not is_attractive and repulsive_stride > 1:
            keep = tuple(slice(None, None, repulsive_stride) for _ in shape)
            a, u, v = a[keep], u[keep], v[keep]

        us.append(u.ravel())
        vs.append(v.ravel())
        priorities.append((a if is_attractive else 1.0 - a).ravel())
        attractive.append(np.full(u.size, is_attractive, dtype=bool))

    return (
        np.concatenate(us),
        np.concatenate(vs),
        np.concatenate(priorities).astype(np.float32),
        np.concatenate(attractive),
    )


def mutex_watershed(
    u: np.ndarray, v: np.ndarray, priority: np.ndarray, attractive: np.ndarray, n_nodes: int
) -> np.ndarray:
    """Mutex watershed: edges in descending priority -> a label per node.

    Union-find, plus a set of forbidden partners per cluster. Walking edges from most to least
    confident:

      * an **attractive** edge merges its two clusters, unless a mutex forbids it;
      * a **repulsive** edge records a mutex between them, so no later (weaker) attractive edge can
        join them.

    No threshold appears anywhere. Every attractive edge would eventually merge if left alone, so
    it is the repulsive constraints that carve the partition -- which is why the long-range
    channels matter and why discarding them forces a threshold to do their job.
    """
    parent = np.arange(n_nodes, dtype=np.int64)
    forbidden: dict[int, set[int]] = {}

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:      # path compression
            parent[x], x = root, parent[x]
        return int(root)

    order = np.argsort(-priority, kind="stable")
    for e in order:
        ru, rv = find(int(u[e])), find(int(v[e]))
        if ru == rv:
            continue
        if attractive[e]:
            if rv in forbidden.get(ru, ()):
                continue
            # Union by set size, so the mutex sets merge in the cheaper direction.
            big, small = (ru, rv) if len(forbidden.get(ru, ())) >= len(forbidden.get(rv, ())) else (rv, ru)
            parent[small] = big
            moved = forbidden.pop(small, set())
            if moved:
                target = forbidden.setdefault(big, set())
                for other in moved:
                    target.add(other)
                    partners = forbidden.get(other)
                    if partners is not None:
                        partners.discard(small)
                        partners.add(big)
        else:
            forbidden.setdefault(ru, set()).add(rv)
            forbidden.setdefault(rv, set()).add(ru)

    roots = np.array([find(i) for i in range(n_nodes)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return (labels + 1).astype(np.uint32)


def segment(affinities: np.ndarray, repulsive_stride: int) -> np.ndarray:
    """(6, X, Y, Z) affinities -> (X, Y, Z) uint32 instance labels."""
    shape = affinities.shape[1:]
    u, v, priority, attractive = build_edges(affinities, repulsive_stride)
    labels = mutex_watershed(u, v, priority, attractive, int(np.prod(shape)))
    return labels.reshape(shape)


# ------------------------------------------------------------------ validation


def self_test() -> None:
    """Cases whose answers follow from the definition, not from a reference implementation."""
    # 1. Two nodes, one attractive edge -> one cluster.
    out = mutex_watershed(np.array([0]), np.array([1]), np.array([0.9]), np.array([True]), 2)
    assert len(set(out.tolist())) == 1, out

    # 2. The same pair, but a stronger repulsive edge first -> two clusters. This is the whole
    #    point: repulsion seen earlier blocks a later merge.
    out = mutex_watershed(np.array([0, 0]), np.array([1, 1]), np.array([0.9, 0.5]),
                          np.array([False, True]), 2)
    assert len(set(out.tolist())) == 2, out

    # 3. Weaker repulsion, stronger attraction -> merged, because the merge is processed first and
    #    the mutex arrives too late to undo it.
    out = mutex_watershed(np.array([0, 0]), np.array([1, 1]), np.array([0.9, 0.2]),
                          np.array([True, False]), 2)
    assert len(set(out.tolist())) == 1, out

    # 4. Transitivity of the constraint: a-b merge, b-c mutex, then a-c attractive must be blocked.
    out = mutex_watershed(
        np.array([0, 1, 0]), np.array([1, 2, 2]), np.array([0.9, 0.8, 0.7]),
        np.array([True, False, True]), 3)
    assert out[0] == out[1] != out[2], out

    # NOTE both remaining cases use a volume LARGER than the long-range offset. At 8^3 with
    # LONG=10 there are no long-range edges at all, so a test there would pass or fail for
    # reasons that have nothing to do with repulsion.
    side = 3 * LONG

    # 5. Repulsive channels at affinity 1 mean repulsion 0, i.e. no constraints, so every
    #    connected component collapses to one label -- MWS degenerates to connected components
    #    over the attractive graph, as it must.
    rng = np.random.default_rng(0)
    aff = np.zeros((6, side, side, side), dtype=np.float32)
    aff[:3] = rng.random((3, side, side, side))    # arbitrary attractive weights
    aff[3:] = 1.0
    labels = segment(aff, 1)
    assert len(np.unique(labels)) == 1, np.unique(labels)

    # 6. A plane of repulsion across x splits the volume. The short-range x edge at the plane is
    #    cut, and every long-range x pair straddling it repels.
    cut = side // 2
    aff = np.zeros((6, side, side, side), dtype=np.float32)
    aff[:3] = 0.9
    aff[3:] = 1.0
    aff[0, cut] = 0.0                              # attractive x edge across the plane: no pull
    aff[3, max(cut - LONG + 1, 0) : cut + 1] = 0.0  # long-range x pairs straddling it: full push
    labels = segment(aff, 1)
    assert len(np.unique(labels)) >= 2, f"expected a split, got {np.unique(labels)}"

    print("  self-test: 6/6 cases pass", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("affinities", type=Path, nargs="?", help="6-channel affinity zarr")
    p.add_argument("--skeleton", type=Path, help="the cube's skeleton.pkl")
    p.add_argument("--out", type=Path, help="scores JSON")
    p.add_argument("--origin", type=int, nargs=3, default=None,
                   help="sub-block origin within the affinity array (default: its own origin)")
    p.add_argument("--size", type=int, nargs=3, default=None, help="sub-block size")
    p.add_argument("--repulsive-stride", type=int, default=4,
                   help="subsample long-range edges by this factor per axis")
    p.add_argument("--also-cc", action="store_true",
                   help="also score thresholded connected components on the identical block, so "
                        "the two post-processings are compared on the same data")
    p.add_argument("--cc-logits", type=float, nargs="+", default=[5, 6, 7])
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        if args.affinities is None:
            return

    from mia_evals.utils.connected_components import compute_connected_component_segmentation
    from mia_evals.utils.instance_metrics import compute_metrics

    store = zarr.open(str(args.affinities), mode="r")
    stored_origin = list(store.attrs.get("origin", [0, 0, 0]))
    full = tuple(int(s) for s in store.shape[1:])
    off = [0, 0, 0] if args.origin is None else [
        a - b for a, b in zip(args.origin, stored_origin, strict=True)
    ]
    size = list(full) if args.size is None else list(args.size)
    block = (slice(None), *(slice(o, o + s) for o, s in zip(off, size, strict=True)))
    aff = np.asarray(store[block]).astype(np.float32)
    if aff.shape[0] < 6:
        raise SystemExit(
            f"{args.affinities} has {aff.shape[0]} channels; mutex watershed needs the 6-channel "
            "output (re-run mia_predict.py with --channels 6)"
        )
    absolute = [a + b for a, b in zip(off, stored_origin, strict=True)]
    print(f"affinities {aff.shape} at absolute origin {absolute} "
          f"(run={store.attrs.get('run')} step={store.attrs.get('step')})", flush=True)

    # Crop the skeleton to the block, as mia_score.py does for sub-volumes.
    from mia_score import crop_skeleton
    with open(args.skeleton, "rb") as handle:
        skeleton = pickle.load(handle)
    cropped = crop_skeleton(skeleton, absolute, tuple(aff.shape[1:]))
    if cropped.number_of_nodes() == 0:
        raise SystemExit("no skeleton nodes in this block")
    skeleton_path = str(args.out.with_suffix(".skeleton.pkl"))
    with open(skeleton_path, "wb") as handle:
        pickle.dump(cropped, handle)
    print(f"  {cropped.number_of_nodes()} skeleton nodes in the block; scores are comparable "
          "between methods but are NOT the benchmark's numbers", flush=True)

    results: dict[str, object] = {
        "run": store.attrs.get("run"), "step": store.attrs.get("step"),
        "origin": absolute, "shape": list(aff.shape[1:]),
        "skeleton_nodes": cropped.number_of_nodes(),
        "repulsive_stride": args.repulsive_stride,
    }

    print("\n=== mutex watershed (no threshold) ===", flush=True)
    labels = segment(aff, args.repulsive_stride)
    scores = {k: float(v) for k, v in compute_metrics(labels, skeleton_path).items()
              if isinstance(v, (int, float, np.integer, np.floating))}
    results["mws"] = scores
    print(f"  segments {len(np.unique(labels)):,}  nerl={scores.get('nerl', float('nan')):.4f} "
          f"voi_sum={scores.get('voi_sum', float('nan')):.4f} "
          f"mergers={int(scores.get('n_non0_mergers', -1))} "
          f"splits={int(scores.get('n_splits', -1))}", flush=True)

    if args.also_cc:
        print("\n=== thresholded connected components, same block ===", flush=True)
        cc = []
        for logit in args.cc_logits:
            threshold = float(1.0 / (1.0 + np.exp(-0.2 * logit)))
            seg = compute_connected_component_segmentation(aff[:3] > threshold)
            s = {k: float(v) for k, v in compute_metrics(seg, skeleton_path).items()
                 if isinstance(v, (int, float, np.integer, np.floating))}
            s |= {"logit": logit, "threshold": threshold}
            cc.append(s)
            print(f"  logit {logit:+.0f}: nerl={s.get('nerl', float('nan')):.4f} "
                  f"voi_sum={s.get('voi_sum', float('nan')):.4f} "
                  f"mergers={int(s.get('n_non0_mergers', -1))} "
                  f"splits={int(s.get('n_splits', -1))}", flush=True)
        results["cc"] = cc
        best = max(cc, key=lambda r: r.get("nerl", -1))
        results["cc_best"] = best
        print(f"\n  CC best nERL  {best.get('nerl'):.4f} (logit {best['logit']:+.0f})")
        print(f"  MWS      nERL  {scores.get('nerl'):.4f}")

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
