"""Stage 1 of scoring a mia-train checkpoint on NISB: affinities over a cube.

Run with mia-train's environment (torch + the mia-train sources); it needs no BANIS dependency.
Stage 2 (`mia_score.py`) turns the output into instances and scores it, and runs in the separate
`banisvenv` that carries funlib.evaluate and numba. The split is the point: mia-train stays free
of the evaluation dependencies, and nothing here imports BANIS.

    <mia-train venv>/bin/python mia_predict.py <run_dir> --out aff.zarr [--origin X Y Z --size N]

`run_dir` is a mia-train run directory -- it already records `resolved_config.json`, so the model
is rebuilt from what the run actually used rather than from a config that may have moved on.

Two conventions are copied from BANIS deliberately, because the thresholds in stage 2 only mean
the same thing if both halves agree:

  * **`sigmoid(0.2 * logit)`**, BANIS' `scale_sigmoid`. Both codebases train with plain
    `binary_cross_entropy_with_logits`, so the logits are directly comparable; the 0.2 scaling is
    only how BANIS stores and thresholds them, and its `eval_ranges` are `sigmoid(0.2 * L)` for
    integer logits L. Applying it here makes a threshold mean the same thing in both pipelines.
  * **Blending in probability space, weighted by distance from the patch centre.** Overlapping
    patches are averaged after the sigmoid, not before, matching BANIS' accumulation. A weighted
    mean of logits is not the logit of a weighted mean of probabilities, so the order matters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

from mia_nisb import open_raw, read_patch, spatial_shape

# mia-train's sources, which this script imports to rebuild a trained algorithm from its run
# directory. Overridable with the `MIA_TRAIN_SRC` environment variable; the default is a sibling
# checkout, so clones of the two repositories beside each other need no configuration.
#
# Transitional. Rebuilding a training run from outside the repository that trained it is the
# coupling this split exists to remove: prediction is moving into mia-train as its own entrypoint,
# after which mia-evals reads the artifact and this path disappears. See the README.
MIA_TRAIN_SRC = Path(
    os.environ.get("MIA_TRAIN_SRC", Path(__file__).resolve().parent.parent / "mia-train" / "src")
)
# BANIS keeps only the 3 short-range channels at inference (`prediction_channels=3`), and the
# default here matches that so scores stay comparable to the published baselines. The long-range
# channels are trained regardless -- they are 3 of the 6 terms in the loss -- and `--channels 6`
# keeps them, which is what mutex watershed needs as repulsive edges.
SHORT_RANGE_CHANNELS = 3


def scale_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """BANIS' `scale_sigmoid`, reproduced so this file has no BANIS import."""
    return torch.sigmoid(0.2 * x)


def patch_weight(size: int) -> np.ndarray:
    """Confidence of a patch's own prediction: low at its faces, high at its centre.

    A patch sees no context beyond its border, so its edge voxels are its worst; weighting
    overlapping predictions this way hides the seams that would otherwise cut objects at patch
    boundaries -- which connected components would then report as split errors.

    Numerically identical to BANIS' `get_single_pred_weight`, which is
    `distance_transform_cdt` of a ones-cube padded by one voxel. For that particular input the
    chessboard distance transform has a closed form -- the per-axis distance to the outside,
    minimised over axes -- so this needs no scipy, and mia-train's environment stays as it is.
    Verified equal to the scipy result for sizes 8 through 512.
    """
    axis = (np.minimum(np.arange(size), size - 1 - np.arange(size)) + 1).astype(np.float32)
    return np.minimum(
        np.minimum(axis[:, None, None], axis[None, :, None]), axis[None, None, :]
    )


def tile_starts(extent: int, size: int, stride: int) -> list[int]:
    """Patch origins covering `extent`, with the last one pulled back to land inside."""
    starts = list(range(0, max(extent - size, 0) + 1, stride))
    if not starts or starts[-1] != extent - size:
        starts.append(max(extent - size, 0))
    return sorted(set(starts))


def assert_checkpoint_is_fully_consumed(algorithm, checkpoint_dir: Path) -> None:
    """Fail if the checkpoint holds model tensors this rebuild has nowhere to put.

    **DCP loads *into* a state dict, and skips whatever the template does not ask for, silently.**
    That makes an incomplete rebuild the worst kind of bug here: a model reconstructed without its
    LoRA adapter loads every base weight, ignores every `lora_a`/`lora_b`, and predicts with the
    *un-adapted* encoder -- which scores near the released-checkpoint baseline it started from. A
    plausible number, attributed to the wrong model, and nothing anywhere says so.

    Compared against `get_model_state_dict`, the same function `CheckpointManager` saves through,
    rather than against `named_parameters()`: an algorithm may hold one module under two names
    (`affinity_seg` exposes its encoder as both `model` and `encoder`), `named_parameters()`
    deduplicates by tensor identity and would report only one of them, and the other would then look
    orphaned. Only the `model.` half of the checkpoint is checked -- `optim.` and `train_state.` are
    not rebuilt here and are not meant to be.
    """
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    stored = set(FileSystemReader(checkpoint_dir).read_metadata().state_dict_metadata)
    have = {f"model.{key}" for key in get_model_state_dict(algorithm)}
    orphaned = sorted(key for key in stored if key.startswith("model.") and key not in have)
    if not orphaned:
        return

    hint = ""
    if any(".lora_" in key for key in orphaned):
        hint = (
            "\nThese are LoRA adapter tensors. The run trained an adapted encoder, so its "
            "resolved_config.json must carry a [lora] section for this rebuild to reproduce it. If "
            "the section is present and this still fires, the model and the checkpoint disagree "
            "about which projections were adapted."
        )
    raise SystemExit(
        f"{len(orphaned)} tensor(s) in {checkpoint_dir} have no slot in the rebuilt model, so DCP "
        f"would load the rest and ignore these without a word: {orphaned[:6]}"
        f"{' ...' if len(orphaned) > 6 else ''}{hint}"
    )


def load_algorithm(run_dir: Path, device: torch.device, step: int | None = None):
    """Rebuild the trained algorithm (encoder + affinity decoder) from a run directory."""
    # Checked before the import, because `import components` failing names neither the path it
    # looked in nor the variable that fixes it.
    if not (MIA_TRAIN_SRC / "components.py").is_file():
        raise SystemExit(
            f"mia-train's sources are not at {MIA_TRAIN_SRC}. This script rebuilds the trained "
            "algorithm from mia-train's own registries, so it needs that checkout: set "
            "MIA_TRAIN_SRC=<path>/mia-train/src, or clone mia-train beside this repository."
        )
    sys.path.insert(0, str(MIA_TRAIN_SRC))
    import components  # noqa: F401  (populates the registries)
    from algorithms.registry import AlgorithmRegistry
    from engine.checkpoint import CheckpointManager
    from engine.config import LoRAConfig
    from engine.lora import apply_lora
    from models.registry import ModelRegistry

    # `resolved_config.json` records each section as {"name": ..., "kwargs": {...}}, which is
    # exactly what the registries take -- so the model is rebuilt from what the run really used.
    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    model_cfg, algo_cfg, data_cfg = resolved["model"], resolved["algorithm"], resolved["data"]
    model = ModelRegistry.build(model_cfg["name"], **model_cfg["kwargs"])

    # Low-rank adaptation, in the same position `engine.run.build_trainer` applies it: on the bare
    # model, before the algorithm wraps it. `[lora]` is a top-level section rather than part of
    # `[model]`, so rebuilding from `model_cfg` alone produces a *plain* encoder -- see
    # `assert_checkpoint_is_fully_consumed` for what that costs. Absent from every run predating the
    # feature, hence the default: `LoRAConfig()` has rank 0 and is disabled.
    lora_cfg = LoRAConfig(**resolved.get("lora", {}))
    if lora_cfg.enabled():
        print(f"[lora] {apply_lora(model, lora_cfg).summary()}", flush=True)

    algorithm = AlgorithmRegistry.build(
        algo_cfg["name"], model, None,
        input_axes=data_cfg["kwargs"]["output_axes"],
        **algo_cfg["kwargs"],
    )
    algorithm.to(device).eval()

    # Reuse the run's own checkpoint reader so sharded DCP directories are handled the same way
    # training would; the optimizer is a throwaway here and only exists to satisfy the signature.
    optimizer = torch.optim.AdamW(algorithm.parameters(), lr=1e-4)
    manager = CheckpointManager(algorithm, optimizer, run_dir / "checkpoints")

    # Checked before loading, so a mismatch costs a second rather than a 35-minute GPU job whose
    # output is quietly wrong. Only when the directory is actually there: a missing checkpoint is
    # already reported below, and `load_step` names the steps that *do* exist, which is the more
    # useful message of the two.
    path = manager.latest_checkpoint() if step is None else run_dir / "checkpoints" / f"step_{step}"
    if path is not None and path.is_dir():
        assert_checkpoint_is_fully_consumed(algorithm, path)

    loaded = manager.load_latest() if step is None else manager.load_step(step)
    if loaded == 0:
        raise SystemExit(f"no checkpoint found under {run_dir / 'checkpoints'}")
    print(f"loaded step {loaded} from {run_dir.name}", flush=True)
    return algorithm, loaded


@torch.no_grad()
def predict(algorithm, image: zarr.Array, origin, shape, size: int, stride: int,
            device: torch.device, channels: int = SHORT_RANGE_CHANNELS) -> np.ndarray:
    """Blended short-range affinity probabilities over a region -> (3, X, Y, Z) float16."""
    total = np.zeros((channels, *shape), dtype=np.float32)
    weight = np.zeros((1, *shape), dtype=np.float32)
    single = patch_weight(size)[None]

    starts = [tile_starts(shape[axis], size, stride) for axis in range(3)]
    coords = [(x, y, z) for x in starts[0] for y in starts[1] for z in starts[2]]
    print(f"{len(coords)} patches of {size}^3 at stride {stride} over {shape}", flush=True)

    for index, (x, y, z) in enumerate(coords):
        block = read_patch(image, (origin[0] + x, origin[1] + y, origin[2] + z), size)
        # NGFF cubes are already (c, x, y, z), which is what mia-train's encoder wants once a
        # batch axis is added. The old flat cubes were (x, y, z, c) and needed a `moveaxis` here;
        # doing that now would transpose a spatial axis into the channel slot.
        patch = np.asarray(block)[None].astype(np.float32) / 255.0
        volumes = torch.from_numpy(patch).to(device)

        with torch.autocast(device.type, dtype=torch.bfloat16):
            tokens, grid = algorithm.encoder.patch_features(volumes)
            logits = algorithm._decode(tokens, grid, volumes.shape[2:])
        probability = scale_sigmoid(logits.float())[0, :channels].cpu().numpy()

        total[:, x : x + size, y : y + size, z : z + size] += probability * single
        weight[:, x : x + size, y : y + size, z : z + size] += single
        if (index + 1) % 25 == 0 or index + 1 == len(coords):
            print(f"  {index + 1}/{len(coords)}", flush=True)

    # Normalised a slab at a time rather than as one expression. `(total / weight).astype(f16)`
    # materialises a full float32 quotient before downcasting, which for a 1628^3 read region is an
    # extra 86 GB on top of the 86 GB accumulator -- and for a whole cube, 272 GB. Slabbing it costs
    # nothing numerically (the arithmetic is elementwise) and bounds the temporary at one slab.
    out = np.empty(total.shape, dtype=np.float16)
    for start in range(0, total.shape[1], 128):
        stop = start + 128
        out[:, start:stop] = total[:, start:stop] / np.maximum(weight[:, start:stop], 1e-8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="a mia-train run directory")
    parser.add_argument("--cube", type=Path, required=True,
                        help="a published NISB cube, e.g. .../dev/nisb/base/val/seed100.zarr")
    parser.add_argument("--out", type=Path, required=True, help="output affinity zarr")
    parser.add_argument("--channels", type=int, default=SHORT_RANGE_CHANNELS,
                        help="affinity channels to keep: 3 (short-range only, BANIS' convention) "
                             "or 6 (also the long-range channels, for mutex watershed)")
    parser.add_argument("--step", type=int, default=None,
                        help="checkpoint step to load; default is the newest")
    parser.add_argument("--origin", type=int, nargs=3, default=[0, 0, 0])
    parser.add_argument("--size", type=int, nargs=3, default=None, help="default: the whole cube")
    parser.add_argument("--patch", type=int, default=512, help="inference patch size")
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="patch stride; defaults to half the patch (BANIS' overlap). Set equal to --patch to "
        "disable overlap, which is faster and visibly worse at the seams.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algorithm, step = load_algorithm(args.run_dir, device, args.step)

    image = open_raw(args.cube)
    # `spatial_shape`, not `image.shape[:3]`: the NGFF layout puts the channel first, so the old
    # expression would return (c, x, y) and silently predict over the wrong extent.
    shape = tuple(args.size) if args.size else spatial_shape(args.cube)
    stride = args.stride or args.patch // 2

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    affinities = predict(algorithm, image, args.origin, shape, args.patch, stride,
                         device, args.channels)
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        name = torch.cuda.get_device_properties(0)
        print(f"peak GPU memory {peak:.1f} GiB of {name.total_memory / 2**30:.0f} GiB "
              f"({name.name})", flush=True)

    store = zarr.open(str(args.out), mode="w", shape=affinities.shape, dtype="f2",
                      chunks=(1, 256, 256, 256))
    store[:] = affinities
    store.attrs.update(
        run=args.run_dir.name, step=step, origin=list(args.origin), shape=list(shape),
        patch=args.patch, stride=stride, cube=str(args.cube),
        channels=args.channels,
        convention="sigmoid(0.2 * logit), blended in probability space",
    )
    print(f"wrote {args.out}  {affinities.shape} float16", flush=True)


if __name__ == "__main__":
    main()
