"""Look at predicted affinities beside the ground truth they were trained on.

    mia-evals-viz-affinities --affinities <aff.zarr> --cube <cube> [--slices 4]

Reads a prediction artifact, like everything else here -- it does not run a model, so it needs
neither torch nor a GPU, only numpy, zarr and PIL. Produce the artifact first with mia-train's
`src/predict.py`. It takes any affinity zarr that carries the standard attrs, so it is tied to no
particular experiment.

Figures go beside the artifact on /nrs rather than into the repo -- they are binary and
regenerable, and keeping them beside the checkpoint and `resolved_config.json` that produced them
means a figure is never orphaned from the run it describes.

**The scale is fixed to [0, 1] on purpose.** Per-panel autoscaling is the default in most plotting
code and would be actively misleading here: the failure mode worth seeing is that predictions sit
in a narrow band around the positive rate (measured 0.46-0.89 on arm B) instead of committing near
0 or 1. Stretching each panel to its own range would render an uncommitted field as a confident
one. `--stretch` enables it anyway for reading faint structure, and labels the panel when it does.

Values are shown as stored by prediction, i.e. `sigmoid(0.2 * logit)` -- BANIS' convention,
and the number that actually gets thresholded downstream. The header prints the range so the
compression is legible as a number, not just a shade.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr
from PIL import Image, ImageDraw

PAD = 6
HEADER = 34
LABEL = 16


#: NISB's own layout. Local to this module rather than shared: these two readers were the only part
#: of the deleted `mia_nisb.py` still in use, and a repository moving to generic OME-Zarr access
#: through `miao` should not grow a second home for one benchmark's hardcoded label key.
RAW_KEY = "raw"
LABEL_KEY = "labels/public_gt-cell-nisb"
NATIVE_LEVEL = "s0"


def open_raw(cube: Path, level: str = NATIVE_LEVEL) -> zarr.Array:
    """The EM image of a cube as **(c, x, y, z)** uint8 -- channel first."""
    return zarr.open(str(cube), mode="r")[f"{RAW_KEY}/{level}"]


def open_labels(cube: Path, level: str = NATIVE_LEVEL) -> zarr.Array:
    """The ground-truth instance segmentation as (x, y, z) uint16."""
    return zarr.open(str(cube), mode="r")[f"{LABEL_KEY}/{level}"]



def colourise_segmentation(seg: np.ndarray) -> np.ndarray:
    """Instance ids -> stable pseudo-colours, background black.

    Hashed rather than sequential so the same neuron keeps its colour between figures, which is
    what makes two panels comparable by eye.
    """
    out = np.zeros((*seg.shape, 3), dtype=np.uint8)
    ids = np.unique(seg)
    for i in ids[ids > 0]:
        rng = np.random.default_rng(int(i))
        out[seg == i] = rng.integers(60, 256, size=3)
    return out


def affinity_rgb(aff: np.ndarray, stretch: bool) -> np.ndarray:
    """(3, H, W) affinities -> an RGB image: red = x, green = y, blue = z.

    Packing the three short-range channels into one panel is the usual connectomics view: a
    membrane perpendicular to x darkens the red channel only, so the colour says which direction
    is cut. White is "glued in every direction", black is "cut in every direction".
    """
    a = aff.astype(np.float32)
    if stretch:
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / max(hi - lo, 1e-6)
    return (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8).transpose(1, 2, 0)


def grayscale_rgb(img: np.ndarray) -> np.ndarray:
    return np.repeat(img[..., None].astype(np.uint8), 3, axis=2)


def compose(panels: list[list[tuple[str, np.ndarray]]], header: str, out: Path) -> None:
    """A grid of labelled RGB panels -> one PNG."""
    rows, cols = len(panels), len(panels[0])
    h, w = panels[0][0][1].shape[:2]
    canvas = Image.new(
        "RGB",
        (cols * (w + PAD) + PAD, HEADER + rows * (h + LABEL + PAD) + PAD),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, PAD), header, fill=(235, 235, 235))
    for r, row in enumerate(panels):
        for c, (title, arr) in enumerate(row):
            x = PAD + c * (w + PAD)
            y = HEADER + r * (h + LABEL + PAD)
            draw.text((x, y), title, fill=(190, 190, 190))
            canvas.paste(Image.fromarray(arr), (x, y + LABEL))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=95)
    print(f"wrote {out}  ({canvas.width}x{canvas.height})", flush=True)


def ground_truth_affinity(seg: np.ndarray) -> np.ndarray:
    """(X+1, Y+1, Z+1) labels -> (3, X, Y, Z) binary affinities, as training builds them."""
    x, y, z = (s - 1 for s in seg.shape)
    core = seg[:x, :y, :z]
    fg = core > 0
    aff = np.zeros((3, x, y, z), dtype=np.float32)
    aff[0] = (core == seg[1:, :y, :z]) & fg
    aff[1] = (core == seg[:x, 1:, :z]) & fg
    aff[2] = (core == seg[:x, :y, 1:]) & fg
    return aff


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--affinities", type=Path, required=True,
                   help="an affinity zarr from mia-train's src/predict.py")
    p.add_argument(
        "--cube",
        type=Path,
        default=Path("/groups/miaai/miaai/lmd-v0.0.1/dev/nisb/train_100/val/seed100.zarr"),
    )
    p.add_argument("--origin", type=int, nargs=3, default=[1024, 1024, 512])
    p.add_argument("--size", type=int, default=256, help="edge length of the region shown")
    p.add_argument("--slices", type=int, default=4, help="how many z slices to lay out")
    p.add_argument("--stretch", action="store_true",
                   help="rescale each panel to its own range; off by default because it makes an "
                        "uncommitted prediction look confident")
    p.add_argument("--out", type=Path, default=None,
                   help="default: beside the affinity zarr")
    args = p.parse_args()

    origin, n = args.origin, args.size

    store = zarr.open(str(args.affinities), mode="r")
    off = np.asarray(origin) - np.asarray(store.attrs.get("origin", [0, 0, 0]))
    pred = np.asarray(
        store[:, off[0]:off[0] + n, off[1]:off[1] + n, off[2]:off[2] + n]
    ).astype(np.float32)
    step = store.attrs.get("step")
    title = f"{store.attrs.get('run', args.affinities.name)} @ step {step}"
    out_dir = args.out or args.affinities.parent

    # Labels are (x, y, z); the raw image is (c, x, y, z), so its channel is selected from the
    # front. Before the NGFF reorganisation both were spatial-first and the channel came last.
    seg_all = open_labels(args.cube)
    seg = np.asarray(
        seg_all[origin[0]:origin[0] + n + 1, origin[1]:origin[1] + n + 1,
                origin[2]:origin[2] + n + 1]
    ).astype(np.int64)
    img = np.asarray(
        open_raw(args.cube)[
            0, origin[0]:origin[0] + n, origin[1]:origin[1] + n, origin[2]:origin[2] + n]
    )
    gt = ground_truth_affinity(seg)

    # Split by ground truth: the single most diagnostic pair of numbers for a dense affinity head.
    same, cut = pred[gt > 0.5], pred[gt <= 0.5]
    header = (
        f"{title}   region {n}^3 at {tuple(origin)}\n"
        f"predicted range [{pred.min():.3f}, {pred.max():.3f}]   "
        f"mean where GT=1: {same.mean():.3f}   where GT=0: {cut.mean():.3f}   "
        f"separation {same.mean() - cut.mean():+.3f}"
        + ("   [PANELS STRETCHED]" if args.stretch else "")
    )

    zs = np.linspace(n // 8, n - n // 8 - 1, args.slices).astype(int)
    panels = []
    for z in zs:
        err = np.abs(pred[:, :, :, z] - gt[:, :, :, z])
        panels.append([
            (f"image  z={z}", grayscale_rgb(img[:, :, z])),
            ("ground-truth instances", colourise_segmentation(seg[:n, :n, z])),
            ("GT affinity  (r=x g=y b=z)", affinity_rgb(gt[:, :, :, z], False)),
            ("predicted affinity", affinity_rgb(pred[:, :, :, z], args.stretch)),
            ("|error|", affinity_rgb(err, False)),
        ])

    # The step belongs in the filename, not only in the header. Without it every checkpoint of a
    # run writes to the same path, so scoring a second one silently replaces the first -- which
    # happened, and left a figure whose header and filename disagreed about which step it showed.
    stem = title.split()[0].replace("/", "_")
    name = f"affinities_{stem}_step{step}_{n}_{'-'.join(map(str, origin))}.png"
    compose(panels, header, Path(out_dir) / name)
    print(header, flush=True)


if __name__ == "__main__":
    main()
