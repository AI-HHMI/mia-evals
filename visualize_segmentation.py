"""Look at a predicted instance segmentation beside the ground truth it is scored against.

    python visualize_segmentation.py --prediction <artifact>.zarr --logit 0 [--slices 4]
    python visualize_segmentation.py --prediction <artifact>.zarr --block 128 128 128 256
    python visualize_segmentation.py --prediction <labelling>.zarr --min-size 5000

An affinity artifact is thresholded into components here. A labelling artifact -- such as a stored
mutex watershed partition -- is rendered as it stands, because recomputing one costs ~33 minutes
and tens of GB and belongs in a batch job that persists its result, not in a figure script.

Why this exists: PQ = 0.0031 and voi_merge = 6.74 are not legible. They cannot distinguish "the
model merged everything into one object" from "the model shattered everything into dust", and those
are opposite failures needing opposite fixes. One picture separates them immediately.

Reads artifacts only -- no model, no torch, no GPU -- and turns the prediction into a labelling
through the **same postprocess registry the scorer uses**, so what is drawn is what was scored
rather than a second implementation of thresholded components that could disagree with it.

Panels, left to right:

  ground truth      each id its own colour, background black. Many colours = many objects.
  prediction        the same, after postprocessing. ONE colour over everything is the
                    under-segmentation failure; confetti is the over-segmentation one.
  largest component the single biggest predicted object, in white. If it covers the frame, the
                    model has fused the tissue -- which the instance *count* actively hides,
                    because a fused prediction still reports thousands of ids when most of them
                    are two-voxel dust.
  agreement         per ground-truth object: green if the prediction's majority label there is
                    unique to it, red if that label is shared with another object (merged).

**Connected components are computed in 3D and then sliced**, never per slice: 3D connectivity is
what fuses objects, so a 2D pass would show a different -- and much better-looking -- segmentation
than the one that was scored. That is why `--block` exists: a 7-gigavoxel volume cannot be
componented for a figure, and a sub-block honestly labelled is better than a per-slice lie. Note a
sub-block's blob is necessarily no larger than the full volume's, so a block view *understates* the
merging.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

PAD = 6
LABEL_HEIGHT = 18


def palette(ids: np.ndarray, seed: int = 0) -> dict[int, tuple[int, int, int]]:
    """A deterministic colour per id, background black.

    Hashed rather than drawn in sequence so the same id keeps its colour between panels and between
    runs, and so neighbouring ids -- which in a components pass are usually spatial neighbours --
    do not come out in similar shades.
    """
    colours: dict[int, tuple[int, int, int]] = {0: (0, 0, 0)}
    for identifier in ids:
        value = int(identifier)
        if value == 0:
            continue
        rng = np.random.default_rng((value, seed))
        # Bounded away from black so no object is mistaken for background, and away from white so
        # the largest-component panel stays distinguishable.
        colours[value] = tuple(int(c) for c in rng.integers(60, 235, size=3))
    return colours


def colourise(labels: np.ndarray, colours: dict[int, tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for identifier in np.unique(labels):
        out[labels == identifier] = colours.get(int(identifier), (128, 128, 128))
    return out


def agreement(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Per true object: correct (green), merged with another (red), or not predicted (blue).

    Per *object* rather than per voxel, because that is what the metrics count: a true object whose
    voxels mostly carry a label some other true object also mostly carries has been fused with it,
    which is what voi_merge measures and PQ punishes.

    **Three categories, not two.** An earlier version had only "majority label is shared" versus
    "not shared", and background counted as a label like any other -- so once a size filter deleted
    the small components, every true object whose territory became background shared label 0 and
    rendered as *merged*. Missed and merged are opposite failures needing opposite fixes, and
    telling them apart is the entire reason this panel exists; conflating them made the panel
    actively misleading exactly when the filter was doing its job.
    """
    out = np.zeros((*truth.shape, 3), dtype=np.uint8)
    majority: dict[int, int] = {}
    for identifier in np.unique(truth):
        if identifier == 0:
            continue
        inside = prediction[truth == identifier]
        if inside.size == 0:
            continue
        values, counts = np.unique(inside, return_counts=True)
        majority[int(identifier)] = int(values[counts.argmax()])

    # Background is excluded before asking which labels are shared, so "several objects were all
    # deleted" is never reported as "several objects were fused together".
    foreground = [label for label in majority.values() if label != 0]
    shared = {label for label in foreground if foreground.count(label) > 1}
    for identifier, label in majority.items():
        if label == 0:
            out[truth == identifier] = (60, 110, 200)      # not predicted at all
        elif label in shared:
            out[truth == identifier] = (200, 40, 40)       # fused with another true object
        else:
            out[truth == identifier] = (40, 170, 60)       # its own label
    return out


def strip(panels: list[tuple[str, np.ndarray]]) -> Image.Image:
    """Lay panels left to right with captions."""
    height, width = panels[0][1].shape[:2]
    canvas = Image.new(
        "RGB",
        (len(panels) * width + (len(panels) + 1) * PAD, height + LABEL_HEIGHT + 2 * PAD),
        (20, 20, 20),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (caption, image) in enumerate(panels):
        x = PAD + index * (width + PAD)
        canvas.paste(Image.fromarray(image), (x, LABEL_HEIGHT + PAD))
        draw.text((x, PAD // 2), caption, fill=(230, 230, 230))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prediction", type=Path, required=True,
                        help="a prediction artifact; its <name>.gt.zarr sibling is the truth")
    parser.add_argument("--logit", type=float, default=0.0,
                        help="threshold as a logit, for an affinity artifact (ignored for a "
                             "labelling). Use the value the scorer fitted, or the figure shows a "
                             "segmentation nobody scored")
    parser.add_argument("--slices", type=int, default=3, help="how many z slices to lay out")
    parser.add_argument("--min-size", type=int, default=0,
                        help="drop components below this many voxels, as the scorer's fitted "
                             "min_size does. Changes the picture far more than it changes the "
                             "topology: it clears the speckle out of the prediction panel and "
                             "leaves the largest component untouched")
    parser.add_argument("--block", type=int, nargs=4, default=None,
                        metavar=("Z", "Y", "X", "SIZE"),
                        help="component and render only this sub-block, for a volume too large to "
                             "process whole. A block understates merging, never overstates it")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import components  # noqa: F401  (populates the registries)
    from artifact import open_artifact
    from postprocess.registry import PostprocessRegistry

    prediction_artifact = open_artifact(args.prediction)
    truth_path = args.prediction.parent / f"{args.prediction.name.removesuffix('.zarr')}.gt.zarr"
    if not truth_path.exists():
        raise SystemExit(
            f"no ground-truth artifact at {truth_path}. This renders a prediction against the "
            "labelling written beside it by mia-train's src/predict.py."
        )
    truth_artifact = open_artifact(truth_path)

    if args.block:
        z, y, x, size = args.block
        origin = (z, y, x)
        shape = (size,) * 3
    else:
        origin, shape = prediction_artifact.origin, prediction_artifact.spatial_shape

    # The same postprocessor the scorer runs, from the same registry, at the same threshold.
    if prediction_artifact.kind == "affinity":
        processor = PostprocessRegistry.build("cc_threshold", logits=[args.logit])
        params = {"logit": args.logit}
        note = f"cc_threshold(logit={args.logit:+g})"
    else:
        processor = PostprocessRegistry.build("identity")
        params = {}
        # A stored labelling records how it was produced in `convention`; using it means the panel
        # caption names the real algorithm rather than the no-op that read it back.
        note = str(prediction_artifact.convention or "identity")
    if args.min_size:
        note += f" + min_size={args.min_size:,d}"

    print(f"reading {shape} at {origin} from {args.prediction.name}", flush=True)
    raw = prediction_artifact.read(origin, shape, processor.reads_channels())
    print(f"postprocessing with {note} (3D, then sliced)", flush=True)
    predicted = np.asarray(processor(raw, **params))
    if args.min_size:
        from postprocess.size_filter import drop_small_components
        predicted = drop_small_components(predicted.copy(), args.min_size)
    truth = truth_artifact.read(origin, shape)

    sizes = np.bincount(predicted.ravel())
    sizes[0] = 0
    biggest = int(sizes.argmax()) if sizes.size > 1 else 0
    foreground = int((truth != 0).sum())
    covered = int((predicted == biggest).sum()) if biggest else 0
    print(f"  ground truth: {len(np.unique(truth)) - 1} objects over {foreground:,} voxels")
    print(f"  predicted:    {len(np.unique(predicted)) - 1} components; "
          f"largest holds {covered:,} voxels "
          f"({100.0 * covered / max(foreground, 1):.1f}% of GT foreground)")

    truth_colours = palette(np.unique(truth), seed=1)
    predicted_colours = palette(np.unique(predicted), seed=2)

    rows = []
    for index in range(args.slices):
        k = int((index + 1) * shape[0] / (args.slices + 1))
        t, p = truth[k], predicted[k]
        rows.append(strip([
            (f"ground truth  z={k}", colourise(t, truth_colours)),
            (f"prediction  {note}", colourise(p, predicted_colours)),
            ("largest component", np.where(
                (p == biggest)[..., None], np.uint8(255), np.uint8(25)
            ).repeat(3, axis=-1) if biggest else np.zeros((*p.shape, 3), np.uint8)),
            ("green=ok  red=merged  blue=missed", agreement(t, p)),
        ]))

    total_height = sum(r.height for r in rows) + PAD * (len(rows) + 1)
    canvas = Image.new("RGB", (rows[0].width, total_height), (20, 20, 20))
    offset = PAD
    for row in rows:
        canvas.paste(row, (0, offset))
        offset += row.height + PAD

    # Named after the postprocessing actually applied, not after a threshold that a labelling
    # artifact never used -- otherwise an mws figure lands on top of a cc_threshold one.
    slug = "".join(c if c.isalnum() else "_" for c in note).strip("_").replace("__", "_")
    out = args.out or args.prediction.parent / (
        f"seg_{args.prediction.name.removesuffix('.zarr')}_{slug}.png"
    )
    canvas.save(out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
