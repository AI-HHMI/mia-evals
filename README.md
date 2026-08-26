# mia-evals

Scoring and leaderboards for volumetric instance and semantic segmentation.

`mia-evals` answers one question — *how good is this segmentation?* — for models that disagree
about what they emit. Some predict affinity maps, some predict per-class scores, some hand you
finished instance masks. It takes all of them, because it scores **prediction artifacts** rather
than models.

Training lives in [`mia-train`](https://github.com/AI-HHMI/mia-train); data is read through
[`miao`](https://pypi.org/project/miao-io/).

> **Status.** The registries, the artifact contract, the runner and the leaderboard are built and
> tested. What is *not* yet done is the parity gate: `mia_score.py` at the repository root is the
> original NISB scorer, kept until the new path is shown to reproduce its numbers on the same
> artifact. Until then it, not `src/`, is what published NISB figures came from. This repository
> derives from BANIS; see [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).

## Where the boundary is

Whatever produced the prediction writes an artifact; `mia-evals` reads it. Nothing in the core
imports torch, and there is no privileged path for "our own" models.

```
  mia-train (or anything else)                    mia-evals
  ────────────────────────────                    ─────────
  train.py                                        postprocess ─► metrics ─► record ─► leaderboard
  predict.py ──► prediction artifact ────────────►
                 (zarr + self-describing attrs)    numpy · zarr · miao · cc3d · funlib
                                                   no torch, ever
```

Two things follow, and both are the point:

- **A segmentation someone emails you is a first-class submission.** It enters at the artifact
  boundary with no model, no checkpoint and no config.
- **The model is never rebuilt from outside.** Reconstructing a training run in a second repository
  means tracking its config schema, its checkpoint layout and the order it applies LoRA — and
  getting that wrong loads the base weights, silently ignores the adapter, and scores the
  *un-adapted* encoder to a plausible-looking number. Prediction belongs with the code that trained
  the model.

## How format diversity is handled

Many artifact kinds and many postprocessors funnel into exactly **two** scoreable forms. Metrics
attach to those, never to the kind, so nERL does not know or care whether the labelling came from
affinities, a watershed, or a `.zarr` from a collaborator.

| artifact kind | shape | postprocessor | → canonical form |
| --- | --- | --- | --- |
| `affinity` | `(2·rank, *spatial)` | `cc_threshold`, `mws` | instance labelling |
| `boundary` | `(1, *spatial)` | `threshold_cc`, `seeded_watershed` | instance labelling |
| `embedding` | `(D, *spatial)` | `mean_shift` | instance labelling |
| `instances` | `(*spatial)` int | `identity` | instance labelling |
| `class_scores` | `(K, *spatial)` | `argmax`, `per_class_threshold` | class labelling |
| `class_labels` | `(*spatial)` int | `identity` | class labelling |

Adding a model that emits masks directly is an `identity` entry, not a new code path. A 2D model is
handled upstream of the boundary by orthoplane averaging, so it arrives as a 3D artifact like
anything else.

Three rules make that hold:

1. **`background_id` and `ignore_id` are recorded in the artifact.** Producers disagree about `0`.
   For thresholded connected components `0` means "no edge survived here", *not* background — and a
   submission where `0` is a real instance would otherwise have its largest object silently scored
   as background.
2. **Each metric declares what it consumes** — `labels`, `scores`, or both — and retention follows.
   mIoU needs the argmax; AP needs the scores. A 51 GB affinity artifact deleted after scoring
   nERL cannot be revisited for AP.
3. **The hyperparameter fit belongs to the postprocessor.** `cc_threshold` fits a scalar, `mws` a
   stride, `per_class_threshold` K values, `identity` nothing. Fit on val, apply to test — always,
   because a threshold chosen on the split being reported is selecting on the number.

## Configuration

A task is one `.toml`. It references a `miao` YAML for the data rather than restating it, because
those configs are generated with provenance headers and a drift check, and because `miao`'s schema
is `extra="forbid"` — a data config cannot carry task or metric keys.

```toml
task_name = "nisb_base_neuron_instance"

[data]
config_path = "/groups/miaai/miaai/lmd-v0.0.1/configs/evals/v1/neuron_instance_seg/ac3_ac4_mouse_atum.yaml"
volumes = ["em-mouse-Kasthuri15-ac3ac4-cortex/crop-001_ac3_100slices"]   # this task's test split

[predict]      kind = "affinity", tile = 256, overlap = 128, channels = 6
[postprocess]  name = "cc_threshold", fit_on = "val", sweep_logits = [3, 4, 5, 6, 7]
[metric]       names = ["nerl", "voi"], rank_by = "nerl", higher_is_better = true
```

Split membership lives here, as a name filter over the YAML's volumes, for the same reason: `miao`
rejects a `split:` key. One data config per dataset; one task file per (dataset × split × task).

## Leaderboard

Derived, never hand-edited. Each eval writes one small `record.json` — metrics per split, the
checkpoint's run directory and step, **a copy of** that run's resolved config and git commit (so
the entry survives its run directory being deleted), the postprocessor and its fitted
hyperparameter, the region scored, and every component version. Records are git-tracked and
reviewable in a PR; the artifacts they describe stay on `/nrs`.

Two things the renderer has to enforce:

- **Show the postprocessor as a column.** Affinities-plus-CC against a model emitting masks
  directly is a fair end-to-end comparison, but "A beats B" can be a post-processing difference,
  and a table that hides it invites the wrong reading.
- **Never mix extents in one table.** nERL is not comparable across regions — the same model scored
  0.3045 over a whole cube and 0.4192 on a 512³ block of it.

## Running it

```bash
pip install -e '.[instance]'          # + numba, cc3d, networkx, funlib.evaluate

# 1. produce the artifact, in the repo that owns the model
python <mia-train>/src/predict.py <run_dir> --cube <cube>.zarr --out aff.zarr --patch 256

# 2. fit the threshold on val, report on test, write a record
python src/evaluate.py score configs/tasks/nisb_base_neuron_instance.toml \
    --val aff_seed100.zarr --test aff_seed101.zarr --run-dir <run_dir>

# 3. rebuild the table (CI runs the same with --check)
python src/evaluate.py leaderboard

# look at what was predicted, beside the ground truth
python visualize_affinities.py --affinities aff.zarr --cube <cube>.zarr
```

Step 2 refuses to run a multi-candidate sweep without `--val`: sweeping on the reported split and
keeping the best is selecting on the number being published. A single-candidate postprocessor —
`identity`, for a finished segmentation — needs no `--val` at all.

**Layout.** `src/artifact.py` is the contract; `src/postprocess/`, `src/metrics/` and `src/tasks/`
are the three registries; `src/config.py` parses a task `.toml` and resolves the `miao` YAML it
references; `src/report/` writes records and renders the table; `src/utils/` holds the two
functions recycled verbatim from BANIS. Adding a component is one line in `src/components.py`.

Prediction and pseudo-labelling now live in `mia-train` (`src/predict.py`, `src/pseudolabel.py`) —
they run a model, which is the other side of the boundary. Only the pseudo-label *scoring*
subcommands belong here, and move once the metrics they need exist.

## Refactoring plan

1. **Parity.** ✅ registries, artifact contract, runner, leaderboard. ⬜ the gate itself: score one
   artifact through both `mia_score.py` and `src/evaluate.py` and require *exact* equality, then
   separately compare the hand-rolled tiler against a `miao`-sequential one before thresholding.
   Confounding the two makes a discrepancy undiagnosable, which is why they are two gates.
2. **Generalise.** ✅ voxel instance metrics (VOI/Rand/PQ, pinned to funlib's VOI convention);
   ⬜ absorb `mia-train/src/evals/` (registered there but
   skeleton-based, and the corpus' instance volumes ship dense voxel GT and no skeletons.
3. **OME-Zarr semantic segmentation.** Replace `mia-train`'s HuggingFace CellMap path with a `miao`
   config over `/groups/miaai/miaai/lmd-v0.0.1/data`. Verify the 2D orthoplane path survives first
   — it is the reason the 2D dataset exists.
4. **Leaderboard.** Records, renderer, `--check` in CI; backfill from the existing `*_scores.json`.
5. **Clean up `mia-train`.** Move prediction in, move the pseudo-labelling scorer out, point the
   experiment scripts at this CLI.

## Licence

MIT, and MIT upstream: two functions are used verbatim from BANIS, whose notice travels with them
in [`third_party_licenses/banis-MIT.txt`](third_party_licenses/banis-MIT.txt). See
[ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for what was recycled, from whom, and why those two
files must not be tidied.
