# mia-evals

Scoring and leaderboards for volumetric instance and semantic segmentation.

`mia-evals` is agnostic to output format. It can work with affinity maps, boundaries, embeddings, 
per-class scores, or finished instance masks. More precisely, it scores **prediction artifacts** 
rather than models.

[`mia-train`](https://github.com/AI-HHMI/mia-train) is the sister repository that handles everything
related to model training. Both repositories use [`miao`](https://pypi.org/project/miao-io/) as 
their dataset interface.

## The boundary between `mia-train` and `mia-evals`

Whatever produced the prediction writes an "artifact". `mia-evals` simply reads and scores it. Nothing in 
the core imports `torch`, and there is no privileged path for "our own" models.

```
  mia-train (or anything else)                    mia-evals
  ────────────────────────────                    ─────────
  train.py                                        postprocess ─► metrics ─► record ─► leaderboard
  predict.py ──► prediction artifact ────────────►
                 (zarr + self-describing attrs)    numpy · zarr · miao · cc3d · funlib
```

This means that:

- A third-party generated segmentation is a first-class submission here. It enters at the artifact
  boundary with no model, no checkpoint and no config.
- The model is never rebuilt here, so generating model predictions belongs to the code that trained
  the model.

## How format diversity is handled

Many artifact kinds and many postprocessors funnel into exactly two scoreable forms. Metrics
attach to those, never to the kind, so, for example, nERL does not know or care whether the labelling 
came from affinities, a watershed, or a `.zarr` from a collaborator.

| artifact kind | shape | postprocessor | → canonical form |
| --- | --- | --- | --- |
| `affinity` | `(2·rank, *spatial)` | `cc_threshold`, `mws` | instance labelling |
| `boundary` | `(1, *spatial)` | `threshold_cc`, `seeded_watershed` | instance labelling |
| `embedding` | `(D, *spatial)` | `mean_shift` | instance labelling |
| `instances` | `(*spatial)` int | `identity` | instance labelling |
| `class_scores` | `(K, *spatial)` | `argmax`, `per_class_threshold` | class labelling |
| `class_labels` | `(*spatial)` int | `identity` | class labelling |

A 2D model is handled upstream of the boundary by orthoplane averaging, so it arrives as a 3D artifact 
like anything else.

## Configuration

A task is define in a `.toml` file. It references a `miao` YAML for the data rather than restating it, 
because those configs are generated with provenance headers and a drift check, and because `miao`'s schema
declares `extra="forbid"`, so a data config cannot carry task or metric keys.

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
rejects a `split:` key. One data config per dataset, one task file per (dataset × split × task).

## Leaderboard

Leaderboard entries are always derived, not hand-edited. Each eval writes one small `record.json` 
that includes metrics per split, the checkpoint's run directory and step, a copy of that run's resolved 
config and git commit (so the entry survives its run directory being deleted), the postprocessor and its 
fitted hyperparameter, the region scored, and every component version. Records are git-tracked and
reviewable in a PR.

Two things the renderer has to enforce:

- **Show the postprocessor as a column.** Affinities-plus-CC against a model emitting masks
  directly is a fair end-to-end comparison, but "A beats B" can be a post-processing difference,
  and a table that hides it invites the wrong reading.
- **Never mix extents in one table.** nERL is not comparable across regions.

## Running it

```bash
pip install -e '.[instance]'          # + numba, cc3d, networkx, funlib.evaluate

# 1. produce the artifact, in the repo that owns the model
python <mia-train>/src/predict.py <run_dir> --cube <cube>.zarr --out aff.zarr --patch 256

# 2. fit the threshold on val, report on test, write a record
mia-evals score configs/tasks/nisb_base_neuron_instance.toml \
    --val aff_seed100.zarr --test aff_seed101.zarr --run-dir <run_dir>

# 3. rebuild the table (CI runs the same with --check)
mia-evals leaderboard

# look at what was predicted, beside the ground truth
mia-evals-viz-affinities --affinities aff.zarr --cube <cube>.zarr
mia-evals-viz-segmentation --prediction <artifact>.zarr --logit 0 --min-size 5000
```

Installed entry points rather than scripts at the repository root: `pip install mia-evals` ships
these, and none of them depend on the working directory. `mia-evals` is `src/evaluate.py`, and the
two figure commands are in [`src/viz/`](src/viz/).

Step 2 refuses to run a multi-candidate sweep without `--val`: sweeping on the reported split and
keeping the best is selecting on the number being published. A single-candidate postprocessor,
`identity` for a finished segmentation, needs no `--val` at all.

**Layout.** `src/artifact.py` is the contract; `src/postprocess/`, `src/metrics/` and `src/tasks/`
are the three registries; `src/config.py` parses a task `.toml` and resolves the `miao` YAML it
references; `src/report/` writes records and renders the table; `src/utils/` holds the two
functions recycled verbatim from BANIS. Adding a component is one line in `src/components.py`.

Prediction and pseudo-labelling now reside in `mia-train` (`src/predict.py`, `src/pseudolabel.py`). 
They run a model, which is the other side of the boundary. Only the pseudo-label scoring
subcommands belong here, and move once the metrics they need exist.

## Refactoring plan

1. **Parity.** ✅ registries, artifact contract, runner, leaderboard. ✅ the gate itself: exact
   equality on 65 values across 5 logits, recorded in `tests/parity/expected/` and checked by
   `pytest -m parity` (needs the `instance` *and* `dev` extras). ⬜ separately compare the
   hand-rolled tiler against a `miao`-sequential one before thresholding. Confounding the two makes
   a discrepancy undiagnosable, which is why they are two gates.
2. **Generalise.** ✅ voxel instance metrics (VOI/Rand/PQ, pinned to funlib's VOI convention);
   ⬜ absorb `mia-train/src/evals/` (registered there but
   skeleton-based, and the corpus' instance volumes ship dense voxel GT and no skeletons.
3. **OME-Zarr semantic segmentation.** Replace `mia-train`'s HuggingFace CellMap path with a `miao`
   config over `/groups/miaai/miaai/lmd-v0.0.1/data`. Verify the 2D orthoplane path survives first
   (it is the reason the 2D dataset exists).
4. **Leaderboard.** Records, renderer, `--check` in CI; backfill from the existing `*_scores.json`.
5. **Clean up `mia-train`.** Move prediction in, move the pseudo-labelling scorer out, point the
   experiment scripts at this CLI.

## Licence

MIT, and MIT upstream: two functions are used verbatim from BANIS, whose notice travels with them
in [`third_party_licenses/banis-MIT.txt`](third_party_licenses/banis-MIT.txt). See
[ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for what was recycled, from whom, and why those two
files must not be tidied.
