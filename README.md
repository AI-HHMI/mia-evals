# mia-evals

Scoring and leaderboards for volumetric instance and semantic segmentation.

`mia-evals` is agnostic to output format. It can work with affinity maps, boundaries, embeddings, 
per-class scores, or finished instance masks. More precisely, it scores **prediction artifacts** 
rather than models.

[`mia-train`](https://github.com/AI-HHMI/mia-train) is the sister repository that handles everything
related to model training. Both repositories use [`miao`](https://pypi.org/project/miao-io/) as 
their dataset interface.

## Boundary between `mia-train` and `mia-evals`

```
  mia-train (or anything else)              mia-evals
  ────────────────────────────              ─────────
  train.py
      │
      ▼
  predict.py ──► prediction artifact ─────► postprocess ─► metrics ─► record ─► leaderboard
           (zarr + self-describing attrs)
```

A trained model generates a "prediction artifact" (affinity or boundary maps, per-voxel embeddings, 
instance masks, or per-class scores). `mia-evals` simply reads and scores it. A third-party generated 
segmentation works perfectly fine here. It enters at the artifact boundary with no model, no checkpoint, 
and no config.

## How format diversity is handled

Many artifact kinds and many postprocessors funnel into exactly two scoreable forms. Metrics
attach to those, never to the kind, so, for example, nERL does not know or care whether the labelling 
came from affinities, a watershed, or a `.zarr` from a collaborator.

| artifact kind | shape | postprocessor | canonical form |
| --- | --- | --- | --- |
| `affinity` | `(2·rank, *spatial)` | `cc_threshold`, `mws` | instance labelling |
| `boundary` | `(1, *spatial)` | `threshold_cc`, `seeded_watershed` | instance labelling |
| `embedding` | `(D, *spatial)` | `mean_shift` | instance labelling |
| `instances` | `(*spatial)` int | `identity` | instance labelling |
| `class_scores` | `(K, *spatial)` | `argmax`, `per_class_threshold` | class labelling |
| `class_labels` | `(*spatial)` int | `identity` | class labelling |

A 2D model is handled upstream of the boundary by orthoplane averaging, so it arrives as a 3D artifact 
like anything else.

## Configs

A task is defined in a `.toml` config file. It references a `miao` YAML for the data rather than restating
it, so the same volume definitions serve training and scoring:

```toml
task_name = "nisb_base_neuron_instance"

[data]
config_path = "../data/nisb_base_test.yaml"

[task]
name = "instance_seg"
truth_kind = "skeleton"          # NISB ships a traced skeleton inside each cube's zarr group
skeleton_name = "skeleton.pkl"

[postprocess]
name = "cc_threshold"
logits = [3, 4, 5, 6, 7]         # fitted on --val, applied to --test
min_sizes = [0]                  # drop components below N voxels; swept like the threshold

[metric]
names = ["skeleton_erl"]
rank_by = "skeleton_erl"         # ranks on that metric's own `primary` key and direction
```

Every section is required except `[task]`'s optional keys. `rank_by` names a metric, not a metric
*key*: which number ranks and whether higher is better are properties of the metric class
(`primary`, `higher_is_better`), so a config cannot declare a direction that contradicts the metric
it is ranking. 

**Splits.** `miao` validates strictly and rejects keys it does not recognise, so a per-volume
`split: test` in the data YAML makes it reject the entire file. Split membership therefore lives
outside the YAML, and there are two ways to express it:

- **A data config per split:** what every config here currently does: `nisb_base_val.yaml` and
  `nisb_base_test.yaml`, or the `lmd_*_singlescale.yaml` pair. Readable, and the file name states
  the split.
- **A name filter in the task file:** `volumes = ["...crop-001_ac3_100slices"]` under `[data]`,
  selecting a subset of one dataset-wide YAML. Avoids duplicating volume definitions when a dataset
  is split several ways.

Either way you need one task file per (dataset × split × task): three independent choices
multiplying, which is why `configs/tasks/` grows faster than the number of datasets suggests. The
runner refuses to fit and report on overlapping volumes, so the two splits must genuinely differ.

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
