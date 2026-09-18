# mia-evals

`mia-evals` scores volumetric instance and semantic segmentations, and maintains a per-task
leaderboard of the results. It is built for 3D electron and light microscopy volumes stored as
OME-Zarr.

The central idea is that `mia-evals` scores *files*, not models. Whatever produced a prediction
writes it to disk as a self-describing Zarr array (bare, or the single level of an OME-Zarr
group that also records where it sits in the volume) called a **prediction artifact**, and `mia-evals`
reads that artifact and scores it. Nothing in this repository imports `torch`, loads a checkpoint,
or rebuilds a network. A segmentation produced by a collaborator, a published tool, or a manual
proofreading pass is therefore a first-class submission, on exactly the same footing as one of our
own models.

[`mia-train`](https://github.com/AI-HHMI/mia-train) is the sister repository that trains models and
writes prediction artifacts. Both repositories read data through
[`miao`](https://pypi.org/project/miao-io/).

The prediction artifact is the whole interface between the two repositories. `mia-train` trains a
model and runs inference with it, writing the result to a Zarr array. `mia-evals` picks that array
up and does everything afterwards: post-processing it into a labelling, scoring that labelling
against ground truth, writing a record, and rendering the leaderboard.

Because the boundary is a file on disk rather than a Python API, the two sides are installed, run,
and versioned independently, and neither needs to import the other.

## Installation

`mia-evals` requires Python 3.11 or newer, and is developed and tested on 3.14. After cloning the
repo, install in editable mode:
```bash
pip install -e .
```

This installs everything needed to score every task in this repository, produce the figures, and 
render the leaderboard.

For development, add the `[dev]` extra, which brings in `pytest`, `ruff` and `mypy`:
```bash
pip install -e '.[dev]'
```

### `funlib.evaluate`

The `skeleton_erl` metric computes expected run length through
[`funlib.evaluate`](https://github.com/funkelab/funlib.evaluate), which is not on PyPI and cannot be
declared as a normal dependency. Install it explicitly if you need skeleton scoring:

```bash
pip install cython scipy
pip install --no-build-isolation git+https://github.com/funkelab/funlib.evaluate.git
```

Everything except `skeleton_erl` works without it. The import is lazy, so its absence is only felt
if you actually run that metric, and the tests that need it skip themselves.

### Verifying an installation

Run these from the repository root, since the leaderboard commands default to `./leaderboard/`:

```bash
mia-evals --help                   # the scorer
mia-evals leaderboard --check      # confirms the committed table matches the records
pytest -m unit                     # requires the `dev` extra
```

Expect `pytest -m unit` to report a small number of skips if you have not installed
`funlib.evaluate`.

## Quick start

Scoring takes three steps. The first happens in whichever repository produced the model.

```bash
# 1. Produce prediction artifacts for both halves of the eval set. This step belongs to the producer, not to mia-evals.
python <mia-train>/src/predict.py <run_dir> --step 50000 --data-config configs/data/lmd_ssl_v1_test.yaml     --out <artifacts>/test
python <mia-train>/src/predict.py <run_dir> --step 50000 --data-config configs/data/lmd_ssl_v1_finetune.yaml --out <artifacts>/finetune

# 2. Fit the post-processing hyperparams on the finetune half, report on the test half, and write a record.
mia-evals score configs/tasks/lmd_ssl_v1_neuron_instance.toml \
    --val <artifacts>/finetune --test <artifacts>/test --run-dir <run_dir>

# 3. Rebuild a table from its records. Scoring already does this for the task it scored; this is for after editing or removing a record by hand.
mia-evals leaderboard --task lmd_ssl_v1_neuron_instance     # one task
mia-evals leaderboard                                       # every task, plus the index
```

To visualize a prediction (which is usually the fastest way to understand a disappointing score), *e.g.*:

```bash
mia-evals-viz-segmentation --prediction <artifacts>/test/kasthuri15_ac4.zarr --logit 3 --min-size 50000   # the values the scorer fitted (postprocess.params in the record)
mia-evals-viz-segmentation --prediction <labellings>/test/kasthuri15_ac4.zarr --min-size 50000            # a stored labelling needs no --logit
```

`mia-evals`, `mia-evals-viz-affinities` and `mia-evals-viz-segmentation` are installed console
entry points, so none of them depend on the working directory.

## Prediction artifacts

An artifact must be a **Zarr array**, written as a single resolution level, carrying a few attributes
that say what its numbers mean. Zarr v2 and v3 are both readable, and any store `zarr.open` accepts
will do.

It is deliberately not multiscale OME-Zarr, even though the source volumes read through `miao`
are. Scoring compares one voxel lattice against the ground truth, and a pyramid does not say which
level that should be, so a Zarr *group* is rejected. If your prediction is already a pyramid, point
at a single level's path, such as `prediction.zarr/s0`. No OME-NGFF metadata is required, and none
is read: the attributes below are the entire contract.

Required attributes:

| attribute | required | meaning |
| --- | --- | --- |
| `kind` | always | one of the seven kinds in the table below, which decides what may post-process it |
| `background_id` | labelling kinds only | the value meaning "no object here", commonly `0` |

`background_id` is required rather than assumed for `instances` and `class_labels` because
thresholded connected components emit `0` for "no affinity edge survived here", which happens at
real membrane and also wherever the model was merely unsure. That is a different claim from
"background", and a producer whose `0` is a genuine instance would otherwise have its largest
object scored as background, yielding a plausible-looking wrong number.

Optional attributes:

| attribute | default | meaning |
| --- | --- | --- |
| `origin` | all zeros | the array corner's position within the source volume, in absolute voxel coordinates |
| `ignore_id` | none | a value meaning "unannotated, do not score", commonly `-1` |
| `convention` | empty | free text describing any transform already applied, such as `sigmoid(0.2 * logit)` |

Anything else you add is preserved and copied into the record, so producers are encouraged to write
whatever provenance they have, such as the run directory, checkpoint step, patch size and stride.

When an artifact is opened, two rules are enforced, so a mislabelled one fails immediately rather
than after an hour of scoring: 
- The **channel count must match the declared kind**: `affinity` needs
exactly two channels per spatial axis, `boundary` and `sdt` exactly one, and `instances` and
`class_labels` no channel axis at all. 
- The **dtype must match the declared kind**: labelling kinds
must be an integer type, because float32 represents integers exactly only below `2**24`, so a
labelling stored as float would silently merge distinct ids while still reading back as a valid
labelling.

The simplest way to produce a conforming artifact is the helper, which validates by reading back
what it wrote:

```python
from artifact import write_artifact

write_artifact(
    "prediction.zarr", labels, kind="instances",
    background_id=0, origin=(0, 0, 0),
    run="my_run", step=50000,          # any extra keyword is kept as provenance
)
```

A producer that writes tile by tile sets the same attributes itself; nothing requires this helper.

## How a score is computed

Every evaluation follows the same four stages, and each stage is a registry you can extend without
touching the engine.

**The artifact kind** describes what is in the file, and is recorded in the artifact's own
attributes. A **post-processor** converts that kind into one of exactly two scoreable forms: an
instance labelling or a class labelling. A **metric** attaches to one of those two canonical forms,
never to an artifact kind, which is why expected run length neither knows nor cares whether the
labelling arrived as affinities, as a watershed, or as a finished mask from a collaborator. A
**task** supplies the ground truth and decides which region is scored.

### Artifact kinds

| kind | array shape | post-processors | canonical form |
| --- | --- | --- | --- |
| `affinity` | `(2·rank, *spatial)` | `cc_threshold`, `mws` | instance labelling |
| `instances` | `(*spatial)`, integer | `identity`, `size_filter` | instance labelling |
| `class_scores` | `(K, *spatial)` | `argmax`, `per_class_threshold` | class labelling |
| `class_labels` | `(*spatial)`, integer | `identity` | class labelling |
| `boundary` | `(1, *spatial)` | none yet | instance labelling |
| `embedding` | `(D, *spatial)` | none yet | instance labelling |
| `sdt` | `(1, *spatial)` | none yet | instance labelling |

The last three kinds are recognised but have no post-processor, so an artifact of those kinds
currently cannot be scored. Adding one is the normal way to extend the repository.

### Post-processors

| name | accepts | produces | swept parameters |
| --- | --- | --- | --- |
| `cc_threshold` | `affinity` | instances | `logits`, `min_sizes` |
| `mws` | `affinity` | instances | `repulsive_strides`, `min_sizes` |
| `size_filter` | `instances` | instances | `min_sizes` |
| `identity` | `instances`, `class_labels` | either | none |
| `argmax` | `class_scores` | classes | none |
| `per_class_threshold` | `class_scores` | classes | `thresholds` |

`cc_threshold` thresholds the affinities and takes connected components. `mws` runs a mutex
watershed, which uses the long-range affinity channels that a threshold discards and needs no
threshold at all; it is more accurate on the volumes measured here but far more expensive, and
`src/postprocess/mws.py` documents both.

### Metrics

| name | canonical form | ranks on | also reported |
| --- | --- | --- | --- |
| `voxel_instance` | instances | `pq` (panoptic quality) | VOI split and merge, SQ, RQ, adapted Rand error |
| `skeleton_erl` | instances | `nerl` (normalised expected run length) | VOI split and merge, merger and split counts |
| `semantic` | classes | `mean_iou` | mean Dice, pixel accuracy, classes present |

### Tasks

Currently, `mia-evals` supports `instance_seg` and `semantic_seg`. 

For instance segmentation, `[task].truth_kind` selects where the ground truth comes from:
- `skeleton`: a traced skeleton, read from `skeleton.pkl` inside the volume's Zarr group
- `instances`: a dense instance labelling, read from the volume's own label array over the region the artifact covers
- `instances_resampled`: a dense instance labelling written by the producer on the prediction's own grid

`instances_resampled` exists because a prediction is not always on the volume's own voxel grid. If
the producer resampled the image before predicting (a 6 nm volume predicted at 8 nm, say), one
prediction voxel is no longer one label voxel, and no `origin` can line the prediction up with the
volume's labels. Only the producer knows the exact grid it used, so it writes the ground truth onto
that grid as a second artifact, and the scorer compares the two arrays voxel for voxel.

`semantic_seg` has no `truth_kind`. Its truth is always the volume's label array.

## Task configuration

A task is a `.toml` file in `configs/tasks/`. It references a `miao` YAML for the data rather than
restating it, so prediction and scoring read the same volume definitions.

```toml
task_name = "lmd_ssl_v1_neuron_instance"

[data.test]                                      # the reported volumes
config_path = "../data/lmd_ssl_v1_test.yaml"

[data.fit]                                       # where the sweep below is fitted; may not share a volume with test
config_path = "../data/lmd_ssl_v1_finetune.yaml"

[task]
name = "instance_seg"
truth_kind = "instances_resampled"   # <volume>.gt.zarr beside each prediction, written by mia-train's predict.py

[postprocess]
name = "cc_threshold"
logits = [0, 3, 6]                   # fitted on --val, applied to --test
min_sizes = [0, 500, 5000, 50000]    # drop components below N voxels; swept like the threshold

[metric]
names = ["voxel_instance"]
rank_by = "voxel_instance"           # ranks on its `primary` key, pq; direction is the metric's own
```

`rank_by` names a metric rather than one of its keys. Which number ranks, and whether
higher is better, are properties of the metric class, so a config cannot declare a ranking direction
that contradicts the metric it ranks on.

## Validation and test splits

`mia-evals score` takes two sets of artifacts. It sweeps the post-processor's parameters over
`--val`, selects whichever scores best there, applies that single choice to `--test`, and reports
only the test numbers. This is what keeps a swept hyperparameter from quietly selecting on the
number being published.

Which volumes form each split is declared in the task file: `[data.test]` names the reported
volumes and `[data.fit]` the ones the sweep is fitted on. Each names a data config and may add a
`volumes = [...]` filter to select a subset of it, so a split can be its own YAML
(`lmd_ssl_v1_finetune.yaml` beside `lmd_ssl_v1_test.yaml`, which makes the split obvious from the
file name) or a filter over one dataset-wide YAML. Split membership never lives inside the `miao`
YAML itself, because `miao` validates strictly and rejects keys it does not recognise. The loader
refuses a fit split that shares a volume with the reported one, and the scorer refuses `--val` for
a task that declares no `[data.fit]`. A task whose post-processor has a single candidate (such as
`identity`) needs no fit split and may write a plain `[data]` with `config_path`.

## The leaderboard

Every file under `leaderboard/` is generated and should not be edited by hand. The directory holds
one subdirectory per task:

```
leaderboard/
  README.md                        index: which tasks exist, and how many entries. No scores.
  <task_name>/
    README.md                      that task's table, rendered from ./records/
    records/<identifier>.json      one record per evaluation
```

`mia-evals score` writes the record and re-renders that task's table, so the two cannot drift
apart through a forgotten second command; every other task's file is left untouched. 
`mia-evals leaderboard` rebuilds everything, and `--task <name>` rebuilds a specific task. A
record carries the scores per volume and in aggregate, the producing run and step, that run's
resolved config and git commit copied inline, the post-processor and the parameters that won on
validation, the exact region scored, and the versions of every component that could move a number.
Copying the provenance inline rather than referencing it means a record stays checkable after its
run directory has been deleted. Records are git-tracked, so a new entry arrives as a reviewable
diff.

The renderer enforces two rules that are easy to get wrong by hand. It always shows the
post-processor as a column, because "A beats B" can be a post-processing difference rather than a
model difference. It also refuses to put two different scored extents in one table, because several
of these metrics change with extent.

**What defines a task:** A task is defined by its name, its reported volumes together with their ground truth (store path, label key or skeleton, bounding box), and the metric it ranks on. Nothing else: the
post-processor, the route the truth is read by (`truth_kind`), the split a sweep was fitted on and
the producer are properties of a submission, recorded in full and shown as columns where they vary
within a table. `mia-evals score` refuses a record whose test set or ranking metric differs from
the records already under its `task_name`, and rendering or checking a task whose records disagree
fails the same way. Anything scored on a different set, or ranked differently, needs its own task
name.

## Repository layout

| path | contents |
| --- | --- |
| `src/artifact.py` | reads a prediction artifact and its self-describing attributes |
| `src/config.py` | parses and validates a task `.toml` |
| `src/evaluate.py` | the `mia-evals` command: the scoring engine and the CLI |
| `src/components.py` | imports every implementation so the registries populate |
| `src/tasks/` | ground truth and scored region, per task type |
| `src/postprocess/` | artifact kind to canonical form, including the mutex watershed |
| `src/metrics/` | the metrics, each declaring its own primary key and direction |
| `src/report/` | the record format and the leaderboard renderer |
| `src/viz/` | the two figure commands |
| `src/utils/` | two modules recycled verbatim from BANIS; see `ACKNOWLEDGEMENTS.md` |
| `configs/data/`, `configs/tasks/` | `miao` data YAMLs and task definitions |
| `docs/controls.md` | control experiments: baseline task metric scores without a model |
| `tests/unit/` | fast, single-process tests |

## Extending `mia-evals`

Adding a post-processor, a metric or a task requires writing one class, decorating it with the 
relevant registry, and adding one import line to `src/components.py`. The engine and the registries
themselves never change. A post-processor declares which artifact kinds it `accepts` and which
canonical form it `produces`, and returns its parameter sweep from `search_space()`; the runner
handles fitting that sweep on validation. A metric declares its canonical form, its `primary` key
and its direction.

## Testing

```bash
pytest                  # everything
pytest -m unit          # fast, no large data or external packages
pytest -m parity        # pins voxel_instance's VOI to funlib.evaluate's numbers; skipped without funlib
```

## Licence

MIT, and MIT upstream. Two modules in `src/utils/` are used verbatim from BANIS, whose notice travels
with them in [`third_party_licenses/banis-MIT.txt`](third_party_licenses/banis-MIT.txt). See
[ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for what was recycled, from whom, and why those two files
must not be tidied.
