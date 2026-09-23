# mia-evals

`mia-evals` scores model predictions on volumetric instance and semantic segmentation tasks, and maintains a per-task
leaderboard of the results.

The central idea is that `mia-evals` scores model predictions, not models themselves. Whatever produced a prediction
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

Run these from the repository root (the leaderboard commands default to the checkout's
`leaderboard/`, or to `./leaderboard/` for a non-editable install):

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
python <mia-train>/src/predict.py <run_dir> --step 50000 --data-config configs/lmd_ssl_v1_neuron_instance/data/test.yaml --out <artifacts>/test
python <mia-train>/src/predict.py <run_dir> --step 50000 --data-config configs/lmd_ssl_v1_neuron_instance/data/fit.yaml  --out <artifacts>/finetune

# 2. Fit the post-processing hyperparams on the finetune half, report on the test half, and write a record.
mia-evals score configs/lmd_ssl_v1_neuron_instance/cc_threshold.toml \
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

An artifact is a **single-resolution Zarr array** carrying a few attributes that say what its
numbers mean. It may be written in either of two forms, which score identically:

- a bare Zarr array, which is what the `write_artifact` helper below produces; or
- a single-level OME-Zarr group whose `multiscales` name exactly one dataset (mia-train's
  `predict.py` writes `<volume>.zarr/s0`). The group's attributes count as the artifact's, and its
  OME voxel size and offset let a neuroglancer view place the prediction on the raw volume; a
  bare array is viewable too, but without that placement.

Zarr v2 and v3 are both readable, and any store `zarr.open` accepts will do. A multiscale *pyramid*
is rejected, even though the source volumes read through `miao` are pyramids: scoring compares one
voxel lattice against the ground truth, and a pyramid does not say which level that should be. If
your prediction is already a pyramid, point at one level's path, such as `prediction.zarr/s0`.
Beyond the single-level `multiscales` entry, no OME-NGFF metadata is required for scoring: the
attributes below are the contract.

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

Optional attributes, each with a specific effect when present:

| attribute | default | meaning |
| --- | --- | --- |
| `origin` | all zeros | the array corner's position within the source volume, in absolute voxel coordinates; a block of a volume must state it or it is scored against the wrong region |
| `ignore_id` | none | a value meaning "unannotated, do not score", commonly `-1` |
| `convention` | empty | free text describing any transform already applied, such as `sigmoid(0.2 * logit)` |
| `source_path` | none | the store the prediction was made from; when present, scoring refuses to match the artifact to a task volume with a different path |
| `covers_full_box` | none | whether the array covers the volume's whole annotated region; decides whether a skeleton is cropped to the artifact and whether a row is labelled a sub-region |
| `run`, `step` | none | name the record (`<run>.step<step>.<route>`) and appear in the table |
| `run_dir` | none | where the producing run's `resolved_config.json` and git commit are copied from when `--run-dir` is not given, and the checkpoint link in the table |
| `source_image_key` | `raw` | the image array in the source store that a neuroglancer view shows under the prediction |

Anything else you add is preserved and copied into the record, so producers are encouraged to write
whatever provenance they have, such as patch size and stride.

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

`mws` always runs the compiled kernel (`src/postprocess/mws_kernel.py`). A block with at most
`max_in_memory_edges` edges, 8 G by default, is watershedded with all its edges in memory, which
peaks at about 37 bytes per edge; a larger one has its edges streamed through the scorer's
`--scratch` in priority bands (`src/postprocess/mws_stream.py`). Both give exactly the partition
of the plain Python implementation, `mutex_watershed_reference`, which the tests use as the
oracle and nothing scores with. So the setting changes time and memory, never a number; a record
names the path taken under `region.volumes.<volume>.postprocess_run`. An 896^3 block, 4.3 G edges
at repulsive stride 1, takes about 20 minutes when the job has its node to itself, and two to five
times longer beside another memory-heavy job, so give mws scoring jobs a whole node. Before
2026-09-23 the scorer ran the Python implementation, about two and a half hours per block.
Re-scoring every gary_comparison mws record then reproduced each pq, fitted parameter and scored
partition exactly. The one visible trace of the change is in VOI: the labels are numbered
differently, VOI sums in label order, and so it can differ in the 15th digit from a record scored
before that date.

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

## Scoring configuration

Configs are laid out one directory per task, named exactly as the task: `configs/<task_name>/<route>.toml`
is a scoring config (its file stem is its `route`) and `configs/<task_name>/data/{test,fit}.yaml` are that
task's data configs, referenced from the scoring config as `data/test.yaml`. A scoring config says which task is scored (`task_name`,
the reported volumes and the ranking metric), where the post-processing sweep is fitted, and which
post-processing route turns the artifact into a labelling; several scoring configs may score one
task through different routes, as `mws.toml` does. Data is referenced as `miao` YAMLs rather
than restated, so prediction and scoring read the same volume definitions.

```toml
task_name = "lmd_ssl_v1_neuron_instance"

[data.test]                                      # the reported volumes
config_path = "data/test.yaml"

[data.fit]                                       # where the sweep below is fitted; may not share a volume with test
config_path = "data/fit.yaml"

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

`route` (top-level, optional) is the short name of the scoring route and the last part of every
record's identifier (see below). It defaults to the post-processor's name; a config sets it when that
name would mislead, as `mws.toml` of the lmd tasks does (`size_filter` runs there, over a stored mutex-watershed
labelling, so `route = "mws"`). Two configs that score one task through different routes must differ
in `route`, or their records would collide.

`rank_by` names a metric rather than one of its keys. Which number ranks, and whether
higher is better, are properties of the metric class, so a config cannot declare a ranking direction
that contradicts the metric it ranks on.

## Validation and test splits

`mia-evals score` takes two sets of artifacts. It sweeps the post-processor's parameters over
`--val`, selects whichever scores best there, applies that single choice to `--test`, and reports
only the test numbers.

Which volumes form each split is declared in the scoring config: `[data.test]` names the reported
volumes and `[data.fit]` the ones the sweep is fitted on. Each names a data config and may add a
`volumes = [...]` filter to select a subset of it, so a split can be its own YAML
(`fit.yaml` beside `test.yaml` in the task's `data/`, which makes the split obvious from the file name) 
or a filter over one dataset-wide YAML. The loader refuses a fit split that shares a volume 
with the reported one, and the scorer refuses `--val` for a task that declares no `[data.fit]`. 
A task whose post-processor has a single candidate (such as `identity`) needs no fit split and may write 
a plain `[data]` with `config_path`.

## The leaderboard

Every file under `leaderboard/` is generated and should not be edited by hand. The directory holds
one subdirectory per task:

```
leaderboard/
  README.md                        index: which tasks exist, and how many entries. No scores.
  fileglancer_shares.json          untracked: this machine's fileglancer data-link keys (see below)
  <task_name>/
    README.md                      that task's table, rendered from ./records/
    records/<identifier>.json      one record per evaluation
```

**Record names.** A record is named `<run>.step<N>.<route>`, and only that way: `<run>` is the
name of the run directory that was passed to the producer's `predict.py` (which already carries the
experiment, the arm and the launch time, e.g. `gary__1a_dinov3_axial_subpixel_20260916_215544`),
`<N>` the checkpoint step, `<route>` the scoring config's `route`. So
`lmd1__2c_dinov3_lvd_ft_subpixel_20260824_183919.step50000.mws` says exactly which checkpoint was
scored and how, and its checkpoint directory can be found by name. There is no `--label`: hand-written
names (`2c_step50000`, `2c_step50000_sizefilter`, ...) made the tables unreadable and were replaced on
2026-09-18. Scoring a run, step and route that already has a record is refused rather than
overwritten; delete the old record if the new scoring supersedes it, or give the config a distinct
`route` if it is a different protocol (`cc_threshold_nosizesweep` is the one legacy example).

`mia-evals score` writes the record and re-renders that task's table, so the two cannot drift
apart through a forgotten second command; every other task's file is left untouched.
`mia-evals leaderboard` rebuilds everything, and `--task <name>` rebuilds a specific task. A
record carries the scores per volume and in aggregate, the producing run and step, that run's
resolved config and git commit copied inline, the post-processor and the parameters that won on
validation, the exact region scored, the paths of the artifacts and kept labellings, the
neuroglancer view links, and the versions of every component. Copying the provenance inline rather than
referencing it means a record stays checkable after its run directory has been deleted. Records are git-tracked, 
so a new entry arrives as a reviewable diff.

**What defines a task:** A task is defined by its name, its reported volumes together with their ground truth 
(store path, label key or skeleton, bounding box), and the metric it ranks on. The
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
| `src/config.py` | parses and validates a scoring config |
| `src/evaluate.py` | the `mia-evals` command: the scoring engine and the CLI |
| `src/components.py` | imports every implementation so the registries populate |
| `src/tasks/` | ground truth and scored region, per task type |
| `src/postprocess/` | artifact kind to canonical form, including the mutex watershed |
| `src/metrics/` | the metrics, each declaring its own primary key and direction |
| `src/report/` | the record format, the leaderboard renderer, and the fileglancer / neuroglancer links |
| `src/viz/` | the two figure commands |
| `src/utils/` | two modules recycled verbatim from BANIS; see `ACKNOWLEDGEMENTS.md` |
| `configs/<task_name>/` | one directory per task: `<route>.toml` scoring configs (task + splits + post-processing route) and `data/{test,fit}.yaml`, the `miao` data configs of that task |
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
