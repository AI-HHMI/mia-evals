# mia-evals

`mia-evals` scores volumetric instance and semantic segmentations, and maintains a per-task
leaderboard of the results. It is built for 3D electron and light microscopy volumes stored as
OME-Zarr.

The central idea is that `mia-evals` scores *files*, not models. Whatever produced a prediction
writes it to disk as a self-describing Zarr array called a **prediction artifact**, and `mia-evals`
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
declared as a normal dependency. Its build requires Cython but does not say so, so pip's build
isolation fails on it, and a direct git URL in the package metadata would make `mia-evals` itself
impossible to publish. Install it explicitly if you need skeleton scoring:

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
# 1. Produce a prediction artifact. This step belongs to the producer, not to mia-evals.
python <mia-train>/src/predict.py <run_dir> --cube <cube>.zarr --out aff.zarr --patch 256

# 2. Fit the post-processing hyperparameter on validation, report on test, and write a record.
mia-evals score configs/tasks/nisb_base_neuron_instance.toml \
    --val aff_seed100.zarr --test aff_seed101.zarr --run-dir <run_dir>

# 3. Rebuild the leaderboard table from all records.
mia-evals leaderboard
```

To visualize the predictions (which is usually the fastest way to understand a disappointing score):

```bash
mia-evals-viz-affinities   --affinities aff.zarr --cube <cube>.zarr
mia-evals-viz-segmentation --prediction <artifact>.zarr --logit 0 --min-size 5000
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

`instance_seg` and `semantic_seg`. For instance segmentation, `truth_kind` selects where the ground
truth comes from: `skeleton` reads a traced skeleton from inside the volume's Zarr group, `labels`
reads a dense label array through `miao`, and `sibling_artifact` reads a labelling the producer
wrote on the prediction's own grid, which is necessary when prediction and ground truth do not share
a voxel lattice.

## Task configuration

A task is a `.toml` file in `configs/tasks/`. It references a `miao` YAML for the data rather than
restating it, so the same volume definitions serve both training and scoring.

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
rank_by = "skeleton_erl"
```

`rank_by` names a metric rather than one of its keys. Which number ranks, and whether
higher is better, are properties of the metric class, so a config cannot declare a ranking direction
that contradicts the metric it ranks on.

## Validation and test splits

`mia-evals score` takes two artifacts. It sweeps the post-processor's parameters over `--val`,
selects whichever scores best there, applies that single choice to `--test`, and reports only the
test numbers. This is what keeps a swept hyperparameter from quietly selecting on the number being
published, and the runner refuses to proceed if the two splits share a volume.

Split membership lives outside the `miao` YAML, because `miao` validates strictly and rejects keys
it does not recognise. There are two ways to express it. The usual one is a data config per split,
such as `nisb_base_val.yaml` beside `nisb_base_test.yaml`, which makes the split obvious from the
file name. The alternative is a `volumes = [...]` name filter under `[data]` in the task file,
selecting a subset of one dataset-wide YAML, which avoids duplicating volume definitions when a
dataset is split several ways.

When the two splits need different task files, pass the validation one explicitly:

```bash
mia-evals score configs/tasks/lmd_ssl_v1_neuron_instance_test.toml \
    --test  <artifacts>/test  \
    --val   <artifacts>/finetune \
    --val-config configs/tasks/lmd_ssl_v1_neuron_instance_fit.toml \
    --run-dir <run_dir>
```

Because dataset, split and task are three independent choices, `configs/tasks/` grows faster than
the number of datasets alone would suggest.

## The leaderboard

`leaderboard/README.md` is generated and must never be edited by hand. Each evaluation writes one
small JSON record under `leaderboard/records/`, and the table is rendered from those records. A
record carries the scores per volume and in aggregate, the producing run and step, that run's
resolved config and git commit copied inline, the post-processor and the parameters that won on
validation, the exact region scored, and the versions of every component that could move a number.
Copying the provenance inline rather than referencing it means a record stays checkable after its
run directory has been deleted. Records are git-tracked, so a new entry arrives as a reviewable
diff.

The renderer enforces two rules that are easy to get wrong by hand. 1) It always shows the
post-processor as a column, because "A beats B" can be a post-processing difference rather than a
model difference. It also refuses to put two different scored extents in one table, because several
of these metrics change with extent.

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
| `tests/unit/`, `tests/parity/` | fast tests, and tests that reproduce a recorded number |

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
pytest -m parity        # reproduces recorded numbers; needs funlib.evaluate and data on /nrs
```

The `parity` tests exist to catch silent changes in scoring behaviour by re-deriving numbers that
were recorded before a refactor. One of them has already caught a real bug in how the scored region
was inferred.

## Licence

MIT, and MIT upstream. Two modules in `src/utils/` are used verbatim from BANIS, whose notice travels
with them in [`third_party_licenses/banis-MIT.txt`](third_party_licenses/banis-MIT.txt). See
[ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for what was recycled, from whom, and why those two files
must not be tidied.
