# Acknowledgements

`mia-evals` began as a working copy of **BANIS**, the reference baseline for the
[Neuron Instance Segmentation Benchmark (NISB)](https://structuralneurobiologylab.github.io/nisb/):

> **BANIS: Baseline for Affinity-based Neuron Instance Segmentation**
> Franz Rieger and Zuzana Urbanová, Structural Neurobiology Lab,
> Max Planck Institute for Biological Intelligence.
> <https://github.com/StructuralNeurobiologyLab/banis> — MIT licensed,
> Copyright (c) 2024 Franz Rieger, Zuzana Urbanová.

Our own evaluation work was developed inside a clone of that repository over 2026, so this
repository starts from that working tree rather than from nothing. It does **not** carry BANIS' git
history: every commit in it is upstream's, describing a PyTorch Lightning / MedNeXt trainer that
`mia-evals` does not contain and does not replace. Keeping the history would have attached nine
commits of unrelated provenance to a repository whose subject is different. Instead the dependency
is recorded here, in full, and upstream's licence travels with the code that came from it, in
[`third_party_licenses/banis-MIT.txt`](third_party_licenses/banis-MIT.txt).

## What is recycled, verbatim

Two functions are used **exactly** as upstream wrote them. Both carry the same attribution in
their own module header, so the provenance survives a file being read on its own.

| now at | from | author | why it is kept |
| --- | --- | --- | --- |
| [`src/mia_evals/utils/connected_components.py`](src/mia_evals/utils/connected_components.py) | `inference.py` `compute_connected_component_segmentation` | Franz Rieger, extended by Zuzana Urbanová (`aecf188`, `7e4834c`) | the post-processing half of the published baseline |
| [`src/mia_evals/utils/instance_metrics.py`](src/mia_evals/utils/instance_metrics.py) | `metrics.py` `compute_metrics`, `adapted_erl`, `evaluate_skeletons` | Zuzana Urbanová (`cb31913`), in turn adapting `funlib.evaluate.expected_run_length` | the scoring half of the published baseline |

Both were copied from upstream commit `0ca3682` and are byte-identical to it. The only changes are
the module docstring, the import block, and the removal of `metrics.py`'s `__main__` handler now
that it is a library rather than a script.

**These two files are not to be tidied, optimised or rewritten.** Every number this repository
reports against the published NISB baselines is comparable *only* because it is the same code. A
faster or cleaner reimplementation would mean scoring against a private approximation of the
benchmark — the one thing that must not differ. A different segmentation rule or a different metric
belongs beside them as a new postprocessor or a new metric, never as an edit to these.

They were extracted rather than imported because `inference.py` pulls in torch, dask,
`dask.distributed`, `filelock` and scipy at module scope — an inference stack this repository
replaced — to reach forty lines of numba that need none of it.

## Conventions inherited, not copied

Two of BANIS' conventions are reproduced in our own code, and are load-bearing rather than
stylistic. They are noted here because they look arbitrary until you know where they came from:

- **`sigmoid(0.2 · logit)`** (`scale_sigmoid`) is how BANIS stores and thresholds affinities, and
  its threshold sweep is `sigmoid(0.2 · L)` for integer logits `L`. A threshold only means the same
  thing in both pipelines if both apply it.
- **Blending overlapping patches in probability space, weighted by distance from the patch
  centre.** A weighted mean of logits is not the logit of a weighted mean of probabilities, so the
  order matters. Our weight function is a closed form of BANIS'
  `distance_transform_cdt`-of-a-padded-ones-cube, verified equal to the scipy result for sizes 8
  through 512.

## What was deliberately not carried over

BANIS' training stack, which `mia-train` replaced: `BANIS.py` (PyTorch Lightning + MedNeXt),
`data.py` (the numpy/monai NISB loader), `config.yaml`, `environment.yaml`, `aff_train.sh`,
`slurm_job_scheduler.py`, `validation_watcher.py`, `show_data.py`, and the dask-based prediction
half of `inference.py`. None of it is a judgement on that code — it solves a problem this
repository does not have.

## Other upstream work this repository stands on

- **[`funlib.evaluate`](https://github.com/funkelab/funlib.evaluate)** (Funke Lab, HHMI Janelia) —
  `expected_run_length`, `rand_voi`, `get_skeleton_lengths`. Expected run length and VOI are the
  benchmark's metrics; this is their reference implementation.
- **Mutex watershed** — Wolf et al., *The Mutex Watershed: Efficient, Parameter-Free Image
  Partitioning* (ECCV 2018). Our implementation in `mia_score_mws.py` is written from the paper,
  because the reference `affogato` is CMake-only and does not pip-install; it is checked against
  cases whose answers follow from the definition (`--self-test`).
- **[`miao`](https://pypi.org/project/miao-io/)** — the OME-NGFF loader every dataset here is read
  through.
- **Affinity-based instance segmentation** — Turaga et al. (2009); Funke et al.,
  [arXiv:1706.00120](https://arxiv.org/abs/1706.00120).
