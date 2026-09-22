# Scoring your own affinity predictions on `gary_comparison_neuron_instance`

Step-by-step instructions for putting an externally produced affinity map on the
`gary_comparison_neuron_instance` leaderboard with `mia-evals`. Written 2026-09-21 for Gary; the
same steps work for any task and any producer.

Everything below runs on the Janelia cluster. Nothing needs a GPU. Budget about half a day of wall
clock, most of it one unattended 5-hour CPU job.

## 0. What you need before starting

1. **Affinity predictions for TWO blocks of the hemibrain Ellipsoid Body crop**
   (`/groups/miaai/miaai/lmd-v0.0.1/data/em-drosophila-flyem-hemibrain/crop-001_EllipsoidBody_x24000_y23000_z17000.zarr`,
   level 0, 8 nm, axes `x, y, z` on disk, 5000^3 voxels). Coordinates below are level-0 voxel
   indices of that store in `[x, y, z]` order, half-open `[lo, hi)`:

   | block | what it is for | the box we scored ourselves | full annotated box |
   | --- | --- | --- | --- |
   | **test** | the reported score | `[4052, 4948) x [4052, 4948) x [4052, 4948)` (896^3) | `[4000, 5000)^3` |
   | **fit** | fitting the size filter, never reported | `[4052, 4948) x [4052, 4948) x [3052, 3948)` (896^3) | `[4000, 5000) x [4000, 5000) x [3000, 4000)` |

   The fit block is required, not optional: the scorer sweeps the size filter and refuses to pick
   the best setting on the test block ("selecting on the number being reported"). Predict the fit
   block with the same model and settings as the test block.

   **Predict exactly the 896^3 test box above if you want your row to sit in the same table group as
   ours.** The leaderboard groups rows by the region they were scored on and marks groups scored on
   different extents as not comparable. Our rows are on that 896^3 centre (it is what our tiling
   lattice covers); a prediction on the full 1000^3 corner is welcome but lands in its own group.

2. **Six affinity channels per voxel**, in this order, with these meanings. `mia-evals` hard-codes
   them (`src/postprocess/mws.py`, `SHORT_OFFSETS` and `LONG_OFFSETS`):

   | channel | offset (in voxels, along the store's axes) | role in the mutex watershed |
   | ---: | --- | --- |
   | 0 | `(+1, 0, 0)` | attractive |
   | 1 | `(0, +1, 0)` | attractive |
   | 2 | `(0, 0, +1)` | attractive |
   | 3 | `(+10, 0, 0)` | repulsive |
   | 4 | `(0, +10, 0)` | repulsive |
   | 5 | `(0, 0, +10)` | repulsive |

   Channel `c` at voxel `p` is the affinity between voxel `p` and voxel `p + offset_c`.
   **1 means "same object", 0 means "different objects / boundary".** Values must be probabilities in
   `[0, 1]`; if your network emits logits, apply a sigmoid first. (The watershed only ranks edges, so
   any symmetric squash of the logits gives the same segmentation; the cheap `cc_threshold` route
   does depend on the squash and assumes ours, `sigmoid(0.2 * logit)`.) If your long-range offsets
   are not exactly 10 voxels, or you have only the three short-range channels, tell us before
   scoring: the routes would silently misread your channels.

3. **Array layout `(6, X, Y, Z)`** in the store's own axis order (x, y, z). If your arrays are
   `(6, Z, Y, X)`, transpose the spatial axes AND permute the channels to match (channel 0 must be
   the +x offset, and so on).

4. Read access to the store above (group `miaai`), a GitHub account in the `AI-HHMI` organisation,
   and a directory of your own for the outputs (a few tens of GB) that the `miaai` group can read.

## 1. Install `mia-evals` (once, about 10 minutes)

The cluster's system Python is 3.9; `mia-evals` needs 3.11 or newer. The shared `micromamba`
binary gives you one without admin rights:

```bash
/misc/sc/micromamba create -y -p ~/envs/mia-evals -c conda-forge python=3.12 pip
git clone git@github.com:AI-HHMI/mia-evals.git ~/mia-evals
cd ~/mia-evals
~/envs/mia-evals/bin/pip install -e .
~/envs/mia-evals/bin/mia-evals --help        # prints the `score` and `leaderboard` sub-commands
```

Any other Python 3.11+ (conda, uv) works the same way; only the first line changes. Skip
`funlib.evaluate` from the README: it is for skeleton metrics, and this task uses voxel metrics.

## 2. Write the two artifact directories (one short job, about 15 minutes)

`mia-evals` scores a *prediction artifact*: a Zarr array plus a few attributes saying what its
numbers mean. For this task it also needs the ground truth **on the prediction's own grid**, as a
second array beside it, named `<volume>.gt.zarr` (the task's config says
`truth_kind = "instances_resampled"`, which is how all our rows were scored). The script below
writes both for each block. File names are fixed by the task's volume names:

```
<OUT>/test/hemibrain_eb_test.zarr      your affinities, test block
<OUT>/test/hemibrain_eb_test.gt.zarr   ground truth, same box, written by the script
<OUT>/fit/hemibrain_eb_fit.zarr        your affinities, fit block
<OUT>/fit/hemibrain_eb_fit.gt.zarr     ground truth, same box, written by the script
```

Save this as `prepare_artifacts.py`, edit the four `EDIT` lines, and run it as a cluster job
(it holds one block's affinities in memory, so give it 4 slots = 60 GB):

```python
"""Write mia-evals artifacts (affinities + ground truth on the same grid) for both blocks.

    bsub -P miaai -q local -n 4 -W 2:00 -o prepare_%J.log \
        ~/envs/mia-evals/bin/python prepare_artifacts.py
"""
import numpy as np
import zarr
from artifact import write_artifact          # from mia-evals (`pip install -e .` above)

STORE = "/groups/miaai/miaai/lmd-v0.0.1/data/em-drosophila-flyem-hemibrain/crop-001_EllipsoidBody_x24000_y23000_z17000.zarr"
LABEL_KEY = "labels/proofread-cell-hemibrain-v1.2"

RUN = "gary_mymodel_20260921"                                    # EDIT: names your row, <RUN>.mws -- letters, digits, _ and - only, NO dots
OUT = "/groups/miaai/miaai/<your dir>/gary_comparison_eval"    # EDIT: outputs; keep it readable by group miaai
SOURCES = {                                                      # EDIT: your affinity arrays, (6, X, Y, Z) in x,y,z order, values in [0, 1]
    "test": "/path/to/my_test_affinities.zarr",
    "fit": "/path/to/my_fit_affinities.zarr",
}
BOXES = {                                                        # EDIT only if you predicted different boxes ([x0,x1],[y0,y1],[z0,z1], level-0 voxels)
    "test": [[4052, 4948], [4052, 4948], [4052, 4948]],
    "fit": [[4052, 4948], [4052, 4948], [3052, 3948]],
}
ANNOTATED = {"test": [[4000, 5000], [4000, 5000], [4000, 5000]],
             "fit": [[4000, 5000], [4000, 5000], [3000, 4000]]}
NAMES = {"test": "hemibrain_eb_test", "fit": "hemibrain_eb_fit"}   # fixed by the task's data configs

labels = zarr.open(STORE, mode="r")[f"{LABEL_KEY}/s0"]
for split in ("test", "fit"):
    (x0, x1), (y0, y1), (z0, z1) = BOXES[split]
    aff = np.asarray(zarr.open(SOURCES[split], mode="r")[:], dtype=np.float32)
    # If your array is (6, Z, Y, X): aff = aff.transpose(0, 3, 2, 1)[[2, 1, 0, 5, 4, 3]]
    assert aff.shape == (6, x1 - x0, y1 - y0, z1 - z0), f"{split}: got {aff.shape}"
    assert 0.0 <= aff.min() and aff.max() <= 1.0, f"{split}: values must be probabilities in [0, 1]"
    common = dict(origin=(x0, y0, z0), axes="xyz", source_path=STORE, source_label_key=LABEL_KEY,
                  native_box=BOXES[split], annotated_box=ANNOTATED[split],
                  covers_full_box=(BOXES[split] == ANNOTATED[split]))
    write_artifact(f"{OUT}/{split}/{NAMES[split]}.zarr", aff.astype(np.float16), kind="affinity",
                   convention="sigmoid(logit)",                  # EDIT: free text, how you squashed your logits
                   run=RUN, **common)
    gt = np.asarray(labels[x0:x1, y0:y1, z0:z1])                 # uint64 ids, 0 = background
    write_artifact(f"{OUT}/{split}/{NAMES[split]}.gt.zarr", gt, kind="instances", background_id=0, **common)
    print(split, "written:", aff.shape, "instances in truth:", len(np.unique(gt)) - 1, flush=True)
```

Notes on the attributes it writes (`README.md`, "Prediction artifacts", has the full contract):
`kind="affinity"` is what lets the routes accept the array; `origin` is the box corner in
absolute voxel coordinates; `run` names your leaderboard row (`<RUN>.mws`; add `step=<int>` to the
affinity call if you want `<RUN>.step<N>.mws`); `source_path` lets the scorer refuse a file filed
under the wrong volume; `native_box` must be identical on the prediction and its `.gt.zarr`, which
the script guarantees.

Check the result in seconds (no job needed):

```bash
cd ~/mia-evals
~/envs/mia-evals/bin/python -c "
from artifact import open_artifact
for p in ['<OUT>/test/hemibrain_eb_test.zarr', '<OUT>/test/hemibrain_eb_test.gt.zarr',
          '<OUT>/fit/hemibrain_eb_fit.zarr', '<OUT>/fit/hemibrain_eb_fit.gt.zarr']:
    a = open_artifact(p); print(a.kind, a.shape, a.origin, a.attrs.get('run'))"
```

Expected: `affinity (6, 896, 896, 896) (4052, 4052, 4052) gary_mymodel_20260921`, then
`instances (896, 896, 896) ...`, and the same pair for the fit block. Anything mislabelled fails
here rather than five hours into the scoring job.

## 3. Score (one CPU job, about 5 hours)

Run from the repository root so the record lands in the checkout's `leaderboard/` directory.
The mutex-watershed route is the one our rows are ranked by:

```bash
cd ~/mia-evals
bsub -P miaai -q local -n 32 -W 24:00 -J score_mws -o score_mws_%J.log \
  ~/envs/mia-evals/bin/mia-evals score configs/gary_comparison_neuron_instance/mws.toml \
    --test <OUT>/test --val <OUT>/fit \
    --scored-out <OUT>/scored/mws --scratch <OUT>/scratch
```

32 slots because the watershed on 896^3 peaks at about 250 GB (the `local` queue gives 15 GB per
slot); it took 5.1 hours for our rows. The job fits the size filter on the fit block, applies the
winner to the test block, and prints the score. The optional second route, thresholded connected
components on the short-range channels, is the cheap baseline (30 minutes, 8 slots):

```bash
bsub -P miaai -q local -n 8 -W 6:00 -J score_cc -o score_cc_%J.log \
  ~/envs/mia-evals/bin/mia-evals score configs/gary_comparison_neuron_instance/cc_threshold.toml \
    --test <OUT>/test --val <OUT>/fit \
    --scored-out <OUT>/scored/cc_threshold --scratch <OUT>/scratch
```

Watch the log with `tail -f score_mws_<jobid>.log`. Success looks like:

```
fit volumes: ['hemibrain_eb_fit']
fitting mws on fit
chose mws(repulsive_stride=1, min_size=20000)
scoring test
scored labellings kept under <OUT>/scored/mws
    hemibrain_eb_test                pq = 0.1xxx
  voxel_instance.pq = 0.1xxx (unweighted mean over 1 volumes)
record: /path/to/mia-evals/leaderboard/gary_comparison_neuron_instance/records/gary_mymodel_20260921.mws.json
updated: .../leaderboard/gary_comparison_neuron_instance/README.md
```

Panoptic quality (`pq`) ranks the table; VOI split and merge, SQ, RQ and adapted Rand error are
reported beside it. For scale: our DINOv3-initialised model reached pq 0.164 on this box through
the same route after 500k training steps, and 0.153 from random initialisation. The post-processed
labelling that was scored is kept under `--scored-out`, so you can open it in neuroglancer next to
the ground truth.

Two refusals you may meet, both deliberate: `a record named '...' already exists` means that
`RUN` (and route) was scored before, so pick a new `RUN` or delete the old JSON; `... has 5
candidate settings ... Pass --val` means the fit block was not given.

## 4. Put it on the shared leaderboard

The record is a JSON file in your clone. Commit **only the record** on a branch and open a pull
request against `AI-HHMI/mia-evals`, or send us the file; we re-render the table (it also gets
neuroglancer links, which need a fileglancer key that only exists on our side):

```bash
cd ~/mia-evals
git checkout -b gary-mws-record
git add leaderboard/gary_comparison_neuron_instance/records/gary_mymodel_20260921.mws.json
git commit -m "gary_comparison_neuron_instance: add gary_mymodel_20260921.mws"
git push -u origin gary-mws-record
```

Leave `leaderboard/gary_comparison_neuron_instance/README.md` out of the commit: your render lacks
the view links and would replace ours. Keep `<OUT>` in place and group-readable
(`chmod -R g+rX <OUT>`): the record links to your artifacts and scored labelling by path.

## 5. If you would rather not run it yourself

Do step 2 only, or even just give us the affinity arrays with their boxes and the note on channel
order and squashing, and we run steps 3 and 4. Everything the scorer needs is in those four Zarr
directories.

## Checklist of the ways this goes silently wrong

- Spatial axes not in the store's `x, y, z` order, or channels not permuted with them. The scorer
  cannot detect this; the score is simply low.
- Channel 3 to 5 not 10-voxel offsets, or only three channels. Six channels are required by the
  `affinity` kind, and the routes assume the offsets in the table above.
- Values as logits or on `[0, 255]`. The script's assertion catches this.
- Test box other than `[4052, 4948)^3`: scored correctly, but grouped separately from our rows.
- Fit block predicted with different settings than the test block: the fitted size filter then does
  not fit the test predictions.
- A dot in `RUN`: record names use dots as separators.
- Running the scoring on a login node instead of through `bsub`: it needs 250 GB of memory.
