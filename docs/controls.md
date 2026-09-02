# Control experiments: what a task scores with no model

A leaderboard number means little without a floor. A metric can be high because a model is good, or
because the post-processor, the fitted hyperparameter and the metric's own definition combine to
reward something that has nothing to do with the prediction. The only way to tell them apart is to
run the identical pipeline on affinities, scores or labellings that carry no model information, and
see what it returns.

This document describes how to build those controls for any task on the leaderboard, and records
the results for the tasks where they have been measured.

## A control replaces the prediction's source, never the prediction

`mia-evals` scores files, so every post-processor needs an artifact to read. There is no
model-free mode. This is most obvious for `mws`, which is *defined* as a single global ordering of
weighted edges — strip the weights and every edge ties, so the tie-break alone decides the
partition — but it is equally true of `cc_threshold`, which has nothing to threshold, and of
`argmax`, which has nothing to compare.

So a control does not remove the prediction. It substitutes a different **source** for the numbers
in the artifact, leaving the post-processor, the hyperparameter sweep, the metric and the scored
region untouched. And the moment that source is an expression over the raw image, that expression
is itself a hand-designed model. The question a control answers is therefore not "model versus no
model" but **"learned versus not learned"**, which is the sharper and more useful comparison.

## The three kinds of arm

| kind | what it is | what it isolates |
| --- | --- | --- |
| **zero-information** | values drawn from a fixed-seed RNG | the floor: what the pipeline scores knowing nothing |
| **distribution-matched** | the model's own artifact, rearranged in space | whether the *specific* predictions matter, or only their statistics |
| **hand-designed** | a closed-form expression over the raw image | what training bought over classical image processing |

A **distribution-matched** arm is the most informative of the three and the easiest to get wrong.
Translating the whole artifact by one vector is the good version: it preserves every value, every
relationship between channels, and the local texture, so the post-processor still produces a
structurally valid result — of the wrong location. Rearranging channels *independently* is a weaker
control, because it destroys inter-channel coherence at the same time and no longer isolates one
variable.

The arms are cheap to write because `segment_streaming` and the post-processors take the artifact
as an argument, so an arm is a substitute reader rather than a fork of any algorithm.

### Instantiated for an `affinity` artifact

| arm | source |
| --- | --- |
| `random` | uniform `[0, 1]`, six channels, fixed seed |
| `rolled` | the model's affinities, every channel translated by the same vector |
| `shuffled` | the model's affinities, each channel translated by a different vector |
| `intensity` | `1 - abs(I(x) - I(x+o))` from the raw image |
| `membrane` | the darkest voxel on the path `x -> x+o`, exploiting that membranes are dark in EM |

For a `class_scores` artifact the same three kinds apply with different expressions: random logits,
a translated score volume, and a thresholded intensity or texture feature. For `instances`, a
control is a labelling built without the model, such as a Voronoi partition of random seeds matched
to the true object-size distribution.

## Adding controls for a new task

Four things keep a control honest.

**Use the task's real config.** Run the arm through the same task `.toml`, the same post-processor
and the same metric as the entry it is a floor for. A control that changes two things at once
measures neither.

**Apply the same fit discipline.** Each arm gets its own sweep fitted on the validation split and
reported on test, exactly as a real entry does. An arm handed the entry's fitted hyperparameter is
not a control on the fit, and the fit is often where the suspicion lies.

**Score the same extent.** `_region_key` groups leaderboard rows by the volumes and shapes scored,
and refuses to rank across groups, because several of these metrics change with extent. A control
run on a subset of the volumes is internally valid but is not a floor for the published number, and
must be reported as its own group.

**Include the model arm.** Re-run the real model through the same harness on the same volumes. It
costs little, it is the in-group reference, and it is the check that the harness reproduces the
library's scoring rather than something adjacent to it.

Controls are deliberately **not** leaderboard records. They are not submissions, they are measured
once and do not move, and provenancing them would cost an artifact and a config per arm.

## Two methodology notes that generalise

**Rank-normalise each channel of a hand-designed feature.** Any statistic taken over a longer path
is systematically different from the same statistic over a shorter one: `min` over a 10-voxel
segment is lower than over 1 voxel purely because there are more samples to minimise over, which
fakes strong repulsion on every long-range edge regardless of the image. Post-processors that depend
on how channels *interleave* in one global ordering are sensitive to this, so it is not a harmless
monotone rescaling. Mapping each channel through its own empirical CDF fixes it and uses no labels.

**Verify image alignment empirically rather than trusting metadata.** A hand-designed arm reads the
raw image, and an arm built on a misaligned image scores at the floor for a reason that has nothing
to do with hand design. Raw data reaches the prediction lattice via the artifact's `native_box`,
`scale` and `image_level`, and `native_ext * scale == pred_shape` should hold exactly. Confirm the
result by requiring a signal that must exist — for EM, mean intensity on ground-truth boundaries
below that in object interiors — to peak at zero shift and decay monotonically as the image is
deliberately displaced. Report it against the spread over many independent slabs, because an
underpowered version of this check has run-to-run noise equal to its effect size. Do not trust the
artifact's `read_shape` attribute; it does not equal the native extent under any consistent ratio.

## Measured results

### `lmd_ssl_v1_neuron_instance`

Six affinity sources through one unchanged pipeline: `mws` at `repulsive_stride = 1`, then a
`size_filter` fitted per arm, then panoptic quality. Fitted on `kasthuri15_ac3` and
`liconn_mouse_dg`, reported on `kasthuri15_ac4` and `liconn_mouse_hippocampus`, which hold 273 and
192 true objects. `seg@0` is the segment count before any size filter.

**Every source without genuine learned predictions scores exactly zero.**

| arm | fit pq | **test pq** | fitted `min_size` | TP of 232 | sq | rq | seg@0 | vs `model` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `model` | 0.4806 | **0.3741** | 50000 | 114 | 0.734 | 0.508 | 44,909 | 1.00x |
| `membrane` | 0.1141 | **0.0356** | 50000 | 8 | 0.655 | 0.060 | 264,954 | 5.90x |
| `intensity` | 0.0000 | **0.0000** | — | 0 | 0.250 | 0.000 | 510,358 | 11.36x |
| `rolled` | 0.0000 | **0.0000** | — | 0 | 0.000 | 0.000 | 46,228 | **1.03x** |
| `shuffled` | 0.0000 | **0.0000** | — | 0 | 0.000 | 0.000 | 27,920 | 0.62x |
| `random` | 0.0000 | **0.0000** | — | 0 | 0.000 | 0.000 | 226,591 | 5.05x |

Per-volume, at each arm's fitted `min_size`:

| arm | `kasthuri15_ac4` | `liconn_mouse_hippocampus` |
| --- | --- | --- |
| `model` | pq 0.4425, 153/273 TP, 87 FP | pq 0.3057, 74/192 TP, 86 FP |
| `membrane` | pq 0.0039, 1/273 TP, 100 FP | pq 0.0672, 14/192 TP, 37 FP |
| `intensity` | pq 0.0000, 1/273 TP, 277,816 FP | pq 0.0000, 0/192 TP, 742,899 FP |
| `rolled` | pq 0.0000, 0/273 TP, 58,473 FP | pq 0.0000, 0/192 TP, 33,984 FP |
| `shuffled` | pq 0.0000, 0/273 TP, 11,970 FP | pq 0.0000, 0/192 TP, 43,871 FP |
| `random` | pq 0.0000, 0/273 TP, 199,277 FP | pq 0.0000, 0/192 TP, 253,905 FP |

The harness reproduces `mia-evals score` exactly on both reported volumes — 0.4425 and 0.3057, to
four decimals — and independently re-fitted `min_size` to the same 50000 the published entry uses.

#### What the results mean

**Mutex watershed contributes nothing on its own.** Its entire value is reading a model's output
better than thresholded components does: 0.2287 against 0.1420 on *identical* affinities, and
0.0000 with none. Concretely it exploits the three long-range channels that a threshold discards,
which is the argument in `src/postprocess/mws.py`, now with a floor under it.

**`rolled` clears panoptic quality of the charge that mattered.** Its field carries the model's
exact per-channel distribution and yields 58,473 and 33,984 segments against the model's 57,542 and
32,276 — within 1.6% and 5.3%, so it is nearly indistinguishable in segment count and size
distribution. Translate it and it scores exactly 0.0000, with zero matched objects. Panoptic quality
at `IoU > 0.5` therefore measures genuine spatial agreement and cannot be gamed by producing
plausibly-sized fragments. That was the live worry, given that the fitted `size_filter(min_size =
50000)` deletes roughly 43% of the objects mutex watershed recovers, and it is ruled out.
`tests/unit/test_metric_guards.py` keeps it that way.

**Do not read this as "the model beats classical methods by 10x."** The two hand-designed arms
differ by the entire distance from floor to baseline — `intensity` at 0.0000 and `membrane` at
0.0356 — so *feature design*, not the absence of learning, explains most of that spread. A better
hand-crafted feature would narrow it. `membrane` is a loose lower bound on classical performance,
not a characterisation of it.

**`membrane` generalises worst of any arm** (0.1141 fit to 0.0356 test, a 3.2x drop, against the
model's 0.4806 to 0.3741). Its `min_size` does not transfer, consistent with a feature whose value
swings with modality: the alignment check measures a 10x difference in membrane contrast between
these two datasets, an interior-minus-boundary intensity gap of 0.097 on kasthuri against 0.009 on
liconn.

**`shuffled` is the weakest arm of the six and should not be over-read.** It breaks two things at
once, and its 27,920 segments — *fewer* than the model's — show why: with each channel translated
separately, the attractive channels' membranes no longer coincide with the repulsive channels', so
repulsive edges keep arriving at pairs already merged and place no constraint. It records 0.071
mutex insertions per voxel against the model's 1.066. `rolled` is the arm to cite; it changes one
thing.

#### Scope and reproduction

Four volumes, not the leaderboard's eight, so these numbers are a floor for this pipeline rather
than for the published eight-volume figure. A random affinity field fragments far harder than the
model's, so its mutex table is larger, and the model run alone cost 10.9 h at 1.70 TB resident on
the zebrafish doublecube. The four here are 105-134 Mvox and take 5-16 min per arm per volume.

```bash
cd /nrs/scicompsoft/orhane/mia-train-scratch/mws_null
for a in model random shuffled rolled intensity membrane; do
    bsub -P miaai -n 12 -q local -o log_$a.txt \
        "$BANISVENV"/bin/python null_arms.py "$a"
done
```

`null_arms.py` holds the arm definitions and the full rationale; `align_raw.py` is the alignment
check. Both are indexed in `/nrs/scicompsoft/orhane/INDEX.md`. Per-arm results are written to
`summary_<arm>.json`, and the cached labellings (`<vol>_<arm>.npy`, ~11 GB total) can be deleted and
regenerated.
