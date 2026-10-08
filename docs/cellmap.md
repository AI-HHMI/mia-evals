# CellMap organelle semantic segmentation

Two tasks score 48-class organelle segmentation in FIB-SEM volumes, on the densely annotated crops of
the [CellMap Segmentation Challenge](https://cellmapchallenge.janelia.org/) training release
(collection [10.25378/janelia.c.7456966](https://doi.org/10.25378/janelia.c.7456966), CC-BY-4.0;
Heinrich et al. 2021, *Nature*). They share their classes, metric and scoring rule, and differ in how
many crops a model may be fine-tuned on:

| task | fit (fine-tuning) crops | test crops |
| --- | --- | --- |
| `cellmap_organelle_semantic` | 228 (every crop not in test, partial ones included) | 43, from 16 datasets |
| `cellmap_organelle_semantic_fewshot` | 17, one complete crop per dataset | 193, from 16 datasets |

The splits nest: the fewshot task's fit crops are among the large task's, and the large task's test
crops among the fewshot task's. The two tasks still rank on different test sets, so their numbers do
not compare, and a model fine-tuned on the large task's fit crops has seen fewshot test crops: each
task needs its own fine-tuned model.

## Data

Every volume is a crop of an lmd-v0.0.1 store, which holds the challenge's 22 datasets (raw EM plus
CellMap's annotations):

```
/groups/miaai/miaai/lmd-v0.0.1/data/<em-...-CellMap-jrc-...>/crop-001_fullvol_tissuecrop.zarr
    raw/                                       the EM, s0 at the dataset's native voxel size
    labels/manual_gt-all_organelles-crop<N>    one crop: uint8, one leaf-class id per voxel
    classes.csv                                CellMap's 74 ids, leaves and composites
```

A crop's label array is CellMap's combined `all` array. It is placed in the store by its own OME
translation and is usually painted at half the raw's voxel size (4 nm labels on 8 nm raw, 2 nm on
4 nm). Checked on 2026-10-08:

- **The labels are exact copies.** lmd's 287 crops match `/nrs/cellmap/data` in shape, voxel size and
  per-id voxel counts.
- **`all` is enough.** Deriving every per-class array from `all` and `classes.csv` reproduces
  CellMap's own per-class arrays on 23,557 of 23,575 (crop, class) pairs. The 18 misses are `cell`
  arrays CellMap left empty.
- **The raw is the same image.** lmd's raw is the source raw shifted by one whole-voxel offset per
  dataset. `jrc_sum159-4` and `jrc_ctl-id8-1` are contrast-inverted relative to CellMap's
  `fibsem-uint8` arrays, which matters only to a model trained on the other copy.
- **Two crops are missing from lmd:** `jrc_cos7-1a` 247 and `jrc_hela-3` 102, both partial. Neither
  task uses them.

**The challenge's own test crops are not used.** Their labels are held out by the challenge. So both
tasks' test crops come from its public training crops, and any model trained on that release (for
example a challenge entry) has seen them and cannot be scored here.

## Classes

The 48 classes the challenge scores (its `tested_classes.csv`; its site says 47) are 31 leaves and
17 composites. Each composite is the union of its leaves:

| leaves (CellMap id) |
| --- |
| `ecs` 1 extracellular space, `pm` 2 plasma membrane, `cyto` 35 cytosol |
| `mito_mem` 3, `mito_lum` 4, `mito_ribo` 5 (mitochondrial ribosomes) |
| `golgi_mem` 6, `golgi_lum` 7; `ves_mem` 8, `ves_lum` 9 (vesicles); `endo_mem` 10, `endo_lum` 11 (endosomes) |
| `lyso_mem` 12, `lyso_lum` 13; `ld_mem` 14, `ld_lum` 15 (lipid droplets); `perox_mem` 47, `perox_lum` 48 |
| `er_mem` 16, `er_lum` 17; `eres_mem` 18, `eres_lum` 19 (ER exit sites) |
| `ne_mem` 20, `ne_lum` 21 (nuclear envelope); `np_out` 22, `np_in` 23 (nuclear pores) |
| `hchrom` 24, `echrom` 26 (hetero-, euchromatin), `nucpl` 28 (nucleoplasm); `mt_out` 30, `mt_in` 36 (microtubules) |

| composite | its leaves |
| --- | --- |
| `mito` | mito_mem, mito_lum, mito_ribo |
| `golgi`, `ves`, `endo`, `lyso`, `ld`, `eres`, `perox`, `mt` | their `_mem` and `_lum` (`mt`: `_out`, `_in`) |
| `er` | er_mem, er_lum, eres_mem, eres_lum, ne_mem, ne_lum, np_out, np_in |
| `ne` | ne_mem, ne_lum, np_out, np_in |
| `np` | np_out, np_in |
| `chrom` | hchrom, echrom, and the untested nhchrom, nechrom |
| `nuc` | the nuclear envelope, pores, chromatin, nucleoplasm and nucleolus |
| `er_mem_all` | er_mem, eres_mem, ne_mem |
| `ne_mem_all` | ne_mem, np_out, np_in |
| `cell` | every intracellular leaf |

The scoring configs list each class as every id a voxel of it may carry: its own id, its leaves'
ids, and those of composites made only of its leaves. So `mito = [3, 4, 5, 50]`, where 50 is the id a
crop uses where only the whole organelle was painted. A voxel whose truth is a composite id counts for
the composite and against each of its leaves, which is how CellMap's per-class arrays count it.

Eight untested leaves also occur in the crops, each in at most ten: `bm` (basement membrane),
`cent_dapp` and `cent_sdapp` (centriole appendages), `nhchrom` and `nechrom` (two further chromatin
classes), `nucleo` (nucleolus), `vim` (vimentin) and `tbar` (synaptic T-bars). They are scored only
through the composites that contain them. The other untested classes never occur (ribosomes,
glycogen, actin, centrioles, insulin granules, plant organelles).

## Splits

`python -m truth.cellmap --configs configs` regenerates every data and scoring config of both tasks
from the stores. The split is deterministic given its seed (0), and the builder states its rules.

- **Complete crops only in test.** 224 of the 287 crops annotate all 48 classes. In the others, a 0
  means "not this class" for the few classes painted and nothing for the rest, so they are fit-only.
- **Overlapping crops stay together.** Crops whose boxes intersect form one group, and a group is
  never split between fit and test.
- **Re-annotated regions are set aside.** Six regions were painted two to four times by different
  annotators: `jrc_fly-vnc-1` 173/185, `jrc_hela-3` 101/181, `jrc_jurkat-1` 126/180/182, and
  `jrc_mus-liver` 157/183, 171/416, 172/417.
  - Voxel by voxel, 6.9-15.5% of their voxels differ, and no shift of up to 2 voxels reduces that.
  - Some differences are class decisions: one annotator's lysosomes are another's endosomes, and ER
    exit sites or mitochondrial ribosomes appear in only one annotation.
  - Picking one annotation as the truth would decide the score, so all 13 crops are used in neither
    task; they measure annotator agreement instead (below).
  - `jrc_cos7-1b` 240/291 is one region painted twice with 0.4% of voxels changed, so 240 is kept and
    291 dropped.
- **Large task:** per dataset, 20% of the complete crops (rounded, in whole groups) are test.
  - Every class must be present in at least 3 test crops and 3 fit crops.
  - Of 2,000 seeded draws meeting that, the one whose per-class test share is closest to 20% wins.
  - Fit is every other crop, partial ones included. A partial crop in a test group is dropped.
- **Fewshot task:** fit is one complete crop per dataset from the large task's fit. Crops are chosen
  greedily to cover new classes, preferring crops whose group holds no other complete crop. Together
  the 17 cover all 48 classes. Test is every other eligible complete crop outside the fit crops'
  groups.

| dataset | raw voxel (nm) | crops | complete | set aside | test | fit | fewshot fit | fewshot test |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| jrc_cos7-1a | 2 x 2 x 2 | 11 | 10 | 0 | 2 | 9 | 1 | 9 |
| jrc_cos7-1b | 2 x 2 x 2 | 11 | 10 | 1 | 2 | 8 | 1 | 8 |
| jrc_ctl-id8-1 | 3.48 x 4 x 4 | 5 | 5 | 0 | 1 | 4 | 1 | 4 |
| jrc_fly-mb-1a | 4 x 4 x 4 | 6 | 6 | 0 | 1 | 5 | 1 | 5 |
| jrc_fly-vnc-1 | 4 x 4 x 4 | 6 | 6 | 2 | 1 | 3 | 1 | 3 |
| jrc_hela-2 | 5.24 x 4 x 4 | 26 | 23 | 0 | 5 | 21 | 1 | 22 |
| jrc_hela-3 | 3.24 x 4 x 4 | 18 | 15 | 2 | 3 | 13 | 1 | 12 |
| jrc_jurkat-1 | 3.44 x 4 x 4 | 20 | 17 | 3 | 3 | 14 | 1 | 13 |
| jrc_macrophage-2 | 3.36 x 4 x 4 | 18 | 15 | 0 | 3 | 15 | 1 | 14 |
| jrc_mus-heart-1 | 8 x 8 x 8 | 2 | 0 | 0 | 0 | 2 | 0 | 0 |
| jrc_mus-kidney | 8 x 8 x 8 | 23 | 17 | 0 | 3 | 20 | 1 | 16 |
| jrc_mus-kidney-3 | 8 x 8 x 8 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| jrc_mus-kidney-glomerulus-2 | 4 x 4 x 4 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| jrc_mus-liver | 8 x 8 x 8 | 24 | 23 | 6 | 3 | 15 | 1 | 16 |
| jrc_mus-liver-3 | 8 x 8 x 8 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| jrc_mus-liver-zon-1 | 8 x 8 x 8 | 42 | 24 | 0 | 5 | 36 | 1 | 23 |
| jrc_mus-liver-zon-2 | 8 x 8 x 8 | 19 | 8 | 0 | 2 | 17 | 1 | 7 |
| jrc_mus-nacc-1 | 4 x 4 x 4 | 1 | 1 | 0 | 0 | 1 | 1 | 0 |
| jrc_sum159-1 | 4.56 x 4 x 4 | 12 | 9 | 0 | 2 | 10 | 1 | 8 |
| jrc_sum159-4 | 8 x 8 x 8 | 18 | 17 | 0 | 3 | 14 | 1 | 16 |
| jrc_ut21-1413-003 | 8 x 8 x 8 | 18 | 18 | 0 | 4 | 14 | 1 | 17 |
| jrc_zf-cardiac-1 | 8 x 8 x 8 | 4 | 0 | 0 | 0 | 4 | 0 | 0 |
| **total** | | **287** | **224** | **14** | **43** | **228** | **17** | **193** |

"set aside" counts the 13 re-annotated crops and the dropped near-copy. The rarest classes in the
large task's test set are Golgi (4 crops), lipid droplets and peroxisomes (5 each), and ER exit sites
and euchromatin (6 each); the builder prints the full per-class table.

## Scoring

- **On the label grid.** `semantic_seg` scores each crop on its label array's own grid. The
  prediction is placed by its OME geometry, and each label voxel takes the class of the prediction
  voxel holding its centre. A model may predict at any resolution; a 2 nm crop is still scored at 2 nm,
  and two rows that predicted at different resolutions score the same voxels.
- **Whole crops.** A row's region is the crop in label voxels. It is whole only when the prediction
  covered every label voxel, and a table holds only rows that scored the same regions.
- **Metric.** One confusion matrix is pooled over all test crops. The ranking is mean IoU over the
  48 classes (all present in both test sets), with mean Dice and per-class IoU also reported. The
  pooled confusion matrix is kept in each record (`details.semantic.confusion`), so another class
  grouping can be scored without re-reading a voxel.
- **Ignored voxels.** Voxels whose truth is 0 are left out (`ignore_truth = [0]`). In complete crops,
  0 occurs in only five crops (`jrc_mus-liver-zon-1` 319, 320; `jrc_ut21-1413-003` 191, 196, 214).
  There it covers 0.5-2.4% of the crop: lipid-droplet regions CellMap marked unknown for `cyto`,
  `ld_mem` and `ld_lum`. CellMap leaves them out of those three classes only; here they are left out
  of all classes.

## Submitting

One artifact per test volume, named `<volume>.zarr` (the volume names in `data/test.yaml`), each a
single-level OME-Zarr group:

- spatial axes `z`, `y`, `x`, in nanometres, in that order, with the group's `multiscales` giving
  the voxel size and the centre of the first voxel (OME's convention, which lmd and CellMap follow);
- covering the whole crop. The crops are small (median 100 voxels per side at 8 nm), so predict over
  the crop with surrounding context and the scorer keeps the crop;
- either `kind = "class_labels"`, uint8 CellMap ids with `background_id = 0`, scored by
  `identity.toml`; or `kind = "class_scores"`, channel *k* scoring CellMap id *k*, scored by
  `argmax.toml`. Labels are about 1% of the size.

```bash
mia-evals score configs/cellmap_organelle_semantic/identity.toml --test <artifacts> --run-dir <run>
```

## Reference numbers

Scored on the real configs with artifacts that hold no model (`controls/` beside this task's other
outputs on `/nrs`, `make_controls.py`). Mean IoU over the 48 classes:

| arm | what it is | `cellmap_organelle_semantic` | `..._fewshot` |
| --- | --- | ---: | ---: |
| `truth` | the label crop itself | 1.0000 | 1.0000 |
| `raw_mode` | the truth block-mode-downsampled to the raw's grid: a perfect model at the raw's resolution | 0.9040 | 0.9115 |
| `8nm_mode` | the same at 8 nm | 0.8527 | 0.8596 |
| `constant_cyto` | every voxel cytosol | 0.0240 | 0.0247 |

- **`truth` is the pipeline check.** It confirms the grid placement end to end on every crop.
- **The resolution arms are ceilings.** Even a perfect prediction loses about a tenth of the score at
  the raw's resolution, mostly on the thin classes: membranes and microtubules fall to about 0.8 IoU.
  It loses about a seventh at 8 nm.
- **`constant_cyto` is the floor.** It scores 0.44 in pixel accuracy, which is why the ranking is
  on IoU.
- **Cost.** Scoring a submission takes about 75 s (43 crops) or 5 min (193), and about 18 GB of
  memory.

**Annotator agreement.** In the six set-aside regions, one annotation was scored against another with
the same metric: mean IoU 0.46-0.68 per pair, and 0.617 pooled over the regions (mean Dice 0.729, 23
classes present). Those six regions come from four datasets, so this is indicative only. If they are
typical, a model that agrees with CellMap's annotators as well as they agree with each other scores
about 0.6.
