# Neurite tracing benchmark

Expected run length (ERL) over skeletons, on the volumes of the LSD paper (Sheridan et al. 2023,
*Local shape descriptors for neuron segmentation*, Nat Methods 20:295), scored by their own
protocol, and on NISB's synthetic cubes. ERL is the expected length of error-free neurite a tracer
would follow from a random point, and a segment that merges two neurites makes all of its cable
wrong, so these tasks reward a segmentation that stays correct over long distances.

| task | region | size | test ground truth | perfect ERL |
|---|---|---|---|---|
| `zebrafinch_neurite_tracing` | LSD's benchmark region of j0126 (zebra finch, SBEM 9x9x20 nm), 87.3 x 83.7 x 106 um | 478 GVox | 50 hand-traced skeletons: 3,410 pieces, 431,659 nodes, 75.95 mm | 240.66 um |
| `zebrafinch_neurite_tracing_11um` | the 10.8 um cube at its centre | 0.78 GVox | the same skeletons inside it: 67 pieces, 3,488 nodes, 0.52 mm | 17.91 um |
| `hemibrain_eb_neurite_tracing` | LSD's three ellipsoid-body cubes of the hemibrain (FIB-SEM 8 nm): 12, 22 and 17 um | 3.2 / 20.2 / 9.3 GVox | skeletons derived from the whitelisted proofread neurons: 364 / 921 / 734 objects, 35,142 / 158,338 / 95,183 nodes, 4.9 / 22.3 / 13.2 mm | 44.6 / 117.9 / 74.0 um |
| `nisb_base_neurite_tracing` | NISB's base test cube, seed101 (synthetic, 9x9x20 nm), 27 x 27 x 27 um | 12.2 GVox | the generator's own skeleton: 791,035 nodes | 256.57 um |

All four rank on nERL (ERL divided by the perfect ERL; for the hemibrain task, the unweighted mean
over its three regions) and also report ERL in um, VOI split and merge over nodes, merge and split
counts, and nerl_5 / 20 / 100 / inf, which ignore merges where a neuron has at most that many nodes
in the segment.

## Protocol

The LSD authors' (Supplementary Note D.4, and their `05_evaluate_annotations_zfinch_masked.py`):

1. The skeletons are cut to the region and to the FFN neuropil mask (no cell bodies, myelin, blood
   vessels or background), and their pieces relabelled as connected components. Each piece is one
   unit of run length.
2. The labelling is restricted to the same mask and its segments relabelled into connected
   components inside the region; otherwise a correct segment joining two pieces through a masked
   cell body would count as a merge.
3. Each node is looked up in the labelling; funlib's `expected_run_length` runs over the pieces, a
   segment touching two pieces making all its edges wrong; nERL divides by the ERL of the perfect
   lookup; VOI is `rand_voi` over nodes.
4. Post-processing parameters are chosen on the 12 validation skeletons in the same region: shared
   voxels, different neurons, which is how both LSD and FFN chose theirs. Every route so far has a
   single candidate, so nothing is fitted yet.

The ground truth is LSD's own per-region node sets from their release, not a re-derivation from the
traced NML files. `python -m truth.lsd_zebrafinch` asserts the properties the metric relies on:
whole-voxel positions, pieces equal to the connected components of the kept graph, no piece
numbered 0.

## Published numbers (LSD Supplementary Table 2, benchmark region, test skeletons)

Threshold chosen on the validation skeletons by best VOI sum (A) or best ERL (B); FFN has one
segmentation, so its row is the same in both.

| method | A: VOI split | VOI merge | VOI sum | ERL (um) | nERL | B: VOI sum | ERL (um) | nERL |
|---|---|---|---|---|---|---|---|---|
| Baseline | 1.115 | 2.741 | 3.856 | 9.147 | 0.038 | 5.343 | 9.113 | 0.038 |
| LR | 2.072 | 2.286 | 4.358 | 9.517 | 0.040 | 5.756 | 9.044 | 0.038 |
| FFN | 1.068 | 1.188 | 2.256 | 16.747 | 0.070 | 2.256 | 16.747 | 0.070 |
| MtLsd | 0.625 | 2.794 | 3.420 | 8.855 | 0.037 | 3.247 | 11.352 | 0.047 |
| AcLsd | 1.192 | 1.222 | 2.414 | 12.886 | 0.054 | 2.554 | 11.419 | 0.047 |
| AcRLsd | 0.944 | 1.346 | 2.290 | 12.667 | 0.053 | 2.239 | 13.470 | 0.056 |

FFN is re-scored here from its released segmentation (`ffn_januszewski2018.identity` rows); the LSD
methods are not, since they ship as fragments and region graphs of 121-392 GB each.

## Hemibrain

The release has voxel ground truth only -- the whitelisted proofread neurons, restricted to the
ellipsoid body, relabelled into connected components and eroded (`consolidated_ids`) -- and the
paper reports voxel VOI only. The erosion removes every voxel within one voxel, in 3D, of a label
boundary: 2 voxels (16 nm) come off every surface, and touching neurons end up 4 voxels apart
(measured on roi_1 against LSD's own pre-erosion labels; 99.97% of voxels follow that rule). The
skeletons are derived here
(`python -m truth.lsd_hemibrain`): kimimaro's TEASAR on every object of at least 1,000 voxels, in
1024^3 blocks, each branch end cut back inside its object (below), thinned to ~150 nm between
nodes, with every node inside its own object. The unit of run length is the object, not a
connected piece of its skeleton: the erosion leaves many objects in pieces (425 pieces for roi_1's
364 objects, 1,572 and 1,173 for roi_2 and roi_3), and a segmentation that keeps such a neuron
whole must not be scored as merging its pieces. There is no validation set -- the paper chose its
thresholds on test -- so the task has no fit split.

**Branch ends lie more than 2 voxels (16 nm) inside their object.** TEASAR runs every branch out
to its object's surface, and those surfaces are FFN's moved inward by the erosion, because the
ground truth is proofread FFN. The erosion trims less at a branch's tip than on a flat face, so
the ends sit only about 2 voxels inside FFN's boundary. Where a segmentation's boundary is that
far off FFN's, an end on the surface lands in the neighbouring segment, and run length scores that
as a merge that voids the neighbour's whole run: a penalty FFN, whose boundaries these are, never
pays. So each end walks back along its own
skeleton path to the first voxel that deep, by at most one node spacing and never past a branch
point; a neurite too thin to hold such a voxel keeps its end at its most interior one. On roi_1
this removes 1.9% of the cable. Measured there against gary_comparison 5a's segmentation (a
pipeline check, not a row: 5a was trained on voxels that include roi_1):

| roi_1 | skeleton | nERL | nerl_5 | segments touching a second neuron by <= 5 nodes | other merging segments |
|---|---|---|---|---|---|
| 5a | ends on the surface | 0.417 | 0.939 | 64 | 1 |
| 5a | ends inset | 0.828 | 0.948 | 8 | 1 |
| FFN | ends on the surface | 0.879 | 0.879 | 0 | 3 |
| FFN | ends inset | 0.879 | 0.879 | 0 | 3 |

FFN's score does not move, and both keep every merge that reaches more than 5 nodes of a second
neuron. The alternatives were measured on the same pair: dropping end nodes removed 8.5-10% of the
cable, and pruning short terminal branches left most of the contacts.

Skeletonising in blocks traces a neurite lying in a shared block plane twice. Measured before the
inset, roi_1 had 5.02 mm of cable against 4.77 mm from one unblocked run. That barely reaches the
ranking number: FFN's roi_1 nERL was 0.8792 on the production skeletons and 0.8809 on the
unblocked ones, with identical merge and split counts.

FFN's reference is the release's `FFN/roi_k/consolidated_ids` (cropped, restricted to the
ellipsoid body, relabelled). Its voxel VOI against the ground truth is within about 1% of
Supplementary Table 3:

| region | ours, VOI split / merge | Table 3 |
|---|---|---|
| roi_1 (12 um) | 0.1297 / 0.0461 | 0.129 / 0.046 |
| roi_3 (17 um) | 0.3434 / 0.0245 | 0.347 / 0.024 |
| roi_2 (22 um) | 0.2410 / 0.0366 | 0.242 / 0.036 |

**FFN is not an independent baseline on these regions.** The hemibrain's proofread neurons were
made by proofreading an FFN segmentation (Scheffer et al. 2020), so the ground truth inherits its
boundaries, and every error left in FFN is one the proofreaders fixed. Its nERL -- 0.879 / 0.848 /
0.749 for roi_1 / roi_2 / roi_3 -- is a reference point, not a bar a model clears on equal terms.

## NISB (base)

NISB, the Neuron Instance Segmentation Benchmark, is synthetic: one fixed procedure generates every
cube, at 9 x 9 x 20 nm and 3000 x 3000 x 1350 voxels, from
`/groups/miaai/miaai/lmd-v0.0.1/dev/nisb/base`. The ground truth is each cube's own
`skeleton.pkl`, written by the generator, so no segmenter's boundaries are built into it. The task
scores whole cubes, as the benchmark's own evaluation does: the val cube (seed100; 784,783 nodes,
perfect ERL 259.24 um) is the fit split and the test cube (seed101) the reported one, which the
benchmark's rules score once. Each cube's own labels score nERL 1 and VOI 0 on it (checked
2026-10-02). A prediction has to cover the whole cube, which there is no context beyond: predict
with mia-train's `predict.py --cover-box`. A table's rows must all score the same region, so a
first row predicted without it would lock the table to a smaller one. On NISB, `mws_blockwise` does
not reproduce exact mutex watershed at either block size tried (the gate under "Segmenting affinities
at this scale"), so its NISB numbers compare models scored with the same blocks rather than
measure `mws`; `cc_threshold`, the benchmark's own post-processing, has no blocks.

## Data

`/groups/miaai/miaai/mia-evals-data/zebrafinch_j0126/` and `.../hemibrain_eb_lsd/`, built by
`python -m truth.lsd_zebrafinch` and `python -m truth.lsd_hemibrain` (`--out <that dir>`); each
`manifest.json` records sources, hashes and the command. The image, mask, label and FFN arrays
are symlinks into the LSD release on the Janelia cluster; the public copies are listed in the
manifests.

## Submitting

- Predict on the store's `raw` (zebrafinch 20x9x9 nm; hemibrain 8 nm, the release's own contrast)
  in the store's own axis order, z, y, x (`output_axes: lczyx` in the data configs), so that
  origins, boxes and skeleton nodes agree; the scorer compares a skeleton's `axes` with an OME
  artifact's before any lookup.
- Predict over the region plus a margin of at least 3/4 of a patch on every side. mia-train's
  `predict.py` centres the largest whole-tile lattice in the box it is given and leaves the faces
  out, unless `--cover-box` asks it to cover the box exactly (at the store's own resolution), and
  voxels near the edge of what it covers see little context. The scorer crops to the region, and
  counts a prediction covering it as the whole region. NISB has no raw beyond its cubes, so there
  the box is the cube itself and every prediction needs `--cover-box`.
- A `predict.py` artifact records its position as `native_box` (its `origin` is always
  `[0, 0, 0]`), and the scorer places it there. That needs the store's own resolution -- `scale` 1,
  which these tasks' data configs ask for: a resampled prediction is refused, since boxes and
  skeleton nodes count the store's voxels. Any other artifact states its first voxel's position in
  the store's level-0 voxels as `origin`.
- Affinities are scored through the `mws_blockwise` route (below), which applies the mask itself.
  A finished segmentation (`identity` route) must already be masked and relabelled as in step 2 --
  to the neuropil mask for zebrafinch, the ellipsoid body for hemibrain, as FFN's reference
  artifacts are -- because the scorer does not do it.

## Segmenting affinities at this scale: `mws_blockwise`

Exact mutex watershed (`mws`, streamed through disk above 8 G edges) holds every voxel of the region
in one union-find, so it ends at the largest node's memory: the 7-gigavoxel zebrafish doublecube
needed 1.7 TB. The benchmark region is 478 gigavoxels. The `mws_blockwise` route
(`src/postprocess/mws_blockwise.py`) is mutex watershed with one change of order: edges inside a
block are decided before edges crossing a block face.

- Every block runs the same compiled kernel, independently.
- One more pass of the kernel stitches the clusters that touch faces, carrying over the mutexes
  decided inside the blocks. Between two clusters only the first crossing edge can act, so the
  stitch takes one edge per pair and is exact for that order.
- The unit tests require the result to equal the reference algorithm run in that order, and to
  equal `mws` itself when one block covers the region.

How far the change of order moves the result was measured against exact `mws` on the same
affinities (gary_comparison 1a at step 500k on its 896^3 hemibrain test block, scored by that task's
voxel metric at its fitted `min_size` of 20,000):

| segmenter | pq | VOI split | VOI merge | segments, unfiltered | VOI to exact, unfiltered |
|---|---|---|---|---|---|
| `mws` | 0.1636 | 0.954 | 0.421 | 244,284 | -- |
| `mws_blockwise`, 512^3 blocks (8) | 0.1630 | 0.964 | 0.417 | 242,281 | 0.018 / 0.029 |
| `mws_blockwise`, 256^3 blocks (64) | 0.1650 | 0.970 | 0.415 | 237,291 | 0.038 / 0.055 |

Both are inside the pre-registered bounds (pq within 2% relative, VOI sum within 0.02). Deciding
crossing edges last blocks a few merges across faces: VOI split rises by 0.010-0.015 and VOI merge
falls by 0.004-0.006. Larger blocks move it less. 97-99% of voxels lie in segments that match
exact `mws` one to one.

**On NISB the change of order is not small.** Measured the same way on 2026-10-06
(`/nrs/scicompsoft/orhane/mia-train-experiments/large_inputs_nisb/probes/mws_exactness/gate.py`):
large_inputs_nisb's w512_1cube at step 100k, on the val cube (seed100) over x, y in [0, 2048) and
all of z, which is 5.66 G voxels and 373,445 skeleton nodes, the largest region exact `mws` holds on a
1.9 TB node. Scored by the task's skeleton metric on the skeleton cut to the region, which is
pessimistic but alike for all three:

| segmenter | VOI split | VOI merge | splits | mergers | nERL | nERL, merges of <= 5 nodes ignored | nERL, all merges ignored | VOI to exact | voxels matching exact one to one |
|---|---|---|---|---|---|---|---|---|---|
| `mws` | 1.320 | 0.101 | 11,531 | 2,083 | 0.016 | 0.503 | 0.715 | -- | -- |
| `mws_blockwise`, 1024 x 1024 x 1350 (4) | 1.436 | 0.105 | 11,931 | 2,117 | 0.024 | 0.461 | 0.679 | 0.068 / 0.019 | 97.4% |
| `mws_blockwise`, 512 x 512 x 256 (96) | 1.826 | 0.110 | 13,725 | 2,186 | 0.019 | 0.375 | 0.565 | 0.292 / 0.075 | 90.2% |

Neither is inside the bound (skeleton VOI sum within 0.02 of exact's): +0.119 and +0.514. All three
labellings have the same number of segments (2.96 M); blockwise splits neurites at block faces. Larger
blocks move it less, but even the 1024 x 1024 x 1350 blocks, with three faces per cube, cost 0.04 of
the merge-tolerant nERLs. So on NISB `mws_blockwise`'s absolute numbers are biased low at both block
sizes; models scored with the same blocks still compare (both large_inputs_nisb 1-cube arms moved
alike between the two sizes). The near-zero strict nERL is not the blocks: exact `mws` scores 0.016 too.
`mws` assigns every voxel, so a skeleton node near a membrane lands in the neighbour's segment and
merges it, where thresholded components leave it in background, whose edges the metric drops.
Exact `mws` on this region peaked at 1.05 TB (185 bytes a voxel; 1.76 mutex pair insertions a voxel)
and took 5.3 h, so a whole NISB cube, 12.15 G voxels, would need about 2.25 TB.

- **Masking.** `[postprocess] mask = "labels/neuropil_mask"` (zebrafinch) or `"labels/eb_mask"`
  (hemibrain), a key in the volume's store, drops every edge touching a masked voxel. Masked voxels
  are background, and every segment is connected inside the region and mask, so the result is
  the protocol's masked, relabelled labelling, with nothing joined through the margin around the
  region.
- **Running it.** The scorer builds the labelling under `<--scratch>/mws_blockwise/` when it first
  needs it, with a pool of `processes`. For the benchmark region, run the per-block watershed --
  nearly all of the work -- on an LSF array first:
  `python -m postprocess.mws_blockwise <config> --test <affinities> --scratch <scratch>
  --worker $((LSB_JOBINDEX - 1)) --workers <N>`. The scorer, given the same `--scratch`, then only
  stitches and relabels. A process holds one block, up to ~300 bytes a voxel (20 GB at the default
  256 x 512 x 512). On the gate a 256^3 block took 13 s, about 0.8 us a voxel: on the order of 100
  CPU-hours for the benchmark region, with low confidence, since zebrafinch's mutex density is
  unmeasured.

## Known overlap

29 of the 33 dense training cubes the benchmark provides lie inside the benchmark region; 38.9 um
of the 91 mm of test cable passes through them (0.04%). This comes with the published benchmark and
is reported, not removed.

The three hemibrain regions lie inside lmd's ellipsoid-body crop: roi_1 entirely, and about 65% of
roi_2 and 20% of roi_3, inside gary_comparison's training blocks. A model trained with that crop's
labels has seen this ground truth; one pretrained on its raw alone has seen the images.
