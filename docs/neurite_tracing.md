# Neurite tracing benchmark

Expected run length (ERL) over hand-traced skeletons, on the volumes of the LSD paper (Sheridan et
al. 2023, *Local shape descriptors for neuron segmentation*, Nat Methods 20:295), scored by their
own protocol. ERL is the expected length of error-free neurite a tracer would follow from a random
point, and a segment that merges two neurites makes all of its cable wrong, so these tasks reward a
segmentation that stays correct over long distances.

| task | region | size | test ground truth | perfect ERL |
|---|---|---|---|---|
| `zebrafinch_neurite_tracing` | LSD's benchmark region of j0126 (zebra finch, SBEM 9x9x20 nm), 87.3 x 83.7 x 106 um | 478 GVox | 50 hand-traced skeletons: 3,410 pieces, 431,659 nodes, 75.95 mm | 240.66 um |
| `zebrafinch_neurite_tracing_11um` | the 10.8 um cube at its centre | 0.78 GVox | the same skeletons inside it: 67 pieces, 3,488 nodes, 0.52 mm | 17.91 um |
| `hemibrain_eb_neurite_tracing` | LSD's three ellipsoid-body cubes of the hemibrain (FIB-SEM 8 nm): 12, 22 and 17 um | 3.2 / 20.2 / 9.3 GVox | skeletons derived from the whitelisted proofread neurons: 364 / 921 / 734 objects, 35,616 / 160,436 / 96,585 nodes, 5.0 / 22.7 / 13.4 mm | 45.4 / 120.0 / 75.4 um |

All three rank on nERL (ERL divided by the perfect ERL; for the hemibrain task, the unweighted mean
over its three regions) and also report ERL in um, VOI split and merge over nodes, and merge and
split counts.

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
ellipsoid body, relabelled into connected components and slightly eroded (`consolidated_ids`) --
and the paper reports voxel VOI only. The skeletons are derived here
(`python -m truth.lsd_hemibrain`): kimimaro's TEASAR on every object of at least 1,000 voxels, in
1024^3 blocks, thinned to ~150 nm between nodes, with every node inside its own object. The unit
of run length is the object, not a connected piece of its skeleton: the erosion leaves many objects
in pieces (425 pieces for roi_1's 364 objects, 1,572 and 1,173 for roi_2 and roi_3), and a
segmentation that keeps such a neuron whole must not be scored as merging its pieces. There is no
validation set -- the paper chose its thresholds on test -- so the task has no fit split.

Skeletonising in blocks traces a neurite lying in a shared block plane twice: roi_1 has 5.02 mm of
cable against 4.77 mm from one unblocked run. That barely reaches the ranking number: FFN's roi_1
nERL is 0.8792 on the production skeletons and 0.8809 on the unblocked ones, with identical merge
and split counts.

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
- Cover the whole region: a prediction made over a larger box is cropped by the scorer.
- A finished segmentation (`identity` route) must already be masked and relabelled as in step 2 --
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
