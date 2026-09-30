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

Both rank on nERL (ERL divided by the perfect ERL) and also report ERL in um, VOI split and merge
over nodes, and merge and split counts.

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

## Data

`/groups/miaai/miaai/mia-evals-data/zebrafinch_j0126/`, built by
`python -m truth.lsd_zebrafinch --out <that dir>`; its `manifest.json` records sources, hashes and
the command. The image, mask and FFN arrays are symlinks into the LSD release on the Janelia
cluster; the public copies are listed in the manifest.

## Submitting

- Predict on `zebrafinch_j0126.zarr/raw` (20x9x9 nm) in the store's own axis order, z, y, x
  (`output_axes: lczyx` in the data config), so that origins, boxes and skeleton nodes agree; the
  scorer compares a skeleton's `axes` with an OME artifact's before any lookup.
- Cover the whole region: a prediction made over a larger box is cropped by the scorer.
- A finished segmentation (`identity` route) must already be masked and relabelled as in step 2 --
  FFN's reference artifacts are -- because at 478 GVox the scorer does not do it.

## Known overlap

29 of the 33 dense training cubes the benchmark provides lie inside the benchmark region; 38.9 um
of the 91 mm of test cable passes through them (0.04%). This comes with the published benchmark and
is reported, not removed.
