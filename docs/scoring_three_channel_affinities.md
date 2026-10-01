# Scoring three-channel affinities on `gary_comparison_neuron_instance`

For a model that predicts only the three **nearest-neighbour** affinities (one voxel along each
axis), which the main routes cannot score: `mws` needs three more, long-range channels for its
repulsive edges. Lives on the branch **`three-channel-mws`** only; it is not on `main`.

Everything in [scoring_third_party_affinities.md](scoring_third_party_affinities.md) applies; this
page lists only what differs. Read that one first.

## 0. Get the branch

```bash
git clone git@github.com:AI-HHMI/mia-evals.git ~/mia-evals     # if you do not have it yet
cd ~/mia-evals
git fetch origin
git checkout three-channel-mws
~/envs/mia-evals/bin/pip install -e .                            # re-run after switching branch
```

## 1. Your affinities: three channels, `(3, X, Y, Z)`

| channel | offset (voxels, along the store's axes) | meaning |
| ---: | --- | --- |
| 0 | `(+1, 0, 0)` | voxel `p` and `p + x` are the same object |
| 1 | `(0, +1, 0)` | voxel `p` and `p + y` are the same object |
| 2 | `(0, 0, +1)` | voxel `p` and `p + z` are the same object |

Channel `c` at voxel `p` is the affinity between `p` and `p + offset_c`. **1 means "same object",
0 means "boundary".** Values are probabilities in `[0, 1]` (apply your sigmoid). The spatial axes
are the store's `x, y, z`.

Two conversions are common:

```python
# Array stored (3, Z, Y, X), channels in z, y, x order: transpose AND permute the channels.
aff = aff.transpose(0, 3, 2, 1)[[2, 1, 0]]

# gunpowder / LSD / MALA networks: channel i at p is the edge to p - e_i (neighbourhood
# [[-1,0,0],[0,-1,0],[0,0,-1]]). Shift each channel one voxel along its own axis so it is the edge
# to p + e_i. Without this every boundary sits one voxel off and nothing raises.
for i in range(3):
    aff[i] = np.roll(aff[i], -1, axis=i)
```

The value `np.roll` wraps into the last plane is never read: there is no edge out of the volume.

## 2. Write the artifacts

Use the main guide's `prepare_artifacts.py` with three changes:

```python
    assert aff.shape == (3, x1 - x0, y1 - y0, z1 - z0), f"{split}: got {aff.shape}"   # 3, not 6
    ...
    write_artifact(f"{OUT}/{split}/{NAMES[split]}.zarr", aff.astype(np.float16), kind="affinity",
                   convention="sigmoid(logit)",
                   offsets=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],    # NEW: declares the channel table
                   run=RUN, **common)
```

plus the conversion lines above, if they apply, before the assertion. `offsets` is checked by both
routes: anything other than the table above is refused before any work, and the backward
(gunpowder) neighbourhood gets a message saying how to convert it. The check in the main guide
then prints `affinity (3, 896, 896, 896) ...`.

## 3. Score: two routes

Both are fitted on the fit block and applied once to the test block, like `mws`. Neither needs a
threshold you choose; every candidate is printed in the log with its fit-block pq.

| route | what it does | fitted | job |
| --- | --- | --- | --- |
| `ws_agglo` | **LSD's decoder** (Sheridan et al. 2023, after MALA): seeded-watershed fragments, then hierarchical agglomeration by the 50% or 75% quantile of the face affinities, cut at a threshold | merge function, threshold, size filter | 8 slots, about 2 hours |
| `mws3` | mutex watershed on the nearest-neighbour edges, each edge attractive above 0.5 and repulsive below | size filter, fill | 16 slots, about 4 hours |

**Which one fits your model depends on how it was trained:**

- **On eroded labels** (gunpowder's `GrowBoundary`, as LSD and MALA train), your network predicts
  membranes a couple of voxels wide, low in all three channels. `ws_agglo` is the decoder it was
  tuned for, and `mws3` works too.
- **On un-eroded labels**, touching neurons meet in a one-voxel plane and a cut shows in one
  channel only. `ws_agglo` cannot separate those: its "inside" mask is the mean of the three
  channels above 0.5, which a single-channel cut leaves at 2/3. Use `mws3`.

Measured on a 256^3 crop of the fit block, from the ground truth's own affinities plus noise (pq at
the best size filter):

| affinities | `mws3` | `ws_agglo` |
| --- | ---: | ---: |
| un-eroded labels | 0.87 | 0.04 |
| labels eroded by one voxel | 0.61 | 0.56 |

Both need cuts below 0.5: blurred until 17% of the true cuts rose above it, both merged everything.

Each route lands as its own row, `<RUN>.ws_agglo` and `<RUN>.mws3`, in the same table as our `mws`
rows. Pick the route from how your model was trained, before looking at test numbers, or report
both -- choosing it by its test pq would be selecting on the number being reported.

```bash
cd ~/mia-evals
bsub -P miaai -q local -n 8 -W 12:00 -J score_ws_agglo -o score_ws_agglo_%J.log \
  ~/envs/mia-evals/bin/mia-evals score configs/gary_comparison_neuron_instance/ws_agglo.toml \
    --test <OUT>/test --val <OUT>/fit \
    --scored-out <OUT>/scored/ws_agglo --scratch <OUT>/scratch
bsub -P miaai -q local -n 16 -W 12:00 -J score_mws3 -o score_mws3_%J.log \
  ~/envs/mia-evals/bin/mia-evals score configs/gary_comparison_neuron_instance/mws3.toml \
    --test <OUT>/test --val <OUT>/fit \
    --scored-out <OUT>/scored/mws3 --scratch <OUT>/scratch
```

Peak memory measured on synthetic three-channel affinities of these blocks: 50 GB for
`ws_agglo`, about 120 GB for `mws3` (the `local` queue gives 15 GB per slot). The `ws_agglo` log
also prints, per block, how many fragments the watershed made and what fraction of voxels had a
mean affinity above 0.5. Nearly all of this tissue is neuron, so the fraction is high (84% and 89%
on the synthetic eroded-label maps); near 0% or 100% means the values are not probabilities with
1 = same object.

Step 4 of the main guide (the pull request with your record) is unchanged.

## What each route does, briefly

`ws_agglo` (`src/postprocess/ws_agglo.py`): the three channels are averaged; voxels above 0.5 are
inside an object; seeds are maxima of the distance to the nearest outside voxel (10-voxel window),
plus one in every inside piece that has none (LSD's seeding leaves small pieces beside big ones
without, and the flood then joins them to the neighbour across the membrane); a seeded watershed
floods the inverted distance. Neighbouring fragments are merged highest score
first, the score of a pair being the 50% or 75% quantile of the affinities on the faces their
inside voxels share; merging pools the faces, so scores stay exact. The labelling at threshold `t`
is every merge made before the first one scoring below `t`, and outside voxels join their nearest
inside voxel's segment. Thresholds are in affinity units (waterz's are `1 - t`).

`mws3` (`src/postprocess/mws3.py`): one edge per voxel per axis; an affinity above 0.5 is an
attractive edge of priority `a`, one at or below 0.5 a repulsive edge of priority `1 - a`; the
mutex watershed takes them most confident first, merging along attractive edges unless an earlier
repulsive edge forbids it. Never coarser than thresholding at 0.5, and a weak leak through an
otherwise clear membrane does not merge the two sides.

## Checklist of the ways this goes silently wrong

Everything in the main guide's checklist, plus:

- Backward (gunpowder) affinities written as forward ones without the `np.roll` above: every cut
  one voxel off. Declaring `offsets` makes this impossible to miss; leaving it out means the
  channels are read as the table says.
- Channels in z, y, x order over an x, y, z array: read as the wrong edges.
- A six-channel array is fine too: both routes read only its first three channels.
