# Scoring your affinity model on `lmd_ssl_v2_neuron_instance_crop512`

How to put an affinity model on the
[`lmd_ssl_v2_neuron_instance_crop512` leaderboard](../leaderboard/lmd_ssl_v2_neuron_instance_crop512/README.md),
scored exactly like our rows. You run your model on 15 prepared images, then submit one CPU job.
Everything runs on the Janelia cluster.

The task has 15 cell and neuron volumes from EM and expansion-microscopy datasets, all at about
8 nm per voxel:

- 8 **fit** volumes, used only to choose a size threshold.
- 7 **test** volumes, whose mean panoptic quality (pq) is your score.

The scorer segments your affinities with a mutex watershed, then deletes segments smaller than
the size threshold chosen on the fit volumes.

## 1. Predict

The inputs are in `/nrs/scicompsoft/orhane/mia-evals/lmd_ssl_v2_neuron_instance_crop512/inputs`:

```
test/<volume>.image.zarr   7 images, scored
fit/<volume>.image.zarr    8 images, used to choose the size threshold
*/<volume>.gt.zarr         ground truth (the script below links it; you never open it)
```

Each image is a 3-D float32 Zarr array with values in [0, 1], up to 768 x 512 x 512 voxels. The
intensities are already scaled the way our models see them; rescale them if your model expects
something else. Predict all 15 images with the same model and settings.

For an image of shape `(D0, D1, D2)`, your model must return an array of shape `(6, D0, D1, D2)`
with values in [0, 1] (apply a sigmoid to logits). Channel `c` at voxel `p` is the probability
that voxel `p` and voxel `p + offset_c` belong to the same object:

| channel | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| offset | (1, 0, 0) | (0, 1, 0) | (0, 0, 1) | (10, 0, 0) | (0, 10, 0) | (0, 0, 10) |

The offsets count array indices. Some images are stored z, y, x and others x, y, z; do not
transpose them, just follow the array's own axes.

Save this script as `predict_crop512.py` and edit the three `EDIT` lines. It needs only `numpy`
and `zarr`, so it runs in your model's own environment.

```python
from pathlib import Path

import numpy as np
import zarr

INPUTS = Path("/nrs/scicompsoft/orhane/mia-evals/lmd_ssl_v2_neuron_instance_crop512/inputs")
OUT = Path("/path/to/my_crop512_predictions")  # EDIT: a new directory of yours
RUN = "mymodel_v1"  # EDIT: your leaderboard row's name: letters, digits, _ and - only


def predict(image):  # EDIT: run your model here
    """(D0, D1, D2) float32 image in [0, 1] -> (6, D0, D1, D2) affinities in [0, 1]."""
    raise NotImplementedError


for split in ("test", "fit"):
    (OUT / split).mkdir(parents=True, exist_ok=True)
    for path in sorted((INPUTS / split).glob("*.image.zarr")):
        name = path.name.removesuffix(".image.zarr")
        image = zarr.open(str(path), mode="r")
        affinities = np.asarray(predict(image[:]), dtype=np.float32)
        assert affinities.shape == (6, *image.shape), f"{name}: shape {affinities.shape}"
        assert 0 <= affinities.min() and affinities.max() <= 1, f"{name}: values outside [0, 1]"
        out = zarr.open(str(OUT / split / f"{name}.zarr"), mode="w", shape=affinities.shape,
                        chunks=(1, 128, 128, 128), dtype="float16")
        out[:] = affinities
        out.attrs.update({**image.attrs, "kind": "affinity", "run": RUN})
        truth = OUT / split / f"{name}.gt.zarr"
        if not truth.is_symlink():
            truth.symlink_to(INPUTS / split / f"{name}.gt.zarr")
        print(split, name, affinities.shape, flush=True)
```

Run it on a GPU node (any GPU queue your model fits on):

```bash
bsub -q gpu_l4 -gpu "num=1" -n 8 -W 4:00 -o predict_%J.log python predict_crop512.py
```

It writes `<OUT>/test/` and `<OUT>/fit/`, holding one `<volume>.zarr` per image plus a link to its
ground truth. Keep the attributes the script copies from each image: the scorer checks them against
the ground truth.

## 2. Score

Install `mia-evals` once, inside an interactive job (`bsub -Is -n 4 -W 1:00 /bin/bash`):

```bash
/misc/sc/micromamba create -y -p ~/envs/mia-evals -c conda-forge python=3.12 pip
git clone https://github.com/AI-HHMI/mia-evals.git ~/mia-evals
~/envs/mia-evals/bin/pip install -e ~/mia-evals
```

Then submit the scoring job, replacing `<OUT>` with your `OUT`:

```bash
cd ~/mia-evals
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
bsub -n 8 -W 4:00 -o score_%J.log ~/envs/mia-evals/bin/mia-evals score \
    configs/lmd_ssl_v2_neuron_instance_crop512/mws_size_filter.toml \
    --test <OUT>/test --val <OUT>/fit --scored-out <OUT>/scored
```

Our scoring runs took 45 to 90 minutes and peaked at 90 GB of memory; the 8 slots give 120 GB.
A successful log ends like this:

```
chose mws(repulsive_stride=1, min_size=20000)
...
  voxel_instance.pq = 0.2xxx (unweighted mean over 7 volumes)
record: .../mia-evals/leaderboard/lmd_ssl_v2_neuron_instance_crop512/records/mymodel_v1.mws_size_filter.json
```

Our rows score pq 0.22 to 0.30. The segmentations that were scored are saved in `<OUT>/scored`,
so you can look at them.

## 3. Submit

Send us the record file named on the `record:` line, or open a pull request that adds only that
file. Scoring also rewrote the leaderboard's `README.md` and `index.html` files in your clone;
leave those out. Keep `<OUT>` in place and readable by others, because the record points to it.

## Common mistakes

- **Channels in another order, or offsets along transposed axes.** The scorer cannot detect this;
  the score is just low.
- **Fit and test images predicted with different settings.** The threshold chosen on the fit
  volumes then does not suit the test volumes.
- **Reusing a `RUN` name.** The scorer refuses to overwrite an existing record; pick a new name.
