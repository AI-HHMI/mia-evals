#!/usr/bin/env bash
# The README's Quick Start, end to end, with real paths: predict -> score -> leaderboard -> figures.
# Read it top to bottom; every command below is the command, nothing is built or hidden.
#
#     bash docs/quickstart_demo.sh          # from the mia-evals checkout; about four minutes
#
# It predicts a 512^3 block of the NISB val cube and of the NISB test cube (27 tiles each), because a
# whole 3000x3000x1350 cube takes ~35 min to predict and ~10 min per threshold to score. A block's
# nERL is NOT the benchmark's number (branches are truncated) and the table says "(sub-region)".
#
# The GPU step must be an LSF job. The CPU steps are submitted too, so nothing heavy runs inside the
# VS Code job; `bsub -I` streams a job's output here and returns its exit code. Each `bsub` line is
# only the prefix -- the command underneath it is what you would type on a node.
set -euo pipefail
cd ~/projects/mia-evals

# The producer: a mia-train run directory = resolved_config.json + checkpoints/step_N/. The model is
# rebuilt from the former and its weights loaded from the latter.
RUN=/nrs/scicompsoft/orhane/mia-train-runs/subpixel_decoder__subpixel_256_20260811_143947

# Where the artifacts go. Step 1 writes into $ART/val and $ART/test; step 2 reads exactly those.
ART=/nrs/scicompsoft/orhane/scratch/mia-evals-demo/artifacts
mkdir -p $ART
echo "Regenerate with: bash ~/projects/mia-evals/docs/quickstart_demo.sh" > $ART/README.txt

export PATH=~/banisvenv/bin:$PATH             # mia-evals, mia-evals-viz-*: installed console entry points
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
set -x                                        # echo each command, with $RUN and $ART expanded

# ---- 1. Produce the prediction artifacts. This is mia-train's job, run once per cube. -----------
#
#   <run_dir>      $RUN, above
#   --data-config  the same miao YAML the task file references, so producer and scorer agree on the
#                  volume: path, image/label keys, 9x9x20 nm (native, no resampling), patch 256
#   --volume       which volume in that YAML; the artifact is named after it
#   --origin/--size  predict only this block (x y z, s0 voxels); omit them for the whole cube
#   --out          a DIRECTORY: <volume>.zarr (6 affinity channels, float16, with origin/kind/run/step
#                  as attributes) and <volume>.gt.zarr (co-registered ground truth) are written into it
#   PYTHONPATH     mia-train's imports, set for predict.py only: both repos have a top-level
#                  `components` module, so exporting it for the whole script breaks `mia-evals`
bsub -P miaai -q gpu_h100 -gpu "num=1" -n 2 -W 0:30 -I \
  env PYTHONPATH=~/projects/mia-train/src ~/myvenv/bin/python ~/projects/mia-train/src/predict.py $RUN --step 100000 \
    --data-config configs/data/nisb_base_val.yaml --volume nisb_base_val_seed100 \
    --origin 1024 1024 384 --size 512 512 512 \
    --out $ART/val

bsub -P miaai -q gpu_h100 -gpu "num=1" -n 2 -W 0:30 -I \
  env PYTHONPATH=~/projects/mia-train/src ~/myvenv/bin/python ~/projects/mia-train/src/predict.py $RUN --step 100000 \
    --data-config configs/data/nisb_base_test.yaml --volume nisb_base_test_seed101 \
    --origin 1024 1024 384 --size 512 512 512 \
    --out $ART/test

ls $ART/val $ART/test
#   $ART/val:  nisb_base_val_seed100.zarr   nisb_base_val_seed100.gt.zarr
#   $ART/test: nisb_base_test_seed101.zarr  nisb_base_test_seed101.gt.zarr

# ---- 2. Fit the threshold on --val, report on --test, write a record. mia-evals from here on. ----
#
#   --val          the --out directory of the VAL cube. The scorer finds <volume>.zarr in it by the
#                  fit task's volume name, sweeps logits 3..7 there and keeps the best nERL
#   --val-config   the task file naming that val cube (it is a different volume from the reported one;
#                  without this, --val would be fitted against the TEST cube's skeleton)
#   --test         the --out directory of the TEST cube; the chosen threshold is applied once and only
#                  these numbers are reported
#   --run-dir      the run directory again; its resolved config and git commit are copied into the record
#   --label        the record's file name (default: <run>_step<N>)
bsub -P miaai -q interactive -n 4 -W 0:30 -I \
  mia-evals score configs/tasks/nisb_base_neuron_instance.toml \
    --val  $ART/val  --val-config configs/tasks/nisb_base_neuron_instance_fit.toml \
    --test $ART/test \
    --run-dir $RUN --label demo_subpixel_step100000_block512

# ---- 3. Rebuild the tables from the records (scoring already did this for its own task). --------
mia-evals leaderboard --task nisb_base_neuron_instance
mia-evals leaderboard
cat leaderboard/nisb_base_neuron_instance/README.md

# ---- 4. Look at the test prediction: the affinities, then the segmentation that was scored. -----
LOGIT=$(jq .postprocess.params.logit leaderboard/nisb_base_neuron_instance/records/demo_subpixel_step100000_block512.json)

bsub -P miaai -q interactive -n 4 -W 0:30 -I \
  mia-evals-viz-affinities --affinities $ART/test/nisb_base_test_seed101.zarr \
    --cube /groups/miaai/miaai/lmd-v0.0.1/dev/nisb/base/test/seed101.zarr --origin 1024 1024 512 --size 256

bsub -P miaai -q interactive -n 4 -W 0:30 -I \
  mia-evals-viz-segmentation --prediction $ART/test/nisb_base_test_seed101.zarr --logit $LOGIT --min-size 5000

ls $ART/test/*.png            # the figures land beside the artifact

# To take the demo's row out of the leaderboard again:
#   rm leaderboard/nisb_base_neuron_instance/records/demo_subpixel_step100000_block512.json
#   rmdir leaderboard/nisb_base_neuron_instance/records 2>/dev/null && rm -r leaderboard/nisb_base_neuron_instance
#   mia-evals leaderboard
