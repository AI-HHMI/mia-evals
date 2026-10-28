"""Use mia-evals as a library: mutex watershed on synthetic affinities, then score it.

Run with `python examples/library_usage.py` after `pip install -e .`.
"""

import numpy as np

from metrics.voxel_instance import contingency, pq_from_table, rand_error_from_table, voi_from_table
from postprocess.mws import LONG, segment

# Ground truth: two slabs along X, 32 X 16 Y 16 Z.
truth = np.ones((32, 16, 16), dtype=np.int64)
truth[16:] = 2

# Affinities (6 X Y Z): channels 0-2 short-range attractive, 3-5 repulsive at offset LONG.
# Affinity is 1 where the two voxels share a label, else 0.
affinities = np.zeros((6, *truth.shape), dtype=np.float32)
for channel in range(6):
    axis, step = channel % 3, 1 if channel < 3 else LONG
    a = np.take(truth, range(truth.shape[axis] - step), axis=axis)
    b = np.take(truth, range(step, truth.shape[axis]), axis=axis)
    index = [slice(None)] * 3
    index[axis] = slice(0, truth.shape[axis] - step)
    affinities[(channel, *index)] = (a == b)

# Add noise so the result isn't trivially perfect.
rng = np.random.default_rng(0)
affinities = np.clip(affinities + rng.normal(0, 0.2, affinities.shape), 0, 1).astype(np.float32)

labels = segment(affinities, repulsive_stride=1)  # X Y Z, labels 1..k
print("instances found:", labels.max())

table = contingency(truth, labels)  # compute once, share across metrics
print("voi:", voi_from_table(table))
print("adapted rand error:", rand_error_from_table(table))
print("pq:", pq_from_table(table))
