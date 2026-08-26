"""Mutex watershed: affinities -> instances with no threshold at all.

Moved here from the repository root, where it was a standalone script. The algorithm is ours,
written from the paper -- Wolf et al., *The Mutex Watershed* (ECCV 2018) -- because the reference
implementation `affogato` is CMake-only and does not pip-install, and the PyPI package of that name
is an unrelated project. `tests/unit/test_mws.py` checks it against cases whose answers follow from
the definition rather than from a reference.

Why it exists beside `cc_threshold`: the model is trained on six channels and measurement showed
the three that thresholded components discards are its *better* predictions -- the long-range
channels separate same-object from different-object more confidently (+0.40 to +0.46) than the
short-range ones (+0.35), because a 10-voxel relationship is a contextual question a ViT answers
well while a 1-voxel one needs localisation a 16x-upsampled head cannot express. Mutex watershed
uses both: short-range as *attractive* edges, long-range as *repulsive* ones, and no threshold
anywhere -- repulsion does the separating.

**Measured worse on this task, so far.** On a 512^3 block of seed100 with the best available
checkpoint it scored 0.024 nERL against thresholded components' 0.584, with 242 mergers against 4.
Most of that is `repulsive_stride = 4` discarding 64x of the repulsive edges (at stride 1 it
recovers to 0.380), but components still win by 2x. It is kept because the reasoning above is
sound and the failure is a tuning story, not because it is currently the better choice.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BasePostprocess
from .registry import PostprocessRegistry

SHORT_OFFSETS = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
LONG = 10
LONG_OFFSETS = ((LONG, 0, 0), (0, LONG, 0), (0, 0, LONG))


def build_edges(
    affinities: np.ndarray, repulsive_stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(6, X, Y, Z) affinities -> flat edge arrays (u, v, priority, attractive).

    Priority is what the algorithm sorts on, and it differs by edge type. For an attractive edge
    the affinity *is* the merge evidence, so priority = a. For a repulsive edge the evidence is
    that the two voxels are *different*, which is strong when the affinity is low, so
    priority = 1 - a. Sorting both by priority descending puts the most confident assertion of
    either kind first, which is exactly what mutex watershed requires.

    `repulsive_stride` subsamples the long-range edges. Every voxel contributing three repulsive
    edges is affordable at small volumes and not at large ones, and the repulsive edges exist to
    place constraints rather than to cover every pair -- taking every k-th voxel keeps the
    constraint field while cutting the edge count by k^3.
    """
    shape = affinities.shape[1:]
    index = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)

    us, vs, priorities, attractive = [], [], [], []
    for channel, offset in enumerate(SHORT_OFFSETS + LONG_OFFSETS):
        is_attractive = channel < len(SHORT_OFFSETS)
        # max(s - o, 0): a negative stop would wrap and silently produce a `u` and `v` of
        # different lengths, which is reachable whenever an axis is shorter than the long-range
        # offset -- true of any block under 10 voxels deep.
        overlap = tuple(slice(0, max(s - o, 0)) for s, o in zip(shape, offset, strict=True))
        shifted = tuple(slice(o, o + max(s - o, 0)) for s, o in zip(shape, offset, strict=True))
        a = affinities[channel][overlap]
        u, v = index[overlap], index[shifted]

        if not is_attractive and repulsive_stride > 1:
            keep = tuple(slice(None, None, repulsive_stride) for _ in shape)
            a, u, v = a[keep], u[keep], v[keep]

        us.append(u.ravel())
        vs.append(v.ravel())
        priorities.append((a if is_attractive else 1.0 - a).ravel())
        attractive.append(np.full(u.size, is_attractive, dtype=bool))

    return (
        np.concatenate(us),
        np.concatenate(vs),
        np.concatenate(priorities).astype(np.float32),
        np.concatenate(attractive),
    )


def mutex_watershed(
    u: np.ndarray, v: np.ndarray, priority: np.ndarray, attractive: np.ndarray, n_nodes: int
) -> np.ndarray:
    """Mutex watershed: edges in descending priority -> a label per node.

    Union-find, plus a set of forbidden partners per cluster. Walking edges from most to least
    confident:

      * an **attractive** edge merges its two clusters, unless a mutex forbids it;
      * a **repulsive** edge records a mutex between them, so no later (weaker) attractive edge can
        join them.

    No threshold appears anywhere. Every attractive edge would eventually merge if left alone, so
    it is the repulsive constraints that carve the partition -- which is why the long-range
    channels matter and why discarding them forces a threshold to do their job.
    """
    parent = np.arange(n_nodes, dtype=np.int64)
    forbidden: dict[int, set[int]] = {}

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:      # path compression
            parent[x], x = root, parent[x]
        return int(root)

    order = np.argsort(-priority, kind="stable")
    for e in order:
        ru, rv = find(int(u[e])), find(int(v[e]))
        if ru == rv:
            continue
        if attractive[e]:
            if rv in forbidden.get(ru, ()):
                continue
            # Union by set size, so the mutex sets merge in the cheaper direction.
            big, small = (
                (ru, rv) if len(forbidden.get(ru, ())) >= len(forbidden.get(rv, ()))
                else (rv, ru)
            )
            parent[small] = big
            moved = forbidden.pop(small, set())
            if moved:
                target = forbidden.setdefault(big, set())
                for other in moved:
                    target.add(other)
                    partners = forbidden.get(other)
                    if partners is not None:
                        partners.discard(small)
                        partners.add(big)
        else:
            forbidden.setdefault(ru, set()).add(rv)
            forbidden.setdefault(rv, set()).add(ru)

    roots = np.array([find(i) for i in range(n_nodes)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return (labels + 1).astype(np.uint32)


def segment(affinities: np.ndarray, repulsive_stride: int) -> np.ndarray:
    """(6, X, Y, Z) affinities -> (X, Y, Z) uint32 instance labels."""
    shape = affinities.shape[1:]
    u, v, priority, attractive = build_edges(affinities, repulsive_stride)
    labels = mutex_watershed(u, v, priority, attractive, int(np.prod(shape)))
    return labels.reshape(shape)


@PostprocessRegistry.register("mws")
class MutexWatershed(BasePostprocess):
    """Six-channel affinities -> instances, with the repulsive edges doing the separating.

    `repulsive_strides` is the sweep. Subsampling the long-range edges is what makes this
    affordable on a large block -- every voxel contributing three repulsive edges is fine at 128^3
    and not at 2000^3 -- but it is also the setting that decides the result, so it is fitted on
    validation rather than defaulted and forgotten. Stride 1 keeps every repulsive edge.
    """

    accepts = ("affinity",)
    produces = "instances"

    def __init__(
        self,
        repulsive_strides: tuple[int, ...] | list[int] = (1, 2, 4),
        **settings: Any,
    ) -> None:
        super().__init__(repulsive_strides=repulsive_strides, **settings)
        if not repulsive_strides:
            raise ValueError("mws with an empty `repulsive_strides` has nothing to sweep")
        bad = sorted(s for s in repulsive_strides if int(s) < 1)
        if bad:
            raise ValueError(f"repulsive_strides must all be >= 1, got {bad}; 1 keeps every edge")
        self.repulsive_strides = tuple(int(s) for s in repulsive_strides)

    def search_space(self) -> list[dict[str, Any]]:
        return [{"repulsive_stride": stride} for stride in self.repulsive_strides]

    def __call__(self, array: np.ndarray, **params: Any) -> np.ndarray:
        if array.shape[0] < 6:
            raise ValueError(
                f"mutex watershed needs all six affinity channels -- three attractive and three "
                f"repulsive -- but this artifact has {array.shape[0]}. With only the short-range "
                "half there are no repulsive edges, so it would degenerate to connected components "
                "over the attractive graph; use cc_threshold for that."
            )
        return segment(np.asarray(array, dtype=np.float32),
                       int(params["repulsive_stride"])).astype(np.int64)

    def describe(self, params: dict[str, Any]) -> str:
        return f"mws(repulsive_stride={params['repulsive_stride']})"
