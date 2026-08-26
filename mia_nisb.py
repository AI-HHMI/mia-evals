"""Reading a published NISB cube, after the OME-NGFF reorganisation of 2026-08-14.

NISB used to ship each cube as a flat zarr v2 group -- `<cube>/data.zarr` holding `img` as
(x, y, z, c) uint8 and `seg` as (x, y, z). That layout no longer exists anywhere. Cubes are now
OME-NGFF zarr v3, one group per cube:

    <cube>.zarr/
        raw/s0 .. s4                          (c, x, y, z) uint8
        labels/public_gt-cell-nisb/s0 .. s3   (x, y, z)    uint16
        skeleton.pkl                          unchanged
        zarr.json

Three things changed, and only the first announces itself:

  * **The keys were renamed**: `img` -> `raw`, `labels/seg` -> `labels/public_gt-cell-nisb`.
    A stale key raises at the read, so this one is safe.
  * **The channel axis moved to the front.** `raw/s0` is (c, x, y, z) where `img` was (x, y, z, c).
    Code that slices the first three axes as spatial now slices the *channel* axis instead, and a
    1-channel cube makes that a silently empty or wrong-shaped patch rather than an error. This is
    the trap, and it is why these accessors exist rather than a bare `zarr.open` at each callsite.
  * **There is a resolution pyramid now.** `s0` is the native 9 x 9 x 20 nm level and is what
    scoring must use; a coarser level would change what a voxel means and quietly rescale every
    metric.

Centralised so the long label key is written once and the axis convention is stated once. Used by
`mia_predict.py`, `mia_score.py` and `visualize_affinities.py`.
"""

from __future__ import annotations

from pathlib import Path

import zarr

RAW_KEY = "raw"
LABEL_KEY = "labels/public_gt-cell-nisb"
NATIVE_LEVEL = "s0"


def open_raw(cube: Path, level: str = NATIVE_LEVEL) -> zarr.Array:
    """The EM image of a cube as **(c, x, y, z)** uint8 -- channel first, unlike the old `img`."""
    return zarr.open(str(cube), mode="r")[f"{RAW_KEY}/{level}"]


def open_labels(cube: Path, level: str = NATIVE_LEVEL) -> zarr.Array:
    """The ground-truth instance segmentation as (x, y, z) uint16."""
    return zarr.open(str(cube), mode="r")[f"{LABEL_KEY}/{level}"]


def spatial_shape(cube: Path, level: str = NATIVE_LEVEL) -> tuple[int, int, int]:
    """(x, y, z) of a cube, with the channel axis dropped.

    Separate from `open_raw(...).shape` because that now carries a leading channel: every caller
    that used to write `image.shape[:3]` would silently take (c, x, y) after the reorganisation.
    """
    shape = open_raw(cube, level).shape
    return (int(shape[1]), int(shape[2]), int(shape[3]))


def read_patch(raw: zarr.Array, origin, size) -> "zarr.core.Array":
    """A (c, x, y, z) block, indexing the spatial axes and keeping every channel.

    A one-liner, but the one place the channel-first layout is easy to get wrong twice: the
    leading `:` is what keeps this from slicing channels as if they were x.
    """
    x, y, z = origin
    sx, sy, sz = (size, size, size) if isinstance(size, int) else size
    return raw[:, x : x + sx, y : y + sy, z : z + sz]
