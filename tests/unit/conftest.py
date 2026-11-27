"""Shared by the tests that load every scoring config under configs/."""

from __future__ import annotations


def needs_newer_miao(error: Exception) -> bool:
    """Whether a config failed to load only because the installed miao predates `fixed_axes`.

    A data config that pins one frame of a time series needs miao's `fixed_axes`
    (AI-HHMI/miao#39, merged 2026-10-07 without a version bump, so `miao-io>=` cannot ask for it),
    which an older miao rejects as an unknown key. Such a config is skipped rather than failed until
    miao is upgraded; with a miao that knows the setting, it is checked like any other.
    """
    import miao.config

    return "fixed_axes" in str(error) and "fixed_axes" not in miao.config.VolumeConfig.model_fields


def name_axes(store, axes: str, shape: tuple[int, ...], key: str = "raw",
              zarr_format: int = 3) -> None:
    """Give a test store an image group that names its axes, as every real store does.

    The scorer converts a volume's bounding_box to the store's axis order, so a store whose data
    config declares a box must name its axes. The image array is empty: no score reads it.
    """
    from pathlib import Path

    import zarr

    group = zarr.open_group(str(Path(store) / key), mode="a", zarr_format=zarr_format)
    group.create_array("s0", shape=shape, dtype="uint8")
    multiscale = {
        "axes": [{"name": axis, "type": "space"} for axis in axes],
        "datasets": [{"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0] * len(axes)}]}],
    }
    if zarr_format == 3:
        group.attrs["ome"] = {"version": "0.5", "multiscales": [multiscale]}
    else:
        group.attrs["multiscales"] = [{"version": "0.4", **multiscale}]
