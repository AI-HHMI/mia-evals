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
