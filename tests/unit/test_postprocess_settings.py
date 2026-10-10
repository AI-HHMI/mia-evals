"""A [postprocess] key the route does not take is an error, not a setting kept and never read."""
from __future__ import annotations

import glob
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCORING = sorted(glob.glob(str(ROOT / "configs" / "*" / "*.toml")))


@pytest.mark.unit
def test_a_setting_of_another_route_is_refused():
    """`repulsive_strides` is mws's. cc_threshold once kept a key it did not take (`fill_distances`,
    before it took one) and ran without it (2026-09-29)."""
    import components  # noqa: F401
    from postprocess.registry import PostprocessRegistry

    with pytest.raises(ValueError, match=r"unknown key\(s\) \['repulsive_strides'\]"):
        PostprocessRegistry.build("cc_threshold", repulsive_strides=[1])


@pytest.mark.unit
def test_a_postprocessor_without_settings_refuses_any():
    import components  # noqa: F401
    from postprocess.registry import PostprocessRegistry

    with pytest.raises(ValueError, match="it takes none"):
        PostprocessRegistry.build("argmax", min_sizes=[0])


@pytest.mark.unit
@pytest.mark.parametrize("path", SCORING, ids=lambda p: str(Path(p).relative_to(ROOT / "configs")))
def test_every_scoring_config_builds_its_postprocessor(path):
    from conftest import needs_newer_miao

    import components  # noqa: F401
    from config import load_scoring_config
    from postprocess.registry import PostprocessRegistry

    try:
        config = load_scoring_config(Path(path))
    except ValueError as error:
        if needs_newer_miao(error):
            pytest.skip("its data pins a frame with fixed_axes; the installed miao predates it")
        raise
    PostprocessRegistry.build(config.postprocess.name, **config.postprocess.kwargs)
