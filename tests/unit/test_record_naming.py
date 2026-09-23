"""Record identifiers are `<run>.step<N>.<route>`, never a hand-written label."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from report.record import Submission


def _submission(**overrides) -> Submission:
    base = dict(
        task_name="t", scores={}, ranking={"metric": "m", "key": "k", "value": 1.0},
        postprocess={"name": "cc_threshold"}, region={}, config={},
        producer={"run": "/nrs/x/runs/gary__1a_dinov3_axial_subpixel_20260916_215544",
                  "step": 500000, "artifacts": {}},
    )
    base.update(overrides)
    return Submission(**base)


@pytest.mark.unit
def test_identifier_names_run_step_and_route():
    assert _submission(route="mws").identifier() == \
        "gary__1a_dinov3_axial_subpixel_20260916_215544.step500000.mws"


@pytest.mark.unit
def test_route_falls_back_to_the_postprocessor_name():
    assert _submission().identifier().endswith(".step500000.cc_threshold")


@pytest.mark.unit
def test_a_legacy_label_still_names_its_record():
    assert _submission(label="2c_step50000").identifier() == "2c_step50000"


@pytest.mark.unit
def test_route_is_read_from_the_scoring_config(tmp_path):
    from config import load_scoring_config
    data = tmp_path / "d.yaml"
    data.write_text(
        "resolutions: [[8, 8, 8]]\npatch_size: [8, 8, 8]\noutput_axes: lcxyz\n"
        "volumes:\n- name: v\n  path: /nowhere/v.zarr\n  image_key: raw\n  label_key: labels/gt\n"
    )
    body = ("task_name = \"t\"\n[data]\nconfig_path = \"d.yaml\"\n[task]\nname = \"instance_seg\"\n"
            "[postprocess]\nname = \"size_filter\"\n[metric]\nnames = [\"voxel_instance\"]\n")
    (tmp_path / "plain.toml").write_text(body)
    assert load_scoring_config(tmp_path / "plain.toml").route == "size_filter"
    (tmp_path / "named.toml").write_text("route = \"mws\"\n" + body)
    assert load_scoring_config(tmp_path / "named.toml").route == "mws"
    (tmp_path / "bad.toml").write_text("route = \"has space\"\n" + body)
    with pytest.raises(ValueError, match="filesystem-safe"):
        load_scoring_config(tmp_path / "bad.toml")


@pytest.mark.unit
def test_mws_runs_the_watershed_once_per_stride_across_the_size_sweep(monkeypatch):
    from postprocess import mws as module
    calls = []
    real = module.segment
    def counted(affinities, stride, **options):
        calls.append(stride)
        return real(affinities, stride, **options)
    monkeypatch.setattr(module, "segment", counted)
    rng = np.random.default_rng(0)
    aff = rng.random((6, 12, 12, 12), dtype=np.float32)
    proc = module.MutexWatershed(repulsive_strides=[1], min_sizes=[0, 5])
    a = proc(aff.copy(), repulsive_stride=1, min_size=0)      # a fresh array object each time,
    b = proc(aff.copy(), repulsive_stride=1, min_size=5)      # as the scorer re-reads the artifact
    assert calls == [1]
    assert a.shape == b.shape == (12, 12, 12)
    assert (b == 0).sum() >= (a == 0).sum()                   # the filter only removes
    proc(aff.copy() * 0.5, repulsive_stride=1, min_size=0)   # different affinities: a new watershed
    assert calls == [1, 1]
