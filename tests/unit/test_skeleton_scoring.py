"""Skeleton scoring at the LSD benchmark's scale, on synthetic data.

What the zebrafinch tasks rest on, each checked where a mistake would still produce a number:
positions are absolute and cut to the region (a region not starting at voxel 0 used to be scored
uncropped); a lazily read labelling looks up exactly what the array holds; axes are compared before
any lookup; a 478-gigavoxel region is never read; and the ground truth's hash is part of the task.
Plus the two halves of the benchmark-data builder: the OME wrapper as miao reads it, and the checks
on LSD's component ids.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import textwrap
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import zarr

from artifact import LazyLabelling, open_artifact, write_artifact

pytestmark = pytest.mark.unit

ORIGIN = (10, 20, 30)                                # z, y, x of the scored region
SHAPE = (4, 8, 8)
RESOLUTION = (20.0, 9.0, 9.0)


def _skeleton() -> nx.Graph:
    """Two straight pieces, component 1 at z=11, y=22 and component 2 at z=12, y=26, x = 31..36."""
    graph = nx.Graph(axes="zyx")
    node = 0
    for component, (z, y) in ((1, (11, 22)), (2, (12, 26))):
        previous = None
        for x in range(31, 37):
            position = (z, y, x)
            nm = tuple(p * r for p, r in zip(position, RESOLUTION, strict=True))
            graph.add_node(node, id=component, index_position=position, nm_position=nm)
            if previous is not None:
                graph.add_edge(previous, node)
            previous, node = node, node + 1
    return graph


def _labelling(split: bool = False) -> np.ndarray:
    """The region's labelling: each piece its own segment, optionally piece 1 cut in two at x=34."""
    labels = np.zeros(SHAPE, dtype=np.uint32)
    labels[11 - ORIGIN[0], 22 - ORIGIN[1], 31 - ORIGIN[2]:37 - ORIGIN[2]] = 5
    labels[12 - ORIGIN[0], 26 - ORIGIN[1], 31 - ORIGIN[2]:37 - ORIGIN[2]] = 9
    if split:
        labels[11 - ORIGIN[0], 22 - ORIGIN[1], 34 - ORIGIN[2]:37 - ORIGIN[2]] = 6
    return labels


def _context(tmp_path, axes="zyx"):
    return {"origin": ORIGIN, "shape": SHAPE, "whole_region": True, "axes": axes,
            "scratch_dir": tmp_path / "scratch"}


def test_positions_are_absolute_and_cropped_to_a_region_not_at_the_origin(tmp_path):
    pytest.importorskip("funlib.evaluate")
    from metrics.skeleton import SkeletonExpectedRunLength

    metric = SkeletonExpectedRunLength()
    perfect = metric(_labelling(), _skeleton(), **_context(tmp_path))
    assert perfect["nerl"] == pytest.approx(1.0)
    assert perfect["voi_split"] == pytest.approx(0.0) and perfect["voi_merge"] == pytest.approx(0.0)
    assert perfect["n_splits"] == 0 and perfect["skeleton_nodes"] == 12
    assert perfect["erl_um"] == pytest.approx(perfect["erl"] / 1000)

    cut = metric(_labelling(split=True), _skeleton(), **_context(tmp_path))
    assert 0.0 < cut["nerl"] < 1.0 and cut["n_splits"] == 1


def test_axes_are_compared_before_anything_is_looked_up(tmp_path):
    from metrics.skeleton import SkeletonExpectedRunLength

    with pytest.raises(ValueError, match="'zyx' order but the artifact's axes are 'xyz'"):
        SkeletonExpectedRunLength()(_labelling(), _skeleton(), **_context(tmp_path, axes="xyz"))


def _stored(tmp_path, labels, origin=ORIGIN, chunks=(2, 4, 4)) -> Path:
    return write_artifact(tmp_path / "labels.zarr", labels, "instances", background_id=0,
                          origin=origin, chunks=chunks)


def test_a_lazy_labelling_looks_up_exactly_what_the_array_holds(tmp_path):
    rng = np.random.default_rng(0)
    full = rng.integers(0, 1000, size=(6, 12, 12)).astype(np.uint64)
    artifact = open_artifact(_stored(tmp_path, full, origin=(8, 18, 28), chunks=(2, 5, 3)))
    lazy = LazyLabelling(artifact, ORIGIN, SHAPE, workers=4)
    points = np.stack([rng.integers(0, s, size=500) for s in SHAPE], axis=1)
    local = points + (np.asarray(ORIGIN) - (8, 18, 28))
    assert np.array_equal(lazy.lookup(points), full[tuple(local.T)])
    with pytest.raises(TypeError, match="refusing to materialise"):
        np.asarray(lazy)
    with pytest.raises(IndexError):
        lazy.lookup(np.array([[0, 0, SHAPE[2]]]))


def test_a_lazy_labelling_scores_exactly_as_the_array_does(tmp_path):
    pytest.importorskip("funlib.evaluate")
    from metrics.skeleton import SkeletonExpectedRunLength

    labels = _labelling(split=True)
    lazy = LazyLabelling(open_artifact(_stored(tmp_path, labels)), ORIGIN, SHAPE)
    metric = SkeletonExpectedRunLength()
    assert metric(lazy, _skeleton(), **_context(tmp_path)) == \
        metric(labels, _skeleton(), **_context(tmp_path))


def _data_config(root: Path, store: Path, name: str) -> Path:
    box = [[o, o + s] for o, s in zip(ORIGIN, SHAPE, strict=True)]
    path = root / "data.yaml"
    path.write_text(textwrap.dedent(f"""\
        resolutions: [[20.0, 9.0, 9.0]]
        patch_size: [4, 4, 4]
        output_axes: lczyx
        volumes:
          - name: {name}
            path: {store}
            image_key: raw
            zarr_version: zarr2
            bounding_box: {json.dumps(box)}
        """))
    return path


def test_scoring_a_stored_labelling_never_reads_the_region(tmp_path, monkeypatch):
    pytest.importorskip("funlib.evaluate")
    import artifact as artifact_module
    import evaluate

    name = "cube_test"
    store = tmp_path / "store.zarr"
    (store / "skeletons").mkdir(parents=True)
    skeleton_file = store / "skeletons" / f"{name}.pkl"
    skeleton_file.write_bytes(pickle.dumps(_skeleton()))
    data = _data_config(tmp_path, store, name)
    config = tmp_path / "skeleton.toml"
    config.write_text(textwrap.dedent(f"""\
        task_name = "unit_tracing"

        [data]
        config_path = "{data}"

        [task]
        name = "instance_seg"
        truth_kind = "skeleton"
        skeleton_name = "skeletons/{{volume}}.pkl"

        [postprocess]
        name = "identity"

        [metric]
        names = ["skeleton_erl"]
        rank_by = "skeleton_erl"
        """))
    labels = tmp_path / "artifacts"
    labels.mkdir()
    stored = write_artifact(labels / f"{name}.zarr", _labelling(split=True), "instances",
                            background_id=0, origin=ORIGIN, chunks=(2, 4, 4), run="unit_run")

    def refuse(*args, **kwargs):
        raise AssertionError("the region was read")

    monkeypatch.setattr(artifact_module.Artifact, "read", refuse)
    records = tmp_path / "leaderboard"
    args = type("Args", (), {
        "config": config, "test": labels, "val": None, "leaderboard": records, "run_dir": None,
        "scored_out": None, "no_scored": False, "scratch": tmp_path / "scratch",
    })()
    evaluate.cmd_score(args)

    [written] = list(records.rglob("records/*.json"))
    payload = json.loads(written.read_text())
    assert 0.0 < payload["ranking"]["value"] < 1.0 and payload["ranking"]["key"] == "nerl"
    assert payload["region"]["volumes"][name]["scored_artifact"] == str(stored)
    assert payload["config"]["volumes"][0]["truth_sha256"] == \
        hashlib.sha256(skeleton_file.read_bytes()).hexdigest()


def _funlib_array(path: Path, shape, resolution, offset=(0, 0, 0)):
    array = zarr.open_array(str(path), mode="w", shape=shape, dtype="u1", chunks=shape,
                            zarr_format=2)
    array.attrs.update(offset=list(offset), resolution=list(resolution))


def test_a_wrapper_gives_miao_the_geometry_the_release_lacks(tmp_path):
    from miao.zarr_meta import read_ome_metadata

    from truth.common import link, wrap_pyramid

    source = tmp_path / "release"
    _funlib_array(source / "s0", (4, 8, 8), (20, 9, 9))
    _funlib_array(source / "s1", (4, 4, 4), (20, 18, 18))
    levels = [("s0", source / "s0"), ("s1", source / "s1")]
    wrapper = tmp_path / "wrapper.zarr"
    wrap_pyramid(wrapper / "raw", "raw", levels, (20.0, 9.0, 9.0))
    meta = read_ome_metadata(wrapper, "raw", "zarr2")
    assert meta.axis_names == ["z", "y", "x"]
    assert meta.scales[0].scale_factors == [20.0, 9.0, 9.0]
    assert meta.scales[1].translation == [0.0, 4.5, 4.5]   # mean-pooled centres of s0's pairs
    assert (wrapper / "raw" / "s0").is_symlink() and meta.scales[1].shape == [4, 4, 4]
    wrap_pyramid(wrapper / "raw", "raw", levels, (20.0, 9.0, 9.0))   # idempotent
    with pytest.raises(FileExistsError):
        link(source / "s1", wrapper / "raw" / "s0")


def test_a_wrapped_array_keeps_its_world_offset(tmp_path):
    """Hemibrain arrays sit at their place in the brain: the translation carries the offset."""
    from miao.zarr_meta import read_ome_metadata

    from truth.common import wrap_pyramid

    _funlib_array(tmp_path / "raw", (4, 4, 4), (8, 8, 8), offset=(1600, 800, 8))
    _funlib_array(tmp_path / "mask", (1, 1, 1), (256, 256, 256), offset=(1024, 512, 0))
    wrapper = tmp_path / "roi.zarr"
    wrap_pyramid(wrapper / "raw", "raw", [("s0", tmp_path / "raw")], (8.0, 8.0, 8.0))
    wrap_pyramid(wrapper / "mask", "mask", [("s0", tmp_path / "mask")], (8.0, 8.0, 8.0))
    assert read_ome_metadata(wrapper, "raw", "zarr2").scales[0].translation == [1600, 800, 8]
    assert read_ome_metadata(wrapper, "mask", "zarr2").scales[0].translation == [1148, 636, 124]


def test_hemibrain_skeletons_lie_inside_their_objects_on_raw_voxels(tmp_path, monkeypatch):
    """Two tubes in a core offset (2, 3, 4) voxels inside its raw: one piece each, nodes inside."""
    pytest.importorskip("kimimaro")
    import truth.lsd_hemibrain as hb

    roi = tmp_path / "roi_1"
    _funlib_array(roi / "raw", (24, 28, 70), (8, 8, 8), offset=(800, 1600, 3200))
    labels = np.zeros((20, 22, 62), dtype=np.uint64)
    labels[4:9, 4:9, 2:60] = 7_000_000_001                 # 64-bit ids, as the release's are
    labels[10:15, 10:15, 2:60] = 7_000_000_002             # off the shared planes at z, y = 16
    array = zarr.open_array(str(roi / "consolidated_ids"), mode="w", shape=labels.shape,
                            dtype="u8", chunks=labels.shape, zarr_format=2)
    array[:] = labels
    array.attrs.update(offset=[816, 1624, 3232], resolution=[8, 8, 8])
    monkeypatch.setattr(hb, "GROUND_TRUTH", tmp_path)
    monkeypatch.setattr(hb, "DOWNSAMPLE", 4)
    monkeypatch.setattr(hb, "BLOCK", 16)                  # each tube crosses four blocks
    monkeypatch.setattr(hb, "DUST", 100)

    graph, stats = hb.region_skeleton("roi_1", workers=1)
    assert hb.core_box("roi_1") == [[2, 22], [3, 25], [4, 66]]
    # One unit per tube, and one piece: the blocks' pieces met on their shared faces.
    assert stats["objects"] == stats["objects_skeletonised"] == 2 and stats["components"] == 2
    assert stats["skeleton_pieces"] == 2
    assert {graph.nodes[n]["id"] for n in graph.nodes} == {1, 2}
    assert {graph.nodes[n]["neuron_id"] for n in graph.nodes} == {7_000_000_001, 7_000_000_002}
    for n in graph.nodes:
        z, y, x = np.asarray(graph.nodes[n]["index_position"]) - (2, 3, 4)
        assert labels[z, y, x] == graph.nodes[n]["neuron_id"]
    # Two tubes of ~57 steps x 8 nm; unblocked kimimaro gives 942 nm. Joining on the three shared
    # planes each tube crosses must not add track.
    assert 880 < stats["cable_mm"] * 1e6 < 1000


def _synthetic_release(monkeypatch, component_ids):
    """LSD's dumps for two 3-node pieces inside the 11 um region, with the given component ids."""
    import truth.lsd_zebrafinch as zf

    z0, y0, x0 = (lo for lo, _ in zf.roi_box("11um"))
    nodes = [{"id": i, "z": (z0 + 1) * 20, "y": (y0 + 1 + i // 3 * 5) * 9, "x": (x0 + i % 3) * 9,
              "neuron_id": 100 + i // 3} for i in range(6)]
    edges = [{"source": 0, "target": 1}, {"source": 1, "target": 2},
             {"source": 3, "target": 4}, {"source": 4, "target": 5}]
    dumps = {
        "components": [{"id": i, "component_id": c} for i, c in enumerate(component_ids)],
        "mask": [{"id": i, "masked": True} for i in range(6)],
    }
    monkeypatch.setattr(zf, "read_bson", lambda path: dumps[
        "components" if "components" in path.name else "mask"])
    monkeypatch.setattr(zf, "sha256", lambda path: "0" * 64)
    return zf, nodes, edges


def test_the_builder_keeps_lsd_components_and_absolute_voxels(monkeypatch):
    zf, nodes, edges = _synthetic_release(monkeypatch, [7, 7, 7, 8, 8, 8])
    graph, stats = zf.region_skeleton(nodes, edges, "test", "11um")
    assert stats["components"] == 2 and stats["nodes"] == 6 and stats["neurons"] == 2
    z0, y0, x0 = (lo for lo, _ in zf.roi_box("11um"))
    assert graph.nodes[4]["index_position"] == (z0 + 1, y0 + 6, x0 + 1)
    assert graph.nodes[4]["id"] == 8 and graph.graph["axes"] == "zyx"


@pytest.mark.parametrize("component_ids, message", [
    ([7, 7, 7, 7, 7, 7], "not the skeleton's connected components"),   # two pieces, one id
    ([7, 7, 9, 8, 8, 8], "not the skeleton's connected components"),   # one piece, two ids
    ([0, 0, 0, 8, 8, 8], "numbered 0"),
])
def test_the_builder_refuses_components_that_are_not_lsds(monkeypatch, component_ids, message):
    zf, nodes, edges = _synthetic_release(monkeypatch, component_ids)
    with pytest.raises(ValueError, match=message):
        zf.region_skeleton(nodes, edges, "test", "11um")
