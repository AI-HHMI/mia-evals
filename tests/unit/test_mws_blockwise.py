"""Blockwise mutex watershed: exactly mutex watershed under one change of order, on any cut.

`postprocess.mws_blockwise` decides every edge inside a block before any edge that crosses a block
face, and is otherwise the algorithm `mws.segment` runs. On volumes small enough for the Python
reference to run the modified order directly, these pin that down: one block covering the region is
`mws.segment` label for label; any cut, with or without a mask and at any repulsive stride, is the
reference's partition on the modified order; and the machinery around it -- a mask placed by OME
geometry, workers, markers, a finisher, a plan refusing a different run, the scorer's lazy path --
leaves the numbers alone.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import textwrap
from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import zarr

from artifact import write_artifact
from postprocess.mws import build_edges, mutex_watershed_reference, segment
from postprocess.mws_blockwise import (
    OFFSETS,
    ArraySource,
    BlockwiseRun,
    Source,
    block_boxes,
    lattice_mask,
)

pytestmark = pytest.mark.unit

SHAPE = (36, 34, 32)
CHUNK = 4                                  # small chunks, so small blocks are legal


def _affinities(seed: int = 0, shape: tuple[int, ...] = SHAPE) -> np.ndarray:
    """Six float16 channels from a blocky labelling plus noise: clusters that cross block faces,
    and float16's ties, which the canonical order has to break."""
    rng = np.random.default_rng(seed)
    grid = np.indices(shape)
    labels = (grid[0] // 7) * 10_000 + (grid[1] // 9) * 100 + grid[2] // 6
    affinities = np.empty((6, *shape), dtype=np.float32)
    for channel, offset in enumerate(OFFSETS):
        same = labels == np.roll(labels, [-o for o in offset], axis=(0, 1, 2))
        affinities[channel] = np.clip(0.1 + 0.8 * same + rng.normal(0, 0.15, shape), 0, 1)
    return affinities.astype(np.float16)


def _mask(seed: int = 1) -> np.ndarray:
    """A half-space plus speckles: masked slabs and isolated holes, across block faces."""
    rng = np.random.default_rng(seed)
    grid = np.indices(SHAPE)
    return (grid[0] + grid[1] < 52) & (rng.random(SHAPE) > 0.08)


def _reference(affinities: np.ndarray, block: tuple[int, ...], stride: int,
               mask: np.ndarray | None) -> np.ndarray:
    """The Python reference on the modified order: within-block edges first, then crossing ones,
    each in the canonical order (descending priority, then offset, then source)."""
    u, v, priority, attractive = build_edges(affinities.astype(np.float32), stride)
    if mask is not None:
        inside = mask.reshape(-1)
        keep = inside[u] & inside[v]
        u, v, priority, attractive = u[keep], v[keep], priority[keep], attractive[keep]
    size = [min(b, s) for b, s in zip(block, SHAPE, strict=True)]
    cu = np.stack(np.unravel_index(u, SHAPE)) // np.asarray(size)[:, None]
    cv = np.stack(np.unravel_index(v, SHAPE)) // np.asarray(size)[:, None]
    crossing = np.any(cu != cv, axis=0)
    canonical = np.argsort(-priority, kind="stable")
    order = np.concatenate([canonical[~crossing[canonical]], canonical[crossing[canonical]]])
    ranks = np.arange(order.size, 0, -1, dtype=np.float64)        # the reference keeps this order
    labels = mutex_watershed_reference(u[order], v[order], ranks, attractive[order],
                                       int(np.prod(SHAPE)))
    return labels.reshape(SHAPE)


def _same_partition(a: np.ndarray, b: np.ndarray, where: np.ndarray) -> bool:
    pairs = np.unique(np.stack([a[where], b[where]]), axis=1)
    return pairs.shape[1] == np.unique(a[where]).size == np.unique(b[where]).size


def _blockwise(tmp_path, affinities, block, stride=1, mask=None, processes=1):
    run = BlockwiseRun(ArraySource(affinities, mask), tmp_path / "labels.zarr", block, stride,
                       chunk=CHUNK)
    run.run(processes=processes)
    return run.read(), run


@pytest.mark.parametrize("stride", [1, 2])
def test_one_block_is_mws_segment_label_for_label(tmp_path, stride):
    affinities = _affinities()
    labels, _ = _blockwise(tmp_path, affinities, SHAPE, stride)
    assert np.array_equal(labels, segment(affinities.astype(np.float32), stride))


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("block", [(12, 12, 12), (16, 20, 12), (12, 36, 16), SHAPE])
def test_any_cut_is_the_reference_on_the_blocks_first_order(tmp_path, block, stride, masked):
    affinities = _affinities()
    mask = _mask() if masked else None
    labels, run = _blockwise(tmp_path, affinities, block, stride, mask)
    inside = np.ones(SHAPE, dtype=bool) if mask is None else mask
    assert np.all(labels[~inside] == 0) and np.all(labels[inside] > 0)
    assert _same_partition(labels, _reference(affinities, block, stride, mask), inside)
    if block != SHAPE:
        # The cut is real: clusters were joined across faces, not merely left alone.
        assert run.stats["crossing_pairs"] > 0
        assert run.stats["segments"] < run.stats["clusters"]


def test_a_pool_changes_nothing(tmp_path):
    affinities = _affinities(seed=3)
    alone, _ = _blockwise(tmp_path / "one", affinities, (12, 12, 12))
    pooled, _ = _blockwise(tmp_path / "pool", affinities, (12, 12, 12), processes=3)
    assert np.array_equal(alone, pooled)


def test_blocks_must_not_share_a_chunk_or_be_shallower_than_the_long_offset():
    with pytest.raises(ValueError, match="multiple of 4"):
        block_boxes(SHAPE, (14, 12, 12), 1, CHUNK)
    with pytest.raises(ValueError, match="at least 10"):
        block_boxes(SHAPE, (8, 12, 12), 1, CHUNK)
    with pytest.raises(ValueError, match="repulsive stride 3"):
        block_boxes(SHAPE, (16, 12, 12), 3, CHUNK)
    boxes = block_boxes(SHAPE, (12, 36, 16), 1, CHUNK)            # 36 >= 34 spans its axis
    assert len(boxes) == 3 * 1 * 2 and boxes[-1] == ((24, 0, 16), (36, 34, 32))


def _ome_group(path, array, axes, scale, translation, channel=False):
    """A single-level OME 0.5 group around `array`, as mia-train's predict.py writes one."""
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.create_array(name="s0", shape=array.shape, dtype=array.dtype)[:] = array
    names = [{"name": a, "type": "space", "unit": "nanometer"} for a in axes]
    if channel:
        names.insert(0, {"name": "c", "type": "channel"})
        scale, translation = [1.0, *scale], [0.0, *translation]
    group.attrs.update(ome={"version": "0.5", "multiscales": [{
        "axes": names, "datasets": [{"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": list(scale)},
            {"type": "translation", "translation": list(translation)}]}]}]})
    return path


def test_a_mask_is_placed_by_both_sides_ome_geometry(tmp_path):
    """A 20 x 18 x 18 nm mask on a 20 x 9 x 9 nm lattice that starts two z planes into it: each
    lattice voxel takes the mask voxel its centre falls in, as the zebrafinch neuropil mask is."""
    from artifact import open_artifact

    rng = np.random.default_rng(0)
    mask = (rng.random((12, 8, 8)) > 0.5).astype(np.uint8)
    _ome_group(tmp_path / "mask.zarr", mask, "zyx", [20.0, 18.0, 18.0], [0.0, 4.5, 4.5])
    lattice = np.zeros((6, 8, 16, 16), dtype=np.float16)
    path = _ome_group(tmp_path / "aff.zarr", lattice, "zyx", [20.0, 9.0, 9.0],
                      [40.0, 0.0, 0.0], channel=True)
    zarr.open_group(str(path), mode="a").attrs.update(kind="affinity")
    artifact = open_artifact(path)

    placed = lattice_mask(artifact, tmp_path / "mask.zarr", (1, 2, 4), (6, 10, 8))
    expected = np.repeat(np.repeat(mask, 2, axis=1), 2, axis=2)[2:][1:7, 2:12, 4:12] != 0
    assert placed.dtype == bool and np.array_equal(placed, expected)
    with pytest.raises(ValueError, match="does not cover the region"):
        lattice_mask(artifact, tmp_path / "mask.zarr", (5, 0, 0), (6, 16, 16))
    moved = replace(artifact, origin=(100, 200, 300))      # as `BaseTask.place` moves a prediction
    assert np.array_equal(lattice_mask(moved, tmp_path / "mask.zarr", (101, 202, 304), (6, 10, 8)),
                          placed)


def _source(tmp_path, affinities, origin=(5, 6, 7)):
    path = write_artifact(tmp_path / "aff.zarr", affinities, "affinity", origin=origin)
    return Source(str(path), origin, SHAPE)


def test_workers_share_the_blocks_and_one_finisher_completes(tmp_path):
    affinities = _affinities(seed=2)
    source = _source(tmp_path, affinities)
    out = tmp_path / "out" / "v.zarr"
    out.parent.mkdir()

    def run(**kwargs):
        return BlockwiseRun(source, out, (12, 12, 12), 1, chunk=CHUNK).run(**kwargs)

    assert run(worker=0, workers=2) is None and not out.exists()
    assert run(worker=1, workers=2) is None
    assert run() == out                                   # the finisher: nothing left to segment
    assert not out.with_name("v.zarr.partial").exists()
    assert not out.with_name("v.zarr.blocks").exists()
    expected, _ = _blockwise(tmp_path / "memory", affinities, (12, 12, 12))
    assert np.array_equal(np.asarray(zarr.open_array(str(out / "s0"), mode="r")[:]), expected)
    assert run() == out                                   # complete: left alone


def test_a_worker_that_loses_the_race_to_make_the_partial_joins_the_winners(tmp_path, monkeypatch):
    """Workers started together all find no partial, and each makes one (2026-10-05, NISB: one of
    nine lost zarr's look-then-create race and crashed; in a stress test, a second creator found
    the other workers' segmented blocks and deleted the partial they were writing into). Replays
    the worst order: another worker publishes its partial, and segments its share into it, while
    this one is still building its own. This one's copy must be discarded and the other's used,
    and the labelling must be the one a single worker makes."""
    affinities = _affinities(seed=2)
    source = _source(tmp_path, affinities)
    out = tmp_path / "out" / "v.zarr"
    out.parent.mkdir()

    def run(**kwargs):
        return BlockwiseRun(source, out, (12, 12, 12), 1, chunk=CHUNK).run(**kwargs)

    real = BlockwiseRun._make_partial
    built = []

    def make(self, where):
        built.append(where)
        if len(built) == 1:                     # the worker started alongside gets there first
            assert run(worker=1, workers=2) is None
        real(self, where)

    monkeypatch.setattr(BlockwiseRun, "_make_partial", make)
    assert run(worker=0, workers=2) is None
    assert len(built) == 2 and not any(where.exists() for where in built)
    assert not list(out.parent.glob(".v.zarr.partial.*"))
    assert run() == out
    expected, _ = _blockwise(tmp_path / "memory", affinities, (12, 12, 12))
    assert np.array_equal(np.asarray(zarr.open_array(str(out / "s0"), mode="r")[:]), expected)


def test_the_plan_is_written_once_and_never_replaced(tmp_path):
    """Every worker that finds no plan writes one. Replacing a plan that is there would swap the
    file under a worker on another node reading it, which NFS answers with ESTALE (seen in the
    stress test, 2026-10-05): the first plan stays, the same file, and no temporary is left."""
    from postprocess.mws_blockwise import _write_json_once

    path = tmp_path / "plan.json"
    _write_json_once(path, {"plan": 1})
    inode = path.stat().st_ino
    _write_json_once(path, {"plan": 2})
    assert json.loads(path.read_text()) == {"plan": 1} and path.stat().st_ino == inode
    assert [p.name for p in tmp_path.iterdir()] == ["plan.json"]


def _held(root) -> list[str]:
    """Files under `root` this process holds open or memory-mapped."""
    held = []
    for fd in Path("/proc/self/fd").iterdir():
        try:
            held.append(os.readlink(fd))
        except OSError:
            continue
    held += [line.split(maxsplit=5)[5].strip() for line in
             Path("/proc/self/maps").read_text().splitlines() if len(line.split()) >= 6]
    return sorted({p for p in held if p.startswith(str(root))})


def test_nothing_is_held_open_when_the_work_is_deleted(tmp_path, monkeypatch):
    """On NFS a file deleted while still open or mapped stays as a hidden `.nfs*` entry and its
    directory cannot be removed -- the finisher once held the stitched table mapped and failed
    there. Local disks hide it, so the handles are checked at the moment of deletion."""
    probe = tmp_path / "probe.npy"                      # the check sees a mapping, as it must
    np.save(probe, np.zeros(4096))
    mapped = np.load(probe, mmap_mode="r")
    assert _held(tmp_path) == [str(probe)]
    del mapped
    assert _held(tmp_path) == []
    real = shutil.rmtree

    def checked(path, *args, **kwargs):
        held = _held(path)
        assert not held, f"{held} still held while {path} is deleted"
        return real(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", checked)
    BlockwiseRun(ArraySource(_affinities()), tmp_path / "labels.zarr", (12, 12, 12), 1,
                 chunk=CHUNK).run()
    assert not (tmp_path / "labels.zarr.blocks").exists()


def test_a_partial_run_is_continued_only_by_the_same_watershed(tmp_path):
    source = _source(tmp_path, _affinities())
    out = tmp_path / "v.zarr"
    BlockwiseRun(source, out, (12, 12, 12), 1, chunk=CHUNK).run(worker=0, workers=2)
    with pytest.raises(SystemExit, match="different watershed"):
        BlockwiseRun(source, out, (16, 12, 12), 1, chunk=CHUNK).run(worker=1, workers=2)
    shutil.rmtree(out.with_name("v.zarr.partial"))           # the labels gone, the markers left
    with pytest.raises(SystemExit, match="was missing"):
        BlockwiseRun(source, out, (12, 12, 12), 1, chunk=CHUNK).run(worker=1, workers=2)
    assert not out.with_name("v.zarr.partial").exists()      # the partial made to find out: gone
    assert not list(tmp_path.glob(".v.zarr.partial.*"))


@pytest.mark.parametrize("placed", [False, True])
def test_the_scorer_builds_the_labelling_once_and_reads_only_what_the_skeleton_touches(
        tmp_path, placed):
    """`mia-evals score` with route mws_blockwise: the labelling is built under the scratch
    directory over the scored region only, the record names it, and the score is the one the same
    labelling gets in memory -- whether the affinities state their position as `origin` or, as
    mia-train's predict.py writes them, as `native_box` (`BaseTask.place`), which the pool
    processes that reopen them must honour too."""
    pytest.importorskip("funlib.evaluate")
    import evaluate
    from metrics.skeleton import SkeletonExpectedRunLength

    affinities = _affinities(seed=4)
    origin, box = (5, 6, 7), [[9, 33], [8, 40], [7, 39]]       # the region: inside the artifact
    name = "cube_test"
    store = tmp_path / "store.zarr"
    (store / "skeletons").mkdir(parents=True)
    graph, node = nx.Graph(axes="zyx"), 0
    for component, (z, y) in enumerate(((12, 15), (20, 30)), start=1):
        for x in range(10, 36):
            graph.add_node(node, id=component, index_position=(z, y, x),
                           nm_position=(z * 20.0, y * 9.0, x * 9.0))
            if x > 10:
                graph.add_edge(node - 1, node)
            node += 1
    (store / "skeletons" / f"{name}.pkl").write_bytes(pickle.dumps(graph))
    data = tmp_path / "data.yaml"
    data.write_text(textwrap.dedent(f"""\
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
    config = tmp_path / "mws_blockwise.toml"
    config.write_text(textwrap.dedent(f"""\
        task_name = "unit_tracing"

        [data]
        config_path = "{data}"

        [task]
        name = "instance_seg"
        truth_kind = "skeleton"
        skeleton_name = "skeletons/{{volume}}.pkl"

        [postprocess]
        name = "mws_blockwise"
        block = [12, 12, 12]
        chunk = {CHUNK}

        [metric]
        names = ["skeleton_erl"]
        rank_by = "skeleton_erl"
        """))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    if placed:
        write_artifact(artifacts / f"{name}.zarr", affinities, "affinity", run="unit_run",
                       step=1, scale=[1.0, 1.0, 1.0], covers_full_box=False,
                       native_box=[[o, o + s] for o, s in zip(origin, SHAPE, strict=True)])
    else:
        write_artifact(artifacts / f"{name}.zarr", affinities, "affinity", origin=origin,
                       run="unit_run", step=1)
    scratch = tmp_path / "scratch"
    args = type("Args", (), {
        "config": config, "test": artifacts, "val": None, "leaderboard": tmp_path / "board",
        "run_dir": None, "scored_out": None, "no_scored": False, "scratch": scratch,
    })()
    evaluate.cmd_score(args)

    [written] = list((tmp_path / "board").rglob("records/*.json"))
    payload = json.loads(written.read_text())
    built = scratch / "mws_blockwise" / "repulsive_stride1" / f"{name}.zarr"
    assert payload["region"]["volumes"][name]["scored_artifact"] == str(built / "s0")
    assert written.name == "unit_run.step1.mws_blockwise.json"

    window = tuple(slice(lo - o, hi - o) for (lo, hi), o in zip(box, origin, strict=True))
    region = affinities[(slice(None), *window)]
    expected, _ = _blockwise(tmp_path / "memory", region, (12, 12, 12))
    stored = np.asarray(zarr.open_array(str(built / "s0"), mode="r")[:])
    assert np.array_equal(stored, expected)
    context = {"origin": tuple(lo for lo, _ in box), "shape": expected.shape,
               "whole_region": True, "axes": None, "scratch_dir": tmp_path / "metric"}
    in_memory = SkeletonExpectedRunLength()(expected, graph, **context)
    assert payload["ranking"]["value"] == pytest.approx(in_memory["nerl"])
