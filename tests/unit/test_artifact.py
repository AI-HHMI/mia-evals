"""The artifact contract. Every test here is a way to score the wrong thing without an error."""

from __future__ import annotations

import numpy as np
import pytest

from artifact import KINDS, open_artifact, write_artifact

pytestmark = pytest.mark.unit


def test_every_kind_declares_a_canonical_form():
    """A kind with no canonical form could be postprocessed into either, unchecked."""
    from artifact import CANONICAL_FORMS

    for kind, spec in KINDS.items():
        assert spec["canonical"] in CANONICAL_FORMS, kind


def test_missing_kind_is_rejected_with_a_usable_message(tmp_path):
    """An artifact predating the spec must not be guessed at."""
    import zarr

    path = tmp_path / "legacy.zarr"
    store = zarr.open(str(path), mode="w", shape=(6, 4, 4, 4), dtype="f2")
    store[:] = 0.5
    with pytest.raises(ValueError, match="declares no `kind`"):
        open_artifact(path)


def test_labelling_without_background_id_is_rejected(tmp_path):
    """The distinction this exists for: 0 from connected components is not background."""
    import zarr

    path = tmp_path / "labels.zarr"
    store = zarr.open(str(path), mode="w", shape=(4, 4, 4), dtype="i4")
    store[:] = 1
    store.attrs.update(kind="instances")
    with pytest.raises(ValueError, match="no `background_id`"):
        open_artifact(path)


def test_affinity_channel_count_is_checked_against_rank(tmp_path):
    """Three channels over three axes is the short-range half, not a 3D affinity artifact."""
    with pytest.raises(ValueError, match="needs 6 channels"):
        write_artifact(tmp_path / "half.zarr", np.zeros((3, 4, 4, 4), np.float16), "affinity")

    written = write_artifact(
        tmp_path / "full.zarr", np.zeros((6, 4, 4, 4), np.float16), "affinity"
    )
    assert open_artifact(written).channels == 6


def test_read_uses_absolute_coordinates(tmp_path):
    """A caller holding a bounding box has it in volume coordinates, not artifact-local ones."""
    data = np.arange(4 * 4 * 4, dtype=np.int32).reshape(4, 4, 4)
    path = write_artifact(
        tmp_path / "block.zarr", data, "instances", origin=(10, 20, 30), background_id=0
    )
    artifact = open_artifact(path)
    assert artifact.origin == (10, 20, 30)

    block = artifact.read((11, 21, 31), (2, 2, 2))
    assert np.array_equal(block, data[1:3, 1:3, 1:3])

    # Asking outside the covered extent is an error, not a silently shifted read.
    with pytest.raises(ValueError, match="Origins are absolute"):
        artifact.read((0, 0, 0), (2, 2, 2))


def test_canonical_form_follows_the_kind(tmp_path):
    affinity = write_artifact(
        tmp_path / "a.zarr", np.zeros((6, 4, 4, 4), np.float16), "affinity"
    )
    scores = write_artifact(
        tmp_path / "s.zarr", np.zeros((5, 4, 4), np.float32), "class_scores"
    )
    assert open_artifact(affinity).canonical == "instances"
    assert open_artifact(scores).canonical == "classes"


def test_read_honours_a_channel_limit(tmp_path):
    """A consumer that reads three of six channels must not pay for six.

    Not a micro-optimisation at the sizes this runs at: the zebrafish doublecube's six affinity
    channels are 85 GB as float16 while `cc_threshold` reads three, and reading all of them then
    slicing is the difference between fitting a 300 GB reservation and being killed part-way
    through a ten-hour job.
    """
    data = np.arange(6 * 4 * 4 * 4, dtype=np.float16).reshape(6, 4, 4, 4)
    path = write_artifact(tmp_path / "aff.zarr", data, "affinity")
    artifact = open_artifact(path)

    assert artifact.read().shape == (6, 4, 4, 4)
    assert artifact.read(channels=3).shape == (3, 4, 4, 4)
    assert np.array_equal(artifact.read(channels=3), data[:3])

    block = artifact.read((1, 1, 1), (2, 2, 2), channels=3)
    assert block.shape == (3, 2, 2, 2)
    assert np.array_equal(block, data[:3, 1:3, 1:3, 1:3])

    # Asking for more than there are is clamped, not an error: a postprocessor declaring 3 against
    # a 1-channel artifact is caught by the kind check, not here.
    assert artifact.read(channels=99).shape == (6, 4, 4, 4)


def test_a_labelling_read_needs_no_channel_axis(tmp_path):
    labels = np.arange(4 * 4 * 4, dtype=np.int32).reshape(4, 4, 4)
    path = write_artifact(tmp_path / "seg.zarr", labels, "instances", background_id=0)
    artifact = open_artifact(path)
    assert artifact.channels is None
    assert np.array_equal(artifact.read(channels=3), labels)


def test_multiscale_group_is_rejected_with_a_usable_message(tmp_path):
    """Pointing at an OME-Zarr pyramid is the natural mistake, so it needs a real message.

    The source volumes read through `miao` are multiscale OME-Zarr groups, so a producer or a
    collaborator handing one to the scorer is expected rather than perverse. Before this check it
    raised `AttributeError: 'Group' object has no attribute 'shape'`, which names neither the
    problem nor the fix.
    """
    import zarr

    path = tmp_path / "pyramid.zarr"
    group = zarr.open_group(str(path), mode="w")
    for level, extent in ((0, 8), (1, 4)):
        array = group.create_array(name=f"s{level}", shape=(extent,) * 3, dtype="u4")
        array[:] = 1
    group.attrs.update({"kind": "instances", "background_id": 0})

    with pytest.raises(ValueError, match="is a zarr group, not an array"):
        open_artifact(path)
    # The message has to say which level to name, or it only halves the problem.
    with pytest.raises(ValueError, match="s0"):
        open_artifact(path)


def test_a_single_level_of_a_pyramid_is_a_valid_artifact(tmp_path):
    """The fix the message recommends must actually work."""
    import zarr

    path = tmp_path / "pyramid.zarr"
    group = zarr.open_group(str(path), mode="w")
    array = group.create_array(name="s0", shape=(8, 8, 8), dtype="u4")
    array[:] = 1
    array.attrs.update({"kind": "instances", "background_id": 0})

    artifact = open_artifact(path / "s0")
    assert artifact.kind == "instances"
    assert artifact.spatial_shape == (8, 8, 8)


def _ome_group(path, array, *, axes="zyx", voxel=(8.0, 8.0, 8.0), shift=(0.0, 0.0, 0.0),
               datasets=("s0",), **attrs):
    """A single-level OME-Zarr 0.5 group the way mia-train's predict.py writes one."""
    import zarr

    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    for name in datasets:
        level = group.create_array(name=name, shape=array.shape, dtype=array.dtype)
        level[:] = array
        level.attrs.update(**attrs)
    group.attrs.update(
        ome={"version": "0.5", "multiscales": [{
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in axes],
            "datasets": [{"path": name, "coordinateTransformations": [
                {"type": "scale", "scale": list(voxel)},
                {"type": "translation", "translation": list(shift)},
            ]} for name in datasets],
        }]},
        **attrs,
    )
    return path


def test_a_single_level_ome_group_is_an_artifact(tmp_path):
    """The layout predict.py writes so viewers can place the prediction on the raw volume: the
    one dataset is the array, the group's attrs are the artifact's, and reads are unchanged."""
    labels = np.arange(4 * 4 * 4, dtype=np.uint32).reshape(4, 4, 4)
    path = _ome_group(tmp_path / "v.zarr", labels, kind="instances", background_id=0,
                      origin=[10, 20, 30], run="r")

    artifact = open_artifact(path)
    assert artifact.kind == "instances" and artifact.spatial_shape == (4, 4, 4)
    assert artifact.origin == (10, 20, 30) and artifact.attrs["run"] == "r"
    assert "ome" not in artifact.attrs
    assert artifact.array_path == path / "s0"
    assert np.array_equal(artifact.read((11, 20, 30), (2, 4, 4)), labels[1:3])
    assert np.array_equal(artifact.load(), labels)


def test_an_ome_pyramid_is_still_refused(tmp_path):
    """Two levels do not name the lattice; the message still says which level to name."""
    labels = np.ones((8, 8, 8), dtype=np.uint32)
    path = _ome_group(tmp_path / "p.zarr", labels, datasets=("s0", "s1"), kind="instances",
                      background_id=0)
    with pytest.raises(ValueError, match="zarr group, not an array"):
        open_artifact(path)
    with pytest.raises(ValueError, match="s0"):
        open_artifact(path)


@pytest.mark.parametrize("kind", ["instances", "class_labels"])
def test_a_float_labelling_is_rejected(tmp_path, kind):
    """float32 holds integers exactly only below 2**24, so ids above that silently merge.

    The damage is invisible downstream: the array still reads back as a labelling with a plausible
    object count, so it must be refused at the read rather than trusted and cast.
    """
    labels = np.zeros((8, 8, 8), dtype=np.float32)
    labels[2:6, 2:6, 2:6] = 1
    with pytest.raises(ValueError, match="must be an integer type"):
        write_artifact(tmp_path / f"{kind}.zarr", labels, kind, background_id=0)


@pytest.mark.parametrize("kind", ["affinity", "boundary", "sdt", "class_scores", "embedding"])
def test_float_is_still_fine_for_the_score_kinds(tmp_path, kind):
    """The dtype check must not catch the kinds that are genuinely floating-point."""
    channels = 6 if kind == "affinity" else 1 if kind in ("boundary", "sdt") else 3
    array = np.full((channels, 4, 4, 4), 0.5, dtype=np.float32)
    artifact = open_artifact(write_artifact(tmp_path / f"{kind}.zarr", array, kind))
    assert artifact.kind == kind


def test_a_scored_labelling_keeps_the_source_geometry_and_shifts_to_its_region(tmp_path):
    """The voxels a row was scored on, written so a viewer places them where the artifact was."""
    from artifact import open_artifact, write_scored

    labels = np.arange(4 * 6 * 8, dtype=np.int64).reshape(4, 6, 8) % 7
    like = _ome_group(tmp_path / "pred.zarr", np.ones((4, 6, 8), dtype=np.uint32),
                      voxel=(8.0, 8.0, 8.0), shift=(100.0, 200.0, 300.0),
                      kind="instances", background_id=0, origin=[0, 0, 0])
    source = open_artifact(like)
    # Scored over a sub-region starting one voxel in along the first axis.
    path = write_scored(tmp_path / "scored.zarr", labels[1:], source, (1, 0, 0),
                        convention="size_filter(min_size=5)")
    scored = open_artifact(path)
    assert scored.kind == "instances" and scored.origin == (1, 0, 0)
    assert scored.attrs["source_artifact"] == str(like)
    assert np.array_equal(scored.load(), labels[1:]) and scored.load().dtype == np.uint32
    import zarr
    ome = dict(zarr.open_group(str(path), mode="r").attrs)["ome"]
    (dataset,) = ome["multiscales"][0]["datasets"]
    assert dataset["coordinateTransformations"][1]["translation"] == [108.0, 200.0, 300.0]
    with pytest.raises(ValueError, match="negative"):
        write_scored(tmp_path / "bad.zarr", np.array([[[-1]]]), source, (0, 0, 0))
