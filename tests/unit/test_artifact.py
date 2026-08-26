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
