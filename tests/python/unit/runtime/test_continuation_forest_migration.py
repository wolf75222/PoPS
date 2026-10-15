"""Rank-table handoff is separate from native numerical migration acceptance."""

import numpy as np
import pytest

from pops.runtime._amr_checkpoint_program_shards import capture_program_shards, select_program_shard


def archive():
    result = {}
    capture_program_shards(result, (b"POPSANDF-old-0", b"POPSANDF-old-1"))
    result["program_accepted_state_source_authority"] = np.frombuffer(b"POPSRSA1-sealed", dtype=np.uint8)
    return result


def test_rank_change_hands_all_images_to_native_migration_without_altering_archive():
    payload = archive()
    before = {key: value.copy() for key, value in payload.items()}
    selected = select_program_shard(payload, source_ranks=2, target_ranks=3, target_rank=2,
                                    allow_native_migration=True)
    assert tuple(selected["program_accepted_state_rank_%d" % i].tobytes() for i in range(2)) == (
        b"POPSANDF-old-0", b"POPSANDF-old-1")
    assert "program_accepted_state_rank_2" not in selected
    for key, value in before.items():
        np.testing.assert_array_equal(payload[key], value)


@pytest.mark.parametrize("bad", (True, 0, 1, "yes", None))
def test_migration_capability_never_replaces_archive_authentication(bad):
    payload = archive()
    del payload["program_accepted_state_rank_1"]
    with pytest.raises((TypeError, ValueError)):
        select_program_shard(payload, source_ranks=2, target_ranks=3, target_rank=2,
                             allow_native_migration=bad)


def test_rank_change_with_capability_still_requires_ranked_native_authority():
    payload = archive()
    payload["program_accepted_state_source_authority"] = np.frombuffer(b"legacy", dtype=np.uint8)
    with pytest.raises(ValueError, match="ranked native source authority"):
        select_program_shard(payload, source_ranks=2, target_ranks=3, target_rank=2,
                             allow_native_migration=True)


@pytest.mark.parametrize("capacity", (False, 0, -1, 1 << 63))
def test_invalid_migration_workspace_rejects_before_native_or_archive_access(capacity):
    from pops.runtime._amr_checkpoint_v3 import prepare_v3
    with pytest.raises(ValueError, match="workspace_byte_capacity"):
        prepare_v3(None, None, {}, workspace_byte_capacity=capacity)


def test_direct_amr_restart_forwards_explicit_workspace_as_runtime_policy(monkeypatch):
    from pops.output import _checkpoint_collective
    from pops.runtime._amr_system_io import _AmrSystemIO
    calls = []
    monkeypatch.setattr(_checkpoint_collective, "restore_checkpoint_path",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    owner = _AmrSystemIO()
    owner.restart("checkpoint.npz", workspace_byte_capacity=123456789)
    assert calls[0][1]["workspace_byte_capacity"] == 123456789
    assert calls[0][1]["bit_identical"] is False
    assert calls[0][0] == (owner, owner, "checkpoint.npz")
