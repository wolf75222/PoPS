"""Temporary source archive budgets are explicit and do not bypass rank-table seals."""
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from pops.identity import make_identity
from pops.output._checkpoint_collective import decode_checkpoint_bytes
from pops.output._checkpoint_restart_budget import restart_checkpoint_resource_budget
from pops.runtime._amr_checkpoint_program_shards import capture_program_shards, select_program_shard
from pops.runtime._checkpoint_manifest import _seal_checkpoint_payload_with_identities
from pops.runtime._checkpoint_resource_budget import _producer_checkpoint_resource_budget


def archive(ranks):
    data = {"t": np.asarray(0.125), "macro_step": np.asarray(2, dtype=np.int64),
            "n_ranks": np.asarray(ranks, dtype=np.int64),
            "program_accepted_state_source_authority": np.frombuffer(b"POPSRSA1-fixture", dtype=np.uint8)}
    capture_program_shards(data, [b"POPSANDF" + bytes([rank]) for rank in range(ranks)])
    _seal_checkpoint_payload_with_identities(data, runtime_kind="amr", **{
        name: make_identity(name if name != "run" else "run", {"fixture": 1})
        for name in ("semantic", "artifact", "bind", "run")})
    target = BytesIO()
    np.savez_compressed(target, **data)
    return data, target.getvalue()


def owner():
    data, _ = archive(1)
    return SimpleNamespace(_checkpoint_resource_budget=_producer_checkpoint_resource_budget(
        data, runtime_kind="amr", authority="installed-single-rank-schema"))


def test_three_to_one_decode_uses_requested_source_workspace_and_preserves_full_table():
    target = owner()
    data, raw = archive(3)
    base = target._checkpoint_resource_budget
    with pytest.raises(ValueError, match="member count"):
        decode_checkpoint_bytes(raw, base)
    budget = restart_checkpoint_resource_budget(target, 1 << 20)
    decoded = decode_checkpoint_bytes(raw, budget)
    selected = select_program_shard(decoded, source_ranks=3, target_ranks=1,
                                    target_rank=0, allow_native_migration=True)
    for rank in range(3):
        name = "program_accepted_state_rank_%d" % rank
        np.testing.assert_array_equal(selected[name], data[name])
    assert target._checkpoint_resource_budget is base
    with pytest.raises(ValueError, match="member count"):
        decode_checkpoint_bytes(raw, restart_checkpoint_resource_budget(target, 1))


def test_workspace_does_not_bypass_payload_digest_or_missing_rank_rejection():
    target = owner()
    data, _ = archive(3)
    data["program_accepted_state_rank_2"] = np.frombuffer(b"POPSANDFchanged", dtype=np.uint8)
    buffer = BytesIO()
    np.savez_compressed(buffer, **data)
    with pytest.raises(ValueError):
        decode_checkpoint_bytes(buffer.getvalue(), restart_checkpoint_resource_budget(target, 1 << 20))
    data, raw = archive(3)
    decoded = dict(decode_checkpoint_bytes(raw, restart_checkpoint_resource_budget(target, 1 << 20)))
    del decoded["program_accepted_state_rank_2"]
    with pytest.raises(ValueError, match="incomplete"):
        select_program_shard(decoded, source_ranks=3, target_ranks=1,
                             target_rank=0, allow_native_migration=True)


@pytest.mark.parametrize("capacity", (True, 0, -1, 1.5, sys.maxsize + 1))
def test_restart_workspace_rejects_invalid_resource_authority(capacity):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        restart_checkpoint_resource_budget(owner(), capacity)


def test_identical_and_uniform_restarts_preserve_installed_budget():
    target = owner()
    base = target._checkpoint_resource_budget
    assert restart_checkpoint_resource_budget(target, 1 << 20, bit_identical=True) is base
    target._checkpoint_resource_budget = replace(base, runtime_kind="uniform")
    assert restart_checkpoint_resource_budget(target, 1 << 20) is target._checkpoint_resource_budget
