"""Explicit-lifetime native communicator used by post-commit observer workers."""
from __future__ import annotations

import pytest

from pops import _pops


def test_observer_lane_is_functional_until_an_idempotent_collective_close():
    world = _pops.mpi_world()
    lane = world.duplicate_observer_lane("pytest-post-commit-observer")
    rank = int(world.rank)
    size = int(world.size)

    try:
        assert lane.identity == "%s/pytest-post-commit-observer" % world.identity
        assert lane.active is True
        assert lane.closed is False
        assert lane.rank == rank
        assert lane.size == size

        root_payload = b"root\x00observer"
        assert lane.broadcast_bytes(root_payload if rank == 0 else b"ignored") == root_payload
        expected = tuple(("rank-%d" % source).encode() for source in range(size))
        local = ("rank-%d" % rank).encode()
        assert lane.allgather_bytes(local) == expected
        gathered = lane.gather_bytes(local)
        if rank == 0:
            assert gathered == expected
        else:
            assert gathered is None
        lane.barrier()

        if _pops.__has_mpi__:
            assert isinstance(lane.fortran_handle, int)
        else:
            with pytest.raises(RuntimeError, match="serial PoPS build"):
                _ = lane.fortran_handle
    finally:
        lane.close_collectively()

    lane.close_collectively()
    assert lane.active is False
    assert lane.closed is True
    assert lane.identity.endswith("/pytest-post-commit-observer")
    with pytest.raises(RuntimeError, match="closed"):
        lane.barrier()


def test_observer_lane_rejects_an_empty_collective_identity():
    with pytest.raises(ValueError, match="identit"):
        _pops.mpi_world().duplicate_observer_lane("")
