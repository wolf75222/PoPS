"""Native communicator primitive reception; run with MPI2/MPI3 after the coordinated build.

These checks do not qualify a complete Program or inject allocator faults into CPython itself.
MemoryError is raised inside the guarded callback on one rank to test the native phase boundary.
"""
from __future__ import annotations

import struct

import pytest

from pops import _pops


@pytest.fixture(params=("world", "observer"))
def lane(request):
    world = _pops.mpi_world()
    value = (world if request.param == "world" else
             world.duplicate_observer_lane("pytest-v10-local-preparation"))
    try:
        yield value
    finally:
        if request.param == "observer":
            value.close_collectively()


def test_collective_prepare_returns_exact_owned_object(lane):
    result = {"integer": 2**60 + 17, "bytes": b"a\x00\xff", "rank": int(lane.rank)}
    assert lane.collective_prepare(lambda: result) is result
    assert lane.agree_control_bytes(b"closed-type", struct.pack("<q", result["integer"]))


def test_one_rank_callback_memory_failure_converges_before_followup(lane):
    def prepare():
        if int(lane.rank) == int(lane.size) - 1:
            raise MemoryError("injected local diagnostic allocation failure")
        return {"pending": 2**60 + 19}

    with pytest.raises(RuntimeError, match="collective Python local preparation failed"):
        lane.collective_prepare(prepare)
    assert lane.allgather_bytes(b"alive") == (b"alive",) * int(lane.size)


def test_one_rank_noncallable_is_checked_inside_collective(lane):
    callback = None if int(lane.rank) == int(lane.size) - 1 else lambda: "pending"
    with pytest.raises(RuntimeError, match="collective Python local preparation failed"):
        lane.collective_prepare(callback)
    assert lane.agree_control_bytes(b"schema", b"still-collective")


def test_agreement_is_exact_across_chunks_schema_and_payload(lane):
    payload = (bytes(range(256)) * 193) + struct.pack("<q", 2**60 + 17)
    assert lane.agree_control_bytes(b"schema\x00\xff", payload)
    different = payload[:-1] + b"\x11" if int(lane.rank) == int(lane.size) - 1 else payload
    assert lane.agree_control_bytes(b"schema", different) is (int(lane.size) == 1)
    schema = b"schema2" if int(lane.rank) == int(lane.size) - 1 else b"schema"
    assert lane.agree_control_bytes(schema, payload) is (int(lane.size) == 1)
    assert lane.agree_control_bytes(b"", b"")


@pytest.mark.parametrize("bad_member", ("schema", "payload"))
def test_one_rank_invalid_bytes_argument_converges(lane, bad_member):
    args = {"schema": b"schema", "payload": b"payload"}
    if int(lane.rank) == int(lane.size) - 1:
        args[bad_member] = bytearray(args[bad_member])
    with pytest.raises(RuntimeError, match="collective Python local preparation failed"):
        lane.agree_control_bytes(**args)
    assert lane.agree_control_bytes(b"schema", b"recovered")


def test_allgather_cast_failure_converges_and_preserves_followup_order(lane):
    payload = bytearray(b"invalid") if int(lane.rank) == int(lane.size) - 1 else b"valid"
    with pytest.raises(RuntimeError, match="collective Python local preparation failed"):
        lane.allgather_bytes(payload)
    payload = b"" if int(lane.rank) == 0 else struct.pack("<q", 2**60 + int(lane.rank))
    expected = tuple(b"" if rank == 0 else struct.pack("<q", 2**60 + rank)
                     for rank in range(int(lane.size)))
    assert lane.allgather_bytes(payload) == expected


def test_closed_observer_refuses_new_prepare_without_running_callback():
    lane = _pops.mpi_world().duplicate_observer_lane("pytest-v10-closed-preparation")
    lane.close_collectively()
    ran = []
    with pytest.raises(RuntimeError, match="closed"):
        lane.collective_prepare(lambda: ran.append(True))
    assert ran == []
    with pytest.raises(RuntimeError, match="closed"):
        lane.agree_control_bytes(b"schema", b"payload")
    with pytest.raises(RuntimeError, match="closed"):
        lane.allgather_bytes(b"payload")
