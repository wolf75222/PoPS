"""Strict Python framing around the native MPI byte collectives."""
from __future__ import annotations

from typing import Any

import pytest

import pops._native_collectives as collectives


class _GatherWorld:
    def __init__(self, *, rank: int, size: int, response: Any) -> None:
        self.rank = rank
        self.size = size
        self.response = response
        self.calls: list[tuple[bytes, int]] = []

    def gather_bytes(self, payload: bytes, root: int) -> Any:
        self.calls.append((payload, root))
        if self.response == "local-frame":
            return (payload,)
        return self.response


def _install_world(monkeypatch: pytest.MonkeyPatch, world: _GatherWorld) -> _GatherWorld:
    monkeypatch.setattr(
        collectives, "require_communicator", lambda communicator: communicator)
    return world


def test_gather_bytes_frames_and_decodes_contiguous_rank_order_on_root(monkeypatch):
    world = _install_world(monkeypatch, _GatherWorld(
        rank=1,
        size=2,
        response=(b"\x00rank-zero", b"\x00rank-one"),
    ))

    assert collectives.gather_bytes(world, b"local", root=1) == (
        b"rank-zero", b"rank-one",
    )
    assert world.calls == [(b"\x00local", 1)]


def test_gather_bytes_returns_none_only_for_a_native_non_root(monkeypatch):
    world = _install_world(monkeypatch, _GatherWorld(rank=0, size=2, response=None))

    assert collectives.gather_bytes(world, b"local", root=1) is None
    assert world.calls == [(b"\x00local", 1)]


def test_gather_bytes_enters_collective_with_an_exact_local_error_frame(monkeypatch):
    world = _install_world(monkeypatch, _GatherWorld(
        rank=0,
        size=1,
        response="local-frame",
    ))

    with pytest.raises(RuntimeError, match="native byte gather failed: rank 0: TypeError"):
        collectives.gather_bytes(world, bytearray(b"not-exact-bytes"))

    assert len(world.calls) == 1
    assert world.calls[0][0].startswith(b"\x01TypeError:")


def test_gather_bytes_reports_every_invalid_root_frame(monkeypatch):
    world = _install_world(monkeypatch, _GatherWorld(
        rank=0,
        size=2,
        response=(b"", b"\x01ValueError: rejected"),
    ))

    with pytest.raises(RuntimeError) as error:
        collectives.gather_bytes(world, b"local")

    message = str(error.value)
    assert "rank 0: native gather returned a malformed native frame" in message
    assert "rank 1: ValueError: rejected" in message


@pytest.mark.parametrize("response", [None, [], (b"\x00only-one",)])
def test_gather_bytes_rejects_a_malformed_root_payload_set(monkeypatch, response):
    world = _install_world(monkeypatch, _GatherWorld(
        rank=0,
        size=2,
        response=response,
    ))

    with pytest.raises(RuntimeError, match="invalid root payload set"):
        collectives.gather_bytes(world, b"local")


def test_gather_bytes_rejects_payloads_returned_to_a_non_root(monkeypatch):
    world = _install_world(monkeypatch, _GatherWorld(
        rank=0,
        size=2,
        response=(b"\x00rank-zero", b"\x00rank-one"),
    ))

    with pytest.raises(RuntimeError, match="payloads on a non-root rank"):
        collectives.gather_bytes(world, b"local", root=1)
