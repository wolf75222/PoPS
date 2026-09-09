"""Rank-qualified immutable history-flux snapshots for strict AMR restart."""

from __future__ import annotations

from collections.abc import MutableMapping
import sys
from typing import Any

import numpy as np


SNAPSHOT_STATE_KEY = "program_history_flux_snapshot_state"
SNAPSHOT_OFFSETS_KEY = "program_history_flux_snapshot_offsets"
HISTORY_FLUX_SNAPSHOT_CHECKPOINT_KEYS = frozenset((SNAPSHOT_STATE_KEY, SNAPSHOT_OFFSETS_KEY))


def _exact_capacity(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError("history-flux snapshot capacity must be a non-negative exact integer")
    return value


def capture_history_flux_snapshots(
    topology: Any,
    local_shard: bytes,
    shard_capacity: int,
    payload: MutableMapping[str, Any],
) -> None:
    """Gather one immutable shard per source rank into two canonical NPZ arrays."""
    from pops._native_collectives import allgather_bytes
    from pops.output._checkpoint_collective import consensus

    capacity = None
    present = None
    error = None
    try:
        capacity = _exact_capacity(shard_capacity)
        if type(local_shard) is not bytes:
            raise TypeError("history-flux snapshot shard must be exact bytes")
        if len(local_shard) > capacity:
            raise ValueError("history-flux snapshot shard exceeds its artifact capacity")
        present = bool(local_shard)
    except BaseException as exc:
        error = exc
    rows = consensus(
        topology,
        "history-flux snapshot checkpoint preflight",
        error=error,
        value={"capacity": capacity, "present": present},
    )
    local_contract = rows[0]["value"]
    if any(row["value"] != local_contract for row in rows[1:]):
        raise RuntimeError("history-flux snapshot checkpoint plans differ across ranks")
    capacity = local_contract["capacity"]
    present = local_contract["present"]

    images = (
        allgather_bytes(topology.communicator, local_shard)
        if topology.distributed and present
        else (local_shard,) * topology.size
    )
    error = None
    try:
        if len(images) != topology.size or any(type(image) is not bytes for image in images):
            raise RuntimeError("history-flux snapshot gather returned an invalid rank envelope")
        if local_shard and any(not image for image in images):
            raise RuntimeError("history-flux snapshot metadata presence differs across ranks")
        if not local_shard and any(images):
            raise RuntimeError("empty history-flux snapshot gather is not canonical")
        if any(len(image) > capacity for image in images):
            raise ValueError("history-flux snapshot shard exceeds its artifact capacity")
        if capacity and topology.size > sys.maxsize // capacity:
            raise OverflowError("history-flux snapshot archive capacity exceeds addressable memory")
        offsets = [0]
        for image in images:
            if len(image) > sys.maxsize - offsets[-1]:
                raise OverflowError("history-flux snapshot archive exceeds addressable memory")
            offsets.append(offsets[-1] + len(image))
        payload[SNAPSHOT_STATE_KEY] = np.frombuffer(b"".join(images), dtype=np.uint8).copy()
        payload[SNAPSHOT_OFFSETS_KEY] = np.asarray(offsets, dtype=np.int64)
    except BaseException as exc:
        error = exc
    consensus(topology, "history-flux snapshot checkpoint serialization", error=error)


def prepare_history_flux_snapshots(
    payload: Any,
    *,
    checkpoint_ranks: int,
    shard_capacity: int,
) -> tuple[bytes, ...] | None:
    """Validate and split one sealed source-rank archive before restart mutation."""
    capacity = _exact_capacity(shard_capacity)
    if isinstance(checkpoint_ranks, bool) or not isinstance(checkpoint_ranks, int):
        raise TypeError("history-flux snapshot source rank count must be an exact integer")
    if checkpoint_ranks < 1:
        raise ValueError("history-flux snapshot source rank count must be positive")
    present = HISTORY_FLUX_SNAPSHOT_CHECKPOINT_KEYS.intersection(payload)
    if not present:
        return None
    if present != HISTORY_FLUX_SNAPSHOT_CHECKPOINT_KEYS:
        raise ValueError("restart has a partial history-flux snapshot archive")

    raw = np.asarray(payload[SNAPSHOT_STATE_KEY])
    offsets = np.asarray(payload[SNAPSHOT_OFFSETS_KEY])
    if raw.dtype != np.dtype(np.uint8) or raw.ndim != 1:
        raise ValueError("restart history-flux snapshot state must be one uint8 vector")
    if offsets.dtype != np.dtype(np.int64) or offsets.ndim != 1:
        raise ValueError("restart history-flux snapshot offsets must be one int64 vector")
    if len(offsets) != checkpoint_ranks + 1 or offsets[0] != 0 or offsets[-1] != len(raw):
        raise ValueError("restart history-flux snapshot offsets do not cover every source rank")
    if len(raw):
        if np.any(offsets[1:] <= offsets[:-1]):
            raise ValueError("restart history-flux snapshot shards must be non-empty")
    elif np.any(offsets):
        raise ValueError("restart empty history-flux snapshot offsets are not canonical")

    widths = offsets[1:] - offsets[:-1]
    if np.any(widths > capacity):
        raise ValueError("restart history-flux snapshot shard exceeds its artifact capacity")
    shards = tuple(
        raw[int(first) : int(last)].tobytes()
        for first, last in zip(offsets[:-1], offsets[1:], strict=True)
    )
    if any(shards) and any(not shard for shard in shards):
        raise ValueError("restart history-flux snapshot metadata presence differs across ranks")
    return shards


__all__ = [
    "HISTORY_FLUX_SNAPSHOT_CHECKPOINT_KEYS",
    "SNAPSHOT_OFFSETS_KEY",
    "SNAPSHOT_STATE_KEY",
    "capture_history_flux_snapshots",
    "prepare_history_flux_snapshots",
]
