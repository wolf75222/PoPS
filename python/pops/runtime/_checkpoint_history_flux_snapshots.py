"""Rank-qualified immutable history-flux snapshots for strict AMR restart."""

from __future__ import annotations

from collections.abc import MutableMapping
from hashlib import sha256
from typing import Any, cast

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
    canonicalize: Any,
    payload: MutableMapping[str, Any],
) -> None:
    """Gather and compact rank-owned shards into one rank-independent NPZ image."""
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
        if not callable(canonicalize):
            raise TypeError("history-flux snapshot canonicalizer must be callable")
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
    canonical = b""
    digest = None
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
        canonical = canonicalize(list(images), topology.size) if present else b""
        if type(canonical) is not bytes:
            raise TypeError("history-flux snapshot canonicalizer must return exact bytes")
        if present and not canonical:
            raise RuntimeError("history-flux snapshot canonicalizer dropped live metadata")
        if len(canonical) > capacity:
            raise ValueError("canonical history-flux snapshot exceeds its artifact capacity")
        digest = sha256(canonical).hexdigest()
    except BaseException as exc:
        error = exc
    rows = consensus(
        topology,
        "history-flux snapshot checkpoint canonicalization",
        error=error,
        value=digest,
    )
    if any(row["value"] != digest for row in rows):
        raise RuntimeError("canonical history-flux snapshot differs across ranks")
    # Successful consensus has rethrown every failed exact-bytes validation above.
    canonical_image = cast(bytes, canonical)
    error = None
    try:
        state = np.frombuffer(canonical_image, dtype=np.uint8).copy()
        offsets = np.asarray((0, len(canonical_image)), dtype=np.int64)
        payload[SNAPSHOT_STATE_KEY] = state
        payload[SNAPSHOT_OFFSETS_KEY] = offsets
    except BaseException as exc:
        error = exc
    consensus(topology, "history-flux snapshot checkpoint serialization", error=error)


def stage_history_flux_snapshots(sim: Any, shards: tuple[bytes, ...] | None) -> None:
    """Replace archive authority inside the restart transaction, including legacy absence."""
    restore = getattr(sim, "restore_program_history_flux_snapshots", None)
    if not callable(restore):
        raise TypeError("restart: AMR engine lacks immutable history-flux snapshot restore")
    # Legacy absence authorizes an empty archive, never reuse of pre-restart live samples.
    # A FLX4 descriptor without its archive must then fail; rollback owns the prior live map.
    restore(list(shards) if shards is not None else [b""], 1)


def prepare_history_flux_snapshots(
    payload: Any,
    *,
    shard_capacity: object,
) -> tuple[bytes, ...] | None:
    """Validate one canonical rank-independent archive before restart mutation."""
    capacity = _exact_capacity(shard_capacity)
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
    if len(offsets) != 2 or offsets[0] != 0 or offsets[1] != len(raw):
        raise ValueError("restart history-flux snapshot offsets must describe one canonical image")
    if len(raw) > capacity:
        raise ValueError("restart history-flux snapshot shard exceeds its artifact capacity")
    return (raw.tobytes(),)


__all__ = [
    "HISTORY_FLUX_SNAPSHOT_CHECKPOINT_KEYS",
    "SNAPSHOT_OFFSETS_KEY",
    "SNAPSHOT_STATE_KEY",
    "capture_history_flux_snapshots",
    "prepare_history_flux_snapshots",
    "stage_history_flux_snapshots",
]
