"""Rank-owned native Program diagnostic archive @1, including legacy absence.

The native codec preserves opaque names and Real bits. No rank equality is imposed
on the table: raw records may be local and compiled reductions may be collective.
"""

from __future__ import annotations

import numpy as np

_STATE = "program_diagnostics_state"
_OFFSETS = "program_diagnostics_offsets"
PROGRAM_DIAGNOSTIC_CHECKPOINT_KEYS = frozenset((_STATE, _OFFSETS))


def _exact_native_image(image, *, rank=None, ranks=None):
    if type(image) is not bytes or len(image) < 40 or image[:8] != b"POPSDIA1":
        raise ValueError("Program diagnostic checkpoint is not an exact native @1 image")
    if rank is not None and (
        int.from_bytes(image[16:24], "little") != rank
        or int.from_bytes(image[24:32], "little") != ranks
    ):
        raise ValueError("Program diagnostic checkpoint belongs to another rank authority")
    return image


def capture_checkpoint_program_diagnostics(owner, payload):
    from pops.output._checkpoint_collective import checkpoint_topology, consensus
    from pops._native_collectives import allgather_bytes

    topology = checkpoint_topology(owner)
    error = None
    local = b""
    try:
        from pops.runtime._checkpoint_resource_budget import (
            _require_checkpoint_diagnostic_authority,
        )

        _require_checkpoint_diagnostic_authority(owner)
        local = _exact_native_image(
            owner._s._checkpoint_program_diagnostics(), rank=topology.rank, ranks=topology.size
        )
    except BaseException as exc:
        error = exc
    consensus(topology, "accepted Program diagnostic preparation", error=error)
    images = allgather_bytes(topology.communicator, local) if topology.distributed else (local,)
    error = None
    try:
        if len(images) != topology.size:
            raise ValueError("Program diagnostic checkpoint rank image count changed")
        offsets = [0]
        for rank, image in enumerate(images):
            offsets.append(
                offsets[-1] + len(_exact_native_image(image, rank=rank, ranks=topology.size))
            )
        bound = getattr(owner, "_checkpoint_program_diagnostic_byte_capacity", None)
        capacity = getattr(owner, "_checkpoint_program_diagnostic_capacity_per_rank", None)
        if (
            type(capacity) is not int
            or any(len(image) > capacity for image in images)
            or type(bound) is not int
            or offsets[-1] > bound
        ):
            raise ValueError(
                "Program diagnostic checkpoint exceeds its chosen resource capacity; call runtime.configure_checkpoint_diagnostics(capacity_per_rank=larger_bytes) collectively before capture/restart"
            )
        raw = np.frombuffer(b"".join(images), dtype=np.uint8).copy()
        index = np.asarray(offsets, dtype=np.int64)
        payload[_STATE], payload[_OFFSETS] = raw, index
    except BaseException as exc:
        error = exc
    consensus(topology, "accepted Program diagnostic serialization", error=error)


def validate_checkpoint_program_diagnostic_arrays(payload):
    present = PROGRAM_DIAGNOSTIC_CHECKPOINT_KEYS.intersection(payload)
    if not present:
        return False  # Explicit legacy absence; restart replaces with empty, not stale data.
    if present != PROGRAM_DIAGNOSTIC_CHECKPOINT_KEYS:
        raise ValueError("Program diagnostic checkpoint image/offsets are incomplete")
    raw, index = np.asarray(payload[_STATE]), np.asarray(payload[_OFFSETS])
    if (
        raw.dtype != np.dtype(np.uint8)
        or raw.ndim != 1
        or index.dtype != np.dtype(np.int64)
        or index.ndim != 1
        or len(index) < 2
        or index[0] != 0
        or index[-1] != len(raw)
        or np.any(index[1:] <= index[:-1])
    ):
        raise ValueError("Program diagnostic checkpoint rank image arrays are malformed")
    for rank, (lo, hi) in enumerate(zip(index[:-1], index[1:], strict=True)):
        _exact_native_image(raw[int(lo) : int(hi)].tobytes(), rank=rank, ranks=len(index) - 1)
    return True


def prepare_checkpoint_program_diagnostics(owner, payload):
    from pops.output._checkpoint_collective import checkpoint_topology

    from pops.runtime._checkpoint_resource_budget import _require_checkpoint_diagnostic_authority

    _require_checkpoint_diagnostic_authority(owner)
    topology = checkpoint_topology(owner)
    present = validate_checkpoint_program_diagnostic_arrays(payload)
    image = b""
    if present:
        raw, index = payload[_STATE], payload[_OFFSETS]
        if len(index) != topology.size + 1:
            raise ValueError("Program diagnostic checkpoint belongs to another rank count")
        bound = getattr(owner, "_checkpoint_program_diagnostic_byte_capacity", None)
        capacity = getattr(owner, "_checkpoint_program_diagnostic_capacity_per_rank", None)
        if (
            type(capacity) is not int
            or any(int(hi - lo) > capacity for lo, hi in zip(index[:-1], index[1:], strict=True))
            or type(bound) is not int
            or len(raw) > bound
        ):
            raise ValueError(
                "Program diagnostic checkpoint exceeds its chosen resource capacity; call runtime.configure_checkpoint_diagnostics(capacity_per_rank=larger_bytes) collectively before capture/restart"
            )
        image = np.asarray(raw)[int(index[topology.rank]) : int(index[topology.rank + 1])].tobytes()
    owner._s._validate_checkpoint_program_diagnostics(image)
    return image
