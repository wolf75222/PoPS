"""Rank-qualified accepted exchange checkpoint images; native code owns record validation.

Compatibility is fail-closed: archives written before these required members existed cannot
restore an exact accepted mailbox and must be rejected in preflight. An absent mailbox is never
inferred empty; the producer artifact/bind identity and continuation plan must also match. This
format addition makes no backward bit-identity claim for historical archives.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

_EXCHANGES = "program_exchange_state"
_OFFSETS = "program_exchange_offsets"
_TRANSITIONS = "continuation_transition_plan"


def capture_checkpoint_continuation(owner, payload):
    from pops.output._checkpoint_collective import checkpoint_topology, consensus
    from pops._native_collectives import allgather_bytes
    from pops.runtime._continuation_transitions import ContinuationTransitionPlan

    topology = checkpoint_topology(owner)
    error = None
    local = b""
    try:
        plan = getattr(owner, "_continuation_transition_plan", None)
        if type(plan) is not ContinuationTransitionPlan:
            raise RuntimeError("checkpoint lacks resolved continuation obligations")
        plan.require("restart")
        local = owner._s._checkpoint_program_exchanges()
        if type(local) is not bytes or not local.startswith(b"POPSEX01"):
            raise RuntimeError("checkpoint accepted exchange mailbox is not an exact native image")
    except BaseException as exc:
        error = exc
    consensus(topology, "accepted exchange checkpoint preparation", error=error)
    images = allgather_bytes(topology.communicator, local) if topology.distributed else (local,)
    error = None
    try:
        offsets = [0]
        for image in images:
            offsets.append(offsets[-1] + len(image))
        bound = getattr(owner, "_checkpoint_exchange_byte_capacity", None)
        if type(bound) is not int or offsets[-1] > bound:
            raise RuntimeError("accepted exchange checkpoint exceeds its resolved byte capacity")
        payload[_EXCHANGES] = np.frombuffer(b"".join(images), dtype=np.uint8).copy()
        payload[_OFFSETS] = np.asarray(offsets, dtype=np.int64)
        payload[_TRANSITIONS] = np.asarray(plan._json)
    except BaseException as exc:
        error = exc
    # All per-rank allocations after the gather must converge before any caller's next
    # collective. A partial local payload is discarded by the outer capture transaction.
    consensus(topology, "accepted exchange checkpoint serialization", error=error)


def prepare_checkpoint_continuation(owner, payload):
    from pops.output._checkpoint_collective import checkpoint_topology
    from pops.runtime._continuation_transitions import ContinuationTransitionPlan

    plan = getattr(owner, "_continuation_transition_plan", None)
    if type(plan) is not ContinuationTransitionPlan:
        raise RuntimeError("restart lacks resolved continuation obligations")
    if any(key not in payload for key in (_EXCHANGES, _OFFSETS, _TRANSITIONS)):
        raise ValueError("restart omits required continuation/exchange policy or accepted state")
    contract = np.asarray(payload[_TRANSITIONS])
    if contract.ndim != 0 or contract.dtype.kind != "U" or str(contract) != plan._json:
        raise ValueError("restart continuation policies differ from the resolved retained objects")
    plan.require("restart")
    raw = np.asarray(payload[_EXCHANGES])
    offsets = np.asarray(payload[_OFFSETS])
    if raw.dtype != np.dtype(np.uint8) or raw.ndim != 1 or offsets.dtype != np.dtype(np.int64) \
            or offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 \
            or offsets[-1] != len(raw) or np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("restart accepted exchange image has invalid ranked byte offsets")
    bound = getattr(owner, "_checkpoint_exchange_byte_capacity", None)
    if type(bound) is not int or len(raw) > bound:
        raise ValueError("restart accepted exchange image exceeds its resolved byte capacity")
    topology = checkpoint_topology(owner)
    images = tuple(raw[int(first):int(last)].tobytes()
                   for first, last in zip(offsets[:-1], offsets[1:], strict=True))
    if len(images) != topology.size and any(image != images[0] for image in images[1:]):
        raise ValueError("restart cannot remap rank-dependent accepted exchange mailboxes")
    selected = images[topology.rank] if len(images) == topology.size else images[0]
    owner._s._validate_checkpoint_program_exchanges(selected)
    return selected


def exchange_checkpoint_byte_capacity(program, *, cells, dimension, rank_capacity, resolved_plan):
    """Conservative finite bound over emitted occurrences, native faces and logical invocations.

    No checkpoint data controls this admission limit. Each affine accepted use can emit at most
    one record per resolved occurrence and oriented cell face. Generated exchange publication runs
    once after the selected affine update; nonlinear residual iterations do not emit records.
    Runtime frame fields have fixed-width numeric spellings.
    """
    values = tuple(program._values)
    serial = program.to_data()
    strings = []

    def visit(value):
        if isinstance(value, Mapping):
            for key, item in value.items():
                strings.append(str(key))
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            strings.append(value)

    visit(serial)
    occurrence_count = 0
    for block in resolved_plan.blocks:
        operations = block.resolved_operations
        if operations is not None:
            occurrence_count += sum(len(row.occurrences) for row in operations.evaluations)
            visit(operations.to_data())
    # Joint inventories/native outputs can stage exchanges without a face construction. Each
    # authored output is bounded by its own serialized row and each accepted commit is one use.
    occurrence_count = max(1, occurrence_count, len(values))
    clock_ticks = 1
    for clock in program.temporal_manifest()["clocks"]:
        ticks = clock.get("ticks_per_macro", 1)
        if isinstance(ticks, int):
            clock_ticks = max(clock_ticks, ticks)
    record_count = max(1, len(values)) * occurrence_count * max(1, sum(cells)) \
        * (2 * dimension) * clock_ticks
    max_text = max((len(text.encode("utf-8")) for text in strings), default=1)
    # Four identifiers, including the runtime-qualified context and cell/axis/side quadrature.
    record_bytes = 72 + 4 * max_text + 512
    capacity = rank_capacity * (16 + record_count * record_bytes) + 8 * (rank_capacity + 1)
    if capacity > (1 << 63) - 1:
        raise OverflowError("resolved accepted exchange checkpoint capacity exceeds int64")
    return capacity
