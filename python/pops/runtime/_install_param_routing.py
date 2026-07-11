"""Qualified BindSchema routing for native runtime parameter carriers.

The install layer receives a complete ``{canonical ParamHandle: value}`` map
from ``BindSchema.resolve``.  It never broadcasts a local name and never invents
a ``0.0`` fallback.  These pure helpers only project that resolved plan into the
flat vectors required by the existing C++ block/program ABI.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def _require_schema(schema: Any) -> Any:
    from pops.model.bind_schema import BindSchema

    if not isinstance(schema, BindSchema):
        raise TypeError("parameter routing requires the artifact's BindSchema")
    return schema


def _slot_for_block(schema: Any, block_name: str, local_id: str) -> Any:
    candidates = [
        slot
        for slot in schema.slots
        if slot.handle.is_instance
        and slot.handle.block_ref.local_id == block_name
        and slot.handle.local_id == local_id
        and (
            slot.kind == "runtime"
            or (slot.kind == "derived" and slot.declaration["phase"] == "bind")
        )
    ]
    if len(candidates) != 1:
        raise ValueError(
            "BindSchema/native artifact drift for block %r parameter %r: expected one "
            "runtime slot, found %d" % (block_name, local_id, len(candidates))
        )
    return candidates[0]


def _resolved_value(resolved: Mapping[Any, Any], slot: Any) -> float:
    if slot.handle not in resolved:
        raise ValueError(
            "resolved BindSchema is missing install value for %s" % slot.qid
        )
    # The current native ABI is a pops::Real array. Integer declarations therefore need an explicit
    # exact-representability gate; silently rounding a valid Python integer would violate its dtype.
    raw = resolved[slot.handle]
    try:
        lowered = float(raw)
    except OverflowError:
        raise ValueError(
            "value %r for %s is not representable by the native pops::Real parameter ABI"
            % (raw, slot.qid)
        ) from None
    if not math.isfinite(lowered):
        raise ValueError(
            "value %r for %s is not finite in the native pops::Real parameter ABI"
            % (raw, slot.qid)
        )
    if slot.dtype == "Integer" and int(lowered) != raw:
        raise ValueError(
            "Integer value %r for %s is not exactly representable by the native pops::Real "
            "parameter ABI" % (raw, slot.qid)
        )
    return lowered


def route_block_params(
    resolved_models: Any,
    schema: Any,
    resolved: Any,
) -> dict[str, list[float]]:
    """Build complete per-instance vectors in each compiled model's slot order."""
    schema = _require_schema(schema)
    if not isinstance(resolved, Mapping):
        raise TypeError("resolved parameter values must be a ParamHandle mapping")
    per_block: dict[str, list[float]] = {}
    for block_name, model in resolved_models.items():
        runtime_names = list(getattr(model, "runtime_param_names", ()) or ())
        if not runtime_names:
            continue
        per_block[block_name] = [
            _resolved_value(
                resolved,
                _slot_for_block(schema, block_name, local_id),
            )
            for local_id in runtime_names
        ]
    return per_block


def route_program_params(
    compiled: Any,
    schema: Any,
    resolved: Any,
) -> dict[int, list[float]]:
    """Build complete per-Program-block vectors in emitted within-block order."""
    schema = _require_schema(schema)
    if not isinstance(resolved, Mapping):
        raise TypeError("resolved parameter values must be a ParamHandle mapping")
    program = getattr(compiled, "program", None)
    if program is None:
        return {}
    from pops.time.references import block_name

    blocks = {index: reference for reference, index in program._block_indices().items()}
    vectors: dict[int, list[float | None]] = {}
    captured = getattr(compiled, "program_param_routes", None)
    if captured is None:
        raise ValueError(
            "compiled Program carries no immutable program_param_routes metadata; rebuild it "
            "through pops.compile(...) before binding (bind never re-enters model/codegen analysis)"
        )
    entries = tuple(captured)
    for block_index, local_id, within_index, _neutral in entries:
        reference = blocks.get(block_index)
        if reference is None:
            raise ValueError(
                "compiled Program parameter route references unknown block index %d"
                % block_index
            )
        owner_name = block_name(reference)
        slot = _slot_for_block(schema, owner_name, local_id)
        vector = vectors.setdefault(block_index, [])
        if len(vector) <= within_index:
            vector.extend([None] * (within_index + 1 - len(vector)))
        value = _resolved_value(resolved, slot)
        existing = vector[within_index]
        if existing is not None and existing != value:
            raise ValueError(
                "compiled Program assigns conflicting values to block %r slot %d"
                % (owner_name, within_index)
            )
        vector[within_index] = value

    result: dict[int, list[float]] = {}
    for block_index, vector in vectors.items():
        if any(value is None for value in vector):
            raise ValueError(
                "compiled Program parameter vector for block index %d has an ABI hole"
                % block_index
            )
        result[block_index] = [float(value) for value in vector]
    return result


__all__ = ["route_block_params", "route_program_params"]
