"""Bound resources for immutable native parameter incarnations.

The capacity bounds one carrier, including both payloads and its descriptor. It is
not a bound on all snapshots a caller may retain. The initial total counts one
carrier per installed model/Program block, not one per shallow brick/view alias.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def validate_parameter_byte_capacity(value: Any) -> int:
    if type(value) is not int or value < 0 or value > (1 << 64) - 1:
        raise ValueError("runtime_parameter_byte_capacity must be an unsigned 64-bit integer")
    return value


def _cardinalities(artifact: Any) -> dict[str, int]:
    from pops.runtime._install_param_routing import _slot_for_block

    schema = artifact.bind_schema
    counts: dict[str, int] = {}
    for block in artifact.blocks:
        names = tuple(getattr(block.model, "runtime_param_names", ()) or ())
        if len(names) != len(set(names)):
            raise ValueError("compiled model runtime parameter names are not unique")
        for name in names:
            _slot_for_block(schema, block.name, name)
        if names:
            counts["model:" + block.name] = len(names)
    for row in artifact.layout_programs:
        program = row.program
        if getattr(program, "program", None) is None:
            continue
        routes = getattr(program, "program_block_routes", None)
        parameter_routes = getattr(program, "program_param_routes", None)
        if routes is None or parameter_routes is None:
            raise ValueError("compiled Program lacks immutable runtime parameter routing metadata")
        blocks = dict(routes)
        if len(blocks) != len(routes):
            raise ValueError("compiled Program block route is duplicated")
        slots: dict[int, set[int]] = {}
        for block, name, index, _default in parameter_routes:
            if type(block) is not int or type(index) is not int or index < 0 or block not in blocks:
                raise ValueError("compiled Program parameter route is invalid")
            _slot_for_block(schema, blocks[block], name)
            entries = slots.setdefault(block, set())
            if index in entries:
                raise ValueError("compiled Program parameter route is duplicated")
            entries.add(index)
        for block, entries in slots.items():
            if max(entries) + 1 != len(entries):
                raise ValueError("compiled Program parameter route has a hole")
            counts["program:" + row.layout_id + ":" + str(block)] = len(entries)
    return counts


def resolve_runtime_parameter_storage(
    artifact: Any,
    resources: Mapping[str, Any],
    *,
    storage_bytes: Callable[[int], int] | None = None,
) -> dict[str, Any]:
    counts = _cardinalities(artifact)
    if counts and storage_bytes is None:
        from pops._native_selector import selected_native_module

        storage_bytes = selected_native_module(required=True).runtime_parameter_storage_bytes
    sizes = {
        key: validate_parameter_byte_capacity(storage_bytes(count))
        for key, count in counts.items()
    } if storage_bytes is not None else {}
    required = max(sizes.values(), default=0)
    capacity = validate_parameter_byte_capacity(
        resources.get("runtime_parameter_byte_capacity", required)
    )
    if capacity < required:
        raise ValueError(
            "runtime parameter incarnation requires %d bytes, bound capacity is %d"
            % (required, capacity)
        )
    total = sum(sizes.values())
    if total > (1 << 64) - 1:
        raise OverflowError("initial runtime parameter storage byte count overflows uint64")
    return {
        "contract": "pops.runtime-parameter-storage@1",
        "scope": "immutable_incarnation",
        "byte_capacity": capacity,
        "initial_storage_bytes": total,
        "cardinalities": counts,
        "carrier_bytes": sizes,
    }
