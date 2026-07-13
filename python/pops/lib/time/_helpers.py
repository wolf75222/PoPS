"""Small shared helpers for canonical time Program factories."""
from __future__ import annotations

from typing import Any


def _block_label(state: Any) -> str:
    """Return the qualified display label retained by one TimeState."""
    from pops.time.references import block_name

    return block_name(state.block)


def _stage_point(
    program: Any,
    name: str,
    offset: Any = 0,
    *,
    partitions: Any = None,
) -> Any:
    """Build one exact stage coordinate on the Program clock."""
    from pops.time.points import StagePoint, TimePoint

    coordinates = {"main": offset} if partitions is None else partitions
    if not isinstance(coordinates, dict) or not coordinates:
        raise TypeError("stage partitions must be a non-empty dict")
    return StagePoint(
        name,
        {
            partition: TimePoint(program.clock, coordinate)
            for partition, coordinate in coordinates.items()
        },
    )


def _op_space_arity(program: Any, handle: Any) -> int:
    """Return the declared number of State/Field inputs of one exact operator."""
    from pops.time.operator_resolution import resolve_operator_handle

    operator = resolve_operator_handle(
        program, handle, where="pops.lib.time operator")
    return sum(
        1
        for item in operator.signature.inputs
        if getattr(item, "kind", None) in ("state", "field")
    )


__all__ = ["_block_label", "_op_space_arity", "_stage_point"]
