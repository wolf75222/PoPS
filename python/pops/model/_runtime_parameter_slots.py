"""Slot indices from compiled Program runtime parameter metadata."""

from collections.abc import Iterable


def index_program_parameter_slots(
    parameter_routes: Iterable[tuple[int, str, int, float]],
) -> dict[tuple[int, str], int]:
    """Index the slot of each parameter by Program block and parameter name."""
    return {
        (block, name): slot
        for block, name, slot, _default in parameter_routes
    }
