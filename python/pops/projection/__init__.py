"""Typed initial-data projection policies.

Projection descriptors state how continuous data becomes stored values.  Their accuracy and halo
contracts are intrinsic; users select a policy and never repeat an ``order`` integer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar


def _state_space(state: Any) -> Any:
    declaration = getattr(state, "declaration_ref", None)
    declaration = state if declaration is None else declaration
    space = getattr(declaration, "space", None)
    if space is None:
        raise TypeError("initial projection requires a state carrying a typed StateSpace")
    return space


@dataclass(frozen=True, slots=True)
class ConservativeCellAverage:
    """Second-order conservative projection onto cell averages.

    Formal order, centering and bootstrap phase ordering belong to this brick.  They are not public
    constructor arguments and therefore cannot disagree with AMR transfer or stencil planning.
    """

    native_route: ClassVar[str] = "conservative_cell_average"
    formal_order: ClassVar[int] = 2
    ghost_depth: ClassVar[tuple[int, ...]] = (1,)
    bootstrap_phases: ClassVar[tuple[str, ...]] = (
        "transfer", "projection", "constraint")
    __pops_ir_immutable__ = True

    def validate_for(self, state: Any, value: Any) -> bool:
        space = _state_space(state)
        if getattr(space, "representation", None) != "conservative":
            raise ValueError(
                "ConservativeCellAverage requires a conservative state representation")
        if getattr(space, "centering", None) != "cell":
            raise ValueError("ConservativeCellAverage requires a cell-centred state")
        return True

    def initial_projection_options(self) -> dict[str, Any]:
        return {"projection": self.to_data()}

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "projection": self.native_route,
            "formal_order": self.formal_order,
            "ghost_depth": list(self.ghost_depth),
        }

    canonical_identity = to_data
    inspect = to_data


__all__ = ["ConservativeCellAverage"]
