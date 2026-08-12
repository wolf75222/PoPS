"""Typed authoring contract for the bounded cell-local AMR execution route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CellLocalTimeContract:
    """Exact integer clock selected for prepared cell-local AMR execution.

    ``rung`` is the authored finest-level base. The native provider derives one homogeneous rung
    per level-group from integral power-of-two temporal ratios, so every level executes one
    Forward-Euler batch per hierarchy window. The contract remains explicit rather than inferred
    from ``dt`` so cache identity, checkpoint qualification and unsupported-route diagnostics all
    observe the same authority. Heterogeneous per-cell rungs and non-dyadic relations remain
    outside this bounded envelope.
    """

    tick_denominator: int
    rung: int = 0

    def __post_init__(self) -> None:
        if type(self.tick_denominator) is not int or self.tick_denominator <= 0:
            raise ValueError("Program.cell_local_time tick_denominator must be a positive int")
        if type(self.rung) is not int or self.rung < 0 or self.rung > 30:
            raise ValueError("Program.cell_local_time rung must be an int in [0, 30]")

    def to_data(self) -> dict[str, int]:
        return {
            "schema_version": 1,
            "tick_denominator": self.tick_denominator,
            "rung": self.rung,
        }


def require_cell_local_time_contract(value: Any) -> CellLocalTimeContract:
    if type(value) is not CellLocalTimeContract:
        raise TypeError("Program carries an invalid cell-local time contract")
    return value


__all__ = ["CellLocalTimeContract", "require_cell_local_time_contract"]
