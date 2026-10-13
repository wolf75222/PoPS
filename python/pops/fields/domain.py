"""Immutable field-equation regions, independent of transport topology."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True, init=False)
class CellCentreBall:
    """Strict ball interior, cell-centre staircase, exterior potential exactly zero.

    Coordinates use the same physical coordinate binding as the FieldProblem's storage.
    Multigrid resamples the declaration on every coarse geometry; it never averages a mask.
    """

    __pops_ir_immutable__ = True

    centre: tuple[float, ...]
    radius: float

    def __init__(self, *, centre: tuple[float, ...], radius: float) -> None:
        if type(centre) is not tuple or len(centre) not in (1, 2, 3):
            raise TypeError("CellCentreBall.centre requires an exact coordinate tuple of rank 1, 2 or 3")
        def real(value: float, name: str) -> float:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise TypeError("CellCentreBall.%s must be a finite real" % name)
            return float(value)
        bound_centre = tuple(real(value, "centre") for value in centre)
        bound_radius = real(radius, "radius")
        if bound_radius <= 0.0:
            raise ValueError("CellCentreBall.radius must be strictly positive")
        object.__setattr__(self, "centre", bound_centre)
        object.__setattr__(self, "radius", bound_radius)

    def to_data(self) -> dict:
        return {
            "schema_version": 1,
            "region": "strict-cell-centre-ball",
            "centre": list(self.centre),
            "radius": self.radius,
            "sampling": "resample-each-mg-level",
            "exterior": {"kind": "dirichlet", "value": 0.0},
        }


__all__ = ["CellCentreBall"]
