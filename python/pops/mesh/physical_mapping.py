"""Explicit bounded physical maps, independent of native storage dimension."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Any
from pops._ir.quantity import PhysicalDimension, PhysicalSupport

@dataclass(frozen=True, slots=True)
class VelocityQuadrature:
    """Uniform cell-average integral with exact (upper-lower)/cells weights."""
    lower: Fraction
    upper: Fraction
    cells: int
    dimension: PhysicalDimension

    def __post_init__(self) -> None:
        if type(self.cells) is not int or self.cells < 1:
            raise ValueError("velocity quadrature cells must be a positive integer")
        if type(self.dimension) is not PhysicalDimension:
            raise TypeError("velocity quadrature requires explicit physical units")
        lower, upper = Fraction(self.lower), Fraction(self.upper)
        if upper <= lower:
            raise ValueError("velocity quadrature requires upper > lower")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def weight(self) -> Fraction:
        return (self.upper - self.lower) / self.cells

    def to_data(self) -> dict[str, Any]:
        pair = lambda value: [value.numerator, value.denominator]
        return {"rule": "uniform-cell-average-integral@1", "lower": pair(self.lower),
                "upper": pair(self.upper), "cells": self.cells,
                "weight": pair(self.weight), "dimension": self.dimension.to_data()}

@dataclass(frozen=True, slots=True)
class PhysicalSupportMap:
    """Directional 1x1v reduction or 1x pullback, never an inverse moment closure.

    Field native storage uses Dim=2 with a periodic, unit-measure singleton axis 1.
    Physical x occupies axis 0; velocity occupies axis 1 on the distribution.
    """
    source_support: PhysicalSupport
    target_support: PhysicalSupport
    quadrature: VelocityQuadrature | None = None

    def __post_init__(self) -> None:
        if any(type(item) is not PhysicalSupport for item in
               (self.source_support, self.target_support)):
            raise TypeError("physical map requires exact, explicit supports")
        source, target = self.source_support.coordinates, self.target_support.coordinates
        moment = self.quadrature is not None
        if moment and type(self.quadrature) is not VelocityQuadrature:
            raise TypeError("moment requires an exact VelocityQuadrature")
        phase, physical = (source, target) if moment else (target, source)
        if len(phase) != 2 or len(physical) != 1 or phase[:1] != physical:
            raise ValueError("physical map requires exact 1x1v <-> 1x support identities; "
                             "homonymous coordinates from different domains do not match")

    @property
    def operation_abi(self) -> int:
        return 2 if self.quadrature is not None else 3

    def validate_ports(self, source: Any, target: Any) -> None:
        spaces = tuple(getattr(item.subject, "space", None) for item in (source, target))
        if any(space is None for space in spaces):
            raise ValueError("physical map ports require typed quantity declarations")
        if tuple(space.support for space in spaces) != (self.source_support, self.target_support):
            raise ValueError("physical map source/target quantity support identities disagree")
        if any(space.representation not in ("conservative", "cell_average") or space.sampling != "cell_average"
               for space in spaces):
            raise ValueError("physical maps require explicit cell_average representation/sampling")
        if len(spaces[0].units) != len(spaces[1].units):
            raise ValueError("physical map component counts differ")
        for origin, destination in zip(spaces[0].units, spaces[1].units, strict=True):
            if origin is None or destination is None:
                raise ValueError("physical maps require explicit source and target units")
            powers = dict(origin.powers)
            if self.quadrature is not None:
                for name, power in self.quadrature.dimension.powers:
                    powers[name] = powers.get(name, Fraction(0)) + power
            if PhysicalDimension(tuple(powers.items())) != destination:
                raise ValueError("physical map units disagree with velocity integration/pullback")

    def to_data(self) -> dict[str, Any]:
        return {"kind": "velocity-moment@1" if self.quadrature else "physical-pullback@1",
                "source_support": self.source_support.to_data(),
                "target_support": self.target_support.to_data(),
                "quadrature": None if self.quadrature is None else self.quadrature.to_data(),
                "storage": {"native_dimension": 2, "physical_axis": 0, "velocity_axis": 1,
                            "field_hidden_cells": 1, "field_hidden_measure": 1,
                            "pullback": "constant-extension", "inverse_closure": False}}
