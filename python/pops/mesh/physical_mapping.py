"""Typed support reductions and constant extensions, independent of storage rank."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Any
from pops._ir.quantity import PhysicalDimension, PhysicalSupport


def _pair(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


@dataclass(frozen=True, slots=True, init=False)
class AxisQuadrature:
    """An explicit tensor-product reduction factor on one native storage axis.

    ``lower``, ``upper`` and ``cells`` authenticate the uniform source cell domain.
    ``weights`` includes the integration measure and any authored moment weight.
    Signed and zero weights are allowed: positivity is not a property of every
    physical moment. Omitting weights selects the exact uniform cell measure.
    ``dimension`` is the physical dimension of each complete weight.
    """
    axis: int
    lower: Fraction
    upper: Fraction
    cells: int
    dimension: PhysicalDimension
    weights: tuple[Fraction, ...]

    def __init__(self, axis: int, lower: Fraction, upper: Fraction, cells: int,
                 dimension: PhysicalDimension,
                 weights: tuple[Fraction, ...] | None = None) -> None:
        if type(axis) is not int or axis not in (0, 1, 2):
            raise ValueError("quadrature axis must be a native storage axis 0, 1 or 2")
        if type(cells) is not int or cells < 1:
            raise ValueError("quadrature cells must be a positive integer")
        if type(dimension) is not PhysicalDimension:
            raise TypeError("quadrature requires explicit physical units")
        lower, upper = Fraction(lower), Fraction(upper)
        if upper <= lower:
            raise ValueError("quadrature requires upper > lower")
        resolved_weights = (((upper - lower) / cells,) * cells if weights is None
                            else tuple(Fraction(value) for value in weights))
        if len(resolved_weights) != cells:
            raise ValueError("quadrature requires one exact weight per source cell")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "weights", resolved_weights)

    def to_data(self) -> dict[str, Any]:
        return {"axis": self.axis, "domain": {"lower": _pair(self.lower),
                "upper": _pair(self.upper), "cells": self.cells},
                "rule": "explicit-cell-average-weighted-sum@1",
                "weights": [_pair(value) for value in self.weights],
                "dimension": self.dimension.to_data()}


@dataclass(frozen=True, slots=True)
class VelocityQuadrature:
    """Uniform integration convenience for a map with one eliminated coordinate."""
    lower: Fraction
    upper: Fraction
    cells: int
    dimension: PhysicalDimension

    def __post_init__(self) -> None:
        value = AxisQuadrature(0, self.lower, self.upper, self.cells, self.dimension)
        object.__setattr__(self, "lower", value.lower)
        object.__setattr__(self, "upper", value.upper)

    @property
    def weight(self) -> Fraction:
        return (self.upper - self.lower) / self.cells

    def to_data(self) -> dict[str, Any]:
        return {"rule": "uniform-cell-average-integral@1", "lower": _pair(self.lower),
                "upper": _pair(self.upper), "cells": self.cells,
                "weight": _pair(self.weight), "dimension": self.dimension.to_data()}


@dataclass(frozen=True, slots=True, init=False)
class PhysicalSupportMap:
    """Explicit support-subset reduction or directional constant extension.

    Axis tuples associate each support coordinate with a native storage axis.
    Their defaults follow the support tuple order; explicit tuples can permute
    either embedding. Missing native axes use unit-measure singleton storage.
    A reduction must supply exactly one quadrature for each eliminated axis.
    An extension is authored independently and never inferred as an inverse.
    """
    source_support: PhysicalSupport
    target_support: PhysicalSupport
    quadrature: VelocityQuadrature | None
    reductions: tuple[AxisQuadrature, ...]
    source_axes: tuple[int, ...]
    target_axes: tuple[int, ...]
    native_dimension: int

    def __init__(self, source_support: PhysicalSupport, target_support: PhysicalSupport,
                 quadrature: VelocityQuadrature | None = None,
                 reductions: tuple[AxisQuadrature, ...] = (),
                 source_axes: tuple[int, ...] | None = None,
                 target_axes: tuple[int, ...] | None = None,
                 native_dimension: int | None = None) -> None:
        if any(type(item) is not PhysicalSupport for item in (source_support, target_support)):
            raise TypeError("physical map requires exact, explicit supports")
        source, target = source_support.coordinates, target_support.coordinates
        dimension = max(len(source), len(target)) if native_dimension is None else native_dimension
        if type(dimension) is not int or dimension not in (1, 2, 3):
            raise ValueError("physical support identities require a native dimension of 1, 2 or 3")
        source_axes = tuple(range(len(source))) if source_axes is None else tuple(source_axes)
        target_axes = tuple(range(len(target))) if target_axes is None else tuple(target_axes)
        for name, support, axes in (("source_axes", source, source_axes),
                                    ("target_axes", target, target_axes)):
            if len(axes) != len(support) or len(set(axes)) != len(axes) or any(
                    type(axis) is not int or not 0 <= axis < dimension for axis in axes):
                raise ValueError("physical support identities require distinct valid " + name)
        reductions = tuple(reductions)
        if any(type(row) is not AxisQuadrature for row in reductions):
            raise TypeError("physical reductions require exact AxisQuadrature values")
        if quadrature is not None:
            if type(quadrature) is not VelocityQuadrature:
                raise TypeError("moment requires an exact VelocityQuadrature")
            missing = tuple(axis for coordinate, axis in zip(source, source_axes, strict=True)
                            if coordinate not in target)
            if len(missing) != 1:
                raise ValueError("velocity quadrature support identities require exactly one eliminated coordinate")
            q = quadrature
            uniform = (AxisQuadrature(missing[0], q.lower, q.upper, q.cells, q.dimension),)
            if reductions and reductions != uniform:
                raise TypeError("use either a VelocityQuadrature or explicit reductions")
            reductions = uniform
        reducing = bool(reductions)
        superset, subset = (source, target) if reducing else (target, source)
        if not set(subset) < set(superset):
            raise ValueError("physical map support identities must form an exact strict subset; "
                             "homonymous coordinates from different domains do not match")
        required = {axis for coordinate, axis in zip(source, source_axes, strict=True)
                    if coordinate not in target} if reducing else set()
        if len({row.axis for row in reductions}) != len(reductions) or {
                row.axis for row in reductions} != required:
            raise ValueError("quadrature axes must cover each eliminated support coordinate exactly once")
        object.__setattr__(self, "source_support", source_support)
        object.__setattr__(self, "target_support", target_support)
        object.__setattr__(self, "quadrature", quadrature)
        object.__setattr__(self, "source_axes", source_axes)
        object.__setattr__(self, "target_axes", target_axes)
        object.__setattr__(self, "native_dimension", dimension)
        object.__setattr__(self, "reductions", tuple(sorted(reductions, key=lambda row: row.axis)))

    @property
    def operation_abi(self) -> int:
        return 2 if self.reductions else 3

    @property
    def source_to_target(self) -> tuple[int, ...]:
        target = dict(zip(self.target_support.coordinates, self.target_axes, strict=True))
        correspondence = [-1] * self.native_dimension
        for coordinate, axis in zip(self.source_support.coordinates, self.source_axes, strict=True):
            correspondence[axis] = target.get(coordinate, -1)
        return tuple(correspondence)

    def native_contract(self) -> dict[str, Any]:
        return {"physical_contract": True,
                "physical_source_to_target": self.source_to_target,
                "physical_source_active": tuple(int(axis in self.source_axes)
                                                for axis in range(self.native_dimension)),
                "physical_target_active": tuple(int(axis in self.target_axes)
                                                for axis in range(self.native_dimension))}

    def validate_ports(self, source: Any, target: Any) -> None:
        source_space = getattr(source.subject, "space", None)
        target_space = getattr(target.subject, "space", None)
        if source_space is None or target_space is None:
            raise ValueError("physical map ports require typed quantity declarations")
        spaces = source_space, target_space
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
            for quadrature in self.reductions:
                for name, power in quadrature.dimension.powers:
                    powers[name] = powers.get(name, Fraction(0)) + power
            if PhysicalDimension(tuple(powers.items())) != destination:
                raise ValueError("physical map units disagree with reduction measure/pullback")

    def to_data(self) -> dict[str, Any]:
        return {"kind": "support-reduction@2" if self.reductions else "support-extension@2",
                "source_support": self.source_support.to_data(),
                "target_support": self.target_support.to_data(),
                "reductions": [row.to_data() for row in self.reductions],
                "storage": {"native_dimension": self.native_dimension,
                            "source_axes": list(self.source_axes), "target_axes": list(self.target_axes),
                            "source_to_target": list(self.source_to_target),
                            "hidden_cells": 1, "hidden_measure": 1,
                            "extension": "constant", "inverse_closure": False}}
