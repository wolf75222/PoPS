"""Physical ports of consumed, equation-owned scalar observations."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import math
from pops.model import Handle, FieldSpace, PhysicalDimension


@dataclass(frozen=True, slots=True)
class ConsumedFieldPort:
    """Select an unknown (or one native gradient axis), with an explicit scalar factor.

    This is declaration data, not an array or a replacement Field producer. Its
    support and units come from the registered physical equation. Runtime storage
    is inferred separately from the complete equation input authority.
    """
    field: Handle
    unknown: Handle
    derivative_axis: int | None = None
    factor: Fraction = Fraction(1)

    def __post_init__(self):
        if not isinstance(self.field, Handle) or self.field.kind != "field" \
                or getattr(self.field, "_field_registry", None) is None:
            raise TypeError("consumed Field port requires its registered Case field")
        if not isinstance(self.unknown, Handle) or self.unknown.kind != "field":
            raise TypeError("consumed Field port requires an exact unknown Handle")
        if self.derivative_axis is not None and (type(self.derivative_axis) is not int or self.derivative_axis < 0):
            raise TypeError("consumed Field derivative axis must be an exact nonnegative integer")
        if type(self.factor) not in (int, Fraction):
            raise TypeError("consumed Field factor must be an exact integer or Fraction")
        factor = Fraction(self.factor)
        try:
            representable = math.isfinite(float(factor)) and (factor == 0 or float(factor) != 0)
        except OverflowError:
            representable = False
        if not representable:
            raise ValueError("consumed Field factor must have a finite non-underflowing native value")
        object.__setattr__(self, "factor", factor)
        self.quantity_space()

    def _declaration(self):
        from ._program_problem import _identity
        registration = self.field._field_registry.resolved_registration(self.field)
        problem = registration.operator
        unknowns = tuple(row for row in problem.unknowns if _identity(row) == _identity(self.unknown))
        if len(unknowns) != 1 or not getattr(problem, "unknown_spaces", None):
            raise ValueError("consumed Field port has no exact equation-owned observation declaration")
        return problem, problem.unknown_spaces[unknowns[0]]

    def quantity_space(self):
        problem, scalar = self._declaration()
        unit = scalar.units[0]
        if self.derivative_axis is not None:
            axes = getattr(self.field._field_registry.resolved_registration(self.field).discretization, "observation_axes", None)
            if axes is None or self.derivative_axis not in axes or len(problem.coordinate_units) != len(axes):
                raise ValueError("consumed Field derivative axis has no active physical-coordinate binding")
            coordinate = axes.index(self.derivative_axis)
            powers = dict(unit.powers)
            for base, exponent in problem.coordinate_units[coordinate].powers:
                powers[base] = powers.get(base, 0) - exponent
            unit = PhysicalDimension(tuple(powers.items()))
        return FieldSpace(scalar.name, components=scalar.components, support=scalar.support,
            units=(unit,), sampling="cell", representation="field", centering="cell",
            frame=scalar.frame, clock=scalar.clock)

    def to_data(self):
        from ._program_problem import _identity
        problem, _ = self._declaration()
        return {"contract": "mapped-consumed-output@1", "field": _identity(self.field),
            "field_problem_identity": problem.identity.token, "unknown": _identity(self.unknown),
            "derivative_axis": self.derivative_axis,
            "observation_axes": list(getattr(self.field._field_registry.resolved_registration(self.field).discretization, "observation_axes", ()) or ()),
            "factor": {"numerator": str(self.factor.numerator), "denominator": str(self.factor.denominator)},
            "space": self.quantity_space().to_data()}
