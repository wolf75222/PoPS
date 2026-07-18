"""Shared exact coefficient normalization for temporal method authorities."""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from pops.identity.scalar import scalar_literal


def exact_coefficient(value: Any, where: str) -> Any:
    """Normalize one finite, dimensionless coefficient without a float round-trip."""
    try:
        literal = scalar_literal(value)
    except (TypeError, ValueError) as exc:
        raise type(exc)("%s must be a finite real coefficient" % where) from exc
    if literal.unit is not None or literal.target is not None or literal.cpp is not None:
        raise TypeError("%s cannot carry units, a target, or custom C++ spelling" % where)
    return literal.to_python()


def exact_fraction(value: Any, where: str) -> Fraction:
    """Return the exact rational represented by one normalized coefficient."""
    value = exact_coefficient(value, where)
    return Fraction.from_float(value) if isinstance(value, float) else Fraction(value)


__all__ = ["exact_coefficient", "exact_fraction"]
