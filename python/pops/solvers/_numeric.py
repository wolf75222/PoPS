"""Exact authoring scalars and explicit native conversion for solver descriptors."""
from __future__ import annotations

import math
from typing import Any


def exact_finite_real(value: Any, *, where: str) -> Any:
    """Retain an exact Python scalar, rejecting annotations and non-numeric payloads."""
    # Lazy by design: solver descriptors remain an import-graph sink. Scalar inspection is needed
    # only when a numeric option is actually authored, never while importing pops.solvers.
    from pops._ir.literals import scalar_literal
    if isinstance(value, bool):
        raise TypeError("%s must be a real scalar, not bool" % where)
    try:
        literal = scalar_literal(value)
    except (TypeError, ValueError) as exc:
        raise type(exc)("%s must be a finite real scalar (got %r)" % (where, value)) from exc
    if literal.unit is not None or literal.target is not None:
        raise TypeError("%s cannot carry a unit or target annotation" % where)
    try:
        return literal.to_python()
    except TypeError as exc:
        raise TypeError(
            "%s must be numerically evaluable before native solver lowering" % where) from exc


def exact_positive_real(value: Any, *, where: str) -> Any:
    numeric = exact_finite_real(value, where=where)
    if numeric <= 0:
        raise ValueError("%s must be > 0 (got %r)" % (where, value))
    return numeric


def exact_nonnegative_real(value: Any, *, where: str) -> Any:
    numeric = exact_finite_real(value, where=where)
    if numeric < 0:
        raise ValueError("%s must be >= 0 (got %r)" % (where, value))
    return numeric


def exact_open_unit_real(value: Any, *, where: str) -> Any:
    numeric = exact_finite_real(value, where=where)
    if not 0 < numeric < 1:
        raise ValueError("%s must be in (0, 1) (got %r)" % (where, value))
    return numeric


def optional_positive_int(value: Any, *, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be a Python int or None (got %r)" % (where, value))
    from pops._ir.literals import exact_cpp_int
    return exact_cpp_int(value, where=where, minimum=1)


def strict_bool(value: Any, *, where: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError("%s must be a Python bool (got %r)" % (where, value))
    return value


def native_float(value: Any, *, where: str) -> float:
    """Convert only at the Python/native boundary and refuse overflow/non-finite results."""
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s cannot be represented by the native real type" % where) from exc
    if not math.isfinite(converted):
        raise ValueError("%s must lower to a finite native real" % where)
    return converted


__all__ = [
    "exact_finite_real", "exact_nonnegative_real", "exact_open_unit_real", "exact_positive_real", "native_float",
    "optional_positive_int", "strict_bool",
]
