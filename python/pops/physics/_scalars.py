"""Exact scalar boundaries shared by the physics authoring APIs.

Physics descriptors keep the author's integer, rational, decimal, or binary64
domain until an explicitly named native boundary.  This module deliberately
does not provide a permissive ``float(value)`` helper: every route states
whether it needs an evaluable number or can emit an algebraic C++ spelling.
"""
from __future__ import annotations

import json
import math
from decimal import Decimal
from typing import Any

from pops.ir.literals import ScalarLiteral, scalar_data, scalar_literal


def _unannotated_literal(value: Any, *, where: str) -> ScalarLiteral:
    try:
        literal = scalar_literal(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise type(exc)("%s: %s" % (where, exc)) from exc
    if literal.unit is not None:
        raise TypeError(
            "%s cannot lower the unit annotation %r; convert to the declared physics unit first"
            % (where, literal.unit))
    if literal.target is not None:
        raise TypeError(
            "%s cannot lower the target annotation %r through its pops::Real ABI"
            % (where, literal.target))
    return literal


def exact_physics_scalar(
    value: Any,
    *,
    where: str,
    positive: bool = False,
) -> Any:
    """Return an exact evaluable scalar, rejecting metadata erasure.

    Algebraic/custom literals are rejected here because bytecode and numeric
    descriptor routes have no symbolic scalar slot.  Code-generating routes
    use :func:`physics_scalar_cpp` instead.
    """
    literal = _unannotated_literal(value, where=where)
    try:
        exact = literal.to_python()
    except TypeError as exc:
        raise TypeError(
            "%s does not support algebraic/custom constants; provide an exact int, "
            "Fraction, Decimal, or finite float" % where) from exc
    if positive and not exact > 0:
        raise ValueError("%s must be strictly positive (got %r)" % (where, exact))
    return exact


def physics_scalar_cpp(value: Any, *, where: str) -> str:
    """Lower an unannotated scalar on a C++-generating route.

    Algebraic/custom literals are accepted only when their ``ScalarLiteral``
    supplies a C++ spelling, so unsupported kinds still fail at declaration or
    emission rather than falling through a Python float conversion.
    """
    literal = _unannotated_literal(value, where=where)
    try:
        return literal.to_cpp()
    except (TypeError, ValueError, OverflowError) as exc:
        raise type(exc)("%s: %s" % (where, exc)) from exc


def codegen_physics_scalar(value: Any, *, where: str) -> Any:
    """Normalize a C++-lowerable scalar without discarding symbolic structure."""
    literal = _unannotated_literal(value, where=where)
    try:
        return literal.to_python()
    except TypeError:
        # Prove the algebraic/custom route has a real lowering now, then retain the literal itself.
        physics_scalar_cpp(literal, where=where)
        return literal


def canonical_scalar_data(value: Any, *, where: str) -> dict[str, Any]:
    """JSON-shaped, lossless scalar identity for manifests and cache keys."""
    literal = _unannotated_literal(value, where=where)
    return literal.to_data()


def canonical_scalar_key(value: Any, *, where: str) -> str:
    return json.dumps(
        canonical_scalar_data(value, where=where),
        sort_keys=True,
        separators=(",", ":"),
    )


def native_real(value: Any, *, where: str) -> float:
    """Perform the one explicit conversion to the native binary64 ABI."""
    exact = exact_physics_scalar(value, where=where)
    try:
        result = float(exact)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OverflowError("%s cannot be represented by native pops::Real" % where) from exc
    if not math.isfinite(result):
        raise OverflowError("%s cannot be represented by finite native pops::Real" % where)
    return result


def subtract_exact_integer(value: Any, integer: int, *, where: str) -> Any:
    """Subtract an integer without applying Decimal's ambient precision context."""
    if isinstance(integer, bool) or not isinstance(integer, int):
        raise TypeError("%s: integer operand must be an int" % where)
    exact = exact_physics_scalar(value, where=where)
    if not isinstance(exact, Decimal):
        return exact - integer
    sign, digits, exponent = exact.as_tuple()
    coefficient = int("".join(map(str, digits)))
    if sign:
        coefficient = -coefficient
    if exponent >= 0:
        return Decimal(coefficient * (10 ** exponent) - integer)
    result = coefficient - integer * (10 ** (-exponent))
    result_sign = int(result < 0)
    result_digits = tuple(map(int, str(abs(result)))) if result else (0,)
    return Decimal((result_sign, result_digits, exponent))


def scalar_data_view(value: Any, *, where: str) -> dict[str, Any]:
    """Alias with an intent-revealing name at public inspection boundaries."""
    # Use the public helper after validation to keep one canonical data schema.
    _unannotated_literal(value, where=where)
    return scalar_data(value)


__all__ = [
    "canonical_scalar_data",
    "canonical_scalar_key",
    "codegen_physics_scalar",
    "exact_physics_scalar",
    "native_real",
    "physics_scalar_cpp",
    "scalar_data_view",
    "subtract_exact_integer",
]
