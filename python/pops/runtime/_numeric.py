"""Strict numeric contracts for public runtime descriptors and native ABI seams."""
from __future__ import annotations

from typing import Any

from pops._ir.literals import exact_numeric_scalar, scalar_to_native


def exact_real(
    value: Any,
    *,
    where: str,
    minimum: Any = None,
    maximum: Any = None,
    minimum_open: bool = False,
    maximum_open: bool = False,
) -> Any:
    """Retain an exact finite authoring scalar and validate optional bounds."""
    exact = exact_numeric_scalar(value, where=where)
    if minimum is not None:
        valid = exact > minimum if minimum_open else exact >= minimum
        if not valid:
            relation = ">" if minimum_open else ">="
            raise ValueError("%s %s %r required (got %r)" % (where, relation, minimum, value))
    if maximum is not None:
        valid = exact < maximum if maximum_open else exact <= maximum
        if not valid:
            relation = "<" if maximum_open else "<="
            raise ValueError("%s %s %r required (got %r)" % (where, relation, maximum, value))
    return exact


def positive_int(value: Any, *, where: str) -> int:
    """Require a positive Python int; bool and lossful ``int(x)`` coercions are invalid."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("%s must be a positive Python int (got %r)" % (where, value))
    return value


def optional_positive_int(value: Any, *, where: str, sentinel: int = 0) -> int:
    return sentinel if value is None else positive_int(value, where=where)


def strict_bool(value: Any, *, where: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError("%s must be bool (got %r)" % (where, value))
    return value


def native_real(value: Any, *, where: str) -> float:
    """The one explicit exact-authoring -> native binary64 conversion."""
    return scalar_to_native(value, where=where)


def native_block_scalars(time: Any, spatial: Any, *, where: str) -> tuple[float, ...]:
    """Lower the five real-valued controls of a native block ABI exactly once."""
    from pops.runtime.defaults import (
        NEWTON_DEFAULT_ABS_TOL,
        NEWTON_DEFAULT_DAMPING,
        NEWTON_DEFAULT_FD_EPS,
        NEWTON_DEFAULT_REL_TOL,
    )
    values = (
        ("newton_rel_tol", getattr(time, "newton_rel_tol", NEWTON_DEFAULT_REL_TOL)),
        ("newton_abs_tol", getattr(time, "newton_abs_tol", NEWTON_DEFAULT_ABS_TOL)),
        ("newton_fd_eps", getattr(time, "newton_fd_eps", NEWTON_DEFAULT_FD_EPS)),
        ("newton_damping", getattr(time, "newton_damping", NEWTON_DEFAULT_DAMPING)),
        ("positivity_floor", getattr(spatial, "positivity_floor", 0.0)),
    )
    return tuple(native_real(value, where=where + "." + name) for name, value in values)


__all__ = [
    "exact_real", "native_block_scalars", "native_real",
    "optional_positive_int", "positive_int", "strict_bool",
]
