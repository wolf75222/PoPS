"""Math-like construction helpers for analytic expression trees."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._model import (
    PredicateExpr,
    ScalarExpr,
    _between,
    _coordinate,
    _scalar_binary,
    _scalar_unary,
    _where,
    constant,
    parameter,
    _program_input,
    _time,
)


def coordinate(frame: Any, axis: Any) -> ScalarExpr:
    """Return the coordinate associated with one typed axis of ``frame``."""

    return _coordinate(frame, axis)


def param(value: Any) -> ScalarExpr:
    """Read one typed parameter Handle in an analytic scalar expression."""

    return parameter(value)


def input(value_id: Any, component: Any) -> ScalarExpr:
    """Read one named discrete input supplied by a consuming initial map."""

    return _program_input(value_id, component)


def time(clock: Any) -> ScalarExpr:
    """Read physical time from one exact logical ``Clock`` at native evaluation."""

    return _time(clock)


def x(frame: Any) -> ScalarExpr:
    """Return the typed x coordinate of ``frame``."""

    axis = getattr(frame, "x", None)
    if axis is None:
        raise TypeError("x(frame) requires a frame exposing a typed x axis")
    return _coordinate(frame, axis)


def y(frame: Any) -> ScalarExpr:
    """Return the typed y coordinate of ``frame``."""

    axis = getattr(frame, "y", None)
    if axis is None:
        raise TypeError("y(frame) requires a frame exposing a typed y axis")
    return _coordinate(frame, axis)


def z(frame: Any) -> ScalarExpr:
    """Return the typed z coordinate of a three-dimensional ``frame``."""

    axis = getattr(frame, "z", None)
    if axis is None:
        raise TypeError("z(frame) requires a frame exposing a typed z axis")
    return _coordinate(frame, axis)


def coordinates(frame: Any) -> tuple[ScalarExpr, ...]:
    """Return every typed coordinate of a rank-1/2/3 frame in canonical axis order."""

    axes = getattr(frame, "axes", None)
    if not isinstance(axes, tuple) or len(axes) not in (1, 2, 3):
        raise TypeError("coordinates(frame) requires a typed rank-1/2/3 Cartesian frame")
    return tuple(_coordinate(frame, axis) for axis in axes)


def _coordinate_center(frame: Any, center: Any) -> tuple[Any, ...]:
    axes = getattr(frame, "axes", None)
    if not isinstance(axes, tuple) or len(axes) not in (1, 2, 3):
        raise TypeError("analytic coordinates require a typed rank-1/2/3 frame")
    if center is None:
        return (0.0,) * len(axes)
    if isinstance(center, Mapping):
        if set(center) != set(axes):
            raise ValueError("analytic center must map every frame axis exactly once")
        return tuple(center[axis] for axis in axes)
    if isinstance(center, (str, bytes)):
        raise TypeError("analytic center must be a coordinate sequence")
    try:
        values = tuple(center)
    except TypeError as exc:
        raise TypeError("analytic center must be a coordinate sequence") from exc
    if len(values) != len(axes):
        raise ValueError(
            "analytic center must contain exactly %s components"
            % {1: "one", 2: "two", 3: "three"}[len(axes)]
        )
    return values



def _polar_center(frame: Any, center: Any) -> tuple[Any, Any]:
    axes = getattr(frame, "axes", None)
    if not isinstance(axes, tuple) or len(axes) != 2:
        raise TypeError("analytic polar coordinates require a typed two-dimensional frame")
    values = _coordinate_center(frame, center)
    return values[0], values[1]


def radius(frame: Any, *, center: Any = None) -> ScalarExpr:
    """Return the rank-generic Euclidean radius around ``center``."""

    values = coordinates(frame)
    center_values = _coordinate_center(frame, center)
    return norm(tuple(
        value - origin for value, origin in zip(values, center_values, strict=True)
    ))


def angle(frame: Any, *, center: Any = None) -> ScalarExpr:
    """Return the quadrant-aware polar angle around ``center``."""

    values = coordinates(frame)
    if len(values) != 2:
        raise TypeError("angle(frame) requires a typed two-dimensional frame")
    x_value, y_value = values
    center_x, center_y = _polar_center(frame, center)
    return atan2(y_value - center_y, x_value - center_x)


def sqrt(value: Any) -> ScalarExpr:
    return _scalar_unary("sqrt", value)


def abs(value: Any) -> ScalarExpr:
    return _scalar_unary("abs", value)


def sin(value: Any) -> ScalarExpr:
    return _scalar_unary("sin", value)


def cos(value: Any) -> ScalarExpr:
    return _scalar_unary("cos", value)


def exp(value: Any) -> ScalarExpr:
    return _scalar_unary("exp", value)


def log(value: Any) -> ScalarExpr:
    return _scalar_unary("log", value)


def atan2(y_value: Any, x_value: Any) -> ScalarExpr:
    """Return the quadrant-aware angle ``atan2(y_value, x_value)``."""

    return _scalar_binary("atan2", y_value, x_value)


def hypot(first: Any, second: Any) -> ScalarExpr:
    """Return the overflow-safe Euclidean norm of two scalar expressions."""

    return _scalar_binary("hypot", first, second)


def norm(*components: Any) -> ScalarExpr:
    """Return a balanced, overflow-safe Euclidean norm of one or more components."""

    values: tuple[Any, ...]
    if len(components) == 1 and not isinstance(components[0], (str, bytes, ScalarExpr)) \
            and isinstance(components[0], Sequence):
        values = tuple(components[0])
    else:
        values = components
    if not values:
        raise ValueError("analytic norm requires at least one component")
    expressions = tuple(abs(value) for value in values)
    while len(expressions) > 1:
        next_level = [
            hypot(expressions[index], expressions[index + 1])
            for index in range(0, len(expressions) - 1, 2)
        ]
        if len(expressions) % 2:
            next_level.append(expressions[-1])
        expressions = tuple(next_level)
    if not expressions:
        raise RuntimeError("analytic norm composition unexpectedly became empty")
    return expressions[0]


def minimum(first: Any, second: Any) -> ScalarExpr:
    """Return the pointwise minimum of two scalar expressions."""

    return _scalar_binary("minimum", first, second)


def maximum(first: Any, second: Any) -> ScalarExpr:
    """Return the pointwise maximum of two scalar expressions."""

    return _scalar_binary("maximum", first, second)


def clamp(value: Any, lower: Any, upper: Any) -> ScalarExpr:
    """Clamp a scalar expression to the closed interval ``[lower, upper]``."""

    return minimum(maximum(value, lower), upper)


def between(value: Any, lower: Any, upper: Any) -> PredicateExpr:
    """Build the inclusive predicate ``lower <= value <= upper`` without Python chaining."""

    return _between(value, lower, upper)


def where(predicate: Any, when_true: Any, when_false: Any) -> ScalarExpr:
    """Select between two scalar expressions with an analytic predicate."""

    return _where(predicate, when_true, when_false)


__all__ = [
    "abs",
    "angle",
    "atan2",
    "between",
    "clamp",
    "constant",
    "coordinate",
    "coordinates",
    "cos",
    "exp",
    "hypot",
    "input",
    "log",
    "maximum",
    "minimum",
    "norm",
    "param",
    "radius",
    "sin",
    "sqrt",
    "time",
    "where",
    "x",
    "y",
]
