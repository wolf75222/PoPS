"""Exact immutable coefficient authorities for Runge--Kutta method families."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from pops._ir.literals import scalar_literal


def exact_coefficient(value: Any, where: str) -> Any:
    """Normalize one finite, dimensionless numerical coefficient without a float round-trip."""
    try:
        literal = scalar_literal(value)
    except (TypeError, ValueError) as exc:
        raise type(exc)("%s must be a finite real coefficient" % where) from exc
    if literal.unit is not None or literal.target is not None or literal.cpp is not None:
        raise TypeError("%s cannot carry units, a target, or custom C++ spelling" % where)
    return literal.to_python()


def exact_fraction(value: Any, where: str) -> Fraction:
    value = exact_coefficient(value, where)
    return Fraction.from_float(value) if isinstance(value, float) else Fraction(value)


def _rows(A: Any, stages: int, *, diagonal: bool, where: str) -> tuple[tuple[Any, ...], ...]:
    try:
        source = tuple(tuple(row) for row in A)
    except TypeError as exc:
        raise TypeError("%s must be a finite sequence of coefficient rows" % where) from exc
    if len(source) != stages:
        raise ValueError("%s and weights must share the stage count" % where)
    normalized = []
    for i, raw in enumerate(source):
        width = i + int(diagonal)
        if len(raw) < width or len(raw) > stages:
            raise ValueError("%s[%d] must provide %d active coefficient(s)" % (where, i, width))
        row = tuple(exact_coefficient(item, "%s[%d][%d]" % (where, i, j))
                    for j, item in enumerate(raw))
        if any(exact_fraction(item, where) != 0 for item in row[width:]):
            shape = "lower-triangular" if diagonal else "strictly lower-triangular"
            raise ValueError("%s must be %s" % (where, shape))
        normalized.append(row[:width])
    return tuple(normalized)


def _weights(values: Any, where: str) -> tuple[Any, ...]:
    try:
        result = tuple(exact_coefficient(item, "%s[%d]" % (where, i))
                       for i, item in enumerate(values))
    except TypeError as exc:
        raise TypeError("%s must be a finite coefficient sequence" % where) from exc
    if not result:
        raise ValueError("%s requires at least one stage" % where)
    return result


def _nodes(rows: tuple[tuple[Any, ...], ...], c: Any, where: str) -> tuple[Any, ...]:
    sums = tuple(sum((exact_fraction(item, where) for item in row), Fraction()) for row in rows)
    if c is None:
        return sums
    result = _weights(c, where + ".c")
    if len(result) != len(rows):
        raise ValueError("%s A, b, and c must share the stage count" % where)
    for i, (actual, expected) in enumerate(zip(result, sums, strict=True)):
        if exact_fraction(actual, where) != expected:
            raise ValueError("%s.c[%d] must equal exact row sum %r" % (where, i, expected))
    return result


@dataclass(frozen=True, slots=True, init=False)
class RungeKuttaTableau:
    """Canonical explicit Butcher tableau; coefficients retain their exact authoring domain."""

    A: tuple[tuple[Any, ...], ...]
    b: tuple[Any, ...]
    c: tuple[Any, ...]
    name: str | None
    __pops_ir_immutable__ = True

    def __init__(self, A: Any, b: Any, c: Any = None, name: Any = None) -> None:
        weights = _weights(b, "RungeKuttaTableau.b")
        rows = _rows(A, len(weights), diagonal=False, where="RungeKuttaTableau.A")
        if sum((exact_fraction(x, "RungeKuttaTableau.b") for x in weights), Fraction()) != 1:
            raise ValueError("RungeKuttaTableau weights b must sum exactly to 1")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("RungeKuttaTableau name must be a non-empty string or None")
        object.__setattr__(self, "A", rows)
        object.__setattr__(self, "b", weights)
        object.__setattr__(self, "c", _nodes(rows, c, "RungeKuttaTableau"))
        object.__setattr__(self, "name", name)

    @property
    def stages(self) -> int:
        return len(self.b)

    @property
    def properties(self) -> Any:
        from pops.time.method_properties import analyze_runge_kutta
        return analyze_runge_kutta(self)

    @property
    def certificate(self) -> Any:
        from pops.time.method_properties import certify_runge_kutta
        return certify_runge_kutta(self)


@dataclass(frozen=True, slots=True, init=False)
class AdditiveRungeKuttaTableau:
    """Partitioned ARK authority with explicit and diagonally-implicit coefficient arrays."""

    explicit: RungeKuttaTableau
    implicit_A: tuple[tuple[Any, ...], ...]
    implicit_b: tuple[Any, ...]
    implicit_c: tuple[Any, ...]
    name: str | None
    __pops_ir_immutable__ = True

    def __init__(self, explicit: RungeKuttaTableau, implicit_A: Any,
                 implicit_b: Any, implicit_c: Any = None, name: Any = None) -> None:
        if type(explicit) is not RungeKuttaTableau:
            raise TypeError("AdditiveRungeKuttaTableau explicit must be a RungeKuttaTableau")
        weights = _weights(implicit_b, "AdditiveRungeKuttaTableau.implicit_b")
        if len(weights) != explicit.stages:
            raise ValueError("ARK partitions must share the stage count")
        if sum((exact_fraction(x, "ARK implicit_b") for x in weights), Fraction()) != 1:
            raise ValueError("ARK implicit weights must sum exactly to 1")
        rows = _rows(implicit_A, len(weights), diagonal=True,
                     where="AdditiveRungeKuttaTableau.implicit_A")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("ARK name must be a non-empty string or None")
        object.__setattr__(self, "explicit", explicit)
        object.__setattr__(self, "implicit_A", rows)
        object.__setattr__(self, "implicit_b", weights)
        object.__setattr__(self, "implicit_c", _nodes(rows, implicit_c, "ARK implicit"))
        object.__setattr__(self, "name", name)

    @property
    def stages(self) -> int:
        return self.explicit.stages

    @property
    def abscissae(self) -> tuple[tuple[Any, Any], ...]:
        return tuple(zip(self.explicit.c, self.implicit_c, strict=True))

    @property
    def certificate(self) -> Any:
        from pops.time.method_properties import certify_additive_runge_kutta
        return certify_additive_runge_kutta(self)

    @property
    def properties(self) -> Any:
        return self.certificate.properties


__all__ = [
    "AdditiveRungeKuttaTableau", "RungeKuttaTableau", "exact_coefficient", "exact_fraction",
]
