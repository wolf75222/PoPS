"""``pops.project(e, quadrature=Q)`` and ``pops.project(e, integral=I)``: the projection of an
analytic initial value onto the cell averages of a field (spec 0.3, 2.6.2, CN12; lot D-E of N15).

``e`` is an expression of the coordinates of the support (``Ω.x``), of numbers, constants and
parameters of the model, and of the words of ``pops.math``. ``quadrature=Q`` averages e over each
cell by the rule Q, one ``pops.Quadrature`` for every axis or a tuple of one per axis, mapped from
its interval to the cell: the average is the sum over the tensor nodes of the products of the
weights, divided by the length of the interval on each axis, times e at the mapped nodes.
``integral=I`` is a function of ``cell`` that returns the exact integral of e over the cell,
written with ``cell.lower(axis)``, ``cell.upper(axis)`` and ``cell.measure`` (CN12); PoPS divides
it by the measure of the cell and does not check that it is the integral of e. The choice between
the two is written in the case and never chosen by a heuristic (SU9).

A value that reads no coordinate, a number, a constant or an expression of them, has an exact cell
average and is written alone, ``initial={u: 0.05}``; an expression of the coordinates written alone
is refused, since its projection is a choice (``AmbiguousElaboration``).
"""
from __future__ import annotations

import math as _math
from typing import Any

from ._expr import Expr, walk
from ._geometry import Coordinate
from ._quadrature import Quadrature
from ._refusal import NotYetElaborated, surface_entry, user_site

_GEOMETRY = "a later lot of N15 with the geometry reads of §4.3 (D18)"


class Projection:
    """The value of ``pops.project(e, ...)``: the expression, and its rule or its integral."""

    __slots__ = ("expression", "quadrature", "integral", "site")

    def __init__(self, expression: Any, quadrature: Any, integral: Any, site: Any) -> None:
        self.expression = expression
        self.quadrature = quadrature
        self.integral = integral
        self.site = site

    def rules(self, rank: int) -> tuple:
        """The rule of each axis of a support of rank ``rank``."""
        if isinstance(self.quadrature, tuple):
            if len(self.quadrature) != rank:
                raise ValueError("pops.project(e, quadrature=(...)) gives %d rules for a support "
                                 "of %d axes" % (len(self.quadrature), rank))
            return self.quadrature
        return (self.quadrature,) * rank

    def __repr__(self) -> str:
        from ._expr import text

        how = "quadrature=%r" % (self.quadrature,) if self.integral is None else \
            "integral=%s" % getattr(self.integral, "__name__", "I")
        return "pops.project(%s, %s)" % (text(self.expression), how)


def _rule(rule: Any) -> Quadrature:
    if not isinstance(rule, Quadrature):
        raise TypeError("pops.project(e, quadrature=Q) takes a pops.Quadrature, such as "
                        "GaussLegendre(n) of pops.lib.quadrature, or a tuple of one per axis, "
                        "got %r" % (rule,))
    lower, upper = (float(end) for end in rule.interval)
    if not (_math.isfinite(lower) and _math.isfinite(upper)):
        raise ValueError("pops.project(e, quadrature=%r): a cell average takes a rule on a "
                         "finite interval" % (rule,))
    return rule


@surface_entry
def project(expression: Any = None, *, quadrature: Any = None, integral: Any = None,
            **earlier: Any) -> Projection:
    """``pops.project(e, quadrature=Q)`` or ``pops.project(e, integral=I)`` (2.6.2, CN12)."""
    site = user_site(2)
    if "projection" in earlier or "order" in earlier:
        raise TypeError("pops.project takes quadrature= or integral=, the rule or the exact "
                        "integral of the cell average (CN12)")
    if earlier:
        raise TypeError("pops.project has no option %s" % ", ".join(sorted(earlier)))
    if expression is None:
        raise TypeError("pops.project(e, quadrature=Q) takes the expression e")
    if quadrature is not None and integral is not None:
        raise TypeError("pops.project(e, ...) takes quadrature= or integral=, one of them")
    if quadrature is None and integral is None:
        from ._refusal import AmbiguousElaboration

        raise AmbiguousElaboration(
            "pops.project(%s) states neither the rule nor the integral of its cell average"
            % _text(expression),
            write="pops.project(e, quadrature=GaussLegendre(n)), or integral=I with I a "
                  "function of cell", site=site)
    if quadrature is not None:
        quadrature = tuple(_rule(rule) for rule in quadrature) \
            if isinstance(quadrature, (tuple, list)) else _rule(quadrature)
    elif not callable(integral):
        raise TypeError("pops.project(e, integral=I) takes a function of cell, got %r"
                        % (integral,))
    return Projection(expression, quadrature, integral, site)


def _text(value: Any) -> str:
    from ._expr import text

    return text(value)


def reads_coordinate(value: Any) -> bool:
    """Whether a value reads a coordinate of a support."""
    return any(isinstance(node, Coordinate) for node in walk(value))


class CellBound(Expr):
    """A bound of the cell in the integral of ``pops.project``: ``cell.lower(Ω.x)``,
    ``cell.upper(Ω.x)`` (``index`` the axis, ``side`` lower or upper), or ``cell.measure``
    (``index`` None)."""

    __slots__ = ("index", "side", "spelling")

    def __init__(self, index: Any, side: str, spelling: str) -> None:
        self.index = index
        self.side = side
        self.spelling = spelling

    def _text(self, precedence: int) -> str:
        return self.spelling


class ProjectionCell:
    """``cell`` in the integral of ``pops.project``: the cell whose average is projected, read
    by its bounds."""

    _later = {"width": _GEOMETRY, "center": _GEOMETRY, "shift": _GEOMETRY,
              "displacement": _GEOMETRY, "value": "a later lot of N15 with the data fields of Q3",
              "level": "a later lot of N15 with the hierarchies of N12",
              "parent": "a later lot of N15 with the hierarchies of N12"}

    def __init__(self, support: Any) -> None:
        self._support = support
        self.measure = CellBound(None, "measure", "cell.measure")

    def _axis(self, axis: Any) -> int:
        if not isinstance(axis, Coordinate) or axis.support is not self._support:
            raise TypeError("cell.lower(axis) takes an axis of the support, Ω.x, got %r"
                            % (axis,))
        return axis.index

    def lower(self, axis: Any) -> Any:
        return CellBound(self._axis(axis), "lower", "cell.lower(%s)" % axis.name)

    def upper(self, axis: Any) -> Any:
        return CellBound(self._axis(axis), "upper", "cell.upper(%s)" % axis.name)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._later:
            raise NotYetElaborated("cell.%s in the integral of pops.project" % name,
                                   self._later[name])
        raise AttributeError("cell in the integral of pops.project reads lower(axis), "
                             "upper(axis) and measure, not %s" % name)

    def __repr__(self) -> str:
        return "cell"
