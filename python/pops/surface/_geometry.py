"""Supports of the surface: ``pops.Box`` and ``pops.Periodic`` (spec 0.3, 2.6.2, 2.6.6, §3.2, R20).

``pops.Box(x=(a, b), y=pops.Periodic(c, d))`` declares a box support with one keyword per axis, in
order; a bound is a number or an expression of numbers and ``pi``. ``Ω.x`` is the coordinate x,
which also names the axis x where an axis is expected; ``Ω.axes`` lists the axes and
``Ω.boundary.x_min`` names a boundary piece. An axis may not take the name of a reading of a
support (SU8, ``ReservedName``). The node function of ``pops.Graded`` is called with the logical
coordinate s in [0, 1] of its axis (``LogicalCoordinate``), not with the coordinate.

The elaborated form is a b6 ``CartesianDomain`` named ``box`` with the frame of its axes; the
identification of a periodic axis is carried by the layout of b6 (``PeriodicAxes``). The frames of
b6 name their axes x, y and z, in that order, so this lot elaborates the boxes whose axes are
named so; another name, an unbounded interval, a product and a tagged piece wait for the supports
of G03 and G09.
"""
from __future__ import annotations

import math as _math
from fractions import Fraction
from typing import Any

from ._expr import Arith, Expr, is_number
from ._refusal import NotYetElaborated, ReservedName, user_site

# SU8: the readings of a support, which no axis may take.
SUPPORT_READINGS = ("axes", "factors", "position", "point", "piece", "boundary", "identify", "copy")
# The axis names of the Cartesian frames of b6, in order.
FRAME_AXES = ("x", "y", "z")


class MathConstant(Expr):
    """A constant of pops.math, ``pi`` or ``infinity``: an exact symbol, folded when it is used."""

    __slots__ = ("name", "value")

    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self.value = value

    def _text(self, precedence: int) -> str:
        return self.name


pi = MathConstant("pi", _math.pi)
infinity = MathConstant("infinity", _math.inf)


def fold(value: Any, where: str) -> Any:
    """The number of a bound written with numbers and the constants of pops.math."""
    if is_number(value):
        return value
    if isinstance(value, MathConstant):
        return value.value
    if isinstance(value, Arith):
        args = [fold(arg, where) for arg in value.args]
        if value.op == "neg":
            return -args[0]
        left, right = args
        if value.op == "+":
            return left + right
        if value.op == "-":
            return left - right
        if value.op == "*":
            return left * right
        if value.op == "/":
            return Fraction(left, right) if isinstance(left, int) and isinstance(right, int) \
                else left / right
        return left ** right
    raise TypeError("%s is a number or an expression of numbers and pi, got %r" % (where, value))


class Periodic:
    """``pops.Periodic(lower, upper)``: an interval whose two ends are identified (R20)."""

    __slots__ = ("lower", "upper", "site")

    def __init__(self, lower: Any, upper: Any) -> None:
        self.lower = fold(lower, "pops.Periodic(lower, upper)")
        self.upper = fold(upper, "pops.Periodic(lower, upper)")
        if not (_math.isfinite(float(self.lower)) and _math.isfinite(float(self.upper))):
            raise ValueError("pops.Periodic(lower, upper) takes two finite bounds")
        if not float(self.lower) < float(self.upper):
            raise ValueError("pops.Periodic(lower, upper) needs lower < upper")
        self.site = user_site(2)

    def __repr__(self) -> str:
        return "pops.Periodic(%s, %s)" % (self.lower, self.upper)


class Coordinate(Expr):
    """``Ω.x``: the coordinate x of a support, which also names its axis x."""

    __slots__ = ("support", "name", "index")

    def __init__(self, support: Box, name: str, index: int) -> None:
        self.support = support
        self.name = name
        self.index = index

    def _text(self, precedence: int) -> str:
        return self.name


class LogicalCoordinate(Expr):
    """The logical coordinate s in [0, 1] of an axis, with which PoPS calls the node function of
    ``pops.Graded`` once (D0b; SU5 as section 13 of the design note N17 states it)."""

    __slots__ = ("axis",)

    def __init__(self, axis: Coordinate) -> None:
        self.axis = axis

    def _text(self, precedence: int) -> str:
        return "s"


class Piece:
    """A boundary piece of a box, ``Ω.boundary.x_min``."""

    __slots__ = ("support", "name", "axis", "side")

    def __init__(self, support: Box, name: str, axis: str, side: str) -> None:
        self.support = support
        self.name = name
        self.axis = axis
        self.side = side

    def __repr__(self) -> str:
        return self.name


class Boundary:
    """``Ω.boundary``: the pieces that periodicity does not identify, read by name."""

    __slots__ = ("_support",)

    def __init__(self, support: Box) -> None:
        self._support = support

    def __getattr__(self, name: str) -> Piece:
        if not name.startswith("_"):
            axis, _, side = name.rpartition("_")
            if axis in self._support._names and side in ("min", "max"):
                if self._support.periodic(axis):
                    raise ValueError(
                        "%s is periodic, so Ω.boundary has no piece %s: its two ends are "
                        "identified (R20)" % (axis, name))
                return Piece(self._support, name, axis, side)
        raise AttributeError("the boundary of %r has the pieces %s"
                             % (self._support, ", ".join(self._support.piece_names())))


class Box:
    """``pops.Box(x=(0, 1), y=pops.Periodic(0, 1))``: a box support, one keyword per axis."""

    def __init__(self, **axes: Any) -> None:
        if not axes:
            raise TypeError("pops.Box takes one keyword per axis, pops.Box(x=(0, 1))")
        names = tuple(axes)
        for name in names:
            if name in SUPPORT_READINGS or name.startswith("_"):
                raise ReservedName(
                    "the axis %r of pops.Box takes the name of a reading of a support; the "
                    "reserved names are %s, and no name begins with an underscore"
                    % (name, ", ".join(SUPPORT_READINGS)),
                    write="another axis name, such as pops.Box(x=(0, 1))")
        bounds = []
        for name, value in axes.items():
            if isinstance(value, Periodic):
                bounds.append((value.lower, value.upper, True))
                continue
            if isinstance(value, list) or not isinstance(value, tuple) or len(value) != 2:
                raise TypeError("pops.Box(%s=...) takes an interval (lower, upper) or "
                                "pops.Periodic(lower, upper), got %r" % (name, value))
            lower = fold(value[0], "pops.Box(%s=(lower, upper))" % name)
            upper = fold(value[1], "pops.Box(%s=(lower, upper))" % name)
            if not (_math.isfinite(float(lower)) and _math.isfinite(float(upper))):
                raise NotYetElaborated(
                    "an unbounded interval of pops.Box, for a support that a quadrature rule "
                    "lays out (D0)", "a later lot of N15 with the supports of G03")
            if not float(lower) < float(upper):
                raise ValueError("pops.Box(%s=(lower, upper)) needs lower < upper" % name)
            bounds.append((lower, upper, False))
        self._names = names
        self._bounds = tuple(bounds)
        self._coordinates = tuple(Coordinate(self, name, index) for index, name in enumerate(names))
        self.site = user_site(2)

    @property
    def rank(self) -> int:
        return len(self._names)

    @property
    def axes(self) -> list:
        return list(self._coordinates)

    @property
    def boundary(self) -> Boundary:
        return Boundary(self)

    def periodic(self, axis: str) -> bool:
        return self._bounds[self._names.index(axis)][2]

    def piece_names(self) -> list:
        return ["%s_%s" % (name, side)
                for name, (_, _, periodic) in zip(self._names, self._bounds, strict=True)
                if not periodic for side in ("min", "max")]

    def bounds(self) -> tuple:
        return self._bounds

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        names = self.__dict__.get("_names", ())
        if name in names:
            return self._coordinates[names.index(name)]
        if name in ("position", "factors", "point", "piece", "identify", "copy"):
            raise NotYetElaborated("the reading Ω.%s of a support" % name,
                                   "a later lot of N15 with the supports of G03 and G09")
        raise AttributeError("%r has the axes %s; Ω.%s is none of them"
                             % (self, ", ".join(names), name))

    def __repr__(self) -> str:
        parts = []
        for name, (lower, upper, periodic) in zip(self._names, self._bounds, strict=True):
            parts.append("%s=%s" % (name, "pops.Periodic(%s, %s)" % (lower, upper) if periodic
                                    else "(%s, %s)" % (lower, upper)))
        return "pops.Box(%s)" % ", ".join(parts)

    def __bool__(self) -> bool:
        return True

    __hash__ = object.__hash__
