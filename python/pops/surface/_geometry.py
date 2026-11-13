"""Supports of the surface: ``pops.Box``, ``pops.Periodic`` and the charted support (spec 0.3,
2.6.2, 2.6.6, §3.2, R20, D0b).

``pops.Box(x=(a, b), y=pops.Periodic(c, d))`` declares a box support with one keyword per axis, in
order; a bound is a number or an expression of numbers and ``pi``. ``Ω.x`` is the coordinate x,
which also names the axis x where an axis is expected; ``Ω.axes`` lists the axes and
``Ω.boundary.x_min`` names a boundary piece. An axis may not take the name of a reading of a
support (SU8, ``ReservedName``). The node function of ``pops.Graded`` is called with the logical
coordinate s in [0, 1] of its axis (``LogicalCoordinate``), not with the coordinate.

``σ.chart(x=e_x, y=e_y)`` states the chart of a support of rank two (D0b; design note N17, 3.2 and
DN25 to DN28): its keywords name the axes of the Cartesian frame, which key the components of a
flux and of a vector (``σ.axes``, ``σ.x``), and the coordinates of σ stay the logical axes, which
name the mesh axes, the pieces and the node functions of ``pops.Graded`` (S6). A chart that only
moves the nodes along each axis is refused naming ``pops.Graded``, and a piece on which the chart
does not depend on the other coordinate is degenerate: it leaves ``σ.boundary``.

The elaborated form is a b6 ``CartesianDomain`` named ``box`` with the frame of its axes; the
identification of a periodic axis is carried by the layout of b6 (``PeriodicAxes``). The frames of
b6 name their axes x, y and z, in that order, so this lot elaborates the boxes whose axes are
named so, and the charts whose keywords are x and y (DN26), the logical axes of a chart mapping to
the axes of its reference box by position; another name, an unbounded interval, a product and a
tagged piece wait for the supports of G03 and G09.
"""
from __future__ import annotations

import math as _math
from fractions import Fraction
from typing import Any

from ._expr import Arith, Call, Expr, is_number, text
from ._refusal import NotYetElaborated, ReservedName, SupportMismatch, surface_entry, user_site

# SU8: the readings of a support, which no axis may take (``chart`` and ``weighted`` by section 13
# of the design note N17).
SUPPORT_READINGS = ("axes", "factors", "position", "point", "piece", "boundary", "identify", "copy",
                    "chart", "weighted")
# The axis names of the Cartesian frames of b6, in order.
FRAME_AXES = ("x", "y", "z")
# The words of a coordinate map of the layout, a node function of pops.Graded or a chart: those of
# the analytic programs that generate the static geometry arrays at bind, without erf and erfc,
# which pops.math keeps to initial values.
MAP_WORDS = {"exp": _math.exp, "log": _math.log, "sqrt": _math.sqrt, "sin": _math.sin,
             "cos": _math.cos, "abs": abs}
_N11 = "N11 (expression closure, C1)"
_G03 = "a later lot of N15 with the supports of G03"
_L2C = "lot L2c of N17 (ROM-110), with identify of G09"
_L3 = "lot L3 of N17 (ROM-110), with the measure densities of axisymmetry"
# DN30: a coordinate read on a charted support waits for the reads at the points of a chart.
CHART_POINT_READS = ("lot L2b of N17 (ROM-110), which reads it at the image of the centre of a "
                     "cell and at the centroid of a face (DN7, DN30)")
# DN30: the consumers that the arrays of a chart do not serve yet stay refused, where the case is
# built: a value relation, whose ghost is the reflection of a cell across a face of the chart, and
# traces reconstructed above the cell values, whose slopes read the logical stencil of a cell.
CHART_GHOSTS = ("lot L2b of N17 (ROM-110), which reflects the ghost of a value relation across "
                "the face of a chart (DN25, DN30)")
CHART_RECONSTRUCTIONS = ("lot L2b of N17 (ROM-110), with the centred gradient of a chart that a "
                         "reconstruction above the cell values reads (DN25, DN30)")


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
    """``Ω.x``: the coordinate x of a support, which also names its axis x. On a charted support
    the coordinates of its box are the logical coordinates, which name the logical axes (S6)."""

    __slots__ = ("support", "name", "index")

    def __init__(self, support: Box, name: str, index: int) -> None:
        self.support = support
        self.name = name
        self.index = index

    def _text(self, precedence: int) -> str:
        return self.name


class FrameCoordinate(Expr):
    """``σ.x`` on a charted support σ: the frame coordinate x, whose value is the component x of the
    chart, an expression of the logical coordinates; it also names the frame axis x, which keys the
    components of a flux and of a vector (S6, D0b)."""

    __slots__ = ("support", "name", "index", "expression")

    def __init__(self, support: Charted, name: str, index: int, expression: Any) -> None:
        self.support = support
        self.name = name
        self.index = index
        self.expression = expression

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
    """A boundary piece of a support, ``Ω.boundary.x_min``, named by a logical axis and a side."""

    __slots__ = ("support", "name", "axis", "side")

    def __init__(self, support: Support, name: str, axis: str, side: str) -> None:
        self.support = support
        self.name = name
        self.axis = axis
        self.side = side

    def __repr__(self) -> str:
        return self.name


class Boundary:
    """``Ω.boundary``: the pieces that periodicity does not identify and that are not degenerate,
    read by name."""

    __slots__ = ("_support",)

    def __init__(self, support: Support) -> None:
        self._support = support

    def __getattr__(self, name: str) -> Piece:
        support = self._support
        if not name.startswith("_"):
            axis, _, side = name.rpartition("_")
            if axis in support._names and side in ("min", "max"):
                if support.periodic(axis):
                    raise ValueError(
                        "%s is periodic, so Ω.boundary has no piece %s: its two ends are "
                        "identified (R20)" % (axis, name))
                if name in support._degenerate:
                    other = [other for other in support._names if other != axis]
                    raise SupportMismatch(
                        "%r.boundary has the pieces %s; %s is degenerate: the chart does not "
                        "depend on %s there, so its faces have zero measure and it takes no "
                        "boundary relation (D0b, DN28)"
                        % (support, ", ".join(support.piece_names()) or "none", name,
                           " or ".join(other)),
                        write="a piece of the boundary; a read across %s needs "
                              "Ω.identify(Ω.boundary.%s, Ω.boundary.%s, by={...}) before the "
                              "chart, which %s elaborates" % (name, name, name, _L2C))
                return Piece(support, name, axis, side)
        raise AttributeError("the boundary of %r has the pieces %s"
                             % (support, ", ".join(support.piece_names())))


class Support:
    """What a box and a charted support share: the logical coordinates and their bounds, the
    pieces that periodicity does not identify and that are not degenerate (2.6.2, D0b)."""

    _names: tuple
    _bounds: tuple
    _coordinates: tuple
    # The pieces on which a chart does not depend on the other coordinate (DN28); none on a box.
    _degenerate: tuple = ()
    site: Any

    @property
    def rank(self) -> int:
        return len(self._names)

    @property
    def boundary(self) -> Boundary:
        return Boundary(self)

    @property
    def _box(self) -> Box:
        """The box of the logical coordinates: the box itself, or the box that a chart maps."""
        raise NotImplementedError("a support states the box of its logical coordinates")

    @property
    def _logical_axes(self) -> list:
        """The logical axes, which name the mesh axes: ``cells=`` in their order, ``face.axis``
        and the node functions of ``pops.Graded`` (S6); on a box, its axes."""
        return list(self._coordinates)

    def periodic(self, axis: str) -> bool:
        return self._bounds[self._names.index(axis)][2]

    def piece_names(self) -> list:
        return ["%s_%s" % (name, side)
                for name, (_, _, periodic) in zip(self._names, self._bounds, strict=True)
                if not periodic for side in ("min", "max")
                if "%s_%s" % (name, side) not in self._degenerate]

    def bounds(self) -> tuple:
        return self._bounds

    def __bool__(self) -> bool:
        return True

    __hash__ = object.__hash__


class Box(Support):
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
                    "lays out (D0)", _G03)
            if not float(lower) < float(upper):
                raise ValueError("pops.Box(%s=(lower, upper)) needs lower < upper" % name)
            bounds.append((lower, upper, False))
        self._names = names
        self._bounds = tuple(bounds)
        self._coordinates = tuple(Coordinate(self, name, index) for index, name in enumerate(names))
        self.site = user_site(2)

    @property
    def axes(self) -> list:
        return list(self._coordinates)

    @property
    def _box(self) -> Box:
        """The box of the logical coordinates: the box itself."""
        return self

    @surface_entry
    def chart(self, **frame: Any) -> Charted:
        """``σ.chart(x=e_x, y=e_y)``: the support σ with its chart, one expression of the
        coordinates of σ per axis of the Cartesian frame (D0b; design note N17, 3.2)."""
        return Charted(self, frame)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        names = self.__dict__.get("_names", ())
        if name in names:
            return self._coordinates[names.index(name)]
        if name == "weighted":
            raise NotYetElaborated("the reading Ω.weighted of a support, its measure density "
                                   "(D0b)", _L3)
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


class Charted(Support):
    """``σ.chart(x=e_x, y=e_y)``: a support of rank two with its chart (D0b; design note N17, 3.2).

    The keywords name the axes of the Cartesian frame, which key the components of a flux and of a
    vector: ``σ.axes`` lists them and ``σ.x`` is the frame coordinate x, whose value is the
    component x of the chart. The coordinates of the box stay the logical axes, read by their
    names (``σ.r``, the very coordinate of the box): they name the mesh axes, ``cells=`` in their
    order, the boundary pieces and the node functions of ``pops.Graded`` (S6). A piece on which the
    chart does not depend on the other coordinate is degenerate and leaves ``σ.boundary`` (DN28);
    a case on such a support waits for lot L2c of N17, with ``identify`` of G09. The layout states
    the quadrature rule of the measures, ``pops.Uniform(cells=..., geometry=rule)`` (DN29).
    """

    def __init__(self, box: Box, frame: dict) -> None:
        names, components, degenerate = _chart(box, frame)
        self._reference = box
        self._names = box._names
        self._bounds = box._bounds
        self._coordinates = box._coordinates
        self._frame_names = names
        self._components = components
        self._frame_coordinates = tuple(FrameCoordinate(self, name, index, component)
                                        for index, (name, component)
                                        in enumerate(zip(names, components, strict=True)))
        self._degenerate = degenerate
        self.site = user_site(2)

    @property
    def _box(self) -> Box:
        """The box of the logical coordinates: the box that the chart maps."""
        return self._reference

    @property
    def axes(self) -> list:
        return list(self._frame_coordinates)

    def chart(self, **frame: Any) -> Any:
        raise SupportMismatch(
            "%r has a chart already: one chart per support, written as one expression (D0b)"
            % (self,), write="the composed chart as one expression of the box, Ω.chart(x=..., "
                             "y=...)")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        frame = self.__dict__.get("_frame_names", ())
        if name in frame:
            return self._frame_coordinates[frame.index(name)]
        names = self.__dict__.get("_names", ())
        if name in names:
            return self._coordinates[names.index(name)]
        if name == "identify":
            raise SupportMismatch(
                "Ω.identify on the charted support %r: identify is stated before chart, never "
                "after it, so that a support has one spelling (D0b)" % (self,),
                write="pole = Ω.identify(...) on the box, then pole.chart(x=..., y=...)")
        if name == "weighted":
            raise NotYetElaborated("the reading Ω.weighted of a support, its measure density "
                                   "(D0b)", _L3)
        if name in ("position", "factors", "point", "piece", "copy"):
            raise NotYetElaborated("the reading Ω.%s of a support" % name,
                                   "a later lot of N15 with the supports of G03 and G09")
        raise AttributeError("%r has the frame axes %s and the logical axes %s; Ω.%s is none of "
                             "them" % (self, ", ".join(frame), ", ".join(names), name))

    def __repr__(self) -> str:
        return "%r.chart(%s)" % (self._box, ", ".join(
            "%s=%s" % (name, text(component))
            for name, component in zip(self._frame_names, self._components, strict=True)))


# --- the chart (D0b; design note N17, 3.2, DN26 to DN28) ----------------------------------------

def _chart(box: Box, frame: dict) -> tuple:
    """The frame names and the components of ``box.chart(**frame)``, and its degenerate pieces."""
    if not frame:
        raise TypeError("σ.chart takes one keyword per axis of the frame, σ.chart(x=e_x, y=e_y)")
    for name, value in frame.items():
        if not (is_number(value) or isinstance(value, Expr)):
            raise TypeError("σ.chart(%s=...) takes an expression of the coordinates of σ, got %r"
                            % (name, value))
    spelled = ", ".join("%s=%s" % (name, text(value)) for name, value in frame.items())
    graded = ("the box in the frame coordinates, pops.Box(x=(lower, upper), ...), and the spacing "
              "as layout={Ω: pops.Graded(base=pops.Uniform(cells=c), nodes={Ω.x: f})}")
    if box.rank == 1:
        raise SupportMismatch(
            "the chart %s of %r only moves the nodes along its axis (rank 1): the domain stays a "
            "box and the spacing is a choice of the layout (D0b, DN1)" % (spelled, box),
            write=graded)
    if box.rank > 2:
        raise NotYetElaborated("a chart of rank %d is planned, not covered (D0b, CF10): a chart "
                               "has rank 2, and graded spacing serves every rank" % box.rank, None)
    names = tuple(frame)
    if len(names) != box.rank:
        raise SupportMismatch(
            "%r.chart takes one keyword per axis of the frame, %d for a support of rank %d; it "
            "got %s" % (box, box.rank, box.rank, ", ".join(names)),
            write="%r.chart(x=e_x, y=e_y)" % (box,))
    for name in names:
        if name in box._names:
            raise SupportMismatch(
                "the keyword %s of the chart is an axis of %r: the frame axes and the logical "
                "axes are two families, and their names differ (S6, D0b)" % (name, box),
                write="logical axes named otherwise, such as pops.Box(r=..., theta=...)")
        if name in SUPPORT_READINGS or name.startswith("_"):
            raise ReservedName("the frame axis %r of a chart takes the name of a reading of a "
                               "support" % name, write="the frame axes x and y")
    if names != FRAME_AXES[:box.rank]:
        raise NotYetElaborated(
            "a chart whose frame axes are named %s; this lot elaborates the frame axes x and y "
            "of the Cartesian frames, in that order (DN26)" % ", ".join(names), _G03)
    for name, value in frame.items():
        _chart_reads(value, box, "the component %s of the chart of %r" % (name, box))
    components = tuple(frame.values())
    reads = [_folded(component)[1] for component in components]
    for name, read in zip(names, reads, strict=True):
        if not read:
            raise SupportMismatch(
                "the component %s = %s of the chart of %r reads no coordinate: the image of the "
                "support would have no area, and its cells no measure (D0b)"
                % (name, text(frame[name]), box), write="components that read the coordinates")
    if all(len(read) == 1 for read in reads) and len(frozenset().union(*reads)) == box.rank:
        raise SupportMismatch(
            "the chart %s of %r only moves the nodes along each axis (separable: %s), so the "
            "domain stays a box and the spacing is a choice of the layout (D0b, DN27)"
            % (spelled, box, ", ".join("%s reads %s alone" % (name, box._names[next(iter(read))])
                                       for name, read in zip(names, reads, strict=True))),
            write=graded)
    unread = [box._names[index] for index in range(box.rank)
              if not any(index in read for read in reads)]
    if unread:
        raise SupportMismatch(
            "the chart %s of %r reads the coordinate %s in none of its components: the image of "
            "the support would have no area, and its cells no measure (D0b)"
            % (spelled, box, unread[0]), write="components that read every coordinate of σ")
    return names, components, _degenerate_pieces(box, components)


def _chart_reads(value: Any, box: Box, where: str) -> None:
    """A chart reads the coordinates of its box, numbers, pi and the words of a coordinate map."""
    if is_number(value):
        return
    if isinstance(value, MathConstant):
        if not _math.isfinite(value.value):
            raise SupportMismatch("%s reads %s" % (where, value.name),
                                  write="an expression of the coordinates, numbers and pi")
        return
    if isinstance(value, Coordinate):
        if value.support is not box:
            raise SupportMismatch(
                "%s reads the coordinate %s of %r; a chart reads the coordinates of %r, numbers "
                "and pi (D0b)" % (where, value.name, value.support, box),
                write="an expression of %s" % ", ".join("Ω.%s" % name for name in box._names))
        return
    if isinstance(value, Arith):
        if value.op == "**" and not is_number(value.args[1]):
            raise NotYetElaborated("%s, a power whose exponent is not a number, in %s"
                                   % (text(value), where), _N11)
        for argument in value.args:
            _chart_reads(argument, box, where)
        return
    if isinstance(value, Call):
        if value.word not in MAP_WORDS:
            raise NotYetElaborated("%s in %s" % (text(value), where), _N11)
        for argument in value.args:
            _chart_reads(argument, box, where)
        return
    raise SupportMismatch(
        "%s reads %s; a chart reads the coordinates of %r, numbers and pi (D0b)"
        % (where, text(value), box),
        write="an expression of %s" % ", ".join("Ω.%s" % name for name in box._names))


def _apply(word: str, numbers: list) -> float:
    try:
        if word in MAP_WORDS:
            return float(MAP_WORDS[word](*numbers))
        left, right = numbers
        if word == "+":
            return left + right
        if word == "-":
            return left - right
        if word == "*":
            return left * right
        if word == "/":
            return left / right
        return _math.pow(left, right)
    except (ArithmeticError, ValueError):
        return _math.nan


def _folded(value: Any, fixed: int | None = None, bound: float = 0.0) -> tuple:
    """The number of an expression of a chart, or None, and the indices of the logical
    coordinates that it reads, the coordinate ``fixed`` set to ``bound`` and the products by an
    exact zero folded (DN27, DN28)."""
    none: frozenset = frozenset()
    if is_number(value):
        return float(value), none
    if isinstance(value, MathConstant):
        return float(value.value), none
    if isinstance(value, Coordinate):
        if value.index == fixed:
            return float(bound), none
        return None, frozenset((value.index,))
    parts = [_folded(argument, fixed, bound) for argument in value.args]
    if isinstance(value, Arith) and value.op == "neg":
        number, reads = parts[0]
        return (None if number is None else -number), reads
    numbers = [number for number, _ in parts]
    if isinstance(value, Arith):
        left, right = numbers
        if value.op == "*" and (left == 0.0 or right == 0.0):
            return 0.0, none
        if value.op == "/" and left == 0.0 and right != 0.0:
            return 0.0, none
        if value.op == "**" and left == 0.0 and right is not None and right > 0:
            return 0.0, none
    if all(number is not None for number in numbers):
        return _apply(value.op if isinstance(value, Arith) else value.word, numbers), none
    return None, frozenset().union(*(reads for _, reads in parts))


def _degenerate_pieces(box: Box, components: tuple) -> tuple:
    """The pieces on which, the logical coordinate of their axis set to its bound and the products
    by an exact zero folded, no component of the chart reads the other coordinate (DN28)."""
    pieces = []
    for index, (name, (lower, upper, periodic)) in enumerate(
            zip(box._names, box._bounds, strict=True)):
        if periodic:
            continue
        for side, bound in (("min", lower), ("max", upper)):
            if not any(_folded(component, index, float(bound))[1] for component in components):
                pieces.append("%s_%s" % (name, side))
    return tuple(pieces)
