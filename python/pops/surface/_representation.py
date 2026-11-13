"""A representation of one's own: ``@pops.representation``, ``pops.Coefficients`` and ``pops.Dual``
(spec 0.3, D1, D1b, D2, SU5, SU9; N16 design note rev 3, sections 3, 4.4 and 5.5). Lot 1 of N16.

``@pops.representation def r(cell, o=d)`` records a function. PoPS supplies its entity ``cell``; every
other parameter is an option, given by keyword at the call, ``r(o=v)``, which records the options and
runs nothing (SU5). The function returns ``pops.Coefficients(size=, on=, decode=, pairing=,
encode=, kind=)``, or ``pops.Dual(R)`` for the loads of a representation R.

PoPS runs the function once per key of ``represent=``, when ``pops.Numerics`` receives it, at the
rank of the key's support: the cell entity, ``decode`` and ``encode`` are traced into the leaves of
contract @2 (``pops.representations.definition``), and the result is a
``RepresentationDefinition`` whose identity is the digest of its normal forms, never of a Python
name (P10). Point dependence and linearity are derived on the normal form of decode (IR15).

This lot elaborates no form of its own. A definition equal to the one of ``CellAverage()``, one
constant mode of scale 1 and no rule, takes the b6 route of ``pops.CellAverage()``
(``ConservativeCellAverage()``): the transitional table of N16 5.5, which matches the whole
definition identity, never a name. Every other definition is refused with the lot that elaborates
it: a total by G03 lot M1 (the surface route of a total, with the moving configuration), a sample and
a stated rule by the N16 lot that retires the transitional table (the projection by encode), a
point-dependent decode and the loads by N16 lot 2, node carriers by N16 lot 3, face and edge carriers
by G08.
"""
from __future__ import annotations

import inspect as _inspect
from fractions import Fraction
from typing import Any

from ._refusal import (
    AmbiguousElaboration, NotAPopsOperation, NotYetElaborated, OutOfScope, Site, SurfaceRefusal,
    user_site,
)

_M1 = ("G03 lot M1 (the surface route of a total, with the moving configuration), with the oracle "
       "tests of a total on a graded layout")
_TABLE = ("the N16 lot that retires the transitional table of CellAverage() (the projection by "
          "encode, line 4 of plan/routes_transitoires.md), with the oracle tests on a graded layout")
_LOT_2 = ("N16 lot 2 (point-dependent cell carriers: the current point, the Gram matrix of the "
          "stated rule and the Riesz map)")
_LOT_2_LOADS = "N16 lot 2 (loads, cell.load and the Riesz map of D2)"
_LOT_3 = "N16 lot 3 (node carriers, after G08)"
_G08 = "G08 (face and edge carriers, which evolve by EdgeFlux)"
_KINDS = ("coefficient", "average", "total", "sample", "face integral", "edge circulation")


# --- the carriers, words of pops.discrete -------------------------------------------------------

class Carrier:
    """``Cell``, ``Node``, ``Face`` and ``Edge`` of ``pops.discrete``: the entity that carries the
    coefficients of a representation (``pops.Coefficients(on=...)``)."""

    __slots__ = ("word", "entity")

    def __init__(self, word: str, entity: str) -> None:
        self.word = word
        self.entity = entity

    def __repr__(self) -> str:
        return self.word


Cell = Carrier("Cell", "cell")
Node = Carrier("Node", "node")
Face = Carrier("Face", "face")
Edge = Carrier("Edge", "edge")
CARRIERS = (Cell, Node, Face, Edge)


# --- the entities that a representation receives -------------------------------------------------

def _definition_module() -> Any:
    from pops.representations import definition
    return definition


class _Vector:
    """``point.reference`` or ``point.position`` on a support of rank above one: a vector read by
    axis, ``point.reference[a]`` with a Python integer a (E4)."""

    __slots__ = ("_leaf", "_rank", "_spelling")

    def __init__(self, leaf: Any, rank: int, spelling: str) -> None:
        self._leaf = leaf
        self._rank = rank
        self._spelling = spelling

    def __getitem__(self, axis: Any) -> Any:
        if type(axis) is not int or not 0 <= axis < self._rank:
            raise OutOfScope("%s[%r] reads an axis outside a cell of rank %d"
                             % (self._spelling, axis, self._rank),
                             write="%s[a] with a in cell.axes" % self._spelling)
        return self._leaf[axis]

    def __repr__(self) -> str:
        return self._spelling


class _NodeEntity:
    """A corner node of ``cell.nodes``: its reference point ``node.reference``."""

    __slots__ = ("index", "_rank")

    def __init__(self, index: int, rank: int) -> None:
        self.index = index
        self._rank = rank

    @property
    def reference(self) -> Any:
        leaf = _definition_module().node_reference[self.index]
        if self._rank == 1:
            return leaf[0]
        return _Vector(leaf, self._rank, "node.reference")

    def __repr__(self) -> str:
        return "cell.nodes[%d]" % self.index


class RepresentationCell:
    """The entity ``cell`` of a representation: ``cell.measure``, ``cell.width(axis)``,
    ``cell.axes`` (Python data) and ``cell.nodes`` (the 2^r corners, lower before upper in the
    lexicographic order of the axes)."""

    __slots__ = ("rank",)

    def __init__(self, rank: int) -> None:
        self.rank = rank

    @property
    def measure(self) -> Any:
        return _definition_module().cell.measure

    def width(self, axis: Any) -> Any:
        if type(axis) is not int or not 0 <= axis < self.rank:
            raise OutOfScope("cell.width(%r) reads an axis outside a cell of rank %d"
                             % (axis, self.rank), write="cell.width(a) with a in cell.axes")
        return _definition_module().cell.width(axis)

    @property
    def axes(self) -> tuple:
        return tuple(range(self.rank))

    @property
    def nodes(self) -> tuple:
        return tuple(_NodeEntity(index, self.rank) for index in range(2 ** self.rank))

    def __getattr__(self, name: str) -> Any:
        raise NotYetElaborated("cell.%s in a representation; a representation reads cell.measure, "
                               "cell.width(axis), cell.axes and cell.nodes" % name, _LOT_2)

    def __repr__(self) -> str:
        return "cell"


class _Coefficients:
    """The entity ``coefficients`` of decode: ``coefficients[k]``, k in ``range(size)``."""

    __slots__ = ("_size", "_carrier")

    def __init__(self, size: int, carrier: Carrier) -> None:
        self._size = size
        self._carrier = carrier

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, _NodeEntity):
            raise NotYetElaborated("coefficients[e] with e a node of cell.nodes", _LOT_3)
        if type(index) is not int or not 0 <= index < self._size:
            raise OutOfScope("coefficients[%r] is outside the local set range(%d)"
                             % (index, self._size),
                             write="coefficients[k] with k a Python integer in range(size)")
        return _definition_module().coefficients[index]

    def index(self, name: Any) -> Any:
        raise NotYetElaborated("coefficients.index(%r) in sum_over" % (name,), _LOT_2)

    def __repr__(self) -> str:
        return "coefficients"


class _Point:
    """The entity ``point`` of decode: ``point.reference`` in [-1, 1]^r and ``point.position``,
    numbers on a support of rank one and vectors read by axis otherwise (2.6.3)."""

    __slots__ = ("_rank",)

    def __init__(self, rank: int) -> None:
        self._rank = rank

    def _read(self, which: str) -> Any:
        leaf = getattr(_definition_module().point, which)
        if self._rank == 1:
            return leaf[0]
        return _Vector(leaf, self._rank, "point." + which)

    @property
    def reference(self) -> Any:
        return self._read("reference")

    @property
    def position(self) -> Any:
        return self._read("position")

    def __repr__(self) -> str:
        return "point"


# --- the construct -------------------------------------------------------------------------------

def _plain_function(value: Any, option: str, parameters: tuple) -> Any:
    """A decode or an encode: a plain function of the named entities, passed uncalled (SU5)."""
    from ._scheme import SurfaceScheme, UserScheme

    if isinstance(value, (SurfaceScheme, UserScheme, SurfaceRepresentation,
                          RepresentationValue)):
        raise NotAPopsOperation("pops.Coefficients(%s=...) takes a plain function of %s, passed "
                                "uncalled; %r is not one" % (option, " and ".join(parameters),
                                                             value),
                                write="def %s(%s): ... and %s=%s" % (
                                    option, ", ".join(parameters), option, option))
    if not callable(value) or isinstance(value, type) or not hasattr(value, "__code__"):
        raise TypeError("pops.Coefficients(%s=...) takes a function def %s(%s): ..., got %r"
                        % (option, option, ", ".join(parameters), value))
    return value


def _signature(function: Any, option: str, parameters: tuple, site: Site | None) -> None:
    names = tuple(_inspect.signature(function).parameters)
    if names != parameters:
        raise TypeError("%s: pops.Coefficients(%s=...) takes a function of the entities %s, "
                        "def %s(%s), got def %s(%s)"
                        % (site, option, " and ".join(parameters), option,
                           ", ".join(parameters), function.__name__, ", ".join(names)))


class Coefficients:
    """``pops.Coefficients(size=1, on=Cell, decode=f, pairing=None, encode=None,
    kind="coefficient")``: what a representation returns (D1b, N16 3.1).

    The options are checked when the representation runs; the functions are traced at
    ``pops.Numerics``, at the rank of the key's support."""

    __slots__ = ("size", "on", "decode", "pairing", "encode", "kind", "site")

    def __init__(self, *, decode: Any, size: Any = 1, on: Any = Cell, pairing: Any = None,
                 encode: Any = None, kind: Any = "coefficient") -> None:
        if type(size) is not int or size < 1:
            raise TypeError("pops.Coefficients(size=...) takes a Python integer n >= 1, the "
                            "number of coefficients of each carrier entity (E4), got %r" % (size,))
        if not isinstance(on, Carrier):
            raise TypeError("pops.Coefficients(on=...) takes a carrier of pops.discrete, Cell, "
                            "Node, Face or Edge, got %r" % (on,))
        if kind not in _KINDS:
            raise TypeError("pops.Coefficients(kind=...) is one of %s, documentation that "
                            "pops.inspect and the claims read, never a selector; got %r; the loads "
                            "of a representation R are pops.Dual(R)"
                            % (", ".join(repr(k) for k in _KINDS), kind))
        if pairing is not None:
            from ._quadrature import Quadrature

            if not isinstance(pairing, Quadrature):
                raise TypeError("pops.Coefficients(pairing=...) takes the rule of the pairing, a "
                                "pops.Quadrature (written in the case or taken from "
                                "pops.lib.quadrature, D0), got %r" % (pairing,))
        self.decode = _plain_function(decode, "decode", ("coefficients", "point"))
        self.encode = None if encode is None else _plain_function(encode, "encode",
                                                                  (on.entity, "field"))
        self.size = size
        self.on = on
        self.pairing = pairing
        self.kind = kind
        self.site = user_site(2)

    def __repr__(self) -> str:
        return "pops.Coefficients(size=%d, on=%r, kind=%r)" % (self.size, self.on, self.kind)


class Dual:
    """``pops.Dual(R)``: the loads of R, coefficient s equal to <field, phi_s>, with the carrier
    and the size of R and a covariant type (D2); decoded through the Riesz map of R."""

    __slots__ = ("of", "site")

    def __init__(self, of: Any) -> None:
        if isinstance(of, SurfaceRepresentation):
            raise NotAPopsOperation("pops.Dual(%s) holds the representation uncalled; a call "
                                    "states its options (SU5)" % of.name,
                                    write="pops.Dual(%s())" % of.name)
        if isinstance(of, Dual):
            raise TypeError("pops.Dual(R) takes a representation called with its options, not "
                            "the loads of one")
        if not isinstance(of, RepresentationValue):
            raise TypeError("pops.Dual(R) takes a representation called with its options, such "
                            "as CellAverage() of pops.lib.representations, got %r" % (of,))
        self.of = of
        self.site = user_site(2)

    def __repr__(self) -> str:
        return "pops.Dual(%r)" % (self.of,)


class SurfaceRepresentation:
    """A function decorated with ``@pops.representation``: its body, its name and its options."""

    __slots__ = ("body", "name", "options", "defaults", "site")

    def __init__(self, body: Any) -> None:
        if not callable(body) or not hasattr(body, "__code__"):
            raise TypeError("@pops.representation decorates a function def name(cell, ...)")
        name = body.__name__
        parameters = list(_inspect.signature(body).parameters.values())
        for parameter in parameters:
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                raise TypeError("@pops.representation: %s takes named options, not *%s (SU5)"
                                % (name, parameter.name))
        if not parameters or parameters[0].name != "cell":
            raise TypeError("@pops.representation: PoPS supplies the entity cell first, "
                            "def %s(cell, ...); every other parameter is an option" % name)
        options = [parameter.name for parameter in parameters[1:]]
        self.body = body
        self.name = name
        self.options = tuple(options)
        self.defaults = {parameter.name: parameter.default for parameter in parameters[1:]
                         if parameter.default is not parameter.empty}
        self.site = user_site(2)

    def __call__(self, *args: Any, **options: Any) -> RepresentationValue:
        if args:
            raise TypeError("%s takes its options by keyword, %s(%s=...) (SU5)"
                            % (self.name, self.name, self.options[0] if self.options
                               else "option"))
        unknown = sorted(set(options) - set(self.options))
        if unknown:
            raise TypeError("%s has the options %s, not %s"
                            % (self.name, ", ".join(self.options) or "none", ", ".join(unknown)))
        missing = [name for name in self.options
                   if name not in options and name not in self.defaults]
        if missing:
            raise TypeError("%s needs the options %s, given by keyword: %s(%s)"
                            % (self.name, ", ".join(missing), self.name,
                               ", ".join("%s=..." % name for name in missing)))
        return RepresentationValue(self, {**self.defaults, **options}, tuple(options),
                                   user_site(2))

    def __repr__(self) -> str:
        return "@pops.representation %s" % self.name


def representation(function: Any = None, **options: Any) -> Any:
    """``@pops.representation``, written bare (SU16)."""
    if options or function is None:
        raise TypeError("@pops.representation takes no option and is written bare, "
                        "@pops.representation (SU16)")
    if isinstance(function, SurfaceRepresentation):
        return function
    return SurfaceRepresentation(function)


class RepresentationValue:
    """A representation called with its options: the value of a key of ``represent=``.

    ``definition(rank)`` runs the function at that rank and traces it into contract @2; it is
    computed once per rank."""

    __slots__ = ("representation", "options", "stated", "site", "_definitions", "_returned")

    def __init__(self, representation: SurfaceRepresentation, options: dict, stated: tuple,
                 site: Site | None) -> None:
        self.representation = representation
        self.options = options
        self.stated = stated
        self.site = site
        self._definitions: dict = {}
        self._returned: dict = {}

    @property
    def name(self) -> str:
        return self.representation.name

    def coefficients(self, rank: int) -> Any:
        """What the function returns at a rank: ``pops.Coefficients`` or ``pops.Dual``. The body
        runs once per rank (SU5)."""
        if rank in self._returned:
            return self._returned[rank]
        returned = self.representation.body(RepresentationCell(rank), **self.options)
        if not isinstance(returned, (Coefficients, Dual)):
            raise TypeError("%s: the representation %s returns pops.Coefficients(...) or "
                            "pops.Dual(R), got %r" % (self.site, self.name, returned))
        self._returned[rank] = returned
        return returned

    def definition(self, rank: int) -> Any:
        """The ``RepresentationDefinition`` of contract @2 at a rank, or the refusal of the lot
        that elaborates its carrier or its reads."""
        if rank not in self._definitions:
            self._definitions[rank] = _trace(self.coefficients(rank), rank, self.site)
        return self._definitions[rank]

    def __repr__(self) -> str:
        options = ", ".join("%s=%r" % (name, self.options[name]) for name in self.stated)
        return "%s(%s)" % (self.name, options)


def _body(returned: Any, what: str, site: Site | None) -> Any:
    if isinstance(returned, _definition_module().Body) or (
            isinstance(returned, (int, float, Fraction)) and not isinstance(returned, bool)):
        return returned
    raise NotYetElaborated("%s that returns %r, a value other than the arithmetic of "
                           "coefficients, point and the reads of cell" % (what, returned),
                           "N16 lot 2 with the expressions of N11", site=site)


def _trace(returned: Any, rank: int, site: Site | None) -> Any:
    definition = _definition_module()
    if isinstance(returned, Dual):
        inner = returned.of.coefficients(rank)
        if isinstance(inner, Dual):
            raise TypeError("%s: pops.Dual(R) takes a representation whose coefficients are "
                            "primal; %r gives the loads of another" % (site, returned.of))
        of = returned.of.definition(rank)
        return definition.RepresentationDefinition(
            decode=of.decode, rank=rank, size=of.size, carrier=of.carrier, encode=of.encode,
            measure=of.measure, pairing=inner.pairing, kind="load", dual_of=of)
    coefficients = returned
    if coefficients.on is Face or coefficients.on is Edge:
        raise NotYetElaborated("pops.Coefficients(on=%r)" % coefficients.on, _G08, site=site)
    if coefficients.on is Node:
        raise NotYetElaborated("pops.Coefficients(on=Node)", _LOT_3, site=site)
    _signature(coefficients.decode, "decode", ("coefficients", "point"), site)
    decode = _body(coefficients.decode(_Coefficients(coefficients.size, coefficients.on),
                                       _Point(rank)), "a decode", site)
    encode = None
    if coefficients.encode is not None:
        _signature(coefficients.encode, "encode", (coefficients.on.entity, "field"), site)
        values = coefficients.encode(RepresentationCell(rank), definition.field[0])
        if isinstance(values, (list, tuple)):
            values = tuple(values)
        else:
            values = (values,)
        if len(values) != coefficients.size:
            raise TypeError("%s: the encode of a cell carrier of size=%d returns %d values, one "
                            "per coefficient (E4); write size=%d or return %d values"
                            % (site, coefficients.size, len(values), len(values),
                               coefficients.size))
        if coefficients.size > 1:
            raise NotYetElaborated("a stated encode of size %d" % coefficients.size, _LOT_2,
                                   site=site)
        encode = _body(values[0], "an encode", site)
    try:
        return definition.RepresentationDefinition(
            decode=decode, rank=rank, size=coefficients.size, carrier=coefficients.on.entity,
            encode=encode, pairing=coefficients.pairing, kind=coefficients.kind)
    except (TypeError, ValueError) as error:
        raise type(error)("%s: %s" % (site, error)) from None


# --- the elaboration of lot 1 --------------------------------------------------------------------

def is_cell_average(definition: Any) -> bool:
    """The transitional table of N16 5.5: the definition identity of ``CellAverage()`` at its rank,
    matched whole, never by a name (CF2, FC3)."""
    return definition.identity == _definition_module().cell_average(definition.rank).identity


def elaborate(value: Any, quantity: Any) -> Any:
    """The definition of the representation of ``quantity``, which this lot elaborates on the b6
    route of ``pops.CellAverage()``, or the refusal of the lot that elaborates it."""
    if isinstance(value, SurfaceRepresentation):
        raise NotAPopsOperation("represent={%s: %s} holds the representation uncalled; a call "
                                "states its options and runs nothing (SU5)"
                                % (quantity, value.name), write="%s()" % value.name)
    rank = quantity.support.rank
    returned = value.coefficients(rank)
    definition = value.definition(rank)
    if isinstance(returned, Dual):
        raise NotYetElaborated("represent={%s: %r}, the loads of a representation"
                               % (quantity, value), _LOT_2_LOADS, site=value.site)
    if definition.point_dependent:
        if returned.pairing is None:
            raise AmbiguousElaboration(
                "the decode of %r reads the point and states no rule: the Gram matrix and the "
                "projection depend on the rule, a choice that changes the result (SU9)" % (value,),
                write="pops.Coefficients(..., pairing=Q), Q a pops.Quadrature or a rule of "
                      "pops.lib.quadrature", site=value.site)
        raise NotYetElaborated("represent={%s: %r}, whose decode reads the point"
                               % (quantity, value), _LOT_2, site=value.site)
    if returned.encode is None and definition.linear is not True:
        from ._solver import PropertyNotDerived

        raise PropertyNotDerived(
            "the projection that encodes %r needs the Gram matrix of a decode linear in the "
            "coefficients, and IR15 does not prove the decode of %r linear" % (value, value),
            write="a decode linear in coefficients, or encode=", site=value.site)
    if is_cell_average(definition):
        return definition
    from pops.representations.definition import constant_mode_scale

    scale = constant_mode_scale(definition)
    if scale is not None:
        # A total (c = 1 / cell.measure) is served by G03 lot M1; a sample or a stated rule differs
        # from the table's average by its encode or its projection rule only, and is served by the
        # lot that gives the projection by encode (review of #800, R1; driver 13/11 12:10).
        raise NotYetElaborated("represent={%s: %r}, one constant mode of kind %r%s"
                               % (quantity, value, definition.kind,
                                  "" if definition.pairing is None else " with a stated rule"),
                               _M1 if scale == "unit" else _TABLE, site=value.site)
    raise NotYetElaborated("represent={%s: %r}, whose constant mode is not 1 or 1/cell.measure"
                           % (quantity, value), _LOT_2, site=value.site)


def check(value: Any, quantity: Any, site: Site | None) -> Any:
    """``elaborate`` with the line of ``pops.Numerics`` named in its refusals."""
    try:
        return elaborate(value, quantity)
    except SurfaceRefusal as refusal:
        raise refusal.traced_at(site).with_traceback(None) from None


__all__ = [
    "CARRIERS", "Carrier", "Cell", "Coefficients", "Dual", "Edge", "Face", "Node",
    "RepresentationCell", "RepresentationValue", "SurfaceRepresentation", "check", "elaborate",
    "is_cell_average", "representation",
]
