"""A scheme of one's own: ``@pops.scheme`` and the entities that PoPS supplies (spec 0.3, 2.6.2,
2.6.4, 2.6.6; SU5, SU7, A4, IR25). Lot 2 of N15.

``@pops.scheme def s(face, term, opt=d)`` records a function. Its parameters named after an entity
(``face``, ``cell``, ``term``, ``axis``) are supplied by PoPS; every other parameter is an option,
given by keyword at the call, ``s(opt=v)``, which records the options and runs nothing (SU5).

PoPS runs a scheme once per key, when ``pops.Numerics`` receives it, with ``term`` bound to the key,
and a face scheme once per axis of its support, so that ``face.axis`` is Python data. The entities
record what the body reads as leaves of an expression (``face.left(u)``, ``face.normal``,
``cell.shift(x, -1).value(u)``); the body returns one builder of ``pops.discrete``. The case lowers
the builder when it is built (2.6.6, row of ``@pops.scheme``):

* a ``FaceFlux`` under a ``Flux`` key is the b6 finite volume of the key whose face is a
  ``riemann.User`` body: ``face.left(u)`` and ``face.right(u)`` are the two traces, ``face.normal``
  the canonical normal, ``dot(term.law(states={u: face.left(u)}), face.normal)`` the face input
  ``fL`` (``fR`` on the right), and the stability bound of the face is ``abs(s)`` for
  ``speed=s``, else the face input ``s``, the larger spectral radius of R9 that ``cell.speed``
  falls back to (SD44; review of the lot note, N1). A face scheme runs once per axis, and its
  traces on the axes must agree once ``face.normal`` is the canonical normal, since
  ``riemann.User`` has one body (N3);
* a ``Reconstruct``, the ``reconstruction=`` of a face flux, is a ``reconstruction.User`` body:
  ``cell.shift(axis, k).value(u)`` is the sample at offset k along the axis. Its
  ``formal_order`` is the bound that the stencil proves, the number of cells read along the axis,
  not a claimed order (§6.9, C46; review N2 and Q1).

Every other read of 2.6.4 is importable and refused with ``NotYetElaborated``, which names its lot.
"""
from __future__ import annotations

import inspect as _inspect
from fractions import Fraction
from typing import Any

from ._derive import derivative
from ._expr import Arith, Call, Expr, Predicate, Relation, is_number, text, walk
from ._geometry import Coordinate, MathConstant
from ._method import ENTITY_PARAMETERS
from ._model import DiffusiveFlux, Flux, ModelConstant, Parameter, Quantity, Term
from ._numerics import SchemeValue, TermView
from ._refusal import (
    AmbiguousElaboration, NotAPopsOperation, NotYetElaborated, RealizationMismatch,
    SupportMismatch, user_site,
)

# The entities that a scheme of this lot receives; the others of SU5 belong to later lots.
SCHEME_ENTITIES = ("face", "cell", "term", "axis")
_ENTITY_OWNER = {
    "edge": "a later lot of N15 with the edge fluxes of G06",
    "depth": "a later lot of N15 with the ghosts of G04",
    "interval": "a later lot of N15 with the interval fluxes of D11",
    "incident": "a later lot of N15 with the graph reductions of G08",
    "region": "a later lot of N15 with the regions of level 3 (SU15)",
}
_DIFFUSIVE = ("a later lot of N15 with a b6 face route that covers a diffusive flux and reads the "
              "spacing (C31)")
_GEOMETRY = "a later lot of N15 with the geometry reads of §4.3 (D18)"
_STENCIL = ("a later lot of N15 with the stencil faces of N7a (a face of this lot reads its two "
            "cells; a wider stencil is a Reconstruct)")
_BUILDERS = "a later lot of N15 with the keys that this builder covers"
_N11 = "N11 (expression closure, C1)"
_PER_AXIS = ("a later lot of N15 with a face lowered per axis (riemann.User has one body for the "
             "axes)")


# --- the leaves that the entities record -------------------------------------------------------

class SchemeLeaf(Expr):
    """A read of an entity: the trace of a quantity on a side of the face, the normal of the face,
    or the value of a quantity at a cell along the axis of a reconstruction."""

    __slots__ = ("key", "quantity", "_spelling", "expansion")

    def __init__(self, key: tuple, quantity: Any, spelling: str, expansion: Any = None) -> None:
        self.key = key
        self.quantity = quantity
        self._spelling = spelling
        self.expansion = expansion

    def _text(self, precedence: int) -> str:
        return self._spelling


class LawValue(Expr):
    """``term.law(states=...)``: the flux of the key at a state, a vector of the components of its
    axes, which ``dot(term.law(...), face.normal)`` projects on the normal. ``side`` is the side
    of the face when the state is the bare trace ``face.left(u)`` or ``face.right(u)``."""

    __slots__ = ("value", "side")

    def __init__(self, value: Any, side: str | None) -> None:
        self.value = value
        self.side = side

    def children(self) -> tuple:
        return (self.value,)

    def _text(self, precedence: int) -> str:
        return text(self.value, precedence)


def substitute(value: Any, table: dict) -> Any:
    """``value`` with each quantity of ``table`` replaced by its value, the other nodes kept."""
    if isinstance(value, Quantity):
        for quantity, replacement in table.items():
            if quantity is value:
                return replacement
        return value
    if isinstance(value, LawValue):
        return substitute(value.value, table)
    if isinstance(value, Arith):
        return Arith(value.op, tuple(substitute(arg, table) for arg in value.args))
    if isinstance(value, Call):
        return Call(value.word, tuple(substitute(arg, table) for arg in value.args))
    if isinstance(value, Predicate):
        return Predicate(value.op, *(substitute(arg, table) for arg in value.args))
    return value


# --- the normal form of term.expect (CC2) ------------------------------------------------------

def _atom(value: Any) -> tuple:
    if isinstance(value, (Quantity, ModelConstant, Parameter, Coordinate, MathConstant)):
        return ("symbol", id(value))
    if isinstance(value, SchemeLeaf):
        return ("leaf", value.key)
    if isinstance(value, Call):
        return ("call", value.word, tuple(normal_form(arg) for arg in value.args))
    if isinstance(value, Predicate):
        return ("predicate", value.op, tuple(normal_form(arg) for arg in value.args))
    raise NotYetElaborated("term.expect over %s, outside the algebra that this lot normalizes"
                           % text(value), _N11)


def _times(left: dict, right: dict) -> dict:
    out: dict = {}
    for mono_a, coef_a in left.items():
        for mono_b, coef_b in right.items():
            powers = dict(mono_a)
            for atom, exponent in mono_b:
                powers[atom] = powers.get(atom, 0) + exponent
            mono = tuple(sorted((atom, e) for atom, e in powers.items() if e != 0))
            out[mono] = out.get(mono, 0) + coef_a * coef_b
    return {mono: coef for mono, coef in out.items() if coef != 0}


def _plus(left: dict, right: dict, sign: int = 1) -> dict:
    out = dict(left)
    for mono, coef in right.items():
        out[mono] = out.get(mono, 0) + sign * coef
    return {mono: coef for mono, coef in out.items() if coef != 0}


def _polynomial(value: Any) -> dict:
    """A sum of monomials with exact rational coefficients over the atoms of the value."""
    if isinstance(value, LawValue):
        return _polynomial(value.value)
    if is_number(value):
        number = Fraction(value)
        return {(): number} if number != 0 else {}
    if isinstance(value, Arith):
        if value.op == "neg":
            return {mono: -coef for mono, coef in _polynomial(value.args[0]).items()}
        left, right = value.args
        if value.op in ("+", "-"):
            return _plus(_polynomial(left), _polynomial(right), 1 if value.op == "+" else -1)
        if value.op == "*":
            return _times(_polynomial(left), _polynomial(right))
        if value.op == "/":
            divisor = _polynomial(right)
            if len(divisor) == 1:
                ((mono, coef),) = divisor.items()
                inverse = {tuple((atom, -e) for atom, e in mono): 1 / coef}
                return _times(_polynomial(left), inverse)
            return _times(_polynomial(left), {((("quotient", normal_form(right)), 1),): 1})
        if is_number(right) and Fraction(right).denominator == 1 and 0 <= right <= 16:
            out: dict = {(): Fraction(1)}
            for _ in range(int(right)):
                out = _times(out, _polynomial(left))
            return out
        return {((("power", normal_form(left), normal_form(right)), 1),): Fraction(1)}
    return {((_atom(value), 1),): Fraction(1)}


def normal_form(value: Any) -> tuple:
    """The normal form of a value (CC2): a sum of monomials with exact rational coefficients over
    atoms (quantities, constants, parameters, the reads of the entities, and the calls of
    ``pops.math`` with normalized arguments).

    Two values with equal normal forms are equal. The converse fails: ``exp(a) * exp(b)`` and
    ``exp(a + b)`` have different forms, so ``term.expect`` refuses a key that is equal to its
    form but written otherwise, and its message says that the forms differ, not that the model
    changed. This is the one normalizer of the surface; the exact simplifier of N11 (lot 3)
    extends it."""
    return tuple(sorted(_polynomial(value).items(), key=repr))


# --- pops.discrete: the builders, the words of the kinds and forms, the speeds -----------------

class Word:
    """A word of ``pops.discrete.kinds`` or ``pops.discrete.forms``: Python data, compared with
    ``is``."""

    __slots__ = ("group", "name")

    def __init__(self, group: str, name: str) -> None:
        self.group = group
        self.name = name

    def __repr__(self) -> str:
        return "%s.%s" % (self.group, self.name)


class Words:
    """``pops.discrete.kinds`` and ``pops.discrete.forms``."""

    def __init__(self, group: str, names: tuple) -> None:
        self._group = group
        self._names = names
        for name in names:
            setattr(self, name, Word(group, name))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError("%s has the words %s, not %s"
                             % (self._group, ", ".join(self._names), name))

    def __repr__(self) -> str:
        return self._group


kinds = Words("kinds", ("Flux", "DiffusiveFlux", "Source", "Interaction", "Definition",
                        "Inventory", "Constraint", "Relation", "BoundaryRelation", "Gradient"))
forms = Words("forms", ("Value", "NoData", "Characteristic", "NormalFlux", "NormalComponent",
                        "General"))


class Builder:
    """What a scheme returns: a builder of 2.6.4 with its value."""

    word = ""

    def __init__(self, value: Any) -> None:
        _expression(value, "%s(value)" % self.word)
        self.value = value
        self.site = user_site(2)

    def __repr__(self) -> str:
        return "%s(%s)" % (self.word, text(self.value))


def _expression(value: Any, where: str) -> None:
    if isinstance(value, Relation):
        raise TypeError("%s takes a value; %s is a relation" % (where, value))
    if isinstance(value, Predicate):
        raise TypeError("%s takes a value; %s is a predicate, which where(p, a, b) chooses by"
                        % (where, value))
    if not (is_number(value) or isinstance(value, Expr)):
        raise TypeError("%s takes an expression of the entities, got %r" % (where, value))


class FaceFlux(Builder):
    """``FaceFlux(value, reconstruction=None, speed=None)``: the normal flux density of the key at
    the face, with the reconstruction of its traces and the normal speed that it used (SD44)."""

    word = "FaceFlux"

    def __init__(self, value: Any, reconstruction: Any = None, speed: Any = None,
                 left: Any = None, right: Any = None) -> None:
        if left is not None or right is not None:
            raise NotYetElaborated("FaceFlux(left=, right=), per-side contributions (2.6.8)",
                                   _BUILDERS)
        super().__init__(value)
        if speed is not None:
            _expression(speed, "FaceFlux(speed=)")
        if reconstruction is not None and not (
                isinstance(reconstruction, UserScheme) and reconstruction.scheme.is_cell):
            raise TypeError("FaceFlux(reconstruction=r) takes a scheme that returns Reconstruct, "
                            "called with its options, such as muscl(limiter=minmod); got %r"
                            % (reconstruction,))
        self.reconstruction = reconstruction
        self.speed = speed
        self.site = user_site(2)


class Reconstruct(Builder):
    """``Reconstruct(value)``: the trace of the key on the face on the positive side of ``axis``,
    the negative side mirroring it (2.6.4)."""

    word = "Reconstruct"


def _later_builder(word: str) -> Any:
    class Later:
        def __init__(self, *args: Any, **options: Any) -> None:
            raise NotYetElaborated("pops.discrete.%s" % word, _BUILDERS)

    Later.__name__ = Later.__qualname__ = word
    return Later


def _flux_key(term: Any, word: str) -> Flux:
    key = term._key if isinstance(term, TermEntity) else None
    if not isinstance(key, Flux):
        raise TypeError("%s(term, u, n) reads the term of a scheme under a flux, got %r"
                        % (word, term))
    return key


def _wave(term: Any, value: Any, normal: Any, word: str) -> Any:
    """The wave speed of the key at ``value`` along ``normal`` (R9, 2.6.6).

    The spaces of this lot have one component, so D_u(F(u)·n) is the number dF/du·n, which is the
    wave speed that the elaborated flux declares (``waves=``): its one eigenvalue is the slowest
    and the fastest speed, and its absolute value the spectral radius. A system takes its speeds
    from the wave speeds that its law declares, with the spaces of several components."""
    key = _flux_key(term, word)
    if not isinstance(normal, SchemeLeaf) or normal.key != ("normal",):
        raise TypeError("%s(term, u, n) takes the normal of the face, face.normal, as n; got %r"
                        % (word, normal))
    slope = derivative(key.value, key.quantity)
    if slope is None:
        raise AmbiguousElaboration(
            "%s(term, u, n) under %s = %s, whose wave speed dF/d%s is not derived in this lot, so "
            "the law declares none" % (word, key, text(key.value), key.quantity),
            write="FaceFlux(..., speed=s) with a speed of the scheme's own, or a flux whose "
                  "derivative is derived", site=user_site(3))
    return substitute(slope, {key.quantity: value}) * normal


def spectral_radius(term: Any, value: Any, normal: Any) -> Any:
    """``spectral_radius(term, u, n)``: the largest |λ| of D_u(F(u)·n) at u, from the wave speeds
    that R9 derives; |dF/du n| for a scalar flux.

    At a bare trace, ``face.left(u)`` or ``face.right(u)``, it is a read of the face, which lowers
    to that expansion; and ``largest`` of the radii at the two traces is the face input ``s`` of
    ``riemann.User``, the larger spectral radius that b6 generates from the same waves."""
    radius = abs(_wave(term, value, normal, "spectral_radius"))
    if isinstance(value, SchemeLeaf) and value.key[0] == "trace":
        side = value.key[1]
        return SchemeLeaf(("radius", side), None,
                          "spectral_radius(term, face.%s(%s), face.normal)"
                          % (side, value.quantity), radius)
    return radius


def slowest_speed(term: Any, value: Any, normal: Any) -> Any:
    """``slowest_speed(term, u, n)``: the smallest λ of D_u(F(u)·n) at u (R9)."""
    return _wave(term, value, normal, "slowest_speed")


def fastest_speed(term: Any, value: Any, normal: Any) -> Any:
    """``fastest_speed(term, u, n)``: the largest λ of D_u(F(u)·n) at u (R9)."""
    return _wave(term, value, normal, "fastest_speed")


def is_radius_pair(value: Any) -> bool:
    """Whether a value is ``largest`` of the spectral radii at the two traces: the input ``s``."""
    return (isinstance(value, Call) and value.word == "largest" and len(value.args) == 2
            and all(isinstance(arg, SchemeLeaf) for arg in value.args)
            and {arg.key for arg in value.args} == {("radius", "left"), ("radius", "right")})


def _nonnegative(value: Any) -> bool:
    """Whether a speed is nonnegative by its form, so that b6 reads it as its bound unchanged."""
    if is_radius_pair(value) or (isinstance(value, SchemeLeaf) and value.key[0] == "radius"):
        return True
    if isinstance(value, Call) and value.word == "abs":
        return True
    if isinstance(value, Call) and value.word == "largest":
        return any(_nonnegative(arg) for arg in value.args)
    if isinstance(value, Call) and value.word == "smallest":
        return all(_nonnegative(arg) for arg in value.args)
    return is_number(value) and value >= 0


def _is_normal(value: Any) -> bool:
    return isinstance(value, SchemeLeaf) and value.key == ("normal",)


def dot(a: Any, b: Any) -> Any:
    """``dot(a, b)`` of 2.6.3, as this lot elaborates it: ``dot(term.law(...), face.normal)``, the
    normal component of the flux of the key, which a face scheme runs once per axis to read
    (review of the lot note, Q2). ``dot(term.law(states={u: face.left(u)}), face.normal)`` is the
    face input F(uL)·n. Any other ``dot`` belongs to N11."""
    if _is_normal(a) and not _is_normal(b):
        a, b = b, a
    if _is_normal(b):
        if isinstance(a, LawValue):
            if a.side is not None:
                return SchemeLeaf(("flux", a.side), None,
                                  "dot(term.law(states={u: face.%s(u)}), face.normal)" % a.side)
            return a.value * b
        spelled = text(a) if isinstance(a, Expr) or is_number(a) else repr(a)
        raise SupportMismatch(
            "dot(e, face.normal) projects a flux on the normal, and %s is a scalar" % spelled,
            write="dot(term.law(states=...), face.normal) for the normal flux of the key, or the "
                  "scalar itself", site=user_site(3))
    raise NotYetElaborated("dot(%s, %s)" % (text(a) if isinstance(a, Expr) else repr(a),
                                            text(b) if isinstance(b, Expr) else repr(b)),
                           _N11, site=user_site(3))


# --- the entities ------------------------------------------------------------------------------

def _own_quantity(key: Term, value: Any, read: str) -> Quantity:
    if value is key.quantity:
        return value
    if isinstance(value, Quantity):
        raise NotYetElaborated("%s of %s, a quantity other than the one of the key %s" % (
            read, value, key), "a later lot of N15 with the record states of N8")
    raise TypeError("%s takes the quantity of the key, %s, got %r" % (read, key.quantity, value))


def _traced(key: Term, value: Any, side: str, spelling: str) -> Any:
    """``value`` read on one side of the face: each quantity replaced by its trace."""
    if isinstance(value, Quantity):
        quantity = _own_quantity(key, value, spelling)
        return SchemeLeaf(("trace", side), quantity, "face.%s(%s)" % (side, quantity))
    if isinstance(value, Expr):
        traces = {node: _traced(key, node, side, spelling) for node in walk(value)
                  if isinstance(node, Quantity)}
        return substitute(value, traces)
    if is_number(value):
        return value
    raise TypeError("%s takes a quantity or an expression of quantities, got %r"
                    % (spelling, value))


class _Unread:
    """The reads of an entity that this lot does not elaborate."""

    _later: dict = {}
    _entity = ""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._later:
            raise NotYetElaborated("%s.%s" % (self._entity, name), self._later[name])
        raise AttributeError("%s has no read %s (2.6.4)" % (self._entity, name))


class FaceCell(_Unread):
    """``face.left_cell`` and ``face.right_cell``: the two cells of the face."""

    _entity = "face.left_cell"
    _later = {"shift": _STENCIL, "width": _GEOMETRY, "center": _GEOMETRY, "lower": _GEOMETRY,
              "upper": _GEOMETRY, "measure": _GEOMETRY, "displacement": _GEOMETRY,
              "gradient": _DIFFUSIVE, "load": _BUILDERS, "speed": _BUILDERS,
              "parent": "a later lot of N15 with the hierarchies of N12",
              "level": "a later lot of N15 with the hierarchies of N12",
              "lower_face": _STENCIL, "upper_face": _STENCIL}

    def __init__(self, face: FaceEntity, side: str) -> None:
        object.__setattr__(self, "_face", face)
        object.__setattr__(self, "_side", side)
        object.__setattr__(self, "_entity", "face.%s_cell" % side)

    def value(self, quantity: Any) -> Any:
        return _traced(self._face._key, _own_quantity(self._face._key, quantity,
                                                      "%s.value" % self._entity),
                       self._side, "%s.value" % self._entity)

    def __repr__(self) -> str:
        return self._entity


class FaceEntity(_Unread):
    """``face`` in a scheme that returns ``FaceFlux``."""

    _entity = "face"
    _later = {"distance": _DIFFUSIVE, "measure": _GEOMETRY, "normal_gradient": _DIFFUSIVE,
              "gradient": _DIFFUSIVE, "inner": "a later lot of N15 with the ghosts of G04",
              "flux": "a later lot of N15 with the face fluxes read by another scheme (X1)",
              "velocity": "a later lot of N15 with the moving configurations of D18 and D21"}

    def __init__(self, key: Term, axis: Coordinate) -> None:
        self._key = key
        self.axis = axis
        self.left_cell = FaceCell(self, "left")
        self.right_cell = FaceCell(self, "right")

    @property
    def normal(self) -> Any:
        return SchemeLeaf(("normal",), None, "face.normal")

    def left(self, value: Any) -> Any:
        return _traced(self._key, value, "left", "face.left")

    def right(self, value: Any) -> Any:
        return _traced(self._key, value, "right", "face.right")

    def average(self, value: Any) -> Any:
        return (self.left(value) + self.right(value)) / 2

    def __repr__(self) -> str:
        return "face"


class CellEntity(_Unread):
    """``cell`` in a scheme that returns ``Reconstruct``: a cell at an offset along the axis."""

    _entity = "cell"
    _later = {"width": _GEOMETRY, "center": _GEOMETRY, "lower": _GEOMETRY, "upper": _GEOMETRY,
              "measure": _GEOMETRY, "displacement": _GEOMETRY, "gradient": _DIFFUSIVE,
              "load": _BUILDERS, "speed": _BUILDERS, "lower_face": _STENCIL,
              "upper_face": _STENCIL,
              "parent": "a later lot of N15 with the hierarchies of N12",
              "level": "a later lot of N15 with the hierarchies of N12"}

    def __init__(self, key: Term, axis: Coordinate, offset: int = 0) -> None:
        self._key = key
        self._axis = axis
        self._offset = offset

    def shift(self, axis: Any, k: Any = 1) -> CellEntity:
        if axis is not self._axis:
            if isinstance(axis, Coordinate):
                raise NotYetElaborated("cell.shift(%s, k) along an axis other than the axis of "
                                       "the reconstruction" % axis, _STENCIL)
            raise TypeError("cell.shift(axis, k) takes the axis entity, got %r" % (axis,))
        if type(k) is not int:
            raise TypeError("cell.shift(axis, k) takes an integer k, got %r" % (k,))
        return CellEntity(self._key, self._axis, self._offset + k)

    def value(self, quantity: Any) -> Any:
        quantity = _own_quantity(self._key, quantity, "cell.value")
        spelling = "cell.value(%s)" % quantity if self._offset == 0 else \
            "cell.shift(%s, %d).value(%s)" % (self._axis, self._offset, quantity)
        return SchemeLeaf(("sample", self._offset), quantity, spelling)

    def __repr__(self) -> str:
        return "cell" if self._offset == 0 else "cell.shift(%s, %d)" % (self._axis, self._offset)


class TermEntity(_Unread):
    """``term`` in a scheme: the key that the scheme is placed under."""

    _entity = "term"
    _later = {"quantities": "a later lot of N15 with the record states of N8",
              "unknowns": "a later lot of N15 with the constraints of N10",
              "gradient_variables": _DIFFUSIVE, "first_order_inputs": _DIFFUSIVE,
              "operator": "a later lot of N15 with the constraints of N10",
              "load": "a later lot of N15 with the constraints of N10",
              "data": "a later lot of N15 with the boundary relations of G04",
              "flux": "a later lot of N15 with the boundary relations of G04"}

    def __init__(self, key: Term) -> None:
        self._key = key

    @property
    def kind(self) -> Word:
        return getattr(kinds, self._key.kind)

    @property
    def quantity(self) -> Quantity:
        return self._key.quantity

    @property
    def form(self) -> Any:
        raise AmbiguousElaboration("term.form is the form of a boundary relation, and the key %s "
                                   "is a %s" % (self._key, self._key.kind),
                                   write="term.kind, a word of pops.discrete.kinds")

    def law(self, states: Any = None, gradients: Any = None) -> Any:
        key = self._key
        if gradients is not None:
            raise NotYetElaborated("term.law(gradients=...)", _DIFFUSIVE)
        if isinstance(key, DiffusiveFlux):
            raise NotYetElaborated("term.law of the diffusive flux %s" % key, _DIFFUSIVE)
        if states is None:
            return LawValue(key.value, None)
        if not isinstance(states, dict):
            raise TypeError("term.law(states={q: value}) takes a table keyed by quantities")
        table = {}
        for quantity, value in states.items():
            _own_quantity(key, quantity, "term.law(states=)")
            if not (is_number(value) or isinstance(value, Expr)):
                raise TypeError("term.law(states={%s: v}) takes a value, got %r"
                                % (quantity, value))
            table[quantity] = value
        side = None
        (trace,) = table.values() if len(table) == 1 else (None,)
        if isinstance(trace, SchemeLeaf) and trace.key[0] == "trace":
            side = trace.key[1]
        return LawValue(substitute(key.value, table), side)

    def expect(self, form: Any) -> None:
        """Refuse the scheme unless the key has the form (CC2); return nothing."""
        key = self._key
        _expression(form, "term.expect(form)")
        if normal_form(key.value) != normal_form(form):
            raise RealizationMismatch(
                "the scheme expects the key %s to be %s, and it is %s"
                % (key, text(form), text(key.value)),
                write="a scheme whose term.expect states the form of %s, or the key that it "
                      "expects" % key, site=user_site(2))

    def __repr__(self) -> str:
        return "term"


# --- @pops.scheme ------------------------------------------------------------------------------

class SurfaceScheme:
    """A function decorated with ``@pops.scheme``: its body, its entities and its options."""

    __slots__ = ("body", "name", "entities", "options", "defaults", "site")

    def __init__(self, body: Any) -> None:
        if not callable(body) or not hasattr(body, "__code__"):
            raise TypeError("@pops.scheme decorates a function def name(face, term, ...)")
        name = body.__name__
        entities, options, defaults = [], [], {}
        for parameter in _inspect.signature(body).parameters.values():
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                raise TypeError("@pops.scheme: %s takes named options, not *%s (SU5)"
                                % (name, parameter.name))
            if parameter.name in ENTITY_PARAMETERS:
                if parameter.name in _ENTITY_OWNER:
                    raise NotYetElaborated("a scheme that receives %s" % parameter.name,
                                           _ENTITY_OWNER[parameter.name])
                if parameter.name not in SCHEME_ENTITIES:
                    raise TypeError("@pops.scheme: PoPS supplies %s to a method or an "
                                    "acceptance, not to a scheme" % parameter.name)
                entities.append(parameter.name)
                continue
            options.append(parameter.name)
            if parameter.default is not parameter.empty:
                defaults[parameter.name] = parameter.default
        if "term" not in entities:
            raise TypeError("@pops.scheme: %s takes term, the key that the scheme is placed "
                            "under: def %s(face, term)" % (name, name))
        if ("face" in entities) == ("cell" in entities):
            raise TypeError("@pops.scheme: %s takes face, for a scheme that returns FaceFlux, or "
                            "cell and axis, for a scheme that returns Reconstruct (2.6.4)" % name)
        if "cell" in entities and "axis" not in entities:
            raise NotYetElaborated("a scheme over cell without axis, which returns CellValue, "
                                   "CellRate or CellResidual", _BUILDERS)
        if "face" in entities and "axis" in entities:
            raise TypeError("@pops.scheme: a face scheme reads its axis as face.axis; %s takes "
                            "face, not axis" % name)
        self.body = body
        self.name = name
        self.entities = tuple(entities)
        self.options = tuple(options)
        self.defaults = defaults
        self.site = user_site(2)

    @property
    def is_cell(self) -> bool:
        return "cell" in self.entities

    def __call__(self, *args: Any, **options: Any) -> UserScheme:
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
        return UserScheme(self, {**self.defaults, **options}, tuple(options))

    def __repr__(self) -> str:
        return "@pops.scheme %s" % self.name


def scheme(function: Any = None, **options: Any) -> Any:
    """``@pops.scheme``, written bare (SU16)."""
    if options or function is None:
        raise TypeError("@pops.scheme takes no option and is written bare, @pops.scheme (SU16)")
    if isinstance(function, SurfaceScheme):
        return function
    return SurfaceScheme(function)


class Traced:
    """What a scheme gave under one key: its builder, and the reconstruction that it names."""

    __slots__ = ("builder", "reconstruction")

    def __init__(self, builder: Any, reconstruction: Any) -> None:
        self.builder = builder
        self.reconstruction = reconstruction


class UserScheme(SchemeValue):
    """A scheme of one's own called with its options: the value of a key of ``schemes=``."""

    def __init__(self, scheme_: SurfaceScheme, options: dict, stated: tuple = ()) -> None:
        super().__init__(scheme_.name, **options)
        self.scheme = scheme_
        self.stated = tuple(stated)
        self.site = user_site(3)
        self._traced: list = []

    def __repr__(self) -> str:
        return "%s(%s)" % (self.scheme.name, ", ".join(
            "%s=%s" % (name, _option_text(self.options[name])) for name in self.stated))

    def option_choices(self) -> list:
        """Each option with its value and its origin, stated or default, for ``pops.inspect``."""
        return [(name, _option_text(self.options[name]),
                 "stated" if name in self.stated else "default") for name in self.scheme.options]

    def reconstruction_of(self, view: TermView) -> Any:
        """The ``reconstruction.User`` of this cell scheme under the key of ``view``, as the
        ``reconstruction=`` of a face flux of the library (2.6.9)."""
        if not self.scheme.is_cell:
            raise TypeError("reconstruction= takes a scheme that returns Reconstruct; %r returns "
                            "FaceFlux" % self)
        return realize_reconstruction(self, self.traced(view.term), view)

    @property
    def kinds(self) -> tuple:  # type: ignore[override]
        # check_key refuses, with its own message, a cell scheme (which covers no key) and a face
        # scheme under a diffusive flux (which a later lot elaborates).
        return ("Flux", "DiffusiveFlux")

    def check_key(self, key: Term) -> None:
        """Run the scheme under its key, as ``pops.Numerics`` receives it (SU5)."""
        if self.scheme.is_cell:
            raise RealizationMismatch(
                "%s returns Reconstruct, the trace of a face, which covers no term" % self,
                write="the scheme as the reconstruction= of a face scheme, such as "
                      "rusanov(reconstruction=%s)" % self, site=self.site)
        if isinstance(key, DiffusiveFlux):
            raise NotYetElaborated("the scheme %s under the diffusive flux %s" % (self, key),
                                   _DIFFUSIVE, site=self.site)
        self.traced(key)

    def _run(self, entities: dict, key: Term) -> Any:
        arguments = {name: entities[name] for name in self.scheme.entities}
        arguments.update(self.options)
        result = self.scheme.body(**arguments)
        if result is None:
            raise TypeError("the scheme %s returns no builder; a scheme returns one builder of "
                            "pops.discrete, such as FaceFlux(value)" % self.scheme.name)
        if isinstance(result, tuple):
            raise TypeError("the scheme %s returns a tuple; a scheme returns one builder of "
                            "pops.discrete (SU14)" % self.scheme.name)
        expected = Reconstruct if self.scheme.is_cell else FaceFlux
        if not isinstance(result, expected):
            raise RealizationMismatch(
                "the scheme %s returns %r under %s, a %s, which %s covers"
                % (self.scheme.name, result, key, key.kind, expected.word),
                write="%s(value)" % expected.word, site=self.scheme.site)
        return result

    def traced(self, key: Term) -> Traced:
        """The builder that the scheme gives under ``key``, run once per key (SU5), and for a face
        scheme once per axis of the support, ``face.axis`` being Python data."""
        for known, traced in self._traced:
            if known is key:
                return traced
        axes = key.quantity.support.axes
        if self.scheme.is_cell:
            (axis,) = axes[:1]
            builder = self._run({"cell": CellEntity(key, axis), "axis": axis,
                                 "term": TermEntity(key)}, key)
            _reads(builder.value, ("sample",), self, "cell")
            traced = Traced(builder, None)
        else:
            builder = None
            for axis in axes:
                ran = self._run({"face": FaceEntity(key, axis), "term": TermEntity(key)}, key)
                _reads(ran.value, _FACE_READS, self, "face")
                if ran.speed is not None:
                    _reads(ran.speed, _FACE_READS, self, "face")
                if builder is None:
                    builder = ran
                elif not _same_face(builder, ran):
                    raise NotYetElaborated(
                        "the scheme %s, whose face flux on the axis %s differs from the one on "
                        "the axis %s" % (self, axis, axes[0]), _PER_AXIS, site=self.site)
            assert builder is not None
            recon = builder.reconstruction
            traced = Traced(builder, None if recon is None else recon.traced(key))
        self._traced.append((key, traced))
        return traced

    def realize(self, view: TermView) -> Any:
        from pops.numerics import reconstruction, riemann, variables
        from pops.numerics.spatial import FiniteVolume

        traced = self.traced(view.term)
        builder = traced.builder
        lower = view.lower

        def body(uL: Any, uR: Any, fL: Any, fR: Any, s: Any, *, n: Any) -> tuple:
            return (lower(builder.value, _face_inputs(uL, uR, fL, fR, n, s)),)

        if builder.speed is None:
            # The larger spectral radius of R9 at the two traces: the fallback of cell.speed
            # (2.6.4, SD44), and the bound that b6 requires of every user face (N1).
            def stability(uL: Any, uR: Any, fL: Any, fR: Any, s: Any, *, n: Any) -> Any:
                del uL, uR, fL, fR, n
                return s
        else:
            # A speed may be signed (upwind follows the orientation of the face); b6 reads a
            # nonnegative bound, so a speed that is not nonnegative by its form is taken in
            # absolute value.
            signed = not _nonnegative(builder.speed)

            def stability(uL: Any, uR: Any, fL: Any, fR: Any, s: Any, *, n: Any) -> Any:
                bound = lower(builder.speed, _face_inputs(uL, uR, fL, fR, n, s))
                return abs(bound) if signed else bound

        recon = reconstruction.FirstOrder() if traced.reconstruction is None else \
            realize_reconstruction(builder.reconstruction, traced.reconstruction, view)
        return FiniteVolume(flux=view.elaborated, variables=variables.Conservative(view.state),
                            reconstruction=recon,
                            riemann=riemann.User(body=body, state=view.state,
                                                 stability=stability))

    def describe(self) -> str:
        """The b6 builder, for ``pops.inspect``; the options are in the repr of the value."""
        if self.scheme.is_cell:
            return "reconstruction.User(the Reconstruct of %s)" % self.scheme.name
        first = self._traced[0][1] if self._traced else None
        builder = None if first is None else first.builder
        reconstruction = "FirstOrder()"
        if first is not None and first.reconstruction is not None:
            reconstruction = ("reconstruction.User(the Reconstruct of %r, formal_order=%d, the "
                              "bound of its stencil, not claimed)"
                              % (first.builder.reconstruction,
                                 formal_order(first.reconstruction.builder.value)))
        if builder is None or builder.speed is None:
            speed = "s, the larger spectral radius of R9 at the two traces"
        elif _nonnegative(builder.speed):
            speed = text(builder.speed)
        else:
            speed = "abs(%s)" % text(builder.speed)
        return ("FiniteVolume(variables=Conservative(u), reconstruction=%s, riemann=riemann.User("
                "body=the FaceFlux of %s, stability=%s))" % (reconstruction, self.scheme.name,
                                                             speed))


_FACE_READS = ("trace", "normal", "flux", "radius")


def _option_text(value: Any) -> str:
    if isinstance(value, (Expr, SchemeValue)) or is_number(value) or value is None:
        return text(value) if isinstance(value, Expr) else repr(value)
    name = getattr(value, "__name__", None)
    return name if isinstance(name, str) else repr(value)


def _same_face(one: FaceFlux, other: FaceFlux) -> bool:
    """Whether two runs of a face scheme on two axes give the same face (N3)."""
    if normal_form(one.value) != normal_form(other.value):
        return False
    if (one.speed is None) != (other.speed is None):
        return False
    if one.speed is not None and normal_form(one.speed) != normal_form(other.speed):
        return False
    return repr(one.reconstruction) == repr(other.reconstruction)


def _face_inputs(uL: Any, uR: Any, fL: Any, fR: Any, n: Any, s: Any) -> dict:
    return {("trace", "left"): uL[0], ("trace", "right"): uR[0], ("flux", "left"): fL[0],
            ("flux", "right"): fR[0], ("normal",): n[0], ("speed",): s}


def _reads(value: Any, allowed: tuple, scheme_: Any, entity: str, site: Any = None) -> None:
    """Check the reads of a value that a scheme (or a function of face) returns."""
    site = scheme_.scheme.site if site is None else site
    for node in walk(value):
        if isinstance(node, LawValue):
            raise SupportMismatch(
                "%s returns %s, the flux of the key, a vector, where a scalar is taken"
                % (scheme_, node),
                write="dot(term.law(...), face.normal), its normal component", site=site)
        if isinstance(node, SchemeLeaf) and node.key[0] not in allowed:
            raise TypeError("%s reads %s, which a %s scheme does not receive"
                            % (scheme_, node, entity))
        if isinstance(node, Quantity):
            raise TypeError("%s reads the quantity %s itself; a %s scheme reads it through its "
                            "entity, %s" % (
                                scheme_, node, entity,
                                "face.left(%s) or face.right(%s)" % (node, node)
                                if entity == "face" else "cell.value(%s)" % node))
        if isinstance(node, Term):
            raise NotYetElaborated("the term %s read in the value of a scheme" % node,
                                   "a later lot of N15 with the face fluxes read by another "
                                   "scheme (X1)", site=site)


def formal_order(value: Any) -> int:
    """The ``formal_order`` that b6 records for a reconstruction: the number of cells that the
    trace reads along the axis, a bound that the stencil proves (a trace of k cell values is at
    most of order k), never below the order, so never short of the AMR lookahead that b6 derives
    from it. It is not a claimed order: that is ``Order(...)`` of ``pops.claims`` (§6.9, C46).
    MUSCL reads three cells and is reported 3, its order being 2 (review of the lot note, Q1)."""
    offsets = {node.key[1] for node in walk(value)
               if isinstance(node, SchemeLeaf) and node.key[0] == "sample"}
    return max(1, len(offsets))


def realize_reconstruction(value: Any, traced: Traced, view: TermView) -> Any:
    """The ``reconstruction.User`` of a ``Reconstruct`` under the key of ``view``."""
    from pops.numerics import reconstruction

    expression = traced.builder.value
    lower = view.lower

    def body(sample: Any) -> tuple:
        offsets = {node.key[1] for node in walk(expression)
                   if isinstance(node, SchemeLeaf) and node.key[0] == "sample"}
        inputs = {("sample", k): sample(k)[0] for k in sorted(offsets)}
        return (lower(expression, inputs),)

    return reconstruction.User(body, state=view.state, formal_order=formal_order(expression),
                               name=value.word)


def face_function(function: Any, view: TermView, where: str) -> Any:
    """A plain function of ``face`` given as an option, ``upwind(speed=f)`` (SU5): run once per
    axis on the face of the key of ``view``, as a face scheme runs; return the function of the b6
    face of ``fluxes`` that lowers its value (``face.left(Uh)``, ``face.right(Uh)``,
    ``face.normal``)."""
    check_plain_function(function, where)
    key = view.term
    value = None
    for axis in key.quantity.support.axes:
        ran = function(FaceEntity(key, axis))
        _expression(ran, "the value of %s" % where)
        _reads(ran, ("trace", "normal"), where, "face", site=user_site(2))
        if value is None:
            value = ran
        elif normal_form(value) != normal_form(ran):
            raise NotYetElaborated("%s, whose value on the axis %s differs from the one on the "
                                   "first axis" % (where, axis), _PER_AXIS)
    lower = view.lower

    def lowered(face: Any) -> Any:
        return lower(value, {("trace", "left"): face.left(view.state)[0],
                             ("trace", "right"): face.right(view.state)[0],
                             ("normal",): face.normal[0]})

    return lowered


def check_plain_function(value: Any, where: str) -> None:
    """A function of an entity given as a value is a plain function, passed uncalled (SU5)."""
    if isinstance(value, (SurfaceScheme, UserScheme)):
        raise NotAPopsOperation("%s takes a plain function of face, passed uncalled; %r is a "
                                "scheme" % (where, value), write="def f(face): ... and %s=f"
                                % where.split("(")[0])
    if not callable(value) or isinstance(value, type):
        raise TypeError("%s takes a function of face, def f(face): ..., got %r" % (where, value))


__all__ = [
    "Builder", "CellEntity", "FaceEntity", "FaceFlux", "Reconstruct", "SchemeLeaf",
    "LawValue", "SurfaceScheme", "TermEntity", "UserScheme", "Word", "Words",
    "check_plain_function", "dot", "formal_order", "is_radius_pair",
    "fastest_speed", "forms", "kinds", "normal_form", "realize_reconstruction", "scheme",
    "slowest_speed", "spectral_radius", "substitute",
]
