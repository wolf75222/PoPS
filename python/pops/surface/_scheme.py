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
  not a claimed order (§6.9, C46; review N2 and Q1);
* a ``CellRate`` of a scheme over ``cell`` without ``axis``, under a ``Source`` key, is the value
  of a b6 named source, read at the cell: ``cell.value(u)`` is the cell value (the decoded density
  under a total), and the selection of the source is realized by ``StateStorage()`` (N15,
  FaceFlux/CellRate lot FC-1, note §3.2 to §3.4).

Every other read of 2.6.4 is importable and refused with ``NotYetElaborated``, which names its lot.
"""
from __future__ import annotations

import inspect as _inspect
from fractions import Fraction
from typing import Any

from ._expr import Arith, Call, Expr, Predicate, Relation, is_number, operand, text, walk
from ._geometry import Coordinate, MathConstant
from ._method import ENTITY_PARAMETERS
from ._model import (
    BoundaryRelation, DiffusiveFlux, Flux, ModelConstant, Parameter, Quantity, Source, Term,
)
from ._numerics import SchemeValue, TermView
from ._refusal import (
    AmbiguousElaboration, NotAPopsOperation, NotYetElaborated, RealizationMismatch,
    SupportMismatch, user_site,
)

# The entities that a scheme of this lot receives; the others of SU5 belong to later lots.
SCHEME_ENTITIES = ("face", "cell", "term", "axis", "depth")
_ENTITY_OWNER = {
    "edge": "a later lot of N15 with the edge fluxes of G06",
    "interval": "a later lot of N15 with the interval fluxes of D11",
    "incident": "a later lot of N15 with the graph reductions of G08",
    "region": "a later lot of N15 with the regions of level 3 (SU15)",
}
# A diffusive face of this lot is realized by b6's StencilFlux (N3 lot 1), which is Cartesian,
# uniform and periodic: a face of a diffusive flux that is not -k grad(u) of its own quantity
# belongs to G05. Its walls (G04) are refused with the boundary of its support, and its hierarchies
# (N4, N12) with pops.AMR (``_elaborate.boundaries``, ``_program.AMR``).
_DIFFUSIVE_FORMS = "a later lot of N15 with the coupled and drift realizations of G05"
_SPACING = ("a later lot of N15 with a b6 face route that covers a transport flux and reads the "
            "spacing (C31)")
_GEOMETRY = "a later lot of N15 with the geometry reads of §4.3 (D18)"
_STENCIL = ("a later lot of N15 with the stencil faces of N7a (a face of this lot reads its two "
            "cells; a wider stencil is a Reconstruct)")
_BUILDERS = "a later lot of N15 with the keys that this builder covers"
# N16 lot 2 finds its seams by this name (handover point 8; the SU-T2 tests mark them `# <-`).
_N16_LOT_2 = "N16 lot 2"
_N16_REGION_FORM = ("N16 lot 2, with the region form of CellRate (a cell rate that reads a stencil "
                    "is not a b6 source)")
_N16_PER_SIDE = "N16 lot 2, with the per-side values of D6 (2.6.8)"
_N16_VOLUME = "N16 lot 2, with the volume term of the weak form of div (D1b)"
_N16_INTEGRALS = "N16 lot 2, with the integrals of 4.3 (a total)"

# Refusal tables of the surface (CF2, §21.7): the reads of 2.6.4 that a later lot elaborates and
# whose names the core does not otherwise spell, keyed by the read, with the sentence of their lot.
_LATER_FACE = {
    "gradient": "a later lot of N15 with the gradient schemes of face.gradient(e, scheme=)",
    "velocity": "a later lot of N15 with the moving configurations of D18 and D21",
}
_LATER_CELL = {
    "gradient": "a later lot of N15 with the gradient schemes of cell.gradient(e, scheme=)",
}
_LATER_TERM: dict = {}
_LATER_KINDS = {
    "Gradient": "a later lot of N15 with the schemes given to face.gradient(e, scheme=) and "
                "cell.gradient(e, scheme=)",
}
# The same tables for the entities of a boundary policy (lot B), whose reads G04 elaborates.
_LATER_BOUNDARY_FACE = {
    "gradient": "a later lot of N15 with the boundary relations of G04",
    "normal_gradient": "a later lot of N15 with the boundary relations of G04",
    "velocity": "a later lot of N15 with the moving configurations of D18 and D21",
}
_LATER_INNER = {
    "gradient": "a later lot of N15 with the boundary relations of G04",
}
_LATER_BOUNDARY_TERM = {
    "gradient_variables": "a later lot of N15 with the boundary relations of G04",
}
_N11 = "N11 (expression closure, C1)"
_PER_AXIS_RECONSTRUCTION = ("a later lot of N15 with a reconstruction lowered per axis "
                            "(reconstruction.User has one body for the axes)")
_PER_AXIS = ("a later lot of N15 with a face lowered per axis (riemann.User has one body for the "
             "axes)")


# --- the leaves that the entities record -------------------------------------------------------

class SchemeLeaf(Expr):
    """A read of an entity: the trace of a quantity on a side of the face, the normal of the face,
    or the value of a quantity at a cell along the axis of a reconstruction."""

    __slots__ = ("key", "quantity", "_spelling", "expansion", "rank")

    def __init__(self, key: tuple, quantity: Any, spelling: str, expansion: Any = None,
                 rank: int = 1) -> None:
        self.key = key
        self.quantity = quantity
        self._spelling = spelling
        self.expansion = expansion
        self.rank = rank

    def _text(self, precedence: int) -> str:
        return self._spelling


class LoadValue(Expr):
    """``cell.load(e)``: the load of e on the cell, ∫_K e dμ under the rule of the representation
    (N16 4.3, D2); under one constant mode its rate is the mean of e under that rule (N15
    FaceFlux/CellRate lot FC-2, note §3.6)."""

    __slots__ = ("value",)

    key = ("load",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def children(self) -> tuple:
        return (self.value,)

    def _text(self, precedence: int) -> str:
        return "cell.load(%s)" % text(self.value)


class LawValue(Expr):
    """``term.law(states=...)``: the flux of the key at a state, a vector of the components of its
    axes, which ``dot(term.law(...), face.normal)`` projects on the normal. ``side`` is the side
    of the face when the state is the bare trace ``face.left(u)`` or ``face.right(u)``."""

    __slots__ = ("value", "side", "components")

    def __init__(self, value: Any, side: str | None, components: tuple) -> None:
        self.value = value
        self.side = side
        self.components = components

    def children(self) -> tuple:
        return (self.value,)

    def _text(self, precedence: int) -> str:
        return text(self.value, precedence)


class DiffusiveLaw(LawValue):
    """``term.law(states=..., gradients=...)`` of a diffusive key -k grad(w), each ``grad(w)``
    given as ``g * face.normal``: the flux along the normal, whose ``dot`` with ``face.normal`` is
    its normal component ``value``, -k g, the normal being a unit vector (lot 1 of N3)."""


def _normal_factor(value: Any) -> Any:
    """The scalar g of ``g * face.normal`` (or ``face.normal * g``), else ``None``."""
    if isinstance(value, Arith) and value.op == "*":
        left, right = value.args
        if isinstance(right, SchemeLeaf) and right.key == ("normal",):
            return left
        if isinstance(left, SchemeLeaf) and left.key == ("normal",):
            return right
    return None


def _replace_gradients(value: Any, table: dict) -> Any:
    """``value`` with each ``grad(w)`` of ``table`` replaced by its normal component."""
    if isinstance(value, Call) and value.word == "grad":
        for variable, replacement in table.items():
            if value.args[0] is variable:
                return replacement
        return value
    if isinstance(value, Arith):
        return Arith(value.op, tuple(_replace_gradients(arg, table) for arg in value.args))
    if isinstance(value, Call):
        return Call(value.word, tuple(_replace_gradients(arg, table) for arg in value.args))
    return value


def _diffusive_law(key: Any, states: Any, gradients: Any) -> Any:
    """``term.law(states={q: v}, gradients={w: g * face.normal})`` of a diffusive key."""
    if gradients is None:
        raise RealizationMismatch(
            "the scheme reads term.law without gradients=, the law of a transport flux, and the "
            "key %s is a diffusive flux, whose law reads the gradients of %s"
            % (key, ", ".join(map(str, key.variables))),
            write="a scheme of the kind of the key, or term.law(states=..., gradients={w: ...}) "
                  "for each w of term.gradient_variables", site=user_site(3))
    if not isinstance(gradients, dict):
        raise TypeError("term.law(gradients={w: value}) takes a table keyed by quantities")
    table = {}
    for variable, value in gradients.items():
        if not any(variable is item for item in key.variables):
            raise TypeError("term.law(gradients={%s: ...}): %s is not a gradient variable of %s, "
                            "whose law reads the gradients of %s" % (
                                variable, variable, key, ", ".join(map(str, key.variables))))
        factor = _normal_factor(value)
        if factor is None:
            raise NotYetElaborated(
                "term.law(gradients={%s: %s}), a gradient that is not a scalar times face.normal"
                % (variable, text(value) if isinstance(value, Expr) else repr(value)),
                "a later lot of N15 with the gradient schemes of face.gradient(e, scheme=)")
        table[variable] = factor
    missing = [item for item in key.variables if not any(item is v for v in table)]
    if missing:
        raise TypeError("term.law(gradients=...) of %s omits %s, whose gradient the law reads"
                        % (key, ", ".join(map(str, missing))))
    value = _replace_gradients(key.value, table)
    if states is not None:
        if not isinstance(states, dict):
            raise TypeError("term.law(states={q: value}) takes a table keyed by quantities")
        for quantity, state in states.items():
            _own_quantity(key, quantity, "term.law(states=)")
            if not (is_number(state) or isinstance(state, Expr)):
                raise TypeError("term.law(states={%s: v}) takes a value, got %r"
                                % (quantity, state))
        value = substitute(value, dict(states))
    return DiffusiveLaw(value, None, ())


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
            if len(divisor) == 1 and () in divisor:
                # only a number divides exactly; a quotient by an expression stays an atom, so
                # that u*u*u/(2*u), undefined at u = 0, is not u**2/2 (review of #760)
                return _times(_polynomial(left), {(): 1 / divisor[()]})
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

    def __init__(self, group: str, names: tuple, later: dict | None = None) -> None:
        self._group = group
        self._names = names
        self._later = later or {}
        for name in names:
            setattr(self, name, Word(group, name))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._later:
            raise NotYetElaborated("pops.discrete.%s.%s" % (self._group, name), self._later[name])
        raise AttributeError("%s has the words %s, not %s"
                             % (self._group, ", ".join(self._names), name))

    def __repr__(self) -> str:
        return self._group


kinds = Words("kinds", ("Flux", "DiffusiveFlux", "Source", "Interaction", "Definition",
                        "Inventory", "Constraint", "Relation", "BoundaryRelation"), _LATER_KINDS)
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
                 left: Any = None, right: Any = None, volume: Any = None) -> None:
        if left is not None or right is not None:
            raise NotYetElaborated("FaceFlux(left=, right=), per-side contributions (2.6.8)",
                                   _N16_PER_SIDE)
        if volume is not None:
            raise NotYetElaborated("FaceFlux(volume=), the cell part of a weak divergence",
                                   _N16_VOLUME)
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


class CellRate(Builder):
    """``CellRate(value)``: the rate of the key's quantity at the cell, a density read at the
    centre under one constant mode (2.6.4; N16 4.3), or a load of ``cell.load`` (FC-2)."""

    word = "CellRate"


class Ghost(Builder):
    """``Ghost(value)``: the value of the ghost cell at depth k beyond a face of a piece, which
    ``face.inner(k)`` mirrors (D9, 2.6.4)."""

    word = "Ghost"


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
    slopes = key.slopes()
    if slopes is None:
        for node in walk(key.value):
            if isinstance(node, Call) and node.word in ("largest", "smallest"):
                raise NotYetElaborated("%s in a value of the model, whose derivative %s(term, u, "
                                       "n) reads" % (text(node), word), _N11, site=user_site(3))
        raise AmbiguousElaboration(
            "%s(term, u, n) under %s = %s, whose wave speed dF/d%s is not derived in this lot, so "
            "the law declares none" % (word, key, text(key.value), key.quantity),
            write="FaceFlux(..., speed=s) with a speed of the scheme's own, or a flux whose "
                  "derivative is derived", site=user_site(3))
    return normal_dot(tuple(substitute(slope, {key.quantity: value}) for slope in slopes), normal)


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


def _side_speed(word: str, term: Any, value: Any, normal: Any) -> Any:
    speed = _wave(term, value, normal, word)
    if isinstance(value, SchemeLeaf) and value.key[0] == "trace":
        side = value.key[1]
        return SchemeLeaf((word, side), None, "%s(term, face.%s(%s), face.normal)"
                          % (word, side, value.quantity), speed)
    return speed


def slowest_speed(term: Any, value: Any, normal: Any) -> Any:
    """``slowest_speed(term, u, n)``: the smallest λ of D_u(F(u)·n) at u (R9).

    At a bare trace it is a read of the face, which lowers to its expansion; and ``smallest`` of
    the slowest speeds at the two traces is the first of the face input ``speeds`` of
    ``riemann.User``, the slowest wave speed united over the two traces (N3 lot 1)."""
    return _side_speed("slowest_speed", term, value, normal)


def fastest_speed(term: Any, value: Any, normal: Any) -> Any:
    """``fastest_speed(term, u, n)``: the largest λ of D_u(F(u)·n) at u (R9); ``largest`` of the
    fastest speeds at the two traces is the second of the face input ``speeds``."""
    return _side_speed("fastest_speed", term, value, normal)


def speed_pair_index(value: Any) -> Any:
    """0 when a value is ``smallest`` of the slowest speeds at the two traces, 1 when it is
    ``largest`` of the fastest speeds there (the two entries of the face input ``speeds``), else
    ``None``."""
    for index, (fold, word) in enumerate((("smallest", "slowest_speed"),
                                          ("largest", "fastest_speed"))):
        if (isinstance(value, Call) and value.word == fold and len(value.args) == 2
                and all(isinstance(arg, SchemeLeaf) for arg in value.args)
                and {arg.key for arg in value.args} == {(word, "left"), (word, "right")}):
            return index
    return None


def reads_speed_pair(*values: Any) -> bool:
    """Whether a value reads the face input ``speeds``."""
    return any(speed_pair_index(node) is not None
               for value in values if value is not None for node in walk(value))


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


def normal_dot(components: tuple, normal: SchemeLeaf) -> Any:
    """The dot product of a vector given by its components on the axes with ``face.normal``:
    the component times the normal on a support of rank one (where a vector is written as its
    scalar), else c_1 n_1 + ... + c_d n_d over the components of the canonical normal, which is
    the component of the axis of the face and one body for every axis (lot 2b)."""
    if len(components) == 1:
        return components[0] * normal
    if len(components) != normal.rank:
        raise SupportMismatch(
            "a vector of %d components dotted with the normal of a support of rank %d"
            % (len(components), normal.rank),
            write="vector(%s), one component per axis" % ", ".join(["..."] * normal.rank),
            site=user_site(4))
    total: Any = None
    for index, component in enumerate(components):
        term = component * SchemeLeaf(("normal", index), None, "face.normal[%d]" % index)
        total = term if total is None else total + term
    return total


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
        if isinstance(a, DiffusiveLaw):
            return a.value
        if isinstance(a, LawValue):
            if a.side is not None:
                return SchemeLeaf(("flux", a.side), None,
                                  "dot(term.law(states={u: face.%s(u)}), face.normal)" % a.side)
            return normal_dot(a.components, b)
        if isinstance(a, Call) and a.word == "vector":
            # vector(...) built of the components on the axes, the velocity of upwind(speed=f)
            return normal_dot(tuple(a.args), b)
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
    _later_reads: dict = {}
    _entity = ""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._later:
            raise NotYetElaborated("%s.%s" % (self._entity, name), self._later[name])
        if name in self._later_reads:
            raise NotYetElaborated("%s.%s" % (self._entity, name), self._later_reads[name])
        raise AttributeError("%s has no read %s (2.6.4)" % (self._entity, name))


class FaceCell(_Unread):
    """``face.left_cell`` and ``face.right_cell``: the two cells of the face."""

    _entity = "face.left_cell"
    _later_reads = _LATER_CELL
    _later = {"shift": _STENCIL, "width": _GEOMETRY, "center": _GEOMETRY, "lower": _GEOMETRY,
              "upper": _GEOMETRY, "measure": _GEOMETRY, "displacement": _GEOMETRY,
              "load": _BUILDERS, "speed": _BUILDERS,
              "parent": "a later lot of N15 with the hierarchies of N12",
              "level": "a later lot of N15 with the hierarchies of N12",
              "lower_face": _STENCIL, "upper_face": _STENCIL}

    def __init__(self, face: FaceEntity, side: str) -> None:
        object.__setattr__(self, "_face", face)
        object.__setattr__(self, "_side", side)
        object.__setattr__(self, "_entity", "face.%s_cell" % side)

    def value(self, quantity: Any) -> Any:
        """The average of the cell on that side (2.6.4): the trace of the face where the face has
        no reconstruction; under a reconstruction it is a read of the stencil, which a later lot
        elaborates (review of #760, R1)."""
        quantity = _own_quantity(self._face._key, quantity, "%s.value" % self._entity)
        return SchemeLeaf(("cell", self._side), quantity,
                          "%s.value(%s)" % (self._entity, quantity))

    def __repr__(self) -> str:
        return self._entity


class FaceEntity(_Unread):
    """``face`` in a scheme that returns ``FaceFlux``."""

    _entity = "face"
    _later_reads = _LATER_FACE
    _later = {"measure": _GEOMETRY,
              "inner": "a later lot of N15 with the ghosts of G04",
              "flux": "a later lot of N15 with the face fluxes read by another scheme (X1)"}

    def __init__(self, key: Term, axis: Coordinate) -> None:
        self._key = key
        self.axis = axis
        self.left_cell = FaceCell(self, "left")
        self.right_cell = FaceCell(self, "right")

    @property
    def normal(self) -> Any:
        return SchemeLeaf(("normal",), None, "face.normal",
                          rank=self._key.quantity.support.rank)

    def left(self, value: Any) -> Any:
        return _traced(self._key, value, "left", "face.left")

    def right(self, value: Any) -> Any:
        return _traced(self._key, value, "right", "face.right")

    def average(self, value: Any) -> Any:
        return (self.left(value) + self.right(value)) / 2

    @property
    def distance(self) -> Any:
        """The distance between the centres of the two cells of the face (2.6.4): the spacing of
        the axis on the uniform tiling of this lot."""
        return SchemeLeaf(("distance",), None, "face.distance")

    def normal_gradient(self, value: Any) -> Any:
        """The two-point normal derivative of ``value`` at the face, (right - left) / distance
        (2.6.4)."""
        return (self.right(value) - self.left(value)) / self.distance

    def __repr__(self) -> str:
        return "face"


class CellEntity(_Unread):
    """``cell`` in a scheme that returns ``Reconstruct``: a cell at an offset along the axis."""

    _entity = "cell"
    _later_reads = _LATER_CELL
    _later = {"width": _GEOMETRY, "center": _GEOMETRY, "lower": _GEOMETRY, "upper": _GEOMETRY,
              "measure": _GEOMETRY, "displacement": _GEOMETRY,
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


class RateCell(_Unread):
    """``cell`` in a scheme over ``cell`` without ``axis``, which returns ``CellRate``: the cell
    of the key, read at one place (note §3.2)."""

    _entity = "cell"
    _later_reads = _LATER_CELL
    _later = {"width": _GEOMETRY, "center": _GEOMETRY, "lower": _GEOMETRY, "upper": _GEOMETRY,
              "measure": _GEOMETRY, "displacement": _GEOMETRY,
              "integral": _N16_INTEGRALS,
              "shift": _N16_REGION_FORM, "lower_face": _N16_REGION_FORM,
              "upper_face": _N16_REGION_FORM,
              "speed": _BUILDERS,
              "parent": "a later lot of N15 with the hierarchies of N12",
              "level": "a later lot of N15 with the hierarchies of N12"}

    def __init__(self, key: Term) -> None:
        self._key = key

    def value(self, quantity: Any) -> Any:
        quantity = _own_quantity(self._key, quantity, "cell.value")
        return SchemeLeaf(("sample", 0), quantity, "cell.value(%s)" % quantity)

    def load(self, value: Any) -> Any:
        """``cell.load(e)``: the load of e on the cell (N16 4.3), whose kind the builder types
        (``_builder_value.classify``)."""
        return LoadValue(operand(value, "cell.load(e)"))

    def __repr__(self) -> str:
        return "cell"


class TermEntity(_Unread):
    """``term`` in a scheme: the key that the scheme is placed under."""

    _entity = "term"
    _later = {"unknowns": "a later lot of N15 with the constraints of N10",
              "operator": "a later lot of N15 with the constraints of N10",
              "load": "a later lot of N15 with the constraints of N10",
              "data": "a later lot of N15 with the boundary relations of G04",
              "flux": "a later lot of N15 with the boundary relations of G04"}

    _later_reads = _LATER_TERM

    def __init__(self, key: Term) -> None:
        self._key = key

    @property
    def kind(self) -> Word:
        return getattr(kinds, self._key.kind)

    @property
    def quantity(self) -> Quantity:
        return self._key.quantity

    @property
    def quantities(self) -> list:
        """The quantities that the law of the key reads: its own one, in this lot (a record state
        is a later lot of N15 with the record states of N8)."""
        return [self._key.quantity]

    @property
    def gradient_variables(self) -> list:
        """The variables whose gradient the law of a diffusive key reads (2.6.4); none for a
        transport flux."""
        return list(getattr(self._key, "variables", ()))

    @property
    def first_order_inputs(self) -> list:
        """The fields whose gradient the law reads as a first-order transport input (2.6.4,
        SD51): none for the diffusive fluxes -k grad(u) of this lot, the others being refused
        when the model declares them (G05)."""
        return []

    @property
    def form(self) -> Any:
        raise AmbiguousElaboration("term.form is the form of a boundary relation, and the key %s "
                                   "is a %s" % (self._key, self._key.kind),
                                   write="term.kind, a word of pops.discrete.kinds")

    def law(self, states: Any = None, gradients: Any = None) -> Any:
        key = self._key
        if isinstance(key, DiffusiveFlux):
            return _diffusive_law(key, states, gradients)
        if gradients is not None:
            raise RealizationMismatch(
                "the scheme reads term.law(gradients=...), the law of a diffusive flux, and the "
                "key %s is a %s, whose law reads no gradient" % (key, key.kind),
                write="a scheme of the kind of the key", site=user_site(2))
        components = getattr(key, "components", (key.value,))
        if states is None:
            if isinstance(key, Source):
                return key.value
            return LawValue(key.value, None, components)
        if not isinstance(states, dict):
            raise TypeError("term.law(states={q: value}) takes a table keyed by quantities")
        table = {}
        for quantity, value in states.items():
            _own_quantity(key, quantity, "term.law(states=)")
            if not (is_number(value) or isinstance(value, Expr)):
                raise TypeError("term.law(states={%s: v}) takes a value, got %r"
                                % (quantity, value))
            table[quantity] = value
        if isinstance(key, Source):
            # The law of a source is a scalar of its quantity, not a vector of axes.
            return substitute(key.value, table)
        side = None
        (trace,) = table.values() if len(table) == 1 else (None,)
        if isinstance(trace, SchemeLeaf) and trace.key[0] == "trace":
            side = trace.key[1]
        return LawValue(substitute(key.value, table), side,
                        tuple(substitute(component, table) for component in components))

    def expect(self, form: Any, write: Any = None) -> None:
        """Refuse the scheme unless the key has the form (CC2); return nothing. ``write`` is the
        help of the refusal, what to write instead (D9 of N3); without it the help is generic."""
        key = self._key
        _expression(form, "term.expect(form)")
        if write is not None and not isinstance(write, str):
            raise TypeError("term.expect(form, write=...) takes the text of the help, got %r"
                            % (write,))
        if normal_form(key.value) != normal_form(form):
            raise RealizationMismatch(
                "the scheme expects the key %s to be %s, and it is %s"
                % (key, text(form), text(key.value)),
                write=write if write is not None else
                "a scheme whose term.expect states the form of %s, or the key that it expects"
                % key, site=user_site(2))

    def __repr__(self) -> str:
        return "term"


# --- the entities of a boundary policy (lot B) ---------------------------------------------------

_G04 = "a later lot of N15 with the boundary relations of G04"


class InnerCell(_Unread):
    """``face.inner(k)``: the k-th cell inward from a boundary face, the mirror of the ghost cell
    at depth k (D9)."""

    _later = {"shift": _STENCIL, "width": _GEOMETRY, "center": _GEOMETRY, "lower": _GEOMETRY,
              "upper": _GEOMETRY, "measure": _GEOMETRY, "displacement": _GEOMETRY,
              "load": _BUILDERS, "speed": _BUILDERS}
    _later_reads = _LATER_INNER

    def __init__(self, relation: BoundaryRelation, k: int) -> None:
        object.__setattr__(self, "_relation", relation)
        object.__setattr__(self, "_k", k)
        object.__setattr__(self, "_entity", "face.inner(%d)" % k)

    def value(self, quantity: Any) -> Any:
        relation = self._relation
        if quantity is not relation.quantity:
            if isinstance(quantity, Quantity):
                raise NotYetElaborated("%s.value(%s), a quantity other than the one of the "
                                       "relation %r" % (self._entity, quantity, relation),
                                       "a later lot of N15 with the record states of N8")
            raise TypeError("%s.value(q) takes the quantity of the relation, %s, got %r"
                            % (self._entity, relation.quantity, quantity))
        return SchemeLeaf(("inner", self._k), quantity,
                          "face.inner(%d).value(%s)" % (self._k, quantity))

    def __repr__(self) -> str:
        return self._entity


class BoundaryFaceEntity(_Unread):
    """``face`` in a boundary policy: a face of a piece of the relation, ``face.normal``
    pointing out of the support, ``face.inner(k)`` the k-th cell inward."""

    _entity = "face"
    _later = {"distance": _G04, "measure": _GEOMETRY, "average": _G04,
              "flux": "a later lot of N15 with the face fluxes read by another scheme (X1)",
              "left": _G04, "right": _G04, "left_cell": _G04, "right_cell": _G04}
    _later_reads = _LATER_BOUNDARY_FACE

    def __init__(self, relation: BoundaryRelation, piece: Any) -> None:
        self._relation = relation
        self.piece = piece
        self.axis = getattr(piece.support, piece.axis)

    @property
    def normal(self) -> Any:
        return SchemeLeaf(("normal",), None, "face.normal",
                          rank=self._relation.quantity.support.rank)

    def inner(self, k: Any) -> InnerCell:
        if type(k) is not int or k < 1:
            raise TypeError("face.inner(k) takes an integer k >= 1, the depth, got %r" % (k,))
        return InnerCell(self._relation, k)

    def __repr__(self) -> str:
        return "face"


class BoundaryTermEntity(_Unread):
    """``term`` in a boundary policy: the relation of ``model.on`` that the policy realizes."""

    _entity = "term"
    _later = {"quantities": "a later lot of N15 with the record states of N8",
              "unknowns": "a later lot of N15 with the constraints of N10",
              "law": _G04, "flux": _G04, "expect": _G04, "operator": _G04, "load": _G04,
              "first_order_inputs": _G04}
    _later_reads = _LATER_BOUNDARY_TERM

    def __init__(self, relation: BoundaryRelation) -> None:
        self._relation = relation

    @property
    def kind(self) -> Word:
        return kinds.BoundaryRelation

    @property
    def form(self) -> Word:
        return getattr(forms, self._relation.form)

    @property
    def quantity(self) -> Quantity:
        return self._relation.quantity

    @property
    def data(self) -> Any:
        relation = self._relation
        if relation.form != "Value":
            raise AmbiguousElaboration(
                "term.data of %r, whose form is %s, which carries no data" % (relation,
                                                                            relation.form),
                write="term.data under a relation of form Value, Characteristic or NormalFlux; "
                      "read term.form first", site=user_site(2))
        return relation.data

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
            raise TypeError("@pops.scheme: %s takes face, for a scheme that returns FaceFlux, "
                            "cell and axis, for a scheme that returns Reconstruct, or cell, for "
                            "a scheme that returns CellRate (2.6.4)" % name)
        if "depth" in entities and "face" not in entities:
            raise TypeError("@pops.scheme: %s takes depth, the depth of a ghost, with face, for "
                            "a boundary policy that returns Ghost: def %s(face, term, depth)"
                            % (name, name))
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
        """A scheme over ``cell`` and ``axis``, which returns ``Reconstruct``."""
        return "cell" in self.entities and "axis" in self.entities

    @property
    def is_cell_rate(self) -> bool:
        """A scheme over ``cell`` without ``axis``, which returns ``CellRate`` (note §3.2)."""
        return "cell" in self.entities and "axis" not in self.entities

    @property
    def is_boundary(self) -> bool:
        return "depth" in self.entities

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
    """What a scheme gave under one key: its builder, the reconstruction that it names, and for a
    diffusive key the incident frequency of its face that the elaborator derives (D10 of N3)."""

    __slots__ = ("builder", "reconstruction", "frequency")

    def __init__(self, builder: Any, reconstruction: Any, frequency: Any = None) -> None:
        self.builder = builder
        self.reconstruction = reconstruction
        self.frequency = frequency


class StencilRealization:
    """The realization of a diffusive key under a face scheme: the b6 ``StencilFlux`` whose body
    is the face flux of the scheme, entered in each evaluation of a rate that covers the key as
    ``stencil_flux_divergence`` (N3 lot 1, D7), not as a row of the plan of b6."""

    __slots__ = ("flux", "description")

    def __init__(self, flux: Any, description: str) -> None:
        self.flux = flux
        self.description = description


_LEFT = ("leaf", ("trace", "left"))


def _mentions_trace(atom: Any) -> bool:
    return "'trace'" in repr(atom)


def diffusive_frequency(scheme_: Any, key: Any, value: Any) -> Any:
    """The incident frequency of a diffusive face (D10 of N3): the exact derivative of the face
    value with respect to the trace on its left, divided by ``face.distance``.

    It is derived, never stated (``FaceFlux`` has no frequency, 2.6.4). The derivative is taken
    on the normal form of the value (CC2), exactly; for the face of -k grad(u), k not reading the
    state, it is k / face.distance, written ``(k / face.distance) / face.distance``. Summed over
    the faces of a cell by b6's StencilFlux, that is bitwise the frequency of b6's two-point
    diffusion, sum_a ((k / h_a) + (k / h_a)) / h_a. A face whose derivative is of another form
    (a coefficient that reads the state, a face that is not linear in its traces) belongs to the
    coupled realizations of G05."""
    distance = SchemeLeaf(("distance",), None, "face.distance")
    derivative: dict = {}
    for mono, coef in _polynomial(value).items():
        powers = dict(mono)
        exponent = powers.pop(_LEFT, 0)
        if exponent == 0:
            continue
        if exponent != 1 or any(_mentions_trace(atom) for atom in powers):
            break
        reduced = tuple(sorted(powers.items()))
        derivative[reduced] = derivative.get(reduced, 0) + coef
    else:
        coefficient = key.coefficient
        expected = _polynomial(Arith("/", (coefficient, distance)))
        if {mono: coef for mono, coef in derivative.items() if coef != 0} == expected:
            return Arith("/", (Arith("/", (coefficient, distance)), distance))
    raise NotYetElaborated(
        "the scheme %s under %s, whose face %s is not the two-point flux k (w_R - w_L) / "
        "face.distance of -k grad(w), so its frequency is not derived in this lot"
        % (scheme_, key, text(value)), _DIFFUSIVE_FORMS, site=getattr(scheme_, "site", None))


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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotAPopsOperation(
            "%r is a scheme, which pops.Numerics runs; an option that takes a function of face "
            "takes a plain function, passed uncalled" % self,
            write="def f(face): ... and pass f", site=user_site(2))

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
        # scheme under a diffusive flux (which a later lot elaborates). A scheme over cell without
        # axis returns CellRate, which covers a source in this lot (note §3.2, Q10).
        if self.scheme.is_cell_rate:
            return ("Source",)
        return ("Flux", "DiffusiveFlux")

    def check_key(self, key: Term) -> None:
        """Run the scheme under its key, as ``pops.Numerics`` receives it (SU5)."""
        if self.scheme.is_cell:
            raise RealizationMismatch(
                "%s returns Reconstruct, the trace of a face, which covers no term" % self,
                write="the scheme as the reconstruction= of a face scheme, such as "
                      "rusanov(reconstruction=%s)" % self, site=self.site)
        if self.scheme.is_boundary:
            raise RealizationMismatch(
                "%s returns Ghost, a ghost of a boundary relation, which covers no term" % self,
                write="the policy as pops.Numerics(boundary=%s)" % self, site=self.site)
        self.traced(key)

    def ghosts(self, relation: BoundaryRelation, depths: tuple) -> list:
        """The ghost that this boundary policy gives under ``relation``, on each of its pieces
        and at each depth: a list of (piece, depth, value)."""
        rows = []
        for piece in relation.pieces:
            for k in depths:
                entities = {"face": BoundaryFaceEntity(relation, piece),
                            "term": BoundaryTermEntity(relation), "depth": k}
                arguments = {name: entities[name] for name in self.scheme.entities}
                arguments.update(self.options)
                result = self.scheme.body(**arguments)
                if not isinstance(result, Ghost):
                    raise RealizationMismatch(
                        "the boundary policy %s returns %r under %r, which Ghost covers"
                        % (self.scheme.name, result, relation),
                        write="Ghost(value)", site=self.scheme.site)
                for node in walk(result.value):
                    if isinstance(node, SchemeLeaf) and node.key[0] not in ("inner", "normal"):
                        raise TypeError("%s reads %s, which a boundary policy does not receive"
                                        % (self, node))
                    if isinstance(node, (Quantity, Term)):
                        raise TypeError("%s reads %s itself; a boundary policy reads the inner "
                                        "cells through face.inner(k).value(q)" % (self, node))
                rows.append((piece, k, result.value))
        return rows

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
        expected = Reconstruct if self.scheme.is_cell else \
            CellRate if self.scheme.is_cell_rate else FaceFlux
        if not isinstance(result, expected):
            raise RealizationMismatch(
                "the scheme %s returns %r under %s, a %s, which %s covers"
                % (self.scheme.name, result, key, key.kind, expected.word),
                write="%s(value)" % expected.word, site=self.scheme.site)
        return result

    def reads_speeds(self, key: Term) -> bool:
        """Whether the face of the key reads the face input ``speeds`` (the slowest and the
        fastest wave speeds of R9), which the model then declares (``Model.wave_speeds``)."""
        if self.scheme.is_cell or self.scheme.is_cell_rate or isinstance(key, DiffusiveFlux):
            return False
        builder = self.traced(key).builder
        return reads_speed_pair(builder.value, builder.speed)

    def traced(self, key: Term) -> Traced:
        """The builder that the scheme gives under ``key``, run once per key (SU5), and for a face
        scheme once per axis of the support, ``face.axis`` being Python data."""
        for known, traced in self._traced:
            if known is key:
                return traced
        axes = key.quantity.support.axes
        if self.scheme.is_cell_rate:
            traced = Traced(self._traced_rate(key), None)
            self._traced.append((key, traced))
            return traced
        if self.scheme.is_cell:
            builder = None
            for axis in axes:
                ran = self._run({"cell": CellEntity(key, axis), "axis": axis,
                                 "term": TermEntity(key)}, key)
                _reads(ran.value, ("sample",), self, "cell")
                if builder is None:
                    builder = ran
                elif normal_form(builder.value) != normal_form(ran.value):
                    raise NotYetElaborated(
                        "the scheme %s, whose trace along the axis %s differs from the one along "
                        "the axis %s" % (self, axis, axes[0]), _PER_AXIS_RECONSTRUCTION,
                        site=self.site)
            traced = Traced(builder, None)
        elif isinstance(key, DiffusiveFlux):
            traced = self._traced_diffusive(key, tuple(axes))
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
            if recon is not None and _reads_cells(builder.value, builder.speed):
                raise NotYetElaborated(
                    "the scheme %s, which reads face.left_cell.value or face.right_cell.value "
                    "under the reconstruction %r: the face receives the reconstructed traces, "
                    "and the average of a cell is a read of the stencil" % (self, recon),
                    _STENCIL, site=self.site)
            traced = Traced(builder, None if recon is None else recon.traced(key))
        self._traced.append((key, traced))
        return traced

    def _traced_rate(self, key: Term) -> CellRate:
        """The ``CellRate`` of a scheme over ``cell`` under a source, run once (no axis): its value
        reads ``cell.value`` of the key's quantity, numbers, constants, parameters, the
        coordinates of the support at the centre of the cell and ``cell.load(e)`` (FC-1, FC-2)."""
        ran = self._run({"cell": RateCell(key), "term": TermEntity(key)}, key)
        for node in walk(ran.value):
            if isinstance(node, Call) and node.word == "grad":
                raise NotYetElaborated(
                    "the scheme %s under %s, whose CellRate reads %s" % (self, key, text(node)),
                    "the lot of scheme gradients of N15 (GR), with pointwise(gradient=)",
                    site=self.site)
        _reads(ran.value, ("sample",), self, "cell")
        return ran

    def _traced_diffusive(self, key: DiffusiveFlux, axes: tuple) -> Traced:
        """The face of a diffusive key, run once per axis: it reads the two traces and
        ``face.distance``, names no reconstruction and states no speed; its frequency is derived
        (D10 of N3)."""
        if any(isinstance(node, Quantity) for node in walk(key.coefficient)):
            raise NotYetElaborated(
                "the scheme %s under %s, whose coefficient %s varies with the state"
                % (self, key, text(key.coefficient)),
                "a later lot of N15 with the face means of 2.6.4 (G05)", site=self.site)
        builder = None
        for axis in axes:
            ran = self._run({"face": FaceEntity(key, axis), "term": TermEntity(key)}, key)
            _reads(ran.value, _DIFFUSIVE_READS + ("normal",), self, "face")
            if ran.reconstruction is not None:
                raise NotYetElaborated(
                    "the scheme %s under the diffusive flux %s names a reconstruction"
                    % (self, key),
                    _DIFFUSIVE_FORMS, site=self.site)
            if ran.speed is not None:
                raise TypeError("the scheme %s states FaceFlux(speed=) under the diffusive flux "
                                "%s; the frequency of a diffusive face is derived (D10)"
                                % (self, key))
            if builder is None:
                builder = ran
            elif normal_form(builder.value) != normal_form(ran.value):
                raise NotYetElaborated(
                    "the scheme %s, whose face flux on the axis %s differs from the one on the "
                    "axis %s" % (self, axis, axes[0]), _PER_AXIS, site=self.site)
        assert builder is not None
        _reads(builder.value, _DIFFUSIVE_READS, self, "face")
        return Traced(builder, None, diffusive_frequency(self, key, builder.value))

    def _realize_diffusive(self, view: TermView) -> StencilRealization:
        from pops.numerics.stencil_flux import StencilFlux

        traced = self.traced(view.term)
        value, frequency = traced.builder.value, traced.frequency
        rank = view.term.quantity.support.rank
        lower = view.lower

        def body(sample: Any, spacing: Any, axis: int) -> tuple:
            left = tuple(-1 if d == axis else 0 for d in range(rank))
            inputs = {("trace", "left"): sample(left, 0),
                      ("trace", "right"): sample((0,) * rank, 0),
                      ("distance",): spacing[axis]}
            return (lower(value, inputs),), lower(frequency, inputs)

        flux = StencilFlux(dimension=rank, components=1, radius=1, body=body)
        return StencilRealization(flux, self.describe())

    def is_two_point(self, key: Term) -> bool:
        """Whether the face of this scheme under the diffusive ``key`` of -k grad(w) is the
        two-point flux -k (w_R - w_L) / face.distance of ``two_point()``, by its normal form (CC2).
        The matrix-free apply of a solve reads it so (N3 lot 1, D7): a copy of the body written in
        a case is applied as the word, and any other face is not."""
        if not isinstance(key, DiffusiveFlux) or self.scheme.is_cell or len(key.variables) != 1:
            return False
        (w,) = key.variables
        left = SchemeLeaf(("trace", "left"), w, "face.left(%s)" % w)
        right = SchemeLeaf(("trace", "right"), w, "face.right(%s)" % w)
        distance = SchemeLeaf(("distance",), None, "face.distance")
        form = Arith("*", (Arith("neg", (key.coefficient,)),
                           Arith("/", (Arith("-", (right, left)), distance))))
        return normal_form(self.traced(key).builder.value) == normal_form(form)

    def realize(self, view: TermView) -> Any:
        from pops.numerics import reconstruction, riemann, variables
        from pops.numerics.spatial import FiniteVolume

        if isinstance(view.term, DiffusiveFlux):
            return self._realize_diffusive(view)
        if self.scheme.is_cell_rate:
            # The value of the CellRate is the value of the b6 named source (note §3.4), which
            # the elaborator declares; its selection of the rate is evaluated at the cell.
            from pops.numerics import StateStorage

            return StateStorage()
        traced = self.traced(view.term)
        builder = traced.builder
        lower = view.lower

        body: Any
        stability: Any
        if reads_speed_pair(builder.value, builder.speed):
            # The pair (sL, sR) of the slowest and the fastest wave speeds united over the two
            # traces is the face input speeds of riemann.User (R9), as the b6 HLL reads it.
            def body_of_pair(uL: Any, uR: Any, fL: Any, fR: Any, s: Any, *, n: Any,
                             speeds: Any) -> tuple:
                return (lower(builder.value, _face_inputs(uL, uR, fL, fR, n, s, speeds)),)
            body = body_of_pair
        else:
            def body_of_traces(uL: Any, uR: Any, fL: Any, fR: Any, s: Any, *, n: Any) -> tuple:
                return (lower(builder.value, _face_inputs(uL, uR, fL, fR, n, s)),)
            body = body_of_traces

        if builder.speed is None:
            # The larger spectral radius of R9 at the two traces: the fallback of cell.speed
            # (2.6.4, SD44), and the bound that b6 requires of every user face (N1).
            def radius(uL: Any, uR: Any, fL: Any, fR: Any, s: Any, *, n: Any) -> Any:
                del uL, uR, fL, fR, n
                return s
            stability = radius
        else:
            # A speed may be signed (upwind follows the orientation of the face); b6 reads a
            # nonnegative bound, so a speed that is not nonnegative by its form is taken in
            # absolute value.
            signed = not _nonnegative(builder.speed)

            if reads_speed_pair(builder.speed):
                def bound_of_pair(uL: Any, uR: Any, fL: Any, fR: Any, s: Any, *, n: Any,
                                  speeds: Any) -> Any:
                    bound = lower(builder.speed, _face_inputs(uL, uR, fL, fR, n, s, speeds))
                    return abs(bound) if signed else bound
                stability = bound_of_pair
            else:
                def bound_of_traces(uL: Any, uR: Any, fL: Any, fR: Any, s: Any, *, n: Any) -> Any:
                    bound = lower(builder.speed, _face_inputs(uL, uR, fL, fR, n, s))
                    return abs(bound) if signed else bound
                stability = bound_of_traces

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
        if self.scheme.is_cell_rate:
            return ("StateStorage(), the selection of the named source whose value is the "
                    "CellRate of %s, evaluated at each cell" % self.scheme.name)
        first = self._traced[0][1] if self._traced else None
        if first is not None and first.frequency is not None:
            return ("StencilFlux(radius=1, body=the FaceFlux of %s, frequency=%s, derived), "
                    "entered in each evaluation of a rate as a stencil of the program"
                    % (self.scheme.name, text(first.frequency)))
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


_FACE_READS = ("trace", "normal", "flux", "radius", "cell", "slowest_speed", "fastest_speed")
_DIFFUSIVE_READS = ("trace", "distance")


def _reads_cells(*values: Any) -> bool:
    return any(isinstance(node, SchemeLeaf) and node.key[0] == "cell"
               for value in values if value is not None for node in walk(value))


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


def _face_inputs(uL: Any, uR: Any, fL: Any, fR: Any, n: Any, s: Any,
                 speeds: Any = None) -> dict:
    # A cell read is lowered to the trace only where the face has no reconstruction (traced).
    inputs = {("trace", "left"): uL[0], ("trace", "right"): uR[0], ("flux", "left"): fL[0],
              ("flux", "right"): fR[0], ("normal",): n[0], ("speed",): s,
              ("cell", "left"): uL[0], ("cell", "right"): uR[0]}
    inputs.update({("normal", index): n[index] for index in range(3)})
    if speeds is not None:
        inputs[("speeds", 0)], inputs[("speeds", 1)] = speeds
    return inputs


def _reads(value: Any, allowed: tuple, scheme_: Any, entity: str, site: Any = None) -> None:
    """Check the reads of a value that a scheme (or a function of face) returns."""
    site = scheme_.scheme.site if site is None else site
    for node in walk(value):
        if isinstance(node, LawValue):
            raise SupportMismatch(
                "%s returns %s, the flux of the key, a vector, where a scalar is taken"
                % (scheme_, node),
                write="dot(term.law(...), face.normal), its normal component", site=site)
        if isinstance(node, SchemeLeaf) and node.key == ("normal",) and node.rank > 1:
            raise SupportMismatch(
                "%s uses face.normal, a vector of %d components, where a scalar is taken"
                % (scheme_, node.rank),
                write="dot(e, face.normal) of a vector e, such as dot(term.law(...), "
                      "face.normal)", site=site)
        if isinstance(node, SchemeLeaf) and node.key == ("distance",) and "distance" not in allowed:
            raise NotYetElaborated("%s, a transport face that reads face.distance" % scheme_,
                                   _SPACING, site=site)
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


def face_function(function: Any, view: TermView, where: str, reconstructed: bool = False) -> Any:
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
        _reads(ran, ("trace", "normal", "cell"), where, "face", site=user_site(2))
        if reconstructed and _reads_cells(ran):
            raise NotYetElaborated(
                "%s, which reads face.left_cell.value or face.right_cell.value under a "
                "reconstruction: the face receives the reconstructed traces" % where, _STENCIL)
        if value is None:
            value = ran
        elif normal_form(value) != normal_form(ran):
            raise NotYetElaborated("%s, whose value on the axis %s differs from the one on the "
                                   "first axis" % (where, axis), _PER_AXIS)
    lower = view.lower

    def lowered(face: Any) -> Any:
        inputs = {("trace", "left"): face.left(view.state)[0],
                  ("trace", "right"): face.right(view.state)[0],
                  ("cell", "left"): face.left(view.state)[0],
                  ("cell", "right"): face.right(view.state)[0], ("normal",): face.normal[0]}
        inputs.update({("normal", index): face.normal[index] for index in range(3)})
        return lower(value, inputs)

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
    "BoundaryFaceEntity", "BoundaryTermEntity", "Builder", "CellEntity", "DiffusiveLaw",
    "FaceEntity", "FaceFlux", "Ghost", "Reconstruct", "SchemeLeaf", "StencilRealization",
    "diffusive_frequency", "reads_speed_pair", "speed_pair_index",
    "LawValue", "SurfaceScheme", "TermEntity", "UserScheme", "Word", "Words",
    "check_plain_function", "dot", "formal_order", "is_radius_pair", "normal_dot",
    "fastest_speed", "forms", "kinds", "normal_form", "realize_reconstruction", "scheme",
    "slowest_speed", "spectral_radius", "substitute",
]
