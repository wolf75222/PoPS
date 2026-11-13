"""Relations and solves of a traced method (spec 0.3 §5.5, SV3, T38; N15 lot « solves »).

A method body states an implicit relation in the unknowns of ``unknowns(...)`` and solves it::

    with unknowns(U=B.state.space) as z:
        s = solve(Problem(unknowns=z, equations=(
            z.U == B.state + p.dt * explicit(B.state, B.time) + p.dt * implicit(z.U, B.time + p.dt),
        )), request=p.request)
    U = require_success(s).values.U

``z.U == ...`` is a ``Relation``, which has no truth value. A component called on an unknown is
its image: the rate of the component at the unknown, which the relation keeps pending. This lot
elaborates one unknown and one relation, affine in the unknown:

    a U + sum_k b_k L_k(U) + known = 0,

``a`` and ``b_k`` exact polynomials of ``p.dt`` and ``L_k`` the linear map of a component that
states it. A component states its linear map with ``linear_apply(P, value)``, which writes ``L(v)``
on the matrix-free apply that it receives, and ``linear_properties``, the claims of that map with
their proof (``LinearClaims``). The core never reads what the map is: it composes the operator
``A = a I + sum_k b_k L_k`` and derives its properties from the claims (IR16): ``A`` is symmetric
positive definite when ``a > 0`` is a number, every ``b_k`` is ``-c_k p.dt`` with ``c_k >= 0``, and
every ``L_k`` is claimed symmetric negative semidefinite. Without these claims nothing is derived,
and a solver that requires them refuses the solve with ``PropertyNotDerived`` (K4).

The solve lowers onto b6's matrix-free linear route: ``program.matrix_free_operator`` with that
apply, ``LinearProblem(A, rhs = -known, initial_guess=B.state)`` (SV3: the guess is the value of the
unknown at the start of the attempt) and the solver that the realization of the advance gives its
path, consumed with ``RejectAttempt()`` by ``require_success``: a failed solve fails the attempt
(T38), not the run. The path of a solve is the name of the method when the solve is unnamed and
the only one of the method, and ``method/name`` otherwise (5.5, line 707). The solution is a state
whose increment derives from the relation: ``U = -(known + sum_k b_k L_k(U)) / a``, each
``L_k(U)`` an evaluation of its component at the stage of the solve (T34).
"""
from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Any

from pops._ir.symbolic import SymbolicTruthValueError
from pops.time._program.traced_method import (
    ClosedBodyViolation, NotYetElaborated, TracedField, TracedScalar, _Atom, _Relational,
    _Traced, _active_trace, _coefficient, _expansion_add, _expansion_scale, _fraction, _poly_add,
    _poly_mul, _poly_of, _poly_text, _refuse_foreign,
)
from pops.time.values import _Coeff

_ONE_POLY = ((0, Fraction(1)),)
_LATER = "a later lot of N8 with the solves of N9"


class Request:
    """``Request(rtol=None, atol=None)``: the accuracy that a solve must reach (SV3)."""

    __slots__ = ("rtol", "atol")

    def __init__(self, rtol: Any = None, atol: Any = None, *, observe: Any = None,
                 norm: Any = None) -> None:
        if observe is not None or norm is not None:
            raise NotYetElaborated("Request(observe=..., norm=...): what a request observes and "
                                   "its norm (SV3, D16, D17) are elaborated by %s" % _LATER)
        for name, value in (("rtol", rtol), ("atol", atol)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float, Fraction, Decimal)) \
                    or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError("Request(%s=...) is a positive finite number, got %r"
                                 % (name, value))
        if rtol is None and atol is None:
            raise ValueError("Request states rtol=, atol= or both (SV3)")
        self.rtol = rtol
        self.atol = atol

    def __repr__(self) -> str:
        return "Request(%s)" % ", ".join("%s=%r" % (name, value) for name, value in
                                          (("rtol", self.rtol), ("atol", self.atol))
                                          if value is not None)


class MissingRequest(ValueError):
    """A solve whose request neither ``solve(request=...)`` nor the proposal states (SU9, SV3)."""


class UnknownSolvePath(ValueError):
    """A key of the realization that names no solve of the advance (§5.5, §14.1)."""


class AmbiguousSolvePath(ValueError):
    """Two unnamed solves in one method: each takes ``name=`` (§2.6.2, 5.5)."""


class LinearClaims:
    """The claims of a linear map ``L`` that a component states, with their proof (IR16).

    ``symmetric`` and ``negative_semidefinite`` are the claims that the core reads; ``proof`` is
    the sentence that ``pops.inspect`` reports for them, and ``description`` names the map. A
    component outside the core states them as a mapping with these four keys.
    """

    @classmethod
    def of(cls, value: Any) -> LinearClaims | None:
        """``value`` as claims: claims, or a value of the library that states the four of them as
        attributes, so that ``pops.lib`` imports no private module (CF0b); ``None`` otherwise."""
        if isinstance(value, cls):
            return value
        if value is None or not all(hasattr(value, name) for name in cls.__slots__):
            return None
        if not all(isinstance(getattr(value, name), bool)
                   for name in ("symmetric", "negative_semidefinite")):
            return None
        return cls(**{name: getattr(value, name) for name in cls.__slots__})

    __slots__ = ("symmetric", "negative_semidefinite", "proof", "description")

    def __init__(self, *, symmetric: bool, negative_semidefinite: bool, proof: str,
                 description: str) -> None:
        self.symmetric = bool(symmetric)
        self.negative_semidefinite = bool(negative_semidefinite)
        self.proof = proof
        self.description = description

    def __repr__(self) -> str:
        return "LinearClaims(%s: %s)" % (self.description, self.proof)


# --- unknowns and the affine forms in them -----------------------------------------------------

class TracedUnknown(_Relational):
    """An unknown of ``unknowns(...)``: a field of the space it was given, solved by ``solve``."""

    __slots__ = ("name", "space")

    def __init__(self, trace: Any, name: str, space: Any) -> None:
        self._trace = trace
        self.name = name
        self.space = space

    def _as_affine(self) -> TracedAffine:
        self._trace.require_active("the unknown %s" % self.name)
        return TracedAffine(self._trace, self, (_Coeff({0: 1}), _ONE_POLY), (), None, False)

    def _image_by(self, component: Any, c: Any) -> TracedAffine:
        """``component(U, t)``: the image of ``U`` by the linear map that the component states,
        pending in the relation that reads it (IR16)."""
        component._linear_claims()
        return TracedAffine(self._trace, self, (_Coeff({}), ()),
                            ((component, c, (_Coeff({0: 1}), _ONE_POLY)),), None, False)

    def __repr__(self) -> str:
        return "TracedUnknown(%s)" % self.name


class TracedAffine(_Relational):
    """``a U + sum_k b_k L_k(U) + known``: an affine form in one unknown ``U``.

    ``_a`` and each ``b_k`` are ``(builder coefficient, exact polynomial of p.dt)``; ``_images``
    holds ``(component, c, b_k)`` with ``c`` the time ``B.time + c p.dt`` of the image; ``_known``
    is the traced field that reads no unknown, or ``None``.
    """

    __slots__ = ("_unknown", "_a", "_images", "_known", "_inexact")

    def __init__(self, trace: Any, unknown: TracedUnknown, a: tuple, images: tuple,
                 known: TracedField | None, inexact: bool) -> None:
        self._trace = trace
        self._unknown = unknown
        self._a = a
        self._images = images
        self._known = known
        self._inexact = inexact

    def _as_affine(self) -> TracedAffine:
        self._trace.require_active("a relation")
        return self

    def __repr__(self) -> str:
        parts = ["(%s)*%s" % (_poly_text(self._a[1]), self._unknown.name)] if self._a[1] else []
        parts.extend("(%s)*%s(%s)" % (_poly_text(poly), component._name, self._unknown.name)
                     for component, _, (_, poly) in self._images)
        if self._known is not None:
            parts.append(repr(self._known))
        return "TracedAffine(%s)" % (" + ".join(parts) or "0")


def _add_terms(left: tuple, right: tuple) -> tuple:
    return (left[0] + right[0], _poly_add(left[1], right[1]))


def _scale_term(term: tuple, factor: Any, poly: tuple) -> tuple:
    return (term[0] * factor, _poly_mul(term[1], poly))


def _lift(trace: Any, unknown: TracedUnknown, other: Any, where: str) -> TracedAffine:
    """``other`` as an affine form in ``unknown``: an unknown, an affine form or a known field."""
    if isinstance(other, _Relational):
        other._same_trace_relational(trace, where)
        affine = other._as_affine()
        if affine._unknown is not unknown:
            raise NotYetElaborated(
                "%s in two unknowns, %s and %s: a relation in several unknowns is elaborated by "
                "%s" % (where, unknown.name, affine._unknown.name, _LATER))
        return affine
    if isinstance(other, TracedField):
        if other._trace is not trace:
            raise ClosedBodyViolation("%s mixes values of two method traces (T33)" % where)
        return TracedAffine(trace, unknown, (_Coeff({}), ()), (), other, other._inexact)
    if isinstance(other, _Traced):
        raise TypeError("%s combines fields and unknowns; a duration or a time is not a field "
                        "(T9)" % where)
    _refuse_foreign(other, where)
    raise TypeError("%s combines fields and unknowns; a number is not a field (T9)" % where)


def _sum(left: TracedAffine, right: TracedAffine) -> TracedAffine:
    images = list(left._images)
    for component, c, term in right._images:
        for index, (other, c_other, held) in enumerate(images):
            if other is component and _fraction(c_other) == _fraction(c):
                images[index] = (other, c_other, _add_terms(held, term))
                break
        else:
            images.append((component, c, term))
    if left._known is None or right._known is None:
        known = left._known if right._known is None else right._known
    else:
        known = left._known + right._known
    return TracedAffine(left._trace, left._unknown, _add_terms(left._a, right._a), tuple(images),
                        known, left._inexact or right._inexact)


def _scale(affine: TracedAffine, factor: Any, poly: tuple, inexact: bool) -> TracedAffine:
    return TracedAffine(
        affine._trace, affine._unknown, _scale_term(affine._a, factor, poly),
        tuple((component, c, _scale_term(term, factor, poly))
              for component, c, term in affine._images),
        None if affine._known is None else affine._known * factor,
        affine._inexact or inexact)


def _factor(trace: Any, other: Any, where: str) -> tuple:
    """(builder coefficient, exact polynomial, binary64?) of a factor of an affine form."""
    if isinstance(other, TracedScalar):
        if other._trace is not trace:
            raise ClosedBodyViolation("%s mixes values of two method traces (T33)" % where)
        return other._coeff, _poly_of(other._coeff), other._inexact
    if isinstance(other, _Traced):
        raise TypeError("%s: a relation is affine in its unknowns (T9, 5.5); it is scaled by a "
                        "number or a duration" % where)
    _refuse_foreign(other, where)
    exact, inexact = _coefficient(other)
    return other, ((0, exact),), inexact


def _binary(name: str) -> Any:
    def method(self: Any, other: Any) -> Any:
        affine = self._as_affine()
        if name in ("__mul__", "__rmul__"):
            factor, poly, inexact = _factor(affine._trace, other, "a product with an unknown")
            return _scale(affine, factor, poly, inexact)
        if name == "__truediv__":
            if isinstance(other, _Traced):
                raise TypeError("an affine form in an unknown is divided by a number only (T9)")
            _refuse_foreign(other, "a quotient of an unknown")
            exact, inexact = _coefficient(other)
            return _scale(affine, Fraction(1) / exact if not inexact else 1.0 / other,
                          ((0, 1 / exact),), inexact)
        right = _lift(affine._trace, affine._unknown, other, "a sum with an unknown")
        if name == "__add__":
            return _sum(affine, right)
        if name == "__radd__":
            return _sum(right, affine)
        if name == "__sub__":
            return _sum(affine, _negated(right))
        return _sum(_negated(affine), right)
    method.__name__ = name
    return method


for _name in ("__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
              "__truediv__"):
    setattr(_Relational, _name, _binary(_name))


def _negated(affine: TracedAffine) -> TracedAffine:
    return _scale(affine, -1, ((0, Fraction(-1)),), False)


def _neg(self: Any) -> TracedAffine:
    return _negated(self._as_affine())


def _eq(self: Any, other: Any) -> Relation:
    affine = self._as_affine()
    right = _lift(affine._trace, affine._unknown, other, "a relation")
    return Relation(_sum(affine, _negated(right)))


_Relational.__neg__ = _neg  # type: ignore[method-assign]
_Relational.__eq__ = _eq  # type: ignore[method-assign,assignment]
_Relational._claims_of = staticmethod(LinearClaims.of)  # type: ignore[method-assign]


class Relation:
    """``lhs == rhs`` in the unknowns of a method: the residual ``lhs - rhs = 0`` (5.5).

    A relation is an equation of the attempt, never a truth value: ``bool(relation)`` refuses.
    """

    __slots__ = ("residual",)
    __hash__ = object.__hash__

    def __init__(self, residual: TracedAffine) -> None:
        self.residual = residual

    def __bool__(self) -> bool:
        raise SymbolicTruthValueError(
            self, suggestion="a relation of traced values is an equation that solve(Problem(...)) "
                             "solves (5.5); Python if/and/or/in cannot evaluate it")

    def __repr__(self) -> str:
        return "Relation(%r == 0)" % (self.residual,)


class _Unknowns:
    """``unknowns(U=space, ...)``: the unknowns of a problem, read as attributes (5.5)."""

    __slots__ = ("_items",)

    def __init__(self, items: dict) -> None:
        self._items = items

    def __getattr__(self, name: str) -> TracedUnknown:
        if name.startswith("_"):
            raise AttributeError(name)
        items = object.__getattribute__(self, "_items")
        if name not in items:
            raise AttributeError("the unknowns are %s, not %r" % (", ".join(items), name))
        return items[name]

    def __iter__(self) -> Any:
        return iter(self._items.values())

    def __enter__(self) -> _Unknowns:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def unknowns(**spaces: Any) -> _Unknowns:
    """``unknowns(U=space)``: the unknowns of a relation of the method body (5.5).

    This lot elaborates one unknown, of the space of the evolved state, ``B.state.space``.
    """
    trace = _active_trace()
    if trace is None:
        raise ClosedBodyViolation("unknowns(...) is written inside a method body (5.5, T33)")
    if trace.evaluation is not None:
        raise NotYetElaborated("unknowns(...) inside the component %r: a solve inside a "
                               "component is elaborated by %s" % (trace.evaluation.component,
                                                                  _LATER))
    if len(spaces) != 1:
        raise NotYetElaborated("unknowns(%s): this lot elaborates one unknown; several are "
                               "elaborated by %s" % (", ".join(spaces) or "", _LATER))
    ((name, space),) = spaces.items()
    if space is not trace.temporal.n.space:
        raise NotYetElaborated(
            "unknowns(%s=...) of another space than the evolved state, B.state.space: this lot "
            "elaborates an unknown of the state, other spaces come with %s" % (name, _LATER))
    return _Unknowns({name: TracedUnknown(trace, name, space)})


class Problem:
    """``Problem(unknowns=z, equations=(relation,))``: a problem of the method body (5.5)."""

    __slots__ = ("unknowns", "equations")

    def __init__(self, unknowns: Any, equations: Any) -> None:
        if not isinstance(unknowns, _Unknowns):
            raise TypeError("Problem(unknowns=...) takes the unknowns of unknowns(...)")
        if isinstance(equations, Relation) or not isinstance(equations, (tuple, list)):
            raise TypeError("Problem(equations=...) is a tuple of relations, (rel,)")
        if not all(isinstance(rel, Relation) for rel in equations):
            raise TypeError("Problem(equations=...) holds relations, U == ..., of its unknowns")
        if len(equations) != 1:
            raise NotYetElaborated("Problem with %d relations: this lot elaborates one relation "
                                   "in one unknown; systems come with %s" % (len(equations),
                                                                             _LATER))
        (unknown,) = tuple(unknowns)
        if equations[0].residual._unknown is not unknown:
            raise TypeError("Problem(...): the relation is not in the unknown %s" % unknown.name)
        self.unknowns = unknowns
        self.equations = tuple(equations)


# --- the solve ---------------------------------------------------------------------------------

class SolveSite:
    """A solve of the method body: its path, its site and what it reads (T38, 5.5)."""

    __slots__ = ("path", "name", "site", "fixed", "solver", "request", "claims", "spd", "outcome",
                 "consumed", "problem", "stage")

    def __init__(self, path: str, name: str | None, site: str) -> None:
        self.path = path
        self.name = name
        self.site = site
        self.fixed = False
        self.solver: Any = None
        self.request: Any = None
        self.claims: tuple = ()
        self.spd = False
        self.outcome: Any = None
        self.consumed = False
        self.problem: Any = None
        self.stage: Any = None

    def __repr__(self) -> str:
        return "<SolveSite %s>" % self.path


class TracedSolve:
    """The outcome of ``solve(...)``, which ``require_success`` reads (5.5, T38)."""

    __slots__ = ("_trace", "_site", "_unknown", "_expansion", "_c", "_inexact")

    def __init__(self, trace: Any, site: SolveSite, unknown: TracedUnknown, expansion: tuple,
                 c: Any, inexact: bool) -> None:
        self._trace = trace
        self._site = site
        self._unknown = unknown
        self._expansion = expansion
        self._c = c
        self._inexact = inexact

    def __bool__(self) -> bool:
        raise SymbolicTruthValueError(
            self, suggestion="require_success(s) reads the solution of a solve; the outcome of "
                             "a solve has no truth value while its method is traced")

    def __repr__(self) -> str:
        return "<TracedSolve %s>" % self._site.path


class _Values:
    __slots__ = ("_items",)

    def __init__(self, items: dict) -> None:
        self._items = items

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        items = object.__getattribute__(self, "_items")
        if name not in items:
            raise AttributeError("the solution holds %s, not %r" % (", ".join(items), name))
        return items[name]

    def __getitem__(self, unknown: Any) -> Any:
        for name, value in self._items.items():
            if unknown is name or getattr(unknown, "name", None) == name:
                return value
        raise KeyError(unknown)


class Solution:
    """``require_success(s)``: the values of the unknowns, ``solution.values.U`` (5.5)."""

    __slots__ = ("values",)

    def __init__(self, values: dict) -> None:
        self.values = _Values(values)


def _path(trace: Any, name: str | None) -> str:
    return trace.name if name is None else "%s/%s" % (trace.name, name)


def _realization(trace: Any, site: SolveSite) -> Any:
    realization = trace.realization
    for key in (site.path, "%s/*" % trace.name):
        if key in realization:
            trace.realized.add(key)
            return realization[key]
    raise NotYetElaborated(
        "the solve %s has no solver: the realization keys it by %r or %r; the selection auto() "
        "of SV11 comes with N9" % (site.path, site.path, "%s/*" % trace.name))


def _spd(a: tuple, images: tuple) -> bool:
    """IR16: ``a I + sum_k b_k L_k`` is symmetric positive definite (see the module)."""
    if dict(a[1]).keys() != {0} or dict(a[1])[0] <= 0:
        return False
    for component, _, (_, poly) in images:
        claims = component._linear_claims()
        if not (claims.symmetric and claims.negative_semidefinite):
            return False
        if any(power < 1 or coeff > 0 for power, coeff in poly):
            return False
    return True


def solve(problem: Any, *, request: Any = None, name: str | None = None) -> TracedSolve:
    """``solve(Problem(...), request=p.request, name=None)``: solve a relation of the attempt.

    The request is ``request=`` or else ``p.request``, the request of the proposal (SV3); the path
    is the name of the method, or ``method/name`` for a named solve. The solver is the
    realization of that path (``elaborate(..., realization=...)``).
    """
    trace = _active_trace()
    if trace is None:
        raise ClosedBodyViolation("solve(...) is written inside a method body (5.5, T33)")
    if trace.evaluation is not None:
        raise NotYetElaborated("solve(...) inside the component %r is elaborated by %s"
                               % (trace.evaluation.component, _LATER))
    if not isinstance(problem, Problem):
        raise TypeError("solve(problem) takes Problem(unknowns=..., equations=(...))")
    if name is not None and (not isinstance(name, str) or not name.isidentifier()):
        raise ValueError("solve(name=...) is an identifier, got %r" % (name,))
    (unknown,) = tuple(problem.unknowns)
    if unknown._trace is not trace:
        raise ClosedBodyViolation("solve(...) of an unknown of another method trace (T33)")
    path = _path(trace, name)
    for other in trace.solves:
        if other.path == path:
            raise AmbiguousSolvePath(
                "the method %r holds two solves at the path %r: each solve of a method that "
                "holds several takes name= (5.5), at %s" % (trace.name, path, ", ".join(
                    site.site for site in trace.solves if site.path == path)))
    if request is None:
        request = trace.proposal.request
    if request is None:
        raise MissingRequest(
            "the solve %s of %r has no request: solve(request=Request(...)) or the request of "
            "the proposal, Proposal(request=...), states it (SU9, SV3)" % (path, trace.name))
    if not isinstance(request, Request):
        raise TypeError("solve(request=...) takes a Request, got %r" % (request,))
    residual = problem.equations[0].residual
    if not residual._images:
        raise NotYetElaborated(
            "the relation of %s reads no rate at its unknown: an algebraic relation without an "
            "operator is elaborated by %s" % (path, _LATER))
    times = {_fraction(c) for _, c, _ in residual._images}
    if len(times) != 1:
        raise NotYetElaborated(
            "the relation of %s evaluates its rates at the unknown at several times: this lot "
            "elaborates one time of the solve, others come with %s" % (path, _LATER))
    c = residual._images[0][1]
    a_coeff, a_poly = residual._a
    if not a_poly or dict(a_poly).keys() != {0}:
        raise NotYetElaborated(
            "the relation of %s weighs its unknown by %s: this lot elaborates a weight that is a "
            "nonzero number, so that the increment of the solution derives (T34); others come "
            "with %s" % (path, _poly_text(a_poly) if a_poly else "0", _LATER))
    if dict(a_poly)[0] != 1:
        # The accepted commit attributes the transport of the solution to the right-hand side
        # only when the unknown has the weight 1 in the operator (codegen, the rule linear_solve
        # of the affine quadrature); the division of the relation by its weight is a later lot.
        raise NotYetElaborated(
            "the relation of %s weighs its unknown by %s: this lot elaborates the weight 1, "
            "U == known + sum_k b_k L_k(U); write the relation divided by %s. A relation "
            "divided by its weight comes with %s" % (path, _poly_text(a_poly), _poly_text(a_poly),
                                                     _LATER))
    site = SolveSite(path, name, path + "/solve")
    trace.solves.append(site)
    site.request = request
    site.claims = tuple((component._name, component._linear_claims())
                        for component, _, _ in residual._images)
    site.spd = _spd(residual._a, residual._images)
    known = residual._known
    site.fixed = all(not any(power for power, _ in poly) for _, _, (_, poly) in residual._images) \
        and (known is None or all(atom is trace.state_atom and all(power == 0 for power, _ in p)
                                  for atom, p in known._expansion))
    solver = _realization(trace, site)
    site.solver = solver
    program = trace.program
    ordinal = len(trace.stages)
    point = program.stage("%s_solve%d" % (trace.name, ordinal), c=c)
    trace.stages.append((ordinal, c, "%s_solution%d" % (trace.name, ordinal)))
    site.stage = ordinal
    if known is None:
        raise NotYetElaborated("the relation of %s is homogeneous: its solution is zero, which "
                               "this lot does not elaborate" % path)
    rhs = program.value("%s_rhs%d" % (trace.name, ordinal),
                        (known * (-1))._builder(), at=point)
    operator = program.matrix_free_operator(
        "%s_operator%d" % (trace.name, ordinal), domain="state", range_="state",
        ncomp=len(getattr(trace.temporal.n.space, "components", ())) or 1)
    identity = a_coeff
    terms = tuple((component, coeff) for component, _, (coeff, _) in residual._images)

    def apply(P: Any, _out: Any, value: Any) -> Any:
        result = identity * value
        for component, coeff in terms:
            result = result + coeff * component._linear_apply(P, value)
        return result

    program.set_apply(operator, apply)
    from pops.linalg import LinearOperatorProperties, LinearProblem

    properties = (LinearOperatorProperties.symmetric_positive_definite() if site.spd
                  else LinearOperatorProperties.general())
    required = tuple(getattr(solver, "requires", ()))
    if "symmetric_positive_definite" in required and not site.spd:
        from pops._ir.refusals import PropertyNotDerived

        stated = "; ".join("%s: %s" % (name, claim.proof) for name, claim in site.claims)
        raise PropertyNotDerived(
            "%r needs a symmetric positive definite operator, which the relation of %s does not "
            "derive (IR16): the operator is a I + sum_k b_k L_k, SPD when a > 0 is a number, every "
            "b_k a nonpositive multiple of p.dt and every L_k claimed symmetric negative "
            "semidefinite; the claims stated are %s" % (solver, path, stated or "none"),
            region=path, property="symmetric_positive_definite")
    descriptor = solver.linear_solver(request=request)
    site.problem = LinearProblem(operator, rhs, initial_guess=trace.temporal.n, at=point,
                                 properties=properties, nullspace=None)
    site.outcome = program.solve(site.problem, solver=descriptor, name=path.replace("/", "_"))
    # T34: U = -(known + sum_k b_k L_k(U)) / a, each L_k(U) an evaluation at the stage.
    inverse = ((0, Fraction(-1) / dict(a_poly)[0]),)
    expansion = _expansion_scale(known._expansion, inverse)
    for component, _, (_, poly) in residual._images:
        evaluation = len(trace.evaluations)
        trace.evaluations.append((evaluation, component._name, ordinal))
        atom = _Atom("%s#%d" % (component._name, evaluation), evaluation)
        expansion = _expansion_add(expansion, ((atom, _poly_mul(poly, inverse)),))
    return TracedSolve(trace, site, unknown, expansion, c, residual._inexact)


def require_success(outcome: Any) -> Solution:
    """``require_success(s)``: the solution of a solve, whose failure fails the attempt (T38)."""
    if not isinstance(outcome, TracedSolve):
        raise TypeError("require_success(s) takes the outcome of solve(...)")
    trace = outcome._trace
    trace.require_active("the solve %s" % outcome._site.path)
    site = outcome._site
    if site.consumed:
        raise TypeError("require_success(s) reads the outcome of the solve %s once" % site.path)
    from pops.time.solve_outcome import RejectAttempt

    value = site.outcome._consume(RejectAttempt(), {"trial_site": {
        "site": site.site, "retry_can_change": not site.fixed}})
    site.consumed = True
    field = TracedField(trace, value, outcome._expansion, value=value, c=outcome._c,
                        inexact=outcome._inexact)
    return Solution({outcome._unknown.name: field})


__all__ = [
    "AmbiguousSolvePath", "LinearClaims", "MissingRequest", "Problem", "Relation", "Request",
    "Solution", "SolveSite", "TracedAffine", "TracedSolve", "TracedUnknown", "UnknownSolvePath",
    "require_success", "solve", "unknowns",
]
