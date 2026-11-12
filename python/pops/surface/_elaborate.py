"""The elaboration of a case of the surface into a case of b6 (2.6.6; SU5, SU8, SU9, SU12).

Building ``pops.Case(...)`` traces it (SU5). Its support becomes a b6 ``CartesianDomain`` named
``box`` and its frame; its model a b6 ``pops.Model`` of that frame, each constant a ``ConstParam``,
each parameter a ``RuntimeParam``, each evolved quantity a state of one component, each flux a
flux whose waves are the speed dF/du that R9 derives, each diffusive flux the F^d of R7, the
opposite of the physical flux, and each balance a rate. Its numerics become a
``DiscretizationPlan``: each term of each balance is selected and receives the realization that
its scheme word gives (``SchemeValue.realize``). Its program becomes the traced method of N8,
elaborated for the evolved state with a fixed step, the attempts of ``pops.Advance`` and the
default acceptance ``finite()``; each rate option of the method is the sum of the selections of
the terms that it names, in the order of their balance. Its initial value is a datum supplied at
bind, and its layout a b6 ``Uniform`` layout whose periodic axes are those of the support.

Every name of the elaborated form comes from the case (SU8): the name of the model, of each
declaration and of each named term; an unnamed term takes the name that ``model_names`` derives
from its kind and its quantity. ``pops.inspect(case)`` runs b6's ``validate`` and ``resolve`` on
the elaborated case and reports, row by row, what each construct became, its level, and the
origin of each choice, stated, derived or default (SU12).
"""
from __future__ import annotations

import math as _math
from decimal import Decimal
from fractions import Fraction
from collections.abc import Iterable, Sequence
from typing import Any

from pops import math as _pm
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D, Cartesian2D, Cartesian3D
from pops.initial import InitialCondition
from pops.layouts import Uniform as _Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.params import ConstParam, RuntimeParam
from pops.params import Positive as _Positive
from pops.physics import Model as _Model
from pops.problem import Case as _Case
from pops.projection import ConservativeCellAverage
from pops.time._program import elaboration as _n8
from pops.time._program import relations as _relations

from ._expr import Arith, Call, Predicate, is_number, text
from ._geometry import FRAME_AXES, Box, Coordinate, MathConstant
from ._method import MethodValue, _Trace, specialize, translate
from ._model import (
    Balance, Datum, DiffusiveFlux, Flux, Model, ModelConstant, Parameter, Positive, Quantity,
    Term, model_names,
)
from ._model import _sign as _derived_sign
from ._numerics import Numerics, SchemeValue, TermView
from ._refusal import (
    AmbiguousElaboration, CoverageNotExact, MissingBoundaryRealization, NotYetElaborated,
    SurfaceRefusal, UnknownSolvePath,
)
from ._scheme import StencilRealization

_LATER = "a later lot of N15"
# The words of pops.math that b6's analytic expressions have (pops.analytic), for initial values.
_ANALYTIC_WORDS = ("exp", "log", "sqrt", "sin", "cos", "erf", "erfc", "abs")
_FRAMES = {1: Cartesian1D, 2: Cartesian2D, 3: Cartesian3D}


# --- the report ---------------------------------------------------------------------------------

class Choice:
    """A choice of the elaboration and its origin: ``stated``, ``derived`` or ``default`` (SU9)."""

    __slots__ = ("name", "value", "origin")

    def __init__(self, name: str, value: Any, origin: str) -> None:
        self.name = name
        self.value = value
        self.origin = origin

    def __str__(self) -> str:
        return "%s = %s (%s)" % (self.name, self.value, self.origin)


class Row:
    """One construct: its mathematics, its level, its elaborated form and its choices."""

    __slots__ = ("construct", "level", "elaborated", "choices")

    def __init__(self, construct: str, level: str, elaborated: str,
                 choices: Iterable[Any] = ()) -> None:
        self.construct = construct
        self.level = level
        self.elaborated = elaborated
        self.choices = tuple(choices)

    def lines(self) -> list:
        out = ["[%s] %s" % (self.level, self.construct), "    -> %s" % self.elaborated]
        out.extend("    %s" % choice for choice in self.choices)
        return out

    def __str__(self) -> str:
        return "\n".join(self.lines())


class Report:
    """What ``pops.inspect`` returns: the rows of a value, and for a case its resolved plan."""

    __slots__ = ("subject", "level", "rows", "resolved")

    def __init__(self, subject: str, level: str, rows: Iterable[Any], resolved: Any = None) -> None:
        self.subject = subject
        self.level = level
        self.rows = tuple(rows)
        self.resolved = resolved

    def choices(self) -> tuple:
        return tuple(choice for row in self.rows for choice in row.choices)

    def __str__(self) -> str:
        head = "pops.inspect(%s), level %s" % (self.subject, self.level)
        if self.resolved is not None:
            head += ": validate and resolve passed, nothing compiled"
        return "\n".join([head] + [line for row in self.rows for line in row.lines()])

    __repr__ = __str__


class Elaborated:
    """What a case elaborates into: the b6 case, its layout and block, the rows of its report."""

    __slots__ = ("case", "layout", "block", "model", "plan", "states", "names", "advance", "rows",
                 "params", "outputs", "representations")

    def __init__(self, **fields: Any) -> None:
        for name in self.__slots__:
            setattr(self, name, fields[name])


# --- the case -----------------------------------------------------------------------------------

def elaborate_case(case: Any) -> Elaborated:
    """The b6 case of a surface case; a refusal names the line of the case as well (SU11)."""
    try:
        return _Elaboration(case).run()
    except SurfaceRefusal as refusal:
        raise refusal.traced_at(case.site) from None


def elaborated_representation(quantity: Any, support: Box) -> Any:
    """The elaborated identity of a quantity's representation (G03 lot E, K4).

    b6's state carries neither the support nor the representation of the quantity it elaborates
    (``b6.state(name, components=(name,))``), so two quantities of equal shape on two supports, or
    under two representations, would be interchangeable once elaborated. The surface keeps, beside
    each state, ``Representation(space, definition)`` of contract @2 (G03 lot R): the space is the
    support's coordinates and the scalar value, never the quantity's name (P10), and the definition
    is the one of ``pops.CellAverage()``, the representation that the surface elaborates today.
    Nothing of it enters b6, so no identity or manifest of a case changes (N16 5.5). Published
    arrays carry it, and ``bind`` reads it back (``surface/_run.py``)."""
    from pops.model.spaces import Space
    from pops._ir.quantity import PhysicalSupport
    from pops.representations.definition import ElaboratedRepresentation, cell_average
    coordinates = tuple(
        (name, ("periodic[%r, %r)" if periodic else "[%r, %r]") % (float(lower), float(upper)))
        for name, (lower, upper, periodic) in zip(support._names, support.bounds(), strict=True))
    space = Space("scalar", components=("value",), support=PhysicalSupport(coordinates))
    return ElaboratedRepresentation(space, cell_average(support.rank))


# b6 selects a term of a balance whose weight is 1 or -1 only (the partitions of a physical
# balance).
_WEIGHTED = "a later lot of N15, once a selection of b6 realizes a term of another weight"


def _dissipation(coefficient: Any) -> str:
    """R2, K7: whether the declarations establish that F = -k grad(w) dissipates, k > 0."""
    if _derived_sign(coefficient) == 1:
        return "established: %s > 0" % text(coefficient)
    return "not established: the sign of %s is not derived" % text(coefficient)


def _per_axis(values: tuple | None) -> Any:
    """A value per axis as the surface writes it: the scalar on a support of rank one, else
    ``vector(...)`` of the components (lot 2b)."""
    if values is None:
        return None
    return values[0] if len(values) == 1 else Call("vector", tuple(values))


def _per_axis_text(values: tuple, spell: Any) -> str:
    return spell(values[0]) if len(values) == 1 else \
        "vector(%s)" % ", ".join(spell(value) for value in values)


_G04 = "a later lot of N15 with the boundary relations of G04"
_GENERATED_GHOSTS = "a later lot of N15 with the generated ghost fills of G04"
# The depths at which a boundary policy is checked: those that b6 derives at most (WENO5 reads
# three cells), so that a body that departs from a law of b6 at a depth is refused.
_GHOST_DEPTHS = (1, 2, 3)


_ANALYTIC_LOT = "a later lot of N15 with the generated initial values of G07"
_STEP_REQUEST = "a later lot of N8 (the solve requests of a step rule, 5.5)"
_STEP_PARTS = "a later lot of N8 with the step rules of 5.11 per flux (PF8)"
_FACE_SPEEDS = "a later lot of N8 (the recorded face speeds of ROM-108 in the compiled step rule)"


def _tensor(mapped: list) -> list:
    """The tensor nodes of one list of (point, weight) per axis, the first axis slowest."""
    rows: list = [()]
    for nodes in mapped:
        rows = [row + (node,) for row in rows for node in nodes]
    return rows


def _number(value: Any) -> Any:
    """A number of the surface as b6 takes it: an exact fraction or a decimal becomes a float."""
    if isinstance(value, (Fraction, Decimal)):
        return float(value)
    return value


class _RateSum:
    """A rate option of a method: the sum of the selections of its terms, in balance order.

    At an unknown of a solve, the rate is read as the linear map that the schemes of its terms
    state (``linear_apply`` and ``linear_claims``, the lot « solves »): each term asks its scheme
    for the map when a relation reads it, never before, so that an explicit method never needs one.
    """

    __slots__ = ("views", "linear", "_providers")

    def __init__(self, views: Sequence[Any], linear: Sequence[Any] = ()) -> None:
        self.views = tuple(views)
        self.linear = tuple(linear)
        self._providers: tuple | None = None

    def __call__(self, state: Any) -> Any:
        result = self.views[0](state)
        for view in self.views[1:]:
            result = result + view(state)
        return result

    def providers(self) -> tuple:
        if self._providers is None:
            self._providers = tuple(linear() for linear in self.linear)
        return self._providers

    @property
    def linear_claims(self) -> Any:
        from pops.time._program.relations import LinearClaims

        claims = []
        for provider in self.providers():
            claim = LinearClaims.of(provider.claims())
            if claim is None:
                raise NotYetElaborated(
                    "the linear map %r of a rate read at the unknown of a solve states no claims "
                    "(symmetric, negative_semidefinite, proof, description; IR16)" % (provider,),
                    "G10, the operator generated from the term, with the Newton solves of N9")
            claims.append(claim)
        return LinearClaims(
            symmetric=all(claim.symmetric for claim in claims),
            negative_semidefinite=all(claim.negative_semidefinite for claim in claims),
            proof="; ".join(claim.proof for claim in claims),
            description=" + ".join(claim.description for claim in claims))

    def linear_apply(self, P: Any, value: Any) -> Any:
        providers = self.providers()
        result = providers[0].apply(P, value)
        for provider in providers[1:]:
            result = result + provider.apply(P, value)
        return result


class _StencilTerm:
    """A diffusive term realized by a ``StencilFlux`` (N3 lot 1, D7): in each evaluation of a rate
    that covers it, ``-div_h`` of its face flux, the evaluation k named ``<name>_<k>``.

    ``stencil_flux_divergence`` gives -div_h(F) of the physical flux F, which is the contribution
    of ``-div(F)`` to the balance; a term written ``+div(F)`` is its opposite."""

    __slots__ = ("flux", "name", "weight")

    def __init__(self, flux: Any, name: str, weight: int) -> None:
        self.flux = flux
        self.name = name
        self.weight = weight

    def __call__(self, state: Any) -> Any:
        from pops.time._program.stencil_flux import OPERATION

        program = state.prog
        evaluation = sum(1 for value in program._values
                         if value.op == OPERATION and value.name.startswith(self.name + "_"))
        rate = getattr(program, OPERATION)("%s_%d" % (self.name, evaluation), state,
                                           flux=self.flux)
        return rate if self.weight == -1 else -rate


class _Elaboration:
    def __init__(self, case: Any) -> None:
        self.case = case
        self.model: Model = case.model
        self.numerics: Numerics = case.numerics
        self.program = case.program
        self.rows: list = []
        self.values: dict = {}
        self.names: dict = {}
        self.linear: dict = {}
        self.layout_facts: dict = {}
        self.params: dict = {}
        self.lowered_outputs: tuple = ()
        self.rule: Any = None

    # The support and its frame.

    def support(self) -> Box:
        model = self.model
        spaces = [q.space for q in model.quantities] + [d.space for d in model.data_fields]
        if not spaces:
            raise AmbiguousElaboration("the model %r has no quantity" % model.name,
                                       write='model.quantity("u", pops.Scalar(Ω))',
                                       site=model.site)
        support = spaces[0].support
        for space in spaces[1:]:
            if space.support is not support:
                raise NotYetElaborated("a model whose quantities live on several supports",
                                       "a later lot of N15 with the product supports of G09",
                                       site=space.site)
        names = support._names
        if names != FRAME_AXES[:len(names)]:
            raise NotYetElaborated(
                "a support whose axes are named %s; this lot elaborates the axes x, y and z of the "
                "Cartesian frames, in that order" % ", ".join(names),
                "a later lot of N15 with the supports of G03", site=support.site)
        return support

    def frame(self, support: Box) -> Any:
        lower = tuple(float(lo) for lo, _, _ in support.bounds())
        upper = tuple(float(hi) for _, hi, _ in support.bounds())
        frame = CartesianDomain("box", lower=lower, upper=upper).frame(_FRAMES[support.rank]())
        periodic = [name for name, (_, _, flag)
                    in zip(support._names, support.bounds(), strict=True) if flag]
        self.rows.append(Row(
            repr(support), "model",
            'CartesianDomain("box", lower=%r, upper=%r).frame(%s())'
            % (lower, upper, _FRAMES[support.rank].__name__),
            [Choice("periodic axes", ", ".join(periodic) or "none", "stated")]))
        return frame

    # The model.

    def lower(self, value: Any, inputs: dict | None = None) -> Any:
        """The b6 expression of a value of the model; ``inputs`` gives the b6 expressions of the
        reads of the entities of a scheme, inside the body of a scheme (lot 2)."""
        if inputs is not None:
            scheme = self._scheme_value(value, inputs)
            if scheme is not None:
                return scheme
        if is_number(value):
            return _number(value)
        if isinstance(value, MathConstant):
            if not _math.isfinite(value.value):
                raise NotYetElaborated("%s in a value of the model" % value.name, _LATER)
            return value.value
        if isinstance(value, (ModelConstant, Parameter, Quantity)):
            return self.values[value]
        if isinstance(value, Datum):
            raise NotYetElaborated("the datum %s read in a value of the model (Q3)" % value,
                                   "a later lot of N15 with the data fields of Q3",
                                   site=value.site)
        if isinstance(value, Coordinate):
            raise NotYetElaborated("the coordinate %s read in a value of the model" % value,
                                   "a later lot of N15 with the coordinate fields of G03")
        if isinstance(value, Arith):
            args = [self.lower(arg, inputs) for arg in value.args]
            if value.op == "neg":
                return -args[0]
            left, right = args
            return {"+": lambda: left + right, "-": lambda: left - right,
                    "*": lambda: left * right, "/": lambda: left / right,
                    "**": lambda: left ** right}[value.op]()
        if isinstance(value, Call) and value.word == "abs":
            return abs(self.lower(value.args[0], inputs))
        if isinstance(value, Call) and value.word in ("exp", "expm1", "log", "sqrt", "sin", "cos"):
            return getattr(_pm, value.word)(self.lower(value.args[0], inputs))
        if isinstance(value, Term):
            raise NotYetElaborated("the term %s read in the value of another" % value, _LATER,
                                   site=value.site)
        raise NotYetElaborated("%s in a value of the model" % text(value),
                               "N11 (expression closure, C1)")

    def _scheme_value(self, value: Any, inputs: dict) -> Any:
        """The words that only the body of a scheme lowers in this lot: the reads of its entities,
        ``where`` and its comparisons, ``largest`` and ``smallest`` (b6 ``where``, ``maximum`` and
        ``minimum``, whose derivative stays the one of b6). ``None`` for every other value."""
        from ._scheme import SchemeLeaf, is_radius_pair, speed_pair_index

        if isinstance(value, SchemeLeaf):
            if value.key in inputs:
                return inputs[value.key]
            return self.lower(value.expansion, inputs)
        if ("speed",) in inputs and is_radius_pair(value):
            return inputs[("speed",)]
        pair = speed_pair_index(value) if ("speeds", 0) in inputs else None
        if pair is not None:
            return inputs[("speeds", pair)]
        if isinstance(value, Predicate):
            if value.op not in ("<", "<=", ">", ">="):
                raise NotYetElaborated("%s in the body of a scheme" % text(value), _LATER)
            left, right = (self.lower(arg, inputs) for arg in value.args)
            return {"<": lambda: left < right, "<=": lambda: left <= right,
                    ">": lambda: left > right, ">=": lambda: left >= right}[value.op]()
        if isinstance(value, Call) and value.word == "where":
            predicate, then, otherwise = value.args
            then_value, otherwise_value = self.lower(then, inputs), self.lower(otherwise, inputs)
            return _pm.where(self.lower(predicate, inputs), lambda: then_value,
                             lambda: otherwise_value)
        if isinstance(value, Call) and value.word in ("largest", "smallest"):
            fold = _pm.maximum if value.word == "largest" else _pm.minimum
            items = [self.lower(arg, inputs) for arg in value.args]
            result = items[0]
            for item in items[1:]:
                result = fold(result, item)
            return result
        return None

    def code(self, value: Any) -> str:
        """The b6 spelling of a value, for the report."""
        if isinstance(value, Quantity):
            return "%s[0]" % value.name
        if isinstance(value, Call):
            word = value.word if value.word == "abs" else "pm." + value.word
            return "%s(%s)" % (word, ", ".join(self.code(arg) for arg in value.args))
        if isinstance(value, Arith):
            if value.op == "neg":
                return "-(%s)" % self.code(value.args[0])
            return "(%s %s %s)" % (self.code(value.args[0]), value.op, self.code(value.args[1]))
        if isinstance(value, MathConstant):
            return repr(value.value)
        return text(value)

    def declarations(self, b6: Any) -> dict:
        model = self.model
        for constant in model.constants:
            param = ConstParam(constant.name, float(constant.value))
            self.values[constant] = b6.value(b6.param(param))
            self.rows.append(Row(
                'model.constant(%r, value=%r)' % (constant.name, constant.value), "model",
                '%s = model.value(model.param(ConstParam(%r, %r)))'
                % (constant.name, constant.name, float(constant.value)),
                [Choice("value", constant.value, "stated"),
                 Choice("binding", "folded at compile time", "derived")]))
        for parameter in model.parameters:
            options: dict = {"default": float(parameter.default)}
            if isinstance(parameter.admissible, Positive):
                options["domain"] = _Positive()
            param = RuntimeParam(parameter.name, **options)
            handle = b6.param(param)
            self.params[parameter] = handle
            self.values[parameter] = b6.value(handle)
            choices = [Choice("default", parameter.default, "stated")]
            if parameter.admissible is not None:
                choices.append(Choice("admissible", parameter.admissible, "stated"))
            self.rows.append(Row(
                "model.parameter(%r, default=%r)" % (parameter.name, parameter.default), "model",
                "%s = model.value(model.param(RuntimeParam(%r, default=%r%s)))"
                % (parameter.name, parameter.name, float(parameter.default),
                   ", domain=Positive()" if "domain" in options else ""), choices))
        evolved = [balance.quantity for balance in model.balances]
        states: dict = {}
        for quantity in model.quantities:
            if not any(quantity is other for other in evolved):
                raise NotYetElaborated(
                    "the quantity %s, which no balance evolves (a constraint unknown or a "
                    "definition)" % quantity, "a later lot of N15 with the constraints of N10",
                    site=quantity.site)
            handle = b6.state(quantity.name, components=(quantity.name,))
            states[quantity] = handle
            self.values[quantity] = handle[0]
            self.rows.append(Row(
                "model.quantity(%r, %r)" % (quantity.name, quantity.space), "model",
                "%s = model.state(%r, components=(%r,))"
                % (quantity.name, quantity.name, quantity.name),
                [Choice("evolved", "by the balance of %s" % quantity, "derived")]))
        return states

    def terms(self, b6: Any, frame: Any, states: dict) -> dict:
        model = self.model
        integrated = [term for balance in model.balances for term in balance.terms]
        for term in model.terms:
            if not any(term is other for other in integrated):
                raise CoverageNotExact(
                    "the term %s of %s is in no balance; every term is integrated (T5)"
                    % (term, term.quantity),
                    write="the term in the balance of its quantity, or no such term",
                    site=term.site)
        self.numerics.check_coverage(model.balances, model.terms)
        self.names = model_names(model)
        elaborated: dict = {}
        for term in model.terms:
            name = self.names[term]
            origin = "stated" if term.name is not None else "derived"
            state = states[term.quantity]
            if isinstance(term, DiffusiveFlux):
                elaborated[term] = self.diffusive(b6, term, name, origin, state)
            else:
                elaborated[term] = self.flux(b6, frame, term, name, origin, state)
        return elaborated

    def flux(self, b6: Any, frame: Any, term: Flux, name: str, origin: str, state: Any) -> Any:
        axes = frame.axes
        slopes = term.slopes()
        waves = None if slopes is None else {
            axis: (self.lower(slope),) for axis, slope in zip(axes, slopes, strict=True)}
        handle = b6.flux(name, state=state, frame=frame,
                         components={axis: (self.lower(component),) for axis, component
                                     in zip(axes, term.components, strict=True)}, waves=waves)
        choices = [Choice("name", name, origin)]
        choices.append(Choice("waves", "none: dF/%s is not derived in this lot" % term.quantity
                              if slopes is None else "dF/d%s = %s"
                              % (term.quantity, _per_axis_text(slopes, text)), "derived"))
        signed = ""
        scheme = self.numerics.scheme_of(term)
        if slopes is not None and getattr(scheme, "reads_speeds", lambda key: False)(term):
            # R9: the slowest and the fastest speeds of a scalar flux are both dF/du, the pair
            # that the face input speeds of riemann.User reads (N3 lot 1, hll()).
            b6.wave_speeds(handle, frame=frame, values={
                axis: (self.lower(slope), self.lower(slope))
                for axis, slope in zip(axes, slopes, strict=True)})
            signed = ", and model.wave_speeds(%s, frame=frame, values=the pair (dF/du, dF/du))" \
                % name
            choices.append(Choice("signed waves", "(dF/d%s, dF/d%s), read by %r"
                                  % (term.quantity, term.quantity, scheme), "derived"))
        names = FRAME_AXES[:len(axes)]
        self.rows.append(Row(
            "model.flux(of=%s, value=%s)" % (term.quantity, text(term.value)), "model",
            "%s = model.flux(%r, state=%s, frame=frame, components={%s}, waves=%s)"
            % (name, name, term.quantity.name,
               ", ".join("%s: (%s,)" % (axis, self.code(component))
                         for axis, component in zip(names, term.components, strict=True)),
               "None" if slopes is None else "{%s}" % ", ".join(
                   "%s: (%s,)" % (axis, self.code(slope))
                   for axis, slope in zip(names, slopes, strict=True))) + signed, choices))
        return handle

    def diffusive(self, b6: Any, term: DiffusiveFlux, name: str, origin: str, state: Any) -> Any:
        (variable,) = term.variables
        value = self.lower(term.coefficient) * _pm.grad(self.values[variable])
        handle = b6.diffusive_flux(name, state=state, value=value)
        self.rows.append(Row(
            "model.diffusive_flux(of=%s, value=%s)" % (term.quantity, text(term.value)),
            "model",
            "%s = model.diffusive_flux(%r, state=%s, value=%s * pm.grad(%s[0])), the F^d of R7, "
            "opposite to the physical flux" % (name, name, term.quantity.name,
                                              self.code(term.coefficient), variable.name),
            [Choice("name", name, origin),
             Choice("differentiated variables", ", ".join(map(str, term.variables)), term.origin),
             Choice("k", text(term.coefficient), "derived"),
             Choice("dissipation", _dissipation(term.coefficient), "derived")]))
        return handle

    def rate(self, b6: Any, balance: Balance, terms: dict, states: dict) -> Any:
        """The b6 rate of the transport terms of a balance; a diffusive term is a stencil of the
        program (``_StencilTerm``), so it is not a term of the b6 rate (N3 lot 1)."""
        rhs = None
        parts = []
        for term, coefficient in balance.occurrences:
            if coefficient not in (1, -1):
                raise NotYetElaborated(
                    "the weight %s of div(%s) in the balance of %s, a weight other than 1 or -1"
                    % (coefficient, term, balance.quantity), _WEIGHTED, site=balance.site)
        stencils = []
        for term, coefficient in balance.occurrences:
            if isinstance(term, DiffusiveFlux):
                stencils.append("%sthe StencilFlux of %s"
                                % ("" if coefficient == -1 else "-", self.names[term]))
                continue
            piece = _pm.div(terms[term])
            spelled = "div(%s)" % self.names[term]
            if rhs is None:
                rhs = piece if coefficient == 1 else -piece
                parts.append(spelled if coefficient == 1 else "-" + spelled)
            elif coefficient == 1:
                rhs = rhs + piece
                parts.append("+ " + spelled)
            else:
                rhs = rhs - piece
                parts.append("- " + spelled)
        name = balance.name if balance.name is not None else balance.quantity.name
        program = (", and in the program %s" % " + ".join(stencils)) if stencils else ""
        origin = Choice("name", name, "stated" if balance.name is not None else "derived")
        if rhs is None:
            self.rows.append(Row("model.balance(%s)" % balance, "model",
                                 "no rate of b6: in the program %s" % " + ".join(stencils),
                                 [origin]))
            return None
        handle = b6.rate(name, equation=_pm.ddt(states[balance.quantity]) == rhs)
        self.rows.append(Row(
            "model.balance(%s)" % balance, "model",
            "model.rate(%r, equation=pm.ddt(%s) == %s)%s"
            % (name, balance.quantity.name, " ".join(parts), program), [origin]))
        return handle

    # The numerics.

    def plan(self, rates: dict, terms: dict, states: dict) -> tuple:
        plan = DiscretizationPlan()
        selections: dict = {}
        for balance in self.model.balances:
            weights = {id(term): coefficient for term, coefficient in balance.occurrences}
            if rates[balance] is None:
                # Only stencils: the state has no rate of b6 to carry its storage, so it is stated,
                # with the one ghost layer of the two-point StencilFlux (D7).
                plan.states.add(states[balance.quantity], StateStorage(ghost_depth=1))
            for term in balance.terms:
                scheme = self.numerics.scheme_of(term)
                slope = _per_axis(term.slopes()) if isinstance(term, Flux) else None
                speed = _per_axis(term.unit_fluxes()) if isinstance(term, Flux) else None
                realization = scheme.realize(TermView(term, terms[term], states[term.quantity],
                                                      slope=slope, unit_flux=speed,
                                                      lower=self.lower))
                if isinstance(term, DiffusiveFlux):
                    if not isinstance(realization, StencilRealization):
                        raise TypeError("the scheme %r of the diffusive flux %s gives %r; a "
                                        "diffusive key is realized by a face scheme"
                                        % (scheme, term, realization))
                    selections[term] = _StencilTerm(realization.flux, self.names[term],
                                                    weights[id(term)])
                else:
                    view = rates[balance].select(terms[term])
                    selections[term] = view
                    plan.rates.add(view, realization)
                self.linear[term] = self._linear(scheme, TermView(
                    term, terms[term], states[term.quantity], slope=slope, unit_flux=speed,
                    lower=self.lower), balance)
                choices = [Choice("scheme", scheme, "stated")]
                choices.extend(Choice("option %s" % name, value, origin)
                               for name, value, origin in getattr(scheme, "option_choices",
                                                                  list)())
                if speed is not None:
                    choices.append(Choice("speed", "s = F(1)·n = %s" % text(speed), "derived"))
                self.rows.append(Row(
                    "schemes={%s: %r}" % (term, scheme), "numerics",
                    ("%s, %s" % (self.names[term], scheme.describe()))
                    if isinstance(term, DiffusiveFlux) else
                    "plan.rates.add(%s.select(%s), %s)"
                    % (balance.name or balance.quantity.name, self.names[term],
                       scheme.describe()), choices))
        for quantity in states:
            self.rows.append(Row("represent={%s: %r}" % (quantity, self.numerics
                                                          .representation_of(quantity)),
                                 "numerics", "the cell averages of %s, projected by "
                                 "ConservativeCellAverage()" % quantity,
                                 [Choice("representation", "pops.CellAverage()", "stated")]))
        return plan, selections

    def _linear(self, scheme: Any, view: TermView, balance: Balance) -> Any:
        """The linear map of the rate of a term, asked when a solve reads it, never before, of the
        solver that ``solvers=`` states (N3 lot 1, D7): the matrix-free apply is the solver's,
        beside ``cg()`` in ``pops.lib.solve``, and it recognizes the scheme by the normal form of
        its face, so that a copy of ``two_point()`` written in a case is applied as the word."""
        weight = next(-coefficient if isinstance(term, DiffusiveFlux) else coefficient
                      for term, coefficient in balance.occurrences if term is view.term)

        def linear() -> Any:
            operator = next((solver.linear_operator for _, solver in self.program.solvers
                             if hasattr(solver, "linear_operator")), None)
            if operator is None:
                # no solver states one: the apply of the library, whose solve N8 then refuses
                # without a solver (auto() of SV11)
                from pops.lib.solve import linear_operator as operator
            return operator(view, scheme=scheme, weight=weight, layout=self.layout_facts)
        return linear

    # The program.

    def method_components(self, value: MethodValue, selections: dict) -> dict:
        model = self.model
        components: dict = {}
        covered: list = []
        for name in value.rate_options():
            option = value.options[name]
            if isinstance(option, Model):
                if option is not model:
                    raise ValueError("%s(%s=%r): the model is not the model of the case"
                                     % (value.name, name, option))
                chosen = [term for balance in model.balances for term in balance.terms]
            else:
                listed = option if isinstance(option, list) else [option]
                for term in listed:
                    if term.model is not model:
                        raise ValueError("%s(%s=...): the term %s belongs to another model"
                                         % (value.name, name, term))
                chosen = [term for balance in model.balances for term in balance.terms
                          if any(term is item for item in listed)]
            for term in chosen:
                if any(term is other for other in covered):
                    raise CoverageNotExact(
                        "the term %s is integrated twice by the method %s" % (term, value.name),
                        write="each term in one rate option of %s" % value.name, site=value.site)
                covered.append(term)
            components[name] = _RateSum([selections[term] for term in chosen],
                                        [self.linear[term] for term in chosen])
        for balance in model.balances:
            for term in balance.terms:
                if not any(term is other for other in covered):
                    raise CoverageNotExact(
                        "the term %s of the balance of %s is integrated by no rate option of %s"
                        % (term, balance.quantity, value.name),
                        write="%s(rate=model), or a rate option that names it" % value.name,
                        site=value.site)
        return components

    def advance(self, case6: Any, block: Any, states: dict, selections: dict) -> Any:
        advance = self.program.advance
        value = advance.method
        components = self.method_components(value, selections)
        trace = _Trace(value.site)
        specialized = specialize(value, components, trace)
        retry = _n8.Retry() if advance.attempts is None else _n8.Retry(
            max_attempts=advance.attempts)
        (quantity,) = states
        request = None if advance.tolerance is None else advance.tolerance.request()
        realization = self.realization(value)
        outputs = self.outputs(block)
        rule = self.rule
        if rule is not None and request is not None:
            raise NotYetElaborated("%r with tolerance=, the request of the solves of a step that "
                                   "a step rule chooses" % (rule,), _STEP_REQUEST,
                                   site=advance.site)
        proposal = _n8.Proposal(float(advance.dt), request=request) if rule is None \
            else rule.realize()
        try:
            elaboration = _n8.elaborate(_n8.Advance(specialized, retry=retry), case=case6,
                                        state=block[states[quantity]], proposal=proposal,
                                        realization=realization, outputs=outputs)
        except Exception as error:
            translated = translate(error, trace)
            if translated is error:
                raise
            if isinstance(translated, SurfaceRefusal) and isinstance(
                    error, (_relations.UnknownSolvePath, _relations.MissingRequest)):
                translated.site = self.program.site
                translated.args = (translated._message(),)
            raise translated.with_traceback(None) from None
        self.rows.append(Row(
            repr(advance), "program",
            "elaborate(Advance(%s(%s), retry=Retry(max_attempts=%d)), case=case, state=block[%s], "
            "proposal=%s%s), the traced method of N8"
            % (value.name, ", ".join(components), retry.max_attempts, quantity.name,
               "Proposal(%r%s)" % (float(advance.dt), "" if request is None
                                   else ", request=%r" % (request,)) if rule is None
               else repr(proposal),
               "" if not realization else ", realization={%s}" % ", ".join(
                   "%r: %r" % item for item in realization.items())),
            [Choice("method", "%s, %d evaluations of its rates"
                    % (value.name, len(elaboration.trace.evaluations)), "stated"),
             Choice("dt", advance.dt, "stated"),
             *([] if rule is None else [Choice(
                 "speeds", "the wave speeds dF/du of %s, recorded by b6 on each cell (R9, SD44); "
                           "each step lands on the frontiers of the run" % ", ".join(
                               text(flux) for flux in rule.fluxes), "derived")]),
             Choice("attempts", retry.max_attempts,
                    "default" if advance.attempts is None else "stated"),
             Choice("accept", "finite()", "default")]
            + ([Choice("tolerance", advance.tolerance, "stated")]
               if advance.tolerance is not None else [])))
        for site in elaboration.trace.solves:
            self.rows.append(self.solve_row(value, site))
        return elaboration

    def realization(self, value: MethodValue) -> dict:
        """The realization of N8 that ``solvers=`` states: a method keys every solve of its
        attempt, ``method.name`` the solve named ``name`` (2.6.2, 5.5)."""
        realization: dict = {}
        for key, solver in self.program.solvers:
            method = key if isinstance(key, MethodValue) else key.method
            if method is not value:
                raise UnknownSolvePath(
                    "the key %r of solvers= names no solve: the advance runs %r" % (key, value),
                    write="solvers={%s: solver}, keyed by the method of the advance"
                          % value.name, site=self.program.site)
            path = "%s/*" % value.name if isinstance(key, MethodValue) \
                else "%s/%s" % (value.name, key.name)
            realization[path] = solver
        return realization

    def solve_row(self, value: MethodValue, site: Any) -> Row:
        """The row of ``pops.inspect`` of a solve of the method (5.5, K5)."""
        request = site.request
        claims = "; ".join("%s: %s" % (name, claim.description) for name, claim in site.claims)
        proofs = "; ".join(claim.proof for _, claim in site.claims)
        return Row(
            "pops.solve(...) in %s" % value.name, "program",
            "solve(Problem(unknowns=unknowns(U=B.state.space), equations=(rel,)), request=%r%s), "
            "then require_success: LinearProblem(matrix_free_operator a I + sum_k b_k L_k, "
            "initial_guess=B.state), solver %s"
            % (request, "" if site.name is None else ", name=%r" % site.name,
               site.solver.describe()),
            [Choice("path", site.path, "derived" if site.name is None else "stated"),
             Choice("request", request, "stated"),
             Choice("guess", "the state at the start of the attempt (SV3)", "derived"),
             Choice("solver", site.solver, "stated"),
             Choice("implicit operator", claims, "derived"),
             Choice("SPD certificate", "derived (IR16) from the substituted operator, until "
                    "IR16 derives it from the face blocks (§17.1): %s" % proofs if site.spd
                    else "not derived", "derived"),
             Choice("failure", "SolveFailed fails the attempt (T38)", "derived")])

    def step_rule(self, dt: Any) -> Any:
        """The step rule of a library given as ``dt=`` (lot D-E), checked against what b6's
        selection reads: the speeds of every transport flux of the model, derived (R9), and no
        diffusive limit; ``None`` for a number."""
        from ._step import StepRuleValue

        if not isinstance(dt, StepRuleValue):
            return None
        model = self.model
        for flux in dt.fluxes:
            if flux.model is not model:
                raise ValueError("%r reads %s, a flux of another model" % (dt, text(flux)))
        named = {id(flux) for flux in dt.fluxes}
        # a term in no balance is left to the coverage of the terms, which refuses it (T5)
        integrated = {id(term) for balance in model.balances for term in balance.terms}
        transports = [term for term in model.terms if isinstance(term, Flux)
                      and not isinstance(term, DiffusiveFlux) and id(term) in integrated]
        missing = [term for term in transports if id(term) not in named]
        if missing:
            raise NotYetElaborated(
                "%r, a step rule that leaves out the flux %s of the model, whose speeds b6's "
                "selection reads with the others" % (dt, text(missing[0])), _STEP_PARTS,
                site=dt.site)
        diffusive = [term for term in model.terms if isinstance(term, DiffusiveFlux)
                     and id(term) in integrated]
        if diffusive:
            raise NotYetElaborated(
                "%r in a model with the diffusive flux %s, whose limit b6's selection adds to "
                "the step" % (dt, text(diffusive[0])), _STEP_PARTS, site=dt.site)
        for flux in dt.fluxes:
            if flux.slopes() is None:
                raise NotYetElaborated(
                    "%r reads the speeds of %s, whose waves dF/%s are not derived (R9); the speed "
                    "that the face fluxes record (SD44)" % (dt, text(flux), flux.quantity),
                    _FACE_SPEEDS, site=dt.site)
        return dt

    def outputs(self, block: Any) -> tuple:
        """The Outputs of N8 of the case's outputs, in their order (CC14; lot G)."""
        lowered = []
        for output in self.case.outputs:
            for quantity in output.quantities():
                if quantity.model is not self.model:
                    raise ValueError("pops.Output(%r) publishes %s, a quantity of another model"
                                     % (output.record, quantity))
            value = output.lowered(block)
            lowered.append(value)
            schedule = ("every(%d)" % output.every_steps if output.every_steps is not None
                        else "at=every_dt(%r)" % output.every_time if output.every_time is not None
                        else "at=%r" % (output.times,))
            if output.every_steps is not None:
                schedule += (", after every k-th accepted step of each run, counted from its "
                             "start t0 (N8)")
            elif output.every_time is not None:
                schedule += ", at t0 + k h, t0 the start of each run (N8)"
            self.rows.append(Row(
                repr(output), "program",
                "Output(lambda B, interval: Product(%s), %s), evaluated by the compiled run (CC14)"
                % (", ".join("%s=B.state" % name for name, _ in output.record.items()), schedule),
                [Choice("when", output.when, "stated"),
                 Choice("write", "result.outputs" if output.write is None else output.write,
                        "default" if output.write is None else "stated")]))
        self.lowered_outputs = tuple(lowered)
        return self.lowered_outputs

    def initial(self, case6: Any, block: Any, states: dict, frame: Any) -> None:
        """The initial value of each evolved quantity (CN12, Q3): a datum, supplied at bind; a
        value that reads no coordinate, whose cell average is exact; ``pops.project(e, ...)``,
        lowered on the analytic route of b6 (``Analytic`` under ``ConservativeCellAverage``)
        with the exact cell integral that its rule or its integral gives (lot D-E)."""
        from pops.lib.initial import BindArray   # the datum supplied at bind (Q3), a b6 value

        from ._project import Projection

        program = self.program
        for key, value in program.initial:
            if not any(key is quantity for quantity in states):
                raise ValueError("pops.Program(initial={%s: ...}): %s is not a quantity that the "
                                 "model evolves" % (key, key))
            if isinstance(value, Datum) and value.model is not self.model:
                raise ValueError("pops.Program(initial={%s: %s}): the datum belongs to another "
                                 "model" % (key, value))
        for quantity, state in states.items():
            value = program.initial_of(quantity)
            if value is None:
                raise AmbiguousElaboration(
                    "the evolved quantity %s has no initial value" % quantity,
                    write='pops.Program(initial={%s: d}), d = model.data("d", %r)'
                          % (quantity, quantity.space), site=program.site)
            if isinstance(value, Datum):
                provider, spelling = BindArray(), "BindArray()"
                choices = [Choice("value", "the datum %s, supplied at bind" % value, "stated")]
                stated = str(value)
            elif isinstance(value, Projection):
                provider, spelling, choices = self.projection(value, quantity, frame)
                stated = repr(value)
            else:
                provider, spelling, choices = self.exact_value(value, frame)
                stated = text(value)
            case6.initials.add(InitialCondition(state=block[state], value=provider,
                                                projection=ConservativeCellAverage()))
            self.rows.append(Row(
                "initial={%s: %s}" % (quantity, stated), "program",
                "case.initials.add(InitialCondition(state=block[%s], value=%s, "
                "projection=ConservativeCellAverage()))" % (quantity.name, spelling),
                choices + [Choice("projection", "ConservativeCellAverage()", "derived")]))

    def exact_value(self, value: Any, frame: Any) -> tuple:
        """A value that reads no coordinate: its cell average is the value itself, which b6's
        analytic route gives exactly (one literal, or equal samples)."""
        from pops.lib.initial import Analytic

        lowered = self.analytic(value, frame, ())
        return (Analytic(frame=frame, components=(lowered,)),
                "Analytic(frame=frame, components=(%s,))" % text(value),
                [Choice("value", text(value), "stated"),
                 Choice("cell average", "exact, the value reads no coordinate", "derived")])

    def projection(self, value: Any, quantity: Any, frame: Any) -> tuple:
        """``pops.project(e, quadrature=Q)``: the cell integral measure x the sum over the tensor
        nodes of the weights (each divided by the length of its interval) times e at the nodes
        mapped onto the cell; ``integral=I``: I run on the bounds of the cell (CN12)."""
        from pops.analytic import CellBounds, coordinate
        from pops.lib.initial import Analytic

        from ._project import ProjectionCell, reads_coordinate

        support = quantity.space.support
        bounds = CellBounds(frame)
        axes = frame.axes
        point = tuple(coordinate(frame, axis) for axis in axes)
        component = self.analytic(value.expression, frame, point)
        if value.integral is not None:
            from ._expr import Expr

            result = value.integral(ProjectionCell(support))
            if not (is_number(result) or isinstance(result, Expr)):
                raise TypeError("pops.project(e, integral=I): I(cell) returns the integral of e "
                                "over the cell, an expression, got %r" % (result,))
            if reads_coordinate(result):
                raise TypeError("pops.project(e, integral=I): I(cell) reads the bounds of the "
                                "cell, cell.lower(axis) and cell.upper(axis), not a coordinate")
            integral = self.analytic(result, frame, (), bounds)
            spelling, rule = text(result), Choice("integral", text(result), "stated")
        else:
            rules = value.rules(len(axes))
            mapped = []
            for axis, rule in zip(axes, rules, strict=True):
                low, high = (float(end) for end in rule.interval)
                lower, upper = bounds.lower(axis), bounds.upper(axis)
                width = upper - lower
                nodes = []
                for node, weight in zip(rule.nodes, rule.weights, strict=True):
                    fraction = (node - low) / (high - low)
                    where = lower if fraction == 0.0 else upper if fraction == 1.0 \
                        else lower + width * fraction
                    nodes.append((where, weight / (high - low)))
                mapped.append(nodes)
            terms = []
            for choice in _tensor(mapped):
                weight = 1.0
                for _, factor in choice:
                    weight *= factor
                sample = self.analytic(value.expression, frame,
                                       tuple(where for where, _ in choice))
                terms.append(sample if weight == 1.0 else weight * sample)
            average = terms[0]
            for term in terms[1:]:
                average = average + term
            integral = bounds.measure * average
            spelling = "the cell measure x the rule of %r" % (value.quadrature,)
            rule = Choice("quadrature", value.quadrature, "stated")
        provider = Analytic(frame=frame, components=(component,), cell_integrals=(integral,))
        return (provider,
                "Analytic(frame=frame, components=(%s,), cell_integrals=(%s,))"
                % (text(value.expression), spelling),
                [Choice("value", text(value.expression), "stated"), rule,
                 Choice("cell average", "the cell integral divided by the cell measure, "
                                        "native", "derived")])

    def analytic(self, value: Any, frame: Any, point: tuple, bounds: Any = None) -> Any:
        """The analytic expression of b6 (``pops.analytic``) of a value of an initial value: the
        coordinates read ``point``, the bounds of the cell ``bounds``; a constant of the model is
        its number and a parameter its handle, bound at bind."""
        from pops import analytic as an

        from ._project import CellBound

        if is_number(value):
            return an.constant(float(_number(value)))
        if isinstance(value, MathConstant):
            if not _math.isfinite(value.value):
                raise NotYetElaborated("%s in an initial value" % value.name, _LATER)
            return an.constant(value.value)
        if isinstance(value, ModelConstant):
            return an.constant(float(value.value))
        if isinstance(value, Parameter):
            return an.param(self.params[value])
        if isinstance(value, Coordinate):
            return point[value.index]
        if isinstance(value, CellBound):
            axes = frame.axes
            if value.index is None:
                return bounds.measure
            return getattr(bounds, value.side)(axes[value.index])
        if isinstance(value, Arith):
            args = [self.analytic(arg, frame, point, bounds) for arg in value.args]
            if value.op == "neg":
                return -args[0]
            left, right = args
            if value.op == "**":
                exponent = value.args[1]
                if not is_number(exponent):
                    raise NotYetElaborated("%s, a power whose exponent is not a number, in an "
                                           "initial value" % text(value), _LATER)
                return left ** float(_number(exponent))
            return {"+": lambda: left + right, "-": lambda: left - right,
                    "*": lambda: left * right, "/": lambda: left / right}[value.op]()
        if isinstance(value, Call) and value.word in _ANALYTIC_WORDS:
            args = [self.analytic(arg, frame, point, bounds) for arg in value.args]
            return getattr(an, value.word)(*args)
        if isinstance(value, Call):
            raise NotYetElaborated("%s in an initial value, a word that b6's analytic "
                                   "expressions (pops.analytic) do not have" % text(value),
                                   _ANALYTIC_LOT)
        raise NotYetElaborated("%s in an initial value" % text(value),
                               "N11 (expression closure, C1)")

    def layout(self, support: Box, frame: Any) -> Any:
        case = self.case
        uniform = case.layout_of(support)
        if uniform is None:
            raise AmbiguousElaboration("the support %r has no layout" % (support,),
                                       write="layout={Ω: pops.Uniform(cells=c)}", site=case.site)
        for key, _ in case.layout:
            if key is not support:
                raise ValueError("pops.Case(layout=...) names the support %r, on which no "
                                 "quantity of the model lives" % (key,))
        periodic = tuple(axis for axis, (_, _, flag)
                         in zip(frame.axes, support.bounds(), strict=True) if flag)
        layout = _Uniform(CartesianGrid(frame=frame, cells=uniform.cells,
                                        periodic=PeriodicAxes(periodic) if periodic else None))
        self.rows.append(Row(
            "layout={%r: %r}" % (support, uniform), "case",
            "Uniform(CartesianGrid(frame=frame, cells=%r, periodic=%s))"
            % (uniform.cells, "PeriodicAxes((%s))" % "".join(
                "%s," % axis for axis in support._names if support.periodic(axis))
               if periodic else None),
            [Choice("cells", uniform.cells, "stated")]))
        return layout

    def boundaries(self, support: Box) -> list:
        """The relations of ``model.on`` that close each boundary piece of the support (2.6.2,
        D9): each piece that a flux crosses takes one relation of the quantity, and the relations
        take the boundary policy of the numerics (``MissingBoundaryRealization`` otherwise)."""
        pieces = support.piece_names()
        diffusive = [term for term in self.model.terms if isinstance(term, DiffusiveFlux)]
        if pieces and diffusive:
            raise MissingBoundaryRealization(
                "the diffusive flux %s crosses the boundary pieces %s of %r, which no boundary "
                "relation closes; its face is realized by b6's StencilFlux, which is periodic"
                % (diffusive[0], ", ".join(pieces), support),
                write="pops.Periodic(lower, upper) for an axis whose ends are identified; the "
                      "boundary policies of a diffusive face are elaborated by G04",
                site=support.site)
        relations = list(self.model.boundary_relations)
        if not pieces:
            return relations
        if not self.model.terms:
            if relations:
                raise NotYetElaborated("a boundary relation of a balance with no flux",
                                       _G04, site=relations[0].site)
            return relations
        (balance,) = self.model.balances
        quantity = balance.quantity
        missing = [name for name in pieces
                   if not any(relation.quantity is quantity
                              and any(piece.name == name for piece in relation.pieces)
                              for relation in relations)]
        if missing:
            raise MissingBoundaryRealization(
                "the fluxes of %s cross the boundary pieces %s of %r, which no boundary relation "
                "closes" % (quantity, ", ".join(missing), support),
                write="pops.Periodic(lower, upper) for an axis whose ends are identified, or a "
                      "relation of each piece, model.on(Ω.boundary.%s, %s == g) or "
                      "model.on(..., outflow(%s))" % (missing[0], quantity, quantity),
                site=support.site)
        if self.numerics.boundary is None:
            raise MissingBoundaryRealization(
                "the relation %r has no realization: the numerics state no boundary policy"
                % relations[0], write="pops.Numerics(..., boundary=ghosts())",
                site=self.numerics.site)
        return relations

    def boundary_data(self, relations: list) -> dict:
        """The data g of each value relation, lowered on the model of b6 before its block is
        made, which registers the parameters that g reads."""
        return {relation: _number(relation.data) if is_number(relation.data)
                else self.lower(relation.data)
                for relation in relations if relation.form == "Value"}

    def boundary_conditions(self, support: Box, frame: Any, block: Any, plan: Any,
                            states: dict, relations: list, data: dict) -> None:
        """Run the boundary policy under each relation, check that the ghost it gives is a
        ghost law of b6, and add the transport boundaries of the block to the plan (T5): the
        ghost 2 g - u_k of a value and the ghost u_1 of no data, b6's two laws of master
        (2.6.5), whose b6 conditions ``pops.lib.numerics.ghost_laws`` names (the bridge of the
        boundary policy, until G04 generates the ghost fills)."""
        if not relations:
            return
        from pops.boundary import TransportBoundarySet
        from pops.lib.numerics.ghost_laws import realize

        from ._scheme import SchemeLeaf, normal_form

        policy = self.numerics.boundary
        laws: dict = {}
        for relation in relations:
            quantity = relation.quantity
            state = block[states[quantity]]
            for _piece, k, value in policy.ghosts(relation, _GHOST_DEPTHS):
                if relation.form == "Value":
                    expected = 2 * relation.data - SchemeLeaf(("inner", k), quantity, "u_%d" % k)
                else:
                    expected = SchemeLeaf(("inner", 1), quantity, "u_1")
                if normal_form(value) != normal_form(expected):
                    raise NotYetElaborated(
                        "the ghost %s that %r gives under %r at depth %d, which is not a ghost "
                        "law of b6 (2 g - u_k for a value, u_1 for no data)"
                        % (text(value), policy, relation, k), _GENERATED_GHOSTS,
                        site=relation.site)
            law, spelling, ghost = realize(
                relation.form, state, data.get(relation), "block[%s]" % quantity.name,
                self.code(relation.data) if relation.form == "Value" else "")
            for piece in relation.pieces:
                laws[getattr(frame.boundaries, piece.name)] = law
            self.rows.append(Row(
                repr(relation), "numerics",
                "%s on %s" % (spelling, ", ".join(piece.name for piece in relation.pieces)),
                [Choice("form", relation.form, "derived"),
                 Choice("ghost", "%s, by %r" % (ghost, policy), "derived")]))
        periodic = tuple(axis for axis, (_, _, flag)
                         in zip(frame.axes, support.bounds(), strict=True) if flag)
        plan.boundaries.add(TransportBoundarySet(
            laws, periodic=PeriodicAxes(periodic) if periodic else None))

    def run(self) -> Elaborated:
        model = self.model
        if not model.balances:
            raise NotYetElaborated("a model that evolves no quantity",
                                   "a later lot of N15 with the constraints of N10",
                                   site=model.site)
        if len(model.balances) != 1:
            raise NotYetElaborated(
                "a model of several evolved quantities, whose state is a record (SD42)",
                "a later lot of N15 with the record states of N8", site=model.balances[1].site)
        support = self.support()
        relations = self.boundaries(support)
        self.rule = self.step_rule(self.program.advance.dt)
        frame = self.frame(support)
        self.layout_facts = {"uniform": True, "cartesian": True,
                             "periodic": all(support.periodic(axis) for axis in support._names)}
        b6 = _Model(model.name, frame=frame)
        self.rows.insert(0, Row("pops.Model(%r)" % model.name, "model",
                                "model = pops.Model(%r, frame=frame)" % model.name))
        states = self.declarations(b6)
        terms = self.terms(b6, frame, states)
        rates = {balance: self.rate(b6, balance, terms, states) for balance in model.balances}
        plan, selections = self.plan(rates, terms, states)
        data = self.boundary_data(relations)
        case6 = _Case(model.name)
        block = case6.block(model.name, b6)
        self.boundary_conditions(support, frame, block, plan, states, relations, data)
        case6.numerics(plan, block=block)
        self.initial(case6, block, states, frame)
        advance = self.advance(case6, block, states, selections)
        layout = self.layout(support, frame)
        return Elaborated(case=case6, layout=layout, block=block, model=b6, plan=plan,
                          states=states, names=self.names, advance=advance,
                          rows=tuple(self.rows), params=dict(self.params),
                          outputs=self.lowered_outputs,
                          representations={quantity: elaborated_representation(quantity, support)
                                           for quantity in states})


def inspect_case(case: Any) -> Report:
    """``pops.inspect(case)``: ``resolve(validate(case))`` of the elaborated case, and its rows."""
    from pops._api import resolve, validate

    elaborated = case.elaborated
    resolved = resolve(validate(elaborated.case), layout=elaborated.layout)
    return Report("case %r" % case.model.name, "1 (model, numerics, program)", elaborated.rows,
                  resolved=resolved)


def inspect_value(value: Any) -> Report:
    """The report of a value of the surface: its mathematics and its level (SU12)."""
    from ._method import MethodValue, SurfaceMethod
    from ._program import Advance, Program

    if isinstance(value, Model):
        rows = [Row("pops.Model(%r)" % value.name, "model",
                    "quantities %s; terms %s; balances %s" % (
                        ", ".join(map(str, value.quantities)) or "none",
                        ", ".join(map(str, value.terms)) or "none",
                        "; ".join(map(str, value.balances)) or "none"))]
        return Report("model %r" % value.name, "model", rows)
    if isinstance(value, Balance):
        return Report(text(value.relation), "model",
                      [Row(text(value.relation), "model", "the balance of %s" % value.quantity)])
    if isinstance(value, Term):
        kind = "diffusive flux" if isinstance(value, DiffusiveFlux) else "flux"
        choices = []
        if isinstance(value, DiffusiveFlux):
            choices = [Choice("differentiated variables", ", ".join(map(str, value.variables)),
                              value.origin), Choice("k", text(value.coefficient), "derived")]
        return Report(str(value), "model", [Row("%s of %s: %s" % (kind, value.quantity,
                                                                    text(value.value)),
                                                "model", "a term of %r" % value.model, choices)])
    if isinstance(value, (Quantity, ModelConstant, Parameter, Datum)):
        return Report(str(value), "model", [Row(str(value), "model", type(value).__name__)])
    if isinstance(value, Box):
        return Report(repr(value), "model", [Row(repr(value), "model", "a support of rank %d"
                                                 % value.rank)])
    if isinstance(value, (Numerics, SchemeValue)):
        return Report(repr(value), "numerics", [Row(repr(value), "numerics",
                                                    value.describe() if isinstance(
                                                        value, SchemeValue) else "a numerics")])
    if isinstance(value, (MethodValue, SurfaceMethod, Advance, Program)):
        return Report(repr(value), "program", [Row(repr(value), "program", type(value).__name__)])
    raise TypeError("pops.inspect takes a case or a value of the surface, got %r" % (value,))
