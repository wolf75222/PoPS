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
from collections.abc import Iterable
from typing import Any

from pops import math as _pm
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D, Cartesian2D, Cartesian3D
from pops.initial import InitialCondition
from pops.layouts import Uniform as _Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan
from pops.params import ConstParam, RuntimeParam
from pops.params import Positive as _Positive
from pops.physics import Model as _Model
from pops.problem import Case as _Case
from pops.projection import ConservativeCellAverage
from pops.time._program import elaboration as _n8

from ._derive import derivative, unit_flux
from ._expr import Arith, Call, is_number, text
from ._geometry import FRAME_AXES, Box, Coordinate, MathConstant
from ._method import MethodValue, _Trace, specialize, translate
from ._model import (
    Balance, Datum, DiffusiveFlux, Flux, Model, ModelConstant, Parameter, Positive, Quantity,
    Term, model_names,
)
from ._numerics import Numerics, SchemeValue, TermView
from ._refusal import (
    AmbiguousElaboration, CoverageNotExact, MissingBoundaryRealization, NotYetElaborated,
    SurfaceRefusal,
)

_LATER = "a later lot of N15"
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

    __slots__ = ("case", "layout", "block", "model", "plan", "states", "names", "advance", "rows")

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


def _number(value: Any) -> Any:
    """A number of the surface as b6 takes it: an exact fraction or a decimal becomes a float."""
    if isinstance(value, (Fraction, Decimal)):
        return float(value)
    return value


class _RateSum:
    """A rate option of a method: the sum of the selections of its terms, in balance order."""

    __slots__ = ("views",)

    def __init__(self, views: list) -> None:
        self.views = tuple(views)

    def __call__(self, state: Any) -> Any:
        result = self.views[0](state)
        for view in self.views[1:]:
            result = result + view(state)
        return result


class _Elaboration:
    def __init__(self, case: Any) -> None:
        self.case = case
        self.model: Model = case.model
        self.numerics: Numerics = case.numerics
        self.program = case.program
        self.rows: list = []
        self.values: dict = {}
        self.names: dict = {}

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

    def lower(self, value: Any) -> Any:
        """The b6 expression of a value of the model."""
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
            args = [self.lower(arg) for arg in value.args]
            if value.op == "neg":
                return -args[0]
            left, right = args
            return {"+": lambda: left + right, "-": lambda: left - right,
                    "*": lambda: left * right, "/": lambda: left / right,
                    "**": lambda: left ** right}[value.op]()
        if isinstance(value, Call) and value.word == "abs":
            return abs(self.lower(value.args[0]))
        if isinstance(value, Call) and value.word in ("exp", "expm1", "log", "sqrt", "sin", "cos"):
            return getattr(_pm, value.word)(self.lower(value.args[0]))
        if isinstance(value, Term):
            raise NotYetElaborated("the term %s read in the value of another" % value, _LATER,
                                   site=value.site)
        raise NotYetElaborated("%s in a value of the model" % text(value),
                               "N11 (expression closure, C1)")

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
            self.values[parameter] = b6.value(b6.param(param))
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
        (axis,) = frame.axes
        slope = derivative(term.value, term.quantity)
        waves = None if slope is None else {axis: (self.lower(slope),)}
        handle = b6.flux(name, state=state, frame=frame,
                         components={axis: (self.lower(term.value),)}, waves=waves)
        choices = [Choice("name", name, origin)]
        choices.append(Choice("waves", "none: dF/%s is not derived in this lot" % term.quantity
                              if slope is None else "dF/d%s = %s" % (term.quantity, text(slope)),
                              "derived"))
        self.rows.append(Row(
            "model.flux(of=%s, value=%s)" % (term.quantity, text(term.value)), "model",
            "%s = model.flux(%r, state=%s, frame=frame, components={x: (%s,)}, waves=%s)"
            % (name, name, term.quantity.name, self.code(term.value),
               "None" if slope is None else "{x: (%s,)}" % self.code(slope)), choices))
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
             Choice("k", text(term.coefficient), "derived")]))
        return handle

    def rate(self, b6: Any, balance: Balance, terms: dict, states: dict) -> Any:
        rhs = None
        parts = []
        for term, coefficient in balance.occurrences:
            # c div(F) of the physical flux F; F^d = -F for a diffusive flux (R7)
            weight = -coefficient if isinstance(term, DiffusiveFlux) else coefficient
            piece = _pm.div(terms[term])
            spelled = "div(%s)" % self.names[term]
            if rhs is None:
                rhs = piece if weight == 1 else -piece if weight == -1 else _number(weight) * piece
                parts.append(spelled if weight == 1 else "-" + spelled if weight == -1
                             else "%s * %s" % (weight, spelled))
            elif weight == 1:
                rhs = rhs + piece
                parts.append("+ " + spelled)
            elif weight == -1:
                rhs = rhs - piece
                parts.append("- " + spelled)
            else:
                rhs = rhs + _number(weight) * piece
                parts.append("+ %s * %s" % (weight, spelled))
        name = balance.name if balance.name is not None else balance.quantity.name
        handle = b6.rate(name, equation=_pm.ddt(states[balance.quantity]) == rhs)
        self.rows.append(Row(
            "model.balance(%s)" % balance, "model",
            "model.rate(%r, equation=pm.ddt(%s) == %s)"
            % (name, balance.quantity.name, " ".join(parts)),
            [Choice("name", name, "stated" if balance.name is not None else "derived")]))
        return handle

    # The numerics.

    def plan(self, rates: dict, terms: dict, states: dict) -> tuple:
        plan = DiscretizationPlan()
        selections: dict = {}
        for balance in self.model.balances:
            for term in balance.terms:
                view = rates[balance].select(terms[term])
                selections[term] = view
                scheme = self.numerics.scheme_of(term)
                slope = derivative(term.value, term.quantity) if isinstance(term, Flux) else None
                speed = unit_flux(term.value, term.quantity) if isinstance(term, Flux) else None
                realization = scheme.realize(TermView(term, terms[term], states[term.quantity],
                                                      slope=slope, unit_flux=speed))
                plan.rates.add(view, realization)
                choices = [Choice("scheme", scheme, "stated")]
                if speed is not None:
                    choices.append(Choice("speed", "s = F(1)·n = %s" % text(speed), "derived"))
                self.rows.append(Row(
                    "schemes={%s: %r}" % (term, scheme), "numerics",
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
            components[name] = _RateSum([selections[term] for term in chosen])
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
        try:
            elaboration = _n8.elaborate(_n8.Advance(specialized, retry=retry), case=case6,
                                        state=block[states[quantity]],
                                        proposal=_n8.Proposal(float(advance.dt)))
        except Exception as error:
            translated = translate(error, trace)
            if translated is error:
                raise
            raise translated.with_traceback(None) from None
        self.rows.append(Row(
            repr(advance), "program",
            "elaborate(Advance(%s(%s), retry=Retry(max_attempts=%d)), case=case, state=block[%s], "
            "proposal=Proposal(%r)), the traced method of N8"
            % (value.name, ", ".join(components), retry.max_attempts, quantity.name,
               float(advance.dt)),
            [Choice("method", "%s, %d evaluations of its rates"
                    % (value.name, len(elaboration.trace.evaluations)), "stated"),
             Choice("dt", advance.dt, "stated"),
             Choice("attempts", retry.max_attempts,
                    "default" if advance.attempts is None else "stated"),
             Choice("accept", "finite()", "default")]))
        return elaboration

    def initial(self, case6: Any, block: Any, states: dict) -> None:
        from pops.lib.initial import BindArray   # the datum supplied at bind (Q3), a b6 value

        program = self.program
        for key, datum in program.initial:
            if not any(key is quantity for quantity in states):
                raise ValueError("pops.Program(initial={%s: ...}): %s is not a quantity that the "
                                 "model evolves" % (key, key))
            if datum.model is not self.model:
                raise ValueError("pops.Program(initial={%s: %s}): the datum belongs to another "
                                 "model" % (key, datum))
        for quantity, state in states.items():
            datum = program.initial_of(quantity)
            if datum is None:
                raise AmbiguousElaboration(
                    "the evolved quantity %s has no initial value" % quantity,
                    write='pops.Program(initial={%s: d}), d = model.data("d", %r)'
                          % (quantity, quantity.space), site=program.site)
            case6.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                                projection=ConservativeCellAverage()))
            self.rows.append(Row(
                "initial={%s: %s}" % (quantity, datum), "program",
                "case.initials.add(InitialCondition(state=block[%s], value=BindArray(), "
                "projection=ConservativeCellAverage()))" % quantity.name,
                [Choice("value", "the datum %s, supplied at bind" % datum, "stated"),
                 Choice("projection", "ConservativeCellAverage()", "derived")]))

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
                                        periodic=PeriodicAxes(periodic)))
        self.rows.append(Row(
            "layout={%r: %r}" % (support, uniform), "case",
            "Uniform(CartesianGrid(frame=frame, cells=%r, periodic=PeriodicAxes(%s)))"
            % (uniform.cells, "(%s)" % "".join("%s," % axis for axis in support._names
                                              if support.periodic(axis))),
            [Choice("cells", uniform.cells, "stated")]))
        return layout

    def boundaries(self, support: Box) -> None:
        pieces = support.piece_names()
        if pieces and self.model.terms:
            raise MissingBoundaryRealization(
                "the fluxes of %s cross the boundary pieces %s of %r, which no boundary relation "
                "closes" % (self.model.terms[0].quantity, ", ".join(pieces), support),
                write="pops.Periodic(lower, upper) for an axis whose ends are identified; the "
                      "boundary relations model.on(...) and the boundary policy are elaborated "
                      "by a later lot of N15 with G04", site=support.site)

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
        self.boundaries(support)
        frame = self.frame(support)
        b6 = _Model(model.name, frame=frame)
        self.rows.insert(0, Row("pops.Model(%r)" % model.name, "model",
                                "model = pops.Model(%r, frame=frame)" % model.name))
        states = self.declarations(b6)
        terms = self.terms(b6, frame, states)
        rates = {balance: self.rate(b6, balance, terms, states) for balance in model.balances}
        plan, selections = self.plan(rates, terms, states)
        case6 = _Case(model.name)
        block = case6.block(model.name, b6)
        case6.numerics(plan, block=block)
        self.initial(case6, block, states)
        advance = self.advance(case6, block, states, selections)
        layout = self.layout(support, frame)
        return Elaborated(case=case6, layout=layout, block=block, model=b6, plan=plan,
                          states=states, names=self.names, advance=advance,
                          rows=tuple(self.rows))


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
