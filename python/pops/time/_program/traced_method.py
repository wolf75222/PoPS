"""Traced methods of the temporal program (spec 0.3, sections 5.2 to 5.4): ``method``, ``Candidate``.

A method is a Python function ``f(B, p, *components) -> Candidate``. ``method(f)`` records it,
calling the result with its components specializes it, and ``trace`` runs the body once against
the internal builder (``pops.time.Program``), which elaborates every stage into its SSA. ``B`` is
the accepted continuation (``B.state``, ``B.time``) and ``p`` the proposal (``p.dt``).

T33. The body reads only ``B``, ``p``, its components and immutable captures. A value of a
Program, a builder coefficient (the ambient step size among them), a clock or a time point inside
a traced expression is refused as ``ClosedBodyViolation``, and so is a component that publishes a
state or creates a history. A Python truth test of a traced value raises
``SymbolicTruthValueError`` (E1).

T4. Each rate evaluation states its time, ``flow(U, B.time + c * p.dt)``, and opens its own stage:
two evaluations at one time are two stages. A combination is materialized at the stage of the
first evaluation that reads it, and the expressions written after that evaluation read the stage
value.

T34. Beside the builder SSA, every traced field keeps its expansion over the accepted state and
the rate evaluations, with exact rational coefficients that are polynomials in ``p.dt``. The
increment of the candidate is derived from it: the candidate must be the accepted state plus a
sum of durations ``c * p.dt`` times rates, or the method is refused (``IncrementNotDerivable``).

b6's ``GraphFunction`` admits neither rate evaluations nor states nor the step size (the ``_PURE``
operations of ``python/pops/graph_functions.py``), so the closed body is elaborated directly into
the builder, and ``MethodTrace`` records its closed form.
"""
from __future__ import annotations

import inspect
from decimal import Decimal
from fractions import Fraction
from typing import Any, Callable

from pops._ir.symbolic import SymbolicTruthValueError
from pops.time.points import Clock, StagePoint, TimePoint
from pops.time.value_metadata import _exact_number
from pops.time.values import ProgramValue, _Affine, _Coeff


class ClosedBodyViolation(TypeError):
    """A method body read or published something outside ``B``, ``p`` and its components (T33)."""

    code = "closed_body_violation"


class IncrementNotDerivable(ValueError):
    """The candidate is not the accepted state plus durations times rates, and no increment is written (T34)."""

    code = "increment_not_derivable"


class NotYetElaborated(NotImplementedError):
    """A construct of section 5 that this elaborator does not lower yet; the message names the issue that does."""

    code = "not_yet_elaborated"


# --- exact expansions -------------------------------------------------------------------------
# A polynomial in p.dt is a tuple of (power, Fraction) pairs in increasing powers, without zeros.

def _fraction(value: Any) -> Fraction:
    """The exact rational value of an authored coefficient (a binary64 literal included)."""
    number = _exact_number(value)
    if isinstance(number, bool) or not isinstance(number, (int, float, Fraction, Decimal)):
        raise TypeError("a method coefficient must be a real number, got %r" % (value,))
    return Fraction(number)


def _poly(powers: dict) -> tuple:
    return tuple(sorted((int(power), coefficient) for power, coefficient in powers.items()
                        if coefficient != 0))


def _poly_of(coefficient: _Coeff) -> tuple:
    return _poly({power: _fraction(value) for power, value in coefficient.powers.items()})


def _poly_add(left: tuple, right: tuple) -> tuple:
    out = dict(left)
    for power, coefficient in right:
        out[power] = out.get(power, Fraction(0)) + coefficient
    return _poly(out)


def _poly_mul(left: tuple, right: tuple) -> tuple:
    out: dict = {}
    for p1, c1 in left:
        for p2, c2 in right:
            out[p1 + p2] = out.get(p1 + p2, Fraction(0)) + c1 * c2
    return _poly(out)


_ONE = ((0, Fraction(1)),)


def _poly_text(poly: tuple) -> str:
    if not poly:
        return "0"
    return " + ".join(str(c) if p == 0 else "%s*dt" % c if p == 1 else "%s*dt^%d" % (c, p)
                      for p, c in poly)


def _expansion_add(left: tuple, right: tuple) -> tuple:
    """Sum two expansions ((atom, poly), ...), in order of first appearance, dropping zeros."""
    order, acc = [], {}
    for atom, poly in left + right:
        if atom not in acc:
            order.append(atom)
            acc[atom] = poly
        else:
            acc[atom] = _poly_add(acc[atom], poly)
    return tuple((atom, acc[atom]) for atom in order if acc[atom])


def _expansion_scale(expansion: tuple, poly: tuple) -> tuple:
    return tuple((atom, product) for atom, product in
                 ((atom, _poly_mul(coefficient, poly)) for atom, coefficient in expansion)
                 if product)


class _Atom:
    """An opaque summand of an expansion: the accepted state, or one rate evaluation."""

    __slots__ = ("label", "evaluation")

    def __init__(self, label: str, evaluation: int | None = None) -> None:
        self.label = label
        self.evaluation = evaluation

    def __repr__(self) -> str:
        return self.label


# --- the trace ----------------------------------------------------------------------------------

class _Trace:
    """The state of one tracing pass: the builder, the stages and the evaluations so far."""

    def __init__(self, program: Any, temporal: Any, name: str, proposal: Any) -> None:
        self.program = program
        self.temporal = temporal
        self.name = name
        self.proposal = proposal
        self.active = True
        self.stages: list = []        # (ordinal, c, name of the stage value materialized or None)
        self.evaluations: list = []   # (ordinal, component, stage ordinal)
        self.state_atom = _Atom("B.state")

    def require_active(self, what: str) -> None:
        if not self.active:
            raise ClosedBodyViolation(
                "%s was traced by a method whose trace has ended; a method body reads only B, p "
                "and its components (T33)" % what)


def _effects(program: Any) -> tuple:
    """What a closed body may not change: publications, histories, step control (T33).

    Objects are compared by identity: the equality of a Program value builds an equation.
    """
    return (len(program._commits), len(program._post_sync_commits), len(program._histories),
            len(program._integral_states), len(program._integral_transfers),
            len(program._acceptance_guards), id(program._step_strategy), id(program._dt_bound))


def _foreign(value: Any) -> str | None:
    """Name a symbolic object that a method body cannot read, or None for plain data."""
    from pops.time.expressions import ProgramExpression
    from pops.time.handles import HistoryHandle, StageHandle, StateEndpointHandle, TimeState

    if isinstance(value, _Coeff):
        if any(power != 0 for power in value.powers):
            return "the step size of a Program (use p.dt)"
        return "a coefficient of a Program"
    if isinstance(value, (ProgramValue, _Affine, ProgramExpression)):
        return "a value of a Program"
    if isinstance(value, (TimeState, StageHandle, StateEndpointHandle, HistoryHandle)):
        return "a temporal handle of a Program"
    if isinstance(value, (Clock, TimePoint, StagePoint)):
        return "a clock or a time point (use B.time)"
    return None


def _refuse_foreign(value: Any, where: str) -> None:
    what = _foreign(value)
    if what is not None:
        raise ClosedBodyViolation(
            "%s reads %s; a method body reads only B, p, its components and immutable "
            "captures (T33)" % (where, what))


class _Traced:
    """Common behaviour of traced values: no truth value, no hashing, no NumPy coercion."""

    __slots__ = ("_trace",)
    __array_ufunc__ = None
    __hash__ = None  # type: ignore[assignment]

    def __bool__(self) -> bool:
        raise SymbolicTruthValueError(
            self, suggestion="a method decides on traced values with cond(...) or match(...) "
                             "(E1); Python if/and/or cannot evaluate them")

    def __eq__(self, other: Any) -> Any:
        raise NotYetElaborated(
            "a relation between traced values (section 5.5) is elaborated by a later lot of N8")

    def __ne__(self, other: Any) -> Any:
        return self.__eq__(other)

    def _same_trace(self, other: Any, where: str) -> None:
        self._trace.require_active("a traced value")
        if isinstance(other, _Traced):
            other._trace.require_active("a traced value")
            if other._trace is not self._trace:
                raise ClosedBodyViolation(
                    "%s mixes values of two method traces; a method body reads only B, p and its "
                    "components (T33)" % where)


class TracedScalar(_Traced):
    """An exact polynomial in ``p.dt``: the proposed duration and its multiples."""

    __slots__ = ("_coeff",)

    def __init__(self, trace: _Trace, coeff: _Coeff) -> None:
        self._trace = trace
        self._coeff = coeff

    def _other(self, other: Any, where: str) -> _Coeff:
        self._same_trace(other, where)
        if isinstance(other, TracedScalar):
            return other._coeff
        if isinstance(other, _Traced):
            raise TypeError("%s needs a duration or a number, got %s" % (where, type(other).__name__))
        _refuse_foreign(other, where)
        _fraction(other)
        return _Coeff({0: other})

    def __add__(self, other: Any) -> Any:
        if isinstance(other, TracedTime):
            return other.__add__(self)
        return TracedScalar(self._trace, self._coeff + self._other(other, "a duration sum"))

    def __radd__(self, other: Any) -> Any:
        return TracedScalar(self._trace, self._other(other, "a duration sum") + self._coeff)

    def __sub__(self, other: Any) -> Any:
        return TracedScalar(self._trace, self._coeff - self._other(other, "a duration difference"))

    def __rsub__(self, other: Any) -> Any:
        return TracedScalar(self._trace, self._other(other, "a duration difference") - self._coeff)

    def __neg__(self) -> Any:
        self._trace.require_active("a traced value")
        return TracedScalar(self._trace, -self._coeff)

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, TracedField):
            return other.__rmul__(self)
        return TracedScalar(self._trace, self._coeff * self._other(other, "a duration product"))

    def __rmul__(self, other: Any) -> Any:
        return TracedScalar(self._trace, self._other(other, "a duration product") * self._coeff)

    def __truediv__(self, other: Any) -> Any:
        self._trace.require_active("a traced value")
        if isinstance(other, _Traced):
            raise TypeError("a duration is divided by a number only (T9)")
        _refuse_foreign(other, "a duration quotient")
        _fraction(other)
        return TracedScalar(self._trace, self._coeff / other)

    def __repr__(self) -> str:
        return "TracedScalar(%s)" % _poly_text(_poly_of(self._coeff))


class TracedTime(_Traced):
    """``B.time + c * p.dt``: a time of the attempt, with ``c`` exact (T4)."""

    __slots__ = ("c",)

    def __init__(self, trace: _Trace, c: Any) -> None:
        self._trace = trace
        self.c = c

    def _offset(self, other: Any, where: str) -> Any:
        self._same_trace(other, where)
        if isinstance(other, TracedScalar):
            powers = dict(other._coeff.powers)
            if set(powers) <= {1}:
                return powers.get(1, 0)
            raise TypeError("%s: a time of the attempt is B.time plus a multiple of p.dt (T4), "
                            "got an offset %s" % (where, _poly_text(_poly_of(other._coeff))))
        _refuse_foreign(other, where)
        raise TypeError("%s: a time of the attempt is B.time plus a multiple of p.dt (T4)" % where)

    def __add__(self, other: Any) -> Any:
        return TracedTime(self._trace, (_Coeff({1: self.c}) + _Coeff({1: self._offset(
            other, "a time sum")})).powers.get(1, 0))

    __radd__ = __add__

    def __sub__(self, other: Any) -> Any:
        return TracedTime(self._trace, (_Coeff({1: self.c}) - _Coeff({1: self._offset(
            other, "a time difference")})).powers.get(1, 0))

    def __repr__(self) -> str:
        return "TracedTime(B.time + %s*dt)" % (self.c,)


class TracedField(_Traced):
    """A traced field: its builder spelling and its exact expansion (T9, T34).

    ``_expr`` is the builder spelling, a ``ProgramValue`` or an affine combination of them.
    ``_value`` and ``_c`` are the stage value that materializes the field and the time of its
    stage, set by the first rate evaluation that reads the field; ``B.state`` is born
    materialized at ``B.time``.
    """

    __slots__ = ("_expr", "_expansion", "_value", "_c")

    def __init__(self, trace: _Trace, expr: Any, expansion: tuple,
                 value: Any = None, c: Any = None) -> None:
        self._trace = trace
        self._expr = expr
        self._expansion = expansion
        self._value = value
        self._c = c

    def _builder(self) -> Any:
        return self._expr if self._value is None else self._value

    def _field(self, other: Any, where: str) -> TracedField:
        self._same_trace(other, where)
        if isinstance(other, TracedField):
            return other
        _refuse_foreign(other, where)
        raise TypeError("%s combines fields only; a number or a time is not a field (T9)" % where)

    def _scalar(self, other: Any, where: str) -> tuple:
        """(builder coefficient, exact polynomial) of a factor of a field."""
        self._same_trace(other, where)
        if isinstance(other, TracedScalar):
            return other._coeff, _poly_of(other._coeff)
        if isinstance(other, TracedField):
            raise TypeError("%s: a stage formula is affine in fields (T9); a product of two "
                            "fields is not" % where)
        if isinstance(other, _Traced):
            raise TypeError("%s: a field is scaled by a number or a duration (T9)" % where)
        _refuse_foreign(other, where)
        return other, ((0, _fraction(other)),)

    def __add__(self, other: Any) -> Any:
        o = self._field(other, "a field sum")
        return TracedField(self._trace, self._builder() + o._builder(),
                           _expansion_add(self._expansion, o._expansion))

    def __radd__(self, other: Any) -> Any:
        o = self._field(other, "a field sum")
        return TracedField(self._trace, o._builder() + self._builder(),
                           _expansion_add(o._expansion, self._expansion))

    def __sub__(self, other: Any) -> Any:
        o = self._field(other, "a field difference")
        return TracedField(self._trace, self._builder() - o._builder(),
                           _expansion_add(self._expansion, _expansion_scale(
                               o._expansion, ((0, Fraction(-1)),))))

    def __rsub__(self, other: Any) -> Any:
        o = self._field(other, "a field difference")
        return o.__sub__(self)

    def __neg__(self) -> Any:
        self._trace.require_active("a traced value")
        return TracedField(self._trace, -self._builder(),
                           _expansion_scale(self._expansion, ((0, Fraction(-1)),)))

    def __mul__(self, other: Any) -> Any:
        factor, poly = self._scalar(other, "a field product")
        return TracedField(self._trace, self._builder() * factor,
                           _expansion_scale(self._expansion, poly))

    def __rmul__(self, other: Any) -> Any:
        factor, poly = self._scalar(other, "a field product")
        return TracedField(self._trace, factor * self._builder(),
                           _expansion_scale(self._expansion, poly))

    def __truediv__(self, other: Any) -> Any:
        self._trace.require_active("a traced value")
        if isinstance(other, _Traced):
            raise TypeError("a field is divided by a number only (T9)")
        _refuse_foreign(other, "a field quotient")
        poly = ((0, 1 / _fraction(other)),)
        return TracedField(self._trace, self._builder() / other,
                           _expansion_scale(self._expansion, poly))

    def __repr__(self) -> str:
        return "TracedField(%s)" % " + ".join(
            "(%s)*%r" % (_poly_text(poly), atom) for atom, poly in self._expansion)


class _Continuation:
    """``B`` inside a method body: the accepted state and the time of the attempt (T33)."""

    __slots__ = ("_trace", "_state")

    def __init__(self, trace: _Trace, state: TracedField) -> None:
        self._trace = trace
        self._state = state

    @property
    def state(self) -> TracedField:
        self._trace.require_active("the accepted continuation")
        return self._state

    @property
    def time(self) -> TracedTime:
        self._trace.require_active("the accepted continuation")
        return TracedTime(self._trace, 0)

    @property
    def memory(self) -> Any:
        raise NotYetElaborated("method memory (section 9, T27) is elaborated by a later lot of N8")


class _ProposalView:
    """``p`` inside a method body: the symbolic duration, never its numerical value (T33)."""

    __slots__ = ("_trace",)

    def __init__(self, trace: _Trace) -> None:
        self._trace = trace

    @property
    def dt(self) -> TracedScalar:
        self._trace.require_active("the proposal")
        return TracedScalar(self._trace, _Coeff({1: 1}))

    @property
    def request(self) -> Any:
        return self._trace.proposal.request

    @property
    def inputs(self) -> Any:
        raise NotYetElaborated("the inputs of a child advance (T35) are elaborated by a later "
                               "lot of N8")


class _Component:
    """A component inside a method body: ``flow(U, t)`` evaluates its rate at a new stage (T4)."""

    __slots__ = ("_trace", "_name", "_evaluate")

    def __init__(self, trace: _Trace, name: str, evaluate: Callable) -> None:
        self._trace = trace
        self._name = name
        self._evaluate = evaluate

    def __call__(self, state: Any = None, time: Any = None, **options: Any) -> TracedField:
        trace = self._trace
        trace.require_active("the component %r" % self._name)
        if options:
            raise NotYetElaborated(
                "%s(..., %s): term selection and field requests (T5, T6) are elaborated by a "
                "later lot of N8" % (self._name, ", ".join(sorted(options))))
        if not isinstance(state, TracedField) or not isinstance(time, TracedTime):
            _refuse_foreign(state, "%s(U, t)" % self._name)
            _refuse_foreign(time, "%s(U, t)" % self._name)
            raise TypeError("%s(U, t) evaluates a traced state U at a time t = B.time + c*p.dt "
                            "of the attempt (T4)" % self._name)
        state._same_trace(time, "%s(U, t)" % self._name)
        if state._trace is not trace:
            raise ClosedBodyViolation(
                "%s(U, t) reads a value of another method trace (T33)" % self._name)
        program = trace.program
        ordinal = len(trace.stages)
        point = program.stage("%s_stage%d" % (trace.name, ordinal), c=time.c)
        value, materialized = self._value_at(state, point, ordinal, time.c)
        trace.stages.append((ordinal, time.c, materialized))
        before = _effects(program)
        rate = self._evaluate(value)
        if _effects(program) != before:
            raise ClosedBodyViolation(
                "the component %r published a state, created a history or changed the step "
                "control; a component only evaluates a rate (T33)" % self._name)
        _require_rate(program, rate, self._name)
        evaluation = len(trace.evaluations)
        trace.evaluations.append((evaluation, self._name, ordinal))
        atom = _Atom("%s#%d" % (self._name, evaluation), evaluation)
        return TracedField(trace, rate, ((atom, _ONE),))

    def _value_at(self, state: TracedField, point: StagePoint, ordinal: int, c: Any) -> tuple:
        """The builder value that the rate reads at ``point``, materialized there if needed."""
        program = self._trace.program
        name = "%s_state%d" % (self._trace.name, ordinal)
        if state._value is None:
            value = program.value(name, state._expr, at=point)
            _require_state(value, "%s(U, t) reads a state" % self._name)
            state._value, state._c = value, c
            return value, name
        if _fraction(state._c) == _fraction(c):
            return state._value, None
        # The same state, read at another time of the attempt: a copy at this stage.
        return program.value(name, state._value, at=point), name


def _require_state(value: Any, where: str) -> None:
    if not (isinstance(value, ProgramValue) and value.vtype == "state"):
        raise TypeError("%s; this combination holds rates without the accepted state (T9)"
                        % where)


def _require_rate(program: Any, rate: Any, component: str) -> None:
    values = ([value for value, _ in rate.terms] if isinstance(rate, _Affine)
              else [rate] if isinstance(rate, ProgramValue) else None)
    if not values or any(not (isinstance(v, ProgramValue) and v.is_field()) for v in values):
        raise TypeError("the component %r must return the rate of its state, a field value of "
                        "the Program, got %r" % (component, rate))
    if any(value.prog is not program for value in values):
        raise ClosedBodyViolation(
            "the component %r returned a value of another Program (T33)" % component)


# --- methods ------------------------------------------------------------------------------------

class Candidate:
    """The candidate continuation that a method returns (T34): its state and achieved time.

    ``increment=None`` derives the increment from the candidate. ``Product()`` is not part of
    this lot, so ``None`` stands for the empty records of ``observations`` and ``evidence``.
    """

    __slots__ = ("state", "time", "increment", "memory", "observations", "evidence",
                 "interval_output")

    def __init__(self, state: Any, time: Any, increment: Any = None, memory: Any = None,
                 observations: Any = None, evidence: Any = None,
                 interval_output: Any = None) -> None:
        if not isinstance(state, TracedField):
            _refuse_foreign(state, "Candidate(state=...)")
            raise TypeError("Candidate(state=...) takes the traced state of a method body")
        if not isinstance(time, TracedTime):
            _refuse_foreign(time, "Candidate(time=...)")
            raise TypeError("Candidate(time=...) takes a time of the attempt, B.time + c*p.dt")
        state._same_trace(time, "Candidate(state, time)")
        for label, value, owner in (
                ("increment", increment, "the increment records of section 8"),
                ("memory", memory, "method memory (section 9)"),
                ("observations", observations, "candidate observations"),
                ("evidence", evidence, "candidate evidence"),
                ("interval_output", interval_output, "interval outputs")):
            if value is not None:
                raise NotYetElaborated("Candidate(%s=...): %s are elaborated by a later lot of "
                                       "N8" % (label, owner))
        self.state = state
        self.time = time
        self.increment = increment
        self.memory = memory
        self.observations = observations
        self.evidence = evidence
        self.interval_output = interval_output


class MethodTrace:
    """The closed form of one traced method: stages, evaluations, derived increment, candidate.

    ``increment`` holds ``(evaluation, duration)`` pairs, the duration as the exact factor of
    ``p.dt``: the candidate is ``B.state + sum(duration * p.dt * rate)`` (T34).
    """

    __slots__ = ("name", "components", "stages", "evaluations", "increment", "candidate")

    def __init__(self, name: str, components: tuple, stages: tuple, evaluations: tuple,
                 increment: tuple, candidate: Any) -> None:
        self.name = name
        self.components = components
        self.stages = stages
        self.evaluations = evaluations
        self.increment = increment
        self.candidate = candidate

    def describe(self) -> dict:
        """Plain data for inspection and tests; times and durations as exact fractions of ``p.dt``."""
        return {
            "method": self.name,
            "components": list(self.components),
            "stages": [{"stage": ordinal, "c": str(Fraction(_fraction(c))),
                        "materializes": name} for ordinal, c, name in self.stages],
            "evaluations": [{"evaluation": index, "component": component, "stage": stage}
                            for index, component, stage in self.evaluations],
            "increment": [{"evaluation": index, "duration": "%s*dt" % duration}
                          for index, duration in self.increment],
        }


class SpecializedMethod:
    """A method with its components bound: what ``Advance`` takes (5.2)."""

    __slots__ = ("method", "components")

    def __init__(self, method: Method, components: tuple) -> None:
        self.method = method
        self.components = components

    @property
    def name(self) -> str:
        return self.method.name

    def trace(self, program: Any, temporal: Any, proposal: Any) -> MethodTrace:
        """Run the body once against ``program`` and materialize the candidate at the endpoint."""
        trace = _Trace(program, temporal, self.name, proposal)
        state = TracedField(trace, temporal.n, ((trace.state_atom, _ONE),),
                            value=temporal.n, c=0)
        views = [_Component(trace, name, evaluate)
                 for name, evaluate in zip(self.method.component_names, self.components)]
        before = _effects(program)
        try:
            candidate = self.method.body(_Continuation(trace, state), _ProposalView(trace), *views)
        finally:
            trace.active = False
        if _effects(program) != before:
            raise ClosedBodyViolation(
                "the body of %r published a state, created a history or changed the step "
                "control outside its Candidate (T33)" % self.name)
        if not isinstance(candidate, Candidate):
            raise TypeError("the method %r must return a Candidate, got %r" % (self.name, candidate))
        if candidate.state._trace is not trace:
            raise ClosedBodyViolation("the method %r returned a value of another trace (T33)"
                                      % self.name)
        if _fraction(candidate.time.c) != 1:
            raise NotYetElaborated(
                "the method %r reaches B.time + %s*p.dt: an achieved time other than "
                "B.time + p.dt (T26, T36) is elaborated in Phase 2 of N8"
                % (self.name, Fraction(_fraction(candidate.time.c))))
        increment = _derive_increment(self.name, trace, candidate.state._expansion)
        value = program.value("%s_candidate" % self.name, candidate.state._builder(),
                              at=temporal.next.point)
        _require_state(value, "Candidate(state=...) is a state")
        return MethodTrace(self.name, tuple(self.method.component_names), tuple(trace.stages),
                           tuple(trace.evaluations), increment, value)


def _derive_increment(name: str, trace: _Trace, expansion: tuple) -> tuple:
    """T34, X1: the candidate is B.state + sum(c_i * p.dt * k_i), or the increment is not derivable."""
    state = dict(expansion).get(trace.state_atom, ())
    if state != _ONE:
        raise IncrementNotDerivable(
            "the candidate of %r holds the accepted state with the coefficient %s, not 1: it is "
            "not B.state plus an increment, and no increment is written (T34). Write the "
            "coefficients as exact rationals (fractions.Fraction) when their binary64 values do "
            "not sum to 1" % (name, _poly_text(state)))
    increment = []
    for atom, poly in expansion:
        if atom is trace.state_atom:
            continue
        if len(poly) != 1 or poly[0][0] != 1:
            raise IncrementNotDerivable(
                "the candidate of %r holds the rate %r with the coefficient %s, which is not a "
                "duration c*p.dt: the increment is not a sum of durations times rates, and no "
                "increment is written (T34)" % (name, atom, _poly_text(poly)))
        increment.append((atom.evaluation, poly[0][1]))
    return tuple(increment)


class Method:
    """A recorded method body ``f(B, p, *components) -> Candidate``; calling it specializes it."""

    __slots__ = ("body", "name", "component_names", "memory", "policies")

    def __init__(self, body: Callable, *, memory: Any = None, policies: Any = None) -> None:
        if not callable(body):
            raise TypeError("method(f): f must be a function f(B, p, *components)")
        if memory is not None:
            raise NotYetElaborated("method(memory=...): method memory (section 9, T27) is "
                                   "elaborated by a later lot of N8")
        if policies:
            raise NotYetElaborated("method(policies=...): memory policies (CN5, CN7) are "
                                   "elaborated by a later lot of N8")
        parameters = list(inspect.signature(body).parameters.values())
        if len(parameters) < 2 or any(
                parameter.kind not in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
                or parameter.default is not parameter.empty for parameter in parameters):
            raise TypeError("method(f): f takes B, p and its components as plain positional "
                            "parameters, f(B, p, *components)")
        name = getattr(body, "__name__", "")
        self.body = body
        self.name = name if isinstance(name, str) and name.isidentifier() else "method"
        self.component_names = tuple(parameter.name for parameter in parameters[2:])
        self.memory = memory
        self.policies = policies

    def __call__(self, *components: Any, **named: Any) -> SpecializedMethod:
        names = self.component_names
        if len(components) > len(names):
            raise TypeError("%s takes the components %s" % (self.name, ", ".join(names)))
        bound = dict(zip(names, components))
        for key, value in named.items():
            if key not in names or key in bound:
                raise TypeError("%s: unknown or repeated component %r; its components are %s"
                                % (self.name, key, ", ".join(names)))
            bound[key] = value
        missing = [key for key in names if key not in bound]
        if missing:
            raise TypeError("%s: missing components %s" % (self.name, ", ".join(missing)))
        for key in names:
            if not callable(bound[key]):
                raise TypeError("%s: the component %r must evaluate a rate, component(U) -> rate"
                                % (self.name, key))
        return SpecializedMethod(self, tuple(bound[key] for key in names))


def method(f: Callable | None = None, *, memory: Any = None, policies: Any = None) -> Any:
    """Record ``f(B, p, *components) -> Candidate`` as a method (5.2); usable as ``@method``."""
    if f is None:
        return lambda body: Method(body, memory=memory, policies=policies)
    return Method(f, memory=memory, policies=policies)
