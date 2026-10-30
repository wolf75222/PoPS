"""Traced methods of the temporal program (spec 0.3, sections 5.2 to 5.4): ``method``, ``Candidate``.

A method is a Python function ``f(B, p, *components) -> Candidate``. ``method(f)`` records it,
calling the result with its components specializes it, and ``trace`` runs the body once against
the internal builder (``pops.time.Program``), which elaborates every stage into its SSA. ``B`` is
the accepted continuation (``B.state``, ``B.time``) and ``p`` the proposal (``p.dt``). This lot
traces a method at each elaboration, into the Program that elaborates it; the closed
``GraphFunction`` of section 5.2, traced once per specialization, comes with a later lot of N8.

T33. The body reads only ``B``, ``p``, its components and immutable captures. A value of a
Program, a builder coefficient (the ambient step size among them), a clock or a time point inside
a traced expression is refused as ``ClosedBodyViolation``. So is a component that reads the step
size of its Program or a value it was not given, that evaluates a value at another time than the
state it is given, that returns a rate this evaluation did not compute, or that changes a
publication, a history, an integral, an acceptance guard or the step control of the Program:
those tables are compared, contents included, around each evaluation and around the body. A Python truth test, ``float``, ``int`` or ``numpy.asarray`` of a
traced value raises ``SymbolicTruthValueError`` (E1). Comparisons build predicates (SU4) and come
with a later lot, and Python's ``max`` and ``min`` with them.

T4. Each rate evaluation states its time, ``flow(U, B.time + c * p.dt)``, and opens its own stage:
two evaluations at one time are two stages. A combination is materialized at the stage of the
first evaluation that reads it, and the expressions written after that evaluation read the stage
value.

T34, SU10. Beside the builder SSA, every traced field keeps its expansion over the accepted state
and the rate evaluations, with exact rational coefficients that are polynomials in ``p.dt``. A
duration is such a polynomial that vanishes when ``p.dt`` is zero; the other durations of SU10,
``(1 - exp(-nu*p.dt))/nu`` among them, come with the expressions of a later lot. The state that a
rate reads and the candidate are states: the accepted state with the weight one, exactly, or within
1e-12 when a binary64 literal enters the weights (the trace lists those), plus durations times
rates. A rate without a duration, or states whose weights sum to anything else, is refused as
``IncrementTypeMismatch``; a candidate that weighs the accepted state by a function of the step, a
closed form that no rate built, as ``IncrementNotDerivable``. The increment of the candidate is
the sum of its durations times rates.

b6's ``GraphFunction`` admits neither rate evaluations nor states nor the step size (the ``_PURE``
operations of ``python/pops/graph_functions.py``), so the closed body is elaborated directly into
the builder, and ``MethodTrace`` records its closed form.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from decimal import Decimal
from fractions import Fraction
from typing import Any, NoReturn

from pops._ir.symbolic import SymbolicTruthValueError
from pops.time.points import Clock, StagePoint, TimePoint
from pops.time.value_metadata import CoeffPolynomial, _exact_number
from pops.time.values import ProgramValue, _Affine, _Coeff


class ClosedBodyViolation(TypeError):
    """A method body read or published something outside ``B``, ``p`` and its components (T33)."""

    code = "closed_body_violation"


class IncrementTypeMismatch(TypeError):
    """A combination breaks the step algebra (SU10): a rate without a duration, or states whose weights do not sum to one."""

    code = "increment_type_mismatch"


class IncrementNotDerivable(ValueError):
    """The candidate is a closed form that no rate built, and no increment derives from it (T34)."""

    code = "increment_not_derivable"


class NotYetElaborated(NotImplementedError):
    """A construct of section 5 that this elaborator does not lower yet; the message names the issue that does."""

    code = "not_yet_elaborated"


# --- exact expansions -------------------------------------------------------------------------
# A polynomial in p.dt is a tuple of (power, Fraction) pairs in increasing powers, without zeros.

# SU10: binary64 weights sum to one within this bound; exact weights sum to one exactly.
_WEIGHT_TOLERANCE = Fraction(1, 10**12)


def _coefficient(value: Any) -> tuple:
    """(exact rational value, whether it is a binary64 literal) of an authored coefficient."""
    number = _exact_number(value)
    if isinstance(number, bool) or not isinstance(number, (int, float, Fraction, Decimal)):
        raise TypeError("a method coefficient must be a real number, got %r" % (value,))
    return Fraction(number), isinstance(number, float)


def _fraction(value: Any) -> Fraction:
    """The exact rational value of an authored coefficient (a binary64 literal included)."""
    return _coefficient(value)[0]


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


def _poly_text(poly: Any) -> str:
    terms = tuple(poly.items()) if isinstance(poly, Mapping) else poly
    if not terms:
        return "0"
    return " + ".join(str(c) if p == 0 else "%s*dt" % c if p == 1 else "%s*dt^%d" % (c, p)
                      for p, c in terms)


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
        self.tolerated: list = []     # (where, weight sum) of binary64 weights within 1e-12
        self.state_atom = _Atom("B.state")

    def require_active(self, what: str) -> None:
        if not self.active:
            raise ClosedBodyViolation(
                "%s was traced by a method whose trace has ended; a method body reads only B, p "
                "and its components (T33)" % what)


# The tables of a Program that a closed body changes only through its Candidate (T33):
# publications, histories, integrals, acceptance guards and the step control.
_GUARDED_TABLES = (
    "_commits", "_post_sync_commits", "_post_sync_recording", "_geometry_states", "_state_spaces",
    "_time_states", "_time_endpoint_handles", "_histories", "_history_spaces", "_history_blocks",
    "_history_state_refs", "_histories_ncomp", "_history_persistence", "_time_history_handles",
    "_time_history_values", "_time_history_configs", "_time_history_stores",
    "_history_contracts", "_global_field_history_issuance", "_integral_states",
    "_integral_units", "_integral_transfers", "_acceptance_guards", "_transaction_stores",
    "_step_strategy", "_dt_bound", "_cadence", "_cell_local_time",
)
_ABSENT = object()


def _snapshot(value: Any) -> tuple:
    """The contents of a table: data by value, every other object by identity, kept alive.

    Objects are never compared with ``==``: the equality of a Program value builds an equation.
    """
    if value is None or isinstance(value, (bool, int, float, str, Fraction, Decimal)):
        return ("data", type(value), value)
    if isinstance(value, Mapping):
        return ("map", tuple((_snapshot(key), _snapshot(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_snapshot(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(value))
    return ("object", value)


def _same(left: tuple, right: tuple) -> bool:
    kind = left[0]
    if kind != right[0]:
        return False
    if kind == "data":
        return left[1] is right[1] and (left[2] is right[2] or left[2] == right[2])
    if kind == "object":
        return left[1] is right[1]
    if kind == "set":
        return {id(item) for item in left[1]} == {id(item) for item in right[1]}
    if len(left[1]) != len(right[1]):
        return False
    if kind == "map":
        return all(_same(key, other_key) and _same(item, other_item)
                   for (key, item), (other_key, other_item) in zip(left[1], right[1],
                                                                   strict=True))
    return all(_same(item, other) for item, other in zip(left[1], right[1], strict=True))


def _guarded_tables(program: Any) -> tuple:
    return tuple((name, _snapshot(getattr(program, name, _ABSENT))) for name in _GUARDED_TABLES)


def _changed_tables(before: tuple, after: tuple) -> list:
    """The readable names of the guarded tables that differ between two snapshots."""
    return [name.strip("_").replace("_", " ")
            for (name, old), (_, new) in zip(before, after, strict=True) if not _same(old, new)]


def _reads_step_size(node: Any) -> bool:
    """Whether a coefficient or IR metadata holds a power of dt (the ``GraphFunction`` rule of b6)."""
    if isinstance(node, _Coeff):
        return any(power != 0 for power in node.powers)
    if isinstance(node, CoeffPolynomial):
        return any(power != 0 for power in node)
    if isinstance(node, Mapping):
        return any(_reads_step_size(child) for child in node.values())
    if isinstance(node, (list, tuple, set, frozenset)):
        return any(_reads_step_size(child) for child in node)
    return False


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
    """Common behaviour of traced values: no truth value, no number, no hashing, no NumPy coercion."""

    __slots__ = ("_trace",)
    __array_ufunc__ = None
    __hash__ = None  # type: ignore[assignment]

    def __bool__(self) -> bool:
        raise SymbolicTruthValueError(
            self, suggestion="a method decides on traced values with cond(...) or match(...) "
                             "(E1); Python if/and/or cannot evaluate them")

    def __float__(self) -> NoReturn:
        raise SymbolicTruthValueError(
            self, suggestion="a traced value has no number while its method is traced (E1); "
                             "float(...), int(...) and numpy.asarray(...) cannot evaluate it")

    __int__ = __index__ = __complex__ = __float__

    def __array__(self, dtype: Any = None, copy: Any = None) -> NoReturn:
        self.__float__()

    def __eq__(self, other: Any) -> Any:
        raise NotYetElaborated(
            "a relation between traced values (section 5.5) is elaborated by a later lot of N8")

    def __ne__(self, other: Any) -> Any:
        return self.__eq__(other)

    def __lt__(self, other: Any) -> NoReturn:
        raise NotYetElaborated(
            "a comparison of traced values builds a predicate (SU4), which a later lot of N8 "
            "elaborates, with Python's max(...) and min(...) of traced values")

    __le__ = __gt__ = __ge__ = __lt__

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

    __slots__ = ("_coeff", "_inexact")

    def __init__(self, trace: _Trace, coeff: _Coeff, inexact: bool = False) -> None:
        self._trace = trace
        self._coeff = coeff
        self._inexact = inexact

    def _other(self, other: Any, where: str) -> tuple:
        """(builder coefficient, whether a binary64 literal entered it) of an operand."""
        self._same_trace(other, where)
        if isinstance(other, TracedScalar):
            return other._coeff, other._inexact
        if isinstance(other, _Traced):
            raise TypeError("%s needs a duration or a number, got %s" % (where, type(other).__name__))
        _refuse_foreign(other, where)
        return _Coeff({0: other}), _coefficient(other)[1]

    def _with(self, coeff: _Coeff, inexact: bool) -> TracedScalar:
        return TracedScalar(self._trace, coeff, self._inexact or inexact)

    def __add__(self, other: Any) -> Any:
        if isinstance(other, TracedTime):
            return other.__add__(self)
        coeff, inexact = self._other(other, "a duration sum")
        return self._with(self._coeff + coeff, inexact)

    def __radd__(self, other: Any) -> Any:
        coeff, inexact = self._other(other, "a duration sum")
        return self._with(coeff + self._coeff, inexact)

    def __sub__(self, other: Any) -> Any:
        coeff, inexact = self._other(other, "a duration difference")
        return self._with(self._coeff - coeff, inexact)

    def __rsub__(self, other: Any) -> Any:
        coeff, inexact = self._other(other, "a duration difference")
        return self._with(coeff - self._coeff, inexact)

    def __neg__(self) -> Any:
        self._trace.require_active("a traced value")
        return self._with(-self._coeff, False)

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, TracedField):
            return other.__rmul__(self)
        coeff, inexact = self._other(other, "a duration product")
        return self._with(self._coeff * coeff, inexact)

    def __rmul__(self, other: Any) -> Any:
        coeff, inexact = self._other(other, "a duration product")
        return self._with(coeff * self._coeff, inexact)

    def __truediv__(self, other: Any) -> Any:
        self._trace.require_active("a traced value")
        if isinstance(other, _Traced):
            raise TypeError("a duration is divided by a number only (T9)")
        _refuse_foreign(other, "a duration quotient")
        return self._with(self._coeff / other, _coefficient(other)[1])

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
    materialized at ``B.time``. ``_inexact`` records that a binary64 literal entered the weights.
    """

    __slots__ = ("_expr", "_expansion", "_value", "_c", "_inexact")

    def __init__(self, trace: _Trace, expr: Any, expansion: tuple,
                 value: Any = None, c: Any = None, inexact: bool = False) -> None:
        self._trace = trace
        self._expr = expr
        self._expansion = expansion
        self._value = value
        self._c = c
        self._inexact = inexact

    def _builder(self) -> Any:
        return self._expr if self._value is None else self._value

    def _field(self, other: Any, where: str) -> TracedField:
        self._same_trace(other, where)
        if isinstance(other, TracedField):
            return other
        _refuse_foreign(other, where)
        raise TypeError("%s combines fields only; a number or a time is not a field (T9)" % where)

    def _scalar(self, other: Any, where: str) -> tuple:
        """(builder coefficient, exact polynomial, binary64?) of a factor of a field."""
        self._same_trace(other, where)
        if isinstance(other, TracedScalar):
            return other._coeff, _poly_of(other._coeff), other._inexact
        if isinstance(other, TracedField):
            raise TypeError("%s: a stage formula is affine in fields (T9); a product of two "
                            "fields is not" % where)
        if isinstance(other, _Traced):
            raise TypeError("%s: a field is scaled by a number or a duration (T9)" % where)
        _refuse_foreign(other, where)
        exact, inexact = _coefficient(other)
        return other, ((0, exact),), inexact

    def __add__(self, other: Any) -> Any:
        o = self._field(other, "a field sum")
        return TracedField(self._trace, self._builder() + o._builder(),
                           _expansion_add(self._expansion, o._expansion),
                           inexact=self._inexact or o._inexact)

    def __radd__(self, other: Any) -> Any:
        o = self._field(other, "a field sum")
        return TracedField(self._trace, o._builder() + self._builder(),
                           _expansion_add(o._expansion, self._expansion),
                           inexact=self._inexact or o._inexact)

    def __sub__(self, other: Any) -> Any:
        o = self._field(other, "a field difference")
        return TracedField(self._trace, self._builder() - o._builder(),
                           _expansion_add(self._expansion, _expansion_scale(
                               o._expansion, ((0, Fraction(-1)),))),
                           inexact=self._inexact or o._inexact)

    def __rsub__(self, other: Any) -> Any:
        o = self._field(other, "a field difference")
        return o.__sub__(self)

    def __neg__(self) -> Any:
        self._trace.require_active("a traced value")
        return TracedField(self._trace, -self._builder(),
                           _expansion_scale(self._expansion, ((0, Fraction(-1)),)),
                           inexact=self._inexact)

    def __mul__(self, other: Any) -> Any:
        factor, poly, inexact = self._scalar(other, "a field product")
        return TracedField(self._trace, self._builder() * factor,
                           _expansion_scale(self._expansion, poly),
                           inexact=self._inexact or inexact)

    def __rmul__(self, other: Any) -> Any:
        factor, poly, inexact = self._scalar(other, "a field product")
        return TracedField(self._trace, factor * self._builder(),
                           _expansion_scale(self._expansion, poly),
                           inexact=self._inexact or inexact)

    def __truediv__(self, other: Any) -> Any:
        self._trace.require_active("a traced value")
        if isinstance(other, _Traced):
            raise TypeError("a field is divided by a number only (T9)")
        _refuse_foreign(other, "a field quotient")
        exact, inexact = _coefficient(other)
        return TracedField(self._trace, self._builder() / other,
                           _expansion_scale(self._expansion, ((0, 1 / exact),)),
                           inexact=self._inexact or inexact)

    def __repr__(self) -> str:
        return "TracedField(%s)" % " + ".join(
            "(%s)*%r" % (_poly_text(poly), atom) for atom, poly in self._expansion)


def _state_weights(trace: _Trace, field: TracedField, where: str, *,
                   candidate: bool = False) -> tuple:
    """SU10 where a combination is read as a state; return its increment.

    The accepted state carries the weight one (within 1e-12 for binary64 weights, recorded in
    the trace) and every rate a duration, a polynomial in p.dt without a constant term. The
    increment is ``((evaluation, duration), ...)`` with each duration a ``CoeffPolynomial``.
    """
    increment = []
    for atom, poly in field._expansion:
        if atom is trace.state_atom:
            continue
        if any(power == 0 for power, _ in poly):
            raise IncrementTypeMismatch(
                "%s adds the rate %r to a state with the coefficient %s, which is not a "
                "duration: a rate times a duration of p.dt, which vanishes when p.dt is zero, is "
                "an increment (SU10)" % (where, atom, _poly_text(poly)))
        increment.append((atom.evaluation, CoeffPolynomial(dict(poly))))
    weights = dict(dict(field._expansion).get(trace.state_atom, ()))
    total = weights.get(0, Fraction(0))
    if total != 1:
        if field._inexact and abs(total - 1) <= _WEIGHT_TOLERANCE:
            trace.tolerated.append((where, total))
        elif total == 0 or field._inexact and abs(total) <= _WEIGHT_TOLERANCE:
            raise IncrementTypeMismatch(
                "%s is a difference: the weights of its states sum to zero, and %s, whose weights "
                "sum to one (SU10)" % (where, "a candidate is a state" if candidate
                                       else "a rate reads a state"))
        else:
            raise IncrementTypeMismatch(
                "%s combines states whose weights sum to %s, neither one nor zero (SU10)%s"
                % (where, repr(float(total)) if field._inexact else total,
                   "; binary64 weights are checked within 1e-12" if field._inexact else ""))
    if any(power != 0 for power in weights):
        state = _poly_text(tuple(sorted(weights.items())))
        if candidate:
            raise IncrementNotDerivable(
                "%s weighs the accepted state by %s: a closed form that no rate built, whose "
                "increment does not derive (T34). Name the terms it integrates, "
                "pops.exact_flow(method, of=term), or give its increment, Candidate(state, "
                "increment=...) (SU10); both come with a later lot of N8" % (where, state))
        raise IncrementTypeMismatch(
            "%s weighs the accepted state by %s, which depends on the step: a rate reads a "
            "state, whose weights sum to one (SU10)" % (where, state))
    return tuple(increment)


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
        evaluation = len(trace.evaluations)
        _state_weights(trace, state, "the state of evaluation %d of %r" % (evaluation, self._name))
        program = trace.program
        ordinal = len(trace.stages)
        point = program.stage("%s_stage%d" % (trace.name, ordinal), c=time.c)
        value, materialized = self._value_at(state, point, ordinal, time.c)
        trace.stages.append((ordinal, time.c, materialized))
        before, created = _guarded_tables(program), len(program._values)
        rate = self._evaluate(value)
        changed = _changed_tables(before, _guarded_tables(program))
        if changed:
            raise ClosedBodyViolation(
                "the component %r changed the %s of its Program: a component evaluates a rate, "
                "and publishes no state, history or step control outside the Candidate (T33)"
                % (self._name, ", ".join(changed)))
        _require_rate(program, rate, self._name, program._values[created:], value)
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


def _require_rate(program: Any, rate: Any, component: str, created: list, given: Any) -> None:
    """What a component returns: rates of its Program that this evaluation computed from the
    state ``given`` and at its point, without reading the step size (T33, T4)."""
    terms = (list(rate.terms) if isinstance(rate, _Affine)
             else [(rate, None)] if isinstance(rate, ProgramValue) else None)
    if not terms or any(not (isinstance(value, ProgramValue) and value.is_field())
                        for value, _ in terms):
        raise TypeError("the component %r must return the rate of its state, a field value of "
                        "the Program, got %r" % (component, rate))
    if any(value.prog is not program for value, _ in terms):
        raise ClosedBodyViolation(
            "the component %r returned a value of another Program (T33)" % component)
    if any(_reads_step_size(coefficient) for _, coefficient in terms) \
            or any(_reads_step_size(value.attrs) for value in created):
        raise ClosedBodyViolation(
            "the component %r reads the step size of its Program: a component evaluates a rate "
            "at the state and time it is given, and the method scales it by a duration of p.dt "
            "(T33)" % component)
    others = sorted({value.vtype for value, _ in terms if value.vtype != "rhs"})
    if others:
        raise IncrementTypeMismatch(
            "the component %r returned a value of type %s, not a rate: a component evaluates the "
            "rate of the state it is given (SU10)" % (component, " and ".join(others)))
    computed = {id(value) for value in created}
    if any(id(value) not in computed for value, _ in terms):
        raise ClosedBodyViolation(
            "the component %r returned a rate that this evaluation did not compute, such as a "
            "rate kept from an earlier evaluation: a component evaluates a rate at the state and "
            "time it is given (T33, T4)" % component)
    computed.add(id(given))
    outside = sorted({read.name for value in created for read in value.inputs
                      if id(read) not in computed})
    if outside:
        raise ClosedBodyViolation(
            "the component %r read %s, which it was not given and did not compute: a component "
            "evaluates a rate from the state it is given (T33, T4)" % (component, ", ".join(outside)))
    elsewhere = [value.name for value in created
                 if value.point is not None and value.point != given.point]
    if elsewhere:
        raise ClosedBodyViolation(
            "the component %r evaluated %s at another time than the state it is given: a "
            "component evaluates a rate at the state and time it is given, and the method states "
            "each time, flow(U, B.time + c*p.dt) (T33, T4)" % (component, ", ".join(elsewhere)))


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

    ``increment`` holds ``(evaluation, duration)`` pairs, each duration a ``CoeffPolynomial`` in
    ``p.dt``: the candidate is ``B.state + sum(duration * rate)`` (T34). ``tolerated`` lists the
    combinations whose binary64 weights sum to one within 1e-12, with their exact sums (SU10);
    ``describe()`` gives each sum minus one, which the nearest binary64 of the sum would hide.
    """

    __slots__ = ("name", "components", "stages", "evaluations", "increment", "candidate",
                 "tolerated")

    def __init__(self, name: str, components: tuple, stages: tuple, evaluations: tuple,
                 increment: tuple, candidate: Any, tolerated: tuple = ()) -> None:
        self.name = name
        self.components = components
        self.stages = stages
        self.evaluations = evaluations
        self.increment = increment
        self.candidate = candidate
        self.tolerated = tolerated

    def describe(self) -> dict:
        """Plain data for inspection and tests; times and durations as exact fractions of ``p.dt``."""
        return {
            "method": self.name,
            "components": list(self.components),
            "stages": [{"stage": ordinal, "c": str(Fraction(_fraction(c))),
                        "materializes": name} for ordinal, c, name in self.stages],
            "evaluations": [{"evaluation": index, "component": component, "stage": stage}
                            for index, component, stage in self.evaluations],
            "increment": [{"evaluation": index, "duration": _poly_text(duration)}
                          for index, duration in self.increment],
            "tolerated_weights": [{"where": where, "deviation": repr(float(total - 1))}
                                  for where, total in self.tolerated],
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
                 for name, evaluate in zip(self.method.component_names, self.components,
                                           strict=True)]
        before = _guarded_tables(program)
        try:
            candidate = self.method.body(_Continuation(trace, state), _ProposalView(trace), *views)
        finally:
            trace.active = False
        changed = _changed_tables(before, _guarded_tables(program))
        if changed:
            raise ClosedBodyViolation(
                "the body of %r changed the %s of its Program outside its Candidate (T33)"
                % (self.name, ", ".join(changed)))
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
        increment = _state_weights(trace, candidate.state, "the candidate of %r" % self.name,
                                   candidate=True)
        value = program.value("%s_candidate" % self.name, candidate.state._builder(),
                              at=temporal.next.point)
        _require_state(value, "Candidate(state=...) is a state")
        return MethodTrace(self.name, tuple(self.method.component_names), tuple(trace.stages),
                           tuple(trace.evaluations), increment, value, tuple(trace.tolerated))


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
        bound = dict(zip(names, components, strict=False))
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
