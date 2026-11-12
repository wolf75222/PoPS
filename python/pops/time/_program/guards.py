"""Admissibility guards of a traced method (spec 0.3 IR4, S5, T38, T39; N8 lot 6).

A method body gates a traced field by a predicate component, with the shared vocabulary of IR4:

    U = cond(domain(B.state), lambda: B.state, lambda: fail(NotAdmissible))

``domain`` is a component of the method read without a time: ``domain(U)`` is a predicate that
the guard evaluates on the value of ``U`` where ``U`` is materialized. Two places are elaborated
(lot 6 design note, revision 2, section 3): ``B.state`` (or ``B.state`` already gated), at
``B.time``, at once; and the candidate, at the endpoint ``B.time + p.dt``, where the trace
materializes it without a stage and commits the gated value itself. A guard of an intermediate
stage value, or a guarded combination that a rate reads or that is not the candidate, is not
elaborated. The component returns a scalar ``Bool`` of the builder, such as
``p.min(margin) > 0``. ``cond`` returns ``U`` gated, with the exact expansion of ``U``, since the
other arm fails: a rate that reads it reads the gated value, as b6's ``Program.guard`` returns its
value, and the guard is transparent to SU10, T34 and the classification of T38.

This lot elaborates that form only: ``then`` returns the guarded field itself, ``otherwise``
returns ``fail(NotAdmissible)`` without a payload, and the predicate is a component's. The other
choices of IR4 come with a later lot of N8 (choices between computations in a body).

T38. A guard whose field is ``B.state`` alone, with no coefficient in ``p.dt``, has a dependency
set that excludes the proposal: its failure ends the step with ``NotAdmissible`` (phase
``trial_failure``) and consumes no attempt. Any other guard is a retryable site (phase
``trial``), counted against ``Retry.max_attempts``. The guard lowers onto b6's acceptance guard
with ``RejectAttempt()`` (``Program._trial_guard``, with the checks of ``Program.guard``); its
name is its site, ``<method>/body#k/<component>``, with ``k`` counting the guards of the body,
so two guards by one predicate have two names. Only this lowering adds an acceptance guard
during a body: a component still may not (T33).
"""
from __future__ import annotations

from typing import Any

from pops.time._program.traced_method import (
    ClosedBodyViolation, NotYetElaborated, TracedField, _changed_tables, _guarded_tables,
    _issued_since, _PredicateCall, _reads_step_size, _refuse_foreign,
)
from pops.time.values import ProgramValue

# The closed sum of T39, by name: pops.time does not import the runtime that defines the classes.
_VARIANTS = frozenset(("BudgetExhausted", "ResourceExhausted", "IncompatibleData",
                       "NotAdmissible", "PartialOperation", "SolveFailed", "FrontierMismatch"))
_CHOICES = "a later lot of N8 (choices between computations in a body, IR4)"


class Failure:
    """``fail(variant, payload=...)``: the arm of a choice that fails the trial (IR4, T39)."""

    __slots__ = ("variant", "payload")

    def __init__(self, variant: str, payload: Any) -> None:
        self.variant = variant
        self.payload = payload

    def __repr__(self) -> str:
        return "fail(%s)" % self.variant


def _variant_name(variant: Any) -> str:
    name = getattr(variant, "__name__", None) if isinstance(variant, type) else None
    if name not in _VARIANTS or not any(base.__name__ == "RunFailed" for base in variant.__mro__):
        raise TypeError("fail(variant): a variant of the closed sum of T39 (pops.NotAdmissible "
                        "and the others), got %r" % (variant,))
    return name


def fail(variant: Any, payload: Any = None) -> Failure:
    """``fail(variant, payload=...)``: a typed failure of the trial, in an arm of ``cond``."""
    name = _variant_name(variant)
    if name != "NotAdmissible" or payload is not None:
        raise NotYetElaborated(
            "fail(%s%s): this lot elaborates fail(NotAdmissible) without a payload, the failure "
            "of an admissibility guard; the other variants and the payloads come with %s"
            % (name, "" if payload is None else ", payload=...", _CHOICES))
    return Failure(name, payload)


def _fixed(trace: Any, field: TracedField) -> bool:
    """T38: the field is B.state alone, without p.dt, so the guard excludes the proposal."""
    return all(atom is trace.state_atom and all(power == 0 for power, _ in poly)
               for atom, poly in field._expansion)


class Guard:
    """One admissibility guard of a method body: its predicate, its site and its class (T38)."""

    __slots__ = ("trace", "component", "site", "fixed", "applied")

    def __init__(self, trace: Any, component: Any, site: str, fixed: bool) -> None:
        self.trace = trace
        self.component = component
        self.site = site
        self.fixed = fixed
        self.applied = False

    def apply(self, value: ProgramValue) -> ProgramValue:
        """Evaluate the predicate on ``value`` and gate it; once, where the field materializes."""
        if self.applied:
            raise RuntimeError("the guard %s is applied once" % self.site)
        program = self.trace.program
        before, first = _guarded_tables(program), program._next_id
        condition = self.component._evaluate(value)
        changed = _changed_tables(before, _guarded_tables(program))
        if changed:
            raise ClosedBodyViolation(
                "the predicate %r changed the %s of its Program: a predicate evaluates a "
                "condition and publishes nothing (T33)" % (self.component._name, ", ".join(changed)))
        if not (isinstance(condition, ProgramValue) and condition.vtype == "bool"
                and condition.prog is program):
            raise TypeError("the predicate %r must return a scalar Bool of its Program, such as "
                            "p.min(margin) > 0, got %r" % (self.component._name, condition))
        if any(_reads_step_size(node.attrs) for node in _issued_since(program, first)):
            raise ClosedBodyViolation(
                "the predicate %r reads the step size of its Program: a predicate reads the "
                "value it is given (T33)" % self.component._name)
        guarded = program._trial_guard(self.site, value, condition, {
            "site": self.site, "retry_can_change": not self.fixed, "variant": "NotAdmissible"})
        self.applied = True
        return guarded


def cond(predicate: Any, then: Any, otherwise: Any) -> TracedField:
    """``cond(p, then, otherwise)`` of IR4, in the form of an admissibility guard (module doc)."""
    if not isinstance(predicate, _PredicateCall):
        _refuse_foreign(predicate, "cond(p, ...)")
        raise NotYetElaborated(
            "cond(p, ...) on %r: this lot elaborates the guard of a field by a predicate "
            "component, cond(domain(U), then=..., otherwise=fail(NotAdmissible)); the other "
            "choices of IR4 come with %s" % (predicate, _CHOICES))
    field = predicate.state
    trace = field._trace
    trace.require_active("cond(...)")
    if predicate.consumed:
        raise TypeError("cond(p, ...): the predicate %s(U) is consumed by one cond only (IR4)"
                        % predicate.component._name)
    predicate.consumed = True
    if not callable(then) or not callable(otherwise):
        raise TypeError("cond(p, then, otherwise): then and otherwise are functions of no "
                        "argument, of which only the chosen one runs (IR4)")
    if then() is not field:
        raise NotYetElaborated(
            "cond(domain(U), then=...): this lot elaborates the guard whose then-arm returns U "
            "itself; other arms come with %s" % _CHOICES)
    failure = otherwise()
    if not isinstance(failure, Failure):
        raise NotYetElaborated(
            "cond(domain(U), otherwise=...): this lot elaborates an otherwise-arm that is "
            "fail(NotAdmissible); other arms come with %s" % _CHOICES)
    fixed = _fixed(trace, field)
    if field._value is not None and not fixed:
        raise NotYetElaborated(
            "cond(%s(U), ...) on a stage value: this lot elaborates the guard of B.state and of "
            "the candidate; a guard of an intermediate stage value comes with %s"
            % (predicate.component._name, _CHOICES))
    site = "%s/body#%d/%s" % (trace.name, trace.body_guards, predicate.component._name)
    trace.body_guards += 1
    guard = Guard(trace, predicate.component, site, fixed)
    trace.guards.append(guard)
    if field._value is not None:
        # B.state, or B.state already gated: the guard gates it now, at B.time.
        value = guard.apply(field._value)
        return TracedField(trace, value, field._expansion, value=value, c=field._c,
                           inexact=field._inexact)
    return TracedField(trace, field._expr, field._expansion, inexact=field._inexact,
                       guards=field._guards + (guard,))


__all__ = ["Failure", "Guard", "cond", "fail"]
