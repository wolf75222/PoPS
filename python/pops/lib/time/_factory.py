"""Strict helpers for final ``pops.lib.time`` Program factories."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


def resolve_solve_action(value: Any, where: str) -> Any:
    """Return a validated explicit failure action, defaulting a preset to fail closed."""
    from pops.time import FailRun, SolveAction

    action = FailRun() if value is None else value
    if not isinstance(action, SolveAction):
        raise TypeError("%s solve_action must be FailRun(...) or RejectAttempt(...)" % where)
    return action


def program_factory(name: str, build: Any, *args: Any, **kwargs: Any) -> Any:
    """Build one ordinary Program through the same operations as manual authoring."""
    from pops.provenance import callable_span, source_span
    from pops.time import Program

    program = Program(name)
    program._provenance_context = {
        "caller": source_span(),
        "factory": callable_span(build),
        "authoring_api": "%s.%s" % (build.__module__, build.__qualname__),
    }
    try:
        build(program, *args, **kwargs)
    finally:
        program._provenance_context = None
    return program


def instance_state(program: Any, reference: Any, where: str) -> Any:
    """Select one exact block-owned state; no ``(block, declaration)`` fallback."""
    from pops.model import Handle

    if not isinstance(reference, Handle) or not reference.is_instance:
        raise TypeError(
            "%s requires the exact block-qualified state handle produced by block[state]" % where
        )
    return program.state(reference)


def operator_handle(value: Any, where: str) -> Any:
    """Require an exact owner-qualified operator handle."""
    from pops.model import OperatorHandle

    if not isinstance(value, OperatorHandle):
        raise TypeError(
            "%s must be the exact owner-qualified OperatorHandle returned by a model declarer"
            % where
        )
    return value


def field_handle(value: Any, where: str) -> Any:
    """Require the single Case-owned authority for a Program field solve."""
    from pops.problem.handles import FieldHandle

    if not isinstance(value, FieldHandle):
        raise TypeError("%s must be the exact FieldHandle returned by Case.field(...)" % where)
    return value


def exact_ordered_routes(value: Any, route_type: type, where: str) -> tuple[Any, ...]:
    """Require one immutable route tuple; mutable or pair-shaped substitutes are refused."""
    if type(value) is not tuple:
        kind = "sequence" if isinstance(value, Sequence) else type(value).__name__
        raise TypeError(
            "%s routes must be an exact tuple of %s objects, not %s"
            % (where, route_type.__name__, kind)
        )
    if not value:
        raise ValueError("%s routes must contain at least one route" % where)
    if any(type(route) is not route_type for route in value):
        raise TypeError(
            "%s routes must contain only exact %s objects" % (where, route_type.__name__)
        )
    states = tuple(route.state for route in value)
    if len(set(states)) != len(states):
        raise ValueError("%s routes contain a duplicate state" % where)
    return value


def call_at(
    program: Any,
    handle: Any,
    *candidate_args: Any,
    name: str,
    point: Any,
) -> Any:
    """Call one typed model operator and name its result at an exact point."""
    from ._helpers import _op_space_arity

    arity = _op_space_arity(program, handle)
    # A nullary operator has no ProgramValue from which the callable handle could recover the
    # authoring Program. Presets own that internal lowering boundary explicitly.
    value = handle(program=program) if arity == 0 else handle(*candidate_args[:arity])
    return program.value(name, value, at=point)


@dataclass(frozen=True, slots=True)
class _FieldCall:
    """One field value together with the exact aligned states that produced it."""

    fields: Any
    states: tuple[Any, ...]


def _call_field_at(
    program: Any,
    field: Any,
    *states: Any,
    name: str,
    point: Any,
    solve_action: Any,
) -> _FieldCall:
    """Solve one Case field and retain every exact state used by its provenance."""
    from pops.time import SolveAction, StagePoint, TimePoint

    field = field_handle(field, "time factory field")
    if not isinstance(solve_action, SolveAction):
        raise TypeError("field solve requires solve_action=FailRun(...) or RejectAttempt(...)")
    solve_point = point
    if type(point) is StagePoint:
        try:
            solve_point = point.time
        except ValueError:
            source_points = {state.point for state in states}
            if len(source_points) != 1 or type(next(iter(source_points))) is not TimePoint:
                raise ValueError(
                    "field solve at a partitioned StagePoint requires one unambiguous physical "
                    "coordinate; materialize the stage state at a TimePoint first"
                ) from None
            solve_point = next(iter(source_points))
    if type(solve_point) is not TimePoint:
        raise TypeError("field solve requires an exact TimePoint or StagePoint")
    aligned = tuple(
        state
        if state.point == solve_point
        else program.value("%s_state_%d" % (name, index), state, at=solve_point)
        for index, state in enumerate(states)
    )
    outcome = field(*aligned, name=name)
    value = outcome.consume(action=solve_action)
    return _FieldCall(program.value(name, value, at=point), aligned)


def call_field_at(
    program: Any,
    field: Any,
    *states: Any,
    name: str,
    point: Any,
    solve_action: Any,
) -> Any:
    """Solve one Case field at an exact point and return its consumed field value."""
    return _call_field_at(
        program,
        field,
        *states,
        name=name,
        point=point,
        solve_action=solve_action,
    ).fields


__all__ = [
    "call_at",
    "call_field_at",
    "exact_ordered_routes",
    "field_handle",
    "instance_state",
    "operator_handle",
    "program_factory",
    "resolve_solve_action",
]
