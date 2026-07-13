"""Strict helpers for final ``pops.lib.time`` Program factories."""
from __future__ import annotations

from typing import Any


def resolve_solve_action(value: Any, where: str) -> Any:
    """Return a validated explicit failure action, defaulting a preset to fail closed."""
    from pops.time import FailRun, SolveAction

    action = FailRun() if value is None else value
    if not isinstance(action, SolveAction):
        raise TypeError(
            "%s solve_action must be FailRun(...) or RejectAttempt(...)" % where)
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
            "%s requires the exact block-qualified state handle produced by block[state]" % where)
    return program.state(reference)


def operator_handle(value: Any, where: str) -> Any:
    """Require an exact owner-qualified operator handle."""
    from pops.model import OperatorHandle

    if not isinstance(value, OperatorHandle):
        raise TypeError(
            "%s must be the exact owner-qualified OperatorHandle returned by a model declarer"
            % where)
    return value


def call_at(
    program: Any,
    handle: Any,
    *candidate_args: Any,
    name: str,
    point: Any,
    solve_action: Any = None,
) -> Any:
    """Call one typed operator, explicitly consume a field solve, and name its result."""
    from ._helpers import _op_space_arity
    from pops.time import FieldSolveOutcome, SolveAction

    arity = _op_space_arity(program, handle)
    # A nullary operator has no ProgramValue from which the callable handle could recover the
    # authoring Program. Presets own that internal lowering boundary explicitly.
    value = program._call(handle) if arity == 0 else handle(*candidate_args[:arity])
    if isinstance(value, FieldSolveOutcome):
        if not isinstance(solve_action, SolveAction):
            raise TypeError(
                "field operator %r requires solve_action=FailRun(...) or RejectAttempt(...)"
                % handle.name)
        value = value.consume(action=solve_action)
    elif solve_action is not None:
        raise TypeError(
            "solve_action applies only to a field-operator outcome; %r returned %s"
            % (handle.name, type(value).__name__))
    return program.value(name, value, at=point)


__all__ = [
    "call_at", "instance_state", "operator_handle", "program_factory", "resolve_solve_action",
]
