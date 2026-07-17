"""Callable field operators carry exact owner, stage and transaction provenance."""
from __future__ import annotations

import pytest

from typed_program_support import state_refs, typed_field, typed_state

from pops.time import FailRun, Program, TimePoint
from pops.time.field_context import FieldContext


def _fields(program, *states, name="phi"):
    return typed_field(program, name)(*states).consume(action=FailRun())


def test_context_is_exact_immutable_and_rejects_stale_reads() -> None:
    program = Program("field-context")
    state = typed_state(program, "fluid")
    block, _ = state_refs(program, "fluid")
    field = typed_field(program, "phi")
    context = FieldContext(field, ((block, state.id),), ("phi",))

    assert context.matches(field, block, state.id)
    assert context.output("phi") == "phi"
    with pytest.raises(ValueError, match="incompatible field context"):
        context.require_read(field, block, state.id + 1)
    with pytest.raises((AttributeError, TypeError)):
        context.outputs = ("other",)


def test_callable_field_tracks_every_coupled_stage_source() -> None:
    program = Program("coupled-field")
    left = typed_state(program, "left")
    right = typed_state(program, "right")
    fields = _fields(program, left, right)

    assert fields.field_context.stage_sources == (
        (left.block, left.id), (right.block, right.id))


def test_callable_field_rejects_foreign_values_and_mixed_time_points() -> None:
    program = Program("field-owner")
    left = typed_state(program, "left")
    right = typed_state(program, "right")
    field = typed_field(program, "phi")

    foreign = Program("foreign")
    foreign_state = typed_state(foreign, "foreign")
    with pytest.raises(ValueError, match="different Program"):
        field(left, foreign_state)

    staged = program.value("right-stage", right, at=TimePoint(program.clock, 0.5))
    with pytest.raises(ValueError, match="one exact TimePoint"):
        field(left, staged)


def test_field_solve_requires_explicit_failure_action() -> None:
    program = Program("field-transaction")
    state = typed_state(program, "fluid")
    outcome = typed_field(program, "phi")(state)

    assert "unconsumed" in repr(outcome)
    with pytest.raises(TypeError, match="consume"):
        _ = outcome.token
    assert outcome.consume(action=FailRun()).op == "solve_outcome_component"
