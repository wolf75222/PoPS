"""Fail-closed public outcome contract for Program field solves."""

from __future__ import annotations

import pytest

from pops.time import FailRun, FieldSolveOutcome, Program, RejectAttempt

from typed_program_support import typed_field, typed_state


def _field_solve():
    program = Program("field_outcome")
    state = typed_state(program, "plasma")
    field = typed_field(program, "potential")
    return program, state, program.solve_fields(state=state, field=field)


def test_field_solve_is_unreadable_and_unconsumed_graph_is_rejected() -> None:
    program, _state, outcome = _field_solve()

    assert isinstance(outcome, FieldSolveOutcome)
    with pytest.raises(TypeError, match="not readable"):
        _ = outcome.token
    with pytest.raises(TypeError, match="not iterable"):
        iter(outcome)
    with pytest.raises(ValueError, match="consumed exactly once"):
        program.to_graph()


@pytest.mark.parametrize("action", (FailRun(), RejectAttempt()))
def test_consuming_field_solve_publishes_one_exact_field_context(action) -> None:
    program, state, outcome = _field_solve()
    fields = outcome.consume(action=action)
    graph = program.to_graph()

    assert fields.vtype == "fields"
    assert fields.op == "solve_outcome_component"
    assert fields.field_context.stage_sources == ((state.block, state.id),)
    operations = [getattr(node, "op", None) for node in graph.nodes]
    assert operations.count("solve_fields") == 1
    assert operations.count("solve_outcome") == 1
    assert operations.count("solve_outcome_component") == 1
    with pytest.raises(RuntimeError, match="already been consumed"):
        outcome.consume(action=action)


def test_field_solve_consume_rejects_untyped_failure_action_atomically() -> None:
    program, _state, outcome = _field_solve()
    before = tuple(program._values)

    with pytest.raises(TypeError, match="requires action"):
        outcome.consume(action="fail_run")

    assert tuple(program._values) == before
    with pytest.raises(ValueError, match="consumed exactly once"):
        program.to_graph()
