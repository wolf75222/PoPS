"""Mandatory collectable proofs for the explicit solve-outcome protocol."""
from __future__ import annotations

import pytest

from pops.linalg import LinearProblem
from pops.solvers import CG
from pops.time import Program, RejectAttempt


def _linear_outcome():
    program = Program("outcome-contract")
    operator = program.matrix_free_operator("identity", ncomp=1)
    program.set_apply(operator, lambda _program, _out, value: value)
    rhs = program.scalar_field("rhs")
    return program, program.solve(
        LinearProblem(operator, rhs), solver=CG(max_iter=2))


def test_reject_attempt_is_explicit_in_the_canonical_program_graph():
    program, outcome = _linear_outcome()
    solved = outcome.consume(action=RejectAttempt(statuses=("iteration_limit",)))

    node = next(node for node in program.to_graph().nodes
                if getattr(node, "op", None) == "solve_outcome")
    action = node.attrs.to_data()["attrs"]["action"]
    assert action["kind"] == "reject_attempt"
    assert action["statuses"] == ["iteration_limit"]
    assert solved.op == "solve_outcome_component"


def test_solve_outcome_refuses_implicit_invalid_and_double_consumption():
    _program, outcome = _linear_outcome()
    with pytest.raises(TypeError, match="not readable"):
        _ = outcome.token
    with pytest.raises(TypeError, match="requires action"):
        outcome.consume(action=None)

    outcome.consume(action=RejectAttempt())
    with pytest.raises(RuntimeError, match="already been consumed"):
        outcome.consume(action=RejectAttempt())
