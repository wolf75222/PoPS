"""ADC-666: explicit StepStrategy and StepTransactionReport contracts."""
from __future__ import annotations

import pytest

from pops.time import AdaptiveCFL, FixedDt, Program, StepTransactionReport


def test_program_step_strategy_is_explicit_and_serializable():
    program = Program("tx")
    program.step_strategy(
        FixedDt(0.125),
        staged_effects=("state", "history"),
        guards=("finite_residual",),
        projections=("positivity",),
    )

    plan = program.transaction_plan()
    assert plan is not None
    data = plan.to_data()
    assert "FixedDt" in data["strategy"]
    assert data["staged_effects"] == ["state", "history"]
    assert data["guards"] == ["finite_residual"]
    assert data["projections"] == ["positivity"]
    assert program.validate_runtime_controls({}) is True


def test_runtime_controls_cannot_select_a_strategy_implicitly():
    program = Program("tx")
    with pytest.raises(ValueError, match="runtime controls require Program.step_strategy"):
        program.validate_runtime_controls({"cfl": 0.5})

    program.step_strategy(AdaptiveCFL(cfl=0.8))
    assert program.validate_runtime_controls({}) is True
    assert program.validate_runtime_controls({"dt_min": 1e-6, "dt_max": 1e-2}) is True
    with pytest.raises(ValueError, match="dt_min/dt_max"):
        program.validate_runtime_controls({"cfl": 0.4})


def test_step_transaction_report_refuses_partial_publication():
    accepted = StepTransactionReport(
        status="accepted",
        phase="commit",
        action="commit",
        staged_effects=("state",),
        committed_effects=("state",),
    )
    assert accepted.to_data()["committed_effects"] == ["state"]

    rejected = StepTransactionReport(
        status="rejected",
        phase="solve",
        action="reject_attempt",
        staged_effects=("state", "history"),
        rolled_back_effects=("state", "history"),
    )
    assert rejected.to_data()["rolled_back_effects"] == ["state", "history"]

    with pytest.raises(ValueError, match="cannot include committed_effects"):
        StepTransactionReport(
            status="rejected",
            phase="guard",
            action="reject_attempt",
            committed_effects=("state",),
        )
    with pytest.raises(ValueError, match="cannot include rolled_back_effects"):
        StepTransactionReport(
            status="accepted",
            phase="commit",
            action="commit",
            rolled_back_effects=("state",),
        )
