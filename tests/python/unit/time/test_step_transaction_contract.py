"""ADC-666: typed, lowered and identity-bearing macro-step transactions."""
from __future__ import annotations
from pops.codegen.program_codegen import emit_cpp_program

import pytest

import pops.time as time
from pops.time import (
    ALL_PROVISIONAL_STORES,
    AdaptiveCFL,
    BlockProjection,
    ErrorControlledDt,
    FailRun,
    FixedDt,
    GuardRole,
    Program,
    ProjectAndRecheck,
    ProvisionalStore,
    RejectAttempt,
    StepTransactionReport,
)
from typed_program_support import typed_state


def _error_strategy():
    return ErrorControlledDt(
        dt_init=0.1,
        rtol=1.0e-3,
        atol=1.0e-8,
        dt_min=1.0e-5,
        dt_max=0.5,
        max_rejections=5,
    )


def test_program_step_strategy_owns_typed_stores_and_is_serialized():
    program = Program("tx")
    program.step_strategy(FixedDt(0.125), stores=ALL_PROVISIONAL_STORES)

    data = program.transaction_plan().to_data()
    assert data["strategy"]["kind"] == "fixed_dt"
    assert data["stores"] == [store.value for store in ProvisionalStore]
    assert program._serialize()["step_transaction"] == data
    with pytest.raises(TypeError, match="ProvisionalStore"):
        program.step_strategy(FixedDt(0.125), stores=("states",))


def test_project_and_recheck_is_lazy_lowered_and_visible_in_transaction_identity():
    program = Program("guarded")
    temporal = typed_state(program, "fluid", state_name="U")
    state = program.value("candidate", temporal.n, at=temporal.next.point)
    error = program.norm_inf(state)
    strategy = _error_strategy()
    condition = error <= strategy.atol + strategy.rtol * program.norm_inf(state)
    guarded = program.guard(
        "embedded_error",
        state,
        condition,
        action=ProjectAndRecheck(BlockProjection(), on_failure=RejectAttempt()),
        role=GuardRole.ERROR_ESTIMATE,
        recheck=lambda P, projected: P.norm_inf(projected)
        <= strategy.atol + strategy.rtol * P.norm_inf(projected),
    )
    program.commit(temporal.next, guarded)
    program.step_strategy(strategy)

    plan = program.transaction_plan()
    assert plan.guards[0].role is GuardRole.ERROR_ESTIMATE
    assert plan.projections == (BlockProjection(),)
    source = emit_cpp_program(program)
    assert "ctx.apply_projection(0," in source
    assert '"guard"' in source
    assert "StepAttemptRejected" in source
    assert "ctx.commit_many({" in source


def test_error_controlled_strategy_requires_an_actual_lowered_error_guard():
    program = Program("missing_guard")
    program.step_strategy(_error_strategy())
    with pytest.raises(ValueError, match="ERROR_ESTIMATE"):
        program.transaction_plan()


def test_runtime_controls_cannot_select_a_strategy_implicitly():
    program = Program("tx")
    with pytest.raises(ValueError, match="Program.step_strategy"):
        program.validate_runtime_controls({"dt_max": 0.5})
    program.step_strategy(AdaptiveCFL(cfl=0.8))
    assert program.validate_runtime_controls({"dt_min": 1.0e-6, "dt_max": 1.0e-2}) is True
    with pytest.raises(ValueError, match="dt_min/dt_max"):
        program.validate_runtime_controls({"cfl": 0.4})


def test_step_transaction_report_refuses_partial_publication():
    accepted = StepTransactionReport(
        status="accepted",
        phase="commit",
        action="commit",
        staged_effects=("states",),
        committed_effects=("states",),
    )
    assert accepted.to_data()["committed_effects"] == ["states"]
    with pytest.raises(ValueError, match="cannot include committed_effects"):
        StepTransactionReport(
            status="rejected",
            phase="guard",
            action="reject_attempt",
            committed_effects=("states",),
        )
    with pytest.raises(ValueError, match="cannot include rolled_back_effects"):
        StepTransactionReport(
            status="accepted",
            phase="commit",
            action="commit",
            rolled_back_effects=("states",),
        )


def test_compiled_time_is_removed_instead_of_aliased():
    assert not hasattr(time, "CompiledTime")
    with pytest.raises(ImportError):
        exec("from pops.time import CompiledTime", {})


def test_fail_run_guard_is_typed_and_lowered_as_fatal():
    program = Program("fatal_guard")
    temporal = typed_state(program, "fluid", state_name="U")
    state = program.value("candidate", temporal.n, at=temporal.next.point)
    guarded = program.guard(
        "finite_state", state, program.norm_inf(state) >= 0.0, action=FailRun())
    program.commit(temporal.next, guarded)
    program.step_strategy(FixedDt(0.1))
    assert "throw std::runtime_error" in emit_cpp_program(program)
