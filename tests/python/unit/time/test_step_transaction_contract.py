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
    projection = source.index("ctx.apply_projection(0,")
    projection_report = source.index(
        'ctx.note_step_projection("embedded_error");', projection)
    recheck_reduction = source.index("ctx.norm_inf(0,", projection_report)
    recheck_branch = source.index("if (!(", recheck_reduction)
    rejection = source.index("StepAttemptRejected", recheck_branch)
    commit = source.index("ctx.commit_many({", rejection)
    assert projection < projection_report < recheck_reduction < recheck_branch < rejection < commit

    report = StepTransactionReport(
        status="accepted",
        phase="commit",
        action="commit",
        projections=tuple(
            guard.name for guard in plan.guards
            if type(guard.action) is ProjectAndRecheck
        ),
    )
    assert report.to_data()["projections"] == ["embedded_error"]


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
        projections=("realizability",),
    )
    assert accepted.to_data()["committed_effects"] == ["states"]
    assert accepted.to_data()["projections"] == ["realizability"]
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
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        StepTransactionReport(
            status="accepted",
            phase="commit",
            action="commit",
            projections=("realizability", "realizability"),
        )
    with pytest.raises(ValueError, match="staged_effects must contain non-empty effect names"):
        StepTransactionReport(
            status="rejected",
            phase="guard",
            action="reject_attempt",
            staged_effects=("",),
            rolled_back_effects=("",),
        )
    with pytest.raises(ValueError, match="committed_effects cannot contain duplicates"):
        StepTransactionReport(
            status="accepted",
            phase="commit",
            action="commit",
            staged_effects=("states",),
            committed_effects=("states", "states"),
        )
    with pytest.raises(ValueError, match="commit every staged_effect"):
        StepTransactionReport(
            status="accepted",
            phase="commit",
            action="commit",
            staged_effects=("states", "fields"),
            committed_effects=("states",),
        )
    with pytest.raises(ValueError, match="roll back every staged_effect"):
        StepTransactionReport(
            status="rejected",
            phase="guard",
            action="reject_attempt",
            staged_effects=("states", "fields"),
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


def test_global_guard_carries_the_observed_predicate_at_its_compiled_site():
    from pathlib import Path

    from pops.time._program.guard_failure_sites import SCHEMA, guard_failure_sites

    program = Program("guarded")
    temporal = typed_state(program, "fluid", state_name="U")
    state = temporal.n
    condition = (program.min(state) > 0) & (program.max(state) < 2)
    current = program.guard("admitted", state, condition, action=FailRun())
    stage = program.value("stage", current, at=temporal.next.point)
    guarded = program._trial_guard("checked_stage", stage, program.min(stage) > 0, {
        "site": "checked_stage", "variant": "NotAdmissible", "retry_can_change": False,
    })
    program.commit(temporal.next, guarded)
    program.step_strategy(FixedDt(0.125))

    source = emit_cpp_program(program)
    terminal, trial = guard_failure_sites(program, program._ir_hash())
    assert terminal.name == "admitted" and terminal.action == "fail_run"
    assert trial.name == "checked_stage" and trial.action == "reject_attempt"
    assert terminal.predicate == condition.id and terminal.token != trial.token
    assert Path(terminal.python_origin.file).name == Path(__file__).name
    assert terminal.python_origin.line < trial.python_origin.line
    assert f'StepTerminalFailure("acceptance guard' in source
    assert f'"trial_failure", "checked_stage failed: NotAdmissible", "{SCHEMA}"' in source
    for site, node in ((terminal, current), (trial, guarded)):
        observed = f"guard_observed_{node.id}"
        assert f"const bool {observed} = " in source
        assert f"""("{site.payload_prefix()}") + ({observed} ? '1' : '0')""" in source


@pytest.mark.parametrize("checked", (False, True))
@pytest.mark.parametrize("another_reader", (False, True))
def test_owned_guard_fold_requires_an_exclusive_cut(checked, another_reader):
    program = Program("owned_cells")
    temporal = typed_state(program, "fluid", state_name="U")
    state = temporal.n
    if checked:
        component = state[0]
        mapped = program.value("nonlinear_margin", (component * component - 1,))
        reduced = program.min(mapped)
        predicate = reduced > 0
    else:
        reduced = program.min(state)
        norm = program.norm_inf_all(state)
        predicate = program.isfinite(norm) & (reduced > 0)
    candidate = program.guard("original_site", state, predicate, action=FailRun())
    if another_reader:
        candidate = program.guard("second_site", candidate, reduced < 100, action=FailRun())
    program.commit(temporal.next, program.value("next", candidate, at=temporal.next.point))
    program.step_strategy(FixedDt(0.125))

    source = emit_cpp_program(program)
    selected = "ctx.owned_guard_fold<true>" if checked else "ctx.owned_guard_fold<false>"
    if another_reader:
        assert "ctx.owned_guard_fold<" not in source
        assert "prepare_owned_guard_host_lane" not in source
    else:
        assert source.count(selected) == 1
        assert source.count("prepare_owned_guard_host_lane") == 1
        assert "const bool guard_observed_" in source
    assert source.count("ctx.min_component(") == 1
    if checked:
        assert source.count("ctx.pointwise_status_max(") == 1
    else:
        assert source.count("ctx.norm_inf_all(") == 1
