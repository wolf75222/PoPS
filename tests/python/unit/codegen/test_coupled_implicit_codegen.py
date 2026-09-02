"""Fail-closed generic multi-block implicit Program lowering (ADC-690)."""
from __future__ import annotations
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_persistent_plan import (
    get_program_resource_plan,
    persistent_slot_token,
)

import pytest

from typed_program_support import typed_state

time = pytest.importorskip("pops.time")
from pops import model  # noqa: E402
from pops._ir.expr import Const  # noqa: E402
from pops.codegen.program_emit_ops import _append_local_nonlinear_report  # noqa: E402
from pops.solvers.nonlinear import LocalNewton  # noqa: E402


def _program(*, consume=True, coefficient=1, action=None):
    module = model.Module("implicit_exchange")
    electrons = module.state_space("electron_state", ("ne", "pex", "pey"))
    ions = module.state_space("ion_state", ("ni", "pix", "piy"))
    bundle = model.RateBundle({
        "electrons": model.Rate(electrons),
        "ions": model.Rate(ions),
    })
    ne, pex, pey = module.state_symbols(electrons)
    ni, pix, piy = module.state_symbols(ions)
    collision = module.operator(
        name="collision",
        signature=model.Signature((electrons, ions), bundle),
        kind="coupled_rate",
        expr={
            "electrons": [ni - ne, pix - pex, piy - pey],
            "ions": [ne - ni, pex - pix, pey - piy],
        },
    )
    program = time.Program("implicit_collision")
    electron_handle = module.state_handle(electrons)
    ion_handle = module.state_handle(ions)
    electron_n = typed_state(
        program, "electrons", space=electrons, model=module, state=electron_handle)
    ion_n = typed_state(program, "ions", space=ions, model=module, state=ion_handle)
    outcome = program.solve(
        time.CoupledImplicitEuler(
            collision, (electron_n, ion_n), coefficient=coefficient),
        solver=LocalNewton(
            tolerance=1.0e-11, max_iterations=12, finite_difference_step=1.0e-6),
        name="collision_step")
    electron_next = typed_state(
        program, "electrons", state_name=electrons.name, space=electrons,
        model=module, state=electron_handle).next
    ion_next = typed_state(
        program, "ions", state_name=ions.name, space=ions,
        model=module, state=ion_handle).next
    if not consume:
        program.commit_many({
            electron_next: program.value("electron_identity", electron_n, at=electron_next.point),
            ion_next: program.value("ion_identity", ion_n, at=ion_next.point),
        })
        return module, program, outcome
    solved = outcome.consume(action=time.RejectAttempt() if action is None else action)
    program.commit_many({
        electron_next: solved[electron_n.block],
        ion_next: solved[ion_n.block],
    })
    return module, program, solved


def _coupled_implicit_resource_slot(program):
    """Resolve the solve-owned dense slot used by its prepared scratch family."""

    solve = next(value for value in program._values if value.op == "solve_coupled_implicit")
    plan = get_program_resource_plan(program, target="system")
    slot = persistent_slot_token(program, solve, target="system")
    entry = next(entry for entry in plan.entries if entry.slot == int(slot))
    assert entry.key.value_id == solve.id
    return slot


def test_coupled_implicit_uses_one_prepared_provider_with_explicit_action():
    _module, program, solved = _program()
    source = emit_cpp_program(program, model=None)
    slot = _coupled_implicit_resource_slot(program)

    assert len(solved) == 2
    assert source.count("pops::for_each_cell") == 1
    assert "pops::prepare_local_nonlinear_problem<6>" in source
    assert "pops::solve_prepared_local_nonlinear(prepared_, G_)" in source
    assert "pops::detail::mat_inverse<6>" not in source
    assert "Jinv_" not in source
    assert "for (int it_ =" not in source
    assert "Ueval[0] - G_[0] - static_cast<pops::Real>(pops::Real(1)) * dt *" in source
    scalar_prepare = "ctx.prepare_scalar_scratch(%s, 0, 0, 11, 0);" % slot
    scalar_access = "ctx.scalar_scratch(%s, 0, u0, 11, 0)" % slot
    assert scalar_prepare in source
    assert scalar_access in source
    assert source.index(scalar_prepare) < source.index(scalar_access)
    assert "encode_ranked_local_nonlinear_failure" not in source
    assert "const pops::ExecutionLane& ci_report_2_lane = ctx.prepared_execution_lane();" in source
    assert (
        "pops::all_reduce_max(pops::reduce_max_local(ci_status_2, 10), "
        "ci_report_2_lane)" in source
    )
    assert source.count("ctx.sum_component(0, ci_status_2, 9)") == 1
    assert "pops::all_reduce_sum(pops::reduce_sum_local(ci_status_2, 9)" not in source
    assert "pops::reduce_sum_local(ci_status_2, 9)" not in source
    assert "pops::local_nonlinear_status_priority(solved_.status)" in source
    assert "pops::local_nonlinear_status_from_priority(" in source
    assert (
        "pops::collective_first_local_nonlinear_failure("
        "ci_status_2, ci_report_2_priority, 10, 8, ci_report_2_lane)" in source
    )
    assert "collective status/location precedence mismatch" in source
    assert "pops::SolveAction::kRejectAttempt" in source
    assert "SolveStatus::kSingular" in source
    assert "program_reject_step(ctx," in source
    assert "throw pops::runtime::program::StepAttemptRejected" not in source
    assert source.count("ctx.commit_many(") == 1
    assert "{&u0, &ci2_electrons}" in source
    assert "{&u1, &ci2_ions}" in source


def test_coupled_reject_attempt_codegen_filters_selected_statuses_and_fails_closed():
    _module, program, _solved = _program(
        action=time.RejectAttempt(statuses=("iteration_limit",)))
    source = emit_cpp_program(program, model=None)
    start = source.index("if (!ci_outcome_")
    end = source.index(".action_name()", start)
    guard = source[start:end]

    assert "SolveStatus::kIterationLimit" in guard
    assert "SolveStatus::kSingular" not in guard
    assert "SolveStatus::kInvalidEvaluation" not in guard
    assert ".action == pops::SolveAction::kRejectAttempt" in guard
    assert "program_reject_step(ctx," in guard
    assert "throw pops::runtime::program::StepAttemptRejected" not in guard


def test_coupled_implicit_euler_carries_exact_stage_coefficient_and_typed_problem_kind():
    _module, program, _solved = _program(coefficient=0.5)
    token = next(value for value in program._values
                 if value.op == "solve_coupled_implicit")
    source = emit_cpp_program(program, model=None)

    assert token.attrs["problem_kind"] == "coupled_implicit_euler"
    assert "problem_identity" not in token.attrs
    assert token.attrs["coefficient"] == 0.5
    assert "static_cast<pops::Real>(0.5) * dt *" in source


def test_unconsumed_coupled_implicit_is_rejected_by_graph_validation_and_codegen():
    _module, program, outcome = _program(consume=False)
    with pytest.raises(ValueError, match="consumed exactly once"):
        program.to_graph()
    with pytest.raises(ValueError, match="must have exactly one explicit"):
        emit_cpp_program(program, model=None)
    with pytest.raises(TypeError, match="not readable"):
        _ = outcome.token


def test_failed_coupled_implicit_never_aliases_live_state_before_guard():
    _module, program, _solved = _program()
    source = emit_cpp_program(program, model=None)
    slot = _coupled_implicit_resource_slot(program)
    guard = source.index("if (!ci_outcome_")
    first_commit = source.index("ctx.commit_many(", guard)

    for subslot, block, state in ((0, 0, "u0"), (1, 1, "u1")):
        prepare = "ctx.prepare_state_scratch(%s, %d, %d);" % (slot, subslot, block)
        access = "ctx.scratch_state(%s, %d, %s)" % (slot, subslot, state)
        assert prepare in source
        assert access in source
        assert source.index(prepare) < source.index(access) < guard
    assert "ctx.commit_many(" not in source[:guard]
    assert guard < first_commit


def test_nonlinear_status_reduction_requires_an_authenticated_owner():
    _module, program, _solved = _program()
    solve = next(value for value in program._values if value.op == "solve_coupled_implicit")

    with pytest.raises(ValueError, match="exact authenticated owner index"):
        _append_local_nonlinear_report(
            program,
            solve,
            "ci_status",
            "ci_report",
            [],
            owner_index=None,
        )


def test_same_component_names_are_qualified_by_state_space():
    module = model.Module("overlapping_components")
    electrons = module.state_space("electrons", ("density",))
    ions = module.state_space("ions", ("density",))
    (electron_density,) = module.state_symbols(electrons)
    (ion_density,) = module.state_symbols(ions)
    exchange = module.operator(
        name="exchange",
        signature=model.Signature(
            (electrons, ions),
            model.RateBundle({"electrons": model.Rate(electrons),
                              "ions": model.Rate(ions)}),
        ),
        kind="coupled_rate",
        expr={
            "electrons": [ion_density - electron_density],
            "ions": [electron_density - ion_density],
        },
    )
    program = time.Program("overlapping_components")
    electron_n = typed_state(
        program, "electrons", space=electrons, model=module,
        state=module.state_handle(electrons))
    ion_n = typed_state(
        program, "ions", space=ions, model=module, state=module.state_handle(ions))
    solved = program.solve(
        time.CoupledImplicitEuler(exchange, (electron_n, ion_n)),
        solver=LocalNewton()).consume(
        action=time.RejectAttempt())
    electron_next = typed_state(
        program, "electrons", state_name=electrons.name, space=electrons,
        model=module, state=module.state_handle(electrons)).next
    ion_next = typed_state(
        program, "ions", state_name=ions.name, space=ions,
        model=module, state=module.state_handle(ions)).next
    program.commit_many({
        electron_next: solved[electron_n.block],
        ion_next: solved[ion_n.block],
    })

    source = emit_cpp_program(program, model=None)
    assert electron_density.name in source
    assert ion_density.name in source
    assert electron_density.name != ion_density.name


def test_dense_newton_dimension_follows_the_typed_rate_bundle():
    module = model.Module("seventeen_components")
    left = module.state_space("left", tuple("l%d" % index for index in range(9)))
    right = module.state_space("right", tuple("r%d" % index for index in range(8)))
    zero = Const(0)
    operator = module.operator(
        name="large_exchange",
        signature=model.Signature(
            (left, right),
            model.RateBundle({"left": model.Rate(left), "right": model.Rate(right)}),
        ),
        kind="coupled_rate",
        expr={"left": [zero] * 9, "right": [zero] * 8},
    )
    program = time.Program("seventeen_components")
    left_n = typed_state(
        program, "left", space=left, model=module, state=module.state_handle(left))
    right_n = typed_state(
        program, "right", space=right, model=module, state=module.state_handle(right))
    solved = program.solve(
        time.CoupledImplicitEuler(operator, (left_n, right_n)),
        solver=LocalNewton()).consume(
        action=time.RejectAttempt())
    left_next = typed_state(
        program, "left", state_name=left.name, space=left,
        model=module, state=module.state_handle(left)).next
    right_next = typed_state(
        program, "right", state_name=right.name, space=right,
        model=module, state=module.state_handle(right)).next
    program.commit_many({left_next: solved[left_n.block], right_next: solved[right_n.block]})

    source = emit_cpp_program(program, model=None)
    assert "pops::prepare_local_nonlinear_problem<17>" in source
    assert "pops::LocalNonlinearCellResult<17>" in source
    assert "pops::detail::mat_inverse<17>" not in source
