"""Regression coverage for live ``dt`` in install-time matrix-free ApplyFns."""

from __future__ import annotations

from fractions import Fraction
import re
from types import SimpleNamespace

import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_solve import _matrix_free_consumer_block
from pops.linalg import LinearProblem
from pops.model import StateSpace
from pops.solvers.krylov import Richardson
from pops.time import FailRun, Program, SampleAndHold
from pops.time.points import Clock, TimePoint
from typed_program_support import typed_state


def _matrix_free_program() -> Program:
    program = Program("matrix_free_live_dt")
    temporal = typed_state(
        program,
        "transport",
        state_name="U",
        space=StateSpace("U", ("u",)),
    )
    operator = program.matrix_free_operator("A")

    def apply(builder, _out, value):
        laplacian = builder.scalar_field("laplacian")
        builder.laplacian(laplacian, value)
        polynomial = -builder.dt + Fraction(2, 3) * builder.dt * builder.dt
        return Fraction(3, 2) * value + polynomial * laplacian

    program.set_apply(operator, apply)
    rhs = program.value("rhs", temporal.n, at=temporal.next.point)
    solution = program.solve(
        LinearProblem(operator, rhs, nullspace=None),
        solver=Richardson(max_iter=4, rel_tol=Fraction(1, 10**8)),
    ).consume(action=FailRun())
    program.commit(temporal.next, solution)
    return program


def _subcycled_matrix_free_program() -> Program:
    program = Program("matrix_free_child_dt")
    temporal = typed_state(
        program,
        "transport",
        state_name="U",
        space=StateSpace("U", ("u",)),
    )
    operator = program.matrix_free_operator("A", domain="state", range_="state", ncomp=1)

    def apply(builder, _out, value):
        laplacian = builder.scalar_field("laplacian")
        builder.laplacian(laplacian, value)
        return value - builder.dt * laplacian

    program.set_apply(operator, apply)
    fast = Clock("fast", owner=program.owner_path)
    child = program.synchronize(
        temporal.n,
        at=TimePoint(fast),
        relation=SampleAndHold(),
        name="to_fast",
    )

    def child_tick(builder, value):
        return builder.solve(
            LinearProblem(operator, value, nullspace=None),
            solver=Richardson(max_iter=4, rel_tol=Fraction(1, 10**8)),
        ).consume(action=FailRun())

    advanced = program.subcycle(
        child,
        clock=fast,
        within=program.clock,
        count=2,
        body_fn=child_tick,
        name="two_fast_ticks",
    )
    returned = program.synchronize(
        advanced,
        at=temporal.next.point,
        relation=SampleAndHold(),
        name="to_macro",
    )
    program.commit(temporal.next, returned)
    return program


def _structured_region_krylov_program(kind: str) -> Program:
    """Place prepared solves in one exact structured region without changing their contract."""
    program = Program("matrix_free_%s" % kind)
    temporal = typed_state(
        program,
        "transport",
        state_name="U",
        space=StateSpace("U", ("u",)),
    )
    operator = program.matrix_free_operator(
        "A", domain="state", range_="state", ncomp=1)

    def apply(builder, _out, value):
        laplacian = builder.scalar_field("laplacian")
        builder.laplacian(laplacian, value)
        return value - builder.dt * laplacian

    def solve(builder, value):
        return builder.solve(
            LinearProblem(operator, value, nullspace=None),
            solver=Richardson(max_iter=4, rel_tol=Fraction(1, 10**8)),
        ).consume(action=FailRun())

    program.set_apply(operator, apply)
    if kind == "while_cond":
        advanced = program.while_(
            temporal.n,
            lambda builder, value: builder.norm2(solve(builder, value)) > 0,
            lambda _builder, value: value,
        )
    elif kind == "while_body":
        advanced = program.while_(
            temporal.n,
            lambda builder, value: builder.norm2(value) > 0,
            solve,
        )
    elif kind == "range":
        advanced = program.range(temporal.n, 2, solve)
    elif kind == "branch":
        condition = program.norm2(temporal.n) > 0
        advanced = program.branch(
            condition,
            lambda builder: solve(builder, temporal.n),
            lambda builder: solve(builder, temporal.n),
        )
    else:  # pragma: no cover - test helper contract
        raise ValueError("unknown structured region %r" % kind)
    program.commit(
        temporal.next,
        program.value("advanced", advanced, at=temporal.next.point),
    )
    return program


def test_matrix_free_apply_reads_live_step_dt_before_prepared_krylov():
    source = emit_cpp_program(_matrix_free_program())
    match = re.search(
        r"auto (apply_dt\d+) = std::make_shared<pops::Real>\("
        r"static_cast<pops::Real>\(0\)\);",
        source,
    )
    assert match is not None
    dt_capture = match.group(1)
    operator_match = re.search(
        r"auto (operator_dt\d+) = std::make_shared<pops::Real>\("
        r"static_cast<pops::Real>\(0\)\);",
        source,
    )
    assert operator_match is not None
    operator_dt = operator_match.group(1)

    factory_start = source.index(
        "pops::PreparedAffineOperatorSessionFactory<pops::kNativeDimension> make_apply_A"
    )
    lambda_start = source.index("pops::ApplyFn<pops::kNativeDimension> apply =", factory_start)
    install_start = source.index("state->step = [ctx_owner = state->ctx_owner](double dt)")
    operator_refresh = source.index(
        f"*{operator_dt} = static_cast<pops::Real>(dt);", install_start)
    apply_refresh = source.index(
        f"*{dt_capture} = static_cast<pops::Real>(dt);", install_start)
    prepare_call = source.index("->prepare(", apply_refresh)
    krylov_call = source.index("ctx.solve_prepared_linear(", prepare_call)

    assert (
        match.start() < lambda_start < install_start
        < operator_refresh < apply_refresh < prepare_call < krylov_call
    )
    assert dt_capture in source[lambda_start:source.index("](", lambda_start)]
    assert operator_dt in source[lambda_start:source.index("](", lambda_start)]
    assert f"const pops::Real dt = *{dt_capture};" in source[lambda_start:install_start]

    # The prepared affine operator retains the exact authored coefficients and reads the step's
    # refreshed operator dt rather than freezing a compile-time or prepare-time scalar.
    assert "static_cast<pops::Real>((pops::Real(3) / pops::Real(2)))" in source
    assert (
        "pops::Real(-1) * (*%s) + (pops::Real(2) / pops::Real(3)) * (*%s) * (*%s)"
        % (operator_dt, operator_dt, operator_dt)
    ) in source[lambda_start:install_start]


def test_matrix_free_templates_are_plan_primed_before_step_and_never_allocate_in_apply():
    source = emit_cpp_program(_matrix_free_program())
    begin_step = source.index("state->step = [ctx_owner = state->ctx_owner](double dt)")

    assert "ctx.alloc_scalar_field(" not in source
    assert source.count("ctx.prepare_scalar_scratch(") == 3
    assert source.count("ctx.scalar_scratch(") == 3
    assert source.index("ctx.prepare_scalar_scratch(") < begin_step
    assert source.index("ctx.scalar_scratch(") < begin_step


def test_amr_matrix_free_sessions_retain_prepared_handles_without_candidate_copies():
    """A requalified AMR level bundle owns handles, not copied active-level fabs."""

    source = emit_cpp_program(_matrix_free_program(), target="amr_system")
    factory = source.split("auto _make_level_program", 1)[1].split(
        "struct _PopsAmrLevelProgramAuthority", 1
    )[0]

    assert factory.count("ctx.prepare_scalar_scratch(") == 3
    assert factory.count("ctx.prepared_scalar_scratch_handle(") == 3
    assert "std::make_shared<pops::MultiFab<pops::kNativeDimension>>(*" not in source
    assert "const std::size_t session_field_count = std::size_t{0};" in factory
    assert "pops::PreparedOperatorConcurrency::Exclusive" in factory
    assert factory.index("ctx.prepare_scalar_scratch(") < factory.index(
        "ctx.prepared_scalar_scratch_handle("
    )
    # Each level closure executes through the retained live provider.  A forward overlay is a
    # cold-only preparation authority: capturing it in the hot closure would leave current_dt and
    # the accepted facade on different service objects after publication.
    assert "auto _make_level_program = [](auto ctx_owner, auto& ctx, bool prepare_resources)" in source
    assert "if (!ctx_owner)" in factory
    assert "auto& ctx = *ctx_owner;" in factory
    assert "auto _make_level_program = [](auto& ctx, bool prepare_resources)" not in source
    assert "_make_level_program(ctx_owner, ctx, prepare_resources)" in source
    assert "_make_level_program(ctx_owner, *owner, true)" in source
    assert "_make_level_program(owner, *owner, true)" not in source


def test_amr_level_authority_is_one_rollbackable_snapshot_bundle():
    source = emit_cpp_program(_matrix_free_program(), target="amr_system")

    assert "struct _PopsAmrLevelProgramAuthoritySlot" in source
    assert "_PopsAcceptedProgramExecutionServicesSnapshot" in source
    assert "prepare_forward_execution_bundle" in source
    assert "with_forward_execution_overlay" in source
    assert (
        "void refresh_from_owner_preallocated() override { "
        "inner_->refresh_from_owner_preallocated(); }"
        in source
    )
    assert "swap(slot_->active, staged_)" in source
    assert "AMR Program level authority is absent or stale" in source
    forward_bundle = source.split(
        "void prepare_forward_execution_bundle", 1
    )[1].split("prepare_forward_accepted_context", 1)[0]
    assert forward_bundle.index("inner_->prepare_forward_execution_bundle(erased)") < (
        forward_bundle.index("owner_->with_forward_execution_overlay")
    )


def test_uniform_matrix_free_sessions_keep_independent_private_templates():
    source = emit_cpp_program(_matrix_free_program())
    assert "pops::PreparedOperatorConcurrency::Independent" in source
    assert "pops::PreparedOperatorConcurrency::Exclusive" not in source


def test_matrix_free_owner_inference_refuses_conflicting_solve_consumers():
    operator = SimpleNamespace(
        op="matrix_free_operator", block=None, attrs={}, inputs=(), name="A")
    first = SimpleNamespace(
        op="solve_linear", inputs=(operator,), block=object(), attrs={}, name="left")
    second = SimpleNamespace(
        op="solve_linear", inputs=(operator,), block=object(), attrs={}, name="right")
    program = SimpleNamespace(_values=(operator, first, second))

    with pytest.raises(ValueError, match="conflicting owner blocks"):
        _matrix_free_consumer_block(program, operator, where="test matrix-free owner")


def test_subcycle_refreshes_both_matrix_free_dt_captures_inside_each_child_tick():
    source = emit_cpp_program(_subcycled_matrix_free_program())
    operator_dt = re.search(r"auto (operator_dt\d+) =", source)
    apply_dt = re.search(r"auto (apply_dt\d+) =", source)
    assert operator_dt is not None and apply_dt is not None

    loop_start = source.index("for (int i")
    logical_scope = source.index("ctx.logical_evaluation_scope(", loop_start)
    child_dt = source.index(".dt();", logical_scope)
    operator_refresh = source.index(
        f"*{operator_dt.group(1)} = static_cast<pops::Real>(dt);", child_dt)
    apply_refresh = source.index(
        f"*{apply_dt.group(1)} = static_cast<pops::Real>(dt);", child_dt)
    snapshot = source.index("ctx.operator_evaluation_snapshot(", apply_refresh)
    prepare = source.index("->prepare(", snapshot)
    loop_end = source.index(".finish();", prepare)

    assert loop_start < logical_scope < child_dt < operator_refresh < apply_refresh < snapshot \
        < prepare < loop_end
    assert source.count(
        f"*{operator_dt.group(1)} = static_cast<pops::Real>(dt);") == 1
    assert source.count(
        f"*{apply_dt.group(1)} = static_cast<pops::Real>(dt);") == 1


@pytest.mark.parametrize(
    ("kind", "region_marker", "solve_count"),
    (
        ("while_cond", "for (;;)", 1),
        ("while_body", "for (;;)", 1),
        ("range", "for (int i", 1),
        ("branch", "if ((", 2),
    ),
)
def test_structured_regions_hoist_prepared_storage_and_emit_solve_in_region(
        kind, region_marker, solve_count):
    source = emit_cpp_program(_structured_region_krylov_program(kind))
    install = source.index("state->step = [ctx_owner = state->ctx_owner](double dt)")
    region = source.index(region_marker, install)
    solution_owners = [match.start() for match in re.finditer(r"auto sf_sol\d+ =", source)]
    solve_calls = [
        match.start() for match in re.finditer(r"ctx\.solve_prepared_linear\(", source)]

    assert len(solution_owners) == solve_count
    assert len(solve_calls) == solve_count
    assert all(owner < install for owner in solution_owners)
    assert all(call > region for call in solve_calls)
    assert source.count("ctx.operator_evaluation_snapshot(") == solve_count
    assert source.count('" action=" +') == solve_count
    assert source.count(".action_name()") == solve_count

    if kind == "while_cond":
        assert solve_calls[0] < source.index("break;", region)
    elif kind == "while_body":
        assert source.index("break;", region) < solve_calls[0]
    elif kind == "branch":
        otherwise = source.index("} else {", source.index("if (", install))
        assert solve_calls[0] < otherwise < solve_calls[1]
