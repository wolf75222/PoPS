"""Regression coverage for live ``dt`` in install-time matrix-free ApplyFns."""

from __future__ import annotations

from fractions import Fraction
import re

import pytest

from pops.codegen.program_codegen import emit_cpp_program
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

    lambda_start = source.index("pops::ApplyFn apply_A")
    install_start = source.index("ctx.install([=](double dt)")
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
    install = source.index("ctx.install([=](double dt)")
    region = source.index(region_marker, install)
    solution_owners = [match.start() for match in re.finditer(r"auto sf_sol\d+ =", source)]
    solve_calls = [
        match.start() for match in re.finditer(r"ctx\.solve_prepared_linear\(", source)]

    assert len(solution_owners) == solve_count
    assert len(solve_calls) == solve_count
    assert all(owner < install for owner in solution_owners)
    assert all(call > region for call in solve_calls)
    assert source.count("ctx.operator_evaluation_snapshot(") == solve_count
    assert source.count(" action=fail_run") == solve_count

    if kind == "while_cond":
        assert solve_calls[0] < source.index("break;", region)
    elif kind == "while_body":
        assert source.index("break;", region) < solve_calls[0]
    elif kind == "branch":
        otherwise = source.index("} else {", source.index("if (", install))
        assert solve_calls[0] < otherwise < solve_calls[1]
