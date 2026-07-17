"""Regression coverage for live ``dt`` in install-time matrix-free ApplyFns."""

from __future__ import annotations

from fractions import Fraction
import re

from pops.codegen.program_codegen import emit_cpp_program
from pops.linalg import LinearProblem
from pops.model import StateSpace
from pops.solvers.krylov import Richardson
from pops.time import FailRun, Program
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
        LinearProblem(operator, rhs),
        solver=Richardson(max_iter=4, rel_tol=Fraction(1, 10**8)),
    ).consume(action=FailRun())
    program.commit(temporal.next, solution)
    return program


def test_matrix_free_apply_reads_live_step_dt_before_krylov_and_preserves_ledger_terms():
    source = emit_cpp_program(_matrix_free_program())
    match = re.search(
        r"auto (apply_dt\d+) = std::make_shared<pops::Real>\("
        r"static_cast<pops::Real>\(0\)\);",
        source,
    )
    assert match is not None
    dt_capture = match.group(1)

    lambda_start = source.index("pops::ApplyFn apply_A")
    install_start = source.index("ctx.install([=](double dt)")
    step_refresh = source.index(
        f"*{dt_capture} = static_cast<pops::Real>(dt);", install_start)
    krylov_call = source.index("ctx.solve_linear_matfree(", step_refresh)

    assert match.start() < lambda_start < install_start < step_refresh < krylov_call
    assert dt_capture in source[lambda_start:source.index("](", lambda_start)]
    assert f"const pops::Real dt = *{dt_capture};" in source[lambda_start:install_start]

    # The numerical coefficients and their exact AMR-ledger polynomials share the same live dt.
    assert ", dt, {{0, 3, 2}});" in source
    assert ", dt, {{1, -1, 1}, {2, 2, 3}});" in source
