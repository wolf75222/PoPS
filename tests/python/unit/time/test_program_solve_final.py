"""Final typed Program.solve contract: one verb, typed problems, executable solvers."""
from __future__ import annotations

import inspect

import pytest

import pops
from pops.ir.literals import scalar_data
from pops.linalg import LinearProblem
from pops.solvers import DenseLU, GMRES
from pops.solvers import preconditioners
from pops.time import FailRun, LocalLinear, Program


def _matrix_free():
    program = Program("typed-linear")
    operator = program.matrix_free_operator("identity", ncomp=1)
    program.set_apply(operator, lambda _program, _out, value: value)
    return program, operator, program.scalar_field("rhs")


def test_one_public_solve_verb_and_no_parallel_solve_verbs():
    assert tuple(inspect.signature(Program.solve).parameters) == (
        "self", "problem", "solver", "name")
    for legacy in (
        "solve_linear", "solve_local_linear", "solve_local_nonlinear", "solve_residual"):
        assert not hasattr(Program, legacy)


def test_krylov_descriptor_owns_every_algorithm_control_and_outcome_is_consumed():
    program, operator, rhs = _matrix_free()
    solver = GMRES(
        max_iter=17, rel_tol=1.0e-7, restart=9,
        preconditioner=preconditioners.Identity())
    solved = program.solve(
        LinearProblem(operator, rhs), solver=solver, name="linear",
    ).consume(action=FailRun())

    token = next(value for value in program._values if value.op == "solve_linear")
    assert token.attrs["method"] == "gmres"
    assert scalar_data(token.attrs["tol"]) == scalar_data(1.0e-7)
    assert token.attrs["max_iter"] == 17
    assert token.attrs["restart"] == 9
    assert token.attrs["preconditioner"] == "identity"
    assert token.attrs["problem_kind"] == "matrix_free_linear"
    assert "problem_identity" not in token.attrs
    assert solved.op == "solve_outcome_component"


def test_typed_linear_problem_lowers_to_the_real_native_krylov_route():
    model = pops.Model("linear-model")
    state = model.state("U", components=("u",))
    block = pops.Case("linear-case").block("fluid", model)
    program = Program("linear-codegen")
    temporal = program.state(block[state])
    operator = program.matrix_free_operator("identity", ncomp=1)
    program.set_apply(operator, lambda _program, _out, value: value)
    solved = program.solve(
        LinearProblem(operator, temporal.n, at=temporal.next.point),
        solver=GMRES(max_iter=5, restart=3), name="linear",
    ).consume(action=FailRun())
    program.commit(temporal.next, solved)

    source = program.emit_cpp_program()
    assert "ctx.solve_linear_matfree" in source


def test_problem_is_only_algebra_and_solver_rejects_option_bags():
    assert tuple(inspect.signature(LinearProblem).parameters) == (
        "operator", "rhs", "initial_guess", "at", "scope")
    assert "method" not in inspect.signature(LinearProblem).parameters
    with pytest.raises(TypeError):
        GMRES(max_iter=4, undocumented=True)


def test_dense_lu_is_executable_only_for_local_linear_problem():
    program, operator, rhs = _matrix_free()
    with pytest.raises((TypeError, ValueError), match="LocalLinear"):
        program.solve(LinearProblem(operator, rhs), solver=DenseLU())


def test_final_catalogs_do_not_publish_unavailable_placeholders():
    import pops.solvers as solvers
    from pops.fields.catalog import fields
    from pops.numerics.projections import projections
    from pops.numerics.reconstruction.limiters import limiters

    assert not hasattr(fields, "Poisson")
    assert not hasattr(fields, "Helmholtz")
    assert not hasattr(fields, "EllipticSolve")
    assert not hasattr(projections, "bound_preserving")
    assert not hasattr(limiters, "MC")
    assert not hasattr(limiters, "Superbee")
    assert not hasattr(preconditioners, "Jacobi")
    assert not hasattr(preconditioners, "BlockJacobi")
    assert not hasattr(solvers, "Schur")
    assert not hasattr(solvers, "CondensedSchur")
