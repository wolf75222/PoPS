"""Final typed Program.solve contract: one verb, typed problems, executable solvers."""
from __future__ import annotations
from pops.codegen.program_codegen import emit_cpp_program

from dataclasses import replace
import inspect

import pytest

import pops
from pops._ir.literals import PREPARED_GMRES_MAX_RESTART, scalar_data
from pops.fields import ConstantNullspace, MeanValueGauge
from pops.linalg import LinearOperatorProperties, LinearProblem
from pops.solvers import CG, DenseLU, GMRES
from pops.solvers import preconditioners
from pops.time import FailRun, Program


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
        LinearProblem(operator, rhs, nullspace=None), solver=solver, name="linear",
    ).consume(action=FailRun())

    token = next(value for value in program._values if value.op == "solve_linear")
    assert token.attrs["method"] == "gmres"
    assert scalar_data(token.attrs["tol"]) == scalar_data(1.0e-7)
    assert token.attrs["max_iter"] == 17
    assert token.attrs["restart"] == 9
    assert token.attrs["preconditioner"] == "identity"
    assert token.attrs["nullspace_contract"] == {
        "schema_version": 1, "kind": "none"}
    assert token.attrs["gauge_contract"] == {
        "schema_version": 1, "kind": "none"}
    assert token.attrs["problem_kind"] == "matrix_free_linear"
    assert "problem_identity" not in token.attrs
    assert solved.op == "solve_outcome_component"


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"max_iterations": 1 << 31}, r"signed C\+\+ int"),
        (
            {"restart": PREPARED_GMRES_MAX_RESTART + 1},
            "native batched robust-dot collective capacity",
        ),
    ],
)
def test_program_reauthenticates_prepared_krylov_integer_capacity(overrides, error):
    program, operator, rhs = _matrix_free()
    prepared = replace(
        GMRES(max_iter=2, restart=2).prepare_program_solve(), **overrides)

    class ForgedDescriptor:
        def prepare_program_solve(self):
            return prepared

    with pytest.raises(ValueError, match=error):
        program.solve(
            LinearProblem(operator, rhs, nullspace=None), solver=ForgedDescriptor())


def test_codegen_rejects_forged_krylov_integers_before_emission_or_allocation():
    from pops.codegen.krylov_contract import validated_krylov_footprint

    program, operator, rhs = _matrix_free()
    program.solve(
        LinearProblem(operator, rhs, nullspace=None),
        solver=GMRES(max_iter=2, restart=2),
    ).consume(action=FailRun())
    solve = next(value for value in program._values if value.op == "solve_linear")
    authenticated_operator = solve.inputs[0]

    max_iter_attrs = dict(solve.attrs)
    max_iter_attrs["max_iter"] = 1 << 31
    with pytest.raises(ValueError, match="max_iter"):
        validated_krylov_footprint(max_iter_attrs, operator=authenticated_operator)

    restart_attrs = dict(solve.attrs)
    restart_attrs["restart"] = PREPARED_GMRES_MAX_RESTART + 1
    restart_footprint = dict(restart_attrs["krylov_footprint"])
    restart_footprint["restart"] = PREPARED_GMRES_MAX_RESTART + 1
    restart_attrs["krylov_footprint"] = restart_footprint
    with pytest.raises(ValueError, match="restart"):
        validated_krylov_footprint(restart_attrs, operator=authenticated_operator)


def test_typed_linear_problem_lowers_to_the_real_native_krylov_route():
    model = pops.Model("linear-model")
    state = model.state("U", components=("u",))
    block = pops.Case("linear-case").block("fluid", model)
    program = Program("linear-codegen")
    temporal = program.state(block[state])
    operator = program.matrix_free_operator("identity", ncomp=1)
    program.set_apply(operator, lambda _program, _out, value: value)
    solved = program.solve(
        LinearProblem(operator, temporal.n, at=temporal.next.point, nullspace=None),
        solver=GMRES(max_iter=5, restart=3), name="linear",
    ).consume(action=FailRun())
    program.commit(temporal.next, solved)

    source = emit_cpp_program(program)
    assert "ctx.solve_prepared_linear" in source
    assert "pops::PreparedAffineLinearProblem" in source


def test_problem_is_only_algebra_and_solver_rejects_option_bags():
    assert tuple(inspect.signature(LinearProblem).parameters) == (
        "operator", "rhs", "initial_guess", "at", "scope", "properties",
        "nullspace", "gauge")
    _program, operator, rhs = _matrix_free()
    with pytest.raises(TypeError, match="nullspace"):
        LinearProblem(operator, rhs)
    assert (
        LinearProblem(operator, rhs, nullspace=None).properties
        == LinearOperatorProperties.general()
    )
    assert "method" not in inspect.signature(LinearProblem).parameters
    with pytest.raises(TypeError):
        GMRES(max_iter=4, undocumented=True)


def test_cg_requires_an_explicit_spd_certificate():
    program, operator, rhs = _matrix_free()
    with pytest.raises(ValueError, match="CG requires"):
        program.solve(
            LinearProblem(operator, rhs, nullspace=None), solver=CG(max_iter=4))
    outcome = program.solve(
        LinearProblem(
            operator, rhs,
            properties=LinearOperatorProperties.symmetric_positive_definite(),
            nullspace=None),
        solver=CG(max_iter=4))
    assert outcome.consume(action=FailRun()).op == "solve_outcome_component"

    with pytest.raises(ValueError, match="positive_definite"):
        LinearOperatorProperties(symmetric=False, positive_definite=True)


def test_constant_nullspace_cg_requires_spd_on_the_complement_and_is_scalar_only():
    program, operator, rhs = _matrix_free()
    with pytest.raises(ValueError, match="on_nullspace_complement"):
        program.solve(
            LinearProblem(
                operator, rhs,
                properties=LinearOperatorProperties.symmetric_operator(),
                nullspace=ConstantNullspace(), gauge=MeanValueGauge(0)),
            solver=CG(max_iter=4),
        )
    outcome = program.solve(
        LinearProblem(
            operator,
            rhs,
            properties=(
                LinearOperatorProperties
                .symmetric_positive_definite_on_nullspace_complement()
            ),
            nullspace=ConstantNullspace(),
            gauge=MeanValueGauge(0),
        ),
        solver=CG(max_iter=4),
    )
    assert outcome.consume(action=FailRun()).op == "solve_outcome_component"
    token = next(value for value in program._values if value.op == "solve_linear")
    assert token.attrs["nullspace_contract"] == {
        "schema_version": 1, "kind": "constant"}
    assert token.attrs["gauge_contract"]["kind"] == "mean_value"

    vector_program = Program("vector-nullspace-rejected")
    vector_operator = vector_program.matrix_free_operator(
        "vector", domain="vector", range_="vector", ncomp=2)
    vector_program.set_apply(
        vector_operator, lambda _program, _out, value: value)
    with pytest.raises(ValueError, match="scalar-only"):
        vector_program.solve(
            LinearProblem(
                vector_operator,
                vector_program.scalar_field("rhs", ncomp=2),
                properties=LinearOperatorProperties.symmetric_operator(),
                nullspace=ConstantNullspace(),
                gauge=MeanValueGauge(0),
            ),
            solver=GMRES(max_iter=4, restart=2),
        )


def test_constant_nullspace_refuses_uncertified_geometric_mg_preconditioner():
    program, operator, rhs = _matrix_free()
    with pytest.raises(NotImplementedError, match="no explicit public certificate"):
        program.solve(
            LinearProblem(
                operator, rhs,
                properties=LinearOperatorProperties.symmetric_operator(),
                nullspace=ConstantNullspace(), gauge=MeanValueGauge(0)),
            solver=GMRES(
                max_iter=4,
                restart=2,
                preconditioner=preconditioners.GeometricMG(),
            ),
        )


def test_dense_lu_is_executable_only_for_local_linear_problem():
    program, operator, rhs = _matrix_free()
    with pytest.raises((TypeError, ValueError), match="LocalLinear"):
        program.solve(LinearProblem(operator, rhs, nullspace=None), solver=DenseLU())


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
