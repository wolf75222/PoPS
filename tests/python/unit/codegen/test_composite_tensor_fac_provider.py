"""Direct CompositeTensorFAC identity and authoring contract."""

from __future__ import annotations

from fractions import Fraction

import pytest

from pops._ir.literals import scalar_data
from pops.linalg import LinearProblem
from pops.solvers import CG, CompositeTensorFAC, Hierarchy, solvers
from pops.time import Program


@pytest.mark.parametrize(
    "option",
    [
        {"max_iter": True},
        {"max_iter": 0},
        {"max_iter": 1.5},
        {"rel_tol": True},
        {"rel_tol": 0},
        {"rel_tol": 1},
        {"rel_tol": float("nan")},
        {"fine_sweeps": True},
        {"fine_sweeps": 1.5},
        {"fine_sweeps": 0},
        {"coarse_cycles": False},
        {"coarse_cycles": "4"},
        {"coarse_cycles": -1},
        {"coarse_rel_tol": True},
        {"coarse_rel_tol": 0},
        {"coarse_rel_tol": 1},
        {"coarse_rel_tol": float("nan")},
        {"coarse_rel_tol": float("inf")},
        {"verbose": 0},
        {"verbose": "true"},
    ],
)
def test_solver_options_are_strict(option):
    with pytest.raises((TypeError, ValueError)):
        CompositeTensorFAC(**option)


def test_identity_owns_complete_flat_and_refined_solve_contract():
    default = CompositeTensorFAC()
    assert solvers.CompositeTensorFAC is CompositeTensorFAC
    assert default.max_iter == 30
    assert default.rel_tol == 1.0e-9
    assert default.fine_sweeps is None
    assert default.coarse_rel_tol is None
    assert default.coarse_cycles is None
    assert default.verbose is None

    configured = CompositeTensorFAC(
        max_iter=23,
        rel_tol=Fraction(3, 100_000_000),
        fine_sweeps=7,
        coarse_rel_tol=Fraction(1, 8),
        coarse_cycles=9,
        verbose=False,
    )
    identity = configured.canonical_identity()
    assert identity == {
        "schema_version": 1,
        "solver_id": "composite_tensor_fac",
        "capabilities": [
            "amr_hierarchy",
            "flat_bicgstab",
            "scalar",
            "tensor_elliptic",
        ],
        "options": {
            "max_iter": 23,
            "rel_tol": scalar_data(Fraction(3, 100_000_000)),
            "fine_sweeps": 7,
            "coarse_rel_tol": scalar_data(Fraction(1, 8)),
            "coarse_cycles": 9,
            "verbose": False,
        },
    }
    assert configured.identity != default.identity
    prepared = configured.prepare_program_solve()
    assert prepared.identity_data == identity
    assert prepared.identity.token == configured.identity.token


def test_krylov_descriptor_rejects_hierarchy_scope_before_codegen():
    program = Program("krylov-hierarchy-rejected")
    operator = program.matrix_free_operator("operator", scope=Hierarchy())
    rhs = program.scalar_field("rhs")
    problem = LinearProblem(operator, rhs, scope=Hierarchy())

    with pytest.raises(TypeError, match="CompositeTensorFAC.*Krylov descriptors solve Level"):
        program.solve(problem, solver=CG(max_iter=11, rel_tol=1.0e-6))


def test_hierarchy_operator_has_no_provider_slot_and_is_scalar_only():
    program = Program("direct-hierarchy-contract")
    with pytest.raises(TypeError, match="unexpected keyword argument 'provider'"):
        program.matrix_free_operator(
            "legacy", scope=Hierarchy(), provider=CompositeTensorFAC()  # type: ignore[call-arg]
        )
    with pytest.raises(ValueError, match="only a scalar operator with ncomp=1"):
        program.matrix_free_operator(
            "vector", domain="vector", range_="vector", ncomp=2, scope=Hierarchy()
        )


def test_hierarchy_apply_rejects_any_unproven_operator_shape():
    program = Program("unproven-hierarchy-apply")
    operator = program.matrix_free_operator("operator", scope=Hierarchy())
    with pytest.raises(ValueError, match="exactly one scalar scratch"):
        program.set_apply(operator, lambda _program, _out, value: value)
