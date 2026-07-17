"""Direct CompositeTensorFAC identity and authoring contract."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import pytest

from pops._ir.literals import scalar_data, scalar_literal
from pops.fields import ConstantNullspace, MeanValueGauge
from pops.linalg import LinearOperatorProperties, LinearProblem
from pops.solvers import CG, CompositeTensorFAC, Hierarchy, solvers
from pops.time import Program


@pytest.mark.parametrize(
    "option",
    [
        {"max_iter": True},
        {"max_iter": 0},
        {"max_iter": 1.5},
        {"max_iter": 1 << 31},
        {"rel_tol": True},
        {"rel_tol": 0},
        {"rel_tol": 1},
        {"rel_tol": float("nan")},
        {"abs_tol": True},
        {"abs_tol": -1},
        {"abs_tol": float("nan")},
        {"fine_sweeps": True},
        {"fine_sweeps": 1.5},
        {"fine_sweeps": 0},
        {"fine_sweeps": 1 << 31},
        {"coarse_cycles": False},
        {"coarse_cycles": "4"},
        {"coarse_cycles": -1},
        {"coarse_cycles": 1 << 31},
        {"coarse_rel_tol": True},
        {"coarse_rel_tol": 0},
        {"coarse_rel_tol": 1},
        {"coarse_rel_tol": float("nan")},
        {"coarse_rel_tol": float("inf")},
        {"coarse_abs_tol": True},
        {"coarse_abs_tol": -1},
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
    assert default.abs_tol == 0.0
    assert default.fine_sweeps is None
    assert default.coarse_rel_tol is None
    assert default.coarse_abs_tol is None
    assert default.coarse_cycles is None
    assert default.verbose is None

    configured = CompositeTensorFAC(
        max_iter=23,
        rel_tol=Fraction(3, 100_000_000),
        abs_tol=Fraction(1, 1_000_000_000_000),
        fine_sweeps=7,
        coarse_rel_tol=Fraction(1, 8),
        coarse_abs_tol=Fraction(1, 10_000_000_000_000),
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
            "abs_tol": scalar_data(Fraction(1, 1_000_000_000_000)),
            "fine_sweeps": 7,
            "coarse_rel_tol": scalar_data(Fraction(1, 8)),
            "coarse_abs_tol": scalar_data(Fraction(1, 10_000_000_000_000)),
            "coarse_cycles": 9,
            "verbose": False,
        },
    }
    assert configured.identity != default.identity
    prepared = configured.prepare_program_solve()
    assert prepared.identity_data == identity
    assert prepared.identity.token == configured.identity.token


def test_provider_integer_controls_accept_the_complete_native_int_range():
    cpp_int_max = (1 << 31) - 1
    configured = CompositeTensorFAC(
        max_iter=cpp_int_max,
        fine_sweeps=cpp_int_max,
        coarse_cycles=cpp_int_max,
    )
    assert configured.max_iter == cpp_int_max
    assert configured.fine_sweeps == cpp_int_max
    assert configured.coarse_cycles == cpp_int_max


@pytest.mark.parametrize("name", ["max_iter", "fine_sweeps", "coarse_cycles"])
def test_codegen_rejects_forged_composite_fac_integer_overflow(name):
    from pops.codegen.program_emit_solve import _composite_tensor_fac_options

    solver = CompositeTensorFAC()
    identity = deepcopy(solver.canonical_identity())
    identity["options"][name] = 1 << 31
    node = SimpleNamespace(attrs={
        "hierarchy_solver_identity": identity,
        "hierarchy_solver": "composite_tensor_fac",
        "max_iter": solver.max_iter,
        "tol": scalar_literal(solver.rel_tol),
        "abs_tol": scalar_literal(solver.abs_tol),
        "method": "bicgstab",
        "preconditioner": "identity",
        "restart": None,
        "ncomp": 1,
        "hierarchy_block_index": 0,
    })

    with pytest.raises(ValueError, match=name):
        _composite_tensor_fac_options(node)


def test_codegen_rejects_flat_absolute_tolerance_that_disagrees_with_provider_identity():
    from pops.codegen.program_emit_solve import _composite_tensor_fac_options

    solver = CompositeTensorFAC(abs_tol=Fraction(1, 10_000))
    node = SimpleNamespace(attrs={
        "hierarchy_solver_identity": solver.canonical_identity(),
        "hierarchy_solver": "composite_tensor_fac",
        "max_iter": solver.max_iter,
        "tol": scalar_literal(solver.rel_tol),
        "abs_tol": scalar_literal(0),
        "method": "bicgstab",
        "preconditioner": "identity",
        "restart": None,
        "ncomp": 1,
        "hierarchy_block_index": 0,
    })

    with pytest.raises(ValueError, match="convergence controls disagree"):
        _composite_tensor_fac_options(node)


def test_program_rejects_forged_composite_fac_negative_absolute_tolerance_before_codegen():
    from test_hierarchy_scoped_solve_emit import _build

    prepared = replace(
        CompositeTensorFAC().prepare_program_solve(), absolute_tolerance=Fraction(-1, 10)
    )

    class ForgedDescriptor:
        def prepare_program_solve(self):
            return prepared

    with pytest.raises(ValueError, match="CompositeTensorFAC abs_tol"):
        _build(ForgedDescriptor())


def test_krylov_descriptor_rejects_hierarchy_scope_before_codegen():
    program = Program("krylov-hierarchy-rejected")
    operator = program.matrix_free_operator("operator", scope=Hierarchy())
    rhs = program.scalar_field("rhs")
    problem = LinearProblem(operator, rhs, scope=Hierarchy(), nullspace=None)

    with pytest.raises(TypeError, match="CompositeTensorFAC.*Krylov descriptors solve Level"):
        program.solve(problem, solver=CG(max_iter=11, rel_tol=1.0e-6))


def test_composite_provider_refuses_constant_nullspace_until_multilevel_gauge_is_wired():
    program = Program("composite-nullspace-rejected")
    problem = LinearProblem(
        object(),
        object(),
        scope=Hierarchy(),
        properties=LinearOperatorProperties.symmetric_operator(),
        nullspace=ConstantNullspace(),
        gauge=MeanValueGauge(0),
    )

    with pytest.raises(NotImplementedError, match="complete AMR hierarchy"):
        program.solve(problem, solver=CompositeTensorFAC())


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
