from __future__ import annotations

import pytest

from pops.fields._prepared_field_solver_registry import PreparedFieldSolverFacts
from pops.solvers.elliptic import CartesianCG, GeometricMG
from pops.solvers.tolerances import AbsoluteFloor, Relative


def _facts(
    *,
    target: str = "system",
    dynamic: bool = False,
    iterate_dependent: bool = False,
    nonlinear: bool = False,
) -> PreparedFieldSolverFacts:
    return PreparedFieldSolverFacts(
        target=target,
        operator={"principal": "scalar-laplacian", "screened": False, "reaction": None},
        layout={
            "kind": "uniform" if target == "system" else "amr",
            "levels": 1 if target == "system" else 2,
            "transition_ratios": () if target == "system" else (2,),
            "embedded_boundary": False,
            "adaptive": target != "system",
            "cells": (16, 16),
            "topology_identity": "test-topology",
            "topology_recipe": {},
        },
        hierarchy={
            "policy_id": (
                "pops.field-hierarchy.level-local"
                if target == "system"
                else "pops.field-hierarchy.composite"
            ),
            "interface_version": 1,
            "option_schema": "pops.field-hierarchy.options.empty@1",
            "options": {},
        },
        boundary={
            "faces": (
                {"type": "dirichlet"},
                {"type": "dirichlet"},
                {"type": "dirichlet"},
                {"type": "dirichlet"},
            ),
            "dynamic": dynamic,
            "dependent": False,
            "state_dependent": False,
            "field_dependent": False,
            "logical_time_coordinates": (),
            "iterate_dependent": iterate_dependent,
        },
        nonlinear=nonlinear,
    )


def _prepare(solver, facts: PreparedFieldSolverFacts):
    provider, options = solver._prepared_field_solver()
    return provider.prepare(options=options, facts=facts, where="test field solver")


def test_cartesian_cg_has_its_own_exact_options_and_native_identity() -> None:
    solver = CartesianCG(
        tolerance=Relative(2.0e-7, AbsoluteFloor(3.0e-13)),
        max_iterations=91,
    )
    binding = _prepare(solver, _facts())

    assert solver.native_id == "pops::elliptic::nd::CartesianPoissonSolver<Dim>"
    assert solver.scheme == "cartesian_cg"
    assert solver.cg_options() == {
        "rel_tol": 2.0e-7,
        "abs_tol": 3.0e-13,
        "max_iterations": 91,
    }
    assert binding.provider["provider_id"] == "pops.field-solver.cartesian-cg"
    assert binding.resolution.native_contract == {
        "factory_route": "cartesian_cg",
        "schema_identity": "pops.system.cartesian-cg-options@1",
        "options": solver.cg_options(),
    }


def test_cartesian_cg_options_change_the_authenticated_binding() -> None:
    default = _prepare(CartesianCG(), _facts())
    configured = _prepare(CartesianCG(max_iterations=17), _facts())

    assert default.identity != configured.identity
    assert default.resolution.native_contract["options"] == {
        "rel_tol": 1.0e-10,
        "abs_tol": 0.0,
        "max_iterations": 2000,
    }


def test_solver_families_refuse_the_wrong_runtime_target() -> None:
    with pytest.raises(ValueError, match="GeometricMG is reserved.*AMR"):
        _prepare(GeometricMG(), _facts())
    with pytest.raises(ValueError, match="CartesianCG implements only a uniform System"):
        _prepare(CartesianCG(), _facts(target="amr_system"))


def test_cartesian_cg_accepts_prepared_dynamic_boundaries() -> None:
    binding = _prepare(CartesianCG(), _facts(dynamic=True))
    assert binding.resolution.native_contract["factory_route"] == "cartesian_cg"


def test_cartesian_cg_requires_newton_for_iterate_dependent_boundaries() -> None:
    with pytest.raises(ValueError, match="requires a prepared Newton-Krylov"):
        _prepare(CartesianCG(), _facts(dynamic=True, iterate_dependent=True))
    binding = _prepare(
        CartesianCG(),
        _facts(dynamic=True, iterate_dependent=True, nonlinear=True),
    )
    assert binding.resolution.native_contract["factory_route"] == "cartesian_cg"


def test_cartesian_cg_rejects_foreign_controls() -> None:
    with pytest.raises(TypeError, match="tolerance"):
        CartesianCG(tolerance="relative")
    with pytest.raises(ValueError, match="max_iterations"):
        CartesianCG(max_iterations=0)
