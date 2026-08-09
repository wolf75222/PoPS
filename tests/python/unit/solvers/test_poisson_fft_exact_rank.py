"""Exact prepared-provider contract for the concrete rank-two FFT Poisson engine."""
from __future__ import annotations

import pytest

from pops.fields._prepared_field_solver_registry import PreparedFieldSolverFacts
from pops.solvers.elliptic import FFT


def _facts(
    cells: tuple[int, ...] = (16, 16), *, periodic: bool = True,
) -> PreparedFieldSolverFacts:
    face_type = "periodic" if periodic else "dirichlet"
    return PreparedFieldSolverFacts(
        target="system",
        operator={"principal": "scalar-laplacian", "screened": False, "reaction": None},
        layout={
            "kind": "uniform",
            "levels": 1,
            "transition_ratios": (),
            "embedded_boundary": False,
            "adaptive": False,
            "cells": cells,
            "topology_identity": "fft-test-topology",
            "topology_recipe": {},
        },
        hierarchy={
            "policy_id": "pops.field-hierarchy.level-local",
            "interface_version": 1,
            "option_schema": "pops.field-hierarchy.options.empty@1",
            "options": {},
        },
        boundary={
            "faces": tuple({"type": face_type} for _ in range(2 * len(cells))),
            "dynamic": False,
            "dependent": False,
            "state_dependent": False,
            "field_dependent": False,
            "logical_time_coordinates": (),
            "iterate_dependent": False,
        },
        nonlinear=False,
    )


def _prepare(solver: FFT, facts: PreparedFieldSolverFacts):
    provider, options = solver._prepared_field_solver()
    return provider.prepare(options=options, facts=facts, where="test FFT field solver")


def test_fft_prepares_one_discrete_exact_rank_two_route() -> None:
    solver = FFT()
    binding = _prepare(solver, _facts())

    assert solver.native_id == "pops::PoissonFFTSolver<2>"
    assert solver.scheme == "fft"
    assert solver.options() == {}
    provider = binding.to_data()["provider"]
    assert provider["provider_id"] == "pops.field-solver.fft"
    assert provider["version"] == 2
    assert provider["resolver_id"] == "pops.field-solver.fft.resolve@2"
    assert provider["installer_id"] == "pops.field-solver.fft.install@2"
    assert provider["use_policy"] == {
        "policy_id": "pops.field-solver.fft.use",
        "version": 2,
        "capabilities": {
            "targets": ["system"],
            "layout": "uniform-exact-rank-two-power-of-two",
            "boundary": "fully-periodic",
            "operator": "discrete-five-point-poisson",
        },
    }
    assert binding.resolution.native_contract == {
        "factory_route": "fft",
        "schema_identity": "pops.system.fft-discrete-rank2-options.empty@1",
        "options": {},
    }


@pytest.mark.parametrize("cells", ((16,), (16, 16, 16)))
def test_fft_provider_fails_closed_outside_exact_rank_two(cells: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="requires exact Dim=2"):
        _prepare(FFT(), _facts(cells))


def test_fft_provider_refuses_non_power_of_two_instead_of_falling_back() -> None:
    with pytest.raises(ValueError, match="power-of-two"):
        _prepare(FFT(), _facts((12, 16)))


def test_fft_provider_refuses_non_periodic_boundary() -> None:
    with pytest.raises(ValueError, match="fully periodic"):
        _prepare(FFT(), _facts(periodic=False))


def test_retired_continuous_symbol_has_no_python_or_catalog_route() -> None:
    from pops.runtime import _generated_component_routes as routes

    with pytest.raises(TypeError):
        FFT(spectral=True)
    field_solver_tokens = tuple(row[0] for row in routes.ROUTE_TABLES["field_solver"])
    assert field_solver_tokens == ("geometric_mg", "fft", "polar", "cartesian_cg")
