"""Resolved AMR carrier assertions plus one complete public root lifecycle.

The first tests deliberately exercise the private native carrier seam after obtaining the plan
through public validation/resolution.  Only the final test is an end-to-end
``compile -> bind -> run`` assertion through the public root API.
"""
from __future__ import annotations

import numpy as np
import pytest

import pops
import pops.lib.time as libtime
from pops.runtime._system import AmrSystem
from pops.solvers.elliptic import GeometricMG
from pops.solvers.options import CompositeFAC
from pops.solvers.tolerances import Relative
from pops.time import FailRun, FixedDt
from tests.python.integration._final_field_program import (
    resolve_periodic_field_program,
    scalar_advection_field_model,
)


pytestmark = (pytest.mark.compiler, pytest.mark.native_loader, pytest.mark.kokkos)

_DT = 1.0e-4

_FAC_DEFAULTS = {
    "max_iters": 30,
    "fine_sweeps": 400,
    "rel_tol": 1.0e-9,
    "abs_tol": 0.0,
    "coarse_rel_tol": 1.0e-12,
    "coarse_abs_tol": 0.0,
    "coarse_cycles": 100,
    "verbose": False,
}
_FAC_CONFIGURED = {
    "max_iters": 11,
    "fine_sweeps": 17,
    "rel_tol": 2.0e-7,
    "abs_tol": 3.0e-12,
    "coarse_rel_tol": 4.0e-9,
    "coarse_abs_tol": 5.0e-14,
    "coarse_cycles": 19,
    "verbose": True,
}


def _field_program(state, rate, field):
    program = libtime.ForwardEuler(
        state, rate=rate, fields=field, solve_action=FailRun())
    program.step_strategy(FixedDt(_DT))
    return program


def _resolve(solver: GeometricMG):
    model = scalar_advection_field_model("native-composite-fac-carrier-model")
    return resolve_periodic_field_program(
        model,
        _field_program,
        name="native-composite-fac-carrier",
        block_name="plasma",
        target="amr_system",
        n=16,
        regrid_every=1,
        field_solver=solver,
    )


def _install_resolved_plan_on_native_carrier(solver: GeometricMG):
    """Install one publicly resolved plan through the deliberate private carrier seam."""
    resolved = _resolve(solver)
    assert len(resolved.field_plans) == 1
    field_name, field_plan = next(iter(resolved.field_plans.items()))
    engine = AmrSystem(n=16, L=1.0)
    engine._install_field_plan(field_name, field_plan)
    configuration = engine.field_solver_configuration(
        field_plan.native_options["provider_slot"])
    assert configuration["schema_version"] == 1
    assert configuration["provider_slot"] == field_plan.native_options["provider_slot"]
    assert configuration["plan_identity"] == field_plan.identity.token
    return field_plan, configuration


def _assert_options(actual, expected) -> None:
    assert set(actual) == set(expected)
    for name, value in expected.items():
        if isinstance(value, bool):
            assert actual[name] is value
        else:
            assert actual[name] == pytest.approx(value)


@pytest.mark.parametrize(
    "solver",
    (GeometricMG(), GeometricMG(fac=CompositeFAC())),
    ids=("absent", "explicit-empty"),
)
def test_absent_or_empty_composite_fac_installs_native_fac_defaults(
    solver: GeometricMG,
) -> None:
    field_plan, configuration = _install_resolved_plan_on_native_carrier(solver)

    assert field_plan.native_options["hierarchy"] == "composite"
    assert configuration["solver"] == "geometric_mg"
    assert configuration["hierarchy"] == "composite"
    _assert_options(configuration["fac"], _FAC_DEFAULTS)


def test_partial_fac_overrides_do_not_inherit_or_replace_geometric_mg_options() -> None:
    solver = GeometricMG(
        tolerance=Relative(6.0e-6),
        max_cycles=7,
        min_coarse=3,
        pre_sweeps=4,
        post_sweeps=5,
        bottom_sweeps=6,
        fac=CompositeFAC(max_iters=11, abs_tol=3.0e-12, verbose=True),
    )
    _field_plan, configuration = _install_resolved_plan_on_native_carrier(solver)

    _assert_options(configuration["mg"], {
        "rel_tol": 6.0e-6,
        "abs_tol": 0.0,
        "max_cycles": 7,
        "min_coarse": 3,
        "pre_smooth": 4,
        "post_smooth": 5,
        "bottom_sweeps": 6,
        "coarse_threshold": 0,
    })
    _assert_options(configuration["fac"], {
        **_FAC_DEFAULTS,
        "max_iters": 11,
        "abs_tol": 3.0e-12,
        "verbose": True,
    })


def test_fac_overrides_propagate_through_a_refined_final_root_lifecycle(
    isolated_native_cache, native_cxx, kokkos_root,
) -> None:
    del isolated_native_cache, native_cxx, kokkos_root
    solver = GeometricMG(
        fac=CompositeFAC(**_FAC_CONFIGURED)
    )
    resolved = _resolve(solver)
    artifact = pops.compile(resolved)

    n = 16
    coordinate = (np.arange(n, dtype=np.float64) + 0.5) / n
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    density = 0.1 + np.exp(-80.0 * ((x - 0.35) ** 2 + (y - 0.55) ** 2))
    simulation = pops.bind(
        artifact,
        initial_state={"plasma": density[None, :, :]},
    )
    report = pops.run(simulation, t_end=2.0 * _DT, max_steps=2)

    assert report.accepted_steps == 2
    assert simulation.n_levels() == 2
    slot, = simulation.field_provider_slots()
    assert simulation.field_provider_levels(slot) == 2
    fine = np.asarray(simulation.field_potential_level_global(slot, 1), dtype=np.float64)
    assert fine.size > 0 and np.isfinite(fine).all()
    assert np.max(np.abs(fine - fine.mean())) > 1.0e-10

    provider, = simulation.inspect().to_dict()["instance"]["field_providers"]
    assert provider["provider_slot"] == slot
    assert provider["materialized"] is True
    assert provider["solver_configuration"]["hierarchy"] == "composite"
    _assert_options(provider["solver_configuration"]["fac"], _FAC_CONFIGURED)
