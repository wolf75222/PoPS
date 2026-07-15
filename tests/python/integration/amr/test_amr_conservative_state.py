"""Full conservative AMR initialization through the final public lifecycle.

This test deliberately starts with non-zero momentum.  It proves that ``pops.bind`` transports the
complete conservative vector to the level-zero AMR state, rather than silently retaining the former
density-only seed.  The AMR hierarchy also declares its level-clock relation explicitly: temporal
subcycling is an authored authority, never an inference from the spatial ratio.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pops
import pytest
from pops.amr import (
    AMRClockRelation,
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import BindArray, Gaussian
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.physics import Density, Model, Momentum
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, StagePoint, TimePoint, every


ROOT = Path(__file__).resolve().parents[4]
N = 16
DT = 1.0e-4

pytestmark = [
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
]


def _gas_model(frame):
    x_axis, y_axis = frame.axes
    model = Model("amr-conservative-gas", frame=frame)
    state = model.state(
        "U",
        components=("rho", "rho_u", "rho_v"),
        roles={
            "rho": Density(),
            "rho_u": Momentum(axis=x_axis),
            "rho_v": Momentum(axis=y_axis),
        },
    )
    rho, rho_u, rho_v = state
    model.flux(
        "transport",
        frame=frame,
        state=state,
        components={
            x_axis: (rho_u, 0.0 * rho_u, 0.0 * rho_v),
            y_axis: (rho_v, 0.0 * rho_u, 0.0 * rho_v),
        },
        waves={x_axis: (1.0, 0.0, -1.0), y_axis: (1.0, 0.0, -1.0)},
    )
    rate = model.rate("transport_rate", equation=ddt(state) == -div(model.fluxes["transport"]))
    return model, state, model.fluxes["transport"], rate


def _marker_model(frame):
    x_axis, y_axis = frame.axes
    model = Model("amr-conservative-marker", frame=frame)
    # ``U`` is the model-level conservative-state identifier.  Keeping it canonical avoids
    # authoring a second implicit state and makes this block's selected state unambiguous.
    state = model.state("U", components=("marker",))
    (marker,) = state
    flux = model.flux(
        "marker_transport",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * marker,), y_axis: (0.0 * marker,)},
        waves={x_axis: (0.0,), y_axis: (0.0,)},
    )
    rate = model.rate("marker_rate", equation=ddt(state) == -div(flux))
    return model, state, flux, rate


def _forward_euler(states_and_rates):
    """One public Program commits both block states at the same accepted-step clock."""
    program = pops.Program("amr_conservative_forward_euler")
    endpoints = []
    for name, state, rate in states_and_rates:
        temporal = program.state(state)
        stage = StagePoint(name + "_stage", {"main": TimePoint(program.clock, 0)})
        rhs = program.value(name + "_rhs", rate(temporal.n), at=stage)
        next_value = program.value(
            name + "_next", temporal.n + program.dt * rhs, at=temporal.next.point)
        endpoints.append((temporal.next, next_value))
    for endpoint, value in endpoints:
        program.commit(endpoint, value)
    program.step_strategy(FixedDt(DT))
    return program


def _resolved(native_cxx):
    frame = Rectangle("amr-conservative-domain", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    gas_model, gas_state, gas_flux, gas_rate = _gas_model(frame)
    marker_model, marker_state, marker_flux, marker_rate = _marker_model(frame)

    case = pops.Case("amr-conservative-case")
    gas = case.block("gas", gas_model, states=(gas_state,))
    marker = case.block("marker", marker_model, states=(marker_state,))
    gas_instance, marker_instance = gas[gas_state], marker[marker_state]
    for block, state, flux, rate in (
        (gas, gas_state, gas_flux, gas_rate),
        (marker, marker_state, marker_flux, marker_rate),
    ):
        numerics = DiscretizationPlan()
        numerics.rates.add(
            rate,
            FiniteVolume(
                flux=flux,
                variables=variables.Conservative(state),
                reconstruction=reconstruction.FirstOrder(),
                riemann=riemann.Rusanov(),
            ),
        )
        case.numerics(numerics, block=block)

    program = _forward_euler((("gas", gas_instance, gas_rate), ("marker", marker_instance, marker_rate)))
    case.program(program)
    case.initials.add(InitialCondition(
        state=gas_instance, value=BindArray(), projection=ConservativeCellAverage()))
    x_axis, y_axis = frame.axes
    case.initials.add(InitialCondition(
        state=marker_instance,
        value=Gaussian(
            frame=frame, center={x_axis: 0.5, y_axis: 0.5},
            background=0.0, amplitude=1.0, inverse_width=100.0),
        projection=ConservativeCellAverage(),
    ))
    threshold = case.param(RuntimeParam("marker_refine_threshold", default=0.2))
    transfer = AMRTransfer()
    transfer.state(gas_instance, StateTransfer())
    transfer.state(marker_instance, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(frame=frame, cells=(N, N), periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(Tag(ValueExpr(marker_instance) > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(1, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    )
    plan = pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include"), "cxx": native_cxx},
    )
    return plan, gas_instance


def _full_conservative_state() -> np.ndarray:
    points = (np.arange(N, dtype=np.float64) + 0.5) / N
    x, y = np.meshgrid(points, points, indexing="xy")
    rho = 1.0 + 0.2 * np.exp(-80.0 * ((x - 0.35) ** 2 + (y - 0.55) ** 2))
    return np.ascontiguousarray(np.stack((rho, 0.3 * rho, -0.15 * rho)))


def test_public_amr_bind_preserves_every_conservative_component(native_cxx, isolated_native_cache, kokkos_root):
    del isolated_native_cache, kokkos_root
    initial = _full_conservative_state()
    zero_momentum = initial.copy()
    zero_momentum[1:] = 0.0
    plan, gas_state = _resolved(native_cxx)
    artifact = pops.compile(plan)
    simulation = pops.bind(artifact, initial_values={gas_state: initial})
    stationary = pops.bind(artifact, initial_values={gas_state: zero_momentum})

    level_zero = np.asarray(simulation.block_level_state_global("gas", 0), dtype=np.float64)
    np.testing.assert_array_equal(level_zero.reshape(initial.shape), initial)
    stationary_level_zero = np.asarray(
        stationary.block_level_state_global("gas", 0), dtype=np.float64)
    np.testing.assert_array_equal(stationary_level_zero.reshape(initial.shape), zero_momentum)
    # Bootstrap materializes the resolved two-level hierarchy during bind; it is not deferred to
    # the first time step and must not mutate the authenticated coarse state while prolonging it.
    assert simulation.n_levels() == 2

    report = pops.run(simulation, t_end=2.0 * DT, max_steps=2)
    stationary_report = pops.run(stationary, t_end=2.0 * DT, max_steps=2)
    evolved = np.asarray(simulation.block_level_state_global("gas", 0), dtype=np.float64)
    stationary_evolved = np.asarray(
        stationary.block_level_state_global("gas", 0), dtype=np.float64).reshape(initial.shape)
    evolved = evolved.reshape(initial.shape)
    assert report.accepted_steps == simulation.macro_step() == 2
    assert stationary_report.accepted_steps == stationary.macro_step() == 2
    assert simulation.n_levels() == 2
    assert np.isfinite(evolved).all()
    np.testing.assert_allclose(
        evolved.sum(axis=(1, 2)), initial.sum(axis=(1, 2)), rtol=0.0, atol=2.0e-12)

    # The seeded momentum is executable physics, not merely preserved storage: against the exact
    # same artifact and density with zero momentum, it transports the density bump in +x.
    x = (np.arange(N, dtype=np.float64) + 0.5) / N

    def centroid_x(field):
        weight = field - field.min()
        return float((weight * x[None, :]).sum() / weight.sum())

    assert centroid_x(evolved[0]) > centroid_x(stationary_evolved[0]) + 1.0e-8
