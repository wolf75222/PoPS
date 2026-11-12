"""N11 lot 1 (spec 0.3 R7): an inline operand of a balance runs as the source it declares.

A balance written with algebra in its right-hand side (``ddt(U) == k*q*(1 - q)``) compiles and
runs bit for bit as the same balance written with a declared source, and both match the explicit
Euler recurrence computed with numpy. M03 (stiffened-gas Euler) with an optically thin loss
written inline, a product, a quotient and an elementary function of the state, runs bit for bit
as its declared spelling and matches a NumPy SSPRK2 oracle of the loss on uniform data.
"""

import numpy as np
import pytest

import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops import math as pm
from examples.api_v02.b6_numerics import rusanov
from pops.lib.time import SSPRK2, ForwardEuler
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FiniteVolume, StateStorage
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import VanLeer
from pops.numerics.riemann import ScalarUpwind
from pops.numerics.variables import Conservative
from pops.projection import ConservativeCellAverage
from pops.time import ExternalTimeGrid
from tests.python.support.native_execution_context import artifact_execution_context


CELLS = 8
DT = 0.01
STEPS = 3
GROWTH = 3.
LOSS = 0.5


def _initial():
    x = (np.arange(CELLS) + 0.5) / CELLS
    profile = 0.2 + 0.6 * np.add.outer(np.sin(2. * np.pi * x) ** 2, x) / 2.
    return profile.reshape(1, CELLS, CELLS)


def _case(spelling):
    frame = Rectangle("closure_box_" + spelling, lower=(0., 0.),
                      upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("closure_" + spelling, frame=frame)
    state = model.state("U", components=("q",))
    (q,) = state
    plan = DiscretizationPlan()
    if spelling == "inline":
        rate = model.rate("balance", equation=ddt(state) == GROWTH * q * (1. - q) - LOSS * q ** 2)
    elif spelling == "declared":
        growth = model.source("growth", on=state, value=(GROWTH * q * (1. - q),))
        loss = model.source("loss", on=state, value=(-(LOSS * q ** 2),))
        rate = model.rate("balance", equation=ddt(state) == growth + loss)
    else:
        flux = model.flux("F", frame=frame, state=state,
                          components={frame.x: (q,), frame.y: (0 * q,)},
                          waves={frame.x: (1,), frame.y: (0,)})
        velocity = model.vector("a", frame=frame, components={frame.x: 1, frame.y: 0})
        if spelling == "transport_inline":
            rate = model.rate("balance", equation=ddt(state) == -div(flux) - LOSS * q ** 2)
        else:
            loss = model.source("loss", on=state, value=(-(LOSS * q ** 2),))
            rate = model.rate("balance", equation=ddt(state) == -div(flux) + loss)
    case = pops.Case("closure_" + spelling)
    block = case.block("material", model)
    if spelling.startswith("transport"):
        plan.rates.add(rate, FiniteVolume(flux=flux, variables=Conservative(state),
                                          reconstruction=MUSCL(VanLeer()),
                                          riemann=ScalarUpwind(velocity=velocity)))
    else:
        plan.rates.add(rate, StateStorage())
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(ExternalTimeGrid("time_grid"))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(CELLS, CELLS),
                                   periodic=PeriodicAxes(frame.axes)))
    return case, layout


def _advance(artifact, initial, *, dt, steps, block="material"):
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    runtime = pops.bind(artifact, initial_values={subject: initial.copy()},
                        resources={"execution_context": artifact_execution_context(artifact)})
    grid = tuple(dt * step for step in range(steps + 1))
    report = pops.run(runtime, t_end=grid[-1], max_steps=steps, time_grid=grid)
    assert report.accepted_steps == steps and report.rejected_steps == 0
    return np.asarray(runtime.state_global(block)).reshape(initial.shape)


def _run(spelling):
    case, layout = _case(spelling)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    return _advance(artifact, _initial(), dt=DT, steps=STEPS)


# M03: stiffened-gas Euler (examples/api_v02 build_m03) with an optically thin loss on the energy.
GAMMA, P_INF, M03_CELLS, M03_STEPS, RADIATION = 1.4, 0.2, (16, 8), 2, 0.5
M03_DT = 0.02 / max(M03_CELLS)


def _m03_artifact(spelling):
    frame = Rectangle("closure_m03_" + spelling, lower=(0., 0.),
                      upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("closure_m03_" + spelling, frame=frame)
    state = model.state("conserved", components=("rho", "mx", "my", "energy"))
    rho, mx, my, energy = state
    vx, vy = mx / rho, my / rho
    pressure = (GAMMA - 1) * (energy - (mx * mx + my * my) / (2 * rho)) - GAMMA * P_INF
    sound = pm.sqrt(GAMMA * (pressure + P_INF) / rho)
    x, y = frame.axes
    flux = model.flux("Euler_flux", state=state, frame=frame,
                      components={x: (mx, mx * vx + pressure, my * vx, (energy + pressure) * vx),
                                  y: (my, mx * vy, my * vy + pressure, (energy + pressure) * vy)},
                      waves={x: (vx - sound, vx, vx, vx + sound),
                             y: (vy - sound, vy, vy, vy + sound)})
    temperature = pressure / rho
    loss = (0, 0, 0, -RADIATION * rho ** 2 * pm.sqrt(temperature))
    if spelling == "inline":
        rate = model.rate("Euler_balance", equation=ddt(state) == -div(flux) + loss)
        assert [item.kind for item in rate.occurrences] == ["flux", "source"]
    else:
        radiation = model.source("radiation", on=state, value=loss)
        rate = model.rate("Euler_balance", equation=ddt(state) == -div(flux) + radiation)
    assert rate.view.legacy_incompatibility() is None
    case = pops.Case("closure_m03_" + spelling)
    block = case.block("fluid", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, rusanov(flux, state))
    case.numerics(plan, block=block)
    program = SSPRK2(block[state], rate=rate)
    program.step_strategy(ExternalTimeGrid("time_grid"))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=M03_CELLS,
                                   periodic=PeriodicAxes(frame.axes)))
    return pops.compile(pops.resolve(pops.validate(case), layout=layout))


def _m03_wave():
    """The M03 entropy wave."""
    x = (np.arange(M03_CELLS[0]) + .5) / M03_CELLS[0]
    rho = np.broadcast_to(1 + .1 * np.cos(2 * np.pi * (x - .3)), (M03_CELLS[1], M03_CELLS[0]))
    return np.stack((rho, .3 * rho, .1 * rho, (1 + GAMMA * P_INF) / (GAMMA - 1) + .05 * rho))


def _m03_uniform():
    ones = np.ones((M03_CELLS[1], M03_CELLS[0]))
    return np.stack((ones, .3 * ones, .1 * ones, ((1 + GAMMA * P_INF) / (GAMMA - 1) + .05) * ones))


def _loss_oracle(initial):
    """SSPRK2 of dE/dt = -L rho^2 sqrt(p/rho) on uniform data, where the divergence vanishes."""
    rho, mx, my, energy = initial

    def rate(e):
        pressure = (GAMMA - 1) * (e - (mx * mx + my * my) / (2 * rho)) - GAMMA * P_INF
        return -RADIATION * rho ** 2 * np.sqrt(pressure / rho)

    for _ in range(M03_STEPS):
        first = energy + M03_DT * rate(energy)
        energy = 0.5 * energy + 0.5 * (first + M03_DT * rate(first))
    return energy


@pytest.mark.compiler
@pytest.mark.native_loader
def test_native_inline_reaction_runs_as_its_declared_source(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    inline = _run("inline")
    declared = _run("declared")
    assert np.array_equal(inline, declared)
    expected = _initial()
    for _ in range(STEPS):
        expected = expected + DT * (GROWTH * expected * (1. - expected) - LOSS * expected ** 2)
    np.testing.assert_allclose(inline, expected, rtol=1.e-13, atol=0.)


@pytest.mark.compiler
@pytest.mark.native_loader
def test_native_transport_with_an_inline_loss_runs_as_its_declared_source(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    inline = _run("transport_inline")
    declared = _run("transport_declared")
    assert np.array_equal(inline, declared)
    assert not np.array_equal(inline, _initial())


@pytest.mark.compiler
@pytest.mark.native_loader
def test_native_m03_with_an_inline_nonlinear_loss_runs_as_its_declared_source(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    inline, declared = _m03_artifact("inline"), _m03_artifact("declared")
    wave = _m03_wave()
    final = _advance(inline, wave, dt=M03_DT, steps=M03_STEPS, block="fluid")
    assert np.array_equal(final, _advance(declared, wave, dt=M03_DT, steps=M03_STEPS,
                                          block="fluid"))
    assert np.all(np.isfinite(final))
    # The loss acts on the energy only: mass and momentum stay conserved (M03's invariants).
    for component in range(3):
        assert abs(np.mean(final[component]) - np.mean(wave[component])) <= 2.e-10
    assert np.mean(final[3]) < np.mean(wave[3])
    uniform = _m03_uniform()
    still = _advance(inline, uniform, dt=M03_DT, steps=M03_STEPS, block="fluid")
    assert np.array_equal(still[:3], uniform[:3])
    np.testing.assert_allclose(still[3], _loss_oracle(uniform), rtol=1.e-13, atol=0.)
