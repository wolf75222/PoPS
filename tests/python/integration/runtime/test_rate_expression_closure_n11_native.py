"""N11 lot 1 (spec 0.3 R7): an inline operand of a balance runs as the source it declares.

A balance written with algebra in its right-hand side (``ddt(U) == k*q*(1 - q)``) compiles and
runs bit for bit as the same balance written with a declared source, and both match the explicit
Euler recurrence computed with numpy.
"""

import numpy as np
import pytest

import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.time import ForwardEuler
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


def _run(spelling):
    case, layout = _case(spelling)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    initial = _initial()
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    runtime = pops.bind(artifact, initial_values={subject: initial.copy()},
                        resources={"execution_context": artifact_execution_context(artifact)})
    grid = tuple(DT * step for step in range(STEPS + 1))
    report = pops.run(runtime, t_end=grid[-1], max_steps=STEPS, time_grid=grid)
    assert report.accepted_steps == STEPS and report.rejected_steps == 0
    return np.asarray(runtime.state_global("material")).reshape(initial.shape)


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
