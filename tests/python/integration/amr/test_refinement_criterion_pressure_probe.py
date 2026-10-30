"""Probe: a refinement criterion that the tag-graph VM cannot express, added without an engine change.

The criterion refines where the gradient of the pressure p = (gamma - 1) (E - |m|^2 / (2 rho)) of a
four-component state is steep. The VM leaves read one component of one state; this criterion reads
four, combines them and differentiates the result. Written as functions of ``cell`` (spec 0.3, DM13b),
it is generated into the compiled Program like any region: no C++ of the engine names it (CF1, CF5b).

The state is advected at constant velocity on a periodic square, so p stays a smooth bump that moves.
At bootstrap and after each accepted step, the level-0 requests of the native tagging plan must equal a
NumPy evaluation of the same arithmetic on the coarse state, on every cell of the level: the reads of a
cell on the boundary cross the periodic identification, which the ghosts of the level hold.
"""
from __future__ import annotations

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
    Coarsen,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.analytic import exp, x, y
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer, gradient_norm
from pops.lib.initial import Analytic
from pops.lib.time import SSPRK2
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import FixedDt, every

pytestmark = [pytest.mark.kokkos, pytest.mark.regression]

N = 32
GAMMA = 1.4
VELOCITY = (1.0, 0.25)
REFINE, COARSEN = 1.5, 0.3
DT = 4.0e-3
STEPS = 6


def _pressure(cell, state):
    rho, mx, my, energy = (cell.value(ValueExpr(state)[name]) for name in ("rho", "mx", "my", "E"))
    return (GAMMA - 1.0) * (energy - 0.5 * (mx * mx + my * my) / rho)


def _case():
    domain = Rectangle("periodic_square", lower=(0.0, 0.0), upper=(1.0, 1.0)).tag("fluid")
    frame = domain.frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    grid = CartesianGrid(frame=frame, cells=(N, N), periodic=PeriodicAxes(frame.axes))
    model = pops.Model("pressure_probe", frame=frame)
    U = model.state("U", components=("rho", "mx", "my", "E"), representation=Conservative(),
                    space=CellState(frame=frame))
    ax, ay = VELOCITY
    flux = model.flux("advection", frame=frame, state=U,
                      components={x_axis: tuple(ax * q for q in U),
                                  y_axis: tuple(ay * q for q in U)},
                      waves={x_axis: (ax,) * 4, y_axis: (ay,) * 4})
    rate = model.rate("advection_rate", equation=ddt(U) == -div(flux))
    method = FiniteVolume(
        flux=flux, variables=variables.Conservative(U),
        reconstruction=reconstruction.MUSCL(limiters.VanLeer()), riemann=riemann.Rusanov())
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, method)
    case = pops.Case("pressure_probe")
    block = case.block("probe", model=model)
    state = block[U]
    case.numerics(numerics, block=block)
    program = SSPRK2(state, rate=rate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    bump = exp(-120.0 * ((x(frame) - 0.35) ** 2 + (y(frame) - 0.45) ** 2))
    rho = 1.0 + 0.0 * x(frame)
    pressure = 1.0 + 0.4 * bump
    case.initials.add(InitialCondition(
        state=state,
        value=Analytic(frame=frame, components=(
            rho, ax * rho, ay * rho,
            pressure / (GAMMA - 1.0) + 0.5 * (ax * ax + ay * ay) * rho)),
        projection=ConservativeCellAverage()))
    refine = case.value(case.param(RuntimeParam("pressure_refine", default=REFINE)))
    coarsen = case.value(case.param(RuntimeParam("pressure_coarsen", default=COARSEN)))

    stencil = method.amr_indicator_stencil(dimension=len(frame.axes))

    def steepness(cell):
        return gradient_norm(cell, lambda at: _pressure(at, state), axes=frame.axes,
                             stencil=stencil)

    tagging = AMRTagging(
        rules=(Tag(lambda cell: steepness(cell) > refine),
               Coarsen(lambda cell: steepness(cell) < coarsen),
               Buffer(cells=1)),
        hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
        conflict_policy=ConflictPolicy.REFINE_WINS)
    transfer = AMRTransfer()
    transfer.state(state, StateTransfer())
    layout = AMR(grid=grid, hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)), tagging=tagging,
                 regrid=AMRRegrid(schedule=every(2, clock=program.clock)), transfer=transfer,
                 execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)))
    return case, layout


def _expected_requests(coarse):
    """The arithmetic of the generated criterion, in the same order, on every periodic cell."""
    rho, mx, my, energy = coarse
    p = (GAMMA - 1.0) * (energy - 0.5 * (mx * mx + my * my) / rho)
    h = 1.0 / N
    dx = (-0.5 * np.roll(p, 1, axis=1) + 0.5 * np.roll(p, -1, axis=1)) / h
    dy = (-0.5 * np.roll(p, 1, axis=0) + 0.5 * np.roll(p, -1, axis=0)) / h
    steep = np.sqrt(dx * dx + dy * dy)
    return np.where(steep > REFINE, 1, np.where(steep < COARSEN, 2, 0)).astype(np.uint8)


@pytest.mark.compiler
@pytest.mark.native_loader
def test_pressure_gradient_criterion_runs_without_an_engine_change(isolated_native_cache) -> None:
    del isolated_native_cache
    case, layout = _case()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    (field,) = resolved.bootstrap_plan.tagging.fields
    assert dict(zip(field.variables, field.components)) == {"rho": 0, "mx": 1, "my": 2, "E": 3}
    assert field.reach == (1, 1)
    runtime = pops.bind(pops.compile(resolved))
    seen = set()
    for step in range(STEPS + 1):
        coarse = np.asarray(runtime.block_level_state_global("probe", 0), dtype=np.float64)
        coarse = coarse.reshape(4, N, N)
        requests = np.frombuffer(
            runtime._executor._tagging_requests(0), dtype=np.uint8).reshape(N, N)
        np.testing.assert_array_equal(requests, _expected_requests(coarse),
                                      err_msg="step %d" % step)
        seen.update(np.unique(requests).tolist())
        assert int(runtime.n_levels()) == 2
        if step < STEPS:
            pops.step(runtime, t_bound=float(runtime.time()) + DT)
    assert {1, 2} <= seen
