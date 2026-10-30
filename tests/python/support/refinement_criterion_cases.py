"""Adaptive scalar-advection cases whose refinement criterion is a tag graph or regions of cell.

The case is tutorial 09 of the scalar-advection series (SSPRK2, MUSCL-Van Leer, upwind, two levels,
regrid every two steps). Only the rules of its AMRTagging change, so a test can run the same physics
with the bytecode tag graph and with the criterion generated into the compiled Program (DM13b).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import pops
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
)
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Inflow, Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import Gaussian
from pops.lib.time import SSPRK2
from pops.math import ddt, div
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import AdaptiveCFL, every

CELLS = (32, 32)
T_END = 0.10


def advection_case(
    rules: Callable[[SimpleNamespace], tuple[Any, ...]],
    *,
    thresholds: dict[str, float] | None = None,
    hysteresis: Hysteresis | None = None,
    conflict_policy: ConflictPolicy = ConflictPolicy.REFINE_WINS,
    name: str = "refinement_criterion_case",
) -> tuple[Any, Any, SimpleNamespace]:
    """(case, layout, names); ``rules(names)`` returns the Tag and Coarsen rules of the case.

    Each threshold is a RuntimeParam of the Case: ``names.params[key]`` is its Handle and
    ``names.value(key)`` its value in an expression. ``refine`` and ``coarsen`` default to the
    thresholds of tutorial 09. ``names.gradient`` is the discrete gradient of the finite-volume
    method, the stencil the tag graph reads for ``grad(...)``.
    """
    domain = Rectangle("unit_square", lower=(0.0, 0.0), upper=(1.0, 1.0)).tag("fluid")
    frame = domain.frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    grid = CartesianGrid(frame=frame, cells=CELLS)
    model = pops.Model("scalar_advection_refinement_criterion", frame=frame)
    U = model.state("U", components=("u",), representation=Conservative(),
                    space=CellState(frame=frame))
    (u,) = U
    velocity = model.vector("a", frame=frame, components={x_axis: 1.0, y_axis: 0.25})
    flux = model.flux("advection_flux", frame=frame, state=U,
                      components={x_axis: (1.0 * u,), y_axis: (0.25 * u,)},
                      waves={x_axis: (1.0,), y_axis: (0.25,)})
    rate = model.rate("advection_rate", equation=ddt(U) == -div(flux))
    method = FiniteVolume(
        flux=flux, variables=variables.Conservative(U),
        reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
        riemann=riemann.ScalarUpwind(velocity=velocity))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, method)
    case = pops.Case(name)
    tracer = case.block("tracer", model=model)
    state = tracer[U]
    boundaries = frame.boundaries
    numerics.boundaries.add(TransportBoundarySet({
        boundaries.x_min: Inflow(state=state, value=0.05),
        boundaries.x_max: Outflow(state=state),
        boundaries.y_min: Inflow(state=state, value=0.05),
        boundaries.y_max: Outflow(state=state),
    }))
    case.numerics(numerics, block=tracer)
    program = SSPRK2(state, rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=0.45, max_dt=1.0e-2))
    case.program(program)
    case.initials.add(InitialCondition(
        state=state,
        value=Gaussian(frame=frame, center={x_axis: 0.30, y_axis: 0.35}, background=0.05,
                       amplitude=0.95, inverse_width=120.0),
        projection=ConservativeCellAverage()))
    defaults = {"refine": 0.10, "coarsen": 0.04, **(thresholds or {})}
    params = {key: case.param(RuntimeParam("%s_threshold" % key, default=value))
              for key, value in defaults.items()}
    names = SimpleNamespace(
        case=case, frame=frame, state=state, params=params,
        value=lambda key: case.value(params[key]),
        gradient=method.amr_indicator_stencil(dimension=len(frame.axes)))
    tagging = AMRTagging(
        rules=tuple(rules(names)) + (Buffer(cells=2),),
        hysteresis=hysteresis or Hysteresis(0, EqualityPolicy.HOLD),
        conflict_policy=conflict_policy)
    transfer = AMRTransfer()
    transfer.state(state, StateTransfer())
    layout = AMR(grid=grid, hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)), tagging=tagging,
                 regrid=AMRRegrid(schedule=every(2, clock=program.clock)), transfer=transfer,
                 execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)))
    return case, layout, names


__all__ = ["CELLS", "T_END", "advection_case"]
