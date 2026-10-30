"""A refinement criterion that reads a non-finite sample rejects the level, on both routes (N12).

The case advects two components with the method of the parity cases: ``u``, the Gaussian of
tutorial 09, and ``q = 1e200 x``, whose discrete gradient overflows. Two rules read that gradient
behind a left operand: ``(u > and_left) & (|grad q| > 1)`` and ``(u > or_left) | (|grad q| > 1)``.
The tag-graph VM evaluates every leaf of a rule, so it rejects the level at ``bind`` whatever the
left operands decide; written as functions of ``cell``, the same rules must be rejected with the
same message, both when the left operands leave the right ones to decide and when they decide alone
(``u > 10`` is false and ``u > 0`` true on every cell): a region evaluates every comparison before
``&`` and ``|`` join them, as IR4 evaluates the operands of an operation.
"""
from __future__ import annotations

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
from pops.analytic import exp, x, y
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Inflow, Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer, gradient_norm
from pops.lib.initial import Analytic
from pops.lib.time import SSPRK2
from pops.math import ValueExpr, ddt, div, grad, norm
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import AdaptiveCFL, every

pytestmark = [pytest.mark.kokkos, pytest.mark.regression]

N = 32
REFUSAL = "rejected a non-finite indicator sample"


def _case(route):
    domain = Rectangle("unit_square", lower=(0.0, 0.0), upper=(1.0, 1.0)).tag("fluid")
    frame = domain.frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("refinement_criterion_overflow", frame=frame)
    U = model.state("U", components=("u", "q"), representation=Conservative(),
                    space=CellState(frame=frame))
    flux = model.flux("advection_flux", frame=frame, state=U,
                      components={x_axis: tuple(1.0 * q for q in U),
                                  y_axis: tuple(0.25 * q for q in U)},
                      waves={x_axis: (1.0, 1.0), y_axis: (0.25, 0.25)})
    rate = model.rate("advection_rate", equation=ddt(U) == -div(flux))
    method = FiniteVolume(
        flux=flux, variables=variables.Conservative(U),
        reconstruction=reconstruction.MUSCL(limiters.VanLeer()), riemann=riemann.Rusanov())
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, method)
    case = pops.Case("refinement_criterion_nonfinite")
    tracer = case.block("tracer", model=model)
    state = tracer[U]
    boundaries = frame.boundaries
    numerics.boundaries.add(TransportBoundarySet({
        boundaries.x_min: Inflow(state=state, value=(0.05, 0.05)),
        boundaries.x_max: Outflow(state=state),
        boundaries.y_min: Inflow(state=state, value=(0.05, 0.05)),
        boundaries.y_max: Outflow(state=state),
    }))
    case.numerics(numerics, block=tracer)
    program = SSPRK2(state, rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=0.45, max_dt=1.0e-2))
    case.program(program)
    bump = exp(-120.0 * ((x(frame) - 0.30) ** 2 + (y(frame) - 0.35) ** 2))
    case.initials.add(InitialCondition(
        state=state,
        value=Analytic(frame=frame, components=(0.05 + 0.95 * bump, 1.0e200 * x(frame))),
        projection=ConservativeCellAverage()))
    params = {key: case.param(RuntimeParam("%s_threshold" % key, default=value))
              for key, value in {"and_left": 0.0, "or_left": 10.0, "right": 1.0}.items()}
    value = {key: case.value(handle) for key, handle in params.items()}
    u, q = ValueExpr(state)["u"], ValueExpr(state)["q"]
    if route == "graph":
        rules = (Tag((u > value["and_left"]) & (norm(grad(q)) > value["right"])),
                 Tag((u > value["or_left"]) | (norm(grad(q)) > value["right"])))
    else:
        stencil = method.amr_indicator_stencil(dimension=2)

        def steep(cell):
            return gradient_norm(cell, q, axes=frame.axes, stencil=stencil) > value["right"]

        rules = (Tag(lambda cell: (cell.value(u) > value["and_left"]) & steep(cell)),
                 Tag(lambda cell: (cell.value(u) > value["or_left"]) | steep(cell)))
    tagging = AMRTagging(rules=rules + (Buffer(cells=2),),
                         hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                         conflict_policy=ConflictPolicy.REFINE_WINS)
    transfer = AMRTransfer()
    transfer.state(state, StateTransfer())
    layout = AMR(grid=CartesianGrid(frame=frame, cells=(N, N)),
                 hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)), tagging=tagging,
                 regrid=AMRRegrid(schedule=every(2, clock=program.clock)), transfer=transfer,
                 execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)))
    return case, layout, params


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("route", ("graph", "region"))
def test_a_non_finite_sample_rejects_the_level_on_both_routes(route, isolated_native_cache) -> None:
    del isolated_native_cache
    case, layout, params = _case(route)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    # The left operands leave the gradient to decide, then decide alone without reading it.
    for and_left, or_left in ((0.0, 10.0), (10.0, 0.0)):
        with pytest.raises(RuntimeError, match=REFUSAL):
            pops.bind(artifact, params={params["and_left"]: and_left,
                                        params["or_left"]: or_left})
