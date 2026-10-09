"""Public constant-gradient diffusion fixture and independent physical face amounts."""
import numpy as np
import pops

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import ddt, div, grad
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan
from pops.physics.diffusion import DiffusiveBoundary
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt

DT, DIFFUSIVITY, Q0, LEFT0 = 1e-4, .1, .7, -.2


def build_diffusion_integral_case(kind="uniform", *, proposed_dt=DT, selected_axis=0,
                                  diagnostic=False, mixed_transport=False):
    n = 32 if kind == "amr2" else 8
    frame = Rectangle("diffusive_trace", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("diffusive_trace_model", frame=frame)
    state = model.state("U", components=("density",))
    boundaries = tuple(DiffusiveBoundary(axis, side, "value", 1., (1., 0.)) if axis == 0
                       else DiffusiveBoundary(axis, side, "periodic")
                       for axis in range(2) for side in ("lower", "upper"))
    flux = model.diffusive_flux("diffusion", state=state, value=DIFFUSIVITY * grad(state[0]),
                                boundaries=boundaries)
    equation = div(flux)
    transport_method = None
    if mixed_transport:
        from pops.numerics import reconstruction, riemann, variables
        from pops.numerics.spatial import FiniteVolume
        transport = model.flux("transport", frame=frame, state=state,
            components={axis: (state[0],) for axis in frame.axes},
            waves={axis: (1.,) for axis in frame.axes})
        equation = equation - div(transport)
        transport_method = FiniteVolume(flux=transport, variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov())
    rate = model.rate("balance", equation=ddt(state) == equation)
    case = pops.Case("diffusive_integral_" + kind)
    block = case.block("fluid", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, Diffusion(flux=flux, transport=transport_method))
    if mixed_transport:
        from pops.boundary import TransportBoundarySet
        from pops.boundary.transport import Outflow
        plan.boundaries.add(TransportBoundarySet({
            frame.boundaries.x_min: Outflow(state=block[state]),
            frame.boundaries.x_max: Outflow(state=block[state]),
        }, periodic=PeriodicAxes((frame.axes[1],))))
    case.numerics(plan, block=block)
    program = pops.Program("diffusive_integral_forward_euler")
    temporal = program.state(block[state])
    selected = rate(temporal.n)
    accepted = program.value("accepted", temporal.n + program.dt * selected,
                             at=temporal.next.point)
    program.commit(temporal.next, accepted)
    right = program.integral_state("right", initial=Q0)
    left = program.integral_state("left", initial=LEFT0)
    program.accept_external_trace(right, rate=rate(temporal.n) if diagnostic else selected,
                                  axis=selected_axis, side=1, component=0)
    program.accept_external_trace(left, rate=selected, axis=0, side=0, component=0)
    program.step_strategy(FixedDt(proposed_dt))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    grid = CartesianGrid(frame=frame, cells=(n, n), periodic=PeriodicAxes((frame.axes[1],)))
    layout = Uniform(grid)
    if kind == "amr2":
        from pops.amr import (AMRClockRelation, AMRExecution, AMRHierarchy, AMRRegrid,
                              AMRTagging, AMRTransfer, Buffer, ConflictPolicy,
                              EqualityPolicy, Hysteresis, Tag)
        from pops.layouts import AMR
        from pops.lib.amr import StateTransfer
        from pops.math import ValueExpr
        from pops.params import RuntimeParam
        from pops.time import every
        transfer = AMRTransfer()
        transfer.state(block[state], StateTransfer())
        threshold = case.param(RuntimeParam("refine_threshold", default=1.5))
        layout = AMR(grid=grid, hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=AMRTagging(rules=(Tag(ValueExpr(block[state])[state.components[0]]
                                        > case.value(threshold)), Buffer(cells=1)),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                conflict_policy=ConflictPolicy.REFINE_WINS),
            regrid=AMRRegrid(schedule=every(100, clock=program.clock)), transfer=transfer,
            execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)))
    initial = np.broadcast_to(1. + (np.arange(n) + .5) / n, (1, n, n)).copy()
    return case, layout, right, left, initial
