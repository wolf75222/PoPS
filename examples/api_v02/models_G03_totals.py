"""Cases of G03 lot R-b (ROM-21): a state represented by cell totals (note §4.3, §4.6).

Each builder states the representation of its state with ``plan.representations.add``:
``cell_total(1)`` holds U_K = ∫_K u, ``cell_average(1)`` (or no row) holds the average. The
total's lowering decodes ū = U/μ(K), runs today's rate on ū and encodes the rate of U as μ(K)
times it, so a run of the total divided by the cell measures is the run of the average, to
rounding. The builders return the Case, its layout, the initial values that ``pops.bind``
receives (totals for a total) and the cell averages of the initial state.

- ``build_burgers``: Burgers on 16 uniform cells, an inflow value at the lower end and a closed
  upper end, or periodic;
- ``build_graded_burgers``: Burgers on the 32 graded cells of N17's segment (widths in a ratio of
  about 92), outflow ends;
- ``build_graded_diffusion_total``: N17's graded diffusion with no-flux ends, under a total.
"""
from __future__ import annotations

import math

import numpy as np
import pops
from pops import math as pm
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Inflow, NoFlux, Outflow
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, MovingControlVolumes, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.physics.diffusion import DiffusiveBoundary
from pops.projection import ConservativeCellAverage
from pops.representations.definition import cell_average, cell_total
from pops.time import FixedDt

from examples.api_v02.models_static_geometry import (
    CONDUCTIVITY, GRADED_CELLS, graded_map, graded_nodes)

BURGERS_CELLS = 16
BURGERS_DT = 0.01
BURGERS_STEPS = 12
INFLOW_VALUE = 1.5
GRADED_BURGERS_DT = 2.0e-4
GRADED_BURGERS_STEPS = 10
DIFFUSION_DT = 1.0e-5
DIFFUSION_STEPS = 8

_REPRESENTATIONS = {"total": cell_total(1), "average": cell_average(1), None: None}


def _ssprk2(case, block, state, rate, dt, name, step=None):
    """Two authored stages, the SSPRK2 of the corpus builders."""
    program = pops.Program(name)
    temporal = program.state(block[state])
    u0 = temporal.n
    k0 = program.value(name + "_k0", rate(u0), at=program.stage(name + "_stage0", c=0))
    point1 = program.stage(name + "_stage1", c=1)
    u1 = program.value(name + "_predictor", u0 + program.dt*k0, at=point1)
    k1 = program.value(name + "_k1", rate(u1), at=point1)
    program.commit(temporal.next, program.value(name + "_accepted", .5*u0 + .5*(u1 + program.dt*k1),
                                                at=temporal.next.point))
    program.step_strategy(FixedDt(dt) if step is None else step)
    case.program(program)
    return program


def _represent(plan, state, represent):
    if represent not in _REPRESENTATIONS:
        raise ValueError("represent is 'total', 'average' or None")
    if _REPRESENTATIONS[represent] is not None:
        plan.representations.add(state, _REPRESENTATIONS[represent])


def _bound(case, subject, averages, measures, represent, initial="bound", frame=None):
    """The arrays that bind receives: the totals mu(K) u_K under a total, the averages otherwise.
    ``initial="analytic"`` projects the averages of u = 1 + 0.5 sin(2 pi x) on b6's analytic route
    instead, which a total refuses (it projects averages)."""
    if initial == "analytic":
        from pops.analytic import coordinates, sin
        from pops.lib.initial import Analytic

        (x,) = coordinates(frame)
        case.initials.add(InitialCondition(
            state=subject, value=Analytic(frame=frame, components=(1.0 + 0.5*sin(2*math.pi*x),)),
            projection=ConservativeCellAverage()))
        return {}
    values = np.asarray(averages, dtype=float)
    if represent == "total":
        values = values*np.asarray(measures, dtype=float)
    case.initials.add(InitialCondition(state=subject, value=BindArray(),
                                       projection=ConservativeCellAverage()))
    return {subject: np.ascontiguousarray(values)}


def _burgers_model(name, frame):
    (x,) = frame.axes
    model = pops.Model(name, frame=frame)
    U = model.state("u", components=("u",))
    (u,) = U
    flux = model.flux("burgers_flux", state=U, frame=frame, components={x: (0.5*u*u,)},
                      waves={x: (u,)})
    rate = model.rate("burgers", equation=pm.ddt(U) == -pm.div(flux))
    return model, U, flux, rate


def _finite_volume(flux, U):
    return FiniteVolume(flux=flux, variables=variables.Conservative(U),
                        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov())


def burgers_centres(cells=BURGERS_CELLS):
    return (np.arange(cells) + 0.5)/cells


def build_burgers(*, represent="total", periodic=False, step=None, initial="bound"):
    """Burgers on 16 uniform cells of [0, 1]: u = 1 + 0.5 sin(2 pi x).

    Not periodic, the lower end holds the inflow value 1.5 (the ghost 2g - u) and the upper end is
    closed (NoFlux); periodic, the ends are identified."""
    frame = CartesianDomain("g03_rb_burgers", (0.0,), (1.0,)).frame(Cartesian1D())
    model, U, flux, rate = _burgers_model("G03_Rb_burgers", frame)
    case = pops.Case("G03_Rb_burgers_%s_%s" % (represent, "periodic" if periodic else "walls"))
    block = case.block("line", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, _finite_volume(flux, U))
    _represent(plan, U, represent)
    boundaries = frame.boundaries
    axes = PeriodicAxes((frame.axes[0],)) if periodic else None
    if not periodic:
        plan.boundaries.add(TransportBoundarySet({
            boundaries.x_min: Inflow(state=block[U], value=INFLOW_VALUE),
            boundaries.x_max: NoFlux(state=block[U]),
        }))
    case.numerics(plan, block=block)
    _ssprk2(case, block, U, rate, BURGERS_DT, "G03_Rb_burgers_SSPRK2", step)
    layout = Uniform(CartesianGrid(frame=frame, cells=(BURGERS_CELLS,), periodic=axes))
    averages = (1.0 + 0.5*np.sin(2*math.pi*burgers_centres()))[None]
    measures = np.full(BURGERS_CELLS, 1.0/BURGERS_CELLS)
    return case, layout, _bound(case, block[U], averages, measures, represent, initial,
                                frame), averages


def build_graded_burgers(*, represent="total", step=None):
    """Burgers on the 32 graded cells of N17's segment, u = 1 + 0.5 cos(pi x), outflow ends."""
    frame = CartesianDomain("g03_rb_graded_logical", (0.0,), (1.0,)).frame(Cartesian1D())
    model, U, flux, rate = _burgers_model("G03_Rb_graded_burgers", frame)
    case = pops.Case("G03_Rb_graded_burgers_%s" % represent)
    block = case.block("line", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, _finite_volume(flux, U))
    _represent(plan, U, represent)
    boundaries = frame.boundaries
    plan.boundaries.add(TransportBoundarySet({
        boundaries.x_min: Outflow(state=block[U]),
        boundaries.x_max: Outflow(state=block[U]),
    }))
    case.numerics(plan, block=block)
    _ssprk2(case, block, U, rate, GRADED_BURGERS_DT, "G03_Rb_graded_SSPRK2", step)
    layout = MovingControlVolumes(Uniform(CartesianGrid(frame=frame, cells=(GRADED_CELLS,))),
                                  evolution=graded_map(frame))
    nodes = graded_nodes()
    centres = (nodes[:-1] + nodes[1:])/2
    averages = (1.0 + 0.5*np.cos(math.pi*centres))[None]
    return case, layout, _bound(case, block[U], averages, np.diff(nodes), represent), averages


def build_graded_diffusion_total(*, represent="total"):
    """N17's diffusion with no-flux ends on the 32 graded cells, under a total."""
    frame = CartesianDomain("g03_rb_graded_heat", (0.0,), (1.0,)).frame(Cartesian1D())
    model = pops.Model("G03_Rb_graded_diffusion", frame=frame)
    U = model.state("heat", components=("u",))
    walls = (DiffusiveBoundary(0, "lower", "conormal", 0.0),
             DiffusiveBoundary(0, "upper", "conormal", 0.0))
    flux = model.diffusive_flux("conduction", state=U,
                                value=pm.CoeffGradient(U[0], ((CONDUCTIVITY,),)),
                                boundaries=walls)
    rate = model.rate("heat_balance", equation=pm.ddt(U) == pm.div(flux))
    case = pops.Case("G03_Rb_graded_diffusion_%s" % represent)
    block = case.block("rod", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, Diffusion(flux=flux))
    _represent(plan, U, represent)
    case.numerics(plan, block=block)
    _ssprk2(case, block, U, rate, DIFFUSION_DT, "G03_Rb_graded_heat_SSPRK2")
    layout = MovingControlVolumes(Uniform(CartesianGrid(frame=frame, cells=(GRADED_CELLS,))),
                                  evolution=graded_map(frame))
    nodes = graded_nodes()
    centres = (nodes[:-1] + nodes[1:])/2
    averages = (1.0 + 0.5*np.cos(math.pi*centres))[None]
    return case, layout, _bound(case, block[U], averages, np.diff(nodes), represent), averages
