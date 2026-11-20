"""A total transported on a moving configuration, rank 1 (G03 lot M1b, ROM-21).

Note §7.2, §7.6: the state U of block "line" holds cell totals (``cell_total(1)``); the members
(x_left, x_right, V) of a moving configuration are the state of block "configuration" (lot M1a).
One SSP-RK3 step advances both. At each stage the members' rate k_M is evaluated first, then the
rate of U reads U, the members and k_M of the same stage:

    ū_K = U_K / V_K,   U_K' = -(F_{K+1/2} - F_{K-1/2}),

with F the Rusanov flux of f(ū) - w ū at the speed |λ - w|, w the node rate read on the copies of
each cell, kept in its physical and swept parts until the difference of the cell. The node rate is w(x, t) = A sin(2 pi x) cos(2 pi t) on [0, 1], zero at both ends.

The physics is Burgers, f(u) = u²/2, λ = u, or the constant advection f(u) = a u (``law=``);
the ends are outflow. ``initial="constant"`` starts from ū = 2, a power of two, so U = 2 V holds
bit for bit along the run (CF11).
"""
from __future__ import annotations

import math

import numpy as np
import pops
from pops import math as pm
from pops.analytic import coordinate, cos, sin, time
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Outflow
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, MovingControlVolumes
from pops.mesh.moving_configuration import MovingConfiguration
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.representations.definition import cell_average, cell_total
from pops.time import FixedDt

from examples.api_v02.models_static_geometry import graded_map

CELLS = 32
AMPLITUDE = 0.05
ADVECTION = 0.5
DT = 1.0/160
STEPS = 16
GRADED_DT = 5.0e-4
CONSTANT = 2.0

_REPRESENTATIONS = {"total": cell_total(1), "average": cell_average(1)}


def _model(name, frame, law):
    (x,) = frame.axes
    model = pops.Model(name, frame=frame)
    U = model.state("u", components=("u",))
    (u,) = U
    if law == "burgers":
        components, waves = (0.5*u*u,), (u,)
    elif law == "advection":
        components, waves = (ADVECTION*u,), (ADVECTION + 0*u,)
    else:
        raise ValueError(law)
    flux = model.flux("transport_flux", state=U, frame=frame, components={x: components},
                      waves={x: waves})
    rate = model.rate("transport", equation=pm.ddt(U) == -pm.div(flux))
    return model, U, flux, rate


def initial_density(centres, initial):
    if initial == "constant":
        return np.full_like(centres, CONSTANT)
    if initial == "wave":
        return 1.0 + 0.5*np.sin(2*math.pi*centres)
    raise ValueError(initial)


def build_transport(*, law="burgers", initial="wave", base="uniform", represent="total",
                    dt=DT, strategy=None, static_rate=False, face="rusanov"):
    """The case, its layout, its initial values and its configuration.

    ``static_rate`` writes the rate of U as the static face pass ``rate(u)`` instead of the
    configuration's transport, and ``face="hll"`` declares an HLL face, for the refusal tests.

    The dt margin: on the uniform base the cells are 1/32 wide and the speeds |λ - w| stay under
    1.6, a Courant number of 0.32 at dt = 1/160; on the graded base the smallest cell is about
    0.00102 wide and |λ - w| stays under 0.55, a Courant number of 0.27 at dt = 5e-4."""
    frame = CartesianDomain("g03_m1b_transport", (0.0,), (1.0,)).frame(Cartesian1D())
    grid = Uniform(CartesianGrid(frame=frame, cells=(CELLS,)))
    reference = grid if base == "uniform" else MovingControlVolumes(grid, evolution=graded_map(frame))
    case = pops.Case("G03_M1b_%s_%s_%s" % (law, initial, base))
    name = "G03_M1b_SSPRK3"
    program = pops.Program(name)
    x, t = coordinate(frame, frame.axes[0]), time(program.clock)
    configuration = MovingConfiguration(
        reference, node_rate=AMPLITUDE*sin(2*math.pi*x)*cos(2*math.pi*t))
    members = configuration.members(case)
    model, U, flux, rate = _model("G03_M1b_%s" % law, frame, law)
    block = case.block("line", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(U),
                                      reconstruction=reconstruction.FirstOrder(),
                                      riemann=riemann.Rusanov() if face == "rusanov"
                                      else riemann.HLL()))
    plan.representations.add(U, _REPRESENTATIONS[represent])
    boundaries = frame.boundaries
    plan.boundaries.add(TransportBoundarySet({boundaries.x_min: Outflow(state=block[U]),
                                              boundaries.x_max: Outflow(state=block[U])}))
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[U], value=BindArray(),
                                       projection=ConservativeCellAverage()))

    totals, geometry = program.state(block[U]), program.state(members.block[members.state])
    transport = members.transport(rate)

    def rates(u, m, point, label):
        k_m = program.value("%s_km%s" % (name, label), members.rate(m), at=point)
        if static_rate:
            k_u = program.value("%s_ku%s" % (name, label), rate(u), at=point)
        else:
            k_u = program.value("%s_ku%s" % (name, label), transport(u, m, k_m), at=point)
        return k_u, k_m

    u0, m0 = totals.n, geometry.n
    point0 = program.stage(name + "_stage0", c=0)
    ku0, km0 = rates(u0, m0, point0, "0")
    point1 = program.stage(name + "_stage1", c=1)
    u1 = program.value(name + "_u1", u0 + program.dt*ku0, at=point1)
    m1 = program.value(name + "_m1", m0 + program.dt*km0, at=point1)
    ku1, km1 = rates(u1, m1, point1, "1")
    point2 = program.stage(name + "_stage2", c=0.5)
    u2 = program.value(name + "_u2", u0 + program.dt*(0.25*ku0 + 0.25*ku1), at=point2)
    m2 = program.value(name + "_m2", m0 + program.dt*(0.25*km0 + 0.25*km1), at=point2)
    ku2, km2 = rates(u2, m2, point2, "2")
    weights = (1/6, 1/6, 2/3)
    program.commit_many({
        totals.next: program.value(name + "_u_next", u0 + program.dt*(
            weights[0]*ku0 + weights[1]*ku1 + weights[2]*ku2), at=totals.next.point),
        geometry.next: program.value(name + "_m_next", m0 + program.dt*(
            weights[0]*km0 + weights[1]*km1 + weights[2]*km2), at=geometry.next.point),
    })
    program.step_strategy(FixedDt(dt) if strategy is None else strategy)
    case.program(program)
    start = configuration.initial_members()
    centres = 0.5*(start[0] + start[1])
    values = members.initial_values()
    values[block[U]] = np.ascontiguousarray((initial_density(centres, initial)*start[2])[None])
    return case, configuration.layout, values, configuration


def _rusanov(law, left, right, w):
    """The two parts of the Rusanov flux of f(u) - w u at the speed |λ - w|: the physical part
    (f(u_L) + f(u_R))/2 - a (u_R - u_L)/2 and the swept part w (u_L + u_R)/2."""
    if law == "burgers":
        fl, fr, sl, sr = 0.5*left*left, 0.5*right*right, left, right
    else:
        fl, fr, sl, sr = ADVECTION*left, ADVECTION*right, ADVECTION + 0*left, ADVECTION + 0*right
    speed = np.maximum(np.abs(sl - w), np.abs(sr - w))
    return 0.5*(fl + fr) - 0.5*speed*(right - left), w*(0.5*(left + right))


def ssprk3_oracle(configuration, steps, dt, *, law="burgers", initial="wave"):
    """An independent NumPy assembly of the route: the members advanced as in lot M1a, the totals
    by the relative Rusanov flux on ū = U/V with outflow ends, w read on the nodes of each stage.
    Returns the totals, the members and the budget of the ends: the integral over the run of the
    flux through the two ends, by the weights of the method."""
    reference = configuration.reference_nodes()
    nodes = configuration.base_nodes()
    members = np.stack((nodes[:-1], nodes[1:], np.diff(nodes)))
    centres = 0.5*(members[0] + members[1])
    totals = initial_density(centres, initial)*members[2]

    def rates(totals, members, t):
        wl = configuration.node_rate_at(members[0], reference[:-1], t)
        wr = configuration.node_rate_at(members[1], reference[1:], t)
        density = totals/members[2]
        padded = np.concatenate((density[:1], density, density[-1:]))
        w = np.concatenate((wl, wr[-1:]))
        physical, swept = _rusanov(law, padded[:-1], padded[1:], w)
        ends = -(physical[-1] - physical[0]) + (swept[-1] - swept[0])
        return (-(physical[1:] - physical[:-1]) + (swept[1:] - swept[:-1]),
                np.stack((wl, wr, wr - wl)), ends)

    t, budget = 0.0, 0.0
    for _ in range(steps):
        ku0, km0, e0 = rates(totals, members, t)
        u1, m1 = totals + dt*ku0, members + dt*km0
        ku1, km1, e1 = rates(u1, m1, t + dt)
        u2, m2 = totals + dt*(0.25*ku0 + 0.25*ku1), members + dt*(0.25*km0 + 0.25*km1)
        ku2, km2, e2 = rates(u2, m2, t + 0.5*dt)
        totals = totals + dt*(ku0/6 + ku1/6 + (2/3)*ku2)
        members = members + dt*(km0/6 + km1/6 + (2/3)*km2)
        budget += dt*(e0/6 + e1/6 + (2/3)*e2)
        t += dt
    return totals, members, budget
