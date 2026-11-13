"""The members X and V of a moving configuration, rank 1, without transport (G03 lot M1a, ROM-21).

Note §7.2: the corner copies X = (x_left, x_right) and the volume V of each cell are the state of a
block of their own, advanced by the step's own method, here SSP-RK3:

    x_left' = w(x_left, t),   x_right' = w(x_right, t),   V' = w(x_right, t) - w(x_left, t).

The node rate takes the two forms of §7.4 on [0, 1]:

- ``expression``: w(x, t) = A sin(2 pi x) cos(2 pi t), read at each copy's own position;
- ``map``: the coordinate map Φ(ξ, t) = ξ + A sin(2 pi ξ) sin(2 pi t) / (2 pi) of b6, whose
  node rate ∂ₜΦ(ξ, t) = A sin(2 pi ξ) cos(2 pi t) is derived symbolically and read at the reference
  coordinate of each node; the nodes then follow the map to the order of the method.

The base is 32 uniform cells, or N17's graded segment (``base="graded"``).
"""
from __future__ import annotations

import math

import numpy as np
import pops
from pops.analytic import coordinate, cos, sin, time
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, GeometryEvolution, MovingControlVolumes
from pops.mesh.moving_configuration import MovingConfiguration
from pops.time import FixedDt, RejectAttempt

from examples.api_v02.models_static_geometry import graded_map

CELLS = 32
AMPLITUDE = 0.05
DT = 1.0/40
STEPS = 10


def node_rate(frame, clock, form):
    """w of form ``expression``, or the map Φ of form ``map``, on ``frame`` and ``clock``."""
    x, t = coordinate(frame, frame.axes[0]), time(clock)
    if form == "expression":
        return AMPLITUDE*sin(2*math.pi*x)*cos(2*math.pi*t)
    if form == "map":
        return GeometryEvolution((x + AMPLITUDE*sin(2*math.pi*x)*sin(2*math.pi*t)/(2*math.pi),))
    if form == "rest":
        return 0*x*cos(2*math.pi*t)
    raise ValueError(form)


def build_members(*, form="expression", base="uniform", dt=DT, strategy=None, reject=False,
                  layout=None):
    """The case, its layout, its initial values and its configuration.

    ``strategy`` replaces ``FixedDt(dt)``; ``reject`` guards the candidate with a condition that
    never holds, so that every attempt is rejected; ``layout`` replaces the base given to resolve.
    """
    frame = CartesianDomain("g03_m1a_members", (0.0,), (1.0,)).frame(Cartesian1D())
    grid = Uniform(CartesianGrid(frame=frame, cells=(CELLS,)))
    reference = grid if base == "uniform" else MovingControlVolumes(grid, evolution=graded_map(frame))
    case = pops.Case("G03_M1a_members_%s_%s" % (form, base))
    name = "G03_M1a_SSPRK3"
    program = pops.Program(name)
    configuration = MovingConfiguration(reference, node_rate=node_rate(frame, program.clock, form))
    members = configuration.members(case)
    if members is None:
        return case, configuration.layout, {}, configuration
    temporal = program.state(members.block[members.state])
    u0 = temporal.n
    rate = members.rate
    k0 = program.value(name + "_k0", rate(u0), at=program.stage(name + "_stage0", c=0))
    point1 = program.stage(name + "_stage1", c=1)
    u1 = program.value(name + "_u1", u0 + program.dt*k0, at=point1)
    k1 = program.value(name + "_k1", rate(u1), at=point1)
    point2 = program.stage(name + "_stage2", c=0.5)
    u2 = program.value(name + "_u2", u0 + program.dt*(0.25*k0 + 0.25*k1), at=point2)
    k2 = program.value(name + "_k2", rate(u2), at=point2)
    candidate = program.value(name + "_candidate", u0 + program.dt*(k0/6 + k1/6 + (2/3)*k2),
                              at=temporal.next.point)
    if reject:
        increment = program.value(name + "_increment", candidate - u0, at=temporal.next.point)
        candidate = program.guard("never_accepted", candidate,
                                  program.norm_inf(increment) < 0.0, action=RejectAttempt())
    program.commit(temporal.next, candidate)
    program.step_strategy(FixedDt(dt) if strategy is None else strategy)
    case.program(program)
    return case, configuration.layout if layout is None else layout, \
        members.initial_values(), configuration


def ssprk3_oracle(configuration, steps, dt):
    """An independent NumPy run of the members: the nodes advanced by SSP-RK3 under w, the volume
    by the rate w(x_right) - w(x_left) of each cell, with the stage times 0, 1 and 1/2."""
    nodes = configuration.base_nodes()
    reference = configuration.reference_nodes()
    left, right = nodes[:-1].copy(), nodes[1:].copy()
    volume = right - left

    def rates(left, right, t):
        wl = configuration.node_rate_at(left, reference[:-1], t)
        wr = configuration.node_rate_at(right, reference[1:], t)
        return np.stack((wl, wr, wr - wl))

    state = np.stack((left, right, volume))
    t = 0.0
    for _ in range(steps):
        k0 = rates(state[0], state[1], t)
        u1 = state + dt*k0
        k1 = rates(u1[0], u1[1], t + dt)
        u2 = state + dt*(0.25*k0 + 0.25*k1)
        k2 = rates(u2[0], u2[1], t + 0.5*dt)
        state = state + dt*(k0/6 + k1/6 + (2/3)*k2)
        t += dt
    return state
