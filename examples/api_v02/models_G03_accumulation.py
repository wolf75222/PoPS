"""The witness of G03 lot A (ROM-21): a nonlinear accumulation on the explicit route (note §6).

``nonlinear_inventory_witness``, below the surface: the state E holds U, the cell averages of
Q(T) = T + T² under ``cell_average(1, GaussLegendre(2))``; the coordinates T are recovered from U
on each point as the root of Q(T) = U on T >= -1/2; the conduction is -k ∂ₓT with no-flux ends, on
the 64 cells of N17's graded segment (widths in a ratio of about 97). The program is SSP-RK3 in its
increment form, U_new = U_old + Δ with Δ = dt (k0/6 + k1/6 + 2 k2/3); on uniform cells it keeps
U_old and Δ in its history, so that a test reads the increment that the commit adds.

``initial={T: e}`` is mapped to E by ``project_accumulation``: the rule of the representation
integrates Q(e(x)) on each cell, never Q of the average of e (K5, K6).
"""
from __future__ import annotations

import math

import numpy as np
import pops
from pops import math as pm
from pops._ir.local_root import LocalRoot
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.quadrature import GaussLegendre
from pops.mesh import CartesianGrid, MovingControlVolumes
from pops.numerics import Diffusion, DiscretizationPlan
from pops.physics.diffusion import DiffusiveBoundary
from pops.projection import ConservativeCellAverage
from pops.representations.definition import cell_average
from pops.time import FixedDt

from examples.api_v02.models_static_geometry import graded_map, graded_nodes

CELLS = 64
CONDUCTIVITY = 0.01
DT = 1.0e-5
STEPS = 20
LOWER = -0.5
TOLERANCE = 1.0e-13


def coordinate_profile(x):
    """e(x) = 0.3 + 0.2 sin(2 pi x), the initial value of the coordinates T."""
    return 0.3 + 0.2*np.sin(2*math.pi*np.asarray(x, dtype=float))


def project_accumulation(stored, profile, nodes):
    """The cell averages of Q(e(x)) under the rule of ``stored`` on the cells [nodes[i],
    nodes[i+1]]: encode(Q(e)), the projection that ``initial={T: e}`` asks for (K5)."""
    rule_nodes, weights, interval = stored.quadrature
    nodes = np.asarray(nodes, dtype=float)
    lower, upper = (float(value) for value in interval)
    weights = np.asarray([float(w) for w in weights])
    reference = (np.asarray([float(t) for t in rule_nodes]) - lower)/(upper - lower)
    points = nodes[:-1, None] + np.diff(nodes)[:, None]*reference[None, :]
    values = np.asarray(stored.law.eval({stored.coordinates.name: profile(points)}), dtype=float)
    return (values*weights[None, :]).sum(axis=1)/weights.sum()


def build_witness(*, recovery=None, admissible=(LOWER, None), quadrature=True, graded=True):
    """The case, its layout, the bound initial values, and the record of the accumulation.

    ``graded=False`` runs the same case on 64 uniform cells and keeps U_old and Δ in the history:
    a history is refused with the static geometry of the graded segment (N17), so the bitwise
    reading of the increment is made on the uniform cells."""
    frame = CartesianDomain("g03_a_inventory", (0.0,), (1.0,)).frame(Cartesian1D())
    model = pops.Model("G03_A_nonlinear_inventory", frame=frame)
    E = model.state("E", components=("e",))
    T = model.coordinate("T")
    representation = cell_average(1, GaussLegendre(2) if quadrature else None)
    stored = model.accumulation(
        E, coordinates=T, law=T + T*T, admissible=admissible, representation=representation,
        recovery=LocalRoot(tolerance=TOLERANCE, max_iterations=200)
        if recovery is None else recovery)
    walls = (DiffusiveBoundary(0, "lower", "conormal", 0.0),
             DiffusiveBoundary(0, "upper", "conormal", 0.0))
    conduction = model.diffusive_flux("conduction", state=E,
                                      value=pm.CoeffGradient(T, ((CONDUCTIVITY,),)),
                                      boundaries=walls)
    rate = model.rate("inventory", equation=pm.ddt(stored) == pm.div(conduction))
    case = pops.Case("G03_A_nonlinear_inventory_witness")
    block = case.block("rod", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, Diffusion(flux=conduction))
    case.numerics(plan, block=block)
    _ssprk3_increment(case, block, E, rate, DT, history=not graded)
    grid = Uniform(CartesianGrid(frame=frame, cells=(CELLS,)))
    if graded:
        layout = MovingControlVolumes(grid, evolution=graded_map(frame))
        nodes = graded_nodes(CELLS)
    else:
        layout, nodes = grid, np.arange(CELLS + 1)/CELLS
    initial = project_accumulation(stored, coordinate_profile, nodes)[None]
    case.initials.add(InitialCondition(state=block[E], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    return case, layout, {block[E]: np.ascontiguousarray(initial)}, initial, stored


def _ssprk3_increment(case, block, state, rate, dt, *, history):
    """SSP-RK3 written as U_new = U_old + Δ, with U_old and Δ kept in the history if asked."""
    name = "G03_A_SSPRK3"
    program = pops.Program(name)
    temporal = program.state(block[state])
    u0 = temporal.n
    k0 = program.value(name + "_k0", rate(u0), at=program.stage(name + "_stage0", c=0))
    point1 = program.stage(name + "_stage1", c=1)
    u1 = program.value(name + "_u1", u0 + program.dt*k0, at=point1)
    k1 = program.value(name + "_k1", rate(u1), at=point1)
    point2 = program.stage(name + "_stage2", c=0.5)
    u2 = program.value(name + "_u2", u0 + program.dt*(0.25*k0 + 0.25*k1), at=point2)
    k2 = program.value(name + "_k2", rate(u2), at=point2)
    delta = program.value(name + "_delta",
                          program.dt*(k0/6 + k1/6 + (2/3)*k2), at=temporal.next.point)
    if history:
        program.store_history("previous", u0)
        program.store_history("increment", delta)
    program.commit(temporal.next, program.value(name + "_accepted", u0 + delta,
                                                at=temporal.next.point))
    program.step_strategy(FixedDt(dt))
    case.program(program)
    return program
