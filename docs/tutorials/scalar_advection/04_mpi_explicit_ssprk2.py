#!/usr/bin/env python3
"""Advection scalaire 2D MPI avec SSPRK2 ecrit explicitement dans pops.Program.

Le script reste top-level et lineaire. Les expressions Python construisent le graphe temporel ;
les stages et les operateurs sont ensuite compiles et executes en C++/Kokkos avec le MPI natif de
PoPS, sans mpi4py.
"""

# ruff: noqa: E402

from fractions import Fraction

import numpy as np

import pops

pops.set_threads(1)

from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Inflow, Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import AdaptiveCFL, StagePoint, TimePoint


NX = 64
NY = 64
AX = 1.0
AY = 0.25
FAR_FIELD = 0.05
CFL = 0.45
MAX_DT = 1.0e-2
T_END = 0.20
MAX_STEPS = 10_000

# 1. Domaine continu, repere cartesien et grille de volumes finis.
domain = Rectangle(
    "unit_square",
    lower=(0.0, 0.0),
    upper=(1.0, 1.0),
).tag("fluid")

frame = domain.frame(Cartesian2D())
x_axis, y_axis = frame.axes
grid = CartesianGrid(frame=frame, cells=(NX, NY))


# 2. Modele physique et flux utilisateur F(U) = a U.
model = pops.Model("scalar_advection", frame=frame)

U = model.state(
    "U",
    components=("u",),
    representation=Conservative(),
    space=CellState(frame=frame),
)
(u,) = U

velocity = model.vector(
    "a",
    frame=frame,
    components={x_axis: AX, y_axis: AY},
)

physical_flux = model.flux(
    "advection_flux",
    frame=frame,
    state=U,
    components={x_axis: (AX * u,), y_axis: (AY * u,)},
    waves={x_axis: (AX,), y_axis: (AY,)},
)

advection_rate = model.rate(
    "advection_rate",
    equation=ddt(U) == -div(physical_flux),
)


# 3. Methode spatiale identique au premier tutoriel.
finite_volume = FiniteVolume(
    flux=physical_flux,
    variables=variables.Conservative(U),
    reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
    riemann=riemann.ScalarUpwind(velocity=velocity),
)

numerics = DiscretizationPlan()
numerics.rates.add(advection_rate, finite_volume)


# 4. Case, bloc qualifie et conditions aux limites.
case = pops.Case("tutorial_scalar_advection")
tracer = case.block("tracer", model=model)
tracer_U = tracer[U]

boundaries = frame.boundaries
transport_boundaries = TransportBoundarySet({
    boundaries.x_min: Inflow(state=tracer_U, value=FAR_FIELD),
    boundaries.x_max: Outflow(state=tracer_U),
    boundaries.y_min: Inflow(state=tracer_U, value=FAR_FIELD),
    boundaries.y_max: Outflow(state=tracer_U),
})

numerics.boundaries.add(transport_boundaries)
case.numerics(numerics, block=tracer)


# 5. SSPRK2 ecrit avec les operations generiques de Program.
program = pops.Program("SSPRK2")
q = program.state(tracer_U)

stage_0 = StagePoint(
    "ssprk2_stage_0",
    {"main": TimePoint(program.clock, 0)},
)
k0 = program.value(
    "ssprk2_k_0",
    advection_rate(q.n),
    at=stage_0,
)

stage_1 = StagePoint(
    "ssprk2_stage_1",
    {"main": TimePoint(program.clock, 1)},
)
q_stage = program.value(
    "ssprk2_U1",
    q.n + program.dt * k0,
    at=stage_1,
)
k1 = program.value(
    "ssprk2_k_1",
    advection_rate(q_stage),
    at=stage_1,
)

half = Fraction(1, 2)
q_next = program.value(
    "ssprk2_step",
    q.n + program.dt * half * k0 + program.dt * half * k1,
    at=q.next.point,
)

program.commit(q.next, q_next)
program.step_strategy(AdaptiveCFL(cfl=CFL, max_dt=MAX_DT))
case.program(program)


# 6. Meme condition initiale que dans le premier tutoriel.
x = (np.arange(NX, dtype=np.float64) + 0.5) / NX
y = (np.arange(NY, dtype=np.float64) + 0.5) / NY
xx, yy = np.meshgrid(x, y, indexing="xy")

initial_u = FAR_FIELD + 0.95 * np.exp(
    -120.0 * ((xx - 0.30) ** 2 + (yy - 0.35) ** 2)
)
initial_state = np.ascontiguousarray(initial_u[np.newaxis, :, :], dtype=np.float64)


# 7. Meme cycle public final que dans le premier tutoriel.
layout = Uniform(grid)
validated = pops.validate(case)
resolved = pops.resolve(validated, layout=layout)
artifact = pops.compile(resolved)

execution_context = pops.ExecutionContext.mpi_world(artifact)
simulation = pops.bind(
    artifact,
    initial_state={"tracer": initial_state},
    resources={"execution_context": execution_context},
)

pops.run(
    simulation,
    t_end=T_END,
    max_steps=MAX_STEPS,
)
