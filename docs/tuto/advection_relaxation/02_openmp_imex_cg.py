#!/usr/bin/env python3
"""Advection explicite et diffusion-relaxation implicite avec le CG natif.

Cette seconde version utilise un solve global car la diffusion couple les cellules. Le programme
reste lineaire : prediction explicite, operateur Helmholtz matrix-free, CG puis commit.
"""

# ruff: noqa: E402

import numpy as np
import pops

pops.set_threads(7)

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.linalg import LinearOperatorProperties, LinearProblem
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.representations import Conservative
from pops.runtime_environment import runtime_environment_report
from pops.solvers import CG
from pops.spaces import CellState
from pops.time import FailRun, FixedDt


NX = 32
NY = 32
AX = 1.0
AY = 0.25
DIFFUSIVITY = 2.0e-3
RELAXATION_RATE = 2.0
DT = 2.0e-3
T_END = 2.0e-2
MAX_STEPS = 100


# 1. Domaine periodique et grille uniforme.
domain = Rectangle(
    "unit_square",
    lower=(0.0, 0.0),
    upper=(1.0, 1.0),
).tag("fluid")
frame = domain.frame(Cartesian2D())
x_axis, y_axis = frame.axes
grid = CartesianGrid(
    frame=frame,
    cells=(NX, NY),
    periodic=PeriodicAxes(frame.axes),
)


# 2. Transport explicite.
model = pops.Model("advection_diffusion_relaxation", frame=frame)
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

finite_volume = FiniteVolume(
    flux=physical_flux,
    variables=variables.Conservative(U),
    reconstruction=reconstruction.FirstOrder(),
    riemann=riemann.ScalarUpwind(velocity=velocity),
)
numerics = DiscretizationPlan()
numerics.rates.add(advection_rate, finite_volume)


# 3. Case et etat qualifie.
case = pops.Case("tutorial_advection_diffusion_relaxation")
tracer = case.block("tracer", model=model)
tracer_U = tracer[U]
case.numerics(numerics, block=tracer)


# 4. Prediction explicite d'IMEX Euler.
program = pops.Program("IMEX Euler with CG")
q = program.state(tracer_U)
explicit_rate = program.value(
    "explicit_advection",
    advection_rate(q.n),
    at=q.n.point,
)
explicit_predictor = program.value(
    "explicit_predictor",
    q.n + program.dt * explicit_rate,
    at=q.next.point,
)


# 5. A(v) = (1 + lambda dt) v - kappa dt Lap(v).
implicit_operator = program.matrix_free_operator(
    "implicit_helmholtz",
    domain="state",
    range_="state",
    ncomp=1,
)
program.set_apply(
    implicit_operator,
    lambda builder, _out, value: (
        (1.0 + RELAXATION_RATE * builder.dt) * value
        - DIFFUSIVITY
        * builder.dt
        * builder.laplacian(
            builder.scalar_field("laplacian"),
            value,
        )
    ),
)


# 6. CG resout le systeme SPD dans la boucle C++ native.
next_state = program.solve(
    LinearProblem(
        implicit_operator,
        explicit_predictor,
        at=q.next.point,
        properties=LinearOperatorProperties.symmetric_positive_definite(),
        nullspace=None,
    ),
    solver=CG(max_iter=80, rel_tol=1.0e-10),
    name="implicit_diffusion_relaxation",
).consume(action=FailRun())

program.commit(q.next, next_state)
program.step_strategy(FixedDt(DT))
case.program(program)


# 7. Condition initiale gaussienne.
x = (np.arange(NX, dtype=np.float64) + 0.5) / NX
y = (np.arange(NY, dtype=np.float64) + 0.5) / NY
xx, yy = np.meshgrid(x, y, indexing="xy")
initial_u = 0.05 + 0.95 * np.exp(
    -100.0 * ((xx - 0.30) ** 2 + (yy - 0.35) ** 2)
)
initial_state = np.ascontiguousarray(initial_u[np.newaxis, :, :])


# 8. Cycle public final.
validated = pops.validate(case)
resolved = pops.resolve(validated, layout=Uniform(grid))
artifact = pops.compile(resolved)
simulation = pops.bind(
    artifact,
    initial_state={"tracer": initial_state},
)
report = pops.run(
    simulation,
    t_end=T_END,
    max_steps=MAX_STEPS,
)

final_state = np.asarray(
    simulation.state_global("tracer"), dtype=np.float64
).reshape(initial_state.shape)

print("PoPS global IMEX/Krylov tutorial finished")
print("  program        : explicit pops.Program")
print("  implicit solve : native matrix-free CG")
print("  Kokkos backend : %s" % runtime_environment_report()["kokkos_backend"])
print("  accepted steps : %d" % report.accepted_steps)
print("  final time     : %.6f" % simulation.time())
print("  initial L2     : %.12e" % np.linalg.norm(initial_state))
print("  final L2       : %.12e" % np.linalg.norm(final_state))
