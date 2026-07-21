#!/usr/bin/env python3
"""Advection scalaire 2D MPI avec les briques preimplementees de PoPS.

Le fichier se lit de haut en bas. Python decrit le probleme, puis PoPS compile et execute les
operateurs en C++/Kokkos et distribue le domaine avec le MPI natif de PoPS. Il n'y a aucune boucle
Python sur les cellules ou les pas de temps et aucun recours a mpi4py.
"""

# ruff: noqa: E402

import numpy as np

import pops

pops.set_threads(1)

from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Inflow, Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import SSPRK2
from pops.math import ddt, div
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import AdaptiveCFL


# Les valeurs faciles a modifier sont regroupees ici, sans interface en ligne de commande.
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


# 2. Modele physique : d_t U + div(F) = 0 avec F(U) = a U.
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


# 3. Methode spatiale : volumes finis, MUSCL-Van Leer et flux upwind scalaire.
finite_volume = FiniteVolume(
    flux=physical_flux,
    variables=variables.Conservative(U),
    reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
    riemann=riemann.ScalarUpwind(velocity=velocity),
)

numerics = DiscretizationPlan()
numerics.rates.add(advection_rate, finite_volume)


# 4. Le Case instancie le modele et fournit le handle qualifie du traceur.
case = pops.Case("tutorial_scalar_advection")
tracer = case.block("tracer", model=model)
tracer_U = tracer[U]


# 5. Conditions aux limites. AX et AY sont positifs : les faces min sont entrantes.
boundaries = frame.boundaries
transport_boundaries = TransportBoundarySet({
    boundaries.x_min: Inflow(state=tracer_U, value=FAR_FIELD),
    boundaries.x_max: Outflow(state=tracer_U),
    boundaries.y_min: Inflow(state=tracer_U, value=FAR_FIELD),
    boundaries.y_max: Outflow(state=tracer_U),
})

numerics.boundaries.add(transport_boundaries)
case.numerics(numerics, block=tracer)


# 6. Programme temporel preimplemente : SSPRK2, avec pas choisi par la CFL native.
program = SSPRK2(tracer_U, rate=advection_rate)
program.step_strategy(AdaptiveCFL(cfl=CFL, max_dt=MAX_DT))
case.program(program)


# 7. Condition initiale : une bosse gaussienne fournie une seule fois au bind.
x = (np.arange(NX, dtype=np.float64) + 0.5) / NX
y = (np.arange(NY, dtype=np.float64) + 0.5) / NY
xx, yy = np.meshgrid(x, y, indexing="xy")

initial_u = FAR_FIELD + 0.95 * np.exp(
    -120.0 * ((xx - 0.30) ** 2 + (yy - 0.35) ** 2)
)
initial_state = np.ascontiguousarray(initial_u[np.newaxis, :, :], dtype=np.float64)


# 8. Cycle public final : validate -> resolve -> compile -> bind -> run.
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
