#!/usr/bin/env python3
"""Advection scalaire 2D OpenMP avec les briques preimplementees de PoPS.

Le fichier se lit de haut en bas. Python decrit le probleme, puis PoPS compile et execute les
operateurs en C++/Kokkos OpenMP sur exactement sept threads. Il n'y a aucune boucle Python sur les
cellules ou les pas de temps.
"""

# ruff: noqa: E402

from pathlib import Path

import numpy as np

import pops

# La configuration OpenMP est une autorite publique explicite. Cet appel precede toute
# initialisation native de Kokkos.
pops.set_threads(7)

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
GAUSSIAN_AMPLITUDE = 0.95
GAUSSIAN_BETA = 120.0
GAUSSIAN_CENTER_X = 0.30
GAUSSIAN_CENTER_Y = 0.35
CFL = 0.45
MAX_DT = 1.0e-2
T_END = 0.20
MAX_STEPS = 10_000

HERE = Path(__file__).resolve().parent
RESULT_FILE = HERE / "results" / "01_openmp_preset_ssprk2.npz"


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

initial_u = FAR_FIELD + GAUSSIAN_AMPLITUDE * np.exp(
    -GAUSSIAN_BETA * (
        (xx - GAUSSIAN_CENTER_X) ** 2
        + (yy - GAUSSIAN_CENTER_Y) ** 2
    )
)
initial_state = np.ascontiguousarray(initial_u[np.newaxis, :, :], dtype=np.float64)


# 8. Cycle public final : validate -> resolve -> compile -> bind -> run.
layout = Uniform(grid)
validated = pops.validate(case)
resolved = pops.resolve(validated, layout=layout)
artifact = pops.compile(resolved)
simulation = pops.bind(
    artifact,
    initial_state={"tracer": initial_state},
)

pops.run(
    simulation,
    t_end=T_END,
    max_steps=MAX_STEPS,
)


# 9. Une copie globale est rapatriee seulement apres le calcul pour tracer et comparer.
final_state = np.asarray(
    simulation.state_global("tracer"),
    dtype=np.float64,
).reshape(initial_state.shape)

RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    RESULT_FILE,
    initial=initial_state,
    final=final_state,
    nx=NX,
    ny=NY,
    ax=AX,
    ay=AY,
    far_field=FAR_FIELD,
    gaussian_amplitude=GAUSSIAN_AMPLITUDE,
    gaussian_beta=GAUSSIAN_BETA,
    gaussian_center_x=GAUSSIAN_CENTER_X,
    gaussian_center_y=GAUSSIAN_CENTER_Y,
    cfl=CFL,
    t_end=T_END,
)

print("  result           : %s" % RESULT_FILE)
