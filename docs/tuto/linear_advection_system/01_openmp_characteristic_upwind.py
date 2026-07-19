#!/usr/bin/env python3
"""Systeme lineaire 2D resolu par un flux upwind caracteristique.

Les deux inconnues sont transportees par des matrices pleines. Python decrit les matrices,
leurs vitesses caracteristiques et le schema ; PoPS compile ensuite le flux de Roe et toute
l'evolution en C++/Kokkos OpenMP.
"""

# ruff: noqa: E402

import numpy as np
import pops

pops.set_threads(7)

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import SSPRK2
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.representations import Conservative
from pops.runtime_environment import runtime_environment_report
from pops.spaces import CellState
from pops.time import FixedDt


NX = 48
NY = 48
DT = 2.0e-3
T_END = 4.0e-2
MAX_STEPS = 100

# A_x et A_y sont pleines et ne commutent pas : aucune base propre unique ne transforme donc
# le probleme 2D en deux transports scalaires independants. Leurs valeurs propres ont les deux
# signes, avec une onde dans chaque sens a toute interface.
ROOT_THREE = 3.0**0.5
A_X = ((0.25, 0.75), (0.75, 0.25))
A_Y = ((0.25, 0.15 * ROOT_THREE), (0.15 * ROOT_THREE, -0.05))
LAMBDA_X = (-0.50, 1.00)
LAMBDA_Y = (-0.20, 0.40)


# 1. Domaine periodique et grille de volumes finis.
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


# 2. Physique : d_t Q + d_x(A_x Q) + d_y(A_y Q) = 0.
model = pops.Model("linear_advection_system", frame=frame)
Q = model.state(
    "U",
    components=("q1", "q2"),
    representation=Conservative(),
    space=CellState(frame=frame),
)
q1, q2 = Q

physical_flux = model.flux(
    "matrix_transport_flux",
    frame=frame,
    state=Q,
    components={
        x_axis: (
            A_X[0][0] * q1 + A_X[0][1] * q2,
            A_X[1][0] * q1 + A_X[1][1] * q2,
        ),
        y_axis: (
            A_Y[0][0] * q1 + A_Y[0][1] * q2,
            A_Y[1][0] * q1 + A_Y[1][1] * q2,
        ),
    },
    waves={x_axis: LAMBDA_X, y_axis: LAMBDA_Y},
)

# Pour ce systeme lineaire, le Jacobien du flux est exactement A_x ou A_y. Le provider
# genere |A| (Q_R - Q_L), c'est-a-dire la dissipation upwind dans la base caracteristique.
model.roe_from_jacobian()

transport_rate = model.rate(
    "matrix_transport_rate",
    equation=ddt(Q) == -div(physical_flux),
)


# 3. Volumes finis et flux de Roe caracteristique natif.
finite_volume = FiniteVolume(
    flux=physical_flux,
    variables=variables.Conservative(Q),
    reconstruction=reconstruction.FirstOrder(),
    riemann=riemann.Roe(),
)
numerics = DiscretizationPlan()
numerics.rates.add(transport_rate, finite_volume)


# 4. Bloc physique et integration SSPRK2.
case = pops.Case("tutorial_linear_advection_system")
waves = case.block("waves", model=model)
waves_Q = waves[Q]
case.numerics(numerics, block=waves)

program = SSPRK2(waves_Q, rate=transport_rate)
program.step_strategy(FixedDt(DT))
case.program(program)


# 5. Seule q1 contient une bosse au depart. La matrice pleine doit donc creer q2.
x = (np.arange(NX, dtype=np.float64) + 0.5) / NX
y = (np.arange(NY, dtype=np.float64) + 0.5) / NY
xx, yy = np.meshgrid(x, y, indexing="xy")
initial_q1 = np.exp(-120.0 * ((xx - 0.35) ** 2 + (yy - 0.45) ** 2))
initial_q2 = np.zeros_like(initial_q1)
initial_state = np.ascontiguousarray(np.stack((initial_q1, initial_q2)))


# 6. Compilation et execution : validate -> resolve -> compile -> bind -> run.
validated = pops.validate(case)
resolved = pops.resolve(validated, layout=Uniform(grid))
artifact = pops.compile(resolved)
simulation = pops.bind(
    artifact,
    initial_state={"waves": initial_state},
)
report = pops.run(
    simulation,
    t_end=T_END,
    max_steps=MAX_STEPS,
)

final_state = np.asarray(
    simulation.state_global("waves"), dtype=np.float64
).reshape(initial_state.shape)


# 7. Sur un domaine periodique, chaque composante conservative garde son integrale.
initial_integrals = initial_state.sum(axis=(1, 2))
final_integrals = final_state.sum(axis=(1, 2))
np.testing.assert_allclose(
    final_integrals,
    initial_integrals,
    rtol=0.0,
    atol=2.0e-12,
)
assert np.linalg.norm(final_state[1]) > 1.0e-3


print("PoPS characteristic-upwind tutorial finished")
print("  matrices         : full 2 x 2 transport matrices")
print("  eigenvalues A_x  : %s" % (LAMBDA_X,))
print("  eigenvalues A_y  : %s" % (LAMBDA_Y,))
print("  Riemann solver   : native generic Roe from the flux Jacobian")
print("  Kokkos backend   : %s" % runtime_environment_report()["kokkos_backend"])
print("  requested threads: 7 via pops.set_threads")
print("  accepted steps   : %d" % report.accepted_steps)
print("  final time       : %.6f" % simulation.time())
print("  conserved sums   : %s" % final_integrals)
print("  generated q2 L2  : %.12e" % np.linalg.norm(final_state[1]))
