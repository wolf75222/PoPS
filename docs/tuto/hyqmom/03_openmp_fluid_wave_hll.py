#!/usr/bin/env python3
"""HyQMOM15 fluid wave periodique, HLL et Euler explicite."""

from pathlib import Path
import time

import numpy as np
import pops

pops.set_threads(7)

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.moments import CartesianVelocityMoments, HyQMOM15Closure
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.time import AdaptiveCFL
from pops.runtime_environment import runtime_environment_report


CELLS = 32
X_MIN = -0.5
X_MAX = 0.5
Y_MIN = -0.5
Y_MAX = 0.5
EPSILON = 0.01
MODE = 15
KX = 4.0 * np.pi / (X_MAX - X_MIN)
KY = 0.0
CFL = 0.4
T_END = 0.05
MAX_STEPS = 200_000_000

HERE = Path(__file__).resolve().parent
RESULT_FILE = HERE / "results" / "03_openmp_fluid_wave_hll.npz"


domain = Rectangle("hyqmom_fluid_wave_square", lower=(X_MIN, Y_MIN), upper=(X_MAX, Y_MAX))
frame = domain.frame(Cartesian2D())
grid = CartesianGrid(frame=frame, cells=(CELLS, CELLS), periodic=PeriodicAxes(frame.axes))

hierarchy = CartesianVelocityMoments(
    4,
    closure=HyQMOM15Closure(),
    robust=False,
    exact_speeds=True,
)
model = hierarchy.build("hyqmom15_fluid_wave", frame=frame)

state = model.states["U"]
physical_flux = model.fluxes["transport"]
explicit_rate = model.operators["transport"]

finite_volume = FiniteVolume(
    flux=physical_flux,
    variables=variables.Conservative(state),
    reconstruction=reconstruction.FirstOrder(),
    riemann=riemann.HLL(waves=riemann.waves.FromJacobian()),
)
numerics = DiscretizationPlan()
numerics.rates.add(explicit_rate, finite_volume)

case = pops.Case("tutorial_hyqmom15_fluid_wave")
plasma = case.block("plasma", model=model)
plasma_state = plasma[state]
case.numerics(numerics, block=plasma)

program = pops.Program("ForwardEuler-HyQMOM15-fluid-wave")
moments = program.state(plasma_state)
rhs = explicit_rate(moments.n)
candidate = program.value("euler_candidate", moments.n + program.dt * rhs, at=moments.next.point)
program.commit(moments.next, candidate)
program.step_strategy(AdaptiveCFL(cfl=CFL))
case.program(program)

base = np.array(
    [1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 3.0],
    dtype=np.float64,
)
J = np.zeros((15, 15), dtype=np.float64)
J[0, 1] = KX
J[0, 5] = KY
J[1, 2] = KX
J[1, 6] = KY
J[2, 3] = KX
J[2, 7] = KY
J[3, 4] = KX
J[3, 8] = KY
J[4, 1] = -6.0 * KX
J[4, 3] = 7.0 * KX
J[4, 5] = -3.0 * KY
J[4, 7] = 6.0 * KY
J[5, 6] = KX
J[5, 9] = KY
J[6, 7] = KX
J[6, 10] = KY
J[7, 8] = KX
J[7, 11] = KY
J[8, 1] = -3.0 * KY
J[8, 3] = KY
J[8, 5] = -3.0 * KX
J[8, 7] = 6.0 * KX
J[8, 10] = 3.0 * KY
J[9, 10] = KX
J[9, 12] = KY
J[10, 11] = KX
J[10, 13] = KY
J[11, 1] = -3.0 * KX
J[11, 3] = KX
J[11, 5] = -3.0 * KY
J[11, 7] = 3.0 * KY
J[11, 10] = 3.0 * KX
J[11, 12] = KY
J[12, 13] = KX
J[12, 14] = KY
J[13, 1] = -3.0 * KY
J[13, 5] = -3.0 * KX
J[13, 7] = 3.0 * KX
J[13, 10] = 6.0 * KY
J[13, 12] = KX
J[14, 1] = -3.0 * KX
J[14, 5] = -6.0 * KY
J[14, 10] = 6.0 * KX
J[14, 12] = 7.0 * KY

eigenvalues, eigenvectors = np.linalg.eig(J)
order = np.argsort(eigenvalues.real)
eigenvector = np.real(eigenvectors[:, order[MODE - 1]])
eigenvector = eigenvector / np.linalg.norm(eigenvector)

dx = (X_MAX - X_MIN) / CELLS
dy = (Y_MAX - Y_MIN) / CELLS
x = np.arange(CELLS, dtype=np.float64) * dx
y = np.arange(CELLS, dtype=np.float64) * dy
X, Y = np.meshgrid(x, y, indexing="ij")
phase = KX * X + KY * Y
initial_state = base[:, None, None] + EPSILON * eigenvector[:, None, None] * np.sin(phase)[None, :, :]

validated = pops.validate(case)
resolved = pops.resolve(validated, layout=Uniform(grid))
artifact = pops.compile(resolved)
simulation = pops.bind(artifact, initial_state={"plasma": initial_state})

start = time.perf_counter()
report = pops.run(simulation, t_end=T_END, max_steps=MAX_STEPS)
elapsed_seconds = time.perf_counter() - start

final_state = np.asarray(simulation.state_global("plasma"), dtype=np.float64).reshape(initial_state.shape)
if not np.isfinite(final_state).all():
    raise RuntimeError("the HyQMOM15 state contains a non-finite value")

RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    RESULT_FILE,
    initial=initial_state,
    final=final_state,
    eigenvalues=eigenvalues,
    selected_eigenvector=eigenvector,
    accepted_steps=report.accepted_steps,
    elapsed_seconds=elapsed_seconds,
)

print("PoPS HyQMOM15 fluid-wave tutorial finished")
print("  Riemann solver   : HLL, full 15 x 15 Jacobian")
print("  selected mode    : %d" % MODE)
print("  Kokkos backend   : %s" % runtime_environment_report()["kokkos_backend"])
print("  accepted steps   : %d" % report.accepted_steps)
print("  result           : %s" % RESULT_FILE)
