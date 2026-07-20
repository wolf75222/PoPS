#!/usr/bin/env python3
"""HyQMOM15 tube a choc 2D, HLL et Euler explicite."""

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


CELLS = 256
X_MIN = -0.5
X_MAX = 0.5
Y_MIN = -0.5
Y_MAX = 0.5
CFL = 0.5
T_END = 1.0
MAX_STEPS = 20_000_000

RHO_LEFT = 1.0
RHO_RIGHT = 0.1
PRESSURE_LEFT = 1.0
PRESSURE_RIGHT = 0.1

HERE = Path(__file__).resolve().parent
RESULT_FILE = HERE / "results" / "06_openmp_shock_tube_hll.npz"


domain = Rectangle("hyqmom_shock_tube_square", lower=(X_MIN, Y_MIN), upper=(X_MAX, Y_MAX))
frame = domain.frame(Cartesian2D())
grid = CartesianGrid(frame=frame, cells=(CELLS, CELLS), periodic=PeriodicAxes(frame.axes))

hierarchy = CartesianVelocityMoments(
    4,
    closure=HyQMOM15Closure(),
    robust=False,
    exact_speeds=True,
)
model = hierarchy.build("hyqmom15_shock_tube", frame=frame)

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

case = pops.Case("tutorial_hyqmom15_shock_tube")
plasma = case.block("plasma", model=model)
plasma_state = plasma[state]
case.numerics(numerics, block=plasma)

program = pops.Program("ForwardEuler-HyQMOM15-shock-tube")
moments = program.state(plasma_state)
rhs = explicit_rate(moments.n)
candidate = program.value("euler_candidate", moments.n + program.dt * rhs, at=moments.next.point)
program.commit(moments.next, candidate)
program.step_strategy(AdaptiveCFL(cfl=CFL))
case.program(program)

left = np.array(
    [
        RHO_LEFT,
        0.0,
        PRESSURE_LEFT,
        0.0,
        3.0 * PRESSURE_LEFT * PRESSURE_LEFT / RHO_LEFT,
        0.0,
        0.0,
        0.0,
        0.0,
        RHO_LEFT,
        0.0,
        PRESSURE_LEFT,
        0.0,
        0.0,
        3.0 * RHO_LEFT,
    ],
    dtype=np.float64,
)
right = np.array(
    [
        RHO_RIGHT,
        0.0,
        PRESSURE_RIGHT,
        0.0,
        3.0 * PRESSURE_RIGHT * PRESSURE_RIGHT / RHO_RIGHT,
        0.0,
        0.0,
        0.0,
        0.0,
        RHO_RIGHT,
        0.0,
        PRESSURE_RIGHT,
        0.0,
        0.0,
        3.0 * RHO_RIGHT,
    ],
    dtype=np.float64,
)
initial_state = np.empty((15, CELLS, CELLS), dtype=np.float64)
for i in range(CELLS):
    profile = left if i < CELLS // 2 else right
    for component in range(15):
        initial_state[component, i, :] = profile[component]

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
    accepted_steps=report.accepted_steps,
    elapsed_seconds=elapsed_seconds,
)

print("PoPS HyQMOM15 shock-tube tutorial finished")
print("  Riemann solver   : HLL, full 15 x 15 Jacobian")
print("  Kokkos backend   : %s" % runtime_environment_report()["kokkos_backend"])
print("  accepted steps   : %d" % report.accepted_steps)
print("  result           : %s" % RESULT_FILE)
