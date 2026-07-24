#!/usr/bin/env python3
"""HyQMOM15 fluid wave periodique, HLL et Euler explicite."""

# ruff: noqa: E402

from pathlib import Path
import os
import time

import numpy as np
import pops

pops.set_threads(int(os.environ.get("POPS_THREADS", "4")))

from _amr_hybrid import (
    BASE_CELLS,
    FINE_CELLS,
    add_static_refinement_marker,
    bind_hybrid,
    build_layout,
    coarse_state,
    paraview_output,
    prepare_output_root,
    require_pvd,
)

from pops.diagnostics import Integral, StepChangeNorm
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.linalg.norms import L2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.moments import CartesianVelocityMoments, HyQMOM15Closure
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.output import ConsoleMonitor, ConsumerGraph
from pops.physics import Density
from pops.runtime_environment import runtime_environment_report
from pops.time import AdaptiveCFL, every


CELLS = BASE_CELLS
X_MIN = -0.5
X_MAX = 0.5
Y_MIN = -0.5
Y_MAX = 0.5
EPSILON = 0.01
MODE = 15
KX = 4.0 * np.pi / (X_MAX - X_MIN)
KY = 0.0
CFL = 0.4
MONITOR_EVERY = 100
ENABLE_MONITOR = True
T_END = 0.05
MAX_STEPS = 200_000_000

HERE = Path(__file__).resolve().parent
RESULT_FILE = HERE / "results" / "03_openmp_fluid_wave_hll.npz"
OUTPUT_ROOT = HERE / "results" / "03_openmp_fluid_wave_hll_amr"


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
    riemann=riemann.HLL(),
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
_, marker_state = add_static_refinement_marker(case, frame, program)
case.program(program)

case.consumers(ConsumerGraph.from_consumers((
    ConsoleMonitor(
        schedule=every(MONITOR_EVERY, clock=program.clock),
        diagnostics=(
            StepChangeNorm(L2(), block=plasma),
            Integral(role=Density(), block=plasma),
        ),
        template=(
            "step={step} t={time:.4e} dt={dt:.3e} "
            "dU_L2={plasma.step_change_l2:.3e} "
            "mass={plasma.integral:.6e}"
        ),
        enabled=ENABLE_MONITOR,
    ),
    paraview_output(program, plasma_state, T_END),
)))

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

layout = build_layout(case, grid, program, plasma_state, marker_state)
validated = pops.validate(case)
resolved = pops.resolve(validated, layout=layout)
artifact = pops.compile(resolved)
simulation, world, rank = bind_hybrid(artifact, plasma_state, initial_state)

prepare_output_root(OUTPUT_ROOT, world, rank)
start = time.perf_counter()
report = pops.run(simulation, t_end=T_END, max_steps=MAX_STEPS, output_dir=OUTPUT_ROOT)
elapsed_seconds = time.perf_counter() - start

final_state = coarse_state(simulation, "plasma", initial_state.shape)
if not np.isfinite(final_state).all():
    raise RuntimeError("the HyQMOM15 state contains a non-finite value")

if rank == 0:
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
    pvd_path = require_pvd(OUTPUT_ROOT)
    print("PoPS HyQMOM15 fluid-wave AMR tutorial finished")
    print("  Riemann solver   : HLL")
    print("  AMR hierarchy    : %d x %d -> %d x %d" % (
        BASE_CELLS, BASE_CELLS, FINE_CELLS, FINE_CELLS,
    ))
    print("  MPI ranks        : %d" % world.size)
    print("  selected mode    : %d" % MODE)
    print("  Kokkos backend   : %s" % runtime_environment_report()["kokkos_backend"])
    print("  accepted steps   : %d" % report.accepted_steps)
    print("  result           : %s" % RESULT_FILE)
    print("  ParaView series  : %s" % pvd_path)
