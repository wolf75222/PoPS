#!/usr/bin/env python3
"""HyQMOM15 magnetic wave periodique, Poisson, source cyclotron, HLL et Euler explicite."""

# ruff: noqa: E402

from pathlib import Path
import os
import time

import numpy as np
import pops

pops.set_threads(int(os.environ.get("POPS_THREADS", "4")))

from _amr_hybrid import (
    BASE_CELLS, FINE_CELLS, add_static_refinement_marker, bind_hybrid,
    build_layout, coarse_state, paraview_output, prepare_output_root, require_pvd,
)

from pops.diagnostics import Integral, StepChangeNorm
from pops.domain import Rectangle
from pops.fields import CellCenteredSecondOrder, ConstantNullspace, FieldDiscretization, MeanValueGauge
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
from pops.frames import Cartesian2D
from pops.linalg.norms import L2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.moments import CartesianVelocityMoments, HyQMOM15Closure, HyQMOM15Relaxation
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.output import ConsoleMonitor, ConsumerGraph
from pops.physics import Density
from pops.runtime_environment import runtime_environment_report
from pops.solvers.elliptic import GeometricMG
from pops.solvers.options import CompositeFAC
from pops.time import AdaptiveCFL, FailRun, every


CELLS = BASE_CELLS
X_MIN = -0.5
X_MAX = 0.5
Y_MIN = -0.5
Y_MAX = 0.5
EPSILON = 0.01
MODE = 15
KX = 2.0 * np.pi / (X_MAX - X_MIN)
KY = 4.0 * np.pi / (Y_MAX - Y_MIN)
OMEGA_P = 20.0
OMEGA_C = -40.0
DEBYE_LENGTH = 1.0 / OMEGA_P
CFL = 0.5
MONITOR_EVERY = 100
ENABLE_MONITOR = True
T_END = float(os.environ.get("POPS_T_END", "1.0"))
MAX_STEPS = 200_000_000

HERE = Path(__file__).resolve().parent
RESULT_FILE = HERE / "results" / "05_openmp_magnetic_wave_hll.npz"
OUTPUT_ROOT = HERE / "results" / "05_openmp_magnetic_wave_hll_amr"


domain = Rectangle("hyqmom_magnetic_wave_square", lower=(X_MIN, Y_MIN), upper=(X_MAX, Y_MAX))
frame = domain.frame(Cartesian2D())
grid = CartesianGrid(frame=frame, cells=(CELLS, CELLS), periodic=PeriodicAxes(frame.axes))

hierarchy = CartesianVelocityMoments(
    4,
    closure=HyQMOM15Closure(),
    robust=False,
    exact_speeds=True,
)
hierarchy.add_poisson_coupling(phi="phi", eps=-(OMEGA_P * OMEGA_P), background=1.0)
hierarchy.add_vlasov_electric_source(("grad_x", "grad_y"), q_over_m=1.0)
hierarchy.add_magnetic_source(omega_c=OMEGA_C)
model = hierarchy.build("hyqmom15_magnetic_wave", frame=frame)

state = model.states["U"]
physical_flux = model.fluxes["transport"]
explicit_rate = model.operators["transport"]
poisson = model.field_operators["fields"]
relaxation = HyQMOM15Relaxation().declare(model, state)

finite_volume = FiniteVolume(
    flux=physical_flux,
    variables=variables.Conservative(state),
    reconstruction=reconstruction.FirstOrder(),
    riemann=riemann.HLL(waves=riemann.waves.FromJacobian()),
)
numerics = DiscretizationPlan()
numerics.rates.add(explicit_rate, finite_volume)

case = pops.Case("tutorial_hyqmom15_magnetic_wave")
plasma = case.block("plasma", model=model)
plasma_state = plasma[state]
case.numerics(numerics, block=plasma)
plasma_field = case.field(
    poisson,
    FieldDiscretization(
        method=CellCenteredSecondOrder(),
        boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Periodic()),),
        solver=GeometricMG(fac=CompositeFAC()),
        nullspace=ConstantNullspace(),
        gauge=MeanValueGauge(0.0),
    ),
)

program = pops.Program("ForwardEuler-HyQMOM15-magnetic-wave")
moments = program.state(plasma_state)
electric_field = plasma_field(moments.n).consume(action=FailRun())
rhs = explicit_rate(moments.n, electric_field)
candidate = program.value("euler_candidate", moments.n + program.dt * rhs, at=moments.next.point)
program.commit(moments.next, candidate)

def _relax(program_body):
    synced = program_body.value(
        "synced_candidate", 1.0 * moments.n, at=moments.next.point)
    relaxed = program_body.transform(
        synced, transform=relaxation, name="relaxed_candidate")
    program_body.commit(moments.next, relaxed)

program.after_synchronization(_relax)
_, marker_state = add_static_refinement_marker(case, frame, program)
program.set_dt_bound(
    lambda P, cfl: (
        cfl * P.hmin() * P.max_wave_speed(moments.n) / (OMEGA_P * OMEGA_P)
    )
)
program.step_strategy(AdaptiveCFL(cfl=CFL))
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
k2 = KX * KX + KY * KY
kl2 = k2 * DEBYE_LENGTH * DEBYE_LENGTH
J = np.zeros((15, 15), dtype=np.complex128)
J[0, 1] = KX
J[0, 5] = KY
J[1, 0] = KX / kl2
J[1, 2] = KX
J[1, 5] = 1j * OMEGA_C
J[1, 6] = KY
J[2, 3] = KX
J[2, 6] = 2j * OMEGA_C
J[2, 7] = KY
J[3, 0] = 3.0 * KX / kl2
J[3, 4] = KX
J[3, 7] = 3j * OMEGA_C
J[3, 8] = KY
J[4, 1] = -6.0 * KX
J[4, 3] = 7.0 * KX
J[4, 5] = -3.0 * KY
J[4, 7] = 6.0 * KY
J[4, 8] = 4j * OMEGA_C
J[5, 0] = KY / kl2
J[5, 1] = -1j * OMEGA_C
J[5, 6] = KX
J[5, 9] = KY
J[6, 2] = -1j * OMEGA_C
J[6, 7] = KX
J[6, 9] = 1j * OMEGA_C
J[6, 10] = KY
J[7, 0] = KY / kl2
J[7, 3] = -1j * OMEGA_C
J[7, 8] = KX
J[7, 10] = 2j * OMEGA_C
J[7, 11] = KY
J[8, 1] = -3.0 * KY
J[8, 3] = KY
J[8, 4] = -1j * OMEGA_C
J[8, 5] = -3.0 * KX
J[8, 7] = 6.0 * KX
J[8, 10] = 3.0 * KY
J[8, 11] = 3j * OMEGA_C
J[9, 6] = -2j * OMEGA_C
J[9, 10] = KX
J[9, 12] = KY
J[10, 0] = KX / kl2
J[10, 7] = -2j * OMEGA_C
J[10, 11] = KX
J[10, 12] = 1j * OMEGA_C
J[10, 13] = KY
J[11, 1] = -3.0 * KX
J[11, 3] = KX
J[11, 5] = -3.0 * KY
J[11, 7] = 3.0 * KY
J[11, 8] = -2j * OMEGA_C
J[11, 10] = 3.0 * KX
J[11, 12] = KY
J[11, 13] = 2j * OMEGA_C
J[12, 0] = 3.0 * KY / kl2
J[12, 10] = -3j * OMEGA_C
J[12, 13] = KX
J[12, 14] = KY
J[13, 1] = -3.0 * KY
J[13, 5] = -3.0 * KX
J[13, 7] = 3.0 * KX
J[13, 10] = 6.0 * KY
J[13, 11] = -3j * OMEGA_C
J[13, 12] = KX
J[13, 14] = 1j * OMEGA_C
J[14, 1] = -3.0 * KX
J[14, 5] = -6.0 * KY
J[14, 10] = 6.0 * KX
J[14, 12] = 7.0 * KY
J[14, 13] = -4j * OMEGA_C

eigenvalues, eigenvectors = np.linalg.eig(J)
# MATLAB trie un spectre complexe par module, puis par angle en cas d'egalite.
order = np.lexsort((np.angle(eigenvalues), np.abs(eigenvalues)))
eigenvector = eigenvectors[:, order[MODE - 1]]
eigenvector = eigenvector / np.linalg.norm(eigenvector)

dx = (X_MAX - X_MIN) / CELLS
dy = (Y_MAX - Y_MIN) / CELLS
x = np.arange(CELLS, dtype=np.float64) * dx
y = np.arange(CELLS, dtype=np.float64) * dy
X, Y = np.meshgrid(x, y, indexing="ij")
phase = KX * X + KY * Y
real_mode = np.real(eigenvector[:, None, None] * np.exp(1j * phase)[None, :, :])
initial_state = base[:, None, None] + EPSILON * real_mode

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
        RESULT_FILE, initial=initial_state, final=final_state,
        eigenvalues=eigenvalues, selected_eigenvector=eigenvector,
        accepted_steps=report.accepted_steps, elapsed_seconds=elapsed_seconds,
    )
    pvd_path = require_pvd(OUTPUT_ROOT)
    print("PoPS HyQMOM15 magnetic-wave AMR tutorial finished")
    print("  Riemann solver   : HLL, full 15 x 15 Jacobian")
    print("  field solver     : CompositeFAC")
    print("  AMR hierarchy    : %d x %d -> %d x %d" % (
        BASE_CELLS, BASE_CELLS, FINE_CELLS, FINE_CELLS,
    ))
    print("  MPI ranks        : %d" % world.size)
    print("  selected mode    : %d" % MODE)
    print("  Kokkos backend   : %s" % runtime_environment_report()["kokkos_backend"])
    print("  accepted steps   : %d" % report.accepted_steps)
    print("  result           : %s" % RESULT_FILE)
    print("  ParaView series  : %s" % pvd_path)
