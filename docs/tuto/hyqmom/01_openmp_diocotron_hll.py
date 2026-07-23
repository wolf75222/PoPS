#!/usr/bin/env python3
"""Diocotron periodique HyQMOM15, HLL et Euler explicite."""

# ruff: noqa: E402

from pathlib import Path
import time

import numpy as np
import pops

pops.set_threads(7)

from pops.diagnostics import Integral, StepChangeNorm
from pops.domain import Rectangle
from pops.fields import (
    CellCenteredSecondOrder,
    ConstantNullspace,
    FieldDiscretization,
    MeanValueGauge,
)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.linalg.norms import L2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.moments import CartesianVelocityMoments, HyQMOM15Closure
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.output import ConsoleMonitor, ConsumerGraph
from pops.params import RuntimeParam
from pops.physics import Density
from pops.runtime_environment import runtime_environment_report
from pops.solvers.elliptic import FFT
from pops.time import AdaptiveCFL, FailRun, every


# Parametres du cas diocotron de l'archive MATLAB.
CELLS = 128
X_MIN = -0.5
X_MAX = 0.5
Y_MIN = -0.5
Y_MAX = 0.5

RHO_MIN = 1.0e-4
RHO_MAX = 1.0
R_INNER = 0.35
R_OUTER = 0.40
PERTURBATION = 0.10
AZIMUTHAL_MODE = 4

OMEGA_P = 20.0
OMEGA_C = -20.0
CFL = 0.5
MONITOR_EVERY = 100
ENABLE_MONITOR = True
T_END = 1.0
MAX_STEPS = 200_000_000

HERE = Path(__file__).resolve().parent
RESULT_FILE = HERE / "results" / "01_openmp_diocotron_hll.npz"


# 1. Domaine periodique et grille cartesienne.
domain = Rectangle(
    "diocotron_square",
    lower=(X_MIN, Y_MIN),
    upper=(X_MAX, Y_MAX),
).tag("plasma")
frame = domain.frame(Cartesian2D())
grid = CartesianGrid(
    frame=frame,
    cells=(CELLS, CELLS),
    periodic=PeriodicAxes(frame.axes),
)


# 2. Hierarchie de 15 moments et fermeture HyQMOM d'ordre quatre.
# Le signe -OMEGA_P**2 donne Delta(phi)=OMEGA_P**2*(rho-rho_background).
# GradientOutput fournit E=-grad(phi), donc q/m=+1 ici.
hierarchy = CartesianVelocityMoments(
    4,
    closure=HyQMOM15Closure(),
    robust=False,
    exact_speeds=True,
)
hierarchy.add_poisson_coupling(
    phi="phi",
    eps=-(OMEGA_P * OMEGA_P),
    background=RuntimeParam("neutralizing_density"),
)
hierarchy.add_vlasov_electric_source("grad_x", "grad_y", q_over_m=1.0)
hierarchy.add_magnetic_source(omega_c=OMEGA_C)
model = hierarchy.build("hyqmom15_diocotron", frame=frame)

state = model.states["U"]
physical_flux = model.fluxes["transport"]
explicit_rate = model.operators["transport"]
poisson = model.field_operators["fields"]
neutralizing_density = model.params["neutralizing_density"]


# 3. Volumes finis constants par cellule et flux HLL fonde sur le Jacobien complet.
finite_volume = FiniteVolume(
    flux=physical_flux,
    variables=variables.Conservative(state),
    reconstruction=reconstruction.FirstOrder(),
    riemann=riemann.HLL(waves=riemann.waves.FromJacobian()),
)
numerics = DiscretizationPlan()
numerics.rates.add(explicit_rate, finite_volume)


# 4. Bloc plasma et solveur de Poisson spectral periodique.
case = pops.Case("tutorial_hyqmom15_diocotron")
plasma = case.block("plasma", model=model)
plasma_state = plasma[state]
case.numerics(numerics, block=plasma)

plasma_field = case.field(
    poisson,
    FieldDiscretization(
        method=CellCenteredSecondOrder(),
        boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Periodic()),),
        solver=FFT(spectral=True),
        nullspace=ConstantNullspace(),
        gauge=MeanValueGauge(0.0),
    ),
)


# 5. Euler explicite. Le champ est recalcule sur U^n avant le taux, comme dans main.m.
program = pops.Program("ForwardEuler-HyQMOM15-diocotron")
moments = program.state(plasma_state)
electric_field = plasma_field(moments.n).consume(action=FailRun())
rhs = explicit_rate(moments.n, electric_field)
candidate = program.value(
    "euler_candidate",
    moments.n + program.dt * rhs,
    at=moments.next.point,
)
program.commit(moments.next, candidate)

# La premiere borne est la CFL hyperbolique native. Cette seconde expression reproduit
# dt_electrostatic=CFL*h*vmax/OMEGA_P**2 ; le runtime prend le minimum des deux.
program.set_dt_bound(
    lambda P, cfl: (
        cfl * P.hmin() * P.max_wave_speed(moments.n) / (OMEGA_P * OMEGA_P)
    )
)
program.step_strategy(AdaptiveCFL(cfl=CFL))
case.program(program)

# Le cout des reductions natives n'est paye que tous les MONITOR_EVERY pas acceptes.
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
)))


# 6. Bootstrap initial du MATLAB : rho -> Poisson -> derive ExB -> gaussienne translatee.
# Ce bloc fabrique seulement l'etat initial. La boucle temporelle reste ensuite native.
dx = (X_MAX - X_MIN) / CELLS
dy = (Y_MAX - Y_MIN) / CELLS
x = X_MIN + (np.arange(CELLS, dtype=np.float64) + 0.5) * dx
y = Y_MIN + (np.arange(CELLS, dtype=np.float64) + 0.5) * dy
X, Y = np.meshgrid(x, y, indexing="ij")
radius = np.sqrt(X * X + Y * Y)
theta = np.arctan2(Y, X)
density = np.where(
    (R_INNER <= radius) & (radius <= R_OUTER),
    RHO_MAX * (1.0 - PERTURBATION + PERTURBATION * np.sin(AZIMUTHAL_MODE * theta)),
    RHO_MIN,
)
neutralizing_density_value = float(np.mean(density, dtype=np.float64))

rhs_poisson = (OMEGA_P * OMEGA_P) * (density - neutralizing_density_value)
kx = 2.0 * np.pi * np.fft.fftfreq(CELLS, d=dx)
ky = 2.0 * np.pi * np.fft.fftfreq(CELLS, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing="ij")
k2 = KX * KX + KY * KY
rhs_hat = np.fft.fft2(rhs_poisson)
potential_hat = np.zeros_like(rhs_hat)
mask = k2 > 0.0
potential_hat[mask] = -rhs_hat[mask] / k2[mask]
grad_phi_x = np.real(np.fft.ifft2(1j * KX * potential_hat))
grad_phi_y = np.real(np.fft.ifft2(1j * KY * potential_hat))

velocity_x = -grad_phi_y / OMEGA_C
velocity_y = grad_phi_x / OMEGA_C
rho = density

initial_state = np.empty((15, CELLS, CELLS), dtype=np.float64)
initial_state[0] = rho
initial_state[1] = rho * velocity_x
initial_state[2] = rho * (velocity_x**2 + 1.0)
initial_state[3] = rho * (velocity_x**3 + 3.0 * velocity_x)
initial_state[4] = rho * (velocity_x**4 + 6.0 * velocity_x**2 + 3.0)
initial_state[5] = rho * velocity_y
initial_state[6] = rho * velocity_x * velocity_y
initial_state[7] = rho * (velocity_x**2 + 1.0) * velocity_y
initial_state[8] = rho * (velocity_x**3 * velocity_y + 3.0 * velocity_x * velocity_y)
initial_state[9] = rho * (velocity_y**2 + 1.0)
initial_state[10] = rho * (velocity_y**2 + 1.0) * velocity_x
initial_state[11] = rho * (
    velocity_x**2 * velocity_y**2 + velocity_x**2 + velocity_y**2 + 1.0
)
initial_state[12] = rho * (velocity_y**3 + 3.0 * velocity_y)
initial_state[13] = rho * (velocity_x * velocity_y**3 + 3.0 * velocity_x * velocity_y)
initial_state[14] = rho * (velocity_y**4 + 6.0 * velocity_y**2 + 3.0)


# 7. Cycle public final : validate -> resolve -> compile -> bind -> run.
validated = pops.validate(case)
neutralizing_density = validated.resolve(neutralizing_density)
resolved = pops.resolve(validated, layout=Uniform(grid))
artifact = pops.compile(resolved)
simulation = pops.bind(
    artifact,
    initial_state={"plasma": initial_state},
    params={neutralizing_density: neutralizing_density_value},
)

(field_slot,) = tuple(simulation.field_provider_slots())
initial_potential = np.asarray(
    simulation.field_potential_global(field_slot), dtype=np.float64
).reshape(CELLS, CELLS)
initial_mass = dx * dy * float(np.sum(initial_state[0], dtype=np.float64))
start = time.perf_counter()
report = pops.run(
    simulation,
    t_end=T_END,
    max_steps=MAX_STEPS,
)
elapsed_seconds = time.perf_counter() - start


# 8. Les tableaux ne reviennent en Python qu'apres le calcul natif.
final_state = np.asarray(
    simulation.state_global("plasma"), dtype=np.float64
).reshape(initial_state.shape)
last_stage_potential = np.asarray(
    simulation.field_potential_global(field_slot), dtype=np.float64
).reshape(CELLS, CELLS)

final_mass = dx * dy * float(np.sum(final_state[0], dtype=np.float64))
mass_error = final_mass - initial_mass
if not np.isfinite(final_state).all():
    raise RuntimeError("the HyQMOM15 state contains a non-finite value")
if np.any(final_state[0] <= 0.0):
    raise RuntimeError("the HyQMOM15 density is not strictly positive")
if not np.isfinite(last_stage_potential).all():
    raise RuntimeError("the last-stage potential contains a non-finite value")

RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    RESULT_FILE,
    initial=initial_state,
    final=final_state,
    initial_potential=initial_potential,
    last_stage_potential=last_stage_potential,
    x=x,
    y=y,
    omega_p=OMEGA_P,
    omega_c=OMEGA_C,
    cfl=CFL,
    t_end=T_END,
    accepted_steps=report.accepted_steps,
    elapsed_seconds=elapsed_seconds,
    initial_mass=initial_mass,
    final_mass=final_mass,
    mass_error=mass_error,
)

print("PoPS HyQMOM15 diocotron tutorial finished")
print("  closure          : polynomial HyQMOM order 4 -> 5")
print("  Riemann solver   : HLL, full 15 x 15 Jacobian")
print("  field solver     : periodic spectral FFT")
print("  time program     : explicit Forward Euler")
print("  Kokkos backend   : %s" % runtime_environment_report()["kokkos_backend"])
print("  requested threads: 7 via pops.set_threads")
print("  accepted steps   : %d" % report.accepted_steps)
print("  final time       : %.12e" % simulation.time())
print("  mass error       : %.12e" % mass_error)
print("  elapsed          : %.6f s" % elapsed_seconds)
print("  result           : %s" % RESULT_FILE)
