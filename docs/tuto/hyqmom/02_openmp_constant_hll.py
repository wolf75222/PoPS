#!/usr/bin/env python3
"""HyQMOM15 constant periodique, HLL et Euler explicite."""

# ruff: noqa: E402

from pathlib import Path
import time

import numpy as np
import pops

pops.set_threads(7)

from pops.diagnostics import Integral, StepChangeNorm
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.linalg.norms import L2
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.moments import closure, moment_flux_expressions
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.output import ConsoleMonitor, ConsumerGraph
from pops.physics import Density, Model
from pops.runtime_environment import runtime_environment_report
from pops.time import AdaptiveCFL, every


CELLS = 64
X_MIN = -0.5
X_MAX = 0.5
Y_MIN = -0.5
Y_MAX = 0.5
CFL = 0.5
MONITOR_EVERY = 100
ENABLE_MONITOR = True
T_END = 1.0
MAX_STEPS = 2_000_000

HERE = Path(__file__).resolve().parent
RESULT_FILE = HERE / "results" / "02_openmp_constant_hll.npz"

MOMENT_COMPONENTS = (
    "M00", "M10", "M20", "M30", "M40",
    "M01", "M11", "M21", "M31",
    "M02", "M12", "M22",
    "M03", "M13",
    "M04",
)


@closure(4)
def user_hyqmom15_closure(S):  # noqa: N803
    """Six moments standardises d'ordre cinq, ecrits par l'utilisateur."""
    s03 = S["S03"]
    s04 = S["S04"]
    s11 = S["S11"]
    s12 = S["S12"]
    s13 = S["S13"]
    s21 = S["S21"]
    s22 = S["S22"]
    s30 = S["S30"]
    s31 = S["S31"]
    s40 = S["S40"]

    return {
        "S50": 0.5 * s30 * (5.0 * s40 - 3.0 * s30 * s30 - 1.0),
        "S41": (
            -0.25 * s30 * (8.0 * s40 - 9.0 * s30 * s30 - 4.0) * s11
            + 0.25 * (10.0 * s40 - 15.0 * s30 * s30 - 6.0) * s21
            + 2.0 * s30 * s31
        ),
        "S32": (
            0.5 * (2.0 * s40 - 3.0 * s30 * s30) * s12
            + 0.5 * (3.0 * s22 - 1.0) * s30
        ),
        "S23": (
            0.5 * (2.0 * s04 - 3.0 * s03 * s03) * s21
            + 0.5 * (3.0 * s22 - 1.0) * s03
        ),
        "S14": (
            -0.25 * s03 * (8.0 * s04 - 9.0 * s03 * s03 - 4.0) * s11
            + 0.25 * (10.0 * s04 - 15.0 * s03 * s03 - 6.0) * s12
            + 2.0 * s03 * s13
        ),
        "S05": 0.5 * s03 * (5.0 * s04 - 3.0 * s03 * s03 - 1.0),
    }


def build_user_hyqmom15_model(name, frame):
    """Construit explicitement le modele a quinze moments dans ce script."""
    model = Model(name, frame=frame)
    state = model.state(
        "U",
        components=MOMENT_COMPONENTS,
        roles={"M00": Density()},
    )

    # Ce generateur Python effectue M -> moments centres -> moments standardises,
    # applique la fermeture ci-dessus, puis reconstruit les moments bruts d'ordre 5.
    expressions = moment_flux_expressions(
        model,
        tuple(state),
        order=4,
        closure=user_hyqmom15_closure,
        robust=False,
    )
    x_axis, y_axis = frame.axes
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={
            x_axis: expressions.x,
            y_axis: expressions.y,
        },
    )

    # Le modele complet est ferme ici : vitesses HLL issues du Jacobien et
    # equation dU/dt = -div(F(U)).
    model.wave_speeds_from_jacobian()
    model.rate("transport", equation=ddt(state) == -div(flux))
    return model


domain = Rectangle("hyqmom_constant_square", lower=(X_MIN, Y_MIN), upper=(X_MAX, Y_MAX))
frame = domain.frame(Cartesian2D())
grid = CartesianGrid(frame=frame, cells=(CELLS, CELLS), periodic=PeriodicAxes(frame.axes))

model = build_user_hyqmom15_model("hyqmom15_constant", frame)

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

case = pops.Case("tutorial_hyqmom15_constant")
plasma = case.block("plasma", model=model)
plasma_state = plasma[state]
case.numerics(numerics, block=plasma)

program = pops.Program("ForwardEuler-HyQMOM15-constant")
moments = program.state(plasma_state)
rhs = explicit_rate(moments.n)
candidate = program.value("euler_candidate", moments.n + program.dt * rhs, at=moments.next.point)
program.commit(moments.next, candidate)
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
)))

base = np.array(
    [1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 3.0],
    dtype=np.float64,
)
initial_state = np.zeros((15, CELLS, CELLS), dtype=np.float64)
for component in range(15):
    initial_state[component, :, :] = base[component]

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

print("PoPS HyQMOM15 constant tutorial finished")
print("  Riemann solver   : HLL, full 15 x 15 Jacobian")
print("  Kokkos backend   : %s" % runtime_environment_report()["kokkos_backend"])
print("  accepted steps   : %d" % report.accepted_steps)
print("  result           : %s" % RESULT_FILE)
