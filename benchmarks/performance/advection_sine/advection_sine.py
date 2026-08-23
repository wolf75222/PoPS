#!/usr/bin/env python3
"""Advection sinusoïdale périodique : cas public mesuré de bout en bout.

Le fichier se lit de haut en bas, comme le tutoriel ``01_openmp_preset_ssprk2``.
Les constantes et la physique restent ici; ``support.py`` ne contient que la
publication JSON.  La fenêtre mesurée est le cycle public complet ``pops.run``:
elle ne mesure ni un kernel C++ isolé, ni une boucle Python par cellule/pas.
"""

# ruff: noqa: E402

from __future__ import annotations

import math
import os
import time
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
PROFILE_SUPPORT = HERE / "profiling"
if str(PROFILE_SUPPORT) not in sys.path:
    sys.path.insert(0, str(PROFILE_SUPPORT))

from support import MEASUREMENT_SCHEMA, arguments, write_rank_measurement
from profile_contract import program_artifact_receipt

# Valeurs par défaut documentaires. Les campagnes les remplacent explicitement
# par arguments; conserver les constantes rend le cas lisible hors ROMEO.
EPSILON = 0.10
WAVE_NUMBERS = (1, 2, 3)
DEFAULT_CFL = 0.40
DISSIPATION_RATIO_MAX = 0.25

args = arguments()
_profile_ready = os.environ.get("POPS_MACOS_PROFILE_READY")
_profile_go = os.environ.get("POPS_MACOS_PROFILE_GO")
_profile_nonce = os.environ.get("POPS_MACOS_PROFILE_NONCE")
_profile_receipt = os.environ.get("POPS_MACOS_PROFILE_RECEIPT")
_profile_values = (_profile_ready, _profile_go, _profile_nonce, _profile_receipt)
if any(value is not None for value in _profile_values) and any(
    not value for value in _profile_values
):
    raise RuntimeError("macOS profiling requires READY, GO, nonce and receipt together")
PROFILE_MODE = all(_profile_values)
if PROFILE_MODE:
    from profile_contract import (
        authenticated_profile_provenance,
    )
    from ready_go import await_go, completed_public_lifecycle, ready_after_bind_warmup

# 1. Configuration de threads, avant tout objet pouvant initialiser Kokkos.
import pops

pops.set_threads(args.threads)

from pops.analytic import coordinates, sin
from pops.domain import CartesianDomain
from pops.frames import Cartesian
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import Analytic
from pops.lib.time import SSPRK2
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes, RegularBlocks
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import FixedDt
from pops.runtime_environment import runtime_environment_report

from helpers.verification.sine_wave import (
    direction_velocity,
    sine_wave_cell_averages,
    weighted_error_norms,
)


# 2. Domaine périodique [0, 1]^d et grille uniforme réellement pavée.
DIMENSION = len(args.resolution)
domain = CartesianDomain("periodic_unit_box", lower=(0.0,) * DIMENSION, upper=(1.0,) * DIMENSION)
frame = domain.frame(Cartesian(DIMENSION))
axes = frame.axes
grid = CartesianGrid(
    frame=frame,
    cells=args.resolution,
    periodic=PeriodicAxes(axes),
    blocks=RegularBlocks(max_cells=args.block_size),
)


# 3. Modèle : d_t q + div(a q) = 0, direction demandée par la campagne.
velocity_values = direction_velocity(args.mode, DIMENSION)
model = pops.Model("periodic_sine_advection", frame=frame)
U = model.state("U", components=("q",), representation=Conservative(), space=CellState(frame=frame))
(q,) = U
velocity = model.vector("a", frame=frame, components=dict(zip(axes, velocity_values, strict=True)))
flux = model.flux(
    "advection_flux",
    frame=frame,
    state=U,
    components={axis: (speed * q,) for axis, speed in zip(axes, velocity_values, strict=True)},
    waves={axis: (speed,) for axis, speed in zip(axes, velocity_values, strict=True)},
)
rate = model.rate("advection_rate", equation=ddt(U) == -div(flux))


# 4. Volumes finis MUSCL Van Leer, flux Scalaire upwind, SSPRK2 et pas dyadique stable.
numerics = DiscretizationPlan()
numerics.rates.add(
    rate,
    FiniteVolume(
        flux=flux,
        variables=variables.Conservative(U),
        reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
        riemann=riemann.ScalarUpwind(velocity=velocity),
    ),
)
case = pops.Case("performance_periodic_sine_advection")
tracer = case.block("tracer", model=model, states=(U,))
tracer_U = tracer[U]
case.numerics(numerics, block=tracer)
program = SSPRK2(tracer_U, rate=rate)
inverse_stable_dt = (
    sum(abs(speed) * cells for speed, cells in zip(velocity_values, args.resolution, strict=True))
    / args.cfl
)
DT = math.ldexp(1.0, -math.ceil(math.log2(inverse_stable_dt)))
program.step_strategy(FixedDt(DT))
case.program(program)


# 5. Sine analytique, projetée par ConservativeCellAverage dans le chemin public.
phase = sum(
    wave * coordinate
    for wave, coordinate in zip(WAVE_NUMBERS[:DIMENSION], coordinates(frame), strict=True)
)
case.initials.add(
    InitialCondition(
        state=tracer_U,
        value=Analytic(frame=frame, components=(1.0 + EPSILON * sin(2.0 * math.pi * phase),)),
        projection=ConservativeCellAverage(),
    )
)


# 6. Validation, résolution et compilation une seule fois. Le contexte MPI est explicite.
layout = Uniform(grid)
validated = pops.validate(case)
resolved = pops.resolve(validated, layout=layout)
artifact = pops.compile(resolved)
artifact_programs = program_artifact_receipt(artifact)
wants_mpi = args.route in {"kokkos_openmp_mpi", "kokkos_cuda_mpi"}
if not wants_mpi and args.expected_ranks != 1:
    raise RuntimeError("a non-MPI route must declare exactly one rank")
execution_context = pops.ExecutionContext.mpi_world(artifact) if wants_mpi else None


# 7. Chaque répétition installe une simulation fraîche hors chronomètre, puis mesure uniquement run.
def bind_fresh():
    resources = {} if execution_context is None else {"execution_context": execution_context}
    return pops.bind(artifact, resources=resources)


def synchronize_world(simulation) -> None:
    # Native synchronization is a RuntimeInstance primitive. MPI barrier is
    # deliberately outside the performance interval and provided by the exact
    # MPI execution context only when the campaign selected MPI.
    simulation.synchronize()
    if execution_context is not None:
        execution_context.communicator.handle.barrier()


for _ in range(args.warmups):
    warmup = bind_fresh()
    synchronize_world(warmup)
    pops.run(warmup, t_end=args.steps * DT, max_steps=args.steps, console=False)
    synchronize_world(warmup)

samples: list[float] = []
last_simulation = None
prepared_profile_simulation = None
prepared_initial_native = None
prepared_initial_native_integral = None
if PROFILE_MODE:
    # The exact measured simulation is fully bound before READY.  The recorder
    # can therefore attach only after compile/bind/warmup, never to setup work.
    from pops._native_selector import select_native_dimension

    prepared_profile_simulation = bind_fresh()
    prepared_initial_native = np.asarray(
        prepared_profile_simulation.state_global("tracer"), dtype=np.float64
    ).reshape(tuple(reversed(args.resolution)))
    prepared_initial_native_integral = float(prepared_profile_simulation.integral("tracer"))
    profile_provenance = authenticated_profile_provenance(
        artifact=artifact,
        args=args,
        case_path=Path(__file__),
        campaign_value=os.environ.get("POPS_MACOS_PROFILE_CAMPAIGN_PATH"),
        expected_command_sha256=os.environ.get("POPS_MACOS_PROFILE_COMMAND_SHA256"),
        expected_source_tree_sha256=os.environ.get("POPS_MACOS_PROFILE_SOURCE_TREE_SHA256"),
        native_module=select_native_dimension(DIMENSION),
        pops_module=pops,
        python_executable=Path(sys.executable),
        runtime_report=runtime_environment_report(),
        source_manifest_value=os.environ.get("POPS_MACOS_PROFILE_SOURCE_MANIFEST"),
        source_root_value=os.environ.get("POPS_MACOS_PROFILE_SOURCE_ROOT"),
        build_receipt_value=os.environ.get("POPS_MACOS_PROFILE_BUILD_RECEIPT"),
    )
    ready_after_bind_warmup(
        ready=Path(_profile_ready), nonce=_profile_nonce, provenance=profile_provenance
    )
    await_go(go=Path(_profile_go), nonce=_profile_nonce)

for repetition in range(1 if PROFILE_MODE else args.repetitions):
    if PROFILE_MODE:
        if repetition != 0 or prepared_profile_simulation is None:
            raise RuntimeError("macOS profiling may acquire exactly one fresh public lifecycle")
        simulation = prepared_profile_simulation
        initial_native = prepared_initial_native
        initial_native_integral = prepared_initial_native_integral
    else:
        simulation = bind_fresh()
        initial_native = np.asarray(simulation.state_global("tracer"), dtype=np.float64).reshape(
            tuple(reversed(args.resolution))
        )
        initial_native_integral = float(simulation.integral("tracer"))
    synchronize_world(simulation)
    started = time.perf_counter()
    report = pops.run(simulation, t_end=args.steps * DT, max_steps=args.steps, console=False)
    simulation.synchronize()
    samples.append(time.perf_counter() - started)
    if int(report.accepted_steps) != args.steps or int(report.rejected_steps) != 0:
        raise RuntimeError(
            "the fixed-dt public lifecycle did not accept exactly the campaign steps"
        )
    last_simulation = simulation

if last_simulation is None:
    raise RuntimeError("campaign requires at least one timed repetition")


# 8. Vérifications hors chronomètre : état global, oracle NumPy, intégrale native, finitude.
environment = runtime_environment_report()
if bool(environment.get("mpi_active")) is not wants_mpi:
    raise RuntimeError("native MPI state differs from the campaign route")
rank = int(environment.get("mpi_rank", 0))
ranks = int(environment.get("mpi_ranks", 1))
if ranks != args.expected_ranks:
    raise RuntimeError("actual MPI rank count differs from campaign")
initial, _coordinates = sine_wave_cell_averages(
    args.resolution, WAVE_NUMBERS[:DIMENSION], epsilon=EPSILON
)
final = np.asarray(last_simulation.state_global("tracer"), dtype=np.float64).reshape(initial.shape)
exact, _ = sine_wave_cell_averages(
    args.resolution,
    WAVE_NUMBERS[:DIMENSION],
    epsilon=EPSILON,
    displacement=tuple(speed * args.steps * DT for speed in velocity_values),
)
volume = 1.0 / math.prod(args.resolution)
initial_norms = weighted_error_norms(initial_native, initial, volume).to_dict()
final_norms = weighted_error_norms(final, exact, volume).to_dict()
stationary_norms = weighted_error_norms(final, initial, volume).to_dict()
if stationary_norms["l2"] <= 1e-12:
    raise RuntimeError("stationary sine error is too small to authenticate advection progress")
dissipation_ratio = final_norms["l2"] / stationary_norms["l2"]
host_integral = float(np.sum(final, dtype=np.float64) * volume)
native_integral = float(last_simulation.integral("tracer"))
initial_host_integral = float(np.sum(initial_native, dtype=np.float64) * volume)
mass_drift = abs(native_integral - initial_native_integral) / max(1.0, abs(initial_native_integral))
local_boxes = [
    {"lower": list(lower), "upper_exclusive": list(upper)}
    for lower, upper in last_simulation.local_boxes("tracer")
]
environment = runtime_environment_report()
expected_backend = {
    "kokkos_serial": "Serial",
    "kokkos_openmp": "OpenMP",
    "kokkos_openmp_mpi": "OpenMP",
    "kokkos_cuda": "Cuda",
    "kokkos_cuda_mpi": "Cuda",
}[args.route]
if environment.get("kokkos_backend") != expected_backend:
    raise RuntimeError("runtime Kokkos backend differs from compiled artifact")
omp_environment = {
    "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
    "omp_places": os.environ.get("OMP_PLACES"),
    "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
}
if omp_environment != {"omp_proc_bind": "spread", "omp_places": "cores", "omp_dynamic": "false"}:
    raise RuntimeError("campaign requires OMP_PROC_BIND=spread OMP_PLACES=cores OMP_DYNAMIC=false")


# 9. Un seul fichier JSON par rang. Le collecteur valide/agrège les rangs hors exécution.
payload = {
    "schema": MEASUREMENT_SCHEMA,
    "campaign": args.campaign,
    "point": args.point,
    "route": args.route,
    "rank": rank,
    "metadata": {
        "execution_space": environment.get("kokkos_backend"),
        "execution_concurrency": environment.get("kokkos_concurrency"),
        "mpi_ranks": ranks,
        "gpu_device_ordinal": environment.get("gpu_device_ordinal"),
        "gpu_uuid": environment.get("gpu_uuid"),
        "gpu_uuid_method": environment.get("gpu_uuid_method"),
        "gpu_uuid_diagnostic": environment.get("gpu_uuid_diagnostic"),
        **omp_environment,
    },
    "program_artifact": artifact_programs,
    "resources": {"nodes": args.nodes, "ranks": ranks, "threads_per_rank": args.threads},
    "problem": {
        "dimension": DIMENSION,
        "resolution": list(args.resolution),
        "mode": args.mode,
        "layout": "uniform",
        "block_size": args.block_size,
        "cfl": args.cfl,
        "dt": DT,
        "steps": args.steps,
    },
    "local_boxes": local_boxes,
    "timing": {"metric": "public_lifecycle_wall_seconds", "samples": samples},
    "validation": {
        "timed": False,
        "passed": bool(
            np.isfinite(initial_native).all()
            and np.isfinite(final).all()
            and max(initial_norms.values()) <= 5e-12
            and dissipation_ratio <= DISSIPATION_RATIO_MAX
            and final_norms["linf"] < 0.2
            and abs(native_integral - host_integral) <= 5e-12
            and abs(initial_native_integral - initial_host_integral) <= 5e-12
            and mass_drift <= 5e-12
        ),
        "initial_exact_errors": initial_norms,
        "final_exact_errors": final_norms,
        "stationary_initial_errors": stationary_norms,
        "final_to_stationary_l2_ratio": dissipation_ratio,
        "native_integral": native_integral,
        "host_integral": host_integral,
        "initial_native_integral": initial_native_integral,
        "initial_host_integral": initial_host_integral,
        "mass_drift": mass_drift,
        "nonfinite_final_cells": int(np.size(final) - np.isfinite(final).sum()),
    },
}
write_rank_measurement(args.output_dir, rank, payload)
if PROFILE_MODE:
    completed_public_lifecycle(receipt=Path(_profile_receipt), nonce=_profile_nonce, returncode=0)
