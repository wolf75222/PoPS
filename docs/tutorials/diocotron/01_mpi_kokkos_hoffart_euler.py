#!/usr/bin/env python3
"""Hoffart's conducting-disk Euler--Poisson benchmark, in mapped finite volumes.

The physical model is the full finite-Omega barotropic Euler--Poisson system.
PoPS stores q=r*(rho,mx,my) on (r,theta), retaining Cartesian momentum components.
The source-first Lie composition uses the full-Gauss-restart Schur system and
SSPRK2 transport. It is first order in splitting time; dt refinement is required.
All simulation construction is deliberately linear and at module scope.
"""
# ruff: noqa: E402

from fractions import Fraction
from pathlib import Path
import json
import math
import os
import time

import numpy as np
import pops

pops.set_threads(int(os.environ.get("POPS_THREADS", "1")))

from mpi4py import MPI
from pops.amr import AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer
from pops.amr import Buffer, Coarsen, ConflictPolicy, EqualityPolicy, Hysteresis, Tag
from pops.amr import PatchLayout
from pops.analytic import CellBounds, between, coordinate, cos, maximum, minimum, sin, where
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import NoFlux, Outflow
from pops.domain import Rectangle
from pops.fields import AnalyticAux, AuxiliaryBoundary
from pops.fields.bcs import Dirichlet, Neumann, Periodic
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import BergerRigoutsos, SpaceFillingCurve, StateTransfer
from pops.lib.initial import Analytic
from pops.linalg import LinearProblem
from pops.math import ValueExpr, Var, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.physics import Density, Momentum
from pops.projection import ConservativeCellAverage
from pops.solvers import CompositeTensorFAC, Hierarchy
from pops.time import AdaptiveCFL, FailRun, every


# 1. Authors' released benchmark, rather than the inconsistent printed scaling.
# ryujin 7e8177dfe35ae5f6a1ae8477f55b1781a1c01c43, mode_5.prm.
R = 16.0
R0, R1 = 6.0, 8.0
BACKGROUND, MEAN_RING, PERTURBATION = 1.0e-6, 0.9, 0.1
ALPHA, OMEGA, TEMPERATURE = 39.4784176e12, -6.28318531e12, 1.0e-24
MODE = int(os.environ.get("POPS_MODE", "5"))
NR = int(os.environ.get("POPS_NR", "16"))
NTHETA = int(os.environ.get("POPS_NTHETA", str(4 * NR)))
MAX_LEVELS = int(os.environ.get("POPS_MAX_LEVELS", "2"))
COARSE_MAX_GRID = int(os.environ.get("POPS_COARSE_MAX_GRID", "8" if NR == 16 else "16"))
# Cluster sizes count parent tagging cells before ratio-two refinement.
CLUSTER_MAX_GRID = int(os.environ.get("POPS_CLUSTER_MAX_GRID", "16" if NR == 16 else "32"))
CFL = float(os.environ.get("POPS_CFL", "0.3"))
MAX_DT = float(os.environ.get("POPS_MAX_DT", "0.001"))
T_END = float(os.environ.get("POPS_T_END", "10.0"))
OUTPUT_INTERVAL = float(os.environ.get("POPS_OUTPUT_INTERVAL", "0.05"))
GROWTH_OUTPUT_INTERVAL = min(OUTPUT_INTERVAL, 0.01)
GROWTH_OUTPUT_END = 1.5
WALL_SECONDS = float(os.environ.get("POPS_RUN_WALLTIME_SECONDS", "inf"))
OUTPUT = Path(os.environ.get("POPS_RUN_OUTPUT", "hoffart-euler-mode%d" % MODE)).resolve()
CHECKPOINT = Path(os.environ.get("POPS_RUN_CHECKPOINT", str(OUTPUT / "checkpoint.npz"))).resolve()
RESTART = os.environ.get("POPS_RUN_RESTART", "")
if MODE not in (3, 4, 5) or NR < 16 or NR % 8 or NTHETA < 32 or NTHETA % 8 or MAX_LEVELS < 1:
    raise ValueError("use mode3/4/5, NR>=16 and Ntheta>=32 divisible by8, and at least one level")
if min(COARSE_MAX_GRID, CLUSTER_MAX_GRID) < 4 or COARSE_MAX_GRID % 2 or CLUSTER_MAX_GRID % 2:
    raise ValueError("coarse patch and parent cluster maxima must be even and at least4")
if (not all(math.isfinite(value) for value in (CFL, MAX_DT, T_END, OUTPUT_INTERVAL))
        or not 0.0 < CFL <= 0.5 or MAX_DT <= 0 or T_END <= 0 or OUTPUT_INTERVAL <= 0):
    raise ValueError("time intervals must be positive and CFL must lie in (0,0.5]")
if math.isnan(WALL_SECONDS) or WALL_SECONDS <= 0:
    raise ValueError("wall budget must be positive (infinity means unbounded)")
WORLD = MPI.COMM_WORLD
RANK = WORLD.Get_rank()
if RANK == 0:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
WORLD.Barrier()


# 2. Exact disk parametrization: no omitted central hole and periodic angle.
domain = Rectangle("polar computational domain", lower=(0.0, 0.0), upper=(R, 2.0 * math.pi))
frame = domain.frame(Cartesian2D())
radial_axis, angular_axis = frame.axes
grid = CartesianGrid(frame=frame, cells=(NR, NTHETA), periodic=PeriodicAxes((angular_axis,)))
r = coordinate(frame, radial_axis)
theta = coordinate(frame, angular_axis)
bounds = CellBounds(frame)
dtheta = bounds.upper(angular_axis) - bounds.lower(angular_axis)
half_angle = dtheta / 2.0


# 3. Cartesian momentum transported through the two polar face families.
# The metric corrections integrate trigonometric face factors. Together with
# the central part of Rusanov, they retain the interior constant-flow identity.
# This uniform-level identity does not assert an exact pole/wall or AMR identity.
radius = Var("radius", "aux")
cosine, sine = Var("cosine", "aux"), Var("sine", "aux")
radial_cosine, radial_sine = Var("radial_cosine", "aux"), Var("radial_sine", "aux")
angular_cosine, angular_sine = Var("angular_cosine", "aux"), Var("angular_sine", "aux")
model = pops.Model("Hoffart full barotropic Euler", frame=frame)
U = model.state("U", components=("density", "mx", "my"),
                roles={"density": Density(), "mx": Momentum(axis=radial_axis),
                       "my": Momentum(axis=angular_axis)})
density, mx, my = U
ux, uy = mx / density, my / density
cartesian_x = (mx, mx * ux + TEMPERATURE * density, my * ux)
cartesian_y = (my, mx * uy, my * uy + TEMPERATURE * density)
ur = radial_cosine * ux + radial_sine * uy
utheta = (-angular_sine * ux + angular_cosine * uy) / radius
sound = math.sqrt(TEMPERATURE)
physical_flux = model.flux("mapped_transport", frame=frame, state=U,
    components={
        radial_axis: tuple(radial_cosine * fx + radial_sine * fy
                           for fx, fy in zip(cartesian_x, cartesian_y, strict=True)),
        angular_axis: tuple((-angular_sine * fx + angular_cosine * fy) / radius
                            for fx, fy in zip(cartesian_x, cartesian_y, strict=True)),
    },
    waves={radial_axis: (ur - sound, ur, ur + sound),
           angular_axis: (utheta - 2.0 * sound / radius, utheta, utheta + 2.0 * sound / radius)})
model.wave_speeds(physical_flux, frame=frame, values={
    radial_axis: (ur - sound, ur + sound),
    angular_axis: (utheta - 2.0 * sound / radius, utheta + 2.0 * sound / radius),
})
transport = model.rate("finite volume transport", equation=ddt(U) == -div(physical_flux))


# 4. Authored Lorentz matrix, physical gradient map C and polar Poisson tensor K.
rotation = model.operator("Lorentz rotation", returns=model.local_linear_operator(
    "Lorentz matrix", on=U,
    matrix=((0., 0., 0.), (0., 0., OMEGA), (0., -OMEGA, 0.))))
gradient_map = model.operator("physical gradient map", returns=model.local_linear_operator(
    "C", on=U, matrix=((0., 0., 0.),
                        (0., cosine, -sine / radius),
                        (0., sine, cosine / radius))))
base_tensor = model.operator("polar Poisson tensor", returns=model.local_linear_operator(
    "K", on=U, matrix=((0., 0., 0.), (0., radius, 0.), (0., 0., 1. / radius))))
module = model.module
for name, expression in (
    ("radius", r), ("cosine", cos(theta)), ("sine", sin(theta)),
    ("radial_cosine", cos(theta) * sin(half_angle) / half_angle),
    ("radial_sine", sin(theta) * sin(half_angle) / half_angle),
    ("angular_cosine", cos(theta) / cos(half_angle)),
    ("angular_sine", sin(theta) / cos(half_angle)),
):
    auxiliary = module.aux_handle(module.aux_field(name, frame=frame.canonical_id))
    module.aux_provider(AnalyticAux(auxiliary, expression, frame=frame,
                                  boundary=AuxiliaryBoundary(width=2, kind="foextrap")))

module.operator_registry().get(physical_flux.reg_name).requirements["aux"] = (
    "radius", "radial_cosine", "radial_sine", "angular_cosine", "angular_sine")
module.operator_registry().get(gradient_map.name).requirements["aux"] = ("radius", "cosine", "sine")
module.operator_registry().get(base_tensor.name).requirements["aux"] = ("radius",)


# 5. Finite volumes, conducting electrostatics and the benchmark's open transport wall.
case = pops.Case("Hoffart Euler mode %d" % MODE)
plasma = case.block("plasma", model=model)
plasma_U = plasma[U]
numerics = DiscretizationPlan()
numerics.rates.add(transport, FiniteVolume(flux=physical_flux,
    variables=variables.Conservative(U),
    reconstruction=reconstruction.MUSCL(limiters.Minmod()), riemann=riemann.Rusanov()))
numerics.boundaries.add(TransportBoundarySet({
    frame.boundaries.x_min: NoFlux(state=plasma_U),
    frame.boundaries.x_max: Outflow(state=plasma_U),
}, periodic=PeriodicAxes((angular_axis,))))
case.numerics(numerics, block=plasma)


# 6. Full-Gauss-restart Schur source, followed by conservative SSPRK2 transport.
# psi=phi/alpha. B=I-s*Omega*J, A=K+s^2*alpha*q00*C.T*B^-1*C.
# The charge RHS folds the Gauss projection into this single composite solve.
program = pops.Program("Lie CN source and SSPRK2 transport")
q = program.state(plasma_U)
half = 0.5
s = half * program.dt
scope = Hierarchy()
coefficients = program.condensed_coeffs("Schur tensor", state=q.n,
    linear_operator=rotation, subset=(1, 2), c=ALPHA * s * s, th_dt=s, c_rho=0,
    gradient_map=gradient_map, base_tensor=base_tensor)
rhs = program.condensed_rhs(program.scalar_field("Schur rhs"), state=q.n,
    linear_operator=rotation, subset=(1, 2), th_dt=s, g=s,
    gradient_map=gradient_map, base_tensor=base_tensor, charge_component=0)
# The zero initial guess avoids interpreting diagnostic history as AB2 lagged data
# when regridding creates new fine cells. The Schur equation and tolerance are unchanged.
elliptic = program.matrix_free_operator("disk Schur operator", scope=scope)
program.set_apply(elliptic, lambda builder, _out, value:
    -builder.apply_laplacian_coeff(builder.scalar_field("Schur action"), value, coefficients))
potential = program.solve(LinearProblem(elliptic, rhs,
    scope=scope, nullspace=None),
    solver=CompositeTensorFAC(max_iter=300, rel_tol=1e-10, abs_tol=1e-12,
        correction_damping=0.5, fine_sweeps=64, coarse_cycles=512,
        boundary_conditions=(Neumann(0.), Dirichlet(0.), Periodic(), Periodic()),
        diagonal_average="arithmetic"), name="midpoint potential").consume(action=FailRun())
program.store_history("plasma.potential", potential)
mean_work = program.value("independent momentum scratch", 1 * q.n, at=q.n.point)
midpoint = program.condensed_reconstruct("midpoint momentum", state=mean_work, phi=potential,
    linear_operator=rotation, subset=(1, 2), th_dt=s, c_rho=0,
    gradient_map=gradient_map, gradient_scale=ALPHA)
source_endpoint = program.value("CN source endpoint", 2 * midpoint - q.n,
                                at=program.stage("source endpoint", c=0))
k0 = program.value("transport k0", transport(source_endpoint))
predictor = program.value("SSPRK2 predictor", source_endpoint + program.dt * k0,
                          at=program.stage("transport predictor", c=1))
k1 = program.value("transport k1", transport(predictor))
accepted = program.value("accepted full-model state",
    source_endpoint + half * program.dt * (k0 + k1), at=q.next.point)
program.commit(q.next, accepted)
program.step_strategy(AdaptiveCFL(cfl=CFL, max_dt=MAX_DT))
case.program(program)


# 7. Positive spatial cell quadrature of the exact initial drift distribution.
# The mode-l disk Green function includes the grounded wall and the background.
ring_density = where(between(r, R0, R1), MEAN_RING + PERTURBATION * sin(MODE * theta), BACKGROUND)
inner_integral = (minimum(maximum(r, R0), R1) ** (MODE + 2) - R0 ** (MODE + 2)) / (MODE + 2)
outer_integral = (R1 ** (2 - MODE) - minimum(maximum(r, R0), R1) ** (2 - MODE)) / (2 - MODE)
whole_integral = (R1 ** (MODE + 2) - R0 ** (MODE + 2)) / (MODE + 2)
psi_mode = PERTURBATION / (2 * MODE) * (
    inner_integral / r**MODE + r**MODE * outer_integral
    - r**MODE / R**(2 * MODE) * whole_integral)
psi_mode_r = PERTURBATION / 2 * (
    -inner_integral / r**(MODE + 1) + r**(MODE - 1) * outer_integral
    - r**(MODE - 1) / R**(2 * MODE) * whole_integral)
psi_r = -BACKGROUND * r / 2 - (MEAN_RING - BACKGROUND) * (
    minimum(maximum(r, R0), R1)**2 - R0**2) / (2 * r) + psi_mode_r * sin(MODE * theta)
initial_ur = -(ALPHA / OMEGA) * MODE * psi_mode * cos(MODE * theta) / r
initial_utheta = (ALPHA / OMEGA) * psi_r
initial_ux = initial_ur * cos(theta) - initial_utheta * sin(theta)
initial_uy = initial_ur * sin(theta) + initial_utheta * cos(theta)
case.initials.add(InitialCondition(state=plasma_U,
    value=Analytic(frame=frame, components=(r * ring_density,
        r * ring_density * initial_ux, r * ring_density * initial_uy)),
    projection=ConservativeCellAverage()))


# 8. Dynamic synchronous AMR; every field solve covers the complete hierarchy.
indicator = ValueExpr(plasma_U)["density"]
refine_threshold = case.param(RuntimeParam("refine_weighted_density", default=0.05))
coarsen_threshold = case.param(RuntimeParam("coarsen_weighted_density", default=0.01))
tagging = AMRTagging(rules=(Tag(indicator > case.value(refine_threshold)),
    Coarsen(indicator < case.value(coarsen_threshold)), Buffer(cells=2)),
    hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS)
transfer = AMRTransfer()
transfer.state(plasma_U, StateTransfer())
layout = AMR(grid=grid,
    patch_layout=PatchLayout(distribute_coarse=True, coarse_max_grid=COARSE_MAX_GRID),
    clustering=BergerRigoutsos(minimum_efficiency=0.7, minimum_box_size=4,
                              maximum_box_size=CLUSTER_MAX_GRID),
    load_balance=SpaceFillingCurve(),
    hierarchy=AMRHierarchy(max_levels=MAX_LEVELS, ratios=(2,) * (MAX_LEVELS - 1)),
    tagging=tagging, regrid=AMRRegrid(schedule=every(10, clock=program.clock)),
    transfer=transfer, execution=AMRExecution.synchronous())


# 9. Public compile/bind. ROMEO stages native checkpoint transactions node-locally.
preparation_wall = time.monotonic()
if RANK == 0:
    print("Validating the authored Euler case", flush=True)
validated = pops.validate(case)
if RANK == 0:
    print("Resolving the finite-volume AMR layout", flush=True)
resolved = pops.resolve(validated, layout=layout)
if RANK == 0:
    print("Compiling the native MPI/Kokkos model and Program", flush=True)
artifact = pops.compile(resolved)
if artifact.platform_manifest.communicator.require("Hoffart MPI case") != "MPI_COMM_WORLD":
    raise RuntimeError("this tutorial requires a PoPS MPI build")
simulation = pops.bind(artifact,
    resources={"execution_context": pops.ExecutionContext.mpi_world(artifact)})
if RANK == 0:
    print("Native case bound in %.3f seconds" % (time.monotonic() - preparation_wall), flush=True)
if RESTART:
    simulation.restart(RESTART)
start_wall = time.monotonic()
# The analytic initial mass is unchanged across scheduler segments; the native
# segment-start integral is recorded separately so restart cannot reset drift.
initial_mass = math.pi * ((R1**2 - R0**2) * MEAN_RING
                         + (R**2 - R1**2 + R0**2) * BACKGROUND)
segment_initial_mass = simulation.integral("plasma", 0)
parameters = dict(model="Euler", mode=MODE, radius=R, ring=(R0, R1), alpha=ALPHA,
    omega=OMEGA, temperature=TEMPERATURE, background=BACKGROUND, mean_ring=MEAN_RING,
    perturbation=PERTURBATION, nr=NR, ntheta=NTHETA, max_levels=MAX_LEVELS, cfl=CFL,
    coarse_max_grid=COARSE_MAX_GRID, cluster_max_grid=CLUSTER_MAX_GRID, distribute_coarse=True,
    potential_history_slot=1, potential_history_contract="scalar-output-field-v1",
    potential_history_transfer="authenticated-1to1-retain-overlap-v1", field_initial_guess="zero",
    time_calendar="absolute cap-safe subdivisions of exact decimal output intervals",
    output_interval=OUTPUT_INTERVAL, growth_output_interval=GROWTH_OUTPUT_INTERVAL,
    growth_output_end=GROWTH_OUTPUT_END,
    max_dt=MAX_DT, t_end=T_END, split="source-first Lie", source="CN Schur, full Gauss restart",
    spatial="mapped MUSCL Minmod Rusanov, SSPRK2", mpi_ranks=WORLD.Get_size(),
    kokkos_threads=int(os.environ.get("POPS_THREADS", "1")),
    artifact_identity=artifact.artifact_identity.token)
if RANK == 0:
    (OUTPUT / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")


# 10. Actual numerical snapshots and checkpoints at accepted native boundaries.
# The first real interval supplies a native near-zero-time Fourier normalization.
# Every potential is timestamped at its actual source midpoint, never at density's endpoint.
# Form the schedule exactly before converting each target once to native binary64.
# This prevents unions of decimal cadences from introducing one-ULP duplicate times.
final_target = Fraction(str(T_END))
target_fractions = {Fraction(0), final_target}
for cadence, stop in ((Fraction(str(OUTPUT_INTERVAL)), final_target),
                      (Fraction(str(GROWTH_OUTPUT_INTERVAL)),
                       min(final_target, Fraction(str(GROWTH_OUTPUT_END))))):
    target_fractions.update(cadence * index for index in range(1, math.floor(stop / cadence) + 1))
for fixed in ("0.0000000001", "0.1", "1.25", "2.5", "3.75", "5", "6.25", "7.5", "8.75", "10"):
    if Fraction(fixed) <= final_target:
        target_fractions.add(Fraction(fixed))
targets = [float(value) for value in sorted(target_fractions)]
if any(right <= left for left, right in zip(targets, targets[1:], strict=False)):
    raise ValueError("distinct requested output times collapse in native binary64")
diagnostics = []
maximum_chunk_seconds = 0.0
last_checkpoint_wall = start_wall
last_checkpoint_step = -1
wall_stop = False
for target_index, target in enumerate(targets):
    if target < simulation.time():
        continue
    # Always anchor subdivisions at the global scheduled interval, including after
    # restart. Regenerating the remaining interval would change accepted dt values.
    interval_start = targets[max(0, target_index - 1)]
    chunk_count = max(1, math.ceil((target - interval_start) / MAX_DT))
    chunk_ends = np.array([target])
    if target > interval_start:
        while True:
            chunk_ends = np.linspace(interval_start, target, chunk_count + 1)
            chunk_widths = np.diff(chunk_ends)
            if np.any(chunk_widths <= 0):
                raise ValueError("requested timestep calendar is not representable in binary64")
            if np.all(chunk_widths <= MAX_DT):
                break
            chunk_count += 1
        chunk_ends = chunk_ends[1:]
    for chunk_end in chunk_ends:
        chunk_end = float(chunk_end)
        if chunk_end <= simulation.time():
            continue
        elapsed = WORLD.allreduce(time.monotonic() - start_wall, op=MPI.MAX)
        if WALL_SECONDS - elapsed <= max(60.0, 2.5 * maximum_chunk_seconds):
            wall_stop = True
            break
        chunk_start = time.monotonic()
        chunk_start_time = simulation.time()
        # AdaptiveCFL may take several smaller steps; public run must reach this
        # absolute endpoint. Neither max_steps nor clock guards are relaxed.
        report = pops.run(simulation, t_end=chunk_end, max_steps=100_000_000, console=False)
        chunk_seconds = WORLD.allreduce(time.monotonic() - chunk_start, op=MPI.MAX)
        maximum_chunk_seconds = max(maximum_chunk_seconds, chunk_seconds)
        last_accepted_dt = simulation.history_slot_dt("plasma.potential", 0, 1)
        if RANK == 0:
            progress = dict(time=simulation.time(), macro_step=simulation.macro_step(),
                n_levels=simulation.n_levels(), elapsed_seconds=time.monotonic() - start_wall,
                chunk_start_time=chunk_start_time, requested_chunk_end=chunk_end,
                calendar_interval_start=interval_start, calendar_interval_end=target,
                calendar_subintervals=chunk_count, chunk_seconds=chunk_seconds,
                accepted_steps_in_chunk=report.accepted_steps,
                rejected_steps_in_chunk=report.rejected_steps, last_accepted_dt=last_accepted_dt)
            progress_tmp = OUTPUT / ".progress.json.tmp"
            progress_tmp.write_text(json.dumps(progress) + "\n")
            progress_tmp.replace(OUTPUT / "progress.json")
            with (OUTPUT / "chunks.jsonl").open("a") as stream:
                stream.write(json.dumps(progress) + "\n")
        checkpoint_elapsed = WORLD.allreduce(time.monotonic() - last_checkpoint_wall, op=MPI.MAX)
        if checkpoint_elapsed >= 300.0:
            simulation.checkpoint(CHECKPOINT.parent / ("step-%012d.npz" % simulation.macro_step()))
            last_checkpoint_wall = time.monotonic()
            last_checkpoint_step = simulation.macro_step()
    levels = simulation.n_levels()
    patch_table = simulation.amr.patch_table().to_dict()
    saved = {}
    for level in range(levels):
        factor = 2**level
        state_array = np.asarray(simulation.block_level_state_global("plasma", level),
                                 dtype=np.float64).reshape(3, NTHETA * factor, NR * factor)
        saved["q0_level%d" % level] = state_array[0]
        if simulation.macro_step() > 0:
            # End-of-step rotation leaves the newest accepted field in raw slot one.
            saved["psi_level%d" % level] = np.asarray(
                simulation.history_global("plasma.potential", level, 1), dtype=np.float64
            ).reshape(NTHETA * factor, NR * factor)
    potential_time = None
    if simulation.macro_step() > 0:
        potential_time = simulation.time() - .5 * simulation.history_slot_dt("plasma.potential", 0, 1)
    mass = simulation.integral("plasma", 0)
    row = dict(time=simulation.time(), potential_time=potential_time,
        macro_step=simulation.macro_step(), mass=mass, initial_mass=initial_mass,
        initial_mass_reference="analytic benchmark", segment_initial_mass=segment_initial_mass,
        elapsed_seconds=time.monotonic() - start_wall, levels=levels,
        fine_patches=patch_table["n_patches"])
    if RANK == 0:
        np.savez_compressed(OUTPUT / ("snapshot-%012d.npz" % simulation.macro_step()),
            **saved, metadata=json.dumps(row), patches=json.dumps(patch_table),
            parameters=json.dumps(parameters))
        diagnostics.append(row)
        with (OUTPUT / "diagnostics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    if simulation.macro_step() > 0 and last_checkpoint_step != simulation.macro_step():
        simulation.checkpoint(CHECKPOINT.parent / ("step-%012d.npz" % simulation.macro_step()))
        last_checkpoint_wall = time.monotonic()
        last_checkpoint_step = simulation.macro_step()
    elapsed = WORLD.allreduce(time.monotonic() - start_wall, op=MPI.MAX)
    if wall_stop or elapsed >= WALL_SECONDS:
        break
if simulation.macro_step() == 0:
    raise RuntimeError("The wall budget ended before the first accepted step; no restart is available")
simulation.checkpoint(CHECKPOINT)
if RANK == 0:
    print("Accepted checkpoint: %s; physical time %.17g" % (CHECKPOINT, simulation.time()), flush=True)
