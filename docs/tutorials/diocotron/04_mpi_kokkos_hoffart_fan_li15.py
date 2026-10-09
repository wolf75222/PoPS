#!/usr/bin/env python3
"""The authors' Hoffart disk benchmark with the separately named Fan--Li15 system.

PoPS stores q=r*M for all fifteen Cartesian velocity moments on (r,theta).
Transport includes both the generalized-Hermite Grad flux and the nonconservative
Fan--Li regularization, with one declared path and common face covector. The
source keeps the coupled Schur/CN mean and uses an exact centered gyro rotation.
Source-first Lie splitting and first-order spatial reconstruction require
independent time/space refinement; no full-source exactness or AP is asserted.

The complete pipeline is authored linearly at module scope. Its native Fan--Li
face/CFL, candidate-admissibility and AMR side-contribution integration is under
development: this file is not an executable or scientifically qualified case in
the current unbuilt worktree. See FAN_LI15_METHOD.md before compiling or running.
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
from pops.lib.amr import CoarseFineInjection, ConservativeInjection, StateTransfer
from pops.lib.amr import BergerRigoutsos, SpaceFillingCurve
from pops.lib.initial import Analytic
from pops.linalg import LinearProblem
from pops.math import ValueExpr, ddt, div
from pops.moments import fan_li15_expressions, moment_indices, moment_names
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics import FanLi15RawMomentPath, PathConservativeFiniteVolume
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
OUTPUT = Path(os.environ.get("POPS_RUN_OUTPUT", "hoffart-fan-li15-mode%d" % MODE)).resolve()
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
model = pops.Model("Hoffart Fan Li15", frame=frame)
U = model.state("U", components=tuple(moment_names(4)),
    roles={"M00": Density(), "M10": Momentum(axis=radial_axis),
           "M01": Momentum(axis=angular_axis)})
# Declare owned geometry reads before F/B; their actual providers are registered
# on the final module after the model's physical operators have been declared.
radius = model.auxiliary("radius", frame=frame.canonical_id)
cosine = model.auxiliary("cosine", frame=frame.canonical_id)
sine = model.auxiliary("sine", frame=frame.canonical_id)
radial_cosine = model.auxiliary("radial_cosine", frame=frame.canonical_id)
radial_sine = model.auxiliary("radial_sine", frame=frame.canonical_id)
angular_cosine = model.auxiliary("angular_cosine", frame=frame.canonical_id)
angular_sine = model.auxiliary("angular_sine", frame=frame.canonical_id)
physical = fan_li15_expressions(U)
# These zero-threshold floating predicates are additional recovery checks.
# They are not an exact-raw SPD certificate. Native face/candidate certification
# must also refuse an indeterminate covariance; no floor or projection is used.
theta_xx, theta_xy, theta_yy = physical.temperature
model.recovery_admissibility(
    M00=physical.rho > 0,
    M20=theta_xx > 0,
    M02=theta_yy > 0,
    M11=theta_xx * theta_yy - theta_xy * theta_xy > 0)
covectors = {
    radial_axis: (radial_cosine, radial_sine),
    angular_axis: (-angular_sine / radius, angular_cosine / radius),
}
physical_flux = model.flux("mapped_transport", frame=frame, state=U,
    components={axis: physical.directional_flux(g) for axis, g in covectors.items()})
regularized_indices = (4, 8, 11, 13, 14)
conserved_components = tuple(name for slot, name in enumerate(moment_names(4))
                             if slot not in regularized_indices)
nonconservative = model.nonconservative_product("Fan Li regularization", state=U,
    matrices={axis: physical.directional_nonconservative_matrix(g)
              for axis, g in covectors.items()},
    conservative_components=conserved_components)
# The five degree-four rows retain B grad(q). The other ten rows are conservative.
# B(q)q=0, so q=r*M introduces no extra radial nonconservative source.
# The native path method supplies its full DF+B face bound and matching CFL;
# a Jacobian of the Grad flux alone is not the Fan--Li characteristic matrix.
transport = model.rate("path-conservative transport",
    equation=ddt(U) == -div(physical_flux) - nonconservative)


# 4. Authored Lorentz matrix, physical gradient map C and polar Poisson tensor K.
rotation_entries = [[0.0 for column in range(15)] for row in range(15)]
rotation_entries[1][5], rotation_entries[5][1] = OMEGA, -OMEGA
map_entries = [[0.0 for column in range(15)] for row in range(15)]
map_entries[1][1], map_entries[1][5] = cosine, -sine / radius
map_entries[5][1], map_entries[5][5] = sine, cosine / radius
tensor_entries = [[0.0 for column in range(15)] for row in range(15)]
tensor_entries[1][1], tensor_entries[5][5] = radius, 1.0 / radius
rotation = model.operator("Lorentz rotation", returns=model.local_linear_operator(
    "Lorentz matrix", on=U, matrix=tuple(tuple(row) for row in rotation_entries)))
gradient_map = model.operator("physical gradient map", returns=model.local_linear_operator(
    "C", on=U, matrix=tuple(tuple(row) for row in map_entries)))
base_tensor = model.operator("polar Poisson tensor", returns=model.local_linear_operator(
    "K", on=U, matrix=tuple(tuple(row) for row in tensor_entries)))
module = model.module
for name, expression in (
    ("radius", r), ("cosine", cos(theta)), ("sine", sin(theta)),
    ("radial_cosine", cos(theta) * sin(half_angle) / half_angle),
    ("radial_sine", sin(theta) * sin(half_angle) / half_angle),
    ("angular_cosine", cos(theta) / cos(half_angle)),
    ("angular_sine", sin(theta) / cos(half_angle)),
):
    auxiliary = module.aux_handle(module.aux()[name])
    module.aux_provider(AnalyticAux(auxiliary, expression, frame=frame,
                                  boundary=AuxiliaryBoundary(width=2, kind="foextrap")))
module.operator_registry().get(physical_flux.reg_name).requirements["aux"] = (
    "radius", "radial_cosine", "radial_sine", "angular_cosine", "angular_sine")
module.operator_registry().get(nonconservative.reg_name).requirements["aux"] = (
    "radius", "radial_cosine", "radial_sine", "angular_cosine", "angular_sine")
module.operator_registry().get(gradient_map.name).requirements["aux"] = ("radius", "cosine", "sine")
module.operator_registry().get(base_tensor.name).requirements["aux"] = ("radius",)


# 5. Finite volumes, conducting electrostatics and the benchmark's open transport wall.
case = pops.Case("Hoffart Fan Li15 mode %d" % MODE)
plasma = case.block("plasma", model=model)
plasma_U = plasma[U]
numerics = DiscretizationPlan()
path = FanLi15RawMomentPath(nonconservative, frame=frame, covectors=covectors)
# Bounded physical raw moments meet a zero-measure face at the disk pole.
# Both flux and path terms vanish there before evaluating an inadmissible q=0
# trace. This geometric contract is separate from a generic NoFlux boundary.
numerics.rates.add(transport, PathConservativeFiniteVolume(flux=physical_flux, path=path,
    variables=variables.Conservative(U),
    reconstruction=reconstruction.FirstOrder(),
    riemann=riemann.Rusanov(), zero_measure_faces=(frame.boundaries.x_min,)))
numerics.boundaries.add(TransportBoundarySet({
    frame.boundaries.x_min: NoFlux(state=plasma_U),
    frame.boundaries.x_max: Outflow(state=plasma_U),
}, periodic=PeriodicAxes((angular_axis,))))
case.numerics(numerics, block=plasma)


# 6. Full-Gauss-restart Schur source, followed by path-conservative SSPRK2 transport.
# psi=phi/alpha. B=I-s*Omega*J, A=K+s^2*alpha*q00*C.T*B^-1*C.
# The charge RHS folds the Gauss projection into this single composite solve.
program = pops.Program("Lie CN mean and exponential centered source with Fan Li SSPRK2")
q = program.state(plasma_U)
half = 0.5
s = half * program.dt
scope = Hierarchy()
coefficients = program.condensed_coeffs("Schur tensor", state=q.n,
    linear_operator=rotation, subset=(1, 5), c=ALPHA * s * s, th_dt=s, c_rho=0,
    gradient_map=gradient_map, base_tensor=base_tensor)
rhs = program.condensed_rhs(program.scalar_field("Schur rhs"), state=q.n,
    linear_operator=rotation, subset=(1, 5), th_dt=s, g=s,
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
        coarse_method="gmres", coarse_restart=64, coarse_preconditioner="polar_poisson",
        interface_coupling="fine_flux",
        boundary_conditions=(Neumann(0.), Dirichlet(0.), Periodic(), Periodic()),
        diagonal_average="arithmetic"), name="midpoint potential").consume(action=FailRun())
program.store_history("plasma.potential", potential)
mean_work = program.value("independent momentum scratch", 1 * q.n, at=q.n.point)
midpoint = program.condensed_reconstruct("midpoint momentum", state=mean_work, phi=potential,
    linear_operator=rotation, subset=(1, 5), th_dt=s, c_rho=0,
    gradient_map=gradient_map, gradient_scale=ALPHA)
mean_endpoint = program.value("CN first-moment endpoint", 2 * midpoint - q.n,
                              at=program.stage("source endpoint", c=0))
# One affine map v_new=exp(dt*Omega*J)*(v_old-u_old)+u_CN acts on every moment.
# It rotates centered tensors exactly for the homogeneous source cell; the mean
# remains CN. This preserves the local discrete CN work identity in exact
# arithmetic, without establishing global energy balance or a full AP method.
source_endpoint = program.affine_moment_update(q.n, mean_endpoint,
    linear_operator=rotation, theta_dt=s, order=4, rotation="exponential",
    name="exponential centered gyro map with CN mean")
source_geometry = program.input_fields(source_endpoint, for_rate=transport,
    name="source-stage transport geometry")
k0 = program.value("transport k0", transport(source_endpoint, source_geometry))
predictor = program.value("SSPRK2 predictor", source_endpoint + program.dt * k0,
                          at=program.stage("transport predictor", c=1))
predictor_geometry = program.input_fields(predictor, for_rate=transport,
    name="predictor-stage transport geometry")
k1 = program.value("transport k1", transport(predictor, predictor_geometry))
accepted = program.value("accepted Fan Li15 candidate",
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
# The thermal Gaussian is retained at the authors' T=1e-24. Conservative spatial
# quadrature includes physical subcell velocity variance; no variance floor or
# replacement Maxwellian is introduced in initialization or transport.
initial_x_moments = []
initial_y_moments = []
for degree in range(5):
    initial_x_moments.append(sum(
        math.comb(degree, 2*k) * math.prod(range(1, 2*k, 2)) * TEMPERATURE**k * initial_ux**(degree-2*k)
        for k in range(degree//2+1)))
    initial_y_moments.append(sum(
        math.comb(degree, 2*k) * math.prod(range(1, 2*k, 2)) * TEMPERATURE**k * initial_uy**(degree-2*k)
        for k in range(degree//2+1)))
initial_components = tuple(r * ring_density * initial_x_moments[i] * initial_y_moments[j]
                           for i, j in moment_indices(4))
case.initials.add(InitialCondition(state=plasma_U,
    value=Analytic(frame=frame, components=initial_components),
    projection=ConservativeCellAverage()))


# 8. Dynamic synchronous AMR; every field solve covers the complete hierarchy.
indicator = ValueExpr(plasma_U)["M00"]
refine_threshold = case.param(RuntimeParam("refine_weighted_density", default=0.05))
coarsen_threshold = case.param(RuntimeParam("coarsen_weighted_density", default=0.01))
tagging = AMRTagging(rules=(Tag(indicator > case.value(refine_threshold)),
    Coarsen(indicator < case.value(coarsen_threshold)), Buffer(cells=2)),
    hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS)
transfer = AMRTransfer()
# Convex parent injection and volume averaging preserve H1 SPD in exact arithmetic.
# This does not prove invariant-domain preservation by the transport update, or
# full degree-four moment realizability of a Fan--Li state. Spatial order is one.
transfer.state(plasma_U, StateTransfer(prolongation=ConservativeInjection(),
                                      coarse_fine=CoarseFineInjection()))
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
    print("Validating the authored Fan Li15 case", flush=True)
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
parameters = dict(model="FanLi15", mode=MODE, radius=R, ring=(R0, R1), alpha=ALPHA,
    omega=OMEGA, temperature=TEMPERATURE, background=BACKGROUND, mean_ring=MEAN_RING,
    perturbation=PERTURBATION, nr=NR, ntheta=NTHETA, max_levels=MAX_LEVELS, cfl=CFL,
    coarse_max_grid=COARSE_MAX_GRID, cluster_max_grid=CLUSTER_MAX_GRID, distribute_coarse=True,
    potential_history_slot=1, potential_history_contract="scalar-output-field-v1",
    potential_history_transfer="authenticated-1to1-retain-overlap-v1", field_initial_guess="zero",
    field_coarse_method="gmres", field_coarse_restart=64, field_coarse_iteration_cap=512,
    field_coarse_preconditioner="polar_poisson",
    field_interface_coupling="fine_flux",
    time_calendar="absolute cap-safe subdivisions of exact decimal output intervals",
    output_interval=OUTPUT_INTERVAL, growth_output_interval=GROWTH_OUTPUT_INTERVAL,
    growth_output_end=GROWTH_OUTPUT_END,
    max_dt=MAX_DT, t_end=T_END, split="source-first Lie",
    source="CN Schur mean, exponential centered gyro phase, full Gauss restart",
    source_rotation="exponential", transport_conserved_components=conserved_components,
    transport_regularized_indices=regularized_indices,
    spatial="mapped first-order path-conservative Rusanov, full DF+B bound, SSPRK2",
    transport_path="straight complete raw moments; analytic polynomial/logarithm integral",
    admissibility="rho>0 and H1 SPD; native certification required, no floor or projection",
    snapshot_raw_moments="all fifteen q=r*M components, q-outer ordering",
    mpi_ranks=WORLD.Get_size(),
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
                                 dtype=np.float64).reshape(15, NTHETA * factor, NR * factor)
        saved["q0_level%d" % level] = state_array[0]
        saved["moments_level%d" % level] = state_array
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
