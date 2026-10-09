#!/usr/bin/env python3
"""Homogeneous isotropic BGK relaxation in the fifteen raw-moment basis.

The collision ODE does not depend on a transport closure. The script compares
actual saved endpoints with the exact continuum and SSPRK2 solutions.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--kn", choices=("0.01", "0.1", "1"), required=True)
parser.add_argument("--steps", type=int, choices=(40, 80, 160), default=80)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError("The actual run needs a new output directory")

import numpy as np
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.moments import moment_indices, moment_names
from pops.numerics import DiscretizationPlan, StateStorage
from pops.time import ExternalTimeGrid, StagePoint, TimePoint

pops.set_threads(int(os.environ.get("POPS_THREADS", "2")))
KN = float(args.kn)
STEPS = args.steps
CELLS = (4, 4)

# Physical initial distribution: a positive mixture of two diagonal Gaussians.
# The independent JSON is checked below; it does not define these moments.
mixture_density = Fraction(3, 2)
mixture_weights = (Fraction(1, 4), Fraction(3, 4))
component_means = ((Fraction(7, 4), Fraction(11, 8)),
                   (Fraction(-1, 4), Fraction(-5, 8)))
component_variances = ((Fraction(17, 4), Fraction(9, 4)),
                       (Fraction(17, 4), Fraction(9, 4)))
assert mixture_density > 0 and sum(mixture_weights) == 1
assert all(weight > 0 for weight in mixture_weights)
assert all(value > 0 for diagonal in component_variances for value in diagonal)
mixture_mean = tuple(sum(weight*mean[axis] for weight, mean in
                        zip(mixture_weights, component_means, strict=True)) for axis in range(2))
mixture_covariance = tuple(tuple(sum(weight*((variances[i] if i == j else 0)
    +(mean[i]-mixture_mean[i])*(mean[j]-mixture_mean[j]))
    for weight, mean, variances in zip(mixture_weights, component_means, component_variances, strict=True))
    for j in range(2)) for i in range(2))
mixture_temperature = (mixture_covariance[0][0]+mixture_covariance[1][1])/2
initial_exact = []
equilibrium_exact = []
for px, py in moment_indices(4):
    raw_moment = Fraction(0)
    for weight, mean, variances in zip(mixture_weights, component_means, component_variances, strict=True):
        axis_moments = []
        for degree, center, variance in zip((px, py), mean, variances, strict=True):
            axis_moments.append(sum(Fraction(math.comb(degree, 2*k)*math.prod(range(1, 2*k, 2)))
                *variance**k*center**(degree-2*k) for k in range(degree//2+1)))
        raw_moment += mixture_density*weight*axis_moments[0]*axis_moments[1]
    initial_exact.append(raw_moment)
    equilibrium_axes = []
    for degree, center in zip((px, py), mixture_mean, strict=True):
        equilibrium_axes.append(sum(Fraction(math.comb(degree, 2*k)*math.prod(range(1, 2*k, 2)))
            *mixture_temperature**k*center**(degree-2*k) for k in range(degree//2+1)))
    equilibrium_exact.append(mixture_density*equilibrium_axes[0]*equilibrium_axes[1])
initial_vector = np.array([float(value) for value in initial_exact])
equilibrium_vector = np.array([float(value) for value in equilibrium_exact])
TAU = KN / (2.0*float(mixture_density)*math.sqrt(float(mixture_temperature)))
T_END = 2.0 * TAU
TIME_GRID = tuple(T_END * k / STEPS for k in range(STEPS)) + (T_END,)
INPUT_FILE = Path(__file__).with_name("analytic-inputs.json")
inputs = json.loads(INPUT_FILE.read_text())
assert inputs["actual_simulation"] is False
assert inputs["ordering"] == list(moment_names(4))
assert initial_exact == [Fraction(value) for value in inputs["exact_initial_rational"]]
assert equilibrium_exact == [Fraction(value) for value in inputs["exact_equilibrium_rational"]]
initial_state = np.broadcast_to(initial_vector[:, None, None], (15, CELLS[1], CELLS[0])).copy()

# =============================================================================
# 1. The physical collision law: isotropic local Maxwellian, not covariance matching.
# =============================================================================
frame = Rectangle("homogeneous_collision_cells", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
model = pops.Model("RIEMOM isotropic BGK raw15", frame=frame)
U = model.state("U", components=tuple(moment_names(4)))
M = dict(zip(moment_indices(4), U, strict=True))
rho = M[(0, 0)]
u, v = M[(1, 0)] / rho, M[(0, 1)] / rho
c20 = M[(2, 0)] / rho - u*u
c11 = M[(1, 1)] / rho - u*v
c02 = M[(0, 2)] / rho - v*v
theta = 0.5 * (c20 + c02)
nu = 2.0 * rho * sqrt(theta) / KN
gx = (1.0, u, u*u + theta, u*u*u + 3.0*u*theta,
      u*u*u*u + 6.0*u*u*theta + 3.0*theta*theta)
gy = (1.0, v, v*v + theta, v*v*v + 3.0*v*theta,
      v*v*v*v + 6.0*v*v*theta + 3.0*theta*theta)
equilibrium = tuple(rho * gx[p] * gy[q] for p, q in moment_indices(4))
source_values = [0.0 if k in (0, 1, 5) else nu * (equilibrium[k] - U[k])
                 for k in range(15)]
# The two diagonal stresses exchange exactly opposite source contributions.
source_values[9] = -source_values[2]
collision_source = model.source("isotropic_BGK", on=U, value=tuple(source_values))
collision = model.rate("collision", equation=ddt(U) == collision_source)

# This identity map only refuses invalid inputs; it does not change any moment.
strict_domain = model.local_transform(
    "strict_collision_domain", tuple(U), on=U,
    valid_if=(rho > 0.0) * (c20 > 0.0) * (theta > 0.0) * (c20*c02 - c11*c11 > 0.0),
)

# =============================================================================
# 2. Source-only cell storage: no spatial flux or numerical characteristic policy.
# =============================================================================
case = pops.Case("separate_Knudsen_BGK_validation")
gas = case.block("gas", model=model, states=(U,))
numerics = DiscretizationPlan()
numerics.rates.add(collision, StateStorage())
case.numerics(numerics, block=gas)
grid = CartesianGrid(frame=frame, cells=CELLS, periodic=PeriodicAxes(frame.axes))

# =============================================================================
# 3. Explicit SSPRK2 with a domain refusal before every source and before commit.
# =============================================================================
program = pops.Program("SSPRK2 isotropic BGK")
q = program.state(gas[U])
q0 = program.transform(q.n, transform=strict_domain, name="checked_initial_stage")
k0 = program.value("collision_k0", collision(q0), at=q.n.point)
stage = StagePoint("collision_stage", {"main": TimePoint(program.clock, 1)})
predictor = program.value("collision_predictor", q.n + program.dt*k0, at=stage)
q1 = program.transform(predictor, transform=strict_domain, name="checked_predictor")
k1 = program.value("collision_k1", collision(q1), at=stage)
candidate = program.value("collision_candidate",
    q.n + Fraction(1, 2)*program.dt*k0 + Fraction(1, 2)*program.dt*k1, at=q.next.point)
checked = program.transform(candidate, transform=strict_domain, name="checked_candidate")
program.commit(q.next, checked)
program.step_strategy(ExternalTimeGrid("time_grid"))
case.program(program)

# =============================================================================
# 4. Compile, bind, advance and checkpoint the native simulation.
# =============================================================================
args.output.mkdir(parents=True, exist_ok=False)
configuration = dict(scope="homogeneous_collision_only_no_transport", kn=KN,
    tau=TAU, t_end=T_END, steps=STEPS, time_grid=TIME_GRID, cells=CELLS,
    input_sha256=hashlib.sha256(INPUT_FILE.read_bytes()).hexdigest(),
    case_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    source_model="RIEMOM isotropic BGK", integrator="SSPRK2", projections=False,
    floors=False, characteristic_or_transport_qualification=False,
    physical_mixture=dict(density=str(mixture_density), weights=[str(value) for value in mixture_weights],
        component_means=[[str(value) for value in mean] for mean in component_means],
        component_variances=[[str(value) for value in diagonal] for diagonal in component_variances],
        derived_mean=[str(value) for value in mixture_mean],
        derived_covariance=[[str(value) for value in row] for row in mixture_covariance],
        derived_temperature=str(mixture_temperature)))
(args.output / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
started = time.perf_counter()
artifact = pops.compile(pops.resolve(pops.validate(case), layout=Uniform(grid)))
artifact.verify()
simulation = pops.bind(artifact, initial_state={"gas": initial_state},
    resources={"execution_context": pops.ExecutionContext.mpi_world(artifact)})
bound_state = np.asarray(simulation.state_global("gas"), dtype=np.float64).reshape(initial_state.shape)
np.savez_compressed(args.output / "actual-initial.npz", state=bound_state,
                    time=simulation.time(), macro_step=simulation.macro_step())
assert np.array_equal(bound_state, initial_state)
report = pops.run(simulation, t_end=T_END, max_steps=STEPS, time_grid=TIME_GRID)
final = np.asarray(simulation.state_global("gas"), dtype=np.float64).reshape(initial_state.shape)
np.savez_compressed(args.output / "actual-final.npz", state=final,
    time=simulation.time(), macro_step=simulation.macro_step(),
    accepted_steps=report.accepted_steps, rejected_steps=report.rejected_steps)
simulation.checkpoint(args.output / "actual-final-checkpoint.npz")

# =============================================================================
# 5. Independent exact continuum and SSPRK2 scalar-amplification comparisons.
# =============================================================================
z = 6.0 * simulation.time() / KN
continuum = equilibrium_vector + math.exp(-z) * (initial_vector - equilibrium_vector)
amplification = math.prod(1.0 - h + 0.5*h*h for h in
    (6.0 * (b-a) / KN for a, b in zip(TIME_GRID[:-1], TIME_GRID[1:], strict=True)))
discrete = equilibrium_vector + amplification * (initial_vector - equilibrium_vector)
scale = np.maximum(1.0, np.maximum(np.abs(initial_vector), np.abs(equilibrium_vector)))
scaled_discrete_error = np.max(np.abs(final - discrete[:, None, None]) / scale[:, None, None])
continuum_error = np.max(np.abs(final - continuum[:, None, None]) / scale[:, None, None])
invariants = np.stack((final[0], final[1], final[5], final[2]+final[9]))
initial_invariants = np.array((initial_vector[0], initial_vector[1], initial_vector[5],
                               initial_vector[2]+initial_vector[9]))
invariant_error = np.max(np.abs(invariants - initial_invariants[:, None, None]) /
                         np.maximum(1.0, np.abs(initial_invariants))[:, None, None])
positive_h1 = 0
for raw in final.reshape(15, -1).T:
    rr, xx, yy, aa, bb, cc = (Fraction(float(raw[k])) for k in (0, 1, 5, 2, 6, 9))
    minors = (rr, rr*aa-xx*xx, rr*(aa*cc-bb*bb)-xx*(xx*cc-bb*yy)+yy*(xx*bb-aa*yy))
    if all(value > 0 for value in minors):
        positive_h1 += 1
roundoff_budget = 2000.0 * STEPS * np.finfo(np.float64).eps
diagnostics = dict(configuration, actual_time=simulation.time(), actual_macro_step=simulation.macro_step(),
    accepted_steps=report.accepted_steps, rejected_steps=report.rejected_steps,
    scaled_discrete_error=float(scaled_discrete_error), scaled_continuum_error=float(continuum_error),
    scaled_invariant_error=float(invariant_error), positive_exact_H1_cells=positive_h1,
    floating_comparison_budget=roundoff_budget, elapsed_seconds=time.perf_counter()-started)
(args.output / "actual-diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
assert report.accepted_steps == STEPS and report.rejected_steps == 0
assert simulation.macro_step() == STEPS and simulation.time() == T_END
assert np.isfinite(final).all() and positive_h1 == CELLS[0]*CELLS[1]
assert scaled_discrete_error <= roundoff_budget and invariant_error <= roundoff_budget
assert np.max(np.abs(final - final[:, :1, :1]) / scale[:, None, None]) <= roundoff_budget
print(json.dumps({"scope": "homogeneous_BGK_only", "diagnostics": str(args.output / "actual-diagnostics.json"),
                  "transport_or_hyperbolicity_qualified": False}))
