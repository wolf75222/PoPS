#!/usr/bin/env python3
"""Finite-volume Cartesian HYQMOM15 shock with isotropic Knudsen BGK collisions.

The full directional Jacobian and strict domain checks are retained. Actual
state snapshots and a final checkpoint are written with measured diagnostics.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError("The actual pilot needs a new output directory")

import numpy as np
import pops
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div, sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.moments import HyQMOM15Closure, moment_flux_expressions, moment_indices, moment_names
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.physics import Density, Momentum
from pops.time import AdaptiveCFL

pops.set_threads(int(os.environ.get("POPS_THREADS", "2")))
NX, NY = 64, 4
KN = 0.1
T_END = 0.01
MAX_STEPS = 10000
MAX_DT = 1.0e-4
OBSERVATION_INTERVALS = 200
TRANSPORT_CFL = 0.1
SOURCE_CFL = 0.05
DX, DY = 1.0/NX, 1.0/NY
physical_targets = tuple(float(Fraction(index, 20000)) for index in range(1, OBSERVATION_INTERVALS+1))
assert physical_targets[-1] == T_END

# =============================================================================
# 1. Exact Cartesian closure and complete directional characteristic policy.
# =============================================================================
frame = Rectangle("x_shock", lower=(-0.5, 0.0), upper=(0.5, 1.0)).frame(Cartesian2D())
x_axis, y_axis = frame.axes
model = pops.Model("Knudsen HYQMOM15 x shock", frame=frame)
U = model.state("U", components=tuple(moment_names(4)),
    roles={"M00": Density(), "M10": Momentum(axis=x_axis), "M01": Momentum(axis=y_axis)})
closed_flux = moment_flux_expressions(model, U, 4, HyQMOM15Closure(), robust=False)
physical_flux = model.flux("transport", frame=frame, state=U,
    components={x_axis: closed_flux.x, y_axis: closed_flux.y})
model.wave_speeds_from_jacobian(blocks=None, im_tol=None, eig_max_iter=1000)

# =============================================================================
# 2. RIEMOM's isotropic BGK law; no realizability relaxation or clipping.
# =============================================================================
M = dict(zip(moment_indices(4), U, strict=True))
rho = M[(0, 0)]
u, v = M[(1, 0)]/rho, M[(0, 1)]/rho
c20 = M[(2, 0)]/rho - u*u
c11 = M[(1, 1)]/rho - u*v
c02 = M[(0, 2)]/rho - v*v
theta = 0.5*(c20+c02)
nu = 2.0*rho*sqrt(theta)/KN
gx = (1.0, u, u*u+theta, u*u*u+3.0*u*theta,
      u*u*u*u+6.0*u*u*theta+3.0*theta*theta)
gy = (1.0, v, v*v+theta, v*v*v+3.0*v*theta,
      v*v*v*v+6.0*v*v*theta+3.0*theta*theta)
equilibrium = tuple(rho*gx[p]*gy[q] for p, q in moment_indices(4))
source_values = [0.0 if k in (0, 1, 5) else nu*(equilibrium[k]-U[k]) for k in range(15)]
source_values[9] = -source_values[2]
collision_source = model.source("isotropic_BGK", on=U, value=tuple(source_values))
balance = model.rate("transport_and_collision", equation=ddt(U) == -div(physical_flux)+collision_source)

# This compiled identity predicate evaluates nu from its actual input state.
# Because every actual dt <= MAX_DT, it proves nu*dt <= SOURCE_CFL before the stage.
stage_domain = model.local_transform("strict_BGK_stage_domain", tuple(U), on=U,
    valid_if=(rho > 0.0)*(c20 > 0.0)*(theta > 0.0)*(c20*c02-c11*c11 > 0.0)
             *(MAX_DT*nu <= SOURCE_CFL))

# =============================================================================
# 3. FirstOrder/HLL, outflow x and periodic y. No mapped metrics or AMR.
# =============================================================================
case = pops.Case("separate_exact_HYQMOM15_Knudsen_x_shock")
gas = case.block("gas", model=model, states=(U,))
gas_U = gas[U]
numerics = DiscretizationPlan()
numerics.rates.add(balance, FiniteVolume(flux=physical_flux,
    variables=variables.Conservative(U), reconstruction=reconstruction.FirstOrder(),
    riemann=riemann.HLL(waves=riemann.waves.FromJacobian())))
numerics.boundaries.add(TransportBoundarySet({
    frame.boundaries.x_min: Outflow(state=gas_U),
    frame.boundaries.x_max: Outflow(state=gas_U),
}, periodic=PeriodicAxes((y_axis,))))
case.numerics(numerics, block=gas)
grid = CartesianGrid(frame=frame, cells=(NX, NY), periodic=PeriodicAxes((y_axis,)))

# =============================================================================
# 4. One real forward-Euler stage of the entire conservative balance and BGK source.
# =============================================================================
program = pops.Program("Forward Euler strict HYQMOM15 BGK")
q = program.state(gas_U)
checked_current = program.transform(q.n, transform=stage_domain, name="checked_current_stage")
rhs = program.value("current_balance", balance(checked_current), at=q.n.point)
candidate = program.value("forward_euler_candidate", q.n+program.dt*rhs, at=q.next.point)
checked_candidate = program.transform(candidate, transform=stage_domain, name="checked_next_domain")
program.commit(q.next, checked_candidate)
program.step_strategy(AdaptiveCFL(cfl=TRANSPORT_CFL, max_dt=MAX_DT))
case.program(program)

# =============================================================================
# 5. Gaussian states with rho/p = 1/1 on the left and 0.1/0.1 on the right.
# =============================================================================
# Array axes are (component,y,x); the discontinuity is the actual x=0 face.
x_centers = -0.5+(np.arange(NX)+0.5)*DX
profile = np.where(x_centers < 0.0, 1.0, 0.1)
unit_gaussian = np.array((1., 0., 1., 0., 3., 0., 0., 0., 0., 1., 0., 1., 0., 0., 3.))
initial_state = np.broadcast_to(unit_gaussian[:, None, None]*profile[None, None, :], (15, NY, NX)).copy()

# =============================================================================
# 6. Explicit physical targets, with the one-step quadrature premise checked on
# =============================================================================
# each real RunReport. A smaller native CFL step is retained/refused, not hidden.
args.output.mkdir(parents=True, exist_ok=False)
configuration = dict(scope="conditional_exact_HYQMOM15_spatial_BGK_pilot", kn=KN,
    cells=(NX, NY), bounds=((-0.5, 0.0), (0.5, 1.0)), t_end=T_END,
    max_steps=MAX_STEPS, max_dt=MAX_DT, transport_cfl=TRANSPORT_CFL,
    physical_targets=physical_targets, intended_target_interval="1/20000",
    source_stage_frequency_bound=SOURCE_CFL, integrator="ForwardEuler",
    characteristic_matrix="full15_each_axis", im_tol=None, eig_max_iter=1000,
    projection=False, variance_floors=False, realizability_relaxation=False,
    initial_density=(1.0, 0.1), initial_pressure=(1.0, 0.1), initial_temperature=(1.0, 1.0),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    native_directional_speed_diagnostics="not exposed by the public RuntimeInstance used here")
(args.output / "configuration.json").write_text(json.dumps(configuration, indent=2)+"\n")
simulation = None
records = []
nonzero_collision_defect_observed = False
started = time.perf_counter()
try:
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=Uniform(grid)))
    artifact.verify()
    simulation = pops.bind(artifact, initial_state={"gas": initial_state},
        resources={"execution_context": pops.ExecutionContext.mpi_world(artifact)})
    current = np.asarray(simulation.state_global("gas"), dtype=np.float64).reshape(15, NY, NX)
    assert np.array_equal(current, initial_state)
    np.savez_compressed(args.output / "actual-step-000000.npz", state=current,
                        time=simulation.time(), macro_step=simulation.macro_step())
    for target in physical_targets:
        if len(records) >= MAX_STEPS:
            raise RuntimeError("The bounded spatial pilot exhausted its accepted-step cap")
        before_time, before_step = simulation.time(), simulation.macro_step()
        if not np.isfinite(current).all() or not (current[0] > 0.0).all():
            raise FloatingPointError("The actual current state is nonfinite or has nonpositive density")
        density = current[0]
        mean_x, mean_y = current[1]/density, current[5]/density
        temperature = 0.5*(current[2]/density-mean_x*mean_x+current[9]/density-mean_y*mean_y)
        if not np.isfinite(temperature).all() or not (temperature > 0.0).all():
            raise FloatingPointError("The actual current state has an invalid BGK temperature")
        frequency = 2.0*density*np.sqrt(temperature)/KN
        if not np.isfinite(frequency).all() or not (MAX_DT*frequency <= SOURCE_CFL).all():
            raise RuntimeError("The actual evolving BGK rate exceeds the authored stage bound")
        actual_gx = (np.ones_like(density), mean_x, mean_x**2+temperature,
                     mean_x**3+3*mean_x*temperature, mean_x**4+6*mean_x**2*temperature+3*temperature**2)
        actual_gy = (np.ones_like(density), mean_y, mean_y**2+temperature,
                     mean_y**3+3*mean_y*temperature, mean_y**4+6*mean_y**2*temperature+3*temperature**2)
        actual_equilibrium = np.stack([density*actual_gx[p]*actual_gy[r] for p, r in moment_indices(4)])
        collision_defect = float(np.max(np.abs(current-actual_equilibrium)))
        nonzero_collision_defect_observed |= collision_defect > 1e-12

        # An independent consistency oracle: FirstOrder Outflow has equal traces,
        # hence physical F(U) equals its consistent HLL boundary flux in real arithmetic.
        # These are host physical-flux values, not claimed native face-buffer readbacks.
        left_flux = np.asarray(model.flux_value(current[:, :, 0], {}, x_axis))
        right_flux = np.asarray(model.flux_value(current[:, :, -1], {}, x_axis))
        assert left_flux.shape == right_flux.shape == (15, NY)
        net_flux = DY*np.sum(left_flux-right_flux, axis=1)
        invariant_boundary_rate = np.array((net_flux[0], net_flux[1], net_flux[5], net_flux[2]+net_flux[9]))
        invariant_before = DX*DY*np.array((current[0].sum(), current[1].sum(), current[5].sum(),
                                         (current[2]+current[9]).sum()))
        report = pops.run(simulation, t_end=target, max_steps=MAX_STEPS-before_step)
        actual = np.asarray(simulation.state_global("gas"), dtype=np.float64).reshape(15, NY, NX)
        now, step = simulation.time(), simulation.macro_step()
        np.savez_compressed(args.output / ("actual-step-%06d.npz" % step), state=actual,
            time=now, macro_step=step, accepted_steps=report.accepted_steps, rejected_steps=report.rejected_steps)
        # Save the actual result before refusing an invalid one-step audit premise.
        if report.accepted_steps != 1 or report.rejected_steps != 0 or step != before_step+1:
            with (args.output / ("actual-interval-%06d-refusal.json" % step)).open("x") as stream:
                json.dump(dict(target=target, before_time=before_time, time=now,
                    before_step=before_step, macro_step=step, accepted_steps=report.accepted_steps,
                    rejected_steps=report.rejected_steps,
                    reason="one-step boundary quadrature premise is not satisfied"), stream, indent=2)
                stream.write("\n")
            raise RuntimeError("The actual interval does not contain exactly one accepted, zero-reject step")
        assert now == target
        clock_dt = now-before_time
        # Clock subtraction has an explicit binary64 rounding allowance; it never
        # changes the native cap or the zero-slack compiled source-frequency guard.
        clock_roundoff = 4*math.ulp(max(abs(now), abs(before_time), MAX_DT))
        assert 0.0 < clock_dt <= MAX_DT+clock_roundoff
        assert np.isfinite(actual).all()
        positive_h1 = 0
        for raw in actual.reshape(15, -1).T:
            rr, xx, yy, aa, bb, cc = (Fraction(float(raw[k])) for k in (0, 1, 5, 2, 6, 9))
            minors = (rr, rr*aa-xx*xx, rr*(aa*cc-bb*bb)-xx*(xx*cc-bb*yy)+yy*(xx*bb-aa*yy))
            if all(value > 0 for value in minors):
                positive_h1 += 1
        assert positive_h1 == NX*NY
        invariant_after = DX*DY*np.array((actual[0].sum(), actual[1].sum(), actual[5].sum(),
                                        (actual[2]+actual[9]).sum()))
        balance_error = np.abs(invariant_after-invariant_before-clock_dt*invariant_boundary_rate)
        balance_scale = np.maximum(1.0, np.maximum(np.abs(invariant_before), np.abs(invariant_after)))
        scaled_balance_error = float(np.max(balance_error/balance_scale))
        uniformity_error = float(np.max(np.abs(actual-actual[:, :1, :])/np.maximum(1.0, np.abs(actual))))
        roundoff_budget = 2000.0*(step+1)*np.finfo(np.float64).eps
        row = dict(before_time=before_time, time=now, macro_step=step, observed_clock_increment=clock_dt,
            clock_roundoff_bound=clock_roundoff, accepted_steps=report.accepted_steps,
            rejected_steps=report.rejected_steps, source_frequency_max=float(frequency.max()),
            authored_max_dt_times_frequency=float(MAX_DT*frequency.max()),
            collision_defect_max=collision_defect, positive_exact_H1_cells=positive_h1,
            scaled_invariant_boundary_balance_error=scaled_balance_error,
            y_uniformity_error=uniformity_error, roundoff_comparison_budget=roundoff_budget,
            physical_outflow_boundary_rate=invariant_boundary_rate.tolist())
        with (args.output / ("actual-step-%06d.json" % step)).open("x") as stream:
            json.dump(row, stream, indent=2, allow_nan=False)
            stream.write("\n")
        records.append(row)
        assert scaled_balance_error <= roundoff_budget and uniformity_error <= roundoff_budget
        current = actual
    assert simulation.time() == T_END and nonzero_collision_defect_observed
    simulation.checkpoint(args.output / "actual-final-checkpoint.npz")
    result = dict(configuration, status="completed_bounded_pilot_pending_independent_scientific_review",
        actual_time=simulation.time(), actual_macro_step=simulation.macro_step(),
        accepted_steps=len(records), observed_nonzero_collision_defect=nonzero_collision_defect_observed,
        elapsed_seconds=time.perf_counter()-started, full_moment_cone_or_global_hyperbolicity_claim=False,
        native_directional_speed_buffer_audit=False, native_boundary_face_buffer_audit=False)
    (args.output / "actual-completion.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
except Exception as failure:
    failure_record = dict(exception=type(failure).__name__, message=str(failure), traceback=traceback.format_exc(),
        elapsed_seconds=time.perf_counter()-started, accepted_observation_count=len(records))
    if simulation is not None:
        try:
            retained = np.asarray(simulation.state_global("gas"), dtype=np.float64).reshape(15, NY, NX)
            failure_record.update(last_observed_time=simulation.time(), last_observed_macro_step=simulation.macro_step())
            np.savez_compressed(args.output / "actual-state-after-failure.npz", state=retained,
                time=simulation.time(), macro_step=simulation.macro_step())
        except Exception as inspection_failure:
            failure_record["state_inspection_failure"] = repr(inspection_failure)
    (args.output / "actual-failure.json").write_text(json.dumps(failure_record, indent=2, allow_nan=False)+"\n")
    raise
