#!/usr/bin/env python3
"""M08 variant: two heterogeneous transported scalars sharing a live Poisson field.

Run with installed Dim=2 PoPS and PYTHONPATH unset.  The PDE is genuinely 2D:
  -Delta phi=q,  u=(d_y phi,-d_x phi),
  q_t+div(q u)=0,  c_t+div(c u)=0 on periodic [0,2pi]^2.
The charge q and tracer c occupy distinct Model/Case blocks.  Both RHS calls
consume gradients published from the same exact stage solve.  A same-time
charge perturbation is an additional diagnostic, not part of the trajectory.
"""
# ruff: noqa: E402
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np
import pops

if os.environ.get("POPS_API040_M08_AUTHORING_ONLY") != "1":
    package = Path(pops.__file__).resolve()
    if "PYTHONPATH" in os.environ or not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("M08 reception requires installed PoPS with PYTHONPATH unset")
else:
    package = Path(pops.__file__).resolve()
pops.set_threads(int(os.environ.get("POPS_THREADS", "1")))

from pops.domain import Rectangle
from pops.fields import FieldBoundary, FieldDiscretization, FieldProblem, SharedMeanGauge, bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div, grad, laplacian
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.solvers import CG
from pops.time import FailRun, FixedDt

from api040_m08_oracle import cell_means, field_metrics, ssprk2_reference
if os.environ.get("POPS_API040_M08_AUTHORING_ONLY") != "1":
    from api040_receipts import receipt_json


# The registry fixes the domain, resolutions, SSPRK2 and field residual; it
# leaves amplitudes and final time open.  These variant values are fixed here
# before native execution, and the receipt identifies the variant explicitly.
RESOLUTIONS = (32, 64)
DT = .02
STEPS = 2
T_END = DT * STEPS
PROBE_FACTOR = 1.1
CRITERIA = {
    "field_residual_max": 1.e-10,
    "fft_potential_max_error": 1.e-9,
    "gradient_max_error": 2.e-10,
    "velocity_divergence_max": 1.e-10,
    "probe_linear_response_max": 2.e-9,
    "probe_potential_separation_min": 1.e-3,
    "stage_potential_separation_min": 1.e-8,
    "reference_fv_max_error": 3.e-7,
    "mean_defect_max": 2.e-10,
    "initial_max_error": 1.e-12,
    "time_error_max": 1.e-12,
}

frame = Rectangle("m08_periodic_square", lower=(0., 0.),
                  upper=(2 * np.pi, 2 * np.pi)).frame(Cartesian2D())
x_axis, y_axis = frame.axes
charge_model = pops.Model("m08_charge_physics", frame=frame)
q_state = charge_model.state("charge_state", components=("charge",))
(q,) = q_state
phi = charge_model.field("potential")
q_gradient = charge_model.vector("charge_field_gradient", frame=frame,
    components={x_axis: grad(phi).x, y_axis: grad(phi).y})
q_flux = charge_model.flux("charge_guiding_transport", frame=frame, state=q_state,
    components={x_axis: (q_gradient.y * q,), y_axis: (-q_gradient.x * q,)},
    waves={x_axis: (q_gradient.y,), y_axis: (-q_gradient.x,)})
q_rate = charge_model.rate("charge_balance", equation=ddt(q_state) == -div(q_flux))

tracer_model = pops.Model("m08_tracer_physics", frame=frame)
c_state = tracer_model.state("tracer_state", components=("concentration",))
(c,) = c_state
tracer_phi = tracer_model.field("potential")
c_gradient = tracer_model.vector("tracer_field_gradient", frame=frame,
    components={x_axis: grad(tracer_phi).x, y_axis: grad(tracer_phi).y})
c_flux = tracer_model.flux("tracer_guiding_transport", frame=frame, state=c_state,
    components={x_axis: (c_gradient.y * c,), y_axis: (-c_gradient.x * c,)},
    waves={x_axis: (c_gradient.y,), y_axis: (-c_gradient.x,)})
c_rate = tracer_model.rate("tracer_balance", equation=ddt(c_state) == -div(c_flux))

case = pops.Case("api040_M08_two_state_guiding_center_variant")
charge_block = case.block("charge", charge_model)
tracer_block = case.block("tracer", tracer_model)
for block, state, rate, flux in (
    (charge_block, q_state, q_rate, q_flux),
    (tracer_block, c_state, c_rate, c_flux),
):
    plan = DiscretizationPlan()
    plan.rates.add(rate, FiniteVolume(flux=flux,
        variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case.numerics(plan, block=block)

problem = FieldProblem("m08_charge_Poisson", unknowns=(phi,),
    equations=(-laplacian(phi) == q,),
    boundaries=(FieldBoundary(phi,
        bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())),),
    gauge=SharedMeanGauge((phi,)))
field = case.field(problem, FieldDiscretization(
    method=CellCenteredSecondOrder(), boundaries=(),
    solver=CG(max_iter=4000, rel_tol=1.e-12, abs_tol=1.e-13)))

program = pops.Program("m08_live_field_ssprk2")
charge_time = program.state(charge_block[q_state])
tracer_time = program.state(tracer_block[c_state])
q0, c0 = charge_time.n, tracer_time.n
charge_module, tracer_module = charge_model.module, tracer_model.module
charge_carrier = charge_block[charge_module.field_handle(
    charge_module.field_spaces()["fields"])]
tracer_carrier = tracer_block[tracer_module.field_handle(
    tracer_module.field_spaces()["fields"])]


def published_stage(label, stage_q, stage_c, point):
    # One solve produces exactly one common field; the tracer state is an
    # explicit supplemental consumer at this very stage.
    observation = field.observe(program.solve(
        field, values={charge_block[q_state]: stage_q}, at=point)
        .consume(action=FailRun()))
    potential = observation[field[phi]]
    gradient = observation.gradient(field[phi], dimension=2)
    bindings = {}
    for carrier in (charge_carrier, tracer_carrier):
        bindings[(carrier, "potential_grad_x")] = (gradient, 0)
        bindings[(carrier, "potential_grad_y")] = (gradient, 1)
    context = observation.publish(bindings, states={tracer_block[c_state]: stage_c})
    program.store_history(label + "_q", stage_q, depth=1)
    program.store_history(label + "_phi", potential, depth=1)
    program.store_history(label + "_gradient", gradient, depth=1)
    return context, potential


initial_point = program.stage("m08_initial_field", c=0)
initial_fields, initial_phi = published_stage("stage0", q0, c0, initial_point)

# A second state at the SAME clock date forces another Poisson query.  Its
# result is diagnostic only; it cannot contaminate the physical SSPRK2 state.
probe_point = program.stage("m08_changed_charge_same_date", c=0)
probe_q = program.value("probe_charge", PROBE_FACTOR * q0, at=probe_point)
probe_observation = field.observe(program.solve(
    field, values={charge_block[q_state]: probe_q}, at=probe_point)
    .consume(action=FailRun()))
program.store_history("probe_q", probe_q, depth=1)
program.store_history("probe_phi", probe_observation[field[phi]], depth=1)

rq0 = q_rate(q0, initial_fields)
rc0 = c_rate(c0, initial_fields)
predictor_point = program.stage("m08_ssprk2_predictor", c=1)
q1 = program.value("predicted_charge", q0 + program.dt * rq0, at=predictor_point)
c1 = program.value("predicted_tracer", c0 + program.dt * rc0, at=predictor_point)
predictor_fields, predictor_phi = published_stage("stage1", q1, c1, predictor_point)
rq1 = q_rate(q1, predictor_fields)
rc1 = c_rate(c1, predictor_fields)
program.commit(charge_time.next, program.value("accepted_charge",
    .5 * q0 + .5 * (q1 + program.dt * rq1), at=charge_time.next.point))
program.commit(tracer_time.next, program.value("accepted_tracer",
    .5 * c0 + .5 * (c1 + program.dt * rc1), at=tracer_time.next.point))
program.step_strategy(FixedDt(DT))
case.program(program)
validated = pops.validate(case)

destination = Path(os.environ.get("POPS_API040_OUTPUT", "outputs/api040_m08"))
records = []
for n in RESOLUTIONS:
    layout = Uniform(CartesianGrid(frame=frame, cells=(n, n),
        periodic=PeriodicAxes(frame.axes)))
    resolved = pops.resolve(validated, layout=layout)
    if os.environ.get("POPS_API040_M08_AUTHORING_ONLY") == "1":
        print("M08 source authoring validated and resolved:", n,
              "blocks=charge,tracer; one shared field; three stage queries")
        continue

    q_initial, c_initial = cell_means(n)
    inputs = {"charge": q_initial[None, :, :], "tracer": c_initial[None, :, :]}
    q_reference, c_reference = ssprk2_reference(q_initial, c_initial, dt=DT, steps=STEPS)
    compiled_at = time.perf_counter()
    artifact = pops.compile(resolved)
    compile_seconds = time.perf_counter() - compiled_at
    if artifact.resolved_dimension != 2:
        raise RuntimeError("M08 requires actual native Dim=2")
    context = pops.ExecutionContext.mpi_world(artifact)
    world = context.communicator.handle
    runtime = pops.bind(artifact, initial_state=inputs,
                        resources={"execution_context": context})
    gathered_initial = {name: np.asarray(runtime.state_global(name))
                        for name in inputs}
    initial_failure = b""
    if world.rank == 0:
        for name, expected in inputs.items():
            if gathered_initial[name].size != expected.size or not np.isfinite(gathered_initial[name]).all():
                initial_failure = b"M08 bound initial state is incomplete or non-finite"
                break
            gathered_initial[name] = gathered_initial[name].reshape(expected.shape).copy()
            if np.max(np.abs(gathered_initial[name] - expected)) > CRITERIA["initial_max_error"]:
                initial_failure = b"M08 bound initial state differs from exact cell averages"
                break
    initial_failure = world.broadcast_bytes(initial_failure, root=0)
    if initial_failure:
        raise RuntimeError(initial_failure.decode())
    run_at = time.perf_counter()
    report = pops.run(runtime, t_end=T_END, max_steps=STEPS)
    run_seconds = time.perf_counter() - run_at
    final = {name: np.asarray(runtime.state_global(name)) for name in inputs}
    history = {name: np.asarray(runtime.history_global(name, 0))
               for name in ("stage0_q", "stage0_phi", "stage0_gradient",
                            "probe_q", "probe_phi", "stage1_q", "stage1_phi",
                            "stage1_gradient")}
    publication_failure = b""
    if world.rank == 0:
        try:
            if any(value.size != n*n for value in final.values()):
                raise RuntimeError("M08 final global state has wrong size")
            for name in final:
                final[name] = final[name].reshape((1, n, n)).copy()
            for name in ("stage0_q", "stage0_phi", "probe_q", "probe_phi",
                         "stage1_q", "stage1_phi"):
                if history[name].size != n*n:
                    raise RuntimeError("M08 field history has wrong scalar shape: " + name)
                history[name] = history[name].reshape(n, n).copy()
            for name in ("stage0_gradient", "stage1_gradient"):
                if history[name].size != 2*n*n:
                    raise RuntimeError("M08 field history has wrong vector shape: " + name)
                history[name] = history[name].reshape(2, n, n).copy()
            destination.mkdir(parents=True, exist_ok=True)
            state_path = destination / ("state_%d.npz" % n)
            np.savez_compressed(state_path,
                initial_q=gathered_initial["charge"][0],
                initial_c=gathered_initial["tracer"][0],
                final_q=final["charge"][0], final_c=final["tracer"][0],
                reference_q=q_reference, reference_c=c_reference,
                time=runtime.time(), cells=n, dt=DT, **history)
            with np.load(state_path) as saved:
                metrics = {label: field_metrics(saved[label + "_q"],
                    saved[label + "_phi"], saved[label + "_gradient"])
                    for label in ("stage0", "stage1")}
                probe_residual = float(np.max(np.abs(
                    # The probe is a 1.1-scaled RHS at unchanged physical time.
                    saved["probe_phi"] - PROBE_FACTOR * saved["stage0_phi"])))
                probe_separation = float(np.max(np.abs(
                    saved["probe_phi"] - saved["stage0_phi"])))
                stage_separation = float(np.max(np.abs(
                    saved["stage1_phi"] - saved["stage0_phi"])))
                fv_error = float(max(np.max(np.abs(saved["final_q"] - saved["reference_q"])),
                                     np.max(np.abs(saved["final_c"] - saved["reference_c"]))))
                mass_defect = float(max(abs(np.mean(saved["final_q"]) - np.mean(saved["initial_q"])),
                                        abs(np.mean(saved["final_c"]) - np.mean(saved["initial_c"]))))
                saved_time = float(saved["time"])
                finite = all(np.isfinite(saved[key]).all() for key in saved.files)
            accepted = (finite and report.accepted_steps == STEPS
                and abs(saved_time - T_END) <= CRITERIA["time_error_max"]
                and max(m["poisson_residual_max"] for m in metrics.values()) <=
                    CRITERIA["field_residual_max"]
                and max(m["fft_potential_max_error"] for m in metrics.values()) <=
                    CRITERIA["fft_potential_max_error"]
                and max(m["gradient_max_error"] for m in metrics.values()) <=
                    CRITERIA["gradient_max_error"]
                and max(m["velocity_divergence_max"] for m in metrics.values()) <=
                    CRITERIA["velocity_divergence_max"]
                and probe_residual <= CRITERIA["probe_linear_response_max"]
                and probe_separation >= CRITERIA["probe_potential_separation_min"]
                and stage_separation >= CRITERIA["stage_potential_separation_min"]
                and fv_error <= CRITERIA["reference_fv_max_error"]
                and mass_defect <= CRITERIA["mean_defect_max"])
            record = {"cells": [n, n], "time": saved_time, "accepted_steps": report.accepted_steps,
                "field_metrics": metrics, "same_time_probe_error": probe_residual,
                "same_time_probe_separation": probe_separation,
                "physical_stage_field_separation": stage_separation,
                "independent_fv_max_error": fv_error, "mass_defect": mass_defect,
                "compile_seconds": compile_seconds, "run_seconds": run_seconds,
                "run_report": report.to_data(), "execution_context": context.to_data(),
                "runtime_backend": context.backend.to_data(), "mpi_ranks": world.size,
                "field_diagnostics": runtime.program_report().diagnostics,
                "saved_state": state_path.name,
                "saved_state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
                "accepted": accepted}
            records.append(record)
            if not accepted:
                raise AssertionError("M08 predeclared criterion failed: " + receipt_json(record))
        except Exception as error:
            publication_failure = (type(error).__name__ + ": " + str(error)).encode()
    publication_failure = world.broadcast_bytes(publication_failure, root=0)
    if publication_failure:
        raise RuntimeError(publication_failure.decode())

if os.environ.get("POPS_API040_M08_AUTHORING_ONLY") != "1":
    final_failure = b""
    if world.rank == 0:
        try:
            from pops import _pops
            native = Path(_pops.__file__).resolve()
            receipt = {"schema_version": 1, "case": "M08",
                "status": "passed" if len(records) == len(RESOLUTIONS) else "failed",
                "scope": "new 2D two-block guiding-center variant; reference M08 not demonstrated",
                "domain": [[0., 2*np.pi], [0., 2*np.pi]],
                "equations": ["-Delta(phi)=q", "u=(d_y phi,-d_x phi)",
                              "q_t+div(q*u)=0", "c_t+div(c*u)=0"],
                "spatial_method": "cell-centred second-order Poisson gradient; first-order Rusanov FV",
                "temporal_method": "SSPRK2", "dt": DT, "t_end": T_END,
                "probe_factor": PROBE_FACTOR, "resolutions": RESOLUTIONS,
                "criteria": CRITERIA, "records": records,
                "package_file": str(package), "package_version": pops.__version__,
                "native_file": str(native),
                "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
                "abi_key": _pops.abi_key(), "prefix": sys.prefix,
                "threads_requested": os.environ.get("POPS_THREADS", "1")}
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "receipt.json").write_text(receipt_json(receipt) + "\n")
            print(receipt_json(receipt))
        except Exception as error:
            final_failure = (type(error).__name__ + ": " + str(error)).encode()
    final_failure = world.broadcast_bytes(final_failure, root=0)
    if final_failure:
        raise RuntimeError(final_failure.decode())
