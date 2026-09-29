#!/usr/bin/env python3
"""M03: 2D Euler Sod slice, exact Riemann cell averages, two authored EOS.

Run the installed Dim=2 package with PYTHONPATH unset. The physical solution is
one-dimensional and invariant in y on genuine N x N meshes; this does not
qualify a Dim=1 build. POPS_API040_M03_CASE selects ideal or stiffened; run both.
This script authors fluxes/EOS in Python and never selects a built-in Euler PDE.
"""
# ruff: noqa: E402
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np
import pops

package = Path(pops.__file__).resolve()
if "PYTHONPATH" in os.environ or not package.is_relative_to(Path(sys.prefix).resolve()):
    raise RuntimeError("M03 reception requires installed PoPS with PYTHONPATH unset")
pops.set_threads(int(os.environ.get("POPS_THREADS", "1")))

from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import SSPRK2
from pops.math import ddt, div, sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import ConstParam
from pops.output import ConsumerGraph, NPZ, ParallelMode, ScientificOutput
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import AdaptiveCFL, every

from api040_m03_riemann_oracle import Gas, Primitive, cell_averages, conservative
from api040_m03_boundary_oracle import boundary_bands, reopen_uniform_state, ssprk2_boundary_increment
from api040_receipts import receipt_json


# Predeclared before any compile/run. N05 in the 0.4.0 reference exercises an
# entropy wave at gamma=1.4, p_inf=.3 in Dim=2; here that EOS gets a shock-tube
# test. The other N05 stiffened choice (1.6,2) sends its right shock beyond
# x=.5 by t=.15, invalidating the prescribed finite-domain far-field balance.
RESOLUTIONS = (100, 200, 400)
T_END = .15
CFL = .25
LEFT = Primitive(1., 0., 1.)
RIGHT = Primitive(.125, 0., .1)
GASES = {"ideal": Gas(1.4, 0.), "stiffened": Gas(1.4, .3)}
CRITERIA = {
    "density_l1_max": {100: .09, 200: .07, 400: .055},
    "density_l1_strictly_decreasing": True,
    "mass_defect_max": 2.e-10,
    "energy_defect_max": 2.e-10,
    "momentum_balance_defect_max": 3.e-9,
    "transverse_momentum_max": 2.e-11,
    "y_invariance_max": 2.e-11,
    "initial_max_error": 1.e-12,
    "time_error_max": 1.e-12,
    "minimum_density": 0.,
    "minimum_internal_energy_density": 0.,
    "minimum_pressure": 0.,
}

label = os.environ.get("POPS_API040_M03_CASE", "ideal")
if label not in GASES:
    raise ValueError("POPS_API040_M03_CASE must be ideal or stiffened")
gas = GASES[label]

# Physical equation: U_t + div F(U) = 0 in two dimensions, with the
# stiffened-gas EOS p=(gamma-1)(E-|m|^2/(2*rho))-gamma*p_inf.
frame = Rectangle("shock_tube", lower=(-.5, 0.), upper=(.5, 1.)).frame(Cartesian2D())
x_axis, y_axis = frame.axes
model = pops.Model("authored_euler_" + label, frame=frame)
component_names = tuple(os.environ.get(
    "POPS_API040_M03_COMPONENT_NAMES",
    "mass,longitudinal_impulse,lateral_impulse,energy").split(","))
parameter_names = tuple(os.environ.get(
    "POPS_API040_M03_PARAMETER_NAMES",
    "adiabatic_index,background_pressure").split(","))
if len(component_names) != 4 or len(set(component_names)) != 4 or len(parameter_names) != 2 \
        or len(set(parameter_names)) != 2:
    raise ValueError("M03 requires four unique state and two unique EOS parameter names")
state_name = os.environ.get("POPS_API040_M03_STATE_NAME", "fluid_conserved_" + label)
block_name = os.environ.get("POPS_API040_M03_BLOCK_NAME", "shock_tube_fluid")
U = model.state(state_name, components=component_names,
                representation=Conservative(), space=CellState(frame=frame))
gamma_handle = model.param(ConstParam(parameter_names[0], gas.gamma))
offset_handle = model.param(ConstParam(parameter_names[1], gas.p_inf))
gamma, pressure_offset = model.value(gamma_handle), model.value(offset_handle)
rho, mx, my, E = U
vx, vy = mx / rho, my / rho
kinetic = (mx * mx + my * my) / (2 * rho)
pressure = (gamma - 1) * (E - kinetic) - gamma * pressure_offset
sound = sqrt(gamma * (pressure + pressure_offset) / rho)
F = model.flux("authored_physical_flux", frame=frame, state=U,
               components={
                   x_axis: (mx, mx * vx + pressure, my * vx, (E + pressure) * vx),
                   y_axis: (my, mx * vy, my * vy + pressure, (E + pressure) * vy),
               },
               waves={
                   x_axis: (vx - sound, vx, vx, vx + sound),
                   y_axis: (vy - sound, vy, vy, vy + sound),
               })
rate = model.rate("authored_balance", equation=ddt(U) == -div(F))

numerics = DiscretizationPlan()
numerics.rates.add(rate, FiniteVolume(
    flux=F, variables=variables.Conservative(U),
    reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
case = pops.Case("api040_M03_" + label)
block = case.block(block_name, model)
numerics.boundaries.add(TransportBoundarySet({
    frame.boundaries.x_min: Outflow(state=block[U]),
    frame.boundaries.x_max: Outflow(state=block[U]),
}, periodic=PeriodicAxes((y_axis,))))
case.numerics(numerics, block=block)
program = SSPRK2(block[U], rate=rate)
program.step_strategy(AdaptiveCFL(cfl=CFL))
case.program(program)
trajectory_format = NPZ(mode=ParallelMode.ROOT, series=True)
case.consumers(ConsumerGraph.from_consumers((ScientificOutput(
    format=trajectory_format, schedule=every(1, clock=program.clock),
    fields=(block[U],), target="accepted/euler"),)))
validated = pops.validate(case)
if os.environ.get("POPS_API040_M03_AUTHORING_ONLY") == "1":
    probe_grid = CartesianGrid(frame=frame, cells=(8, 8), periodic=PeriodicAxes((y_axis,)))
    pops.resolve(validated, layout=Uniform(probe_grid))
    print("M03 public authoring validated and resolved:", label,
          state_name, component_names, parameter_names)
    sys.exit(0)

destination = Path(os.environ.get("POPS_API040_OUTPUT", "outputs/api040_m03")) / label
records = []
for n in RESOLUTIONS:
    initial_line = cell_averages(n, 0., LEFT, RIGHT, gas)
    exact_line = cell_averages(n, T_END, LEFT, RIGHT, gas)
    # Native arrays are (component, y, x): the last index is frame.x.
    initial = np.ascontiguousarray(np.broadcast_to(initial_line[:, None, :], (4, n, n)))
    exact = np.ascontiguousarray(np.broadcast_to(exact_line[:, None, :], (4, n, n)))
    if not np.all(np.isfinite(exact)):
        raise RuntimeError("exact Riemann cell averages are nonfinite")
    # For even N the diaphragm is an exact face. The check is independent of
    # PoPS projection and catches accidental center-sampling or index shifts.
    expected_left, expected_right = conservative(LEFT, gas), conservative(RIGHT, gas)
    if not (np.allclose(initial_line[:, :n // 2], expected_left[:, None], rtol=0., atol=1.e-14)
            and np.allclose(initial_line[:, n // 2:], expected_right[:, None], rtol=0., atol=1.e-14)):
        raise RuntimeError("initial states are not the prescribed conservative cell averages")
    grid = CartesianGrid(frame=frame, cells=(n, n), periodic=PeriodicAxes((y_axis,)))
    compiled_at = time.perf_counter()
    artifact = pops.compile(pops.resolve(validated, layout=Uniform(grid)))
    compile_seconds = time.perf_counter() - compiled_at
    if artifact.resolved_dimension != 2:
        raise RuntimeError("M03 requires a true Dim=2 artifact")
    context = pops.ExecutionContext.mpi_world(artifact)
    world = context.communicator.handle
    simulation = pops.bind(artifact, initial_state={block_name: initial},
                           resources={"execution_context": context})
    bound_initial = np.asarray(simulation.state_global(block_name))
    initial_failure = b""
    if world.rank == 0:
        if bound_initial.size != initial.size:
            initial_failure = b"M03 initial global gather has wrong size"
        else:
            bound_initial = bound_initial.reshape(initial.shape).copy()
            initial_max_error = float(np.max(np.abs(bound_initial - initial)))
            if not np.isfinite(initial_max_error) or initial_max_error > CRITERIA["initial_max_error"]:
                initial_failure = b"M03 bound state differs from exact initial averages"
    initial_failure = world.broadcast_bytes(initial_failure, root=0)
    if initial_failure:
        raise RuntimeError(initial_failure.decode())
    run_at = time.perf_counter()
    trajectory_root = destination / ("trajectory_%d" % n)
    report = pops.run(simulation, t_end=T_END, max_steps=100_000,
                      output_dir=str(trajectory_root))
    run_seconds = time.perf_counter() - run_at
    gathered = np.asarray(simulation.state_global(block_name))
    publication_failure = b""
    if world.rank == 0:
        try:
            if gathered.size != initial.size:
                raise RuntimeError("M03 final global gather has wrong size")
            actual = gathered.reshape(initial.shape).copy()
            destination.mkdir(parents=True, exist_ok=True)
            # Conservation means change equals the accepted boundary exchange.
            # The former far-field-only check failed for the stiffened EOS when
            # numerical tails reached Outflow. Keep that failure; do not tune
            # its EOS, boundaries or tolerance. Reconstruct the actual SSPRK2
            # boundary quadrature independently from authenticated snapshots.
            series_files = tuple(trajectory_root.rglob("*.npz.series"))
            if len(series_files) != 1:
                raise RuntimeError("M03 requires one complete accepted-state NPZ series")
            series = trajectory_format.reopen_series(series_files[0])
            if len(series.samples) != report.accepted_steps:
                raise RuntimeError("M03 trajectory does not contain every accepted step")
            previous = bound_initial.copy()
            previous_time, previous_step = 0., 0
            bands, durations, totals, archive_members = [], [], [np.mean(previous, axis=(1, 2))], []
            for sample in series.samples:
                if sample.macro_step != previous_step + 1 or not sample.time > previous_time:
                    raise RuntimeError("M03 trajectory has a missing or unordered accepted step")
                bands.append(boundary_bands(previous))
                durations.append(sample.time - previous_time)
                previous = reopen_uniform_state(sample.reopen(), initial.shape)
                totals.append(np.mean(previous, axis=(1, 2)))
                archive_members.append({"path": str(sample.path.relative_to(destination)),
                                        "sha256": hashlib.sha256(sample.path.read_bytes()).hexdigest()})
                previous_time, previous_step = sample.time, sample.macro_step
            if not np.array_equal(previous, actual):
                raise RuntimeError("M03 final archived state differs from the native final state")
            budget_path = destination / ("boundary_budget_%d.npz" % n)
            np.savez_compressed(budget_path, bands=np.asarray(bands), dt=np.asarray(durations),
                                native_totals=np.asarray(totals), dx=1 / n, dy=1 / n,
                                gamma=gas.gamma, p_inf=gas.p_inf)
            with np.load(budget_path) as saved_budget:
                increments = np.asarray([ssprk2_boundary_increment(
                    band, dt, float(saved_budget["dx"]), float(saved_budget["dy"]),
                    float(saved_budget["gamma"]), float(saved_budget["p_inf"]))
                    for band, dt in zip(saved_budget["bands"], saved_budget["dt"], strict=True)])
                boundary_increment = np.sum(increments, axis=0)
                step_balance_max = np.max(np.abs(np.diff(saved_budget["native_totals"], axis=0)
                                                - increments), axis=0)
            state_path = destination / ("state_%d.npz" % n)
            np.savez_compressed(state_path, initial=bound_initial, final=actual,
                                exact=exact, time=simulation.time(), cells=n,
                                gamma=gas.gamma, p_inf=gas.p_inf,
                                incoming_boundary_increment=boundary_increment)
            # All diagnostics below read the persisted full-domain state.
            with np.load(state_path) as saved:
                actual = saved["final"]
                initial_saved = saved["initial"]
                exact_saved = saved["exact"]
                saved_time = float(saved["time"])
                density_l1 = float(np.mean(np.abs(actual[0] - exact_saved[0])))
                rho_min = float(np.min(actual[0]))
                kinetic_density = (actual[1]**2 + actual[2]**2) / (2 * actual[0])
                internal_density = actual[3] - kinetic_density
                pressure_actual = (gas.gamma - 1) * internal_density - gas.gamma * gas.p_inf
                e_min = float(np.min(internal_density))
                pressure_min = float(np.min(pressure_actual))
                incoming = saved["incoming_boundary_increment"]
                mass_defect = float(abs(np.mean(actual[0]) - np.mean(initial_saved[0]) - incoming[0]))
                energy_defect = float(abs(np.mean(actual[3]) - np.mean(initial_saved[3]) - incoming[3]))
                expected_momentum = float(np.mean(initial_saved[1]) + incoming[1])
                momentum_defect = float(abs(np.mean(actual[1]) - expected_momentum))
                transverse_momentum = float(np.max(np.abs(actual[2])))
                y_invariance = float(np.max(np.abs(actual - actual[:, :1, :])))
            record = {"eos": label, "gamma": gas.gamma, "p_inf": gas.p_inf,
                      "cells": [n, n], "time": saved_time,
                      "accepted_steps": report.accepted_steps,
                      "density_l1": density_l1, "rho_min": rho_min,
                      "internal_energy_density_min": e_min, "pressure_min": pressure_min,
                      "mass_defect": mass_defect, "energy_defect": energy_defect,
                      "momentum_balance_defect": momentum_defect,
                      "transverse_momentum_max": transverse_momentum,
                      "y_invariance_max": y_invariance,
                      "initial_max_error": initial_max_error,
                      "incoming_boundary_increment": incoming.tolist(),
                      "farfield_flux_deviation": (incoming - np.asarray(
                          (0., T_END * (LEFT.p - RIGHT.p), 0., 0.))).tolist(),
                      "maximum_per_step_balance_defect": step_balance_max.tolist(),
                      "boundary_budget": budget_path.name,
                      "boundary_budget_sha256": hashlib.sha256(budget_path.read_bytes()).hexdigest(),
                      "trajectory_series": str(series_files[0].relative_to(destination)),
                      "trajectory_members": archive_members,
                      "compile_seconds": compile_seconds, "run_seconds": run_seconds,
                      "run_report": report.to_data(), "mpi_ranks": world.size,
                      "execution_context": context.to_data(),
                      "runtime_backend": context.backend.to_data(),
                      "saved_state": state_path.name,
                      "saved_state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest()}
            records.append(record)
            checks = (
                np.isfinite(actual).all() and abs(saved_time - T_END) <= CRITERIA["time_error_max"]
                and density_l1 <= CRITERIA["density_l1_max"][n]
                and rho_min > CRITERIA["minimum_density"]
                and e_min > CRITERIA["minimum_internal_energy_density"]
                and pressure_min > CRITERIA["minimum_pressure"]
                and mass_defect <= CRITERIA["mass_defect_max"]
                and energy_defect <= CRITERIA["energy_defect_max"]
                and momentum_defect <= CRITERIA["momentum_balance_defect_max"]
                and step_balance_max[0] <= CRITERIA["mass_defect_max"]
                and step_balance_max[3] <= CRITERIA["energy_defect_max"]
                and step_balance_max[1] <= CRITERIA["momentum_balance_defect_max"]
                and transverse_momentum <= CRITERIA["transverse_momentum_max"]
                and y_invariance <= CRITERIA["y_invariance_max"])
            if not checks:
                raise AssertionError("M03 predeclared criterion failed: " + receipt_json(record))
        except Exception as error:
            publication_failure = (type(error).__name__ + ": " + str(error)).encode()
    publication_failure = world.broadcast_bytes(publication_failure, root=0)
    if publication_failure:
        raise RuntimeError(publication_failure.decode())

final_failure = b""
if world.rank == 0:
    try:
        errors = [item["density_l1"] for item in records]
        accepted = (len(records) == len(RESOLUTIONS) and
                    (not CRITERIA["density_l1_strictly_decreasing"] or all(
                        errors[i + 1] < errors[i] for i in range(len(errors) - 1))))
        from pops import _pops  # Provenance only; evolution uses public APIs.
        native = Path(_pops.__file__).resolve()
        receipt = {"schema_version": 2, "case": "M03", "eos": label,
                   "balance_oracle": "SSPRK2 boundary quadrature from every saved native accepted state",
                   "balance_revision": "v2: incoming numerical boundary flux, unchanged tolerances",
                   "status": "passed" if accepted else "failed", "criteria": CRITERIA,
                   "scope": "Dim=2 NxN, 1D solution invariant in y; not Dim=1",
                   "method": "authored Euler/EOS; first-order Rusanov; SSPRK2",
                   "cfl": CFL, "t_end": T_END, "resolutions": RESOLUTIONS,
                   "component_names": component_names, "parameter_names": parameter_names,
                   "state_name": state_name, "block_name": block_name,
                   "package_file": str(package), "package_version": pops.__version__,
                   "native_file": str(native),
                   "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
                   "abi_key": _pops.abi_key(), "prefix": sys.prefix,
                   "threads_requested": os.environ.get("POPS_THREADS", "1"),
                   "records": records}
        (destination / "receipt.json").write_text(receipt_json(receipt) + "\n")
        print(receipt_json(receipt))
        if not accepted:
            final_failure = b"M03 density error did not decrease strictly with resolution"
    except Exception as error:
        final_failure = (type(error).__name__ + ": " + str(error)).encode()
final_failure = world.broadcast_bytes(final_failure, root=0)
if final_failure:
    raise RuntimeError(final_failure.decode())
