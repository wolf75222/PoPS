#!/usr/bin/env python3
"""M02: authored Godunov face, Burgers shock and rarefaction cell averages.

Installed native Dim=2, full N x N grids, x Outflow and periodic y. This is a
one-dimensional physical solution in a genuine 2D specialization, not a Dim=1
qualification. Select shock or rarefaction using POPS_API040_M02_CASE.
"""
# ruff: noqa: E402
import hashlib
import inspect
import os
from pathlib import Path
import sys
import time

import numpy as np
import pops

package = Path(pops.__file__).resolve()
if "PYTHONPATH" in os.environ or not package.is_relative_to(Path(sys.prefix).resolve()):
    raise RuntimeError("M02 reception requires installed PoPS with PYTHONPATH unset")
pops.set_threads(int(os.environ.get("POPS_THREADS", "1")))

from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import ddt, div, where
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import AdaptiveCFL

from api040_m02_burgers_oracle import entropy_cell_averages, weak_shock_speed
from api040_receipts import receipt_json


# Acceptance is fixed before compilation or measurement. No observed errors
# are used to tune these bounds, and no failed state is clipped into [0,1].
RESOLUTIONS = (100, 200, 400)
T_END = .2
CFL = .4
FIRST_STEP_DX_RATIO = .01
CASES = {"shock": (1., 0.), "rarefaction": (0., 1.)}
CRITERIA = {
    "integrated_l1_max": {100: .08, 200: .06, 400: .04},
    "integrated_l1_strictly_decreasing": True,
    "boundary_mass_balance_tolerance": 1.e-11,
    "initial_tolerance": 1.e-14,
    "first_step_tolerance": 2.e-13,
    "time_tolerance": 1.e-13,
    "bound_tolerance": 2.e-13,
    "y_invariance_tolerance": 1.e-12,
    "monotonicity_tolerance": 1.e-12,
    "shock_position_error_in_cells": 1.5,
}
label = os.environ.get("POPS_API040_M02_CASE", "shock")
if label not in CASES:
    raise ValueError("POPS_API040_M02_CASE must be shock or rarefaction")
left_state, right_state = CASES[label]

# The conserved quantity remains u, not u^2/2. Conserving the latter after a
# smooth coordinate rewrite would move the 1->0 shock at 2/3 instead of 1/2.
frame = Rectangle("burgers_strip", lower=(-1., 0.), upper=(1., 1.)).frame(Cartesian2D())
x_axis, y_axis = frame.axes
model = pops.Model("authored_burgers", frame=frame)
U = model.state("conserved_u", components=("u",), representation=Conservative(),
                space=CellState(frame=frame))
(u,) = U
F = model.flux("burgers_physical_flux", frame=frame, state=U,
               components={x_axis: (.5 * u * u,), y_axis: (0 * u,)},
               waves={x_axis: (u,), y_axis: (0.,)})
rate = model.rate("conservative_balance", equation=ddt(U) == -div(F))


def godunov_face(left, right, flux_left, flux_right, speed):
    """Complete scalar Burgers entropy flux in the canonical positive normal.

    Use the provided physical fluxes so the zero-flux transverse axis remains
    zero. The body is ordinary symbolic Python, with lazy common-IR where.
    """
    ul, ur = left[0], right[0]
    fl, fr = flux_left[0], flux_right[0]
    shock = where(ul + ur >= 0, lambda: fl, lambda: fr)
    rarefaction = where(ul >= 0, lambda: fl,
                        lambda: where(ur <= 0, lambda: fr, lambda: 0 * fl))
    return (where(ul > ur, lambda: shock, lambda: rarefaction),)


if not hasattr(riemann, "User") or not {"body", "state"}.issubset(
        inspect.signature(riemann.User).parameters):
    raise RuntimeError("M02 requires installed riemann.User(body=..., state=...); "
                       "the legacy C++ brick selector is insufficient")
godunov = riemann.User(body=godunov_face, state=U)
numerics = DiscretizationPlan()
numerics.rates.add(rate, FiniteVolume(
    flux=F, variables=variables.Conservative(U),
    reconstruction=reconstruction.FirstOrder(), riemann=godunov))
case = pops.Case("api040_M02_" + label)
fluid = case.block("burgers", model)
numerics.boundaries.add(TransportBoundarySet({
    frame.boundaries.x_min: Outflow(state=fluid[U]),
    frame.boundaries.x_max: Outflow(state=fluid[U]),
}, periodic=PeriodicAxes((y_axis,))))
case.numerics(numerics, block=fluid)
program = ForwardEuler(fluid[U], rate=rate)
program.step_strategy(AdaptiveCFL(cfl=CFL))
case.program(program)
validated = pops.validate(case)
if os.environ.get("POPS_API040_M02_AUTHORING_ONLY") == "1":
    probe_grid = CartesianGrid(frame=frame, cells=(8, 8), periodic=PeriodicAxes((y_axis,)))
    pops.resolve(validated, layout=Uniform(probe_grid))
    print(receipt_json({"case": "M02", "variant": label, "status": "authoring_only",
                        "native_status": "not_executed"}))
    sys.exit(0)

destination = Path(os.environ.get("POPS_API040_OUTPUT", "outputs/api040_m02")) / label
records = []
for n in RESOLUTIONS:
    dx = 2. / n
    initial_line = entropy_cell_averages(n, 0., left_state, right_state)
    exact_line = entropy_cell_averages(n, T_END, left_state, right_state)
    first_time = FIRST_STEP_DX_RATIO * dx
    first_exact_line = entropy_cell_averages(n, first_time, left_state, right_state)
    # Native component-major layout is (component,y,x), with x fastest.
    initial = np.ascontiguousarray(np.broadcast_to(initial_line, (1, n, n)))
    exact = np.ascontiguousarray(np.broadcast_to(exact_line, (1, n, n)))
    first_exact = np.ascontiguousarray(np.broadcast_to(first_exact_line, (1, n, n)))
    grid = CartesianGrid(frame=frame, cells=(n, n), periodic=PeriodicAxes((y_axis,)))
    started = time.perf_counter()
    artifact = pops.compile(pops.resolve(validated, layout=Uniform(grid)))
    compile_seconds = time.perf_counter() - started
    if artifact.resolved_dimension != 2:
        raise RuntimeError("M02 receipt requires an actual Dim=2 native artifact")
    context = pops.ExecutionContext.mpi_world(artifact)
    world = context.communicator.handle
    simulation = pops.bind(artifact, initial_state={"burgers": initial},
                           resources={"execution_context": context})
    gathered_initial = np.asarray(simulation.state_global("burgers"))
    initial_failure = b""
    if world.rank == 0:
        if gathered_initial.size != initial.size:
            initial_failure = b"M02 initial gather has wrong full-grid size"
        else:
            bound_initial = gathered_initial.reshape(initial.shape).copy()
            initial_error = float(np.max(np.abs(bound_initial - initial)))
            if not np.isfinite(initial_error) or initial_error > CRITERIA["initial_tolerance"]:
                initial_failure = b"M02 native initial state differs from conservative cell averages"
    initial_failure = world.broadcast_bytes(initial_failure, root=0)
    if initial_failure:
        raise RuntimeError(initial_failure.decode())

    # A short first Forward Euler step is exact in cell averages for these
    # face-aligned Riemann data. Rusanov gives a different first step, so a
    # silent substitution cannot pass merely by converging to Burgers later.
    started = time.perf_counter()
    first_report = pops.run(simulation, t_end=first_time, max_steps=1)
    gathered_first = np.asarray(simulation.state_global("burgers"))
    first_failure = b""
    if world.rank == 0:
        try:
            if gathered_first.size != initial.size:
                raise RuntimeError("M02 first-step gather has wrong full-grid size")
            destination.mkdir(parents=True, exist_ok=True)
            first_path = destination / ("first_step_%d.npz" % n)
            np.savez_compressed(first_path, initial=bound_initial,
                                final=gathered_first.reshape(initial.shape),
                                exact=first_exact, time=simulation.time(), cells=n)
            with np.load(first_path) as saved:
                first_error = float(np.max(np.abs(saved["final"] - saved["exact"])))
                first_saved_time = float(saved["time"])
            if (not np.isfinite(first_error) or first_error > CRITERIA["first_step_tolerance"]
                    or abs(first_saved_time - first_time) > CRITERIA["time_tolerance"]
                    or first_report.accepted_steps != 1):
                raise AssertionError("M02 authored Godunov first-step oracle failed")
        except Exception as error:
            first_failure = (type(error).__name__ + ": " + str(error)).encode()
            (destination / "receipt.json").write_text(receipt_json({
                "case": "M02", "variant": label, "status": "failed", "stage": "first_step",
                "cells": [n, n], "reason": first_failure.decode(), "criteria": CRITERIA}) + "\n")
    first_failure = world.broadcast_bytes(first_failure, root=0)
    if first_failure:
        raise RuntimeError(first_failure.decode())

    report = pops.run(simulation, t_end=T_END, max_steps=100_000)
    run_seconds = time.perf_counter() - started
    gathered = np.asarray(simulation.state_global("burgers"))
    publication_failure = b""
    if world.rank == 0:
        try:
            if gathered.size != initial.size:
                raise RuntimeError("M02 final gather has wrong full-grid size")
            state_path = destination / ("state_%d.npz" % n)
            np.savez_compressed(state_path, initial=bound_initial,
                                final=gathered.reshape(initial.shape), exact=exact,
                                time=simulation.time(), cells=n, left=left_state, right=right_state)
            with np.load(state_path) as saved:
                actual = saved["final"]
                if not np.isfinite(actual).all():
                    raise AssertionError("M02 saved state contains nonfinite values")
                saved_time = float(saved["time"])
                l1 = float(2 * np.mean(np.abs(actual - saved["exact"])))
                mass_change = float(2 * np.mean(actual - saved["initial"]))
                expected_mass_change = T_END * .5 * (left_state**2 - right_state**2)
                mass_defect = abs(mass_change - expected_mass_change)
                minimum, maximum = float(actual.min()), float(actual.max())
                y_invariance = float(np.max(np.abs(actual - actual[:, :1, :])))
                profile = np.mean(actual[0], axis=0)
                differences = np.diff(profile)
                monotonicity_defect = float(max(0., np.max(
                    differences if label == "shock" else -differences)))
            record = {"cells": [n, n], "time": saved_time, "integrated_l1": l1,
                      "mass_change": mass_change, "expected_mass_change": expected_mass_change,
                      "boundary_mass_balance_defect": mass_defect,
                      "minimum": minimum, "maximum": maximum,
                      "y_invariance": y_invariance, "monotonicity_defect": monotonicity_defect,
                      "initial_max_error": initial_error, "first_step_max_error": first_error,
                      "first_step_time": first_saved_time, "first_step_report": first_report.to_data(),
                      "accepted_steps": simulation.macro_step(), "run_report": report.to_data(),
                      "compile_seconds": compile_seconds, "run_seconds": run_seconds,
                      "mpi_ranks": world.size, "execution_context": context.to_data(),
                      "runtime_backend": context.backend.to_data(),
                      "saved_state": state_path.name,
                      "saved_state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
                      "first_step_state": first_path.name,
                      "first_step_sha256": hashlib.sha256(first_path.read_bytes()).hexdigest()}
            position_ok = True
            if label == "shock":
                crossings = np.flatnonzero((profile[:-1] >= .5) & (profile[1:] < .5))
                if len(crossings) != 1:
                    raise AssertionError("M02 shock must have one resolved half-height crossing")
                i = int(crossings[0])
                position = -1 + (i + .5) * dx + dx * (profile[i] - .5) / (profile[i] - profile[i + 1])
                exact_speed = weak_shock_speed(left_state, right_state)
                record.update(shock_position=float(position), shock_speed=float(position / saved_time),
                              exact_shock_speed=exact_speed,
                              shock_position_error=float(abs(position - exact_speed * T_END)))
                position_ok = record["shock_position_error"] <= CRITERIA["shock_position_error_in_cells"] * dx
            checks = (abs(saved_time - T_END) <= CRITERIA["time_tolerance"]
                      and abs(report.final_time - saved_time) <= CRITERIA["time_tolerance"]
                      and l1 <= CRITERIA["integrated_l1_max"][n]
                      and mass_defect <= CRITERIA["boundary_mass_balance_tolerance"]
                      and minimum >= -CRITERIA["bound_tolerance"]
                      and maximum <= 1 + CRITERIA["bound_tolerance"]
                      and y_invariance <= CRITERIA["y_invariance_tolerance"]
                      and monotonicity_defect <= CRITERIA["monotonicity_tolerance"] and position_ok)
            record["accepted"] = bool(checks)
            records.append(record)
            if not checks:
                raise AssertionError("M02 predeclared acceptance criterion failed")
        except Exception as error:
            publication_failure = (type(error).__name__ + ": " + str(error)).encode()
            (destination / "receipt.json").write_text(receipt_json({
                "case": "M02", "variant": label, "status": "failed", "stage": "saved_state",
                "reason": publication_failure.decode(), "criteria": CRITERIA, "records": records}) + "\n")
    publication_failure = world.broadcast_bytes(publication_failure, root=0)
    if publication_failure:
        raise RuntimeError(publication_failure.decode())

final_failure = b""
if world.rank == 0:
    try:
        errors = [row["integrated_l1"] for row in records]
        accepted = len(records) == len(RESOLUTIONS) and all(
            errors[i + 1] < errors[i] for i in range(len(errors) - 1))
        from pops import _pops  # Provenance only; evolution uses public APIs.
        native = Path(_pops.__file__).resolve()
        receipt = {"schema_version": 1, "case": "M02", "variant": label,
                   "status": "passed" if accepted else "failed", "criteria": CRITERIA,
                   "scope": "Dim=2 NxN; y-invariant Burgers; no Dim=1 qualification",
                   "equation": "u_t + (u*u/2)_x = 0", "conserved_quantity": "u",
                   "method": "authored riemann.User Godunov; FirstOrder; ForwardEuler",
                   "resolutions": RESOLUTIONS, "t_end": T_END, "cfl": CFL,
                   "first_step_dx_ratio": FIRST_STEP_DX_RATIO,
                   "package_file": str(package), "package_version": pops.__version__,
                   "native_file": str(native),
                   "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
                   "abi_key": _pops.abi_key(), "threads_requested": os.environ.get("POPS_THREADS", "1"),
                   "example_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "records": records}
        (destination / "receipt.json").write_text(receipt_json(receipt) + "\n")
        print(receipt_json(receipt))
        if not accepted:
            final_failure = b"M02 integrated L1 error must decrease strictly under refinement"
    except Exception as error:
        final_failure = (type(error).__name__ + ": " + str(error)).encode()
final_failure = world.broadcast_bytes(final_failure, root=0)
if final_failure:
    raise RuntimeError(final_failure.decode())
