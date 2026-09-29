#!/usr/bin/env python3
"""M05 closed periodic shear subproblem, genuine Dim1 Diffusion reception.

The equation here is u_t = 0.03 u_xx, u(x,0)=sin(2*pi*x). It isolates one
viscous shear component; it is not a compressible Navier--Stokes, wall-Couette,
or heat-conduction qualification. All numerical criteria precede native runs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pops
from pops import math as pops_math
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.time import ForwardEuler
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt

from api040_m05_shear_oracle import (
    energy, forward_euler_states, fourier_amplitudes, initial_means,
    integrated_initial_means, periodic_faces, periodic_rate, semidiscrete_work,
)
from api040_receipts import receipt_json


VISCOSITY = .03
T_END = .1
RESOLUTIONS = (32, 64, 128)
CRITERIA = {
    "initial_max_error": 2e-14,
    "saved_state_max_error": 5e-12,
    "continuous_state_max_error": 2e-4,
    "mass_error": 5e-13,
    "ledger_flux_max_error": 2e-12,
    "ledger_cell_change_error": 5e-13,
    "semidiscrete_work_error": 5e-13,
    "forward_euler_energy_error": 5e-13,
    "time_error": 3e-14,
    "explicit_frequency_fraction_max": .9,
}


def build_case(cells: int):
    frame = CartesianDomain("periodic_shear", lower=(0.,), upper=(1.,)).frame(
        Cartesian1D())
    model = pops.Model("linear_viscous_shear", frame=frame)
    state = model.state("transverse_velocity", components=("u",))
    flux = model.diffusive_flux(
        "viscous_gradient", state=state, value=VISCOSITY * pops_math.grad(state[0]))
    rate = model.rate("shear_balance", equation=pops_math.ddt(state) == pops_math.div(flux))
    case = pops.Case("m05_closed_periodic_shear")
    block = case.block("shear", model, states=(state,))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, Diffusion(flux=flux))
    case.numerics(numerics, block=block)
    step = T_END / cells
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(step))
    case.program(program)
    subject = block[state]
    case.initials.add(InitialCondition(
        state=subject, value=BindArray(), projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(
        frame=frame, cells=(cells,), periodic=PeriodicAxes(frame.axes)))
    return case, layout, subject


def _collective_call(world, label, operation):
    result = None
    failure = b""
    try:
        result = operation()
    except Exception as exception:
        failure = (label+": "+type(exception).__name__+": "+str(exception)).encode()
    failures = world.allgather_bytes(failure)
    if any(failures):
        raise RuntimeError("; ".join(row.decode() for row in failures if row))
    return result


def _prepared_case(cells: int):
    step = T_END / cells
    fraction = step * 2 * VISCOSITY * cells**2
    if fraction > CRITERIA["explicit_frequency_fraction_max"]:
        raise AssertionError("M05 authored Forward Euler frequency bound is inadmissible")
    initial = np.ascontiguousarray(initial_means(cells)[None, :])
    if np.max(np.abs(initial[0] - integrated_initial_means(cells))) > 2e-14:
        raise AssertionError("M05 requires true cell means, not center samples")
    case, layout, subject = build_case(cells)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    return step, fraction, initial, subject, resolved


def _ledger_metrics(rows: list[dict], before: np.ndarray, after: np.ndarray,
                    step: float) -> dict[str, float | int]:
    """Reconstruct the last accepted cell update from genuine face records."""
    cells = len(before)
    faces = periodic_faces(before, VISCOSITY)
    increments = np.zeros(cells)
    max_flux_error = 0.
    quadratures, occurrences, contexts = set(), set(), set()
    if len(rows) != 2 * cells:
        raise AssertionError("M05 last accepted ledger must contain two incidences per cell")
    for row in rows:
        cell_token, axis_token, side_token = row["quadrature_identity"].split("/")
        cell = int(cell_token.split(":")[1])
        axis = int(axis_token.split(":")[1])
        side = int(side_token.split(":")[1])
        if not 0 <= cell < cells or axis != 0 or side not in (0, 1):
            raise AssertionError("M05 ledger has a foreign cell/axis/side")
        if (row["orientation"] != (-1 if side == 0 else 1) or
                row["multiplicity"] != 1 or abs(row["face_measure"] - 1) > 2e-14 or
                abs(row["temporal_weight"] - step) > 2e-14):
            raise AssertionError("M05 ledger has a foreign oriented quadrature weight")
        expected_flux = faces[(cell - 1) % cells if side == 0 else cell]
        max_flux_error = max(max_flux_error, abs(row["numerical_flux"] - expected_flux))
        increments[cell] += row["integrated_amount"]
        quadratures.add(row["quadrature_identity"])
        occurrences.add(row["occurrence_identity"])
        contexts.add(row["evaluation_context"])
    if len(quadratures) != 2 * cells or len(occurrences) != 1 or len(contexts) != 1:
        raise AssertionError("M05 ledger lost occurrence/stage/owned-incidence identity")
    return {
        "accepted_exchange_count": len(rows),
        "unique_quadratures": len(quadratures),
        "ledger_flux_max_error": float(max_flux_error),
        "ledger_cell_change_error": float(np.max(np.abs(
            increments - (after - before) / cells))),
        "ledger_mass_error": float(abs(np.sum(increments))),
    }


def _assess_saved_state(path: Path, rows: list[dict], cells: int,
                        first_report, last_report) -> dict:
    with np.load(path) as saved:
        initial = np.asarray(saved["initial"]).reshape(cells)
        before = np.asarray(saved["before_last"]).reshape(cells)
        final = np.asarray(saved["final"]).reshape(cells)
        time = float(saved["time"])
        step = float(saved["dt"])
    oracle_before, oracle_final = forward_euler_states(cells, VISCOSITY, T_END, cells)
    continuous, semidiscrete, forward_euler, eigenvalue = fourier_amplitudes(
        cells, VISCOSITY, T_END, cells)
    prescribed = initial_means(cells)
    rate = periodic_rate(before, VISCOSITY)
    work, dissipation = semidiscrete_work(before, VISCOSITY)
    temporal_correction = .5 * step**2 * float(np.dot(rate, rate)) / cells
    energy_change = energy(final) - energy(before)
    ledger = _ledger_metrics(rows, before, final, step)
    metrics = {
        "cells": cells, "dt": step, "planned_steps": cells,
        "initial_max_error": float(np.max(np.abs(initial - prescribed))),
        "before_last_max_error": float(np.max(np.abs(before - oracle_before))),
        "saved_state_max_error": float(np.max(np.abs(final - oracle_final))),
        "continuous_state_max_error": float(np.max(np.abs(final - continuous * prescribed))),
        "semidiscrete_state_max_error": float(np.max(np.abs(final - semidiscrete * prescribed))),
        "mass_error": float(abs(np.mean(final - initial))),
        "semidiscrete_work": work, "face_jump_dissipation": dissipation,
        "semidiscrete_work_error": abs(work - dissipation),
        "last_step_energy_change": energy_change,
        "forward_euler_temporal_correction": temporal_correction,
        "forward_euler_energy_error": abs(energy_change - (step * dissipation + temporal_correction)),
        "initial_energy": energy(initial), "final_energy": energy(final),
        "continuous_amplitude": continuous,
        "semidiscrete_amplitude": semidiscrete,
        "forward_euler_amplitude": forward_euler,
        "spatial_symbol": eigenvalue,
        "time_error": abs(time - T_END),
        "accepted_steps": first_report.accepted_steps + last_report.accepted_steps,
        "rejected_steps": first_report.rejected_steps + last_report.rejected_steps,
        **ledger,
    }
    for name, limit in CRITERIA.items():
        if name == "explicit_frequency_fraction_max":
            continue
        if name in metrics and (not math.isfinite(metrics[name]) or metrics[name] > limit):
            raise AssertionError("M05 saved-state criterion failed: " + name)
    if (metrics["before_last_max_error"] > CRITERIA["saved_state_max_error"] or
            metrics["ledger_mass_error"] > CRITERIA["mass_error"] or
            metrics["accepted_steps"] != cells or metrics["rejected_steps"] != 0 or
            not (0 < metrics["final_energy"] < metrics["initial_energy"]) or
            not (work < 0 and dissipation < 0 and energy_change < 0)):
        raise AssertionError("M05 damping, last-step, or accepted-step criterion failed")
    return metrics


def run_and_archive(destination: Path) -> list[dict]:
    from pops._native_selector import select_native_dimension

    native = select_native_dimension(1)
    bootstrap_world = native.mpi_world()

    def authenticate_installation():
        package = Path(pops.__file__).resolve()
        if "PYTHONPATH" in os.environ or not package.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError("M05 reception requires installed PoPS with PYTHONPATH unset")
        pops.set_threads(int(os.environ.get("POPS_THREADS", "1")))
        return Path(destination)

    destination = _collective_call(
        bootstrap_world, "installed identity and output preflight", authenticate_installation)
    from pops.codegen._native_mpi import native_mpi_communicator
    mpi_route = _collective_call(
        bootstrap_world, "MPI route", lambda: native_mpi_communicator(native))
    routes = bootstrap_world.allgather_bytes(mpi_route.encode())
    if len(set(routes)) != 1:
        raise RuntimeError("M05 MPI route differs across ranks")
    compile_once = None
    if mpi_route == "MPI_COMM_WORLD":
        def load_compile_helper():
            repository = str(Path(__file__).resolve().parents[3])
            sys.path.insert(0, repository)
            try:
                from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
            finally:
                sys.path.remove(repository)
            return compile_resolved_plan_once

        compile_once = _collective_call(
            bootstrap_world, "authenticated MPI compiler helper", load_compile_helper)
    records = []
    for cells in RESOLUTIONS:
        step, fraction, initial, subject, resolved = _collective_call(
            bootstrap_world, f"N={cells} numpy/authoring/resolve preflight",
            lambda cells=cells: _prepared_case(cells))
        if compile_once is not None:
            artifact = compile_once(
                bootstrap_world, resolved, route="M05 periodic shear",
                compile_artifact=pops.compile)
        else:
            artifact = pops.compile(resolved)
        def prepare_execution(artifact=artifact):
            if artifact.resolved_dimension != 1:
                raise RuntimeError("M05 requires a genuine one-dimensional artifact")
            return pops.ExecutionContext.mpi_world(artifact)

        execution = _collective_call(
            bootstrap_world, "Dim1 artifact and execution context", prepare_execution)
        world = _collective_call(
            bootstrap_world, "execution communicator",
            lambda execution=execution: execution.communicator.handle)
        mismatch = world.rank != bootstrap_world.rank or world.size != bootstrap_world.size
        if any(row == b"different" for row in bootstrap_world.allgather_bytes(
                b"different" if mismatch else b"same")):
            raise RuntimeError("M05 bound execution world differs from its preflight world")
        simulation = _collective_call(world, "bind", lambda artifact=artifact,
                initial=initial, subject=subject, execution=execution: pops.bind(
                    artifact, initial_values={subject: initial},
                    resources={"execution_context": execution}))
        bound_global = _collective_call(
            world, "initial gather", lambda simulation=simulation:
            simulation.state_global("shear"))
        before_global = None
        first_report = _collective_call(
            world, "first N-1 steps", lambda simulation=simulation, cells=cells,
            step=step: pops.run(
                simulation, t_end=T_END-step, max_steps=cells-1, console=False))
        before_global = _collective_call(
            world, "penultimate gather", lambda simulation=simulation:
            simulation.state_global("shear"))
        last_report = _collective_call(
            world, "last step", lambda simulation=simulation: pops.run(
                simulation, t_end=T_END, max_steps=1, console=False))
        last_dt = _collective_call(
            world, "accepted last duration", lambda simulation=simulation:
            simulation._executor.program_last_dt())
        final_global = _collective_call(
            world, "final gather", lambda simulation=simulation:
            simulation.state_global("shear"))
        local_rows = _collective_call(
            world, "accepted ledger", lambda simulation=simulation:
            simulation._executor._program_exchange_records())
        local_wire = _collective_call(
            world, "ledger serialization", lambda local_rows=local_rows:
            receipt_json(local_rows).encode())
        gathered_rows = world.allgather_bytes(local_wire)
        status = _collective_call(
            world, "status", lambda simulation=simulation,
            first_report=first_report, last_report=last_report, last_dt=last_dt: json.dumps((
                first_report.accepted_steps, first_report.rejected_steps,
                last_report.accepted_steps, last_report.rejected_steps,
                simulation.macro_step(), simulation.time(), last_dt)).encode())
        if len(set(world.allgather_bytes(status))) != 1:
            raise RuntimeError("M05 time/step status differs across ranks")
        failure = b""
        if world.rank == 0:
            try:
                destination.mkdir(parents=True, exist_ok=True)
                saved = destination / f"state_{cells}.npz"
                np.savez_compressed(
                    saved, initial=np.asarray(bound_global).reshape(initial.shape),
                    before_last=np.asarray(before_global).reshape(initial.shape),
                    final=np.asarray(final_global).reshape(initial.shape),
                    dt=last_dt, cells=cells, time=simulation.time())
                rows = [row for piece in gathered_rows for row in json.loads(piece)]
                metrics = _assess_saved_state(
                    saved, rows, cells, first_report, last_report)
                metrics.update({
                    "explicit_frequency_fraction": fraction,
                    "mpi_ranks": world.size,
                    "artifact_identity": artifact.artifact_identity.token,
                    "artifact_abi_key": artifact.abi_key,
                    "execution_context": execution.to_data(),
                    "saved_state": saved.name,
                    "saved_state_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
                    "first_run_report": first_report.to_data(),
                    "last_run_report": last_report.to_data(),
                })
                records.append(metrics)
            except Exception as exception:
                failure = (type(exception).__name__+": "+str(exception)).encode()
        failure = world.broadcast_bytes(failure, root=0)
        if failure:
            raise RuntimeError(failure.decode())
    payload = b""
    failure = b""
    if world.rank == 0:
        try:
            payload = (receipt_json({
                "schema_version": 1, "case": "M05_periodic_linear_viscous_shear",
                "scope": "genuine Dim1 periodic scalar shear reduction, not full Navier-Stokes",
                "status": "passed", "criteria": CRITERIA, "records": records,
                "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                "mpi_ranks": world.size,
                "threads_requested": int(os.environ.get("POPS_THREADS", "1")),
            })+"\n").encode()
            (destination / "receipt.json").write_bytes(payload)
        except Exception as exception:
            failure = (type(exception).__name__+": "+str(exception)).encode()
    failure = world.broadcast_bytes(failure, root=0)
    if failure:
        raise RuntimeError(failure.decode())
    payload = world.broadcast_bytes(payload, root=0)
    return json.loads(payload)["records"]


if __name__ == "__main__":
    run_and_archive(Path(os.environ["POPS_API040_OUTPUT"]))
