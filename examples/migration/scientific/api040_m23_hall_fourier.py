"""Selected M23 reversible-gradient Fourier witness, genuine periodic Dim1.

This is the normalized two-component Hall classification model from the 0.4
mathematical note, not a claim about complete magnetohydrodynamics. D=0 and
R=eta_H*J are retained as different physical component tensors. SSPRK2 has
its own, slightly amplifying imaginary-axis polynomial; receipt criteria
compare that polynomial separately from continuous and spatial phase.
"""
from __future__ import annotations

import hashlib
import json
import math as pymath
import os
import sys
from pathlib import Path

import numpy as np
import pops
from pops import math
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.numerics import CoupledGradient, DiscretizationPlan
from pops.lib.time import SSPRK2
from pops.time import FixedDt
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.initial import InitialCondition
from pops.lib.initial import BindArray
from pops.projection import ConservativeCellAverage
from pops._native_selector import select_native_dimension


K = 2
ETA_H = .3
T_END = .2
DT = .002
STEPS = 100
CELLS = (32, 64)
ORDERS = ((0, 1), (1, 0))
CRITERIA = {
    "initial_max_error": 3e-14,
    "saved_state_max_error": 3e-10,
    "phase_discrete_error": 3e-10,
    "norm_discrete_error": 3e-10,
    "time_error": 3e-14,
}


def build_case(cells: int, *, eta_h: float, order: tuple[int, int]):
    frame = CartesianDomain("periodic_hall", lower=(0.,), upper=(2*pymath.pi,)).frame(
        Cartesian1D())
    model = pops.Model("transverse_hall", frame=frame)
    labels = ("field_a", "field_b")
    state = model.state("w", components=tuple(labels[i] for i in order))
    canonical = ((0., -eta_h), (eta_h, 0.))
    reversible = tuple(tuple(canonical[i][j] for j in order) for i in order)
    physical = model.coupled_gradient_flux(
        "signed_gradient", state=state,
        dissipative=((0., 0.), (0., 0.)), reversible=reversible)
    rate = model.rate("balance", equation=math.ddt(state) == math.div(physical))
    case = pops.Case("m23_hall_fourier")
    block = case.block("transverse", model, states=(state,))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, CoupledGradient(flux=physical))
    case.numerics(numerics, block=block)
    program = SSPRK2(block[state], rate=rate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    case.initials.add(InitialCondition(
        state=block[state], value=BindArray(),
        projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(
        frame=frame, cells=(cells,), periodic=PeriodicAxes(frame.axes)))
    return case, layout, block[state]


def initial_means(cells: int, order: tuple[int, int]):
    centers = 2*pymath.pi*(np.arange(cells)+.5)/cells
    sinc = np.sinc(K/cells)
    canonical = np.array([1., .2])[:, None]*sinc*np.cos(K*centers)[None, :]
    return np.ascontiguousarray(canonical[list(order)])


def discrete_oracle(cells: int, eta_h: float):
    h = 2*pymath.pi/cells
    symbol = 4*np.sin(K*h/2)**2/h**2
    z = eta_h*symbol*DT
    amplification = (1-.5*z*z-1j*z)**STEPS
    return amplification, symbol


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


def run_and_archive(destination: Path):
    from api040_receipts import receipt_json

    native = select_native_dimension(1)
    package = Path(pops.__file__).resolve()
    if "PYTHONPATH" in os.environ or not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("M23 reception requires installed PoPS with PYTHONPATH unset")
    destination = Path(destination)
    records = []
    for eta_h, cells, order in (
            *((ETA_H, n, order) for order in ORDERS for n in CELLS),
            (0., CELLS[0], ORDERS[0])):
        case, layout, subject = build_case(cells, eta_h=eta_h, order=order)
        resolved = pops.resolve(pops.validate(case), layout=layout)
        from pops.codegen._native_mpi import native_mpi_communicator
        if native_mpi_communicator(native) == "MPI_COMM_WORLD":
            # The installed package remains authoritative. Only the repository's
            # MPI cache-publication test helper is loaded from this script's root.
            repository = str(Path(__file__).resolve().parents[3])
            sys.path.insert(0, repository)
            try:
                from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
            finally:
                sys.path.remove(repository)
            artifact = compile_resolved_plan_once(
                native.mpi_world(), resolved, route="M23 Hall Fourier",
                compile_artifact=pops.compile)
        else:
            artifact = pops.compile(resolved)
        if artifact.resolved_dimension != 1:
            raise RuntimeError("M23 requires the genuine one-dimensional package")
        execution = pops.ExecutionContext.mpi_world(artifact)
        world = execution.communicator.handle
        initial = initial_means(cells, order)
        simulation = _collective_call(world, "bind", lambda artifact=artifact,
                initial=initial, execution=execution: pops.bind(
            artifact, initial_values={subject: initial},
            resources={"execution_context": execution}))
        before_global = _collective_call(
            world, "initial gather", lambda simulation=simulation:
            simulation.state_global("transverse"))
        report = _collective_call(
            world, "run", lambda simulation=simulation: pops.run(
                simulation, t_end=T_END, max_steps=STEPS, console=False))
        final_global = _collective_call(
            world, "final gather", lambda simulation=simulation:
            simulation.state_global("transverse"))
        status = _collective_call(
            world, "status", lambda simulation=simulation, report=report: json.dumps((
                report.accepted_steps, report.rejected_steps,
                simulation.macro_step(), simulation.time())).encode())
        statuses = world.allgather_bytes(status)
        if len(set(statuses)) != 1:
            raise RuntimeError("M23 accepted-step/time status differs across MPI ranks")
        error = b""
        if world.rank == 0:
            try:
                before = np.asarray(before_global).reshape(initial.shape)
                final = np.asarray(final_global).reshape(initial.shape)
                destination.mkdir(parents=True, exist_ok=True)
                saved = destination / f"state_{cells}_{''.join(map(str, order))}_{eta_h:.1f}.npz"
                np.savez_compressed(
                    saved, initial=before, final=final, order=order,
                    cells=cells, eta_h=eta_h, dt=DT, time=simulation.time())
                with np.load(saved) as state:
                    canonical_initial = state["initial"][np.argsort(state["order"])]
                    canonical_final = state["final"][np.argsort(state["order"])]
                    amplification, symbol = discrete_oracle(cells, eta_h)
                    oracle_complex = (canonical_initial[0]+1j*canonical_initial[1])*amplification
                    oracle = np.stack((oracle_complex.real, oracle_complex.imag))
                    initial_error = float(np.max(np.abs(state["initial"]-initial)))
                    state_error = float(np.max(np.abs(canonical_final-oracle)))
                    center = 2*pymath.pi*(np.arange(cells)+.5)/cells
                    basis = np.cos(K*center)
                    measured = ((2/cells)*np.dot(
                        canonical_final[0]+1j*canonical_final[1], basis) /
                        ((1+.2j)*np.sinc(K/cells)))
                    phase_error = float(abs(np.angle(measured/amplification)))
                    norm_ratio = float(np.linalg.norm(canonical_final) /
                                       np.linalg.norm(canonical_initial))
                    norm_error = abs(norm_ratio-abs(amplification))
                    time_error = abs(float(state["time"])-T_END)
                if (initial_error > CRITERIA["initial_max_error"] or
                        state_error > CRITERIA["saved_state_max_error"] or
                        phase_error > CRITERIA["phase_discrete_error"] or
                        norm_error > CRITERIA["norm_discrete_error"] or
                        time_error > CRITERIA["time_error"] or
                        report.accepted_steps != STEPS or report.rejected_steps != 0 or
                        simulation.macro_step() != STEPS):
                    raise AssertionError("M23 Fourier saved-state criterion failed")
                records.append({
                    "cells": cells, "order": order, "eta_h": eta_h,
                    "continuous_phase": -eta_h*K*K*T_END,
                    "semidiscrete_phase": -eta_h*symbol*T_END,
                    "ssprk2_phase": float(np.angle(amplification)),
                    "ssprk2_norm_ratio": float(abs(amplification)),
                    "observed_norm_ratio": norm_ratio,
                    "initial_max_error": initial_error,
                    "saved_state_max_error": state_error,
                    "phase_discrete_error": phase_error,
                    "norm_discrete_error": norm_error,
                    "time_error": time_error,
                    "accepted_steps": report.accepted_steps,
                    "mpi_ranks": world.size,
                    "saved_state": saved.name,
                    "saved_state_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
                    "run_report": report.to_data(),
                })
            except Exception as exception:
                error = (type(exception).__name__+": "+str(exception)).encode()
        error = world.broadcast_bytes(error, root=0)
        if error:
            raise RuntimeError(error.decode())
    payload = b""
    error = b""
    if world.rank == 0:
        try:
            payload = (receipt_json({
                "schema_version": 1, "case": "M23_normalized_Hall_Fourier",
                "status": "passed", "scope": "periodic Dim1 transverse linear two-component classification witness",
                "criteria": CRITERIA, "records": records, "native_sha256": hashlib.sha256(
                    Path(native.__file__).read_bytes()).hexdigest(),
                "threads_requested": int(os.environ.get("OMP_NUM_THREADS", "1")),
                "mpi_ranks": world.size,
            })+"\n").encode()
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "receipt.json").write_bytes(payload)
        except Exception as exception:
            error = (type(exception).__name__+": "+str(exception)).encode()
    error = world.broadcast_bytes(error, root=0)
    if error:
        raise RuntimeError(error.decode())
    payload = world.broadcast_bytes(payload, root=0)
    return json.loads(payload)["records"]


if __name__ == "__main__":
    run_and_archive(Path(os.environ["POPS_API040_OUTPUT"]))
