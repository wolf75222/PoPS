#!/usr/bin/env python3
"""Closed periodic M27 mixed linear subcase, with c and mu solved jointly.

This is backward Euler for c_t=Delta(mu), mu=c-epsilon² Delta(c).
It does not qualify the nonlinear double-well potential or phase separation.
All thresholds below are fixed before any native reception.
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
from pops._ir.elliptic import DivCoeffGrad, Reaction
from pops._ir.quantity import PhysicalSupport
from pops.domain import CartesianDomain
from pops.fields import (
    CellCenteredGeneralCoupled, FieldBoundary, FieldDiscretization, FieldProblem, bcs,
)
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model import Handle, OwnerPath
from pops.projection import ConservativeCellAverage
from pops.solvers import GMRES
from pops.time import FailRun, FixedDt

from api040_m27_mixed_oracle import (
    EPSILON, STEP, STEPS, fourier_mixed_step, initial_means, integrated_initial_means,
    original_residuals, quadratic_energy,
)
from api040_receipts import receipt_json


RESOLUTIONS = (16, 32, 64)
CRITERIA = {
    "initial_max_error": 2e-14,
    "native_c_max_error": 3e-10,
    "native_mu_max_error": 3e-10,
    "original_mass_residual": 1e-10,
    "original_chemical_residual": 1e-10,
    "mass_error": 2e-12,
    "energy_increase": 3e-12,
    "time_error": 3e-14,
}


def build_case(cells: int, *, permuted: bool = False, solver_iterations: int = 400,
               solver_restart: int = 128):
    frame = CartesianDomain("mixed_periodic", lower=(0.,), upper=(1.,)).frame(
        Cartesian1D())
    model = pops.Model("linear_concentration", frame=frame)
    state = model.state(
        "concentration", components=("c",), sampling="cell_average",
        support=PhysicalSupport((("x", "mixed_periodic"),)))
    case = pops.Case("m27_linear_mixed_periodic")
    block = case.block("concentration", model, states=(state,))
    c, mu = (Handle(name, kind="field", owner=OwnerPath.model("m27_mixed_fields"))
             for name in ("c_next", "mu_next"))
    equations = {
        c: Reaction(c, 1) + DivCoeffGrad(mu, 1, scale=-STEP) == state[0],
        mu: Reaction(mu, 1) - Reaction(c, 1)
            + DivCoeffGrad(c, EPSILON**2) == 0,
    }
    unknowns = (mu, c) if permuted else (c, mu)
    problem = FieldProblem(
        "mixed_backward_euler", unknowns=unknowns,
        equations=tuple(equations[unknown] for unknown in unknowns),
        boundaries=tuple(FieldBoundary(
            unknown, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic()))
            for unknown in unknowns))
    field = case.field(problem, FieldDiscretization(
        method=CellCenteredGeneralCoupled(), boundaries=(),
        solver=GMRES(max_iter=solver_iterations, restart=solver_restart,
                     rel_tol=1e-12, abs_tol=1e-12)))
    program = pops.Program("m27_mixed_be")
    current = program.state(block[state])
    solved = program.solve(field, values={block[state]: current.n},
                           at=current.next.point).consume(action=FailRun())
    observed = field.observe(solved)
    chemical = observed[field[mu]]
    concentration = observed.cell_mean_state(field[c], target=current.next)
    program.store_history("accepted_mu", chemical, depth=1)
    program.commit(current.next, concentration)
    program.step_strategy(FixedDt(STEP))
    case.program(program)
    subject = block[state]
    case.initials.add(InitialCondition(
        state=subject, value=BindArray(), projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(
        frame=frame, cells=(cells,), periodic=PeriodicAxes(frame.axes)))
    return case, layout, subject


def _collective_call(world, label, operation):
    value = None
    failure = b""
    try:
        value = operation()
    except Exception as exception:
        failure = (label + ": " + type(exception).__name__ + ": " + str(exception)).encode()
    failures = world.allgather_bytes(failure)
    if any(failures):
        raise RuntimeError("; ".join(row.decode() for row in failures if row))
    return value


def _prepared_case(cells: int, permuted: bool, *, solver_iterations: int = 400):
    initial = np.ascontiguousarray(initial_means(cells)[None, :])
    if initial.shape != (1, cells) or not np.all(np.isfinite(initial)) \
            or np.max(np.abs(initial[0] - integrated_initial_means(cells))) > 2e-14:
        raise AssertionError("M27 initial data must be true cell-volume means")
    case, layout, subject = build_case(
        cells, permuted=permuted, solver_iterations=solver_iterations)
    validated = pops.validate(case)
    resolved = pops.resolve(validated, layout=layout)
    if resolved.resolved_dimension != 1:
        raise RuntimeError("M27 requires a genuine one-dimensional plan")
    bindings = resolved.initial_condition_plan.bindings
    if len(bindings) != 1 or bindings[0].subject != validated.resolve(subject):
        raise RuntimeError("M27 initial subject differs from its exact BindArray authority")
    return initial, subject, resolved


def _assess_saved(path: Path, cells: int) -> dict:
    with np.load(path) as saved:
        c = np.asarray(saved["c"], dtype=float)
        mu = np.asarray(saved["mu"], dtype=float)
        times = np.asarray(saved["times"], dtype=float)
    if c.shape != (STEPS + 1, cells) or mu.shape != (STEPS, cells) \
            or times.shape != (STEPS + 1,) or not np.all(np.isfinite(c)) \
            or not np.all(np.isfinite(mu)):
        raise AssertionError("M27 saved native trajectory has missing/nonfinite cell means")
    initial = initial_means(cells)
    oracle_c = initial.copy()
    c_errors = []
    mu_errors = []
    for index in range(STEPS):
        oracle_c, oracle_mu = fourier_mixed_step(oracle_c)
        c_errors.append(float(np.max(np.abs(c[index + 1] - oracle_c))))
        mu_errors.append(float(np.max(np.abs(mu[index] - oracle_mu))))
    energies = [quadratic_energy(value) for value in c]
    residuals = [original_residuals(c[index], c[index + 1], mu[index])
                 for index in range(STEPS)]
    metrics = {
        "cells": cells, "dt": STEP, "planned_steps": STEPS,
        "initial_max_error": float(np.max(np.abs(c[0] - initial))),
        "native_c_max_error": max(c_errors),
        "native_mu_max_error": max(mu_errors),
        "original_mass_residual": max(row[0] for row in residuals),
        "original_chemical_residual": max(row[1] for row in residuals),
        "mass_error": float(np.max(np.abs(np.mean(c, axis=1) - np.mean(c[0])))),
        "energy_increase": float(max(energies[index + 1] - energies[index]
                                     for index in range(STEPS))),
        "time_error": float(np.max(np.abs(times - STEP * np.arange(STEPS + 1)))),
        "initial_energy": energies[0], "final_energy": energies[-1],
    }
    for name, limit in CRITERIA.items():
        if not math.isfinite(metrics[name]) or metrics[name] > limit:
            raise AssertionError("M27 saved-state criterion failed: " + name)
    if not energies[-1] < energies[0]:
        raise AssertionError("M27 quadratic energy did not decay")
    return metrics


def run_and_archive(destination: Path) -> list[dict]:
    from pops._native_selector import select_native_dimension

    native = select_native_dimension(1)
    bootstrap_world = native.mpi_world()

    def authenticate_installation():
        package = Path(pops.__file__).resolve()
        if "PYTHONPATH" in os.environ or not package.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError("M27 reception requires installed PoPS with PYTHONPATH unset")
        pops.set_threads(int(os.environ.get("POPS_THREADS", "1")))
        return Path(destination)

    destination = _collective_call(
        bootstrap_world, "installed identity", authenticate_installation)
    from pops.codegen._native_mpi import native_mpi_communicator
    mpi_route = _collective_call(
        bootstrap_world, "MPI route", lambda: native_mpi_communicator(native))
    if len(set(bootstrap_world.allgather_bytes(mpi_route.encode()))) != 1:
        raise RuntimeError("M27 MPI route differs across ranks")
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
    variants = tuple((cells, False) for cells in RESOLUTIONS) + ((16, True),)
    for cells, permuted in variants:
        initial, subject, resolved = _collective_call(
            bootstrap_world, f"N={cells} permuted={permuted} preflight",
            lambda cells=cells, permuted=permuted: _prepared_case(cells, permuted))
        if compile_once is not None:
            artifact = compile_once(
                bootstrap_world, resolved, route="M27 periodic mixed linear",
                compile_artifact=pops.compile)
        else:
            artifact = pops.compile(resolved)
        def require_dim1_artifact(artifact=artifact):
            if artifact.resolved_dimension != 1:
                raise RuntimeError("M27 requires a true Dim1 artifact")
            return True

        _collective_call(bootstrap_world, "Dim1 artifact", require_dim1_artifact)
        execution = _collective_call(
            bootstrap_world, "execution context", lambda artifact=artifact:
            pops.ExecutionContext.mpi_world(artifact))
        world = _collective_call(
            bootstrap_world, "execution communicator", lambda execution=execution:
            execution.communicator.handle)
        mismatch = world.rank != bootstrap_world.rank or world.size != bootstrap_world.size
        if any(row == b"different" for row in bootstrap_world.allgather_bytes(
                b"different" if mismatch else b"same")):
            raise RuntimeError("M27 bound world differs from its preflight world")
        simulation = _collective_call(world, "bind", lambda artifact=artifact,
                initial=initial, subject=subject, execution=execution: pops.bind(
                    artifact, initial_values={subject: initial},
                    resources={"execution_context": execution}))
        c_global = [_collective_call(world, "initial gather", lambda simulation=simulation:
                                     simulation.state_global("concentration"))]
        mu_global = []
        reports = []
        times = [0.]
        for index in range(STEPS):
            t_end = STEP * (index + 1)
            report = _collective_call(
                world, f"accepted step {index + 1}",
                lambda simulation=simulation, t_end=t_end: pops.run(
                    simulation, t_end=t_end, max_steps=1, console=False))
            reports.append(report)
            c_global.append(_collective_call(
                world, "concentration gather", lambda simulation=simulation:
                simulation.state_global("concentration")))
            mu_global.append(_collective_call(
                world, "chemical-potential history gather", lambda simulation=simulation:
                simulation.history_global("accepted_mu", 1)))
            times.append(_collective_call(
                world, "accepted time", lambda simulation=simulation: simulation.time()))
        status = _collective_call(
            world, "run status", lambda simulation=simulation, reports=reports:
            json.dumps((simulation.macro_step(), simulation.time(),
                        tuple((row.accepted_steps, row.rejected_steps) for row in reports))).encode())
        if len(set(world.allgather_bytes(status))) != 1:
            raise RuntimeError("M27 accepted status differs across ranks")
        failure = b""
        if world.rank == 0:
            try:
                destination.mkdir(parents=True, exist_ok=True)
                saved = destination / f"state_{cells}_{'permuted' if permuted else 'canonical'}.npz"
                np.savez_compressed(
                    saved,
                    c=np.asarray(c_global, dtype=float).reshape(STEPS + 1, cells),
                    mu=np.asarray(mu_global, dtype=float).reshape(STEPS, cells),
                    times=np.asarray(times, dtype=float))
                metrics = _assess_saved(saved, cells)
                if any(row.accepted_steps != 1 or row.rejected_steps != 0 for row in reports):
                    raise AssertionError("M27 native run must accept exactly ten BE steps")
                metrics.update({
                    "permuted": permuted, "mpi_ranks": world.size,
                    "artifact_identity": artifact.artifact_identity.token,
                    "artifact_abi_key": artifact.abi_key,
                    "execution_context": execution.to_data(),
                    "saved_state": saved.name,
                    "saved_state_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
                    "run_reports": [row.to_data() for row in reports],
                })
                records.append(metrics)
            except Exception as exception:
                failure = (type(exception).__name__ + ": " + str(exception)).encode()
        failure = world.broadcast_bytes(failure, root=0)
        if failure:
            raise RuntimeError(failure.decode())
    payload = b""
    failure = b""
    if world.rank == 0:
        try:
            payload = (receipt_json({
                "schema_version": 1, "case": "M27_mixed_linear_periodic",
                "status": "passed", "criteria": CRITERIA, "records": records,
                "scope": "two original linear mixed equations; genuine Dim1 periodic Uniform",
                "nonlinear_double_well_qualified": False,
                "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                "mpi_ranks": world.size,
                "threads_requested": int(os.environ.get("POPS_THREADS", "1")),
                "example_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "oracle_sha256": hashlib.sha256(
                    Path(__file__).with_name("api040_m27_mixed_oracle.py").read_bytes()).hexdigest(),
            }) + "\n").encode()
            (destination / "receipt.json").write_bytes(payload)
        except Exception as exception:
            failure = (type(exception).__name__ + ": " + str(exception)).encode()
    failure = world.broadcast_bytes(failure, root=0)
    if failure:
        raise RuntimeError(failure.decode())
    return json.loads(world.broadcast_bytes(payload, root=0))["records"]


if __name__ == "__main__":
    run_and_archive(Path(os.environ["POPS_API040_OUTPUT"]))
