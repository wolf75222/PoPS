#!/usr/bin/env python3
"""M01/W01: periodic FV cosine, MC/SSPRK2, four full 2D uniform grids.

The one-dimensional physical solution is invariant in y. Both coordinates have
N cells; this is the production Dim=2 specialization, not a Dim=1 qualification.
Run with the installed package, PYTHONPATH unset, POPS_NATIVE_DIM=2, and an
MPI-enabled environment. POPS_THREADS selects CPU threads before Kokkos starts.
"""
# ruff: noqa: E402
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np
import pops
from api040_receipts import receipt_json

package = Path(pops.__file__).resolve()
if "PYTHONPATH" in os.environ or not package.is_relative_to(Path(sys.prefix).resolve()):
    raise RuntimeError("M01 reception requires installed PoPS with PYTHONPATH unset")
pops.set_threads(int(os.environ.get("POPS_THREADS", "1")))

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import SSPRK2
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import AdaptiveCFL


# Physical equation and periodic domain: u_t + u_x = 0, a = 1.
RESOLUTIONS = (40, 80, 160, 320)
CFL = 0.3
T_END = 0.125
CRITERIA = {"minimum_order": 1.6, "mass_tolerance": 1.e-12,
            "time_tolerance": 1.e-14, "initial_tolerance": 1.e-14,
            "bound_tolerance": 1.e-12}


def exact_cell_averages(n, time_value=0.):
    """Analytic cell integral of the translated cosine, divided by its width."""
    centers = (np.arange(n, dtype=float) + .5) / n
    return 1. + .1 * np.sinc(1. / n) * np.cos(2 * np.pi * (centers - time_value))

frame = Rectangle("periodic_square", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
x_axis, y_axis = frame.axes
model = pops.Model("cosine_advection", frame=frame)
U = model.state("U", components=("u",), representation=Conservative(),
                space=CellState(frame=frame))
(u,) = U
F = model.flux("transport", frame=frame, state=U,
               components={x_axis: (u,), y_axis: (0 * u,)},
               waves={x_axis: (1.,), y_axis: (0.,)})
rate = model.rate("balance", equation=ddt(U) == -div(F))

# Space, time and acceptance are separate declarations. The order threshold is
# a reception criterion evaluated against saved states, not a solver option.
numerics = DiscretizationPlan()
numerics.rates.add(rate, FiniteVolume(
    flux=F, variables=variables.Conservative(U),
    reconstruction=reconstruction.MUSCL(limiters.MC()), riemann=riemann.Rusanov()))
case = pops.Case("api040_M01_cell_averages")
tracer = case.block("tracer", model)
case.numerics(numerics, block=tracer)
program = SSPRK2(tracer[U], rate=rate)
program.step_strategy(AdaptiveCFL(cfl=CFL))
case.program(program)
validated = pops.validate(case)

destination = Path(os.environ.get("POPS_API040_OUTPUT", "outputs/api040_m01"))
records = []
for n in RESOLUTIONS:
    # Integral of cos(2*pi*x) divided by cell width, including the exact shift.
    centers = (np.arange(n, dtype=float) + .5) / n
    initial_line = exact_cell_averages(n)
    exact_line = exact_cell_averages(n, T_END)
    edges = np.arange(n + 1, dtype=float) / n
    integral_line = 1. + .1 * np.diff(np.sin(2 * np.pi * edges)) * n / (2 * np.pi)
    point_line = 1. + .1 * np.cos(2 * np.pi * centers)
    projection_gap = float(np.max(np.abs(initial_line - point_line)))
    if (not np.allclose(initial_line, integral_line, rtol=0., atol=CRITERIA["initial_tolerance"])
            or projection_gap <= 1.e-8):
        raise RuntimeError("M01 requires actual cell averages, distinguishable from center samples")
    initial = np.ascontiguousarray(np.broadcast_to(initial_line, (1, n, n)))
    exact = np.ascontiguousarray(np.broadcast_to(exact_line, (1, n, n)))
    grid = CartesianGrid(frame=frame, cells=(n, n), periodic=PeriodicAxes(frame.axes))
    started = time.perf_counter()
    artifact = pops.compile(pops.resolve(validated, layout=Uniform(grid)))
    compiled_seconds = time.perf_counter() - started
    if artifact.resolved_dimension != 2:
        raise RuntimeError("M01 receipt is scoped to the actual Dim=2 artifact")
    context = pops.ExecutionContext.mpi_world(artifact)
    world = context.communicator.handle
    simulation = pops.bind(artifact, initial_state={"tracer": initial},
                           resources={"execution_context": context})
    # Every rank participates in the gather; the selected native world owns MPI.
    bound_initial = np.asarray(simulation.state_global("tracer"))
    initial_error = 0.
    initial_failure = b""
    if world.rank == 0:
        if bound_initial.size != initial.size:
            initial_failure = b"M01 initial gather has the wrong full-grid shape"
        else:
            bound_initial = bound_initial.reshape(initial.shape).copy()
            initial_error = float(np.max(np.abs(bound_initial - initial)))
            if not np.isfinite(initial_error) or initial_error > CRITERIA["initial_tolerance"]:
                initial_failure = b"M01 bound state is not the prescribed cell averages"
    initial_failure = world.broadcast_bytes(initial_failure, root=0)
    if initial_failure:
        raise RuntimeError(initial_failure.decode())
    started = time.perf_counter()
    report = pops.run(simulation, t_end=T_END, max_steps=10_000)
    elapsed = time.perf_counter() - started
    gathered = np.asarray(simulation.state_global("tracer"))
    publication_failure = b""
    if world.rank == 0:
        try:
            if gathered.size != initial.size:
                raise RuntimeError("M01 final gather has the wrong full-grid shape")
            actual = gathered.reshape(initial.shape)
            destination.mkdir(parents=True, exist_ok=True)
            state_path = destination / ("state_%d.npz" % n)
            np.savez_compressed(state_path, initial=bound_initial, final=actual, exact=exact,
                                time=simulation.time(), cells=n)
            # All acceptance statistics use these reopened, saved state bytes.
            with np.load(state_path) as saved:
                error = saved["final"] - saved["exact"]
                l1 = float(np.mean(np.abs(error)))
                mass_error = float(abs(np.mean(saved["final"]) - np.mean(saved["initial"])))
                saved_time = float(saved["time"])
                minimum, maximum = float(saved["final"].min()), float(saved["final"].max())
            records.append({"cells": [n, n], "time": saved_time,
                            "accepted_steps": report.accepted_steps, "l1": l1,
                            "initial_max_error": initial_error, "center_average_gap": projection_gap,
                            "mass_error": mass_error, "minimum": minimum, "maximum": maximum,
                            "compile_seconds": compiled_seconds, "run_seconds": elapsed,
                            "run_report": report.to_data(), "execution_context": context.to_data(),
                            "runtime_backend": context.backend.to_data(),
                            "mpi_ranks": world.size, "mpi_root": world.rank,
                            "saved_state": state_path.name,
                            "saved_state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest()})
        except Exception as error:
            publication_failure = (type(error).__name__ + ": " + str(error)).encode()
    publication_failure = world.broadcast_bytes(publication_failure, root=0)
    if publication_failure:
        raise RuntimeError(publication_failure.decode())

if world.rank == 0:
    try:
        errors = np.asarray([row["l1"] for row in records])
        orders = np.log2(errors[:-1] / errors[1:])
        from pops import _pops  # Provenance only; simulation above uses the public API.
        native = Path(_pops.__file__).resolve()
        accepted = (len(records) == 4 and bool(np.all(orders > CRITERIA["minimum_order"]))
                    and all(row["mass_error"] < CRITERIA["mass_tolerance"]
                            and abs(row["time"] - T_END) < CRITERIA["time_tolerance"]
                            and abs(row["run_report"]["final_time"] - row["time"]) < CRITERIA["time_tolerance"]
                            and row["initial_max_error"] <= CRITERIA["initial_tolerance"]
                            and row["accepted_steps"] > 0
                            and row["minimum"] >= .9 - CRITERIA["bound_tolerance"]
                            and row["maximum"] <= 1.1 + CRITERIA["bound_tolerance"] for row in records))
        receipt = {"schema_version": 1, "case": "M01/W01", "status": "passed" if accepted else "failed",
                   "scope": "periodic cell-average cosine; full uniform Dim=2 grids, invariant y",
                   "equation": "u_t + u_x = 0", "initial": "cell average of 1 + 0.1*cos(2*pi*x)",
                   "method": "MUSCL-MC, Rusanov, SSPRK2", "cfl": CFL, "t_end": T_END,
                   "criteria": CRITERIA, "resolutions": RESOLUTIONS, "l1_orders": orders.tolist(), "runs": records,
                   "package_file": str(package), "package_version": pops.__version__,
                   "native_file": str(native), "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
                   "abi_key": _pops.abi_key(), "threads_requested": os.environ.get("POPS_THREADS", "1")}
        (destination / "result.json").write_text(receipt_json(receipt) + "\n")
        print(receipt_json(receipt))
        acceptance_failure = b"" if accepted else b"M01 scientific acceptance failed; inspect saved states"
    except Exception as error:
        acceptance_failure = (type(error).__name__ + ": " + str(error)).encode()
else:
    acceptance_failure = b""
acceptance_failure = world.broadcast_bytes(acceptance_failure, root=0)
if acceptance_failure:
    raise RuntimeError(acceptance_failure.decode())
