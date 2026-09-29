#!/usr/bin/env python3
"""M04/W02: authored scalar advection-diffusion, exact cosine cell means.

The default native Dim=2 law is u_t + a u_x = D u_xx, with exactly zero y
diffusion. POPS_API040_M04_DIFFUSION=isotropic replays the earlier D(u_xx+u_yy)
variant whose valid combined step exposed the directional-frequency defect.
Both use y-invariant exact cell averages; their different full constitutive
laws and bounds are recorded explicitly. Neither qualifies native Dim=1.
POPS_API040_M04_METHOD=ssprk2 selects a distinct temporal method. The original
ForwardEuler case and its pre-asymptotic order failure are retained explicitly.
Run with installed PoPS, PYTHONPATH unset, and POPS_NATIVE_DIM=2.
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
    raise RuntimeError("M04 reception requires installed PoPS with PYTHONPATH unset")
pops.set_threads(int(os.environ.get("POPS_THREADS", "1")))

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler, SSPRK2
from pops.math import CoeffGradient, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import FixedDt

from api040_m04_oracle import discrete_cell_means, exact_cell_means, frequencies
from api040_receipts import receipt_json


# These are fixed before compilation or native data are observed.
VELOCITY = 1.
DIFFUSIVITY = .01
DIFFUSION_VARIANT = os.environ.get("POPS_API040_M04_DIFFUSION", "x_only")
if DIFFUSION_VARIANT not in ("x_only", "isotropic"):
    raise ValueError("M04 diffusion variant must be x_only or isotropic")
TRANSVERSE_DIFFUSIVITY = 0. if DIFFUSION_VARIANT == "x_only" else DIFFUSIVITY
AMPLITUDE = .2
T_END = .1
RESOLUTIONS = (32, 64, 128)
SAFETY_FACTOR = .9
TEMPORAL_METHOD = os.environ.get("POPS_API040_M04_METHOD", "forward_euler")
if TEMPORAL_METHOD not in ("forward_euler", "ssprk2"):
    raise ValueError("M04 method must be forward_euler or ssprk2")
CRITERIA = {"density_l1_max": {32: .009, 64: .0048, 128: .0024},
            "minimum_observed_order": .7,
            "mass_defect_max": 2.e-11,
            "initial_max_error": 1.e-12,
            "time_error_max": 1.e-12,
            "y_invariance_max": 2.e-11,
            "minimum_state": .7999999999,
            "maximum_state": 1.2000000001}
CRITERIA["discrete_fourier_max_error"] = 3.e-12


def author_case(step: float):
    """Physical law, discrete method, time method, and acceptance stay distinct."""
    # Physics: the tensor is explicit, including its exact zero axis.
    frame = Rectangle("periodic_square", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("scalar_advection_diffusion", frame=frame)
    state = model.state("conserved_tracer", components=("amount",),
                        representation=Conservative(), space=CellState(frame=frame))
    (u,) = state
    advective = model.flux("physical_transport", frame=frame, state=state,
                           components={x_axis: (VELOCITY * u,), y_axis: (0 * u,)},
                           waves={x_axis: (VELOCITY,), y_axis: (0.,)})
    diffusive = model.diffusive_flux("physical_diffusion", state=state,
                                     value=CoeffGradient(u, ((DIFFUSIVITY, 0.),
                                                            (0., TRANSVERSE_DIFFUSIVITY))))
    rate = model.rate("physical_balance", equation=ddt(state) ==
                      -div(advective) + div(diffusive))

    # One combined rate is selected. The method owns both finite-volume
    # transport and diffusion; no separately scheduled additive update exists.
    transport = FiniteVolume(flux=advective, variables=variables.Conservative(state),
                             reconstruction=reconstruction.FirstOrder(),
                             riemann=riemann.Rusanov())
    plan = DiscretizationPlan()
    plan.rates.add(rate, Diffusion(flux=diffusive, transport=transport))
    case = pops.Case("api040_M04_W02")
    block = case.block("tracer", model)
    case.numerics(plan, block=block)
    method = {"forward_euler": ForwardEuler, "ssprk2": SSPRK2}[TEMPORAL_METHOD]
    program = method(block[state], rate=rate)
    program.step_strategy(FixedDt(step))
    case.program(program)
    return pops.validate(case), frame


destination = Path(os.environ.get("POPS_API040_OUTPUT", "outputs/api040_m04"))
records = []
for n in RESOLUTIONS:
    parts = frequencies(n, velocity=VELOCITY, diffusivity=DIFFUSIVITY,
                        transverse_diffusivity=TRANSVERSE_DIFFUSIVITY)
    dt = SAFETY_FACTOR / parts["installed_2d"]
    if not (dt * parts["installed_2d"] <= SAFETY_FACTOR + 1.e-14
            and dt <= SAFETY_FACTOR / parts["physical_1d"]):
        raise RuntimeError("M04 authored step violates its combined frequency")
    validated, frame = author_case(dt)
    layout = Uniform(CartesianGrid(frame=frame, cells=(n, n),
                                   periodic=PeriodicAxes(frame.axes)))
    if os.environ.get("POPS_API040_M04_AUTHORING_ONLY") == "1":
        pops.resolve(validated, layout=layout)
        print("M04 public authoring validated and resolved:", n, dt, parts)
        continue
    initial_line = exact_cell_means(n, 0., velocity=VELOCITY,
                                    diffusivity=DIFFUSIVITY, amplitude=AMPLITUDE)
    exact_line = exact_cell_means(n, T_END, velocity=VELOCITY,
                                  diffusivity=DIFFUSIVITY, amplitude=AMPLITUDE)
    discrete_line = discrete_cell_means(n, T_END, dt, method=TEMPORAL_METHOD,
                                       velocity=VELOCITY, diffusivity=DIFFUSIVITY,
                                       amplitude=AMPLITUDE)
    initial = np.ascontiguousarray(np.broadcast_to(initial_line, (1, n, n)))
    exact = np.ascontiguousarray(np.broadcast_to(exact_line, (1, n, n)))
    # Native arrays are (component,y,x), so the last axis carries x.
    centers = (np.arange(n) + .5) / n
    point_initial = 1 + AMPLITUDE * np.cos(2 * np.pi * centers)
    center_average_gap = float(np.max(np.abs(point_initial - initial_line)))
    if center_average_gap <= 1.e-6:
        raise RuntimeError("M04 initial means are indistinguishable from center samples")
    compile_at = time.perf_counter()
    artifact = pops.compile(pops.resolve(validated, layout=layout))
    compile_seconds = time.perf_counter() - compile_at
    if artifact.resolved_dimension != 2:
        raise RuntimeError("M04 requires actual installed Dim=2")
    context = pops.ExecutionContext.mpi_world(artifact)
    world = context.communicator.handle
    simulation = pops.bind(artifact, initial_state={"tracer": initial},
                           resources={"execution_context": context})
    gathered_initial = np.asarray(simulation.state_global("tracer"))
    initial_failure = b""
    if world.rank == 0:
        if gathered_initial.size != initial.size:
            initial_failure = b"M04 initial gather has wrong size"
        else:
            gathered_initial = gathered_initial.reshape(initial.shape).copy()
            initial_max_error = float(np.max(np.abs(gathered_initial - initial)))
            if not np.isfinite(initial_max_error) or initial_max_error > CRITERIA["initial_max_error"]:
                initial_failure = b"M04 bound initial state differs from exact cell means"
    initial_failure = world.broadcast_bytes(initial_failure, root=0)
    if initial_failure:
        raise RuntimeError(initial_failure.decode())
    run_at = time.perf_counter()
    report = pops.run(simulation, t_end=T_END, max_steps=100_000)
    run_seconds = time.perf_counter() - run_at
    gathered_final = np.asarray(simulation.state_global("tracer"))
    publication_failure = b""
    if world.rank == 0:
        try:
            if gathered_final.size != initial.size:
                raise RuntimeError("M04 final gather has wrong size")
            actual = gathered_final.reshape(initial.shape).copy()
            destination.mkdir(parents=True, exist_ok=True)
            state_path = destination / ("state_%d.npz" % n)
            np.savez_compressed(state_path, initial=gathered_initial, final=actual,
                                exact=exact, discrete_exact=discrete_line,
                                time=simulation.time(), cells=n,
                                dt=dt, velocity=VELOCITY, diffusivity=DIFFUSIVITY,
                                transverse_diffusivity=TRANSVERSE_DIFFUSIVITY)
            # Compute acceptance exclusively from reopened actual state bytes.
            with np.load(state_path) as saved:
                observed = saved["final"]
                density_l1 = float(np.mean(np.abs(observed - saved["exact"])))
                discrete_error = float(np.max(np.abs(observed - saved["discrete_exact"])))
                mass_defect = float(abs(np.mean(observed) - np.mean(saved["initial"])))
                y_variation = float(np.max(np.abs(observed - observed[:, :1, :])))
                minimum, maximum = float(np.min(observed)), float(np.max(observed))
                saved_time = float(saved["time"])
            record = {"cells": [n, n], "time": saved_time, "fixed_dt": dt,
                      "frequency_parts": parts, "density_l1": density_l1,
                      "discrete_fourier_max_error": discrete_error,
                      "mass_defect": mass_defect, "y_invariance_max": y_variation,
                      "minimum": minimum, "maximum": maximum,
                      "initial_max_error": initial_max_error,
                      "center_average_gap": center_average_gap,
                      "accepted_steps": report.accepted_steps,
                      "compile_seconds": compile_seconds, "run_seconds": run_seconds,
                      "run_report": report.to_data(), "execution_context": context.to_data(),
                      "runtime_backend": context.backend.to_data(), "mpi_ranks": world.size,
                      "saved_state": state_path.name,
                      "saved_state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest()}
            records.append(record)
            accepted = (np.isfinite(observed).all()
                        and density_l1 <= CRITERIA["density_l1_max"][n]
                        and discrete_error <= CRITERIA["discrete_fourier_max_error"]
                        and mass_defect <= CRITERIA["mass_defect_max"]
                        and y_variation <= CRITERIA["y_invariance_max"]
                        and minimum >= CRITERIA["minimum_state"]
                        and maximum <= CRITERIA["maximum_state"]
                        and abs(saved_time - T_END) <= CRITERIA["time_error_max"]
                        and report.accepted_steps > 0)
            if not accepted:
                raise AssertionError("M04 predeclared criterion failed: " + receipt_json(record))
        except Exception as error:
            publication_failure = (type(error).__name__ + ": " + str(error)).encode()
    publication_failure = world.broadcast_bytes(publication_failure, root=0)
    if publication_failure:
        raise RuntimeError(publication_failure.decode())

if os.environ.get("POPS_API040_M04_AUTHORING_ONLY") != "1":
    final_failure = b""
    if world.rank == 0:
        try:
            errors = np.asarray([row["density_l1"] for row in records])
            orders = np.log2(errors[:-1] / errors[1:])
            accepted = (len(records) == len(RESOLUTIONS) and
                        bool(np.all(orders >= CRITERIA["minimum_observed_order"])))
            from pops import _pops  # Provenance only; evolution uses public APIs.
            native = Path(_pops.__file__).resolve()
            receipt = {"schema_version": 3, "case": "M04/W02",
                       "status": "passed" if accepted else "failed",
                       "scope": "Dim=2 authored diagonal tensor, y-invariant exact trajectory; not Dim=1",
                       "diffusion_variant": DIFFUSION_VARIANT,
                       "diffusion_tensor": [[DIFFUSIVITY, 0.], [0., TRANSVERSE_DIFFUSIVITY]],
                       "equation_2d": "u_t+a*u_x=Dx*u_xx+Dy*u_yy",
                       "reduced_equation": "u_t+a*u_x=D*u_xx for y-invariant state",
                       "spatial_method": "combined first-order Rusanov+two-point diffusion",
                       "temporal_method": TEMPORAL_METHOD,
                       "criteria": CRITERIA, "resolutions": RESOLUTIONS,
                       "observed_orders": orders.tolist(), "records": records,
                       "package_file": str(package), "package_version": pops.__version__,
                       "native_file": str(native),
                       "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
                       "abi_key": _pops.abi_key(), "prefix": sys.prefix,
                       "threads_requested": os.environ.get("POPS_THREADS", "1")}
            encoded_receipt = receipt_json(receipt)
            (destination / "receipt.json").write_text(encoded_receipt + "\n")
            print(encoded_receipt)
            if not accepted:
                final_failure = b"M04 error order below its predeclared threshold"
        except Exception as error:
            final_failure = (type(error).__name__ + ": " + str(error)).encode()
    final_failure = world.broadcast_bytes(final_failure, root=0)
    if final_failure:
        raise RuntimeError(final_failure.decode())
