#!/usr/bin/env python3
"""M17/W08: authored Fan--Li15 path-conservative transport on a 2D extrusion.

The physical data depend on x only. This is a true Dim=2 Uniform calculation,
not evidence for a Dim=1 specialization. No moment-specific numerical method or
native emitter is selected here. Run with the installed package and PYTHONPATH
unset; POPS_API040_M17_ORDER=reverse checks actual component permutation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div, maximum, minimum, sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.moments.fan_li import FAN_LI15_INDICES, fan_li15_expressions
from pops.numerics import (DiscretizationPlan, PathConservativeFiniteVolume,
                           SymbolicPath, reconstruction, riemann, variables)
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import FailRun, FixedDt

from api040_m17_oracle import (INDICES, TOP, gaussian_mixture, independent_path,
                               solve_reference)
from api040_receipts import receipt_json


# Prescribed corpus M17: N=16, eight SSPRK2 steps. The pinned reference N10
# instead ran N=4, dt=1e-4, t=4e-4; that historical check is not this case.
N = 16
DT = 1.e-4
STEPS = 8
T_END = STEPS * DT
GAUSS4 = ((.06943184420297371, .17392742256872692),
          (.33000947820757187, .32607257743127305),
          (.6699905217924281, .32607257743127305),
          (.9305681557970262, .17392742256872692))
CRITERIA = {"oracle_max_error": 3.e-8, "conserved_inventory": 3.e-13,
            "permutation_max_error": 1.e-12, "path_quadrature_gap": 3.e-8,
            "Gauss4_vs_Gauss48_saved_faces": 3.e-8,
            "nonconservative_max_min": 1.e-8, "y_invariance": 1.e-12,
            "initial_max_error": 1.e-13, "time_error": 1.e-14}


def order_from_environment() -> tuple[tuple[int, int], ...]:
    choice = os.environ.get("POPS_API040_M17_ORDER", "canonical")
    if choice not in ("canonical", "reverse"):
        raise ValueError("POPS_API040_M17_ORDER must be canonical or reverse")
    return INDICES if choice == "canonical" else tuple(reversed(INDICES))


def build_case(order: tuple[tuple[int, int], ...]):
    if len(order) != 15 or set(order) != set(FAN_LI15_INDICES):
        raise ValueError("M17 requires one exact permutation of the fifteen raw moments")
    frame = Rectangle("fan_li_periodic_square", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("authored_fan_li15", frame=frame)
    labels = tuple("M%d%d" % pq for pq in order)
    state = model.state("raw_moments", components=labels,
                        representation=Conservative(), space=CellState(frame=frame))
    slot = {pq: k for k, pq in enumerate(order)}
    canonical = tuple(state[slot[pq]] for pq in INDICES)
    physical = fan_li15_expressions(canonical)
    flux = model.flux("hermite_transport", state=state, frame=frame,
                      components={x_axis: tuple(physical.x[INDICES.index(pq)] for pq in order),
                                  y_axis: tuple(physical.y[INDICES.index(pq)] for pq in order)})

    def physical_matrix(direction):
        rows = physical.directional_nonconservative_matrix(direction)
        return tuple(tuple(rows[INDICES.index(pq)][INDICES.index(qr)] for qr in order)
                     for pq in order)

    product = model.nonconservative_product(
        "full_temperature_regularization", state=state,
        matrices={x_axis: physical_matrix((1, 0)), y_axis: physical_matrix((0, 1))},
        conservative_components=tuple("M%d%d" % pq for pq in order if sum(pq) < 4))

    def whole_path_speed(left, right, axis):
        # Published Fan--Li raw-second-moment majorant of the COMPLETE DF+B.
        # The second-moment/rho quotient along a straight raw segment is a
        # positive-weighted average of its endpoint quotients, so the maximum
        # endpoint bound controls every interior point. This is conditional on
        # rho>0 and SPD Theta, enforced at each authored SSPRK2 state below.
        second = (2, 0) if axis == 0 else (0, 2)
        rho = slot[0, 0]
        moment = slot[second]
        factor = sqrt(6 + sqrt(10))
        return factor * sqrt(maximum(left[moment] / left[rho],
                                     right[moment] / right[rho]))

    path = SymbolicPath(product, frame=frame, quadrature=GAUSS4, speed=whole_path_speed)
    rate = model.rate("complete_fan_li_balance", equation=ddt(state) == -div(flux) - product)

    numerics = DiscretizationPlan()
    numerics.rates.add(rate, PathConservativeFiniteVolume(
        flux=flux, path=path, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case = pops.Case("api040_M17_fan_li15")
    block = case.block("gas", model)
    case.numerics(numerics, block=block)

    # Explicit SSPRK2 authoring lets the domain guard run before both physical
    # evaluations and before the accepted endpoint, rather than only after a
    # ready-made Program has committed. The guard is collective and fatal; it
    # does not floor, project, or hide an invalid moment state.
    program = pops.Program("FanLi_SSPRK2_with_domain")
    temporal = program.state(block[state])

    def guarded(candidate, tag):
        raw = tuple(candidate[slot[pq]] for pq in INDICES)
        rho = raw[0]
        u, v = raw[1] / rho, raw[5] / rho
        a = raw[2] / rho - u*u
        b = raw[6] / rho - u*v
        c = raw[9] / rho - v*v
        margin = minimum(minimum(rho, a), minimum(c, a*c - b*b))
        margin_field = program.value(tag + "_domain_margin",
                                     tuple(margin if k == 0 else 0*candidate[k]
                                           for k in range(15)), at=candidate.point)
        return program.guard(tag + "_positive_SPD", candidate,
                             program.min(margin_field) > 0., action=FailRun())

    stage0 = program.stage("fan_li_stage_0", c=0)
    stage1 = program.stage("fan_li_stage_1", c=1)
    current = guarded(temporal.n, "current")
    k0 = program.value("fan_li_rhs_0", rate(current), at=stage0)
    first = program.value("fan_li_first_stage", temporal.n + program.dt*k0, at=stage1)
    first = guarded(first, "stage_1")
    k1 = program.value("fan_li_rhs_1", rate(first), at=stage1)
    endpoint = program.value("fan_li_ssprk2_endpoint",
                             temporal.n + .5*program.dt*(k0 + k1),
                             at=temporal.next.point)
    program.commit(temporal.next, guarded(endpoint, "endpoint"))
    program.step_strategy(FixedDt(DT))
    case.program(program)
    return case, frame


def initial_state(order, n=N):
    # Pinned reference examples/moments.py: mixture of two uncorrelated
    # Gaussians with rho_1=.55+.03*cos(2*pi*x), rho_2=1-rho_1.
    centers = (np.arange(n, dtype=float) + .5) / n
    weight = .55 + .03*np.sinc(1./n)*np.cos(2*np.pi*centers)
    first = np.asarray(gaussian_mixture(((1., (.15, -.1), (.8, 0., 1.1)),)))
    second = np.asarray(gaussian_mixture(((1., (-.2, .15), (1.2, 0., .9)),)))
    line = first[:, None]*weight[None, :] + second[:, None]*(1-weight[None, :])
    permuted = line[[INDICES.index(pq) for pq in order]]
    return np.ascontiguousarray(np.broadcast_to(permuted[:, None, :], (15, n, n)))


def main():
    package = Path(pops.__file__).resolve()
    if "PYTHONPATH" in os.environ or not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("M17 reception requires installed PoPS with PYTHONPATH unset")
    pops.set_threads(int(os.environ.get("POPS_THREADS", "1")))
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    oracle_sha256 = hashlib.sha256(Path(sys.modules["api040_m17_oracle"].__file__).read_bytes()).hexdigest()
    order = order_from_environment()
    case, frame = build_case(order)
    validated = pops.validate(case)
    layout = Uniform(CartesianGrid(frame=frame, cells=(N, N),
                                   periodic=PeriodicAxes(frame.axes)))
    if os.environ.get("POPS_API040_M17_AUTHORING_ONLY") == "1":
        pops.resolve(validated, layout=layout)
        print("M17 public authoring validated/resolved", order)
        return
    started = time.perf_counter()
    artifact = pops.compile(pops.resolve(validated, layout=layout))
    compile_seconds = time.perf_counter() - started
    if artifact.resolved_dimension != 2:
        raise RuntimeError("M17 requires the actual Dim=2 artifact")
    context = pops.ExecutionContext.mpi_world(artifact)
    world = context.communicator.handle
    initial = initial_state(order)
    simulation = pops.bind(artifact, initial_state={"gas": initial},
                           resources={"execution_context": context})
    gathered_initial = np.asarray(simulation.state_global("gas"))
    failure = b""
    if world.rank == 0:
        if gathered_initial.size != initial.size:
            failure = b"M17 initial gather has wrong shape"
        elif np.max(np.abs(gathered_initial.reshape(initial.shape)-initial)) > CRITERIA["initial_max_error"]:
            failure = b"M17 bound initial state differs from analytic cell averages"
    failure = world.broadcast_bytes(failure, root=0)
    if failure:
        raise RuntimeError(failure.decode())
    started = time.perf_counter()
    report = pops.run(simulation, t_end=T_END, max_steps=STEPS)
    run_seconds = time.perf_counter() - started
    gathered_final = np.asarray(simulation.state_global("gas"))
    destination = Path(os.environ.get("POPS_API040_OUTPUT", "outputs/api040_m17")) / (
        "reverse" if order != INDICES else "canonical")
    failure = b""
    if world.rank == 0:
        try:
            if gathered_final.size != initial.size:
                raise RuntimeError("M17 final gather has wrong shape")
            actual = gathered_final.reshape(initial.shape).copy()
            destination.mkdir(parents=True, exist_ok=True)
            state_path = destination / "state.npz"
            np.savez_compressed(state_path, initial=gathered_initial.reshape(initial.shape),
                                final=actual, time=simulation.time(), cells=N,
                                order=np.asarray(order, dtype=np.int64))
            with np.load(state_path) as saved:
                saved_initial, saved_final = saved["initial"], saved["final"]
                saved_time = float(saved["time"])
                canonical_initial = saved_initial[[order.index(pq) for pq in INDICES], 0, :]
                canonical_final = saved_final[[order.index(pq) for pq in INDICES], 0, :]
                oracle = solve_reference(canonical_initial, T_END)
                error = float(np.max(np.abs(canonical_final-oracle)))
                conserved = [k for k, pq in enumerate(INDICES) if k not in TOP]
                balance = float(np.max(np.abs(canonical_final[conserved].mean(axis=1)
                                              - canonical_initial[conserved].mean(axis=1))))
                y_invariance = float(np.max(np.abs(saved_final-saved_final[:, :1, :])))
                right = np.roll(canonical_initial, -1, axis=1)
                path4 = independent_path(canonical_initial, right, (1., 0.), points=4)
                path24 = independent_path(canonical_initial, right, (1., 0.), points=24)
                path48 = independent_path(canonical_initial, right, (1., 0.), points=48)
                quadrature_gap = float(np.max(np.abs(path24-path48)))
                final_right = np.roll(canonical_final, -1, axis=1)
                final_path4 = independent_path(canonical_final, final_right,
                                               (1., 0.), points=4)
                final_path48 = independent_path(canonical_final, final_right,
                                                (1., 0.), points=48)
                native_rule_gap = float(max(np.max(np.abs(path4-path48)),
                                            np.max(np.abs(final_path4-final_path48))))
                nonconservative = float(np.max(np.abs(path48)))
            other_order = tuple(reversed(INDICES)) if order == INDICES else INDICES
            other_label = "reverse" if order == INDICES else "canonical"
            peer_path = destination.parent / other_label / "state.npz"
            permutation_error = None
            if peer_path.exists():
                from pops import _pops
                current_native = Path(_pops.__file__).resolve()
                peer_receipt_path = peer_path.with_name("result.json")
                peer_receipt = json.loads(peer_receipt_path.read_text())
                if (peer_receipt.get("status") != "passed"
                        or peer_receipt.get("state_sha256") != hashlib.sha256(peer_path.read_bytes()).hexdigest()
                        or peer_receipt.get("native_sha256") != hashlib.sha256(current_native.read_bytes()).hexdigest()
                        or peer_receipt.get("script_sha256") != script_sha256
                        or peer_receipt.get("oracle_sha256") != oracle_sha256
                        or peer_receipt.get("package_file") != str(package)
                        or tuple(tuple(pq) for pq in peer_receipt.get("order", ())) != other_order
                        or peer_receipt.get("grid") != [N, N]
                        or peer_receipt.get("steps") != STEPS
                        or peer_receipt.get("dt") != DT):
                    raise RuntimeError("M17 permutation counterpart lacks matching authenticated receipt")
                with np.load(peer_path) as peer:
                    peer_order = tuple(tuple(int(v) for v in row) for row in peer["order"])
                    if peer_order != other_order or peer["final"].shape != initial.shape:
                        raise RuntimeError("M17 counterpart has wrong component order or shape")
                    peer_initial = peer["initial"][[peer_order.index(pq) for pq in INDICES]]
                    if np.max(np.abs(peer_initial-saved_initial[
                            [order.index(pq) for pq in INDICES]])) > CRITERIA["initial_max_error"]:
                        raise RuntimeError("M17 counterpart has different physical initial state")
                    peer_canonical = peer["final"][[peer_order.index(pq) for pq in INDICES]]
                    permutation_error = float(np.max(np.abs(
                        saved_final[[order.index(pq) for pq in INDICES]]-peer_canonical)))
            accepted = (report.accepted_steps == STEPS
                        and abs(saved_time-T_END) < CRITERIA["time_error"]
                        and error < CRITERIA["oracle_max_error"]
                        and balance < CRITERIA["conserved_inventory"]
                        and quadrature_gap < CRITERIA["path_quadrature_gap"]
                        and native_rule_gap < CRITERIA["Gauss4_vs_Gauss48_saved_faces"]
                        and nonconservative > CRITERIA["nonconservative_max_min"]
                        and y_invariance < CRITERIA["y_invariance"]
                        and (permutation_error is None
                             or permutation_error < CRITERIA["permutation_max_error"]))
            from pops import _pops
            native = Path(_pops.__file__).resolve()
            receipt = {"schema_version": 1, "case": "M17/W08", "status": "passed" if accepted else "failed",
                       "scope": "Uniform Dim=2 x-dependent extrusion; no Dim=1 or AMR claim",
                       "grid": [N, N], "steps": STEPS, "dt": DT, "final_time": saved_time,
                       "order": order, "criteria": CRITERIA,
                       "error_to_DOP853_Gauss24": error, "ten_line_inventory_defect": balance,
                       "nonconservative_path_max": nonconservative,
                       "Gauss24_48_path_gap": quadrature_gap, "y_invariance": y_invariance,
                       "Gauss4_48_saved_face_path_gap": native_rule_gap,
                       "permutation_error_if_counterpart_saved": permutation_error,
                       "permutation_verified": permutation_error is not None,
                       "run_report": report.to_data(), "execution_context": context.to_data(),
                       "runtime_backend": context.backend.to_data(), "mpi_ranks": world.size,
                       "threads_requested": os.environ.get("POPS_THREADS", "1"),
                       "package_file": str(package), "package_version": pops.__version__,
                       "script_sha256": script_sha256, "oracle_sha256": oracle_sha256,
                       "native_file": str(native),
                       "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
                       "abi_key": _pops.abi_key(), "state_file": state_path.name,
                       "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
                       "compile_seconds": compile_seconds, "run_seconds": run_seconds}
            (destination / "result.json").write_text(receipt_json(receipt) + "\n")
            print(receipt_json(receipt))
            if not accepted:
                failure = b"M17 predeclared scientific criteria failed; inspect saved states"
        except Exception as error:
            failure = (type(error).__name__ + ": " + str(error)).encode()
    failure = world.broadcast_bytes(failure, root=0)
    if failure:
        raise RuntimeError(failure.decode())


if __name__ == "__main__":
    main()
