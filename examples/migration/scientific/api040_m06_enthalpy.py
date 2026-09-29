#!/usr/bin/env python3
"""M06 homogeneous enthalpy accumulation and latent plateau (Dim=2 storage).

This closes only the corpus's homogeneous subcase. No Stefan front or heat
diffusion is inferred from this experiment. Run with installed PoPS and no
PYTHONPATH; POPS_API040_OUTPUT selects the receipt directory.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pops
from api040_m06_m13_oracles import enthalpy_exact, enthalpy_thermometer
from api040_m06_m13_receipts import receipt_json, sha256

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, where
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.params import RuntimeParam
from pops.time import FailRun, FixedDt


# H=c*T+L*f at T=Tm; the stored unknown is H, never an Euler update of T.
CAPACITY, LATENT, MELTING = 2., 3., 1.
DT = .125
CRITERIA = {"energy_max_error": 2.e-13, "temperature_max_error": 2.e-13,
            "phase_max_error": 2.e-13, "time_error": 1.e-14,
            "initial_max_error": 1.e-14, "phase_bound_slack": 1.e-14}
SCENARIOS = (("sensible", 1.5, .25, 1.),
             ("melting_plateau", 1.5, 4., .5),
             ("fully_melted", 1.5, 4., 1.),
             ("reverse_plateau", 5.5, -4., .5))
GRIDS = ((4, 4), (6, 8))


def _symbolic_thermometer(h, capacity, latent, melting):
    threshold = capacity * melting
    temperature = where(h < threshold, lambda: h / capacity,
                        lambda: where(h <= threshold + latent,
                                      lambda: 0*h + melting,
                                      lambda: (h - latent) / capacity))
    fraction = where(h < threshold, lambda: 0*h,
                     lambda: where(h <= threshold + latent,
                                   lambda: (h - threshold) / latent,
                                   lambda: 0*h + 1.))
    return temperature, fraction


def build_case(frame, *, name="M06_enthalpy", capacity=CAPACITY,
               latent=LATENT, melting=MELTING):
    enthalpy_thermometer(np.asarray([0.]), capacity=capacity,
                         latent=latent, melting=melting)  # fail before authoring
    model = pops.Model(name + "_physics", frame=frame)
    enthalpy = model.state("enthalpy", components=("H",))
    input_declaration = model.param(RuntimeParam("uniform_energy_input", default=0.))
    input_rate = model.value(input_declaration)
    drive = model.source("heat_input", on=enthalpy,
                         value=(0*enthalpy[0] + input_rate,))
    rate = model.rate("energy_balance", equation=ddt(enthalpy) == drive)
    plan = DiscretizationPlan()
    plan.rates.add(rate, StateStorage())
    case = pops.Case(name)
    block = case.block("material", model)
    case.numerics(plan, block=block)
    program = pops.Program(name + "_Euler_H")
    h = program.state(block[enthalpy])
    candidate = program.value("updated_enthalpy", h.n + program.dt*rate(h.n),
                              at=h.next.point)
    t_expr, f_expr = _symbolic_thermometer(candidate[0], capacity, latent, melting)
    temperature = program.value("temperature_from_H", (t_expr,), at=candidate.point)
    fraction = program.value("liquid_fraction_from_H", (f_expr,), at=candidate.point)
    # Materialize the derived relation in the native Program before publication.
    candidate = program.guard("liquid_fraction_lower", candidate,
                              program.min(fraction) >= 0., action=FailRun())
    candidate = program.guard("liquid_fraction_upper", candidate,
                              program.max(fraction) <= 1., action=FailRun())
    candidate = program.guard("finite_temperature", candidate,
                              program.max(temperature) < 1.e300, action=FailRun())
    candidate = program.guard("finite_temperature_lower", candidate,
                              program.min(temperature) > -1.e300, action=FailRun())
    program.commit(h.next, candidate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    return case, input_declaration


def _installed_identity():
    package = Path(pops.__file__).resolve()
    if "PYTHONPATH" in os.environ or not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("M06 native receipt requires installed PoPS and unset PYTHONPATH")
    from pops import _pops
    native = Path(_pops.__file__).resolve()
    return {"package_file": str(package), "package_version": pops.__version__,
            "native_file": str(native), "native_sha256": sha256(native),
            "abi_key": _pops.abi_key()}


def main():
    identity = _installed_identity()
    pops.set_threads(int(os.environ.get("POPS_THREADS", "1")))
    destination = Path(os.environ.get("POPS_API040_OUTPUT", "outputs/api040_m06"))
    rows = []
    for cells in GRIDS:
        upper = (1., 2.) if cells == GRIDS[0] else (2., 1.)
        frame = Rectangle("M06_domain_%d_%d" % cells, lower=(0., 0.),
                          upper=upper).frame(Cartesian2D())
        case, declaration = build_case(frame, name="M06_%d_%d" % cells)
        validated = pops.validate(case)
        parameter = validated.resolve(declaration)
        grid = CartesianGrid(frame=frame, cells=cells, periodic=PeriodicAxes(frame.axes))
        artifact = pops.compile(pops.resolve(validated, layout=Uniform(grid)))
        if artifact.resolved_dimension != 2:
            raise RuntimeError("M06 needs authentic Dim=2 native artifact")
        for label, h0, input_rate, t_end in SCENARIOS:
            initial = np.full((1, cells[1], cells[0]), h0, dtype=float)
            context = pops.ExecutionContext.mpi_world(artifact)
            world = context.communicator.handle
            simulation = pops.bind(artifact, initial_state={"material": initial},
                                   params={parameter: input_rate},
                                   resources={"execution_context": context})
            bound = np.asarray(simulation.state_global("material"))
            report = pops.run(simulation, t_end=t_end, max_steps=16)
            gathered = np.asarray(simulation.state_global("material"))
            failure = b""
            if world.rank == 0:
                try:
                    exact_h, _, _ = enthalpy_exact(initial, input_rate, t_end,
                                                    capacity=CAPACITY, latent=LATENT,
                                                    melting=MELTING)
                    if bound.size != initial.size or gathered.size != initial.size:
                        raise RuntimeError("incomplete native global state")
                    saved = destination / ("state_%d_%d_%s.npz" % (*cells, label))
                    destination.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(saved, initial=bound.reshape(initial.shape),
                                        final=gathered.reshape(initial.shape),
                                        exact_enthalpy=exact_h, time=simulation.time(),
                                        input_rate=input_rate, cells=cells)
                    with np.load(saved) as data:
                        actual_h = data["final"]
                        actual_t, actual_f = enthalpy_thermometer(
                            actual_h, capacity=CAPACITY, latent=LATENT, melting=MELTING)
                        oracle_t, oracle_f = enthalpy_thermometer(
                            data["exact_enthalpy"], capacity=CAPACITY,
                            latent=LATENT, melting=MELTING)
                        metrics = {"initial_max_error": float(np.max(np.abs(data["initial"]-initial))),
                                   "energy_max_error": float(np.max(np.abs(actual_h-data["exact_enthalpy"]))),
                                   "temperature_max_error": float(np.max(np.abs(actual_t-oracle_t))),
                                   "phase_max_error": float(np.max(np.abs(actual_f-oracle_f))),
                                   "phase_min": float(actual_f.min()),
                                   "phase_max": float(actual_f.max()),
                                   "time_error": abs(float(data["time"])-t_end)}
                    accepted = (all(metrics[key] <= CRITERIA[key] for key in
                                    ("initial_max_error", "energy_max_error",
                                     "temperature_max_error", "phase_max_error", "time_error"))
                                and metrics["phase_min"] >= -CRITERIA["phase_bound_slack"]
                                and metrics["phase_max"] <= 1.+CRITERIA["phase_bound_slack"]
                                and report.accepted_steps == round(t_end/DT)
                                and report.rejected_steps == 0)
                    rows.append({"cells": cells, "domain_upper": upper,
                                 "label": label, "initial_h": h0,
                                 "input_rate": input_rate, "t_end": t_end,
                                 "metrics": metrics, "accepted": accepted,
                                 "accepted_steps": report.accepted_steps,
                                 "saved_state": saved.name, "saved_state_sha256": sha256(saved),
                                 "run_report": report.to_data(),
                                 "execution_context": context.to_data()})
                except Exception as error:
                    failure = (type(error).__name__ + ": " + str(error)).encode()
            failure = world.broadcast_bytes(failure, root=0)
            if failure:
                raise RuntimeError(failure.decode())
    if world.rank == 0:
        receipt = {"schema_version": 1, "case": "M06", "status": "passed" if all(
            row["accepted"] for row in rows) else "failed", "scope": "homogeneous H accumulation only; no Stefan front",
            "equation": "dH/dt=input; H=2T+3f with T=1 on 0<f<1",
            "method": "native StateStorage/Program Euler on H", "dt": DT,
            "criteria": CRITERIA, "scenarios": rows, **identity}
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "result.json").write_text(receipt_json(receipt) + "\n")
        print(receipt_json(receipt))
        failure = b"" if receipt["status"] == "passed" else b"M06 scientific acceptance failed"
    else:
        failure = b""
    failure = world.broadcast_bytes(failure, root=0)
    if failure:
        raise RuntimeError(failure.decode())


if __name__ == "__main__":
    main()
