#!/usr/bin/env python3
"""M13 homogeneous A→B→C network with an exact matrix-exponential oracle.

This is the local chemistry subcase only. No streamer transport, Poisson field,
avalanche front, nonlocal radiation or photon budget is claimed here.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pops
from api040_m06_m13_oracles import chain_exact
from api040_m06_m13_receipts import receipt_json, sha256

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.params import RuntimeParam
from pops.time import FailRun, FixedDt


DT, T_END = .002, 1.
CRITERIA = {"state_max_error": 1.e-6, "inventory_max_error": 2.e-12,
            "minimum_species": -2.e-13, "initial_max_error": 1.e-14,
            "time_error": 1.e-14, "cell_mean_gap_min": 1.e-5}
RATES = (("forward", 1.3, .4), ("reversed_rates", .4, 1.3),
         ("equal_rates", 1., 1.), ("zero_rates", 0., 0.))
CONFIGURATIONS = (((4, 4), ("A", "B", "C")),
                  ((6, 8), ("C", "B", "A")))


def cell_mean_initial(cells, order):
    """Exact x-cell integrals of three nonnegative periodic smooth species."""
    nx, ny = cells
    center = (np.arange(nx) + .5)/nx
    average = np.sinc(1./nx)
    lines = {"A": 1. + .1*average*np.cos(2*np.pi*center),
             "B": .2 + .05*average*np.sin(2*np.pi*center),
             "C": np.full(nx, .1)}
    return np.ascontiguousarray(np.broadcast_to(
        np.stack([lines[name] for name in order])[:, None, :], (3, ny, nx)))


def build_case(frame, order=("A", "B", "C"), *, name="M13_chain"):
    if tuple(sorted(order)) != ("A", "B", "C"):
        raise ValueError("order must permute A,B,C")
    slot = {name: order.index(name) for name in order}
    model = pops.Model(name + "_physics", frame=frame)
    species = model.state("species", components=order)
    ab_decl = model.param(RuntimeParam("rate_A_to_B", default=1.))
    bc_decl = model.param(RuntimeParam("rate_B_to_C", default=1.))
    kab, kbc = model.value(ab_decl), model.value(bc_decl)
    a, b = species[slot["A"]], species[slot["B"]]
    physical_rhs = {"A": -kab*a, "B": kab*a-kbc*b, "C": kbc*b}
    reaction = model.source("stoichiometric_A_B_C", on=species,
                            value=tuple(physical_rhs[name] for name in order))
    model.source("rate_parameters", on=species,
                 value=(kab+0*a, kbc+0*a, 0*a))
    rate = model.rate("local_chemistry", equation=ddt(species) == reaction)
    plan = DiscretizationPlan()
    plan.rates.add(rate, StateStorage())
    case = pops.Case(name)
    block = case.block("reactor", model)
    case.numerics(plan, block=block)

    program = pops.Program(name + "_SSPRK2")
    temporal = program.state(block[species])

    def guard(candidate, tag):
        for component in range(3):
            probe = program.value(tag + "_species_%d" % component,
                                  tuple(candidate[component] if i == 0 else
                                        0*candidate[i] for i in range(3)),
                                  at=candidate.point)
            candidate = program.guard(tag + "_nonnegative_%d" % component,
                                      candidate, program.min(probe) >= -1.e-13,
                                      action=FailRun())
        return candidate

    # Runtime parameters are rebound against this artifact. Their domain is
    # checked in the Program before evaluating either physical reaction RHS.
    param_probe_ab = program.source(model.module.operator_handle("rate_parameters"),
                                    temporal.n)
    param_probe_bc = program.value("rate_B_to_C_domain",
                                   (0*temporal.n[0]+param_probe_ab[1],
                                    0*temporal.n[1], 0*temporal.n[2]),
                                   at=temporal.n.point)
    current = program.guard("nonnegative_rate_AB", temporal.n,
                            program.min(param_probe_ab) >= 0., action=FailRun())
    current = program.guard("nonnegative_rate_BC", current,
                            program.min(param_probe_bc) >= 0., action=FailRun())
    current = guard(current, "current")
    stage0 = program.stage("reaction_stage_0", c=0)
    stage1 = program.stage("reaction_stage_1", c=1)
    rhs0 = program.value("reaction_rhs_0", rate(current), at=stage0)
    first = program.value("reaction_first", temporal.n + program.dt*rhs0,
                          at=stage1)
    first = guard(first, "first")
    rhs1 = program.value("reaction_rhs_1", rate(first), at=stage1)
    end = program.value("reaction_endpoint",
                        temporal.n + .5*program.dt*(rhs0+rhs1),
                        at=temporal.next.point)
    program.commit(temporal.next, guard(end, "endpoint"))
    program.step_strategy(FixedDt(DT))
    case.program(program)
    return case, (ab_decl, bc_decl)


def _identity():
    package = Path(pops.__file__).resolve()
    if "PYTHONPATH" in os.environ or not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("M13 native receipt requires installed PoPS and unset PYTHONPATH")
    from pops import _pops
    native = Path(_pops.__file__).resolve()
    return {"package_file": str(package), "package_version": pops.__version__,
            "native_file": str(native), "native_sha256": sha256(native),
            "abi_key": _pops.abi_key()}


def main():
    identity = _identity()
    pops.set_threads(int(os.environ.get("POPS_THREADS", "1")))
    destination = Path(os.environ.get("POPS_API040_OUTPUT", "outputs/api040_m13"))
    rows = []
    for cells, order in CONFIGURATIONS:
        upper = (1., 2.) if cells == CONFIGURATIONS[0][0] else (2., 1.)
        frame = Rectangle("M13_domain_%d_%d" % cells, lower=(0., 0.),
                          upper=upper).frame(Cartesian2D())
        case, declarations = build_case(frame, order, name="M13_%d_%d" % cells)
        validated = pops.validate(case)
        parameters = tuple(validated.resolve(item) for item in declarations)
        artifact = pops.compile(pops.resolve(validated, layout=Uniform(
            CartesianGrid(frame=frame, cells=cells,
                          periodic=PeriodicAxes(frame.axes)))))
        if artifact.resolved_dimension != 2:
            raise RuntimeError("M13 needs authentic Dim=2 native artifact")
        initial = cell_mean_initial(cells, order)
        point_x = (np.arange(cells[0])+.5)/cells[0]
        gap = float(np.max(np.abs(initial[order.index("A"), 0] -
                                  (1.+.1*np.cos(2*np.pi*point_x)))))
        if gap <= CRITERIA["cell_mean_gap_min"]:
            raise RuntimeError("cell means are indistinguishable from center samples")
        for label, kab, kbc in RATES:
            context = pops.ExecutionContext.mpi_world(artifact)
            world = context.communicator.handle
            simulation = pops.bind(artifact, initial_state={"reactor": initial.copy()},
                                   params=dict(zip(parameters, (kab, kbc))),
                                   resources={"execution_context": context})
            bound = np.asarray(simulation.state_global("reactor"))
            report = pops.run(simulation, t_end=T_END, max_steps=600)
            gathered = np.asarray(simulation.state_global("reactor"))
            failure = b""
            if world.rank == 0:
                try:
                    if bound.size != initial.size or gathered.size != initial.size:
                        raise RuntimeError("incomplete native global state")
                    reorder = [order.index(name) for name in ("A", "B", "C")]
                    exact_abc = chain_exact(initial[reorder], kab, kbc, T_END)
                    exact = exact_abc[[ ("A", "B", "C").index(name) for name in order ]]
                    destination.mkdir(parents=True, exist_ok=True)
                    saved = destination / ("state_%d_%d_%s.npz" % (*cells, label))
                    np.savez_compressed(saved, initial=bound.reshape(initial.shape),
                                        final=gathered.reshape(initial.shape), exact=exact,
                                        time=simulation.time(), rates=(kab,kbc), cells=cells,
                                        order=order)
                    with np.load(saved) as data:
                        actual, expected = data["final"], data["exact"]
                        metrics = {"initial_max_error": float(np.max(np.abs(data["initial"]-initial))),
                                   "state_max_error": float(np.max(np.abs(actual-expected))),
                                   "inventory_max_error": float(np.max(np.abs(
                                       actual.sum(axis=0)-data["initial"].sum(axis=0)))),
                                   "minimum_species": float(actual.min()),
                                   "time_error": abs(float(data["time"])-T_END)}
                    accepted = (metrics["initial_max_error"] <= CRITERIA["initial_max_error"]
                                and metrics["state_max_error"] <= CRITERIA["state_max_error"]
                                and metrics["inventory_max_error"] <= CRITERIA["inventory_max_error"]
                                and metrics["minimum_species"] >= CRITERIA["minimum_species"]
                                and metrics["time_error"] <= CRITERIA["time_error"]
                                and report.accepted_steps == round(T_END/DT)
                                and report.rejected_steps == 0)
                    rows.append({"cells": cells, "domain_upper": upper,
                                 "order": order, "label": label,
                                 "rates": (kab,kbc), "cell_mean_gap": gap,
                                 "metrics": metrics, "accepted": accepted,
                                 "accepted_steps": report.accepted_steps,
                                 "mpi_ranks": world.size,
                                 "saved_state": saved.name, "saved_state_sha256": sha256(saved),
                                 "run_report": report.to_data(),
                                 "execution_context": context.to_data()})
                except Exception as error:
                    failure = (type(error).__name__ + ": " + str(error)).encode()
            failure = world.broadcast_bytes(failure, root=0)
            if failure:
                raise RuntimeError(failure.decode())
    if world.rank == 0:
        receipt = {"schema_version": 1, "case": "M13", "status": "passed" if all(
            row["accepted"] for row in rows) else "failed",
            "scope": "homogeneous local A→B→C chemistry; no streamer/radiation closure",
            "equation": "A'=-kAB*A; B'=kAB*A-kBC*B; C'=kBC*B",
            "method": "native StateStorage/Program SSPRK2", "dt": DT,
            "t_end": T_END, "criteria": CRITERIA, "runs": rows,
            "threads_requested": os.environ.get("POPS_THREADS", "1"), **identity}
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "result.json").write_text(receipt_json(receipt)+"\n")
        print(receipt_json(receipt))
        failure = b"" if receipt["status"] == "passed" else b"M13 scientific acceptance failed"
    else:
        failure = b""
    failure = world.broadcast_bytes(failure, root=0)
    if failure:
        raise RuntimeError(failure.decode())


if __name__ == "__main__":
    main()
