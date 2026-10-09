#!/usr/bin/env python3
"""Native H05: normalized linear reservoirs, not full physical M22 radiation."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pops
from api040_m22_h05_oracle import backward_euler, continuous
from api040_receipts import receipt_json
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.solvers.nonlinear import LocalNewton
from pops.time import FailRun, FixedDt, LocalResidual

INITIAL, K, T_END = (2., .5), .8, .4
STEPS = (1, 2, 4, 8, 16)
CRITERIA = {"discrete_error": 2e-11, "inventory_error": 2e-11,
            "minimum_energy": 0., "initial_error": 1e-14, "time_error": 1e-14,
            "temporal_order_min": .8, "temporal_order_max": 1.1,
            "permutation_error": 2e-11}


def build_case(dt=.4, *, reverse=False):
    frame = Rectangle("H05_box", (0.,0.), (1.,1.)).frame(Cartesian2D())
    radiation = pops.Model("radiation_reservoir", frame=frame)
    material = pops.Model("matter_reservoir", frame=frame)
    energy = radiation.state("U", components=("value",))
    matter = material.state("U", components=("value",))
    coefficient = radiation.param(RuntimeParam("exchange_rate", default=K))
    radiation.source("exchange_parameter", on=energy, value=(radiation.value(coefficient),))
    case = pops.Case("M22_H05_normalized_linear_reservoirs")
    models = (("radiation", radiation), ("matter", material))
    blocks = {name: case.block(name, model) for name,model in (reversed(models) if reverse else models)}
    subjects = (blocks["radiation"][energy], blocks["matter"][matter])
    program = pops.Program("H05_backward_euler_original_equations")
    E,T = tuple(program.state(subject) for subject in subjects)
    k = program.source(radiation.module.operator_handle("exchange_parameter"), E.n)
    k = program.value("fixed_exchange_parameter", (0*E.n[0]+k[0],), at=E.n.point)
    k = program.guard("nonnegative_rate", k, program.min(k) >= 0, action=FailRun())
    e0 = program.guard("initial_radiation_domain", E.n, program.min(E.n) >= 0, action=FailRun())
    t0 = program.guard("initial_matter_domain", T.n, program.min(T.n) >= 0, action=FailRun())
    seeds = {"radiation": program.value("seed_E", e0, at=E.next.point),
             "matter": program.value("seed_T", t0, at=T.next.point)}

    def residual(P, z, *, old_E, old_T, rate):
        exchanged = P.dt*rate[0]*(z["radiation"][0]-z["matter"][0])
        return {"radiation": (z["radiation"][0]-old_E[0]+exchanged,),
                "matter": (z["matter"][0]-old_T[0]-exchanged,)}

    values = program.solve(LocalResidual(residual, seeds,
        captures={"old_E":e0,"old_T":t0,"rate":k}),
        solver=LocalNewton(tolerance=1e-12)).consume(action=FailRun())
    new_E = program.guard("radiation_domain", values[blocks["radiation"]],
                          program.min(values[blocks["radiation"]]) >= 0, action=FailRun())
    new_T = program.guard("matter_domain", values[blocks["matter"]],
                          program.min(values[blocks["matter"]]) >= 0, action=FailRun())
    program.commit_many({E.next:new_E,T.next:new_T})
    program.step_strategy(FixedDt(dt))
    case.program(program)
    for subject in subjects:
        case.initials.add(InitialCondition(state=subject,value=BindArray(),
                                          projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame,cells=(4,4),periodic=PeriodicAxes(frame.axes)))
    return case,layout,subjects,blocks["radiation"][coefficient]


def main():
    if os.environ.get("POPS_API040_M22_AUTHORING_ONLY") == "1":
        case,layout,_,_ = build_case(reverse=os.environ.get("POPS_API040_M22_REVERSE") == "1")
        pops.resolve(pops.validate(case),layout=layout)
        print("M22 H05 source authoring resolved; native not executed")
        return
    package = Path(pops.__file__).resolve()
    if "PYTHONPATH" in os.environ or not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("H05 native receipt requires installed PoPS with PYTHONPATH unset")
    from pops._native_selector import select_native_dimension
    select_native_dimension(2)
    pops.set_threads(int(os.environ.get("POPS_THREADS","1")))
    from pops import _pops
    import hashlib
    native = Path(_pops.__file__).resolve()
    destination = Path(os.environ.get("POPS_API040_OUTPUT","outputs/api040_m22_h05"))
    runs = []
    for reverse in (False,True):
        for count in STEPS:
            case,layout,subjects,coefficient = build_case(T_END/count,reverse=reverse)
            validated = pops.validate(case)
            artifact = pops.compile(pops.resolve(validated,layout=layout))
            if artifact.resolved_dimension != 2:
                raise RuntimeError("H05 requires the actual Dim=2 artifact")
            context = pops.ExecutionContext.mpi_world(artifact)
            world = context.communicator.handle
            initial = np.broadcast_to(np.array(INITIAL)[:,None,None],(2,4,4)).copy()
            simulation = pops.bind(artifact,
                initial_values={subject:initial[i:i+1] for i,subject in enumerate(subjects)},
                params={validated.resolve(coefficient):K},resources={"execution_context":context})
            bound = [np.asarray(simulation.state_global(name)) for name in ("radiation","matter")]
            report = pops.run(simulation,t_end=T_END,max_steps=count,console=False)
            final = [np.asarray(simulation.state_global(name)) for name in ("radiation","matter")]
            failure = b""
            if world.rank == 0:
                try:
                    destination.mkdir(parents=True,exist_ok=True)
                    path = destination/("state_%s_%d.npz" % ("reverse" if reverse else "canonical",count))
                    np.savez_compressed(path,initial=np.stack(bound).reshape(initial.shape),
                        final=np.stack(final).reshape(initial.shape),time=simulation.time(),
                        k=K,dt=T_END/count,steps=count)
                    with np.load(path) as saved:
                        start,actual = saved["initial"],saved["final"]
                        exact = backward_euler(start,float(saved["k"]),float(saved["dt"]),int(saved["steps"]))
                        metrics = {"discrete_error":float(np.max(np.abs(actual-exact))),
                            "inventory_error":float(np.max(np.abs(actual.sum(axis=0)-start.sum(axis=0)))),
                            "minimum_energy":float(actual.min()),
                            "initial_error":float(np.max(np.abs(start-initial))),
                            "time_error":abs(float(saved["time"])-T_END),
                            "continuous_error":float(np.max(np.abs(actual-continuous(start,K,T_END))))}
                    accepted = (all(metrics[key] <= CRITERIA[key] for key in
                        ("discrete_error","inventory_error","initial_error","time_error"))
                        and metrics["minimum_energy"] >= CRITERIA["minimum_energy"]
                        and report.accepted_steps == count and report.rejected_steps == 0)
                    runs.append({"reverse":reverse,"steps":count,"mpi_ranks":world.size,"metrics":metrics,"accepted":accepted,
                        "state_file":path.name,"state_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                        "run_report":report.to_data(),"execution_context":context.to_data(),
                        "semantic_identity":artifact.semantic_identity.token})
                except Exception as error:
                    failure = (type(error).__name__+": "+str(error)).encode()
            failure = world.broadcast_bytes(failure,root=0)
            if failure:
                raise RuntimeError(failure.decode())
    if world.rank == 0:
        try:
            orders = []
            for reverse in (False,True):
                errors = [row["metrics"]["continuous_error"] for row in runs if row["reverse"] == reverse]
                orders.extend(np.log2(np.array(errors[:-1])/errors[1:]).tolist())
            permutation_error = 0.
            for count in STEPS:
                with np.load(destination/("state_canonical_%d.npz" % count)) as a, \
                        np.load(destination/("state_reverse_%d.npz" % count)) as b:
                    permutation_error = max(permutation_error,float(np.max(np.abs(a["final"]-b["final"]))))
            passed = all(row["accepted"] for row in runs) and all(
                CRITERIA["temporal_order_min"] <= order <= CRITERIA["temporal_order_max"] for order in orders) \
                and permutation_error <= CRITERIA["permutation_error"]
            receipt = {"case":"M22/H05","status":"passed" if passed else "failed",
                "scope":"normalized linear reservoirs only; no spatial M1 or physical T^4 source",
                "equations":"E'=-k(E-T), T'=k(E-T)","criteria":CRITERIA,"orders":orders,"runs":runs,
                "permutation_error":permutation_error,
                "package_file":str(package),"native_file":str(native),
                "threads_requested":os.environ.get("POPS_THREADS","1"),
                "native_sha256":hashlib.sha256(native.read_bytes()).hexdigest(),"abi_key":_pops.abi_key()}
            (destination/"result.json").write_text(receipt_json(receipt)+"\n")
            failure = b"" if passed else b"H05 predeclared criteria failed"
        except Exception as error:
            failure = (type(error).__name__+": "+str(error)).encode()
    else:
        failure = b""
    failure = world.broadcast_bytes(failure,root=0)
    if failure:
        raise RuntimeError(failure.decode())


if __name__ == "__main__":
    main()
