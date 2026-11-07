"""M08 Poisson: the Program CG against the native CG, in A B B A blocks (N9, ROM-90, SV-T1).

Spec 0.3 SV-T1 measures the gap between the reference CG and the native CG on M08 Poisson with
the same generated operator and rtol = 1e-10, on ROMEO, from 1 to 64 CPU ranks and 1 to 8 GPUs,
256² to 4096² cells per rank, with the A B B A harness. The threshold is 0.9 of native throughput
(OD9). This harness runs that comparison on the PoPS build of the current PYTHONPATH:

* ``native``: the M08 field problem -Δφ = q, periodic, mean-zero gauge, realized by the native
  ``CG(max_iter, rel_tol=rtol, abs_tol=0)`` of the field route (the prepared affine problem over
  the generated field operator, ``solve_cg`` with its SV8 gate);
* ``program``: the same equation solved by the CG that ``examples/api_v02/method_krylov.py``
  writes as a Program (``iterate`` and ``dot_all``), stopped at ||r|| <= rtol ||b|| as the native
  request is, with x0 = 0.

Each accepted step solves once from the unchanged state q, so a step times one solve. Both routes
are compiled and bound first, warmed up, then advanced in interleaved A B B A blocks of fixed
accepted steps. The JSON output holds the median time per solve, the throughput of ``program``
against ``native`` (OD9: 0.9), the CG iterations of each route, and the difference of the two
potentials (both mean-zero). No time threshold is asserted.

Limits at this tree, which the output states in ``limits`` and which keep SV-T1 open:

* the Program CG applies the Program's Cartesian Laplacian (``program.laplacian`` on cell
  midpoints), not the generated field operator of the native route; the harness checks that both
  solve the same discrete equation (the potentials agree to the solve tolerance), not that they
  run the same kernel. The reference CG of ``pops.lib.solve`` on the generated operator (G10)
  replaces this route when it exists;
* a ``Uniform`` Cartesian layout lowers to one box (``single_box``): under MPI one rank owns every
  cell, so only one rank measures a per-rank size. The multi-rank rows of SV-T1 need a decomposed
  layout.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROUTES = ("native", "program")


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cells", help="NXxNY, for example 1024x1024")
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=2, help="accepted steps (solves) per block")
    parser.add_argument("--blocks", type=int, default=3, help="A B B A repetitions")
    parser.add_argument("--warmup", type=int, default=1, help="steps per route before timing")
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def _source_revision():
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                             capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                    check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"sha": "unknown", "dirty": None}
    return {"sha": sha, "dirty": dirty}


def _load(cells):
    """The M08 density q of examples/api_v02/models_M01_M14.py:build_m08, mean zero."""
    import numpy as np
    x, y = np.meshgrid(2*np.pi*(np.arange(cells[0]) + .5)/cells[0],
                       2*np.pi*(np.arange(cells[1]) + .5)/cells[1])
    q = np.sinc(1/cells[0])*np.sin(x) + .3*np.sinc(2/cells[1])*np.cos(2*y)
    return (q - q.mean())[None]


def _model(name, cells):
    import pops
    from examples.api_v02 import models_M01_M14 as cases
    frame, layout = cases._domain(name + "_periodic", cells, lengths=(2*3.141592653589793,)*2)
    model = pops.Model(name + "_model", frame=frame)
    U = model.state("density", components=("q",))
    case = pops.Case(name)
    block = case.block("fluid", model)
    return pops, cases, frame, layout, model, U, case, block


def build_native(cells, rtol, max_iterations, dt=1.):
    pops, cases, frame, layout, model, U, case, block = _model("N9_M08_native", cells)
    from pops import math as pm
    from pops.fields import (CellCenteredSecondOrder, FieldBoundary, FieldDiscretization,
                             FieldProblem, SharedMeanGauge, bcs)
    from pops.solvers import CG
    from pops.time import FailRun, FixedDt
    phi = model.field("potential")
    problem = FieldProblem("M08_Poisson", unknowns=(phi,), equations=(-pm.laplacian(phi) == U[0],),
        boundaries=(FieldBoundary(phi, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(),
                                                               bcs.Periodic())),),
        gauge=SharedMeanGauge((phi,)))
    field = case.field(problem, FieldDiscretization(method=CellCenteredSecondOrder(), boundaries=(),
        solver=CG(max_iter=max_iterations, rel_tol=rtol, abs_tol=0.)))
    program = pops.Program("N9_M08_native_CG")
    temporal = program.state(block[U])
    q = temporal.n
    point = program.stage("solve", c=0)
    observation = field.observe(program.solve(field, values={block[U]: q}, at=point)
                                .consume(action=FailRun()))
    program.store_history("potential", observation[field[phi]], depth=1)
    program.commit(temporal.next, program.value("unchanged", q, at=temporal.next.point))
    program.step_strategy(FixedDt(dt))
    case.program(program)
    return case, layout, cases._initial(case, block[U], _load(cells)), dt


def build_program(cells, rtol, max_iterations, dt=1.):
    import numpy as np
    pops, cases, frame, layout, model, U, case, block = _model("N9_M08_program", cells)
    from examples.api_v02.method_krylov import conjugate_gradient, laplacian
    from pops.time import FixedDt
    load = _load(cells)
    program = pops.Program("N9_M08_program_CG")
    temporal = program.state(block[U])
    rhs = program.value("rhs", temporal.n, at=temporal.next.point)

    def minus_laplacian(builder, value):
        applied = laplacian(builder, value, name="M08_laplacian")
        return builder.value("M08_A", (-applied[0],), at=value.point)

    # The native request stops at ||b - A x|| <= rtol ||b - A(0)|| = rtol ||b||; this CG stops on
    # its recursive residual at the same bound (method_krylov tests ||r||² < tolerance²).
    tolerance = rtol*float(np.sqrt(np.sum(load*load)))
    potential = conjugate_gradient(program, rhs, minus_laplacian, name="M08_CG",
                                   tolerance=tolerance, max_steps=max_iterations)
    program.store_history("potential", potential, depth=1)
    program.commit(temporal.next, program.value("unchanged", temporal.n, at=temporal.next.point))
    program.step_strategy(FixedDt(dt))
    case.program(program)
    return case, layout, cases._initial(case, block[U], load), dt


def main():
    args = _arguments()
    import numpy as np
    import pops
    cells = tuple(int(count) for count in args.cells.split("x"))
    builders = {"native": build_native, "program": build_program}
    runtimes, dts, compile_seconds, communicator = {}, {}, {}, None
    for route in ROUTES:
        case, layout, initial, dt = builders[route](cells, args.rtol, args.max_iterations)
        plan = pops.resolve(pops.validate(case), layout=layout)
        start = time.perf_counter()
        artifact = pops.compile(plan)
        compile_seconds[route] = time.perf_counter() - start
        communicator = artifact.platform_manifest.communicator.require("N9 M08 communicator")
        runtimes[route] = pops.bind(artifact, initial_values=dict(initial))
        dts[route] = dt

    def advance(route, steps):
        runtime = runtimes[route]
        start = time.perf_counter()
        report = pops.run(runtime, t_end=runtime.time() + steps*dts[route], max_steps=steps + 1,
                          console=False)
        return time.perf_counter() - start, report.accepted_steps

    for route in ROUTES:
        advance(route, args.warmup)
    samples = {route: [] for route in ROUTES}
    order = [route for _ in range(args.blocks) for route in ROUTES + ROUTES[::-1]]
    for route in order:
        elapsed, accepted = advance(route, args.steps)
        if accepted != args.steps:
            raise RuntimeError("%s took %d steps in a block of %d" % (route, accepted, args.steps))
        samples[route].append(elapsed/accepted)

    potentials = {route: runtimes[route].history_global("potential", 1) for route in ROUTES}
    owner = all(value is not None for value in potentials.values())
    result = {"schema": "pops.n9.m08_gap.v1", "cells": cells, "rtol": args.rtol,
              "max_iterations": args.max_iterations, "routes": list(ROUTES),
              "order": "".join("A" if route == "native" else "B" for route in order),
              "steps_per_block": args.steps, "blocks": args.blocks, "warmup": args.warmup,
              "communicator": communicator, "ranks": int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                                                                         os.environ.get("PMI_SIZE", 1))),
              "threads": os.environ.get("OMP_NUM_THREADS"), "host": platform.node(),
              "slurm_job": os.environ.get("SLURM_JOB_ID"),
              "processor": platform.processor() or platform.machine(),
              "source": _source_revision(), "compile_seconds": compile_seconds,
              "limits": ["program route applies program.laplacian, not the generated field "
                         "operator (G10 reference pending)",
                         "Uniform Cartesian layout is single_box: one rank owns every cell"]}
    timing = {}
    for route in ROUTES:
        timing[route] = {"median_s_per_solve": statistics.median(samples[route]),
                         "min_s_per_solve": min(samples[route]),
                         "max_s_per_solve": max(samples[route]), "samples": samples[route]}
    result["timing"] = timing
    result["throughput_program_vs_native"] = (timing["native"]["median_s_per_solve"]
                                              / timing["program"]["median_s_per_solve"])
    result["od9_threshold"] = 0.9
    if owner:
        native = np.asarray(potentials["native"], dtype=float).reshape(-1)
        program = np.asarray(potentials["program"], dtype=float).reshape(-1)
        program = program - program.mean()
        result["parity"] = {"linf": float(np.max(np.abs(native - program))),
                            "linf_native": float(np.max(np.abs(native))),
                            "native_mean": float(native.mean())}
    if pops_rank() == 0:
        args.out.write_text(json.dumps(result, indent=1))
        print(json.dumps({key: value for key, value in result.items() if key != "timing"}
                         | {"timing": {route: {key: value for key, value in row.items()
                                               if key != "samples"}
                                       for route, row in timing.items()}}, indent=1))


def pops_rank():
    return int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", 0)))


if __name__ == "__main__":
    main()
