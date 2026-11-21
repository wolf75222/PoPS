"""L1-d: the relation solves of cg() against the native witness, in A B B A blocks (N9, ROM-90).

The relation route of L1-d (advection-diffusion, ``imex_euler`` with ``solvers={imex: cg()}``)
solves its implicit diffusion with the reference of ``cg()``: a region of the Program that applies
the relation's operator, the StencilFlux of ``two_point()``, at each iterate. The witness is the
hand-written b6 Program of the Trace test: the same right-hand side, ``v - dt nu Lap(v)`` applied
by a field-level ``set_apply`` and solved by the native CG. Both are the builders of
``tests/python/integration/runtime/test_relation_solve_reference_n9_native.py``, whose Trace test
checks that they take the same iterations and reach the same states.

The driver's ruling of 13/11 (``n9/relation_solves/decisions_pilote.md``, 1(c)) bounds the
reference route at 10 times the native witness, at 64² and at 256²: ``ratio`` is the median time
per step of the reference over the native's, and ``within_bound`` reads it against ``bound``.

Both routes are compiled, bound and warmed up, then advanced by ``pops.run`` in interleaved
A B B A blocks of a fixed number of steps; a sample is the time of one block over its accepted
steps. The output also holds the iterations of the last step of each route.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROUTES = ("native", "reference")
BOUND = 10.0


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cells", type=int, help="cells per axis, for example 64 or 256")
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-14)
    parser.add_argument("--steps", type=int, default=10, help="accepted steps per block")
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


def main():
    arguments = _arguments()
    import numpy as np

    import pops
    from tests.python.integration.runtime import test_relation_solve_reference_n9_native as l1d
    from tests.python.support.native_execution_context import artifact_execution_context

    builders = {"native": l1d._witness, "reference": l1d._relation}
    cells = arguments.cells
    runtimes, compile_seconds = {}, {}
    for route in ROUTES:
        if "POPS_CACHE_DIR" not in os.environ:
            cache = tempfile.mkdtemp(prefix="n9-relation-cost-%s-" % route)
            os.environ["POPS_CACHE_DIR"] = os.environ["POPS_NATIVE_CACHE_DIR"] = cache
        started = time.perf_counter()
        artifact = builders[route](cells, arguments.rtol, arguments.atol)
        compile_seconds[route] = time.perf_counter() - started
        (binding,) = tuple(artifact.plan.initial_condition_plan.bindings)
        runtimes[route] = pops.bind(
            artifact, initial_values={binding.subject: np.ascontiguousarray(
                np.transpose(l1d._initial(cells))[np.newaxis, ...])},
            resources={"execution_context": artifact_execution_context(artifact)})

    def advance(route, steps):
        runtime = runtimes[route]
        started = time.perf_counter()
        report = pops.run(runtime, t_end=runtime.time() + steps * l1d.DT, console=False)
        return time.perf_counter() - started, report.accepted_steps

    for route in ROUTES:
        advance(route, arguments.warmup)
    order = ("native", "reference", "reference", "native")
    samples = {route: [] for route in ROUTES}
    for _ in range(arguments.blocks):
        for route in order:
            seconds, steps = advance(route, arguments.steps)
            samples[route].append(seconds / steps)
    median = {route: statistics.median(samples[route]) for route in ROUTES}
    iterations = {route: int(runtimes[route].program_report().diagnostics["cg_iterations"])
                  for route in ROUTES}
    ratio = median["reference"] / median["native"]
    document = {
        "schema": "pops.n9.relation_cost.v1",
        "cells": [cells, cells],
        "request": {"rtol": arguments.rtol, "atol": arguments.atol},
        "order": "".join("A" if route == "native" else "B" for route in order),
        "steps_per_block": arguments.steps,
        "blocks": arguments.blocks,
        "compile_seconds": compile_seconds,
        "timing": {route: {"median_seconds_per_step": median[route],
                           "min_seconds_per_step": min(samples[route]),
                           "max_seconds_per_step": max(samples[route]),
                           "samples": samples[route]} for route in ROUTES},
        "last_step_iterations": iterations,
        "ratio": ratio,
        "bound": BOUND,
        "within_bound": ratio <= BOUND,
        "threads": os.environ.get("OMP_NUM_THREADS"),
        "host": platform.node(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "source": _source_revision(),
    }
    arguments.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"cells": cells, "ratio": round(ratio, 2), "within_bound": ratio <= BOUND,
                      "median_seconds_per_step": median, "iterations": iterations}))


if __name__ == "__main__":
    main()
