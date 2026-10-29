#!/usr/bin/env python3
"""N3a (ROM-82): time per step of the native face and the Python face on M01 or M03.

Every route compiles the same builder, binds the same initial data, is warmed up, then advances
in interleaved A B B A blocks of ``--steps`` accepted steps; the wall time of a block divided by
its steps is one sample. With ``KOKKOS_TOOLS_LIBS`` naming ``kernel_timer.cpp`` built as a
shared library, each block is also split per kernel. After the last block every route has taken
the same steps, and the final states are compared bit for bit.

The script measures; it asserts no time threshold. PF10(1) and OD9 compare the median
``throughput_vs_<first route>`` with 0.9.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROUTES = ("native_rusanov", "python_rusanov", "native_hll", "python_hll_m01")


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", choices=("M01", "M03"))
    parser.add_argument("cells", help="NXxNY, for example 4096x4096")
    parser.add_argument("--routes", default="native_rusanov,python_rusanov",
                        help="comma-separated, first is the reference; from %s" % ", ".join(ROUTES))
    parser.add_argument("--steps", type=int, default=4, help="accepted steps per block")
    parser.add_argument("--blocks", type=int, default=3, help="A B B A repetitions")
    parser.add_argument("--warmup", type=int, default=2, help="steps per route before timing")
    parser.add_argument("--signed", action="store_true",
                        help="declare the signed wave speeds that the native HLL requires")
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def _bind(pops, artifact, spec):
    # One process with host threads: the final states are compared on this process.
    communicator = artifact.platform_manifest.communicator.require("N3a artifact communicator")
    if communicator != "serial":
        raise RuntimeError("the N3a harness runs serial artifacts, not %r" % communicator)
    return pops.bind(artifact, initial_values=dict(spec.initial_values))


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
    args = _arguments()
    routes = args.routes.split(",")
    unknown = sorted(set(routes) - set(ROUTES))
    if unknown or len(set(routes)) != len(routes) or len(routes) < 2:
        raise SystemExit("--routes needs at least two distinct routes from %s" % (ROUTES,))

    import numpy as np
    import pops
    from examples.api_v02 import models_M01_M14 as cases

    timer = None
    if os.environ.get("KOKKOS_TOOLS_LIBS"):
        timer = ctypes.CDLL(os.environ["KOKKOS_TOOLS_LIBS"])
    schemes = {"native_rusanov": cases.native_rusanov, "python_rusanov": cases.rusanov,
               "native_hll": cases.native_hll, "python_hll_m01": cases.m01_hll}
    builder = {"M01": cases.build_m01, "M03": cases.build_m03}[args.case]
    cells = tuple(int(count) for count in args.cells.split("x"))

    runtimes, specs, compile_seconds = {}, {}, {}
    for route in routes:
        spec = builder(cells=cells, scheme=schemes[route], signed_wave_speeds=args.signed)
        plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
        start = time.perf_counter()
        artifact = pops.compile(plan)
        compile_seconds[route] = time.perf_counter() - start
        runtimes[route], specs[route] = _bind(pops, artifact, spec), spec

    def advance(route, steps):
        runtime, spec = runtimes[route], specs[route]
        start = time.perf_counter()
        report = pops.run(runtime, t_end=runtime.time() + steps*spec.dt, max_steps=steps + 1,
                          console=False)
        return time.perf_counter() - start, report.accepted_steps

    for route in routes:
        advance(route, args.warmup)
    samples = {route: [] for route in routes}
    kernels = {route: {} for route in routes}
    dump = args.out.with_suffix(".kernels.tmp.json")
    for route in [route for _ in range(args.blocks) for route in routes + routes[::-1]]:
        if timer is not None:
            timer.n3a_timer_reset()
        elapsed, accepted = advance(route, args.steps)
        if accepted != args.steps:
            raise RuntimeError("%s took %d steps in a block of %d" % (route, accepted, args.steps))
        samples[route].append(elapsed/accepted)
        if timer is not None:
            if timer.n3a_timer_dump(str(dump).encode()) != 0:
                raise RuntimeError("kernel timer could not write " + str(dump))
            for row in json.loads(dump.read_text()):
                entry = kernels[route].setdefault(row["name"], {"kind": row["kind"], "seconds": 0.,
                                                                "calls": 0, "steps": 0})
                entry["seconds"] += row["seconds"]
                entry["calls"] += row["calls"]
            for entry in kernels[route].values():
                entry["steps"] = len(samples[route])*args.steps
    dump.unlink(missing_ok=True)

    states = {route: np.asarray(runtimes[route].state_global("fluid"), dtype=float)
              for route in routes}
    reference = states[routes[0]]
    parity = {}
    for route in routes[1:]:
        difference = states[route] - reference
        parity[route] = {"against": routes[0],
                         "bitwise_equal": bool(np.array_equal(states[route], reference)),
                         "l1": float(np.mean(np.abs(difference))),
                         "linf": float(np.max(np.abs(difference))),
                         "steps": int(runtimes[route].macro_step())}
    timing = {}
    for route in routes:
        timing[route] = {"median_s_per_step": statistics.median(samples[route]),
                         "min_s_per_step": min(samples[route]),
                         "max_s_per_step": max(samples[route]), "samples": samples[route]}
    reference_time = timing[routes[0]]["median_s_per_step"]
    for route in routes:
        timing[route]["throughput_vs_" + routes[0]] = (
            reference_time/timing[route]["median_s_per_step"])

    result = {"schema": "pops.n3a.face_routes.v1", "case": args.case, "cells": cells,
              "routes": routes, "steps_per_block": args.steps, "blocks": args.blocks,
              "warmup": args.warmup, "signed_wave_speeds": args.signed,
              "threads": os.environ.get("OMP_NUM_THREADS"), "host": platform.node(),
              "processor": platform.processor() or platform.machine(),
              "source": _source_revision(), "compile_seconds": compile_seconds,
              "timing": timing, "parity": parity, "kernels": kernels}
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"case": args.case, "cells": cells, "parity": parity,
                      "timing": {route: {key: value for key, value in row.items()
                                         if key != "samples"}
                                 for route, row in timing.items()}}, indent=1))


if __name__ == "__main__":
    main()
