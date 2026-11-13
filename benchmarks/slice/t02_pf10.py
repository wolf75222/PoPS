#!/usr/bin/env python3
"""PF10 of the vertical slice (OD31, T8): tutorial 02 written with the surface against master's.

``measure`` times one size at one thread count in one process. Route ``master`` is master's
tutorial 02 as written in ``docs/tutorials/scalar_advection`` (``riemann.ScalarUpwind``,
``MUSCL(VanLeer)``, SSPRK2 written in ``pops.Program``); route ``slice`` is the slice's case file
``examples/surface/tutorials/scalar_advection/02_openmp_explicit_ssprk2.py``. The bench builds
both cases on its side at ``N x N`` cells and never runs either file: it executes the statements of
each file that build the case, up to its ``pops.compile``, then compiles and binds them itself.
The first bind of the process is the slice's ``pops.bind(..., resources=pops.Resources(threads=n))``,
which fixes the threads of the process (spec, section 12); the bench checks that the run holds
them. Both routes then advance through the same driver, ``pops.step`` per accepted step, warmed
up, in interleaved ``A B B A`` blocks; the wall time of a block divided by its steps is one
sample. After the last block both routes have taken the same steps to the same time, and the
final states are compared (relative max error; the initial data differ by the rounding of exp,
cause C44, G07). With ``KOKKOS_TOOLS_LIBS`` naming N3a's ``kernel_timer.cpp`` built as a shared
library, each block is also split per kernel, for PF10 (2).

``verdict`` reads the measurements and states PF10 (1), the median throughput of the slice at
least 0.9 of master's at each size and thread count (OD9), and PF10 (3), the time per step of
each route growing at most 1.25 times the ratio of the cells when they quadruple (OD24; the step
is linear in the cells). The script asserts no other threshold.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import resource
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "docs" / "tutorials" / "scalar_advection" / "02_openmp_explicit_ssprk2.py"
SLICE = ROOT / "examples" / "surface" / "tutorials" / "scalar_advection" / \
    "02_openmp_explicit_ssprk2.py"
ROUTES = ("master", "slice")
SCHEMA = "pops.slice.t02_pf10.v1"
PF10_1 = 0.9            # OD9
PF10_3 = 1.25           # OD24: the measured growth exceeds the stated complexity by at most 25 %


def _once(source: str, old: str, new: str, where: Path) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError("%s: the statement %r holds %d times, not once" % (where, old, count))
    return source.replace(old, new)


def _prefix(where: Path, stop: str, substitutions=()) -> str:
    """The statements of ``where`` before ``stop``, with each substitution made exactly once."""
    source = where.read_text(encoding="utf-8")
    for old, new in substitutions:
        source = _once(source, old, new, where)
    head, _, _ = _once(source, stop, "\x00", where).partition("\x00")
    return head


def _execute(where: Path, source: str) -> dict:
    namespace = {"__name__": "pf10_" + where.stem, "__file__": str(where)}
    exec(compile(source, str(where), "exec"), namespace)
    return namespace


def build_slice(cells: int):
    """The slice's case at ``cells`` x ``cells``, without its outputs, compiled."""
    namespace = _execute(SLICE, _prefix(SLICE, "\ncase = pops.Case("))
    pops = namespace["pops"]
    case = pops.Case(model=namespace["model"], numerics=namespace["numerics"],
                     program=namespace["program"],
                     layout={namespace["domain"]: pops.Uniform(cells=(cells, cells))})
    return pops, pops.compile(case, target=pops.OpenMP())


def build_master(cells: int):
    """Master's tutorial 02 at ``cells`` x ``cells``, compiled; its initial state. The line that
    fixes the threads from ``POPS_THREADS`` is taken out: the bench fixes them."""
    namespace = _execute(MASTER, _prefix(
        MASTER, "\nsimulation = pops.bind(",
        (("\nNX = 64\n", "\nNX = %d\n" % cells), ("\nNY = 64\n", "\nNY = %d\n" % cells),
         ("\npops.set_threads(THREADS)\n", "\n"))))
    return namespace["artifact"], namespace["initial_state"]


def _source_revision():
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                             capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                    check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"sha": "unknown", "dirty": None}
    return {"sha": sha, "dirty": dirty}


def measure(args) -> int:
    import numpy as np

    import pops as b6

    timer = ctypes.CDLL(os.environ["KOKKOS_TOOLS_LIBS"]) \
        if os.environ.get("KOKKOS_TOOLS_LIBS") else None
    routes = ROUTES if args.route == "both" else (args.route,)
    compile_seconds, runtimes, threads = {}, {}, {"asked": args.threads}
    if "slice" in routes:
        start = time.perf_counter()
        surface, slice_artifact = build_slice(args.cells)
        compile_seconds["slice"] = time.perf_counter() - start
    if "master" in routes:
        start = time.perf_counter()
        master_artifact, initial_state = build_master(args.cells)
        compile_seconds["master"] = time.perf_counter() - start
    if "slice" in routes:
        simulation = surface.bind(slice_artifact,
                                  resources=surface.Resources(threads=args.threads),
                                  directory=str(args.out.parent))
        threads = simulation._threads
        if threads["running"] != args.threads:
            raise RuntimeError("the run holds %d threads, the bench asked %d: an execution space "
                               "started before the first bind" % (threads["running"],
                                                                  args.threads))
        runtimes["slice"] = simulation._simulation._runtime
    else:
        # master alone fixes the threads as its tutorial does, by pops.set_threads
        b6.set_threads(args.threads)
    if "master" in routes:
        runtimes["master"] = b6.bind(master_artifact, initial_state={"tracer": initial_state})
    from pops.runtime import _threading

    native = _threading._selected_native_module()
    concurrency = int(native.runtime_environment_report()["kokkos_concurrency"]) \
        if native is not None else None
    if concurrency != args.threads:
        raise RuntimeError("Kokkos runs %r threads, the bench asked %d" % (concurrency,
                                                                          args.threads))
    threads = dict(threads, running=concurrency, kokkos_concurrency=concurrency)
    blocks = {"slice": "scalar_advection", "master": "tracer"}   # the model's and master's

    def advance(route, steps):
        runtime = runtimes[route]
        start = time.perf_counter()
        for _ in range(steps):
            b6.step(runtime, t_bound=float(runtime.time()) + 1.0)
        return time.perf_counter() - start

    for route in routes:
        advance(route, args.warmup)
    samples = {route: [] for route in routes}
    kernels = {route: {} for route in routes}
    dump = args.out.with_suffix(".kernels.tmp.json")
    for route in [route for _ in range(args.blocks) for route in routes + routes[::-1]]:
        if timer is not None:
            timer.n3a_timer_reset()
        samples[route].append(advance(route, args.steps)/args.steps)
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

    clocks = {route: (int(runtimes[route].macro_step()), float(runtimes[route].time()))
              for route in routes}
    states = {route: np.asarray(runtimes[route].state_global(blocks[route]),
                                dtype=np.float64).reshape(args.cells, args.cells)
              for route in routes}
    result = {"schema": SCHEMA, "cells": [args.cells, args.cells], "routes": list(routes),
              "steps_per_block": args.steps, "blocks": args.blocks, "warmup": args.warmup,
              "threads": threads, "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
              "host": platform.node(), "processor": platform.processor() or platform.machine(),
              "cpus": os.cpu_count(), "source": _source_revision(),
              "compile_seconds": compile_seconds, "clocks": clocks,
              "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "samples": samples, "kernels": kernels}
    if len(routes) == 1:
        # one route per process: merge pairs the processes and compares their final states
        np.save(args.out.with_suffix(".state.npy"), states[routes[0]])
        args.out.write_text(json.dumps(result, indent=1))
        print(json.dumps({"route": routes[0], "clock": clocks[routes[0]],
                          "max_rss_kb": result["max_rss_kb"],
                          "median_s_per_step": statistics.median(samples[routes[0]])}))
        return 0
    result = _finish(result, states)
    args.out.write_text(json.dumps(result, indent=1))
    timing, parity = result["timing"], result["parity"]
    print(json.dumps({"cells": result["cells"], "threads": threads, "parity": parity,
                      "timing": {route: {key: value for key, value in row.items()
                                         if key != "samples"}
                                 for route, row in timing.items()}}, indent=1))
    return 0


def _finish(result: dict, states: dict) -> dict:
    """Timing and parity of a measurement holding the samples of both routes."""
    import numpy as np

    clocks = {route: tuple(clock) for route, clock in result["clocks"].items()}
    if clocks["slice"] != clocks["master"]:
        raise RuntimeError("the routes stopped at different clocks: %r" % (clocks,))
    difference = np.abs(states["slice"] - states["master"])
    result["parity"] = {"against": "master", "steps": clocks["master"][0],
                        "time": clocks["master"][1],
                        "bitwise_equal": bool(np.array_equal(states["slice"], states["master"])),
                        "cells_differing": int(np.count_nonzero(difference)),
                        "relative_max": float(difference.max()/np.abs(states["master"]).max())}
    samples, timing = result.pop("samples"), {}
    for route in ROUTES:
        timing[route] = {"median_s_per_step": statistics.median(samples[route]),
                         "min_s_per_step": min(samples[route]),
                         "max_s_per_step": max(samples[route]), "samples": samples[route]}
    for route in ROUTES:
        timing[route]["throughput_vs_master"] = (timing["master"]["median_s_per_step"]
                                                 / timing[route]["median_s_per_step"])
    result["timing"] = timing
    result["routes"] = list(ROUTES)
    return result


def merge(paths, out: Path) -> dict:
    """One measurement from processes that each ran one route at the same size and threads, in
    the order A B B A across processes; their final states are compared as in one process."""
    import numpy as np

    runs = [(Path(path), json.loads(Path(path).read_text(encoding="utf-8"))) for path in paths]
    keys = {(tuple(run["cells"]), run["threads"]["running"], run["steps_per_block"],
             run["blocks"], run["warmup"]) for _, run in runs}
    if len(keys) != 1 or any(len(run["routes"]) != 1 for _, run in runs):
        raise SystemExit("merge takes one-route measurements of one size, threads and protocol")
    merged = dict(runs[0][1], processes=[str(path) for path, _ in runs],
                  order=[run["routes"][0] for _, run in runs], samples={}, clocks={},
                  kernels={}, compile_seconds={})
    states = {}
    for path, run in runs:
        (route,) = run["routes"]
        merged["samples"].setdefault(route, []).extend(run["samples"][route])
        clock = tuple(run["clocks"][route])
        if merged["clocks"].setdefault(route, clock) != clock:
            raise RuntimeError("two processes of %s stopped at different clocks" % route)
        merged["compile_seconds"].setdefault(route, []).append(run["compile_seconds"][route])
        state = np.load(path.with_suffix(".state.npy"))
        if route in states and not np.array_equal(states[route], state):
            raise RuntimeError("two processes of %s end in different states" % route)
        states[route] = state
    if set(states) != set(ROUTES):
        raise SystemExit("merge needs processes of both routes")
    result = _finish(merged, states)
    out.write_text(json.dumps(result, indent=1))
    return result


def verdict(paths) -> dict:
    """PF10 (1) per measurement, PF10 (3) per route and thread count over the sizes."""
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    for run, path in zip(runs, paths):
        if run.get("schema") != SCHEMA:
            raise SystemExit("%s is not a measurement of this bench" % path)
    first = []
    for run in runs:
        ratio = run["timing"]["slice"]["throughput_vs_master"]
        first.append({"cells": run["cells"], "threads": run["threads"]["running"],
                      "throughput_vs_master": ratio, "holds": ratio >= PF10_1})
    third = []
    by_threads: dict = {}
    for run in runs:
        by_threads.setdefault(run["threads"]["running"], []).append(run)
    for threads, group in sorted(by_threads.items()):
        group.sort(key=lambda run: run["cells"][0]*run["cells"][1])
        for small, large in zip(group, group[1:]):
            cells_ratio = (large["cells"][0]*large["cells"][1])/(small["cells"][0]*small["cells"][1])
            if cells_ratio < 4:
                continue
            for route in ROUTES:
                growth = (large["timing"][route]["median_s_per_step"]
                          / small["timing"][route]["median_s_per_step"])
                third.append({"route": route, "threads": threads, "cells": [small["cells"],
                                                                           large["cells"]],
                              "growth": growth, "bound": PF10_3*cells_ratio,
                              "holds": growth <= PF10_3*cells_ratio})
    return {"pf10_1": first, "pf10_3": third,
            "holds": all(row["holds"] for row in first + third) and bool(third)}


def _arguments():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("measure", help="both routes at one size and thread count")
    one.add_argument("cells", type=int, help="N for N x N cells, for example 2048")
    one.add_argument("--threads", type=int, required=True)
    one.add_argument("--steps", type=int, default=4, help="accepted steps per block")
    one.add_argument("--blocks", type=int, default=3, help="A B B A repetitions")
    one.add_argument("--warmup", type=int, default=2, help="steps per route before timing")
    one.add_argument("--out", required=True, type=Path)
    one.add_argument("--route", choices=ROUTES + ("both",), default="both",
                     help="one route per process when both do not fit in memory; see merge")
    pair = commands.add_parser("merge", help="one measurement from one-route processes")
    pair.add_argument("runs", nargs="+", type=Path)
    pair.add_argument("--out", required=True, type=Path)
    many = commands.add_parser("verdict", help="PF10 (1) and (3) from measurements")
    many.add_argument("runs", nargs="+", type=Path)
    many.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.command == "measure":
        return measure(args)
    if args.command == "merge":
        result = merge(args.runs, args.out)
        print(json.dumps({"parity": result["parity"],
                          "throughput_vs_master":
                              result["timing"]["slice"]["throughput_vs_master"]}, indent=1))
        return 0
    result = verdict(args.runs)
    text = json.dumps(result, indent=1)
    if args.out is not None:
        args.out.write_text(text)
    print(text)
    return 0 if result["holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
