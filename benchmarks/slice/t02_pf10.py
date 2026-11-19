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
is linear in the cells). Both comparisons use the raw samples. Split-process receipts must
follow ABBA and hold the same comparison context. Historical receipts remain readable and
report which provenance was outside their recording protocol. The checkout revision describes
the driver. The imported Python package, native module, effective headers and compiler have
separate identities. Use ``verdict --phase1`` to qualify the four T8 production profiles.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "docs" / "tutorials" / "scalar_advection" / "02_openmp_explicit_ssprk2.py"
SLICE = ROOT / "examples" / "surface" / "tutorials" / "scalar_advection" / \
    "02_openmp_explicit_ssprk2.py"
_INPUTS = (Path(__file__).resolve(), MASTER, SLICE)
ROUTES = ("master", "slice")
SCHEMA = "pops.slice.t02_pf10.v1"
PF10_1 = 0.9            # OD9
PF10_3 = 1.25           # OD24: the measured growth exceeds the stated complexity by at most 25 %
_CONTEXT = ("host", "processor", "cpus", "source", "omp_proc_bind")
_RUNTIME_CONTEXT = ("python_runtime", "include_signature", "native_abi", "compiler")
_OPTIONAL_CONTEXT = ("omp_places", "omp_num_threads", "cpu_affinity",
                     "input_sources", "native_build", "kernel_timer", *_RUNTIME_CONTEXT)
_PHASE1_PROFILES = {(cells, threads) for cells in (2048, 4096) for threads in (1, 4)}


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
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                                     "--untracked-files=no"],
                                    check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"sha": "unknown", "dirty": None}
    return {"sha": sha, "dirty": dirty}


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_runtime(pops):
    """Fingerprint the source package Python located, excluding caches and bundled binaries."""
    location = getattr(pops, "__file__", None)
    if not location:
        return None
    root = Path(location).parent.resolve()
    try:
        sources = [(path.relative_to(root).as_posix(), _file_digest(path))
                   for path in sorted(root.rglob("*.py"))]
    except OSError:
        return None
    if not sources:
        return None
    return {"source_sha256": hashlib.sha256(json.dumps(sources).encode()).hexdigest(),
            "pops_version": getattr(pops, "__version__", None),
            "interpreter": platform.python_implementation() + " " + platform.python_version()}


def _compiler_context():
    """Reuse the generated loader's include and compiler selection."""
    from pops.codegen import toolchain

    result = {}
    try:
        result["include_signature"] = toolchain.pops_header_signature(toolchain.pops_include())
    except (AttributeError, OSError, RuntimeError):
        pass
    try:
        command = toolchain._native_kokkos_compiler(None)
        executable = shutil.which(command) if command else None
        if executable:
            version = subprocess.run([executable, "--version"], check=True, capture_output=True,
                                     text=True, timeout=5).stdout.strip()
            if version:
                result["compiler"] = {"sha256": _file_digest(executable), "version": version}
    except (AttributeError, OSError, RuntimeError, subprocess.SubprocessError):
        pass
    return result


def _measurement_context(native, runtime):
    import pops

    data = {
        "host": platform.node(), "processor": platform.processor() or platform.machine(),
        "cpus": os.cpu_count(), "source": _source_revision(),
        "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
        "omp_places": os.environ.get("OMP_PLACES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "input_sources": {str(path.relative_to(ROOT)): _file_digest(path) for path in _INPUTS},
    }
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        data["cpu_affinity"] = sorted(get_affinity(0))
    python_runtime = _python_runtime(pops)
    if python_runtime is not None:
        data["python_runtime"] = python_runtime
    data.update(_compiler_context())
    abi_key = getattr(native, "abi_key", None)
    if callable(abi_key) and (key := abi_key()) and key != "unknown":
        data["native_abi"] = {"key": key, "version": getattr(native, "__abi_version__", None)}
    module = getattr(native, "__file__", None)
    if module is not None and Path(module).is_file():
        data["native_build"] = {
            "module_sha256": _file_digest(module),
            **{key: runtime[key] for key in ("dimension", "precision", "kokkos_backend")},
            "compile_environment": {key: os.environ.get(key) for key in
                                    ("POPS_CXX", "POPS_KOKKOS_CXX", "POPS_DSL_OPTFLAGS")},
        }
    tools = os.environ.get("KOKKOS_TOOLS_LIBS")
    data["kernel_timer"] = _file_digest(tools) if tools else None
    return data


def measure(args) -> int:
    if min(args.cells, args.threads, args.steps, args.blocks) < 1 or args.warmup < 0:
        raise ValueError("cells, threads, steps and blocks must be positive; warmup is nonnegative")
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
    runtime_report = native.runtime_environment_report() if native is not None else {}
    concurrency = runtime_report.get("kokkos_concurrency")
    if concurrency != args.threads:
        raise RuntimeError("Kokkos runs %r threads, the bench asked %d" % (concurrency,
                                                                          args.threads))
    threads = dict(threads, running=concurrency, kokkos_concurrency=concurrency)
    context = _measurement_context(native, runtime_report)
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
              "threads": threads, **context,
              "compile_seconds": compile_seconds, "clocks": clocks,
              "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "samples": samples, "kernels": kernels}
    _check_receipt(result, args.out)
    for route in routes:
        _check_samples(samples[route], 2 * args.blocks, args.out)
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


def _check_receipt(run, path):
    if not isinstance(run, dict) or run.get("schema") != SCHEMA:
        raise ValueError("%s is not a measurement of this bench" % path)
    cells = run.get("cells")
    if (not isinstance(cells, list) or len(cells) != 2 or cells[0] != cells[1]
            or any(type(size) is not int or size < 1 for size in cells)):
        raise ValueError("%s must have a positive square grid" % path)
    if any(type(run.get(key)) is not int or run[key] < minimum
           for key, minimum in (("steps_per_block", 1), ("blocks", 1), ("warmup", 0))):
        raise ValueError("%s has an invalid timed protocol" % path)
    threads = run.get("threads")
    if (not isinstance(threads, dict) or type(threads.get("running")) is not int
            or threads["running"] < 1
            or any(type(threads.get(key)) is not int or threads[key] != threads["running"]
                   for key in ("asked", "kokkos_concurrency"))):
        raise ValueError("%s reports incompatible requested and active threads" % path)
    if any(key not in run for key in _CONTEXT):
        raise ValueError("%s is missing the comparison context" % path)
    steps = run["warmup"] + 2 * run["blocks"] * run["steps_per_block"]
    routes = run.get("routes")
    if (not isinstance(routes, list) or not routes or any(route not in ROUTES for route in routes)
            or len(routes) != len(set(routes))):
        raise ValueError("%s has an invalid route inventory" % path)
    clocks = run.get("clocks")
    if not isinstance(clocks, dict) or set(clocks) != set(routes):
        raise ValueError("%s has an invalid clock inventory" % path)
    for clock in clocks.values():
        if (not isinstance(clock, (list, tuple)) or len(clock) != 2
                or type(clock[0]) is not int or clock[0] != steps
                or type(clock[1]) not in (int, float) or not math.isfinite(clock[1])
                or clock[1] <= 0):
            raise ValueError("%s has an invalid final clock for %d accepted steps" % (path, steps))


def _check_samples(samples, expected, path):
    if (not isinstance(samples, list) or len(samples) != expected
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in samples)):
        raise ValueError("%s must contain %d positive finite timing samples per route"
                         % (path, expected))


def _check_abba(order):
    if not isinstance(order, list) or len(order) < 4 or len(order) % 4:
        raise ValueError("split-process measurements require complete ABBA repetitions")
    abba = (ROUTES + ROUTES[::-1], ROUTES[::-1] + ROUTES)
    if any(tuple(order[index:index+4]) not in abba for index in range(0, len(order), 4)):
        raise ValueError("split-process measurements must follow ABBA")


def _instrumented(run):
    if "kernel_timer" in run:
        return run["kernel_timer"] is not None
    if "timing_instrumented" in run:
        return run["timing_instrumented"]
    if "processes" in run:
        # Older merged receipts discarded the component kernel reports.
        return None
    return any(run.get("kernels", {}).values())


def _require_same_context(first, other, path):
    for key in (*_CONTEXT, *_OPTIONAL_CONTEXT, "steps_per_block", "blocks", "warmup"):
        if (key in first) != (key in other) or first.get(key) != other.get(key):
            raise ValueError("%s has a different comparison context: %s" % (path, key))
    if _instrumented(first) != _instrumented(other):
        raise ValueError("%s has different timing instrumentation" % path)


def _missing_provenance(run):
    missing = [key for key in _OPTIONAL_CONTEXT if key not in run]
    missing.extend(key for key in _RUNTIME_CONTEXT if key in run and not run[key])
    source = run["source"]
    if not isinstance(source, dict) or not source.get("sha") or source["sha"] == "unknown":
        missing.append("source_revision")
    if not isinstance(source, dict) or source.get("dirty") is not False:
        missing.append("clean_source_tree")
    return missing


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
    result["missing_provenance"] = _missing_provenance(result)
    return result


def merge(paths, out: Path) -> dict:
    """One measurement from processes that each ran one route at the same size and threads, in
    the order A B B A across processes; their final states are compared as in one process."""
    import numpy as np

    paths = [Path(path) for path in paths]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("merge requires distinct component receipt paths")
    runs = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    for path, run in runs:
        _check_receipt(run, path)
        if len(run["routes"]) != 1 or any(key in run for key in ("timing", "order", "processes")):
            raise ValueError("%s is not a raw one-route measurement" % path)
    _check_abba([run["routes"][0] for _, run in runs])
    if len({json.dumps(run, sort_keys=True) for _, run in runs}) != len(runs):
        raise ValueError("merge requires distinct component measurements")
    first = runs[0][1]
    for path, run in runs:
        if run["cells"] != first["cells"] or run["threads"]["running"] != first["threads"]["running"]:
            raise ValueError("%s has a different size or thread count" % path)
        _require_same_context(first, run, path)
        route = run["routes"][0]
        if not isinstance(run.get("samples"), dict) or set(run["samples"]) != {route}:
            raise ValueError("%s has an invalid timing sample inventory" % path)
        _check_samples(run["samples"][route], 2 * run["blocks"], path)
    merged = dict(runs[0][1], processes=[str(path) for path, _ in runs],
                  order=[run["routes"][0] for _, run in runs], samples={}, clocks={},
                  timing_instrumented=_instrumented(first), kernels={}, compile_seconds={})
    states = {}
    for path, run in runs:
        (route,) = run["routes"]
        merged["samples"].setdefault(route, []).extend(run["samples"][route])
        clock = tuple(run["clocks"][route])
        if merged["clocks"].setdefault(route, clock) != clock:
            raise RuntimeError("two processes of %s stopped at different clocks" % route)
        merged["compile_seconds"].setdefault(route, []).append(run["compile_seconds"][route])
        state = np.load(path.with_suffix(".state.npy"), allow_pickle=False)
        if state.shape != tuple(run["cells"]) or state.dtype != np.float64 or not np.isfinite(state).all():
            raise ValueError("%s has an invalid final state shape, dtype or value" % path)
        if route in states and not np.array_equal(states[route].view(np.uint64), state.view(np.uint64)):
            raise RuntimeError("two processes of %s end in different states" % route)
        states[route] = state
    result = _finish(merged, states)
    out.write_text(json.dumps(result, indent=1))
    return result


def _require_phase1(runs):
    profiles = {(run["cells"][0], run["threads"]["running"]) for _, run in runs}
    if profiles != _PHASE1_PROFILES:
        raise ValueError("phase 1 requires 2048 and 4096 square grids at both 1 and 4 threads")
    reference = runs[0][1]
    for path, run in runs:
        if (len(run.get("order", ())) != 12 or run["steps_per_block"] != 2
                or run["blocks"] != 1 or run["warmup"] != 2):
            raise ValueError("%s does not follow the phase 1 split-process ABBA protocol" % path)
        if run["omp_proc_bind"] != "spread" or run.get("omp_places") != "threads":
            raise ValueError("%s does not use the phase 1 OpenMP placement" % path)
        required = (*_RUNTIME_CONTEXT, "native_build", "input_sources")
        missing = [key for key in required if not run.get(key)]
        if missing:
            raise ValueError("%s lacks phase 1 execution provenance: %s" % (path, ", ".join(missing)))
        build = run["native_build"]
        if build.get("dimension") != 2 or build.get("kokkos_backend") != "OpenMP":
            raise ValueError("%s is not a phase 1 Dim2 OpenMP measurement" % path)
        if (not build.get("module_sha256") or not run["python_runtime"].get("source_sha256")
                or not run["native_abi"].get("key") or not run["compiler"].get("sha256")
                or set(run["input_sources"]) != {str(source.relative_to(ROOT)) for source in _INPUTS}):
            raise ValueError("%s lacks phase 1 build or benchmark input identities" % path)
        if _instrumented(run) is not False:
            raise ValueError("%s does not establish uninstrumented phase 1 timing" % path)
        # The actual build and target must agree even between groups with different thread counts.
        for key in (*_CONTEXT, *required, "kernel_timer"):
            if reference.get(key) != run.get(key):
                raise ValueError("%s has a different phase 1 comparison context: %s" % (path, key))


def verdict(paths, *, phase1=False) -> dict:
    """PF10 (1) per measurement, PF10 (3) per route and thread count over the sizes."""
    first, third, keys = [], [], set()
    by_threads: dict = {}
    runs = []
    for path in paths:
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        _check_receipt(run, path)
        if set(run["routes"]) != set(ROUTES) or run["clocks"]["master"] != run["clocks"]["slice"]:
            raise ValueError("%s must compare both routes at the same final clock" % path)
        threads = run["threads"]["running"]
        key = (threads, tuple(run["cells"]))
        if key in keys:
            raise ValueError("%s duplicates a size and thread count" % path)
        keys.add(key)
        expected = 2 * run["blocks"]
        if "order" in run or "processes" in run:
            order, processes = run.get("order"), run.get("processes")
            _check_abba(order)
            if (not isinstance(processes, list) or len(processes) != len(order)
                    or any(not isinstance(process, str) or not process for process in processes)
                    or len(set(processes)) != len(processes)):
                raise ValueError("%s has an invalid split-process inventory" % path)
            expected *= len(order) // 2
        if not isinstance(run.get("timing"), dict) or set(run["timing"]) != set(ROUTES):
            raise ValueError("%s has an invalid route timing inventory" % path)
        medians = {}
        for route in ROUTES:
            row = run["timing"][route]
            samples = row.get("samples") if isinstance(row, dict) else None
            _check_samples(samples, expected, path)
            medians[route] = statistics.median(samples)
        ratio = medians["master"] / medians["slice"]
        first.append({"cells": run["cells"], "threads": threads,
                      "throughput_vs_master": ratio, "holds": ratio >= PF10_1,
                      "missing_provenance": _missing_provenance(run)})
        runs.append((path, run))
        by_threads.setdefault(threads, []).append((path, run, medians, expected))
    if phase1:
        _require_phase1(runs)
    missing_scaling_threads = []
    for threads, group in sorted(by_threads.items()):
        group.sort(key=lambda row: row[1]["cells"][0])
        _, reference, _, count = group[0]
        for path, run, _, expected in group[1:]:
            _require_same_context(reference, run, path)
            if count != expected or ("order" in reference) != ("order" in run):
                raise ValueError("%s has a different sampling protocol for scaling" % path)
        start = len(third)
        for (_, small, small_medians, _), (_, large, large_medians, _) in zip(group, group[1:]):
            cells_ratio = (large["cells"][0]*large["cells"][1])/(small["cells"][0]*small["cells"][1])
            if cells_ratio < 4:
                continue
            for route in ROUTES:
                growth = large_medians[route] / small_medians[route]
                third.append({"route": route, "threads": threads, "cells": [small["cells"],
                                                                           large["cells"]],
                              "growth": growth, "bound": PF10_3*cells_ratio,
                              "holds": growth <= PF10_3*cells_ratio})
        if len(third) == start:
            missing_scaling_threads.append(threads)
    result = {"pf10_1": first, "pf10_3": third,
              "missing_scaling_threads": missing_scaling_threads,
              "holds": bool(third) and not missing_scaling_threads
              and all(row["holds"] for row in first + third)}
    if phase1:
        result["phase1_protocol"] = True
    return result


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
    many.add_argument("--phase1", action="store_true",
                      help="require the four documented production profiles and their provenance")
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
    result = verdict(args.runs, phase1=args.phase1)
    text = json.dumps(result, indent=1)
    if args.out is not None:
        args.out.write_text(text)
    print(text)
    return 0 if result["holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
