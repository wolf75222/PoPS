#!/usr/bin/env python3
"""DM-T2 (spec 0.3 §10.8): time per step of M01 and M03 at a head against b6, ABBA across builds.

N10 phase 1 (ROM-85), driver's decision P2. The gate measures the retyping, not N3a: both builds
run b6's native face route (at the head ``build_m01(cells, scheme=native_rusanov)``, at b6
``build_m01(cells)``, whose default is the native ``riemann.Rusanov()``), and the head may take at
most 1.02 times b6's median time per step on each case, size and thread count. The default route
of the head (the Python face of N3a) is measured apart by ``benchmarks/n3a/face_routes.py``.

The protocol is the one of ``face_routes.py`` (warm-up, interleaved A B B A blocks of accepted
steps, time per step = wall time of a block over its steps, median over blocks, final states
compared bit for bit), across two builds: each build runs in its own worker process, started on
its own source tree and native package, and the driver advances one worker at a time, so the
other sits idle on its bound runtime. Nothing in a worker reads code of this repository but the
case builders and ``pops``, so the same file drives b6, where neither ``face_routes.py`` nor
``scheme=`` exists.

    dmt2.py drive M01 4096x4096 --a b6=RUN:SRC --b head=RUN:SRC:native_rusanov \\
        --steps 3 --blocks 3 --warmup 2 --out m01_4096_t1.json
    dmt2.py gate RESULT.json ...       # prints the table and fails above 1.02

``RUN`` is a directory holding the built ``pops`` package of ``SRC`` (``cp -rL`` of the build
tree's ``python/pops``). A worker imports ``pops`` from ``RUN`` first (``PYTHONPATH=RUN:SRC``) and
compiles its packages against ``SRC/include``: a probe that changes Python code changes it in
``RUN``, or both workers run the same code. The script asserts no threshold while measuring; ``gate`` reads the
results.
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

GATE = 1.02


# --------------------------------------------------------------------------------------- worker

def _worker(args) -> int:
    import numpy as np
    import pops
    from examples.api_v02 import models_M01_M14 as cases

    builder = {"M01": cases.build_m01, "M03": cases.build_m03}[args.case]
    cells = tuple(int(count) for count in args.cells.split("x"))
    if args.scheme:
        spec = builder(cells=cells, scheme=getattr(cases, args.scheme))
    else:
        spec = builder(cells=cells)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    start = time.perf_counter()
    artifact = pops.compile(plan)
    compile_seconds = time.perf_counter() - start
    runtime = pops.bind(artifact, initial_values=dict(spec.initial_values))
    # With KOKKOS_TOOLS_LIBS naming benchmarks/n3a/kernel_timer.cpp, each block is split per
    # kernel, as in face_routes.py; the library is the one Kokkos loaded, reached by its path.
    timer = None
    if os.environ.get("KOKKOS_TOOLS_LIBS"):
        import ctypes

        timer = ctypes.CDLL(os.environ["KOKKOS_TOOLS_LIBS"])
    dump = Path(args.kernels_tmp) if args.kernels_tmp else None

    def advance(steps):
        begin = time.perf_counter()
        report = pops.run(runtime, t_end=runtime.time() + steps*spec.dt, max_steps=steps + 1,
                          console=False)
        return time.perf_counter() - begin, int(report.accepted_steps)

    advance(args.warmup)
    _answer({"ready": True, "compile_seconds": compile_seconds, "pops": pops.__file__,
             "source": _source_revision(Path.cwd())})
    for line in sys.stdin:
        command = json.loads(line)
        if "advance" in command:
            if timer is not None:
                timer.n3a_timer_reset()
            seconds, accepted = advance(int(command["advance"]))
            row = {"seconds": seconds, "accepted": accepted}
            if timer is not None and dump is not None:
                if timer.n3a_timer_dump(str(dump).encode()) != 0:
                    raise RuntimeError("kernel timer could not write %s" % dump)
                row["kernels"] = json.loads(dump.read_text())
            _answer(row)
        elif "state" in command:
            state = np.asarray(runtime.state_global("fluid"), dtype=float)
            np.save(command["state"], state)
            _answer({"steps": int(runtime.macro_step()), "shape": list(state.shape)})
        else:
            _answer({"bye": True})
            return 0
    return 0


def _answer(row) -> None:
    sys.stdout.write("DMT2 " + json.dumps(row) + "\n")
    sys.stdout.flush()


def _source_revision(tree: Path) -> dict:
    try:
        sha = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], check=True,
                             capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(tree), "status", "--porcelain"],
                                    check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"sha": "unknown", "dirty": None}
    return {"sha": sha, "dirty": dirty}


# --------------------------------------------------------------------------------------- driver

class _Build:
    """One worker process: a name, a native package directory, a source tree, a scheme."""

    def __init__(self, text: str, case: str, cells: str, warmup: int, log: Path):
        name, _, rest = text.partition("=")
        parts = rest.split(":")
        if not name or len(parts) not in (2, 3):
            raise SystemExit("a build is NAME=RUN:SRC[:SCHEME], not %r" % text)
        self.name = name
        self.run, self.source = Path(parts[0]).resolve(), Path(parts[1]).resolve()
        self.scheme = parts[2] if len(parts) == 3 else ""
        env = dict(os.environ)
        env.update(PYTHONPATH="%s:%s" % (self.run, self.source),
                   POPS_INCLUDE=str(self.source / "include"))
        command = [sys.executable, str(Path(__file__).resolve()), "worker", case, cells,
                   "--warmup", str(warmup), "--kernels-tmp",
                   str(log.with_suffix(".kernels.tmp.json"))]
        if self.scheme:
            command += ["--scheme", self.scheme]
        self.log = log.open("w", encoding="utf-8")
        self.process = subprocess.Popen(command, cwd=self.source, env=env, text=True,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log)

    def read(self) -> dict:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if line.startswith("DMT2 "):
                return json.loads(line[5:])
            self.log.write(line)
        raise RuntimeError("worker %s ended (exit %s); see %s"
                           % (self.name, self.process.wait(), self.log.name))

    def ask(self, command: dict) -> dict:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()
        return self.read()

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.ask({"quit": True})
            except (RuntimeError, BrokenPipeError):
                pass
            self.process.wait(timeout=60)
        self.log.close()


def _drive(args) -> int:
    import numpy as np

    out = args.out.resolve()  # the workers run in their own source trees
    out.parent.mkdir(parents=True, exist_ok=True)
    # The workers start one after the other: compilation and warm-up are not timed together.
    builds = []
    ready = {}
    try:
        for text, tag in ((args.a, "a"), (args.b, "b")):
            build = _Build(text, args.case, args.cells, args.warmup,
                           out.with_suffix(".%s.log" % tag))
            builds.append(build)
            ready[build.name] = build.read()
        if len({build.name for build in builds}) != 2:
            raise SystemExit("the two builds need distinct names")
        a, b = builds
        samples = {a.name: [], b.name: []}
        kernels: dict = {a.name: {}, b.name: {}}
        order = [build for _ in range(args.blocks) for build in (a, b, b, a)]
        for build in order:
            row = build.ask({"advance": args.steps})
            if row["accepted"] != args.steps:
                raise RuntimeError("%s took %d steps in a block of %d"
                                   % (build.name, row["accepted"], args.steps))
            samples[build.name].append(row["seconds"]/row["accepted"])
            for entry in row.get("kernels", ()):
                total = kernels[build.name].setdefault(
                    entry["name"], {"kind": entry["kind"], "seconds": 0., "calls": 0, "steps": 0})
                total["seconds"] += entry["seconds"]
                total["calls"] += entry["calls"]
            for total in kernels[build.name].values():
                total["steps"] = len(samples[build.name])*args.steps
        states = {}
        steps = {}
        for build in builds:
            path = out.with_suffix(".%s.npy" % build.name)
            steps[build.name] = build.ask({"state": str(path)})["steps"]
            states[build.name] = np.load(path)
            path.unlink()
    finally:
        for build in builds:
            build.close()

    reference, head = a.name, b.name
    timing = {}
    for name, rows in samples.items():
        timing[name] = {"median_s_per_step": statistics.median(rows), "min_s_per_step": min(rows),
                        "max_s_per_step": max(rows), "samples": rows}
    # Paired: each A B B A repetition gives two adjacent (B, A) pairs, A1 B1 and B2 A2.
    paired = []
    for index in range(args.blocks):
        a1, a2 = samples[reference][2*index], samples[reference][2*index + 1]
        b1, b2 = samples[head][2*index], samples[head][2*index + 1]
        paired += [b1/a1, b2/a2]
    difference = states[head] - states[reference]
    result = {
        "schema": "pops.n10.dmt2.v1", "case": args.case, "cells": args.cells,
        "reference": reference, "head": head, "steps_per_block": args.steps,
        "blocks": args.blocks, "warmup": args.warmup,
        "threads": os.environ.get("OMP_NUM_THREADS"),
        "proc_bind": os.environ.get("OMP_PROC_BIND"), "places": os.environ.get("OMP_PLACES"),
        "host": platform.node(), "processor": platform.processor() or platform.machine(),
        "cpus": os.cpu_count(),
        "builds": {build.name: {"run": str(build.run), "source": str(build.source),
                                "scheme": build.scheme or "default", **ready[build.name]}
                   for build in builds},
        "timing": timing,
        "head_over_reference": timing[head]["median_s_per_step"]
        / timing[reference]["median_s_per_step"],
        "paired_head_over_reference": statistics.median(paired), "paired": paired,
        "gate": GATE,
        "parity": {"bitwise_equal": bool(np.array_equal(states[head], states[reference])),
                   "l1": float(np.mean(np.abs(difference))),
                   "linf": float(np.max(np.abs(difference))), "steps": steps},
        "kernels": kernels,
    }
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("case", "cells", "threads",
                                                   "head_over_reference",
                                                   "paired_head_over_reference", "parity")}))
    return 0


def _gate(args) -> int:
    failures = 0
    print("| Case | Cells | Threads | %s s/step | %s s/step | ratio | paired | states |"
          % ("reference", "head"))
    print("|---|---|---|---|---|---|---|---|")
    for path in args.results:
        row = json.loads(path.read_text(encoding="utf-8"))
        reference, head = row["timing"][row["reference"]], row["timing"][row["head"]]
        ratio = row["head_over_reference"]
        verdict = "ok" if ratio <= row["gate"] else "ABOVE %.2f" % row["gate"]
        failures += ratio > row["gate"]
        print("| %s | %s | %s | %.3f (%.3f-%.3f) | %.3f (%.3f-%.3f) | %.3f %s | %.3f | %s |" % (
            row["case"], row["cells"], row["threads"], reference["median_s_per_step"],
            reference["min_s_per_step"], reference["max_s_per_step"],
            head["median_s_per_step"], head["min_s_per_step"], head["max_s_per_step"], ratio,
            verdict, row["paired_head_over_reference"],
            "bitwise equal" if row["parity"]["bitwise_equal"]
            else "linf %.3g" % row["parity"]["linf"]))
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("case", choices=("M01", "M03"))
    worker.add_argument("cells")
    worker.add_argument("--scheme", default="")
    worker.add_argument("--warmup", type=int, default=2)
    worker.add_argument("--kernels-tmp", default="")
    drive = sub.add_parser("drive")
    drive.add_argument("case", choices=("M01", "M03"))
    drive.add_argument("cells", help="NXxNY, for example 4096x4096")
    drive.add_argument("--a", required=True, help="reference build NAME=RUN:SRC[:SCHEME]")
    drive.add_argument("--b", required=True, help="measured build NAME=RUN:SRC[:SCHEME]")
    drive.add_argument("--steps", type=int, default=4, help="accepted steps per block")
    drive.add_argument("--blocks", type=int, default=3, help="A B B A repetitions")
    drive.add_argument("--warmup", type=int, default=2, help="steps per build before timing")
    drive.add_argument("--out", required=True, type=Path)
    gate = sub.add_parser("gate")
    gate.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.command == "worker":
        return _worker(args)
    if args.command == "drive":
        return _drive(args)
    return _gate(args)


if __name__ == "__main__":
    sys.exit(main())
