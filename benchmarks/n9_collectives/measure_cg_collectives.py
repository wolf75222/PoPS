"""Collectives per iteration of b6's Program CG and of the native CG (N9, ROM-90, SV20).

The spec counts at least 12 collectives per iteration for the Program CG at b6, against 2 for
the native CG (spec 0.3 §7.6). This script measures both on the cases that run them:

* T04 global (``examples/api_v02/methods_T01_T12.py:build_t04_global``): two solves by the CG
  that ``examples/api_v02/method_krylov.py`` writes with ``iterate`` and ``dot_all``;
* M08 (``examples/api_v02/models_M01_M14.py:build_m08``): the periodic Poisson solve by the
  native ``CG(max_iter=2000, rel_tol=1e-12, abs_tol=1e-13)``.

Run it under ``mpiexec`` with ``LD_PRELOAD`` set to the PMPI trace library built from
``mpi_collective_trace.c`` (see README.md). The library records every MPI collective call
without changing any PoPS path. For both solvers, an iteration is the period of the longest
periodic stretch of the trace (CG iterations repeat the same collective sequence), searched over
(collective, call site) pairs when POPS_N9_TRACE_FRAMES is set, which also attributes each
collective of the iteration to the PoPS function that issued it.
"""
from __future__ import annotations

import argparse
import bisect
import ctypes
import json
import subprocess
import sys
from collections import Counter

import numpy as np

KINDS = {1: "Allreduce", 2: "Allgather", 3: "Allgatherv", 4: "Bcast", 5: "Barrier", 6: "Reduce",
         7: "Gather", 8: "Gatherv", 9: "Alltoall", 10: "Alltoallv", 11: "Scatter",
         12: "Scatterv", 13: "Reduce_scatter_block", 14: "Scan", 15: "Exscan",
         16: "Iallreduce", 17: "Ibarrier", 18: "Ibcast", 19: "Comm_dup", 20: "Comm_split"}
OPS = {-1: "", 0: "user", 1: "sum", 2: "max", 3: "min", 4: "lor", 5: "bor", 6: "land",
       7: "band", 8: "maxloc", 9: "minloc"}


class Trace:
    """ctypes view of the preloaded PMPI trace library."""

    def __init__(self):
        library = ctypes.CDLL(None)
        try:
            self._size = library.pops_n9_trace_size
        except AttributeError as error:
            raise SystemExit("LD_PRELOAD the PMPI trace library (libpops_n9_mpi_trace.so)") from error
        self._size.restype = ctypes.c_int64
        self._dropped = library.pops_n9_trace_dropped
        self._dropped.restype = ctypes.c_int64
        self._p2p = library.pops_n9_point_to_point
        self._p2p.restype = ctypes.c_int64
        self._reset = library.pops_n9_trace_reset
        self._copy = library.pops_n9_trace_copy
        self._copy.restype = ctypes.c_int64
        self._copy.argtypes = (ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int32))
        self._depth = library.pops_n9_trace_frame_depth
        self._depth.restype = ctypes.c_int
        self._frames = library.pops_n9_trace_frames
        self._frames.restype = ctypes.c_int64
        self._frames.argtypes = (ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_uint64))
        self._resolve = library.pops_n9_resolve
        self._resolve.restype = ctypes.c_int
        self._resolve.argtypes = (ctypes.c_uint64, ctypes.c_char_p, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_uint64))

    def reset(self):
        self._reset()

    def frames(self, first, count):
        """Return addresses of entries [first, first + count), or None without frames."""
        depth = int(self._depth())
        if depth == 0 or count == 0:
            return None
        buffer = (ctypes.c_uint64 * (depth * count))()
        copied = int(self._frames(first, count, buffer))
        return [list(buffer[row * depth:(row + 1) * depth]) for row in range(copied)]

    def resolve(self, address):
        path = ctypes.create_string_buffer(4096)
        base = ctypes.c_uint64(0)
        if not self._resolve(address, path, len(path), ctypes.byref(base)):
            return None
        return path.value.decode(), int(address) - int(base.value)

    def read(self):
        size = int(self._size())
        if int(self._dropped()) != 0:
            raise RuntimeError("PMPI trace overflowed")
        buffer = (ctypes.c_int32 * (4 * max(size, 1)))()
        copied = int(self._copy(0, size, buffer))
        rows = np.frombuffer(buffer, dtype=np.int32)[:4 * copied].reshape(copied, 4)
        return [tuple(int(item) for item in row) for row in rows], int(self._p2p())


def label(entry):
    kind, count, element_bytes, op = entry
    name = KINDS.get(kind, str(kind))
    op_name = OPS.get(op, str(op))
    return "%s(%s%d x %dB)" % (name, op_name + " " if op_name else "", count, element_bytes)


def longest_period(trace, max_period=600, min_repeats=3):
    """Period of the longest periodic stretch of the trace.

    For each period p, the longest run of trace[i] == trace[i + p] covers a stretch of run + p
    entries. The CG iterations form the longest stretch; a multiple of the iteration period
    covers one iteration less, so the smallest period within 95% of the best coverage is the
    iteration. Returns (period, repeats, start), or None without min_repeats repetitions.
    """
    rows = []
    for period in range(1, min(max_period, len(trace) // min_repeats) + 1):
        run = longest = start = 0
        for index in range(len(trace) - period):
            if trace[index] == trace[index + period]:
                run += 1
                if run > longest:
                    longest, start = run, index - run + 1
            else:
                run = 0
        coverage = longest + period
        if coverage >= min_repeats * period:
            rows.append((coverage, period, start))
    if not rows:
        return None
    best = max(row[0] for row in rows)
    coverage, period, start = min((row for row in rows if row[0] >= 0.95 * best),
                                  key=lambda row: row[1])
    return period, coverage // period, start


_TRANSPORT = ("pops::all_reduce", "pops::collectively_rethrow_exception", "pops::detail::",
              "pops::all_ranks_agree", "pops::collective_", "pops::reduce_", "MPI_", "PMPI_",
              "ompi_", "mca_", "pops_n9_")


def _short(name):
    """Function name without parameters or nested template arguments."""
    out, depth = [], 0
    for character in name.split("(")[0] if not name.startswith("auto ") else name[5:].split("(")[0]:
        if character == "<":
            depth += 1
            if depth == 1:
                out.append("<")
            continue
        if character == ">":
            depth -= 1
            if depth == 0:
                out.append(">")
            continue
        if depth == 0:
            out.append(character)
    return "".join(out).replace("pops::runtime::program::", "")


class Symbols:
    """Function containing an offset of an object file, from its symbol table (nm)."""

    def __init__(self):
        self._tables = {}

    def _table(self, path):
        if path not in self._tables:
            rows = []
            for flags in (["-C", "-n", "--defined-only"], ["-C", "-n", "-D", "--defined-only"]):
                listing = subprocess.run(["nm", *flags, path], capture_output=True, text=True)
                for line in listing.stdout.splitlines():
                    parts = line.split(" ", 2)
                    if len(parts) == 3 and parts[1] in "tTwWiu":
                        rows.append((int(parts[0], 16), parts[2]))
                if rows:
                    break
            rows.sort()
            self._tables[path] = ([row[0] for row in rows], [row[1] for row in rows])
        return self._tables[path]

    def name(self, path, offset):
        starts, names = self._table(path)
        index = bisect.bisect_right(starts, offset) - 1
        return names[index] if index >= 0 else "?"


def attribute(trace, first, count):
    """For each entry, the PoPS functions that issued the collective (innermost first), or None
    when the library records no frames (POPS_N9_TRACE_FRAMES unset)."""
    frames = trace.frames(first, count)
    if frames is None:
        return None
    symbols, cache, sites = Symbols(), {}, []
    for row in frames:
        chain = []
        for address in row:
            if not address:
                continue
            if address not in cache:
                where = trace.resolve(address - 1)  # the call instruction, not the return
                cache[address] = None if where is None else _short(symbols.name(*where))
            name = cache[address]
            if name is None or name.startswith(_TRANSPORT) or (chain and chain[-1] == name):
                continue
            chain.append(name)
        sites.append(" <- ".join(chain[:3]) if chain else "?")
    return sites


def compile_and_bind(builder, scope):
    import pops
    from tests.python.support.collective_checks import collective_call
    from tests.python.support.native_execution_context import artifact_execution_context
    from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

    _native, world = native_profile(dimension=2, scope=scope)
    spec = collective_call(world, builder)
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case), layout=spec.layout))
    artifact = compile_profile_plan_once(world, plan, route=scope, compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(
        artifact, initial_values={key: value.copy() for key, value in spec.initial_values.items()},
        resources={"execution_context": artifact_execution_context(artifact)}))
    return spec, runtime, world


def _reraise(failure):
    if failure is not None:
        raise failure


def run_case(spec, runtime, world, trace):
    import pops
    from tests.python.support.collective_checks import collective_call
    from tests.python.support.public_native_profile import profile_gather_value

    trace.reset()
    failure = None
    steps = int(round(spec.t_end / spec.dt))
    try:
        pops.run(runtime, t_end=spec.t_end, max_steps=steps, console=False)
    except Exception as error:  # read the trace first, then converge the failure
        failure = error
    entries, p2p = trace.read()
    collective_call(world, lambda: _reraise(failure))
    rows = profile_gather_value(world, collective_call(world, lambda: spec.invariants(runtime)))
    passed = all(row["passed"] for row in rows if row.get("metrics_owner"))
    return entries, p2p, passed, steps


def iteration_profile(trace, entries):
    """Collectives of one iteration: the period of the longest periodic stretch of the trace.

    With call sites, the stretch is searched over (collective, call site) pairs, so two equal
    collectives issued from two places of one iteration are not mistaken for two iterations.
    """
    sites = attribute(trace, 0, len(entries))
    keys = entries if sites is None else list(zip(entries, sites, strict=True))
    found = longest_period(keys)
    if found is None:
        raise RuntimeError("no periodic iteration in the collective trace")
    period, repeats, start = found
    block = entries[start:start + period]
    profile = {
        "collectives_per_iteration": period,
        "periodic_iterations_seen": repeats,
        "iteration": dict(Counter(label(entry) for entry in block)),
        "iteration_sequence": [label(entry) for entry in block],
    }
    if sites is not None:
        block_sites = sites[start:start + period]
        profile["iteration_sites"] = [
            {"site": site, "collectives": dict(Counter(
                label(entry) for entry, where in zip(block, block_sites, strict=True)
                if where == site))}
            for site, _ in Counter(block_sites).most_common()]
        profile["iteration_site_sequence"] = block_sites
    return profile


def measure(trace, builder, case, scope):
    spec, runtime, world = compile_and_bind(builder, scope)
    entries, p2p, passed, steps = run_case(spec, runtime, world, trace)
    return world, {
        "case": case,
        "passed": bool(passed),
        "steps": steps,
        "collectives_per_run": len(entries),
        "point_to_point_per_run": p2p,
        "run": dict(Counter(label(entry) for entry in entries)),
        **iteration_profile(trace, entries),
    }


def measure_program_cg(trace):
    from examples.api_v02.methods_T01_T12 import build_t04_global

    return measure(trace, build_t04_global,
                   "T04 global, b6 Program CG (examples/api_v02/method_krylov.py)",
                   "N9 Program CG collectives")


def measure_native_cg(trace):
    from examples.api_v02.models_M01_M14 import build_m08

    return measure(trace, build_m08,
                   "M08, native CG(max_iter=2000, rel_tol=1e-12, abs_tol=1e-13)",
                   "N9 native CG collectives")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=("program", "native", "both"), default="both")
    parser.add_argument("--output", help="JSON file written by rank 0")
    args = parser.parse_args(argv)
    trace = Trace()
    results, world = [], None
    if args.case in ("program", "both"):
        world, row = measure_program_cg(trace)
        results.append(row)
    if args.case in ("native", "both"):
        world, row = measure_native_cg(trace)
        results.append(row)
    rank, size = (0, 1) if world is None else (int(world.rank), int(world.size))
    document = {"schema": "pops.n9.cg_collectives.v1", "ranks": size, "rank": rank,
                "results": results}
    if rank == 0:
        text = json.dumps(document, indent=2)
        print(text)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as stream:
                stream.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
