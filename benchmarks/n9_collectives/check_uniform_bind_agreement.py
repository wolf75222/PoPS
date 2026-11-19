"""Probe native MPI CC13 admission on the Dim1 M15 and Dim2 M05 public graphs.

Run from a checkout with the matching installed wheel, the PMPI tracing library
preloaded and a fresh cache. Use the reference mode on the parent wheel to expose
the Uniform bind that admitted different Program code with one structural IR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()))

import pops
from benchmarks.n9_collectives.measure_cg_collectives import Trace
from examples.api_v02.models_M01_M14 import build_m05
from examples.api_v02.models_M15_orders import build_m15_orders
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import native_profile, profile_gather_value


# The fixed success/failure slot and two extrema per byte of the Program
# digest. The existing bind and step collectives
# keep their own trace entries.
AGREEMENT = (1, 1 + 2 * 32, 8, 2)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("m15", "m05"))
    parser.add_argument("mode", choices=("reference", "matching", "alternate-flags", "program", "local-error"))
    args = parser.parse_args()

    dimension = 1 if args.profile == "m15" else 2
    build = build_m15_orders if dimension == 1 else build_m05
    _, world = native_profile(dimension=dimension, scope="CC13 native bind")
    rank = int(world.rank)
    assert int(world.size) in (2, 3)
    trace = Trace()

    spec = collective_call(world, build)
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case), layout=spec.layout))
    from pops.codegen import _compile_drivers

    original_driver = _compile_drivers._compile_resolved_problem
    differs = args.mode in ("reference", "program")
    optflags = ("-O2" if differs and rank else "-O1") + " -DNDEBUG"
    if args.mode == "alternate-flags":
        optflags += " -fno-math-errno"

    def compile_program(*args, **kwargs):
        # Change the real Program toolchain only. An unrelated model package
        # mismatch would be refused by an older native installation control.
        with patch.dict(os.environ, {"POPS_DSL_OPTFLAGS": optflags}):
            return original_driver(*args, **kwargs)

    with patch.object(_compile_drivers, "_compile_resolved_problem", compile_program):
        artifact = collective_call(world, lambda: pops.compile(plan))
    values = {key: value.copy() for key, value in spec.initial_values.items()}
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    original_verify = type(artifact).verify

    def verify(local):
        if rank == 1 and local is artifact and args.mode == "local-error":
            raise ValueError("failed local authentication of compiled image")
        return original_verify(local)

    runtime = None
    failure = None
    trace.reset()
    with patch.object(type(artifact), "verify", verify):
        try:
            runtime = pops.bind(artifact, initial_values=values,
                                resources={"execution_context": context})
        except (RuntimeError, ValueError) as error:
            if args.mode != "reference":
                from pops.runtime._bind_agreement import SchemaMismatch

                assert type(error) is SchemaMismatch
            failure = (type(error).__name__, str(error))
    cold, cold_messages = trace.read()
    record = {
        "rank": rank,
        "program_id": artifact.program_id.token,
        "program_graph": artifact.program.program.to_graph().graph_hash,
        "binary": digest(artifact.so_path),
        "model_binaries": [digest(row.model.so_path) for row in artifact.blocks],
        "so_path": str(artifact.so_path),
        "failure": failure,
        "cold_agreements": cold.count(AGREEMENT),
        "cold_trace": cold,
        "cold_messages": cold_messages,
    }
    if runtime is not None:
        record["installed_program"] = runtime._executor._s.installed_program_hash()
        assert runtime.time() == 0 and runtime.macro_step() == 0
    peers = profile_gather_value(world, record)

    with collective_check(world):
        assert len(peers) == int(world.size)
        assert len({p["program_graph"] for p in peers}) == 1
        if differs:
            assert len({p["program_id"] for p in peers}) == 2
            assert len({p["binary"] for p in peers}) == 2
            assert len({tuple(p["model_binaries"]) for p in peers}) == 1
        else:
            assert len({p["program_id"] for p in peers}) == 1

        if args.mode == "reference":
            assert all(p["failure"] is None and p["cold_agreements"] == 0 for p in peers)
            assert len({p["installed_program"] for p in peers}) == 1
        else:
            assert all(p["cold_agreements"] == 1 and p["cold_trace"][0] == AGREEMENT
                       for p in peers)
        if args.mode in ("program", "local-error"):
            detail = "program_id differs" if differs else "could not authenticate"
            assert runtime is None
            assert all(p["failure"] is not None and p["failure"][0] == "SchemaMismatch"
                       and detail in p["failure"][1] for p in peers), [p["failure"] for p in peers]
            assert len({tuple(p["failure"]) for p in peers}) == 1
            assert all(p["cold_trace"] == [AGREEMENT] and p["cold_messages"] == 0
                       for p in peers), [(p["cold_trace"], p["cold_messages"]) for p in peers]
        if args.mode in ("matching", "alternate-flags"):
            assert all(p["failure"] is None for p in peers), [p["failure"] for p in peers]
            assert len({p["so_path"] for p in peers}) == len(peers)

    result = {"bind": peers}
    if args.mode in ("matching", "alternate-flags"):
        assert runtime is not None
        trace.reset()
        report = pops.run(runtime, t_end=spec.t_end, console=False)
        hot, hot_messages = trace.read()
        manifest = runtime._executor._last_run_manifest
        assert "math_library" not in manifest.controls
        assert type(manifest).from_dict(manifest.to_dict()).run_identity == manifest.run_identity
        metrics = spec.invariants(runtime)
        with collective_check(world):
            assert report.accepted_steps > 0 and runtime.time() == spec.t_end
            assert metrics["passed"]
            assert AGREEMENT not in hot
        result["run"] = profile_gather_value(world, {
            "rank": rank, "time": runtime.time(), "steps": runtime.macro_step(),
            "hot_trace": hot, "hot_messages": hot_messages})
    if rank == 0:
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
