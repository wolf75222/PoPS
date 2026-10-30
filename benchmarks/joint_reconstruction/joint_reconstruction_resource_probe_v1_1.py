#!/usr/bin/env python3
"""Separate, instrumented resource probe for the fixed 3d06cab -> cf6dace case.

This imports the corrected v1.1 timing protocol's case and authentication
helpers but does not produce timing observations or change its results. Run the plan
first; execute only after the two installed snapshots are available and idle.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


SCHEMA = "pops.api040.joint-reconstruction-resources.v1.1"
V1_1_SHA256 = "e59336591c49e5d658cc057bae3ea74b83c506cdec654ff9b929d3bd55319833"
CANDIDATE_COMMIT = "cf6daceaa028df1343a303751ec9ae8f9f5ce6b3"
CANDIDATE_NATIVE = "4b458dabd54357b8aa3a4659cee8363a9a4510e55aaa81706bfe03a445d27037"
CANDIDATE_SOURCES = "8c2d99c5149aee562ba0eddfcb7395958e77a28cac94af014ea0a2fc4caaeb45"
WARMUPS, PROFILED_RUNS, WORKER_TIMEOUT_S = 2, 3, 720
_COUNTER_UNITS = {
    "scratch_allocs": "allocations",
    "scratch_peak_bytes": "bytes_largest_single_scratch_buffer",
    "kernels": "program_kernel_operations_or_batches",
    "kernel_launches": "instrumented_amr_operations_or_batches",
    "mpi_messages": "native_reported_messages",
    "mpi_reductions": "native_reported_reductions",
    "halo_exchanges": "native_reported_exchanges",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_v1_1():
    path = Path(__file__).with_name("joint_reconstruction_benchmark_v1_1.py")
    if _sha(path) != V1_1_SHA256:
        raise ValueError("the frozen v1.1 timing/case protocol changed")
    spec = importlib.util.spec_from_file_location("pops_joint_benchmark_v1_1", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the frozen v1 case")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity(v1, root: Path, lane: str) -> dict:
    identity = v1._load_identity(root)
    expected = (
        (v1.BASELINE_COMMIT, v1.BASELINE_NATIVE, v1.BASELINE_SOURCES, 1079)
        if lane == "baseline" else
        (CANDIDATE_COMMIT, CANDIDATE_NATIVE, CANDIDATE_SOURCES, 1080)
    )
    actual = tuple(identity.get(key) for key in (
        "source_commit", "native_sha256", "source_files_sha256", "verified_source_files"))
    if actual != expected:
        raise ValueError(f"{lane} is not the retained authenticated snapshot")
    return identity


def _counter(counters: dict, name: str) -> dict:
    """A zero is real only when the native snapshot contains the counter key."""
    if name not in counters:
        return {"available": False, "value": None,
                "reason": "counter absent from native profile snapshot"}
    value = counters[name]
    if type(value) is not int or value < 0:
        raise ValueError(f"native counter {name!r} is not a nonnegative integer")
    return {"available": True, "value": value, "unit": _COUNTER_UNITS[name]}


def _selected_counters(counters: dict) -> dict:
    return {name: _counter(counters, name) for name in _COUNTER_UNITS}


def _profile_once(v1, artifact, subject, initial):
    import numpy as np
    import pops
    from pops.runtime._profile import Profile
    from pops.runtime._system import System

    runtime = pops.bind(artifact, initial_values={subject: initial.copy()},
                        resources={"execution_context": v1._execution_context(artifact)})
    engine = runtime._executor  # Private observation seam; public bind/run remain authoritative.
    if not isinstance(engine, System) or not callable(getattr(engine, "profile", None)):
        raise ValueError("this Uniform artifact has no native System.profile authority")
    with engine.profile(Profile.Advanced()) as session:
        pops.run(runtime, t_end=v1.STEPS * v1.DT, max_steps=v1.STEPS, console=False)
        native_snapshot = engine.profile_snapshot()
    summary = session.summary()
    if summary.source != "snapshot" or native_snapshot.get("enabled") is not True:
        raise ValueError("native structured profiling was not enabled during the run")
    counters = summary.counters()
    if not isinstance(counters, dict):
        raise ValueError("native profile returned no counter mapping")
    if "steps" in counters and counters["steps"] != v1.STEPS:
        raise ValueError("profiled native step count differs from accepted steps")
    final = np.asarray(runtime.state_global("field")).reshape(v1.WIDTH, v1.N, v1.N).copy()
    if (not np.all(np.isfinite(final)) or runtime.macro_step() != v1.STEPS
            or not np.isclose(runtime.time(), v1.STEPS * v1.DT, rtol=0, atol=1e-14)):
        raise ValueError("profiled state or clock is invalid")
    if not np.allclose(final.sum(axis=(1, 2)), initial.sum(axis=(1, 2)),
                       rtol=0, atol=v1.MASS_ATOL):
        raise ValueError("profiled run violated component conservation")
    return final, {"profile_source": summary.source, "native_snapshot": native_snapshot,
                   "summary": summary.to_dict(), "raw_counters": counters,
                   "selected_counters": _selected_counters(counters),
                   "raw_scopes": summary.scopes()}


def _worker(args) -> None:
    root = Path(args.package_root).resolve()
    v1 = _frozen_v1_1()
    identity = _identity(v1, root, args.lane)
    sys.path.insert(0, str(root))  # -I removes PYTHONPATH and the current checkout.
    import numpy as np
    import pops
    from pops._native_selector import select_native_dimension

    select_native_dimension(2)
    from pops import _pops
    from pops.codegen import abi, toolchain
    from pops.runtime.doctor import doctor

    if Path(pops.__file__).resolve() != root / "pops" / "__init__.py":
        raise ValueError("PoPS import escaped its authenticated snapshot")
    if v1._sha(Path(_pops.__file__)) != identity["native_sha256"]:
        raise ValueError("loaded native extension differs from snapshot identity")
    include = Path(toolchain.pops_include()).resolve()
    if include != root / "pops" / "include":
        raise ValueError("compiler would read headers outside this snapshot")
    if toolchain.pops_header_signature(include) != abi.module_header_signature():
        raise ValueError("snapshot headers differ from native ABI")
    checks = doctor(verbose=False)
    if not checks or any(passed is not True for passed, _ in checks.values()):
        raise ValueError("snapshot doctor failed")

    case, layout, subject = v1._case()
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    platform_manifest = artifact.platform_manifest
    facts = {name: getattr(platform_manifest, name).require("platform." + name)
             for name in ("backend", "target", "device", "communicator")}
    facts["precision"] = platform_manifest.precision.storage.require("platform.precision.storage")
    if (facts["backend"] != "production" or facts["target"] != "system"
            or facts["device"] not in {"cpu", "host"} or facts["precision"] != "float64"
            or facts["communicator"] != "MPI_COMM_WORLD"):
        raise ValueError(f"resource probe needs fixed CPU/MPI/float64 route: {facts}")
    dsos = v1._dso_sizes(artifact)
    initial = v1._initial()
    warmup_final = None
    for _ in range(WARMUPS):
        state, _, _ = v1._run_once(artifact, subject, initial)
        if warmup_final is not None and not np.array_equal(warmup_final, state):
            raise ValueError("unprofiled fresh instances produced different accepted states")
        warmup_final = state
    samples = []
    final = None
    for _ in range(PROFILED_RUNS):
        state, profile = _profile_once(v1, artifact, subject, initial)
        if not np.array_equal(warmup_final, state):
            raise ValueError("profiling changed the accepted state")
        if final is None:
            final = state
        elif not np.array_equal(final, state):
            raise ValueError("profiled fresh instances produced different accepted states")
        samples.append(profile)
    output = Path(args.output)
    state_path = output.with_suffix(".npy")
    np.save(state_path, final)
    result = {
        "schema": SCHEMA, "lane": args.lane, "source_commit": identity["source_commit"],
        "native_sha256": identity["native_sha256"],
        "package_file": str(Path(pops.__file__).resolve()),
        "native_file": str(Path(_pops.__file__).resolve()),
        "include_root": str(include), "header_signature": abi.module_header_signature(),
        "doctor_checks": sorted(checks), "python_executable": sys.executable,
        "abi_key": str(artifact.abi_key),
        "abi_environment": v1._abi_environment(str(artifact.abi_key)),
        "compiler": str(artifact.cxx), "cxx_standard": str(artifact.std),
        "sdk_version": v1._sdk_version(), "platform_facts": facts,
        "dso_bytes": dsos,
        "compiler_provenance": v1._compiler_provenance(
            tuple(Path(args.codegen_dir).rglob("*.cpp"))),
        "loaded_external_images": v1._loaded_dependency_fingerprint(root, dsos),
        "state_file": str(state_path), "state_sha256": v1._sha(state_path),
        "samples": samples,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _invoke(args, lane: str, root: Path, output: Path) -> dict:
    receipt = output / f"{lane}.json"
    cache = output / "cache" / lane
    codegen = output / "codegen" / lane
    cache.mkdir(parents=True)
    codegen.mkdir(parents=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("POPS_INCLUDE", None)
    env.pop("POPS_NATIVE_VARIANTS_ROOT", None)
    env.update({"PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "POPS_PROFILE": "off", "POPS_CACHE_DIR": str(cache),
                "POPS_CODEGEN_DIR": str(codegen), "POPS_KEEP_GENERATED": "1",
                "POPS_DUMP_CPP": "1"})
    command = [str(Path(args.python).resolve()), "-I", str(Path(__file__).resolve()),
               "--worker", "--lane", lane, "--package-root", str(root),
               "--codegen-dir", str(codegen), "--output", str(receipt)]
    stdout_path, stderr_path = output / f"{lane}.stdout.log", output / f"{lane}.stderr.log"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, env=env, cwd=output, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            exit_code = process.wait(timeout=WORKER_TIMEOUT_S)
        except subprocess.TimeoutExpired as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            (output / f"{lane}.failed.json").write_text(json.dumps({
                "status": "timed_out", "timeout_s": WORKER_TIMEOUT_S,
                "stdout": str(stdout_path), "stderr": str(stderr_path)}, indent=2) + "\n")
            raise TimeoutError(f"{lane} resource worker timed out; logs preserved") from error
    if exit_code:
        (output / f"{lane}.failed.json").write_text(json.dumps({
            "status": "failed", "exit_code": exit_code,
            "stdout": str(stdout_path), "stderr": str(stderr_path)}, indent=2) + "\n")
        raise RuntimeError(f"{lane} resource worker failed; see {stderr_path}")
    return json.loads(receipt.read_text(encoding="utf-8"))


def _aggregate(rows: dict[str, dict]) -> dict:
    """Only counters present in every profiled sample on both lanes are comparable."""
    out = {}
    for name in _COUNTER_UNITS:
        values = {}
        for lane, row in rows.items():
            entries = [sample["selected_counters"][name] for sample in row["samples"]]
            if len(entries) == PROFILED_RUNS and all(item["available"] for item in entries):
                values[lane] = [item["value"] for item in entries]
        out[name] = ({"available": True, "unit": _COUNTER_UNITS[name], "values": values}
                     if len(values) == 2 else
                     {"available": False, "reason": "counter absent in at least one lane/sample",
                      "observed_values": values})
    return out


def _driver(args) -> None:
    if sys.platform != "darwin":
        raise ValueError("retained Darwin artifacts require macOS dyld verification")
    v1 = _frozen_v1_1()
    roots = {"baseline": Path(args.baseline_root).resolve(),
             "candidate": Path(args.candidate_root).resolve()}
    identities = {lane: _identity(v1, root, lane) for lane, root in roots.items()}
    python = Path(args.python).resolve()
    if not python.is_file():
        raise ValueError("the single shared Python executable is absent")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("plan and execution each require a new empty output directory")
    plan = {
        "schema": SCHEMA, "status": "planned" if not args.execute else "running",
        "case_protocol_sha256": V1_1_SHA256, "resource_script_sha256": _sha(Path(__file__)),
        "case": {"N": v1.N, "width": v1.WIDTH, "steps": v1.STEPS, "dt": v1.DT,
                 "rtol": v1.RTOL, "atol": v1.ATOL, "mass_atol": v1.MASS_ATOL},
        "sampling": {"unprofiled_warmups_per_lane": WARMUPS,
                     "profiled_fresh_instances_per_lane": PROFILED_RUNS,
                     "profile_level": "advanced", "boundary": "run only; bind excluded"},
        "worker_timeout_s": WORKER_TIMEOUT_S, "python": str(python),
        "identities": {lane: {key: identity[key] for key in
                        ("source_commit", "native_sha256", "source_files_sha256")}
                       for lane, identity in identities.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if not args.execute:
        print(output / "plan.json")
        return
    rows = {lane: _invoke(args, lane, roots[lane], output) for lane in
            ("baseline", "candidate")}
    for row in rows.values():
        if row["loaded_external_images"].get("available") is not True:
            raise ValueError("dyld dependency inventory unavailable")
    comparable = ("platform_facts", "abi_environment", "compiler", "cxx_standard",
                  "sdk_version", "loaded_external_images")
    for key in comparable:
        if rows["baseline"][key] != rows["candidate"][key]:
            raise ValueError(f"resource lanes differ in {key}")
    import numpy as np

    np.testing.assert_allclose(np.load(rows["baseline"]["state_file"]),
                               np.load(rows["candidate"]["state_file"]),
                               rtol=v1.RTOL, atol=v1.ATOL)
    report = {**plan, "status": "observed", "workers": rows,
              "counter_comparison": _aggregate(rows),
              "numerical_equivalence": {"passed": True, "rtol": v1.RTOL, "atol": v1.ATOL}}
    (output / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output / "result.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root")
    parser.add_argument("--candidate-root")
    parser.add_argument("--python")
    parser.add_argument("--package-root")
    parser.add_argument("--lane", choices=("baseline", "candidate"))
    parser.add_argument("--codegen-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    needed = (("package_root", "lane", "codegen_dir") if args.worker else
              ("baseline_root", "candidate_root", "python"))
    for name in needed:
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if args.worker:
        _worker(args)
    else:
        _driver(args)


if __name__ == "__main__":
    main()
