#!/usr/bin/env python3
"""Authenticated, paired 3d06cab -> T2 scalar-User performance probe.

The baseline cannot author a joint User body. This runner compares the *same*
five-component scalar User/Rusanov calculation on both releases. It does not
attribute a scalar regression or improvement to the new joint policy itself.
This v1.1 rerun corrects only the CompiledArtifact.layout_program_paths
property access. See joint_reconstruction_benchmark_v1_1.md first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import statistics
import subprocess
import sys
import time


SCHEMA = "pops.api040.joint-reconstruction-comparison.v1.1"
N, WIDTH, STEPS, DT = 48, 5, 12, 1.0e-4
WARMUPS, SAMPLES = 2, 5
RTOL, ATOL, MASS_ATOL = 1.0e-12, 1.0e-12, 1.0e-10
COMPILE_TIMEOUT_S, RUNTIME_TIMEOUT_S = 600, 180
ORDER = ("baseline", "candidate", "candidate", "baseline")
BASELINE_COMMIT = "3d06cabee9db4a31c4d07fa4e00dc5a40d164155"
BASELINE_NATIVE = "4574ed6096650aa708ad040fd6ee056414a3cfa1735a0440bd74220170265beb"
BASELINE_SOURCES = "2a76043ebff3bb5405d3408b39867856b9dd635d67a72ca9040d6535794300ce"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _abi_environment(key: str) -> dict[str, str]:
    fields = dict(token.split("=", 1) for token in key.split(";") if "=" in token)
    required = {"compiler", "std", "kokkos", "stdlib", "mpi", "mpi_abi", "dim"}
    if not required <= fields.keys():
        raise ValueError("compiled artifact ABI omits comparable compiler/runtime fields")
    return {name: fields[name] for name in sorted(required)}


def _load_identity(root: Path) -> dict:
    identity_path = root / "identity.json"
    source_path = root / "source-files.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    source_files = json.loads(source_path.read_text(encoding="utf-8"))
    if identity.get("schema_version") != 1 or not isinstance(source_files, dict) or not source_files:
        raise ValueError("artifact snapshot has no exact identity/source-file contract")
    if identity.get("source_diff_sha256") != hashlib.sha256(b"").hexdigest():
        raise ValueError("artifact snapshot was not built from a clean source tree")
    if identity.get("verified_source_files") != len(source_files):
        raise ValueError("artifact source-file count differs from its identity")
    if _sha(source_path) != identity.get("source_files_sha256"):
        raise ValueError("artifact source manifest differs from its recorded digest")
    for name, expected in source_files.items():
        if (not isinstance(name, str) or not isinstance(expected, str)
                or ".." in Path(name).parts or Path(name).is_absolute()):
            raise ValueError("source manifest contains an unsafe member")
        if name.startswith("python/pops/"):
            member = root / "pops" / name.removeprefix("python/pops/")
        elif name.startswith("include/"):
            member = root / "pops" / name
        else:
            raise ValueError(f"unrecognized source member {name!r}")
        if not member.resolve().is_relative_to(root):
            raise ValueError("source manifest escaped the snapshot")
        if not member.is_file() or _sha(member) != expected:
            raise ValueError(f"source member differs from snapshot identity: {name}")
    native = tuple((root / "pops" / "_native" / "dim2").glob("_pops*.so"))
    if len(native) != 1 or _sha(native[0]) != identity.get("native_sha256"):
        raise ValueError("artifact native Dim=2 extension differs from its identity")
    if not (root / "pops" / "__init__.py").is_file():
        raise ValueError("artifact has no installed PoPS package")
    return identity


def _initial():
    import numpy as np

    coordinate = (np.arange(N) + .5) / N
    y, x = np.meshgrid(coordinate, coordinate, indexing="ij")
    return np.ascontiguousarray(np.stack([
        2.0 + .2 * component
        + .11 * np.sinc(1 / N) * np.sin(2 * np.pi * x + .3 * component)
        + .07 * np.sinc(2 / N) * np.cos(4 * np.pi * y - .2 * component)
        for component in range(WIDTH)
    ]))


def _case():
    import pops
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.initial import InitialCondition
    from pops.layouts import Uniform
    from pops.lib.initial import BindArray
    from pops.math import ddt, div
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.projection import ConservativeCellAverage
    from pops.time import FixedDt

    frame = Rectangle("joint_perf_square", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("joint_perf_linear", frame=frame)
    state = model.state("U", components=tuple(f"q{i}" for i in range(WIDTH)))
    velocity = (.7, -.4)
    flux = model.flux("linear", frame=frame, state=state,
        components={axis: tuple(speed * component for component in state)
                    for axis, speed in zip(frame.axes, velocity, strict=True)},
        waves={axis: (abs(speed),) * WIDTH
               for axis, speed in zip(frame.axes, velocity, strict=True)})
    rate = model.rate("transport", equation=ddt(state) == -div(flux))
    method = FiniteVolume(flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.User(
            lambda sample: sample(0) + .2 * (sample(1) - sample(-1)), formal_order=1),
        riemann=riemann.Rusanov())
    case = pops.Case("joint_perf_case")
    block = case.block("field", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    program = pops.Program("joint_perf_euler")
    current = program.state(block[state])
    residual = rate(current.n)
    program.commit(current.next, program.value(
        "accepted", current.n + program.dt * residual, at=current.next.point))
    program.step_strategy(FixedDt(DT))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(N, N),
                                   periodic=PeriodicAxes(frame.axes)))
    return case, layout, case.resolve(block[state])


def _execution_context(artifact):
    from pops import _pops
    from pops._native_collectives import require_world
    from pops._platform_contracts import ExecutionContext, ExecutionResource, validate_launch
    from pops.runtime._platform_manifest import native_runtime_backend, native_device_resource

    platform_manifest = artifact.platform_manifest
    backend = native_runtime_backend(platform_manifest)
    communicator_name = platform_manifest.communicator.require("artifact.communicator")
    datatype_name = platform_manifest.precision.storage.require("artifact.precision.storage")
    if communicator_name == "MPI_COMM_WORLD":
        world = require_world(_pops.mpi_world())
        if world.size != 1 or datatype_name != "float64":
            raise ValueError("benchmark requires one native rank and float64")
        communicator = ExecutionResource("communicator", "MPI_COMM_WORLD", handle=world)
        datatype = ExecutionResource("datatype", datatype_name,
                                     handle=world.datatype_float64)
    else:
        raise ValueError(f"benchmark requires identical MPI-enabled one-rank lanes, got {communicator_name!r}")
    context = ExecutionContext(backend=backend, communicator=communicator,
                               datatype=datatype, device=native_device_resource(backend))
    validate_launch(platform_manifest, context, ())
    return context


def _dso_sizes(artifact):
    paths = {Path(block.model.so_path) for block in artifact.blocks}
    paths.update(Path(item) for item in artifact.layout_program_paths.values())
    paths.add(Path(artifact.so_path))
    return {str(path): {"bytes": path.stat().st_size, "sha256": _sha(path)}
            for path in sorted(paths)}


def _compiler_provenance(generated):
    fields = ("cxx", "std", "cflags", "lflags", "compile_command")
    rows = {}
    for path in generated:
        with path.open(encoding="utf-8", errors="replace") as stream:
            prefix = stream.read(16384)
        if not prefix.startswith("/*\npops.compile provenance banner"):
            continue
        values = {}
        for line in prefix.splitlines():
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            if name.strip() in fields:
                values[name.strip()] = value.strip()
        if set(values) == set(fields):
            rows[str(path)] = values
    return ({"available": True, "sources": rows} if rows else
            {"available": False, "reason": "no retained C++ provenance banner"})


def _sdk_version():
    if sys.platform != "darwin":
        return {"available": False, "reason": "macOS SDK probe does not apply"}
    try:
        result = subprocess.run(("xcrun", "--show-sdk-version"), capture_output=True,
                                text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "reason": type(error).__name__}
    if result.returncode or not result.stdout.strip():
        return {"available": False, "reason": result.stderr.strip() or "xcrun failed"}
    return {"available": True, "version": result.stdout.strip()}


def _loaded_dependency_fingerprint(root: Path, owned_dsos):
    """Hash actual non-system images loaded by dyld, excluding the two changing PoPS DSOs."""
    if sys.platform != "darwin":
        return {"available": False, "reason": "dyld image inventory is macOS-specific"}
    import ctypes

    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    members = {}
    owned = {Path(item).resolve() for item in owned_dsos}
    for index in range(dyld._dyld_image_count()):
        raw = dyld._dyld_get_image_name(index)
        if raw is None:
            continue
        path = Path(raw.decode()).resolve()
        if path.is_relative_to(root) or path in owned:
            continue
        if path.is_file() and (path.suffix in {".dylib", ".so"} or ".framework/" in str(path)):
            members[str(path)] = _sha(path)
    return {"available": True, "images": members}


def _run_once(artifact, subject, initial):
    import numpy as np
    import pops

    started = time.perf_counter_ns()
    runtime = pops.bind(artifact, initial_values={subject: initial.copy()},
                        resources={"execution_context": _execution_context(artifact)})
    bind_s = (time.perf_counter_ns() - started) / 1e9
    started = time.perf_counter_ns()
    pops.run(runtime, t_end=STEPS * DT, max_steps=STEPS, console=False)
    step_s = (time.perf_counter_ns() - started) / 1e9
    final = np.asarray(runtime.state_global("field")).reshape(WIDTH, N, N).copy()
    if (not np.all(np.isfinite(final)) or runtime.macro_step() != STEPS
            or not np.isclose(runtime.time(), STEPS * DT, rtol=0, atol=1e-14)):
        raise ValueError("accepted state or clock is invalid")
    if not np.allclose(final.sum(axis=(1, 2)), initial.sum(axis=(1, 2)),
                       rtol=0, atol=MASS_ATOL):
        raise ValueError("component conservation failed")
    return final, bind_s, step_s


def _worker(args):
    # -I removes PYTHONPATH/current checkout; add only the exact snapshot before importing pops.
    root = Path(args.package_root).resolve()
    identity = _load_identity(root)
    if identity["source_commit"] != args.expected_commit:
        raise ValueError("worker imported a different source revision")
    sys.path.insert(0, str(root))
    import numpy as np
    import pops
    from pops._native_selector import select_native_dimension

    select_native_dimension(2)
    from pops import _pops
    from pops.codegen import abi, toolchain
    from pops.runtime.doctor import doctor

    if Path(pops.__file__).resolve() != root / "pops" / "__init__.py":
        raise ValueError("worker imported PoPS from outside its authenticated snapshot")
    if _sha(Path(_pops.__file__)) != identity["native_sha256"]:
        raise ValueError("worker loaded the wrong native extension")
    include = Path(toolchain.pops_include()).resolve()
    if include != root / "pops" / "include":
        raise ValueError("worker compiler would read headers outside its snapshot")
    if toolchain.pops_header_signature(include) != abi.module_header_signature():
        raise ValueError("snapshot headers differ from the loaded native ABI")
    checks = doctor(verbose=False)
    if not checks or any(passed is not True for passed, _ in checks.values()):
        raise ValueError("snapshot doctor did not pass all checks")
    case, layout, subject = _case()
    started = time.perf_counter_ns()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    resolve_s = (time.perf_counter_ns() - started) / 1e9
    started = time.perf_counter_ns()
    artifact = pops.compile(resolved)
    compile_s = (time.perf_counter_ns() - started) / 1e9
    platform_manifest = artifact.platform_manifest
    platform_facts = {
        "backend": platform_manifest.backend.require("platform.backend"),
        "target": platform_manifest.target.require("platform.target"),
        "precision": platform_manifest.precision.storage.require("platform.precision.storage"),
        "device": platform_manifest.device.require("platform.device"),
        "communicator": platform_manifest.communicator.require("platform.communicator"),
    }
    if (platform_facts["backend"] != "production" or platform_facts["target"] != "system"
            or platform_facts["precision"] != "float64"
            or platform_facts["device"] not in {"cpu", "host"}
            or platform_facts["communicator"] != "MPI_COMM_WORLD"):
        raise ValueError(f"benchmark artifact does not have the fixed CPU/MPI/float64 route: {platform_facts}")
    dso = _dso_sizes(artifact)
    generated = tuple(Path(args.codegen_dir).rglob("*.cpp"))
    result = {
        "schema": SCHEMA, "lane": args.lane, "phase": args.phase,
        "source_commit": identity["source_commit"],
        "native_sha256": identity["native_sha256"],
        "package_file": str(Path(pops.__file__).resolve()),
        "native_file": str(Path(_pops.__file__).resolve()),
        "include_root": str(include), "header_signature": abi.module_header_signature(),
        "doctor_checks": sorted(checks), "python_executable": sys.executable,
        "abi_key": str(artifact.abi_key),
        "abi_environment": _abi_environment(str(artifact.abi_key)),
        "compiler": str(artifact.cxx), "cxx_standard": str(artifact.std),
        "platform_facts": platform_facts,
        "resolve_s": resolve_s, "compile_s": compile_s,
        "dso_bytes": dso,
        "compiler_provenance": _compiler_provenance(generated),
        "sdk_version": _sdk_version(),
        "generated_cpp": ({"available": True, "bytes": sum(path.stat().st_size for path in generated),
                           "count": len(generated)} if generated else
                          {"available": False, "bytes": None, "count": 0,
                           "reason": "requested C++ dumps were not retained"}),
        "process_peak_rss_unit": "bytes" if sys.platform == "darwin" else "KiB",
    }
    if args.phase == "runtime":
        initial = _initial()
        for _ in range(WARMUPS):
            _run_once(artifact, subject, initial)
        bind_samples, step_samples, final = [], [], None
        for _ in range(SAMPLES):
            candidate, bind_s, step_s = _run_once(artifact, subject, initial)
            bind_samples.append(bind_s)
            step_samples.append(step_s)
            if final is None:
                final = candidate
            elif not np.array_equal(final, candidate):
                raise ValueError("repeated runs did not reproduce the same final state")
        state_path = Path(args.output).with_suffix(".npy")
        np.save(state_path, final)
        result.update({"bind_samples_s": bind_samples, "step_samples_s": step_samples,
                       "state_file": str(state_path),
                       "state_sha256": _sha(state_path),
                       "native_counters": {"available": False,
                           "reason": "RuntimeInstance has no public symmetric profiler; no "
                                     "native allocation/kernel/communication count is inferred"}})
    result["process_peak_rss_raw"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["loaded_external_images"] = _loaded_dependency_fingerprint(root, dso)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")


def _invoke(args, lane, phase, index, identity, root, python, output):
    out = output / f"{index:02d}-{lane}-{phase}.json"
    cache = output / "cache" / lane
    codegen = output / "codegen" / lane
    if phase == "compile" and cache.exists() and any(cache.iterdir()):
        raise ValueError(f"cold JIT cache is not empty: {cache}")
    cache.mkdir(parents=True, exist_ok=True)
    codegen.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update({"PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "POPS_CACHE_DIR": str(cache), "POPS_CODEGEN_DIR": str(codegen),
                "POPS_KEEP_GENERATED": "1", "POPS_DUMP_CPP": "1"})
    command = [str(python), "-I", str(Path(__file__).resolve()), "--worker",
               "--package-root", str(root), "--expected-commit", identity["source_commit"],
               "--lane", lane, "--phase", phase, "--codegen-dir", str(codegen),
               "--output", str(out)]
    stdout_path = output / f"{index:02d}-{lane}-{phase}.stdout.log"
    stderr_path = output / f"{index:02d}-{lane}-{phase}.stderr.log"
    timeout = COMPILE_TIMEOUT_S if phase == "compile" else RUNTIME_TIMEOUT_S
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, env=env, cwd=output,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            (output / f"{index:02d}-{lane}-{phase}.failed.json").write_text(json.dumps({
                "status": "timed_out", "lane": lane, "phase": phase, "timeout_s": timeout,
                "stdout": str(stdout_path), "stderr": str(stderr_path)}, indent=2) + "\n")
            raise TimeoutError(f"{lane} {phase} exceeded {timeout}s; logs preserved") from error
    if exit_code:
        (output / f"{index:02d}-{lane}-{phase}.failed.json").write_text(json.dumps({
            "status": "failed", "lane": lane, "phase": phase,
            "exit_code": exit_code,
            "stdout": str(stdout_path), "stderr": str(stderr_path)}, indent=2) + "\n")
        raise RuntimeError(f"{lane} {phase} failed ({exit_code}); see {stderr_path}")
    return json.loads(out.read_text(encoding="utf-8"))


def _driver(args):
    if sys.platform != "darwin":
        raise ValueError("retained Darwin native artifacts require macOS dyld verification")
    baseline, candidate = (Path(args.baseline_root).resolve(),
                           Path(args.candidate_root).resolve())
    identities = {"baseline": _load_identity(baseline), "candidate": _load_identity(candidate)}
    if identities["baseline"]["source_commit"] != BASELINE_COMMIT:
        raise ValueError("baseline is not the retained 3d06cab snapshot")
    if identities["baseline"]["native_sha256"] != BASELINE_NATIVE:
        raise ValueError("baseline native artifact is not the retained 4574ed60 build")
    if (identities["baseline"]["source_files_sha256"] != BASELINE_SOURCES
            or identities["baseline"]["verified_source_files"] != 1079):
        raise ValueError("baseline source manifest is not the retained 1079-member snapshot")
    if identities["candidate"]["source_commit"] != args.expected_candidate_commit:
        raise ValueError("candidate source differs from the independently requested revision")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("plan and execution each require a new empty results directory")
    plan = {"schema": SCHEMA, "scenario": {"N": N, "width": WIDTH, "steps": STEPS,
        "dt": DT, "physics": "linear advection x=.7 y=-.4; scalar User slope=.2; Rusanov",
        "tolerances": {"rtol": RTOL, "atol": ATOL, "mass_atol": MASS_ATOL}},
        "sampling": {"warmups_per_worker": WARMUPS, "timed_samples_per_worker": SAMPLES,
                     "worker_order": ORDER},
        "budgets_s": {"cold_compile_each": COMPILE_TIMEOUT_S,
                      "runtime_worker_each": RUNTIME_TIMEOUT_S},
        "identities": {name: {"source_commit": item["source_commit"],
                              "native_sha256": item["native_sha256"]}
                       for name, item in identities.items()},
        "host": {"platform": platform.platform(), "python": sys.version,
                 "cpu_count": os.cpu_count()},
        "benchmark_script_sha256": _sha(Path(__file__)),
        "status": "planned" if not args.execute else "running"}
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if not args.execute:
        print(output / "plan.json")
        return
    roots = {"baseline": baseline, "candidate": candidate}
    pythons = {"baseline": Path(args.baseline_python).resolve(),
               "candidate": Path(args.candidate_python).resolve()}
    if pythons["baseline"] != pythons["candidate"]:
        raise ValueError("paired CPU comparison requires one identical Python/runtime prefix")
    if not pythons["baseline"].is_file():
        raise ValueError("the paired Python executable is absent")
    cold = {lane: _invoke(args, lane, "compile", index, identities[lane], roots[lane],
                          pythons[lane], output) for index, lane in enumerate(("baseline", "candidate"))}
    runs = [_invoke(args, lane, "runtime", index + 2, identities[lane], roots[lane],
                    pythons[lane], output) for index, lane in enumerate(ORDER)]
    for row in runs:
        parent = cold[row["lane"]]
        if (row["abi_key"] != parent["abi_key"]
                or row["header_signature"] != parent["header_signature"]
                or row["dso_bytes"] != parent["dso_bytes"]
                or row["platform_facts"] != parent["platform_facts"]
                or row["abi_environment"] != parent["abi_environment"]
                or row["compiler"] != parent["compiler"]
                or row["cxx_standard"] != parent["cxx_standard"]
                or row["sdk_version"] != parent["sdk_version"]):
            raise ValueError("runtime worker did not load its lane's cold-compiled DSOs")
    if cold["baseline"]["platform_facts"] != cold["candidate"]["platform_facts"]:
        raise ValueError("baseline and candidate use different execution platform contracts")
    if any(row["loaded_external_images"].get("available") is not True for row in runs):
        raise ValueError("runtime dependency inventory is unavailable for one worker")
    if (cold["baseline"]["abi_environment"] != cold["candidate"]["abi_environment"]
            or cold["baseline"]["compiler"] != cold["candidate"]["compiler"]
            or cold["baseline"]["cxx_standard"] != cold["candidate"]["cxx_standard"]
            or cold["baseline"]["sdk_version"] != cold["candidate"]["sdk_version"]):
        raise ValueError("baseline and candidate use different compiler/runtime ABI fields")
    if any(row["loaded_external_images"] != runs[0]["loaded_external_images"] for row in runs[1:]):
        raise ValueError("baseline and candidate loaded different external library images")
    import numpy as np

    reference = np.load(runs[0]["state_file"])
    for row in runs[1:]:
        np.testing.assert_allclose(np.load(row["state_file"]), reference,
                                   rtol=RTOL, atol=ATOL)
    times = {lane: [sample for row in runs if row["lane"] == lane
                    for sample in row["step_samples_s"]]
             for lane in ("baseline", "candidate")}
    medians = {lane: statistics.median(values) for lane, values in times.items()}
    dispersions = {lane: statistics.median(abs(value - medians[lane]) for value in values)
                   for lane, values in times.items()}
    binds = {lane: [sample for row in runs if row["lane"] == lane
                    for sample in row["bind_samples_s"]]
             for lane in ("baseline", "candidate")}
    report = {**plan, "status": "measured", "cold_compile": cold,
              "runtime_workers": runs, "step_median_s": medians, "step_mad_s": dispersions,
              "bind_median_s": {lane: statistics.median(values)
                                for lane, values in binds.items()},
              "candidate_over_baseline_median": medians["candidate"] / medians["baseline"],
              "numerical_equivalence": {"rtol": RTOL, "atol": ATOL, "passed": True}}
    (output / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output / "result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root")
    parser.add_argument("--candidate-root")
    parser.add_argument("--baseline-python")
    parser.add_argument("--candidate-python")
    parser.add_argument("--expected-candidate-commit")
    parser.add_argument("--package-root")
    parser.add_argument("--expected-commit")
    parser.add_argument("--lane")
    parser.add_argument("--phase", choices=("compile", "runtime"))
    parser.add_argument("--codegen-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.worker:
        _worker(args)
    else:
        for name in ("baseline_root", "candidate_root", "baseline_python", "candidate_python",
                     "expected_candidate_commit"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required")
        _driver(args)


if __name__ == "__main__":
    main()
