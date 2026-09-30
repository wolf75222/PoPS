#!/usr/bin/env python3
"""Reproduce full scientific cases against the authenticated installed package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples/migration/scientific"
CASES = {
    "m01": ("api040_m01_cell_averages.py", "result.json", {}),
    "m02-shock": ("api040_m02_burgers.py", "shock/receipt.json",
                  {"POPS_API040_M02_CASE": "shock"}),
    "m02-rarefaction": ("api040_m02_burgers.py", "rarefaction/receipt.json",
                        {"POPS_API040_M02_CASE": "rarefaction"}),
    "m03-ideal": ("api040_m03_euler_eos.py", "ideal/receipt.json",
                  {"POPS_API040_M03_CASE": "ideal"}),
    "m03-stiffened": ("api040_m03_euler_eos.py", "stiffened/receipt.json",
                      {"POPS_API040_M03_CASE": "stiffened"}),
    "m04": ("api040_m04_advection_diffusion.py", "receipt.json",
            {"POPS_API040_M04_DIFFUSION": "x_only", "POPS_API040_M04_METHOD": "forward_euler",
             "POPS_API040_M04_STEP_POLICY": "combined_bound"}),
    "m04-isotropic": ("api040_m04_advection_diffusion.py", "receipt.json",
                      {"POPS_API040_M04_DIFFUSION": "isotropic",
                       "POPS_API040_M04_METHOD": "forward_euler", "POPS_API040_M04_STEP_POLICY": "combined_bound"}),
    "m04-ssprk2": ("api040_m04_advection_diffusion.py", "receipt.json",
                   {"POPS_API040_M04_DIFFUSION": "x_only", "POPS_API040_M04_METHOD": "ssprk2",
                    "POPS_API040_M04_STEP_POLICY": "combined_bound"}),
    "m04-fe-fixed-courant": ("api040_m04_advection_diffusion.py", "receipt.json",
                             {"POPS_API040_M04_DIFFUSION": "x_only", "POPS_API040_M04_METHOD": "forward_euler",
                              "POPS_API040_M04_STEP_POLICY": "fixed_courant"}),
    "m17": ("api040_m17_fan_li.py", "reverse/result.json", {}),
    "m06": ("api040_m06_enthalpy.py", "result.json", {}),
    "m07-hydrostatic": ("api040_m07_saint_venant.py", "receipt.json", {}),
    "m08": ("api040_m08_guiding_center.py", "receipt.json", {}),
    "m13": ("api040_m13_reaction_chain.py", "result.json", {}),
    "m15-axial-b1": ("api040_m15_hyqmom_axial_b1.py", "receipt.json", {}),
    "m22-h05": ("api040_m22_h05.py", "result.json", {}),
    "cattaneo": ("structural_cattaneo.py", "result.json", {}),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600.,
                        help="maximum seconds per native scientific phase")
    args = parser.parse_args()
    if args.ranks < 1 or args.threads < 1 or args.timeout <= 0:
        parser.error("ranks, threads and timeout must be positive")
    if "PYTHONPATH" in os.environ:
        parser.error("run with env -u PYTHONPATH")
    if any(key.endswith("_AUTHORING_ONLY") and value != "0"
           for key, value in os.environ.items() if key.startswith("POPS_API040_")):
        parser.error("an authoring-only run cannot qualify native execution")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output must be empty to exclude stale receipts")
    script, receipt_name, overrides = CASES[args.case]
    dimension = 1 if args.case in {"m07-hydrostatic", "m15-axial-b1"} else 2
    environment = dict(os.environ, PYTHONNOUSERSITE="1", POPS_NATIVE_DIM=str(dimension),
                       POPS_REQUIRE_NATIVE_TESTS="1", OMP_NUM_THREADS=str(args.threads),
                       POPS_THREADS=str(args.threads), POPS_API040_OUTPUT=str(output / "states"))
    environment.update(overrides)
    sources = [Path(__file__).resolve(), *sorted(EXAMPLES.glob("api040_*.py")),
               *sorted(EXAMPLES.glob("structural_cattaneo*.py"))]
    manifest = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    (output / "example-sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    identity_runner = Path(__file__).with_name("run_installed_checks.py")

    def authenticate(phase: str) -> int:
        with (output / (phase + ".log")).open("w") as log:
            return subprocess.run([sys.executable, str(identity_runner), "--identity-only",
                                   "--output", str(output / phase)], cwd=ROOT,
                                  env=environment, stdout=log, stderr=subprocess.STDOUT).returncode

    before_code = authenticate("before")
    command = [sys.executable, str(EXAMPLES / script)]
    if args.ranks > 1:
        launcher = Path(sys.prefix) / "bin/mpiexec"
        if not launcher.is_file():
            parser.error(f"MPI launcher absent from active environment: {launcher}")
        command = [str(launcher), "-n", str(args.ranks), *command]
    started = time.monotonic()
    code = None
    phases = []
    if before_code == 0:
        with (output / "run.log").open("w") as log:
            variants = ({"POPS_API040_M17_ORDER": "canonical"},
                        {"POPS_API040_M17_ORDER": "reverse"}) if args.case == "m17" else ({},)
            for variant in variants:
                phase_started = time.monotonic()
                process = subprocess.Popen(command, cwd=ROOT, env=dict(environment, **variant),
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                timed_out = False
                try:
                    code = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    code = 124
                phases.append({"settings": variant, "returncode": code,
                               "timeout": timed_out,
                               "seconds": time.monotonic() - phase_started})
                if code:
                    break
    elapsed = time.monotonic() - started
    after_code = authenticate("after")
    unchanged = manifest == {str(path.relative_to(ROOT)): digest(path) for path in sources}
    scientific_path = output / "states" / receipt_name
    scientific = json.loads(scientific_path.read_text()) if scientific_path.is_file() else {}
    before = json.loads((output / "before/identity.json").read_text()) if before_code == 0 else {}
    after = json.loads((output / "after/identity.json").read_text()) if after_code == 0 else {}
    same_native = bool(before) and before.get("native_sha256") == after.get("native_sha256")
    same_sources = bool(before) and before.get("source_files_sha256") == after.get("source_files_sha256")
    records = scientific.get("records", scientific.get("runs", scientific.get("scenarios", [])))
    if args.case == "m17":
        records = [scientific] if scientific else []
    correct_backend = (bool(records) and all(row.get("mpi_ranks") == args.ranks for row in records)
                       and scientific.get("threads_requested") == str(args.threads))
    passed = (code == 0 and before_code == 0 and after_code == 0 and unchanged
              and same_native and same_sources and correct_backend
              and scientific.get("status") == "passed"
              and scientific.get("native_sha256") == before.get("native_sha256")
              and (args.case != "m17" or scientific.get("permutation_verified") is True))
    result = {"schema_version": 1, "case": args.case,
              "status": "passed" if passed else "failed", "command": command,
              "returncode": code, "seconds": elapsed, "ranks": args.ranks,
              "dimension": dimension,
              "phases": phases,
              "threads": args.threads, "authentication_before": before_code,
              "authentication_after": after_code, "example_sources_unchanged": unchanged,
              "same_native": same_native, "same_shipped_sources": same_sources,
              "correct_backend": correct_backend, "native_sha256": before.get("native_sha256"),
              "scientific_receipt": str(scientific_path.relative_to(output)),
              "scientific_receipt_sha256": digest(scientific_path) if scientific_path.is_file() else None,
              "settings": {key: value for key, value in environment.items()
                           if key.startswith("POPS_API040_") or key in ("OMP_NUM_THREADS", "POPS_THREADS")}}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
