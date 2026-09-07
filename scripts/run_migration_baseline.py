#!/usr/bin/env python3
"""Replay the frozen M0 reference examples and preserve exact command evidence.

This runner does not replace a convergence campaign. Example-owned numerical and
restart assertions remain authoritative, and failures are preserved in the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--profile", action="append", choices=(
            "scalar_tutorial_openmp", "scalar_tutorial_mpi2", "scalar_full",
            "multiphysics_full", "imex_amr_full",
        ), help="select complete profiles for independent replay; default: all five",
    )
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    python = str(args.python.resolve())
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(POPS_NATIVE_DIM="2", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2")
    native = subprocess.run(
        [python, "-c", "import json; from pops._native_selector import select_native_dimension; "
         "select_native_dimension(2); import pops; from pops import _pops; "
         "print(json.dumps({'package':pops.__file__,'native':_pops.__file__}))"],
        env=environment, check=True, capture_output=True, text=True,
    )
    installation = json.loads(native.stdout)
    extension = Path(installation["native"])
    installation["sha256"] = hashlib.sha256(extension.read_bytes()).hexdigest()
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"], check=True,
        capture_output=True, text=True,
    ).stdout
    bootstrap = (
        "import runpy,sys; from pops._native_selector import select_native_dimension; "
        "select_native_dimension(2); sys.argv=sys.argv[1:]; "
        "runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    helper = str(source / "scripts/run_installed_example.py")
    profiles = (
        ("scalar_tutorial_openmp", "docs/tuto/scalar_advection/06_openmp_amr_explicit_ssprk2.py", (), 1),
        ("scalar_tutorial_mpi2", "docs/tuto/scalar_advection/08_mpi_amr_explicit_ssprk2.py", (), 2),
        ("scalar_full", "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py", ("--output-dir", str(output / "scalar_full")), 1),
        ("multiphysics_full", "examples/final/EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py", ("--output-dir", str(output / "multiphysics_full")), 1),
        ("imex_amr_full", "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py", ("--output-dir", str(output / "imex_amr_full")), 1),
    )
    if args.profile:
        selected = set(args.profile)
        profiles = tuple(row for row in profiles if row[0] in selected)
    report = {
        "schema": "pops.migration.m0.reference-examples.v1", "phase": args.phase,
        "source": str(source), "revision": revision, "dirty": bool(status),
        "platform": platform.platform(), "python": python, "installation": installation,
        "dimension": 2, "omp_num_threads": 2, "profiles": [],
        "selected_profiles": [row[0] for row in profiles],
        "limitations": ["No GPU qualification", "Reference examples, not a manufactured-solution convergence matrix", "No process-loss tolerance claim"],
    }
    for name, path, forwarded, ranks in profiles:
        command = [python, "-c", bootstrap, helper, "--runtime-sha256", installation["sha256"],
                   "--example", str(source / path), "--", *forwarded]
        if ranks > 1:
            command = [str(args.python.parent / "mpiexec"), "-n", str(ranks), *command]
        log = output / (name + ".log")
        started = time.monotonic()
        timeout = False
        with log.open("w") as stream:
            try:
                completed = subprocess.run(command, cwd=source, env=environment,
                                           stdout=stream, stderr=subprocess.STDOUT,
                                           timeout=args.timeout, check=False)
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code, timeout = 124, True
        row = {"profile": name, "command": command, "mpi_ranks": ranks,
               "returncode": code, "timeout": timeout,
               "seconds": time.monotonic() - started, "log": str(log),
               "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
        report["profiles"].append(row)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: row[key] for key in ("profile", "returncode", "seconds", "log")}), flush=True)
    return int(any(row["returncode"] for row in report["profiles"]))


if __name__ == "__main__":
    raise SystemExit(main())
