"""Receive the 28+12 corpus in separate authentic installed-PoPS processes.

This driver only launches the reception recorder. It supplies no equations,
numerical stepping or alternate runtime. A child receipt is insufficient when
the child fails to exit cleanly, including during MPI finalization.
"""
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


CORPUS = tuple(f"M{i:02d}" for i in range(1, 29)) + tuple(f"T{i:02d}" for i in range(1, 13))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(command, *, environment, log, timeout):
    start = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.Popen(command, env=environment, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Terminate this invocation's process group, including its compiler.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    return {"exit_code": code, "timed_out": timed_out,
            "elapsed_seconds": time.monotonic()-start,
            "log": str(log), "log_sha256": digest(log)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", choices=CORPUS, default=CORPUS)
    parser.add_argument("--timeout", type=int, default=900,
                        help="watchdog in seconds for each native profile, including compilation")
    args = parser.parse_args()
    if args.timeout <= 0 or len(set(args.ids)) != len(args.ids):
        parser.error("require a positive watchdog and distinct corpus identifiers")
    output = args.output.resolve()
    if output.exists():
        parser.error("use a new output directory to preserve previous evidence")
    output.mkdir(parents=True)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    repository = Path(__file__).resolve().parents[2]
    report = {"schema": "pops.api02.native-campaign@1", "python": sys.executable,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=repository, text=True).strip(),
        "source_diff_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "diff", "HEAD", "--binary"], cwd=repository)).hexdigest(),
        "requested_profiles": list(args.ids), "results": [],
        "scope": "installed PoPS; CPU Kokkos; one MPI rank per profile; Dim1 HyQMOM/ALE and Dim2 other profiles",
        "complete_api_02_conformance": False,
        "qualification": "only the explicit equations, variants and checks in each profile"}
    installed_by_dimension = {}
    common_installation = None
    scientific_sources = None
    for identifier in args.ids:
        dimension = 1 if identifier in ("M15", "M28") else 2
        destination = output / identifier
        command = [sys.executable, "-m", "examples.api_v02.reception", "--ids", identifier,
            "--dimension", str(dimension), "--output", str(destination)]
        row = {"id": identifier, "dimension": dimension, "command": command}
        row.update(execute(command, environment=environment, log=output/(identifier+".log"), timeout=args.timeout))
        receipt = destination / "result.json"
        row["passed"] = False
        if receipt.is_file():
            child = json.loads(receipt.read_text())
            row["receipt"] = str(receipt)
            row["receipt_sha256"] = digest(receipt)
            row["profiles"] = child["results"]
            identity = child["package_before"]
            previous = installed_by_dimension.setdefault(dimension, identity)
            common = {key: identity[key] for key in ("package", "python_sources_sha256",
                                                       "sdk_headers", "native_variants")}
            if common_installation is None:
                common_installation = common
            row["same_installation"] = previous == identity and common_installation == common
            sources = [profile.get("scientific_sources") for profile in child["results"]]
            if scientific_sources is None and sources:
                scientific_sources = sources[0]
            row["same_scientific_sources"] = bool(sources) and all(
                source is not None and source == scientific_sources for source in sources)
            row["passed"] = (row["exit_code"] == 0 and child.get("package_unchanged") is True
                and child.get("all_executed_and_assessed") is True and row["same_installation"]
                and row["same_scientific_sources"])
        report["results"].append(row)
        (output/"campaign.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({k:row[k] for k in ("id", "passed", "exit_code", "elapsed_seconds")}),flush=True)
    report["source_diff_sha256_after"] = hashlib.sha256(subprocess.check_output(
        ["git", "diff", "HEAD", "--binary"], cwd=repository)).hexdigest()
    report["source_unchanged"] = report["source_diff_sha256_after"] == report["source_diff_sha256"]
    report["scientific_sources"] = scientific_sources
    report["all_requested_profiles_passed"] = (report["source_unchanged"] and
        all(row["passed"] for row in report["results"]))
    report["entire_corpus_received"] = set(args.ids) == set(CORPUS) and report["all_requested_profiles_passed"]
    report["installed_by_dimension"] = installed_by_dimension
    (output/"campaign.json").write_text(json.dumps(report, indent=2)+"\n")
    return 0 if report["all_requested_profiles_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
