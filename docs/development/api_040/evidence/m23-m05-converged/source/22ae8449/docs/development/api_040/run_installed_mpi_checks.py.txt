#!/usr/bin/env python3
"""Run the same installed-package pytest nodes on every actual native MPI rank.

Each rank owns its log, XML and identity receipt. Counts describe one shared
test selection and are never summed to inflate the number of independent tests.
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
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(output: Path, ranks: int, dimension: int, tests: list[str]) -> int:
    import pops
    from pops._native_selector import select_native_dimension
    from pops._native_collectives import require_world, allgather_value

    native = select_native_dimension(dimension)
    world = require_world(native.mpi_world())
    if world.size != ranks:
        raise RuntimeError(f"expected {ranks} actual native MPI ranks, got {world.size}")
    before = json.loads((output / "before/identity.json").read_text())
    observed = {"rank": world.rank, "ranks": world.size, "dimension": dimension,
                "package_file": str(Path(pops.__file__).resolve()),
                "native_file": str(Path(native.__file__).resolve()),
                "native_sha256": digest(Path(native.__file__))}
    identities = allgather_value(world, observed)
    for rank, row in enumerate(identities):
        if row["rank"] != rank or any(row[key] != before[key] for key in
                                       ("package_file", "native_file", "native_sha256")):
            raise RuntimeError(f"rank {rank} did not import the authenticated installation")
    (output / f"rank{world.rank}.identity.json").write_text(json.dumps(observed, indent=2) + "\n")
    # Redirect file descriptors as well as Python output, to retain native diagnostics.
    with (output / f"rank{world.rank}.log").open("w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        sys.path.insert(0, str(ROOT))  # only repository helpers; never ROOT/python
        import pytest
        code = int(pytest.main(["-v", "--tb=short", "-o", "pythonpath=",
                               f"--junitxml={output / f'rank{world.rank}.xml'}",
                               f"--basetemp={output / f'rank{world.rank}-tmp'}", *tests]))
        sys.stdout.flush()
        sys.stderr.flush()
    return max(allgather_value(world, code))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--dimension", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800.)
    parser.add_argument("--test", action="append", required=True, dest="tests")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if "PYTHONPATH" in os.environ:
        parser.error("run with env -u PYTHONPATH")
    if args.ranks < 2 or args.threads < 1 or args.timeout <= 0:
        parser.error("ranks >= 2, threads >= 1 and positive timeout are required")
    output = args.output.resolve()
    if args.worker:
        return worker(output, args.ranks, args.dimension, args.tests)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output must be empty to exclude stale rank receipts")
    environment = dict(os.environ, PYTHONNOUSERSITE="1", POPS_NATIVE_DIM=str(args.dimension),
                       POPS_REQUIRE_NATIVE_TESTS="1", OMP_NUM_THREADS=str(args.threads),
                       POPS_THREADS=str(args.threads), OMP_PROC_BIND="false")
    identity_runner = Path(__file__).with_name("run_installed_checks.py")
    paths = {Path(__file__).resolve(), identity_runner,
             *(ROOT / test.split("::", 1)[0] for test in args.tests)}
    manifest = {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}
    (output / "test-sources.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def authenticate(phase: str) -> int:
        with (output / (phase + ".log")).open("w") as log:
            return subprocess.run([sys.executable, str(identity_runner), "--identity-only",
                                   "--output", str(output / phase)], cwd=ROOT, env=environment,
                                  stdout=log, stderr=subprocess.STDOUT).returncode

    before_code = authenticate("before")
    launcher = Path(sys.prefix) / "bin/mpiexec"
    if not launcher.is_file():
        parser.error(f"MPI launcher absent from active environment: {launcher}")
    command = [str(launcher), "-n", str(args.ranks), sys.executable, str(Path(__file__).resolve()),
               "--worker", "--output", str(output), "--ranks", str(args.ranks),
               "--dimension", str(args.dimension)]
    for test in args.tests:
        command.extend(("--test", test))
    started = time.monotonic()
    code, timed_out = None, False
    if before_code == 0:
        with (output / "launcher.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
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
    elapsed = time.monotonic() - started
    after_code = authenticate("after")
    before = json.loads((output / "before/identity.json").read_text()) if before_code == 0 else {}
    after = json.loads((output / "after/identity.json").read_text()) if after_code == 0 else {}
    unchanged = manifest == {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}
    same_installation = bool(before) and all(before.get(key) == after.get(key) for key in
                                             ("native_sha256", "source_files_sha256"))
    rank_results = []
    for rank in range(args.ranks):
        xml = output / f"rank{rank}.xml"
        if not xml.is_file():
            rank_results.append({"rank": rank, "status": "missing"})
            continue
        suites = ET.parse(xml).getroot()
        counts = {key: sum(int(suite.get(key, 0)) for suite in suites.iter("testsuite"))
                  for key in ("tests", "failures", "errors", "skipped")}
        nodes = [(node.get("classname"), node.get("name")) for node in suites.iter("testcase")]
        rank_results.append({"rank": rank, "counts": counts, "nodes": nodes,
                             "xml_sha256": digest(xml),
                             "log_sha256": digest(output / f"rank{rank}.log")})
    rank_parity = all(row.get("nodes") == rank_results[0].get("nodes") for row in rank_results)
    clean = all(row.get("counts", {}).get("tests", 0) > 0 and
                all(row["counts"][key] == 0 for key in ("failures", "errors", "skipped"))
                for row in rank_results)
    passed = (code == 0 and before_code == after_code == 0 and unchanged
              and same_installation and rank_parity and clean)
    receipt = {"schema_version": 2, "status": "passed" if passed else "failed",
               "pytest_base_temps": [f"rank{rank}-tmp" for rank in range(args.ranks)],
               "command": command, "returncode": code, "timeout": timed_out,
               "seconds": elapsed, "ranks": args.ranks, "threads": args.threads,
               "dimension": args.dimension,
               "authentication_before": before_code, "authentication_after": after_code,
               "same_installation": same_installation, "test_sources_unchanged": unchanged,
               "rank_test_parity": rank_parity, "rank_results": rank_results,
               "native_sha256": before.get("native_sha256")}
    (output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items()
                      if key not in ("command", "rank_results")}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
