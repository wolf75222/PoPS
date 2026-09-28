#!/usr/bin/env python3
"""Run bounded migration checks only against an authenticated installed PoPS."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
TESTS = (
    "tests/python/unit/time/test_program_expressions.py",
    "tests/python/unit/time/test_program_expressions_adversarial.py",
    "tests/python/unit/numerics/test_symbolic_path.py",
    "tests/python/unit/numerics/test_symbolic_path_adversarial.py",
    "tests/python/unit/numerics/test_symbolic_path_consistency.py",
    "tests/python/unit/problem/test_deep_freeze_storage.py",
    "tests/python/integration/runtime/test_program_expression_runtime.py",
    "tests/python/integration/runtime/test_symbolic_path_runtime.py",
    "tests/python/integration/runtime/test_symbolic_path_param_runtime.py",
    "tests/python/integration/runtime/test_local_transform_runtime.py",
    "tests/python/integration/runtime/test_native_retry_transactions.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--identity-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if "PYTHONPATH" in os.environ:
        raise RuntimeError("run with env -u PYTHONPATH; source/prototype precedence is forbidden")
    sys.path.insert(0, str(ROOT))  # repository test/script helpers, never ROOT/python
    import pops
    from pops._native_selector import select_native_dimension
    from pops.runtime.doctor import doctor
    from scripts.verify_installed_native import verify_installed_native

    package = Path(pops.__file__).resolve().parent
    if not package.is_relative_to(Path(sys.prefix).resolve()) or package.is_relative_to(ROOT):
        raise RuntimeError(f"expected an installed package under {sys.prefix}, got {package}")
    dimension = int(os.environ.get("POPS_NATIVE_DIM", "2"))
    native = select_native_dimension(dimension)
    origin = verify_installed_native(expect_dimension=dimension, expect_mpi=True,
                                     expect_parallel_hdf5=True)
    checks = doctor(verbose=False)
    if any(not passed for passed, _ in checks.values()):
        raise RuntimeError(f"native doctor failed: {checks}")
    sources = {}
    # HEAD diff includes staged new modules. Do not accept a source-only fix in
    # a native receipt merely because an earlier wheel still imports successfully.
    for name in git("diff", "HEAD", "--name-only", "--", "python/pops", "include/pops").splitlines():
        source = ROOT / name
        if not source.is_file() or source.suffix not in {".py", ".hpp", ".h", ".inc"}:
            continue
        relative = name.removeprefix("python/pops/") if name.startswith("python/") else name
        installed = package / relative
        if not installed.is_file() or digest(source) != digest(installed):
            raise RuntimeError(f"installed/source mismatch for {name}: {installed}")
        sources[name] = digest(source)
    identity = {
        "schema_version": 1, "source_commit": git("rev-parse", "HEAD"),
        "source_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary").encode()).hexdigest(),
        "python": sys.executable, "package_file": pops.__file__, "package_version": pops.__version__,
        "native_file": str(origin), "native_sha256": digest(origin), "abi_key": native.abi_key(),
        "doctor": checks, "modified_installed_files": sources,
        "scope": "installed production PoPS, local CPU Kokkos, MPI-enabled Dim=2",
        "sys_path": sys.path,
    }
    (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    if args.identity_only:
        return 0
    environment = dict(os.environ, POPS_REQUIRE_NATIVE_TESTS="1")
    command = [sys.executable, "-m", "pytest", "-q", "-o", "pythonpath=",
               f"--junitxml={output / 'pytest.xml'}", *TESTS]
    start = time.monotonic()
    with (output / "pytest.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    counts = {}
    if (output / "pytest.xml").is_file():
        suites = ET.parse(output / "pytest.xml").getroot()
        counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites.iter("testsuite"))
                  for key in ("tests", "failures", "errors", "skipped")}
    receipt = {"schema_version": 1, "command": command, "returncode": result.returncode,
               "duration_seconds": time.monotonic() - start, "counts": counts,
               "status": "passed" if result.returncode == 0 and counts.get("skipped") == 0 else "failed",
               "identity_sha256": digest(output / "identity.json"), "log_sha256": digest(output / "pytest.log")}
    (output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
