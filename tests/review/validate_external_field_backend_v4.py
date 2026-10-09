"""Bounded source/Host validation with existing read-only dependencies; no GPU qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-command", required=True, type=Path)
    parser.add_argument("--syntax-command", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)  # receipts are never overwritten
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["OMP_PROC_BIND"] = "false"
    inherited = json.loads(args.compile_command.read_text())
    source_suffix = "tests/cpp/unit/runtime/test_external_field_backend_preparation.cpp"
    syntax_suffix = "tests/cpp/unit/runtime/test_prepared_field_solver_nd.cpp"

    def adapt(command: list[str], source: str) -> list[str]:
        adapted = []
        for arg in command:
            if arg.startswith("-I") and arg.endswith("/include") and "/miniforge" not in arg and "/googletest" not in arg:
                arg = "-I" + str(root / "include")
            elif arg.endswith(source):
                arg = str(root / source)
            adapted.append(arg)
        return adapted

    compile_command = adapt(inherited, source_suffix)
    compile_command[compile_command.index("-o") + 1] = str(out / "host-tests")
    syntax_command = adapt(json.loads(args.syntax_command.read_text()), syntax_suffix)
    probe_files = (
        "include/pops/runtime/system/prepared_field_solver_component.hpp",
        "python/pops/fields/providers.py", "python/pops/interfaces.py",
        source_suffix, syntax_suffix,
        "tests/review/test_sol61_external_field_backend_preparation.py",
        "tests/python/unit/fields/test_external_field_solver_provider.py",
    )
    report = {
        "schema_version": 1, "qualification": "source-and-host-only",
        "gpu_compiled": False, "gpu_executed": False, "mpi_executed": False,
        "installed_provider_executed": False,
        "source_hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in probe_files},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_dirty": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
        "borrowed_dependency_command": str(args.compile_command.resolve()),
        "checks": {},
    }

    def run(name: str, command: list[str]) -> bool:
        (out / (name + "-command.json")).write_text(json.dumps(command, indent=2) + "\n")
        with (out / (name + ".log")).open("w") as log:
            result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        report["checks"][name] = result.returncode
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, result.returncode, flush=True)
        return result.returncode == 0

    if not run("compile", compile_command):
        return 1
    if not run("host", [str(out / "host-tests"), "--gtest_output=xml:" + str(out / "host.xml")]):
        return 1
    if not run("syntax", syntax_command):
        return 1
    pytest_code = (
        "import sys,pytest;sys.path.insert(0," + repr(str(root / "python")) + ");"
        "raise SystemExit(pytest.main(" + repr([
            "-q", "tests/review/test_sol61_external_field_backend_preparation.py",
            "tests/python/unit/fields/test_external_field_solver_provider.py",
            "tests/python/architecture/test_prepared_field_provider_core.py",
            "--junitxml=" + str(out / "python.xml"),
        ]) + "))"
    )
    if not run("python", [str(args.python), "-c", pytest_code]):
        return 1
    report["xml_counts"] = {}
    for name in ("host", "python"):
        xml = ET.parse(out / (name + ".xml")).getroot()
        suites = [xml] if xml.tag == "testsuite" else list(xml.findall("testsuite"))
        report["xml_counts"][name] = {
            key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["xml_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
