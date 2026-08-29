#!/usr/bin/env python3
"""Validate and run the deterministic M3 AMR/multi-layout conformance matrix."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from collections.abc import Iterable
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests/gates/m3_amr_multilayout.toml"
TEST_MANIFEST = ROOT / "tests/test_manifest.toml"
EXPECTED_ISSUES = tuple("ADC-%d" % number for number in range(672, 679))
EXPECTED_REQUIREMENTS = {
    "ghost_plan",
    "layout_plan",
    "tagging_graph",
    "hierarchy_regrid",
    "transfer_bootstrap",
    "clocks_reflux",
    "accepted_state",
    "restart_hierarchy_policy",
    "lowering_coverage",
}
ALLOWED_PYTEST_TARGETS = {
    "accepted_state",
    "ghost_plan",
    "layout_plan",
    "lowering_coverage",
    "tagging_graph",
    "hierarchy_regrid",
    "transfer_bootstrap",
    "restart_hierarchy_policy",
}
NATIVE_PYTEST_FILES = {
    "tests/python/integration/amr/test_amr_magnitude_tagging_native.py",
}
_GTEST_DECLARATION = re.compile(
    r"\bTEST(?:_F)?\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\)"
)
_CPP_RAW_STRING_START = re.compile(r'(?:u8|u|U|L)?R"([^\s()\\]{0,16})\(')


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return "%s.%s" % (prefix, node.attr) if prefix else node.attr
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    return ""


def _skip_or_xfail_markers(node: ast.AST) -> list[str]:
    markers = []
    for decorator in getattr(node, "decorator_list", ()):
        name = _dotted_name(decorator)
        if name.endswith((".skip", ".skipif", ".xfail")) or name in {"skip", "skipif", "xfail"}:
            markers.append(name)
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and _dotted_name(child.func) in {
            "pytest.skip",
            "pytest.xfail",
        }:
            markers.append(_dotted_name(child.func))
    return markers


def _ctest_suites() -> dict[str, dict]:
    data = tomllib.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    return {str(row["name"]): row for row in data.get("cpp", {}).get("suite", ())}


def _cpp_code_only(source: str) -> str:
    """Mask comments and literals while preserving source positions and newlines."""
    code = list(source)
    size = len(source)

    def mask(begin: int, end: int) -> None:
        for offset in range(begin, end):
            if code[offset] != "\n":
                code[offset] = " "

    index = 0
    while index < size:
        if source.startswith("//", index):
            end = source.find("\n", index + 2)
            end = size if end < 0 else end
            mask(index, end)
            index = end
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            end = size if end < 0 else end + 2
            mask(index, end)
            index = end
            continue
        raw = _CPP_RAW_STRING_START.match(source, index)
        if raw is not None:
            terminator = ")" + raw.group(1) + '"'
            end = source.find(terminator, raw.end())
            end = size if end < 0 else end + len(terminator)
            mask(index, end)
            index = end
            continue
        if source[index] in {'"', "'"}:
            quote = source[index]
            end = index + 1
            while end < size:
                if source[end] == "\\":
                    end = min(size, end + 2)
                    continue
                end += 1
                if source[end - 1] == quote:
                    break
            mask(index, end)
            index = end
            continue
        index += 1
    return "".join(code)


def _registered_gtest_cases(source: str) -> set[str]:
    code = _cpp_code_only(source)
    return {"%s.%s" % declaration for declaration in _GTEST_DECLARATION.findall(code)}


def _registered_ctest_cases(target: str, suite: dict) -> set[str]:
    """Recover the exact configure-time CTest names owned by one manifest suite."""
    mpi_counts = tuple(suite.get("mpi_nproc", ())) + tuple(suite.get("mpi_variants", ()))
    is_no_discover_mpi = bool(suite.get("mpi_nproc")) or bool(suite.get("mpi_rank_parity"))
    cases: set[str] = set()
    if not is_no_discover_mpi:
        for relative in suite.get("sources", ()):
            source = ROOT / relative
            if not source.is_file():
                continue
            cases.update(_registered_gtest_cases(source.read_text(encoding="utf-8")))
    cases.update(
        "%s_np%d" % (target, nproc)
        for nproc in mpi_counts
        if not isinstance(nproc, bool) and isinstance(nproc, int) and nproc > 0
    )
    if suite.get("mpi_rank_parity"):
        cases.add("%s_rank_parity" % target)
    return cases


def _validate_exact_ctest_selector(
    selector: object,
    target: str,
    suite: dict,
    where: str,
    errors: list[str],
) -> None:
    if not isinstance(selector, str) or not selector:
        errors.append("%s CTest row requires a non-empty test_regex" % where)
        return
    cases = _registered_ctest_cases(target, suite)
    exact = {"^%s$" % re.escape(case) for case in cases}
    if selector not in exact:
        errors.append(
            "%s CTest selector %r is not one exact source-registered case for target %r"
            % (where, selector, target)
        )


def _python_mpi_entrypoints() -> dict[str, int]:
    data = tomllib.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    entries: dict[str, int] = {}
    for suite in data.get("python", {}).get("suite", ()):
        for row in suite.get("mpi_entrypoints", ()):
            path = str(row.get("path", ""))
            nproc = row.get("nproc")
            if not path or isinstance(nproc, bool) or not isinstance(nproc, int) or nproc < 1:
                raise ValueError("invalid Python MPI entrypoint %r" % row)
            if path in entries:
                raise ValueError("duplicate Python MPI entrypoint %s" % path)
            entries[path] = nproc
    return entries


def _python_mpi_orchestrators() -> set[str]:
    data = tomllib.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    orchestrators: set[str] = set()
    for suite in data.get("python", {}).get("suite", ()):
        for row in suite.get("mpi_orchestrators", ()):
            if not isinstance(row, dict) or set(row) != {"path"}:
                raise ValueError(
                    "invalid Python MPI orchestrator %r; expected exactly one path field" % row
                )
            path = row["path"]
            if not isinstance(path, str) or not path:
                raise ValueError("invalid Python MPI orchestrator path %r" % path)
            if path in orchestrators:
                raise ValueError("duplicate Python MPI orchestrator %s" % path)
            orchestrators.add(path)
    return orchestrators


def _validate_python_nodeid(nodeid: object, where: str, errors: list[str]) -> str | None:
    if not isinstance(nodeid, str) or nodeid.count("::") != 1:
        errors.append("%s must contain one exact file::test nodeid" % where)
        return None
    relative, function_name = nodeid.split("::")
    test_path = ROOT / relative
    if not test_path.is_file():
        errors.append("%s references missing test file %s" % (where, relative))
        return None
    tree = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    function = functions.get(function_name)
    if function is None:
        errors.append("%s references missing test function %s" % (where, nodeid))
        return None
    markers = _skip_or_xfail_markers(function)
    module_nodes = [
        node
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    markers.extend(_skip_or_xfail_markers(ast.Module(body=module_nodes, type_ignores=[])))
    if markers:
        errors.append("%s is not mandatory; found %s" % (nodeid, sorted(set(markers))))
    return relative


def validate_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[dict, list[str]]:
    """Return the parsed manifest and every deterministic source-only error."""
    errors: list[str] = []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return {}, ["cannot read M3 gate manifest %s: %s" % (path, exc)]

    if data.get("schema_version") != 1:
        errors.append("schema_version must be exactly 1")
    if data.get("gate") != "m3-amr-multilayout":
        errors.append("gate must be exactly 'm3-amr-multilayout'")
    if set(data) != {"schema_version", "gate", "issues", "deferred", "check"}:
        errors.append("manifest fields must be schema_version/gate/issues/deferred/check")
    if data.get("issues") != list(EXPECTED_ISSUES):
        errors.append("issues must list ADC-672..ADC-678 exactly once")
    if data.get("deferred") != []:
        errors.append("final M3 gate requires deferred = []")

    checks = data.get("check")
    if not isinstance(checks, list) or not checks:
        errors.append("manifest must contain [[check]] rows")
        checks = []

    identities = Counter()
    issue_coverage: dict[str, set[str]] = defaultdict(set)
    requirement_coverage: dict[str, set[str]] = defaultdict(set)
    native_positive_issues: set[str] = set()
    cpp_suites = _ctest_suites()
    try:
        python_mpi_entrypoints = _python_mpi_entrypoints()
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        errors.append("cannot read Python MPI entrypoints: %s" % exc)
        python_mpi_entrypoints = {}
    try:
        python_mpi_orchestrators = _python_mpi_orchestrators()
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        errors.append("cannot read Python MPI orchestrators: %s" % exc)
        python_mpi_orchestrators = set()
    for index, row in enumerate(checks, 1):
        where = "check[%d]" % index
        base = {"issue", "requirement", "polarity", "kind", "target"}
        if row.get("kind") == "pytest":
            expected = base | {"nodeid"}
        elif row.get("kind") == "mpi_python":
            expected = base | {"nodeid", "nproc"}
        else:
            expected = base | {"test_regex"}
        if set(row) != expected:
            errors.append("%s has unknown or missing fields: %s" % (where, sorted(row)))
            continue

        issue = row.get("issue")
        requirement = row.get("requirement")
        polarity = row.get("polarity")
        kind = row.get("kind")
        target = row.get("target")
        if issue not in EXPECTED_ISSUES:
            errors.append("%s has unknown issue %r" % (where, issue))
        if requirement not in EXPECTED_REQUIREMENTS:
            errors.append("%s has unknown requirement %r" % (where, requirement))
        if polarity not in {"positive", "refusal"}:
            errors.append("%s polarity must be positive or refusal" % where)
        else:
            issue_coverage[str(issue)].add(polarity)
            requirement_coverage[str(requirement)].add(polarity)

        identity = (kind, row.get("nodeid", row.get("test_regex")))
        identities[identity] += 1
        if kind == "pytest":
            nodeid = row.get("nodeid")
            if target not in ALLOWED_PYTEST_TARGETS:
                errors.append("%s has unknown pytest target %r" % (where, target))
            relative = _validate_python_nodeid(nodeid, where, errors)
            if (
                relative is not None
                and relative.startswith("tests/python/integration/mpi/")
                and relative not in python_mpi_orchestrators
            ):
                errors.append("%s is not a manifest-owned serial MPI orchestrator" % relative)
            if polarity == "positive" and relative in NATIVE_PYTEST_FILES:
                native_positive_issues.add(str(issue))
        elif kind == "mpi_python":
            nodeid = row.get("nodeid")
            nproc = row.get("nproc")
            if target not in ALLOWED_PYTEST_TARGETS:
                errors.append("%s has unknown MPI Python target %r" % (where, target))
            relative = _validate_python_nodeid(nodeid, where, errors)
            if isinstance(nproc, bool) or not isinstance(nproc, int) or nproc < 1:
                errors.append("%s MPI Python row requires a positive integer nproc" % where)
            elif relative is not None:
                manifest_nproc = python_mpi_entrypoints.get(relative)
                if manifest_nproc is None:
                    errors.append("%s is not a manifest-owned MPI Python entrypoint" % relative)
                elif manifest_nproc != nproc:
                    errors.append(
                        "%s requires nproc=%d, not %d" % (relative, manifest_nproc, nproc)
                    )
            if polarity == "positive":
                native_positive_issues.add(str(issue))
        elif kind == "ctest":
            selector = row.get("test_regex")
            if target not in cpp_suites:
                errors.append("%s references unknown CTest target %r" % (where, target))
                continue
            suite = cpp_suites[target]
            _validate_exact_ctest_selector(selector, target, suite, where, errors)
            for relative in suite.get("sources", ()):
                source = ROOT / relative
                if not source.is_file():
                    errors.append("%s target %r has missing source %s" % (where, target, relative))
                else:
                    text = source.read_text(encoding="utf-8")
                    if "GTEST_SKIP" in text or "DISABLED_" in text:
                        errors.append("%s target %r contains a skip marker" % (where, target))
            if polarity == "positive":
                native_positive_issues.add(str(issue))
        else:
            errors.append("%s kind must be pytest, mpi_python, or ctest" % where)

    duplicates = sorted(identity for identity, count in identities.items() if count > 1)
    if duplicates:
        errors.append("duplicate executable checks: %s" % duplicates)
    for issue in EXPECTED_ISSUES:
        missing = {"positive", "refusal"} - issue_coverage[issue]
        if missing:
            errors.append("%s lacks %s coverage" % (issue, "/".join(sorted(missing))))
        if issue not in native_positive_issues:
            errors.append("%s lacks a mandatory native positive proof" % issue)
    for requirement in sorted(EXPECTED_REQUIREMENTS):
        missing = {"positive", "refusal"} - requirement_coverage[requirement]
        if missing:
            errors.append("%s lacks %s coverage" % (requirement, "/".join(sorted(missing))))
    return data, errors


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def _mpi_python_command(mpi_exec: str, nproc: int, relative: str) -> list[str]:
    if shutil.which(mpi_exec) is None:
        raise RuntimeError("required MPI launcher %r is unavailable" % mpi_exec)
    return [mpi_exec, "-n", str(nproc), sys.executable, str(ROOT / relative)]


def _required_mpi_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["POPS_REQUIRE_MPI_TESTS"] = "1"
    environment["POPS_REQUIRE_NATIVE_TESTS"] = "1"
    return environment


def _pytest_skip_count(report: Path) -> int:
    if not report.is_file():
        raise RuntimeError("M3 pytest did not produce its mandatory JUnit report")
    try:
        root = ET.parse(report).getroot()
    except ET.ParseError as exc:
        raise RuntimeError("M3 pytest produced an invalid JUnit report") from exc
    return len(root.findall(".//skipped"))


def _run_required_pytest(nodeids: list[str]) -> None:
    environment = _required_mpi_environment()
    with tempfile.TemporaryDirectory(prefix="pops-m3-gate-") as temporary:
        report = Path(temporary) / "pytest.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--strict-markers",
            "-o",
            "xfail_strict=true",
            "--junitxml",
            str(report),
            *nodeids,
        ]
        print(
            "+ POPS_REQUIRE_MPI_TESTS=1 POPS_REQUIRE_NATIVE_TESTS=1",
            " ".join(command),
            flush=True,
        )
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=False,
        )
        skipped = _pytest_skip_count(report)
        if skipped:
            raise RuntimeError(
                "M3 pytest reported %d skipped/xfail proof(s); every proof is mandatory" % skipped
            )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _run_ctest(build_dir: Path, target: str, selector: str) -> None:
    listed = subprocess.run(
        ["ctest", "--test-dir", str(build_dir), "-N", "-R", selector],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    if "Total Tests: 0" in listed.stdout or "Test #" not in listed.stdout:
        raise RuntimeError(
            "M3 CTest target %r (%s) is not built in %s" % (target, selector, build_dir)
        )
    _run(["ctest", "--test-dir", str(build_dir), "--output-on-failure", "-R", selector])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-mpi")
    parser.add_argument("--mpi-exec", default="mpiexec")
    args = parser.parse_args(argv)

    data, errors = validate_manifest(args.manifest)
    if errors:
        print("M3 gate manifest is incomplete or invalid:", file=sys.stderr)
        for error in errors:
            print(" -", error, file=sys.stderr)
        return 2
    checks = data["check"]
    print(
        "M3 gate source matrix: OK (%d executable, %d deferred)"
        % (len(checks), len(data["deferred"]))
    )
    if args.check_only:
        return 0

    nodeids = [row["nodeid"] for row in checks if row["kind"] == "pytest"]
    for chunk in _chunks(nodeids, 24):
        _run_required_pytest(chunk)
    mpi_entrypoints = sorted(
        {
            (row["nodeid"].split("::", 1)[0], row["nproc"])
            for row in checks
            if row["kind"] == "mpi_python"
        }
    )
    for relative, nproc in mpi_entrypoints:
        _run(
            _mpi_python_command(args.mpi_exec, nproc, relative),
            env=_required_mpi_environment(),
        )
    if not args.python_only:
        for row in sorted(
            (row for row in checks if row["kind"] == "ctest"),
            key=lambda value: (value["target"], value["test_regex"]),
        ):
            _run_ctest(args.build_dir, row["target"], row["test_regex"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
