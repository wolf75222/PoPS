#!/usr/bin/env python3
"""Validate and run the deterministic M2 temporal-execution conformance matrix."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests/gates/m2_temporal_execution.toml"
TEST_MANIFEST = ROOT / "tests/test_manifest.toml"
EXPECTED_ISSUES = tuple("ADC-%d" % number for number in range(661, 667))
EXPECTED_REQUIREMENTS = {
    "phase_pipeline", "program_graph", "schedules", "residual_operator",
    "solve_outcome", "step_transaction", "restart",
}
EXPECTED_DEFERRED = {"ADC-648", "ADC-667"}
ALLOWED_PYTEST_TARGETS = {
    "architecture", "pipeline", "program_graph", "residual", "schedule",
    "solve", "transaction",
}


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
        if name.endswith((".skip", ".skipif", ".xfail")) or name in {
                "skip", "skipif", "xfail"}:
            markers.append(name)
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and _dotted_name(child.func) in {
                "pytest.skip", "pytest.xfail"}:
            markers.append(_dotted_name(child.func))
    return markers


def _ctest_suites() -> dict[str, dict]:
    data = tomllib.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    return {str(row["name"]): row for row in data.get("cpp", {}).get("suite", ())}


def validate_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[dict, list[str]]:
    """Return the parsed manifest and all deterministic source-only errors."""
    errors: list[str] = []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return {}, ["cannot read M2 gate manifest %s: %s" % (path, exc)]

    if data.get("schema_version") != 1:
        errors.append("schema_version must be exactly 1")
    if data.get("gate") != "m2-temporal-execution":
        errors.append("gate must be exactly 'm2-temporal-execution'")
    if set(data) != {"schema_version", "gate", "issues", "deferred", "check"}:
        errors.append("manifest fields must be schema_version/gate/issues/deferred/check")
    if data.get("issues") != list(EXPECTED_ISSUES):
        errors.append("issues must list ADC-661..ADC-666 exactly once in ascending order")

    deferred = data.get("deferred")
    deferred_ids = []
    if not isinstance(deferred, list):
        errors.append("manifest must contain [[deferred]] rows")
        deferred = []
    for index, row in enumerate(deferred, 1):
        where = "deferred[%d]" % index
        if set(row) != {"issue", "requirement", "reason", "close_condition"}:
            errors.append("%s has unknown or missing fields: %s" % (where, sorted(row)))
            continue
        deferred_ids.append(row["issue"])
        for field in ("requirement", "reason", "close_condition"):
            if not isinstance(row[field], str) or not row[field].strip():
                errors.append("%s.%s must be a non-empty string" % (where, field))
    if set(deferred_ids) != EXPECTED_DEFERRED or len(deferred_ids) != len(EXPECTED_DEFERRED):
        errors.append("deferred issues must be exactly ADC-648 and ADC-667")

    checks = data.get("check")
    if not isinstance(checks, list) or not checks:
        errors.append("manifest must contain [[check]] rows")
        checks = []
    identities = Counter()
    issue_coverage: dict[str, set[str]] = defaultdict(set)
    requirement_coverage: dict[str, set[str]] = defaultdict(set)
    cpp_suites = _ctest_suites()
    for index, row in enumerate(checks, 1):
        where = "check[%d]" % index
        base = {"issue", "requirement", "polarity", "kind", "target"}
        expected = base | ({"nodeid"} if row.get("kind") == "pytest" else {"test_regex"})
        if set(row) != expected:
            errors.append("%s has unknown or missing fields: %s" % (where, sorted(row)))
            continue
        issue = row.get("issue")
        requirement = row.get("requirement")
        polarity = row.get("polarity")
        kind = row.get("kind")
        target = row.get("target")
        if issue not in EXPECTED_ISSUES:
            errors.append("%s has unknown or deferred issue %r" % (where, issue))
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
            if not isinstance(nodeid, str) or nodeid.count("::") != 1:
                errors.append("%s must contain one exact file::test nodeid" % where)
                continue
            relative, function_name = nodeid.split("::")
            test_path = ROOT / relative
            if not test_path.is_file():
                errors.append("%s references missing test file %s" % (where, relative))
                continue
            tree = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
            functions = {
                node.name: node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            function = functions.get(function_name)
            if function is None:
                errors.append("%s references missing test function %s" % (where, nodeid))
                continue
            markers = _skip_or_xfail_markers(function)
            module_nodes = [
                node for node in tree.body
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ]
            markers.extend(_skip_or_xfail_markers(ast.Module(body=module_nodes, type_ignores=[])))
            if markers:
                errors.append("%s is not mandatory; found %s" % (nodeid, sorted(set(markers))))
        elif kind == "ctest":
            selector = row.get("test_regex")
            if not isinstance(selector, str) or not selector:
                errors.append("%s CTest row requires a non-empty test_regex" % where)
            if target not in cpp_suites:
                errors.append("%s references unknown CTest target %r" % (where, target))
                continue
            for relative in cpp_suites[target].get("sources", ()):
                source = ROOT / relative
                if not source.is_file():
                    errors.append("%s target %r has missing source %s" % (where, target, relative))
                elif "GTEST_SKIP" in source.read_text(encoding="utf-8") \
                        or "DISABLED_" in source.read_text(encoding="utf-8"):
                    errors.append("%s target %r contains a skip marker" % (where, target))
        else:
            errors.append("%s kind must be pytest or ctest" % where)

    duplicates = sorted(identity for identity, count in identities.items() if count > 1)
    if duplicates:
        errors.append("duplicate executable checks: %s" % duplicates)
    for issue in EXPECTED_ISSUES:
        missing = {"positive", "refusal"} - issue_coverage[issue]
        if missing:
            errors.append("%s lacks %s coverage" % (issue, "/".join(sorted(missing))))
    for requirement in sorted(EXPECTED_REQUIREMENTS):
        missing = {"positive", "refusal"} - requirement_coverage[requirement]
        if missing:
            errors.append("%s lacks %s coverage" % (requirement, "/".join(sorted(missing))))
    return data, errors


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def _run_ctest(build_dir: Path, target: str, selector: str) -> None:
    listed = subprocess.run(
        ["ctest", "--test-dir", str(build_dir), "-N", "-R", selector],
        cwd=ROOT, check=True, text=True, capture_output=True)
    if "Total Tests: 0" in listed.stdout or "Test #" not in listed.stdout:
        raise RuntimeError("M2 CTest target %r (%s) is not built in %s"
                           % (target, selector, build_dir))
    _run(["ctest", "--test-dir", str(build_dir), "--output-on-failure", "-R", selector])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args(argv)

    data, errors = validate_manifest(args.manifest)
    if errors:
        print("M2 gate manifest is incomplete or invalid:", file=sys.stderr)
        for error in errors:
            print(" -", error, file=sys.stderr)
        return 2
    checks = data["check"]
    print("M2 gate source matrix: OK (%d executable, %d explicitly deferred)"
          % (len(checks), len(data["deferred"])))
    if args.check_only:
        return 0

    nodeids = [row["nodeid"] for row in checks if row["kind"] == "pytest"]
    for chunk in _chunks(nodeids, 24):
        _run([sys.executable, "-m", "pytest", "-q", *chunk])
    if not args.python_only:
        for row in sorted(
                (row for row in checks if row["kind"] == "ctest"),
                key=lambda value: (value["target"], value["test_regex"])):
            _run_ctest(args.build_dir, row["target"], row["test_regex"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
