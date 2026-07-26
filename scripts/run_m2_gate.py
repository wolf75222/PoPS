#!/usr/bin/env python3
"""Validate and run the deterministic M2 temporal-execution conformance matrix."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from collections.abc import Iterable
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests/gates/m2_temporal_execution.toml"
TEST_MANIFEST = ROOT / "tests/test_manifest.toml"
EXPECTED_SUPPORT_ISSUES = ("ADC-648",) + tuple("ADC-%d" % number for number in range(661, 668))
EXPECTED_ISSUES = EXPECTED_SUPPORT_ISSUES + ("ADC-668", "ADC-700")
EXPECTED_SUPPORT_REQUIREMENTS = {
    "amr_step_transaction",
    "dae_residual",
    "hierarchy_solve_ordering",
    "normalized_program_execution",
    "phase_pipeline",
    "program_graph",
    "schedules",
    "residual_operator",
    "solve_outcome",
    "step_transaction",
    "restart",
    "temporal_restart",
}
EXPECTED_DEFERRED = {
    "normalized_program_execution": ("ADC-668",),
    "native_solve_outcome_fault_matrix": ("ADC-665",),
    "atomic_rejection_side_effects": ("ADC-666",),
    "strict_temporal_continuation": ("ADC-667",),
    "native_multiblock_implicit_phase": ("ADC-665", "ADC-666"),
    "refined_hierarchy_native_ordering": ("ADC-648",),
    "legacy_temporal_route_retirement": ("ADC-700",),
}
ALLOWED_PYTEST_TARGETS = {
    "architecture",
    "example",
    "hierarchy",
    "pipeline",
    "program_graph",
    "residual",
    "schedule",
    "solve",
    "transaction",
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


def _gtest_identities(source_text: str) -> set[str]:
    return {
        "%s.%s" % match.groups()
        for match in re.finditer(
            r"\b(?:TEST|TEST_F|TEST_P)\s*\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*\)",
            source_text,
        )
    }


def validate_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[dict, list[str]]:
    """Return the parsed manifest and all deterministic source-only errors."""
    errors: list[str] = []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return {}, ["cannot read M2 gate manifest %s: %s" % (path, exc)]

    if data.get("schema_version") != 2:
        errors.append("schema_version must be exactly 2")
    if data.get("gate") != "m2-temporal-execution":
        errors.append("gate must be exactly 'm2-temporal-execution'")
    if set(data) != {"schema_version", "gate", "issues", "deferred", "check"}:
        errors.append("manifest fields must be schema_version/gate/issues/deferred/check")
    if data.get("issues") != list(EXPECTED_ISSUES):
        errors.append("issues must list ADC-648, ADC-661..ADC-668 and ADC-700 exactly once")

    deferred = data.get("deferred")
    if not isinstance(deferred, list):
        errors.append("manifest must contain [[deferred]] rows")
        deferred = []
    deferred_names = Counter()
    for index, row in enumerate(deferred, 1):
        where = "deferred[%d]" % index
        if not isinstance(row, dict) or set(row) != {"requirement", "blocked_by", "missing"}:
            fields = sorted(row) if isinstance(row, dict) else type(row).__name__
            errors.append("%s has unknown or missing fields: %s" % (where, fields))
            continue
        requirement = row.get("requirement")
        deferred_names[str(requirement)] += 1
        expected_blockers = EXPECTED_DEFERRED.get(str(requirement))
        if expected_blockers is None:
            errors.append("%s has unknown requirement %r" % (where, requirement))
        blocked_by = row.get("blocked_by")
        if not isinstance(blocked_by, list) or not all(
            isinstance(issue, str) for issue in blocked_by
        ):
            errors.append("%s blocked_by must be a list of issue identifiers" % where)
        elif expected_blockers is not None and tuple(blocked_by) != expected_blockers:
            errors.append("%s blockers must be exactly %s" % (requirement, list(expected_blockers)))
        unknown_blockers = (
            sorted(set(blocked_by) - set(data.get("issues", ())))
            if isinstance(blocked_by, list)
            else []
        )
        if unknown_blockers:
            errors.append(
                "%s blockers are absent from manifest issues: %s" % (requirement, unknown_blockers)
            )
        missing = row.get("missing")
        if not isinstance(missing, str) or not missing.strip():
            errors.append("%s missing must explain the absent executable proof" % where)
    duplicate_deferred = sorted(
        requirement for requirement, count in deferred_names.items() if count > 1
    )
    if duplicate_deferred:
        errors.append("duplicate deferred requirements: %s" % duplicate_deferred)
    absent_deferred = sorted(set(EXPECTED_DEFERRED) - set(deferred_names))
    if absent_deferred:
        errors.append("missing deferred requirements: %s" % absent_deferred)

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
        if issue not in EXPECTED_SUPPORT_ISSUES:
            errors.append("%s has unknown or deferred issue %r" % (where, issue))
        if requirement not in EXPECTED_SUPPORT_REQUIREMENTS:
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
                node.name: node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            function = functions.get(function_name)
            if function is None:
                errors.append("%s references missing test function %s" % (where, nodeid))
                continue
            markers = _skip_or_xfail_markers(function)
            module_nodes = [
                node
                for node in tree.body
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
            gtest_identities = set()
            for relative in cpp_suites[target].get("sources", ()):
                source = ROOT / relative
                if not source.is_file():
                    errors.append("%s target %r has missing source %s" % (where, target, relative))
                    continue
                source_text = source.read_text(encoding="utf-8")
                gtest_identities.update(_gtest_identities(source_text))
                if "GTEST_SKIP" in source_text or "DISABLED_" in source_text:
                    errors.append("%s target %r contains a skip marker" % (where, target))
            selector_match = (
                re.fullmatch(r"\^([A-Za-z_]\w*)\\\.([A-Za-z_]\w*)\$", selector)
                if isinstance(selector, str)
                else None
            )
            if selector_match is None:
                errors.append("%s CTest selector must be one exact ^Suite\\.Test$ regex" % where)
            elif ".".join(selector_match.groups()) not in gtest_identities:
                errors.append("%s references missing GoogleTest %s" % (where, selector))
        else:
            errors.append("%s kind must be pytest or ctest" % where)

    duplicates = sorted(identity for identity, count in identities.items() if count > 1)
    if duplicates:
        errors.append("duplicate executable checks: %s" % duplicates)
    for issue in EXPECTED_SUPPORT_ISSUES:
        missing = {"positive", "refusal"} - issue_coverage[issue]
        if missing:
            errors.append("%s lacks %s coverage" % (issue, "/".join(sorted(missing))))
    for requirement in sorted(EXPECTED_SUPPORT_REQUIREMENTS):
        missing = {"positive", "refusal"} - requirement_coverage[requirement]
        if missing:
            errors.append("%s lacks %s coverage" % (requirement, "/".join(sorted(missing))))
    return data, errors


def is_complete(data: dict) -> bool:
    """Whether the executable M2 acceptance matrix has no acknowledged proof gap."""
    return data.get("deferred") == []


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _pytest_command(nodeids: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "scripts.m2_mandatory_pytest_plugin",
        "-q",
        *nodeids,
    ]


def _run_mandatory_pytest(nodeids: list[str]) -> None:
    _run(_pytest_command(nodeids))


def _run_isolated_pytest(nodeids: list[str], *, timeout_seconds: int) -> None:
    """Run native examples in independent process groups and report every failure."""
    failures = []
    for nodeid in nodeids:
        command = _pytest_command([nodeid])
        print("+", " ".join(command), flush=True)
        process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            failures.append("%s (timeout after %ds)" % (nodeid, timeout_seconds))
            continue
        if returncode != 0:
            failures.append("%s (exit %d)" % (nodeid, returncode))
    if failures:
        raise RuntimeError("isolated M2 example proof failures: " + ", ".join(failures))


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
            "M2 CTest target %r (%s) is not built in %s" % (target, selector, build_dir)
        )
    _run(["ctest", "--test-dir", str(build_dir), "--output-on-failure", "-R", selector])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="run landed supporting proofs even while the final M2 gate is incomplete",
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="skip CTest proofs; pytest proofs may still compile or execute native code",
    )
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument(
        "--example-timeout",
        type=int,
        default=1800,
        help="maximum wall time in seconds for each isolated native example",
    )
    args = parser.parse_args(argv)
    if args.example_timeout <= 0:
        parser.error("--example-timeout must be a positive integer")

    data, errors = validate_manifest(args.manifest)
    if errors:
        print("M2 gate manifest is incomplete or invalid:", file=sys.stderr)
        for error in errors:
            print(" -", error, file=sys.stderr)
        return 2
    checks = data["check"]
    complete = is_complete(data)
    state = "COMPLETE" if complete else "INCOMPLETE"
    print(
        "M2 gate source matrix: %s (%d executable supporting proofs, %d deferred)"
        % (state, len(checks), len(data["deferred"]))
    )
    if args.check_only:
        return 0
    if not complete and not args.available_only:
        print(
            "M2 gate cannot pass: executable acceptance proofs are still deferred.",
            file=sys.stderr,
        )
        for row in data["deferred"]:
            print(
                " - %s [%s]: %s"
                % (row["requirement"], ", ".join(row["blocked_by"]), row["missing"]),
                file=sys.stderr,
            )
        print(
            "Use --available-only to run the landed supporting battery without "
            "claiming M2 completion.",
            file=sys.stderr,
        )
        return 3

    nodeids = [
        row["nodeid"] for row in checks if row["kind"] == "pytest" and row["target"] != "example"
    ]
    for chunk in _chunks(nodeids, 24):
        _run_mandatory_pytest(chunk)
    _run_isolated_pytest(
        [
            row["nodeid"]
            for row in checks
            if row["kind"] == "pytest" and row["target"] == "example"
        ],
        timeout_seconds=args.example_timeout,
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
