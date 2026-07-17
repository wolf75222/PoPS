#!/usr/bin/env python3
"""Run the deterministic M1 semantic-core conformance matrix.

The manifest is validated before any executable is started.  Missing nodeids,
unknown CTest targets, duplicate rows and skip/xfail markers are hard errors.
"""
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
DEFAULT_MANIFEST = ROOT / "tests/gates/m1_semantic_core.toml"
TEST_MANIFEST = ROOT / "tests/test_manifest.toml"
EXPECTED_ISSUES = {"ADC-%d" % number for number in range(652, 661)}
ALLOWED_PYTEST_TARGETS = {
    "architecture", "cross_language", "manifest", "pipeline", "python", "snapshot",
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
    decorators = getattr(node, "decorator_list", ())
    for decorator in decorators:
        name = _dotted_name(decorator)
        if name.endswith((".skip", ".skipif", ".xfail")) or name in {"skip", "skipif", "xfail"}:
            markers.append(name)
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = _dotted_name(child.func)
        if name in {"pytest.skip", "pytest.xfail"}:
            markers.append(name)
    return markers


def _ctest_suites() -> dict[str, dict]:
    data = tomllib.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    return {str(row["name"]): row for row in data.get("cpp", {}).get("suite", ())}


def validate_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[dict, list[str]]:
    """Return parsed data and every deterministic source-only validation error."""
    errors: list[str] = []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return {}, ["cannot read M1 gate manifest %s: %s" % (path, exc)]

    if data.get("schema_version") != 1:
        errors.append("schema_version must be exactly 1")
    if data.get("gate") != "m1-semantic-core":
        errors.append("gate must be exactly 'm1-semantic-core'")
    if set(data) != {"schema_version", "gate", "issues", "check", "scan"}:
        errors.append("manifest top-level fields must be schema_version/gate/issues/check/scan")
    issues = data.get("issues")
    expected_issue_order = ["ADC-%d" % number for number in range(652, 661)]
    if issues != expected_issue_order:
        errors.append("issues must list ADC-652..ADC-660 exactly once in ascending order")

    checks = data.get("check")
    if not isinstance(checks, list) or not checks:
        errors.append("manifest must contain [[check]] rows")
        checks = []
    identities = Counter()
    coverage: dict[str, set[str]] = defaultdict(set)
    cpp_suites = _ctest_suites()
    for index, row in enumerate(checks, 1):
        where = "check[%d]" % index
        if set(row) not in ({"issue", "polarity", "kind", "target", "nodeid"},
                            {"issue", "polarity", "kind", "target", "test_regex"}):
            errors.append("%s has unknown or missing fields: %s" % (where, sorted(row)))
            continue
        issue = row.get("issue")
        polarity = row.get("polarity")
        kind = row.get("kind")
        target = row.get("target")
        if issue not in EXPECTED_ISSUES:
            errors.append("%s has unknown issue %r" % (where, issue))
        if polarity not in {"positive", "negative"}:
            errors.append("%s polarity must be positive or negative" % where)
        else:
            coverage[str(issue)].add(polarity)
        identity = (kind, row.get("nodeid", target))
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
            module_statements = [
                node for node in tree.body
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ]
            markers.extend(_skip_or_xfail_markers(ast.Module(body=module_statements,
                                                              type_ignores=[])))
            if markers:
                errors.append("%s is not mandatory; found %s" % (nodeid, sorted(set(markers))))
        elif kind == "ctest":
            if "nodeid" in row:
                errors.append("%s CTest row must not contain nodeid" % where)
            if not isinstance(row.get("test_regex"), str) or not row["test_regex"]:
                errors.append("%s CTest row requires a non-empty test_regex" % where)
            if target not in cpp_suites:
                errors.append("%s references unknown tests/test_manifest.toml CTest target %r"
                              % (where, target))
                continue
            for relative in cpp_suites[target].get("sources", ()):
                source = ROOT / relative
                if not source.is_file():
                    errors.append("%s CTest target %r has missing source %s"
                                  % (where, target, relative))
                    continue
                text = source.read_text(encoding="utf-8")
                if "GTEST_SKIP" in text or "DISABLED_" in text:
                    errors.append("%s CTest target %r is not mandatory; %s contains a skip marker"
                                  % (where, target, relative))
        else:
            errors.append("%s kind must be pytest or ctest" % where)

    duplicates = sorted(identity for identity, count in identities.items() if count > 1)
    if duplicates:
        errors.append("duplicate executable checks: %s" % duplicates)
    for issue in sorted(EXPECTED_ISSUES):
        missing = {"positive", "negative"} - coverage[issue]
        if missing:
            errors.append("%s lacks %s coverage" % (issue, "/".join(sorted(missing))))

    scans = data.get("scan")
    expected_scans = {
        "compiled_live_authoring_refs", "ownerless_handles", "permissive_phase_probing",
        "phase_strict_false", "runtime_legacy_loaders",
    }
    scan_names = [row.get("name") for row in scans] if isinstance(scans, list) else []
    if set(scan_names) != expected_scans or len(scan_names) != len(expected_scans):
        errors.append("[[scan]] must list the five reviewed M1 AST scans exactly once")
    return data, errors


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _run_ctest(build_dir: Path, target: str, selector: str) -> None:
    listed = subprocess.run(
        ["ctest", "--test-dir", str(build_dir), "-N", "-R", selector],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    if "Total Tests: 0" in listed.stdout or ("Test #" not in listed.stdout):
        raise RuntimeError(
            "M1 gate CTest target %r (%s) is not built in %s"
            % (target, selector, build_dir)
        )
    _run(["ctest", "--test-dir", str(build_dir), "--output-on-failure",
          "-R", selector])


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check-only", action="store_true",
                        help="validate source coverage without running tests")
    parser.add_argument("--python-only", action="store_true", help="do not run CTest targets")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args(argv)

    data, errors = validate_manifest(args.manifest)
    if errors:
        print("M1 gate manifest is incomplete or invalid:", file=sys.stderr)
        for error in errors:
            print(" -", error, file=sys.stderr)
        return 2
    print("M1 gate source matrix: OK (%d checks)" % len(data["check"]))
    if args.check_only:
        return 0

    architecture_nodeid = (
        "tests/python/architecture/test_m1_semantic_core_gate.py")
    _run([sys.executable, "-m", "pytest", "-q", architecture_nodeid])
    nodeids = [row["nodeid"] for row in data["check"] if row["kind"] == "pytest"]
    for chunk in _chunks(nodeids, 24):
        _run([sys.executable, "-m", "pytest", "-q", *chunk])
    if not args.python_only:
        for row in sorted(
            (row for row in data["check"] if row["kind"] == "ctest"),
            key=lambda value: value["target"],
        ):
            _run_ctest(args.build_dir, row["target"], row["test_regex"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
