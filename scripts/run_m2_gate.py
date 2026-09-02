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
import subprocess
import sys
import tempfile
import tomllib
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests/gates/m2_temporal_execution.toml"
TEST_MANIFEST = ROOT / "tests/test_manifest.toml"
EXPECTED_ISSUES = ("ADC-648",) + tuple("ADC-%d" % number for number in range(661, 669))
EXPECTED_REQUIREMENTS = {
    "amr_step_transaction",
    "phase_pipeline", "program_graph", "schedules", "residual_operator",
    "solve_outcome", "fallible_nonlinear_evaluation", "fallible_linear_evaluation",
    "native_multiblock_implicit_phase",
    "refined_hierarchy_native_ordering",
    "normalized_program_execution",
    "program_only_temporal_routes",
    "step_transaction", "restart", "temporal_restart",
}
ALLOWED_PYTEST_TARGETS = {
    "architecture", "pipeline", "program_graph", "residual", "schedule",
    "restart", "solve", "transaction",
}
EXPECTED_BACKENDS = {"serial", "mpi"}
SUPPORTED_NATIVE_DIMENSIONS = (1, 2, 3)


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


def _registered_ctest_cases(suite: dict) -> set[str]:
    """Return the CTest names CMake registers for one declared C++ suite.

    MPI-only entries are manually registered wrapper tests. Ordinary suites with `mpi_variants`
    retain their discovered serial GoogleTest cases and add wrapper names for the MPI launches.
    """
    name = str(suite["name"])
    if "mpi_rank_parity" in suite:
        return {name + "_rank_parity"}
    if "mpi_nproc" in suite:
        return {"%s_np%d" % (name, int(rank)) for rank in suite["mpi_nproc"]}
    cases: set[str] = set()
    for relative in suite.get("sources", ()):
        source = (ROOT / relative).read_text(encoding="utf-8")
        for match in re.finditer(r"\bTEST(?:_F)?\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)", source):
            cases.add("%s.%s" % (match.group(1).strip(), match.group(2).strip()))
    cases.update("%s_np%d" % (name, int(rank)) for rank in suite.get("mpi_variants", ()))
    return cases


def _registered_mpi_ctest_cases(suite: dict) -> set[str]:
    """Return the MPI wrapper cases registered for one CTest suite."""
    name = str(suite["name"])
    if "mpi_rank_parity" in suite:
        return {name + "_rank_parity"}
    ranks = suite.get("mpi_nproc") or suite.get("mpi_variants") or ()
    return {"%s_np%d" % (name, int(rank)) for rank in ranks}


def _validate_row_dimensions(
    value: object, where: str, errors: list[str]
) -> tuple[int, ...] | None:
    """Validate one optional row dimension qualifier and return canonical values."""
    if not isinstance(value, list):
        errors.append("%s dimensions must be a list" % where)
        return None
    if not value:
        errors.append("%s dimensions must be non-empty" % where)
        return None
    if any(isinstance(dimension, bool) or not isinstance(dimension, int) for dimension in value):
        errors.append("%s dimensions must contain integers; bool is not accepted" % where)
        return None
    if any(dimension not in SUPPORTED_NATIVE_DIMENSIONS for dimension in value):
        errors.append(
            "%s dimensions must contain only supported values 1, 2, or 3" % where
        )
        return None
    if len(set(value)) != len(value):
        errors.append("%s dimensions must contain unique values" % where)
        return None
    if value != sorted(value):
        errors.append("%s dimensions must use canonical sorted order" % where)
        return None
    return tuple(value)


def _validated_backend(backend: object) -> str:
    """Return one supported M2 execution backend, rejecting programmatic bypasses of argparse."""
    if not isinstance(backend, str) or backend not in EXPECTED_BACKENDS:
        raise ValueError("backend must be exactly 'serial' or 'mpi'")
    return backend


def _selected_backend(explicit: object | None) -> str:
    """Select an explicit backend or the authenticated MPI composition lane."""
    environment_backend = None
    if os.environ.get("POPS_REQUIRE_MPI_TESTS") == "1":
        environment_backend = "mpi"
    if explicit is not None:
        selected = _validated_backend(explicit)
        if environment_backend is not None and selected != environment_backend:
            raise ValueError(
                "conflicting M2 backends: --backend=%s but POPS_REQUIRE_MPI_TESTS=1" % selected
            )
        return selected
    if environment_backend is not None:
        return environment_backend
    raise ValueError(
        "backend is required for execution: pass --backend serial|mpi or set "
        "POPS_REQUIRE_MPI_TESTS=1"
    )


def _validated_native_dimension(dimension: object) -> int:
    """Return one supported native dimension, rejecting programmatic bypasses of argparse."""
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension not in SUPPORTED_NATIVE_DIMENSIONS
    ):
        raise ValueError("native dimension must be exactly 1, 2, or 3")
    return dimension


def _selected_native_dimension(explicit: object | None) -> int:
    """Select a dimension from an explicit flag or an authenticated environment value."""
    environment_value = os.environ.get("POPS_NATIVE_DIM")
    environment_dimension: int | None = None
    if environment_value:
        if environment_value not in {str(value) for value in SUPPORTED_NATIVE_DIMENSIONS}:
            raise ValueError(
                "POPS_NATIVE_DIM must be exactly 1, 2, or 3, got %r" % environment_value
            )
        environment_dimension = int(environment_value)

    if explicit is not None:
        selected = _validated_native_dimension(explicit)
        if environment_dimension is not None and environment_dimension != selected:
            raise ValueError(
                "conflicting native dimensions: --dim=%d but POPS_NATIVE_DIM=%d"
                % (selected, environment_dimension)
            )
        return selected
    if environment_dimension is None:
        raise ValueError(
            "native dimension is required for execution: pass --dim 1|2|3 or set "
            "POPS_NATIVE_DIM=1|2|3"
        )
    return environment_dimension


def _selected_checks(
    checks: Iterable[dict], *, backend: str, dimension: int
) -> list[dict]:
    """Return all M2 rows applicable to one backend and native dimension.

    The MPI lane includes serial proofs because the MPI build contains those ordinary tests as
    well as its ``*_npN`` wrappers.  The serial lane deliberately excludes every MPI wrapper.
    """
    backend = _validated_backend(backend)
    dimension = _validated_native_dimension(dimension)

    def supports_dimension(row: dict) -> bool:
        if "dimensions" not in row:
            return True
        dimensions = row["dimensions"]
        return isinstance(dimensions, list) and dimension in dimensions

    if backend == "mpi":
        return [
            row
            for row in checks
            if row.get("backend") in {"serial", "mpi"} and supports_dimension(row)
        ]
    return [
        row
        for row in checks
        if row.get("backend") == "serial" and supports_dimension(row)
    ]


def _required_ctest_targets(
    checks: Iterable[dict], *, backend: str, dimension: int
) -> tuple[str, ...]:
    """Return exact CTest suite targets needed by the filtered M2 rows."""
    selected = _selected_checks(checks, backend=backend, dimension=dimension)
    return tuple(sorted({row["target"] for row in selected if row["kind"] == "ctest"}))


def _required_environment(*, backend: str, dimension: int) -> dict[str, str]:
    """Build the authenticated native/MPI environment for one filtered M2 lane."""
    backend = _validated_backend(backend)
    dimension = _validated_native_dimension(dimension)
    environment = os.environ.copy()
    environment["POPS_REQUIRE_NATIVE_TESTS"] = "1"
    environment["POPS_EXACT_PROCESS_NODEIDS"] = "1"
    if backend == "mpi":
        environment["POPS_REQUIRE_MPI_TESTS"] = "1"
    else:
        environment.pop("POPS_REQUIRE_MPI_TESTS", None)
    environment["POPS_NATIVE_DIM"] = str(dimension)
    return environment


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
        errors.append("issues must list ADC-648 and ADC-661..ADC-668 exactly once")

    deferred = data.get("deferred")
    if deferred != []:
        errors.append("final M2 gate requires deferred = []")

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
        if not isinstance(row, dict):
            errors.append("%s must be a table" % where)
            continue
        base = {"issue", "requirement", "polarity", "kind", "target", "backend"}
        expected = base | ({"nodeid"} if row.get("kind") == "pytest" else {"test_regex"})
        expected_with_optional_dimensions = expected | (
            {"dimensions"} if "dimensions" in row else set()
        )
        if set(row) != expected_with_optional_dimensions:
            errors.append("%s has unknown or missing fields: %s" % (where, sorted(row)))
            continue

        if "dimensions" in row:
            _validate_row_dimensions(row["dimensions"], where, errors)
        issue = row.get("issue")
        requirement = row.get("requirement")
        polarity = row.get("polarity")
        kind = row.get("kind")
        target = row.get("target")
        backend = row.get("backend")
        if issue not in EXPECTED_ISSUES:
            errors.append("%s has unknown or deferred issue %r" % (where, issue))
        if requirement not in EXPECTED_REQUIREMENTS:
            errors.append("%s has unknown requirement %r" % (where, requirement))
        if polarity not in {"positive", "refusal"}:
            errors.append("%s polarity must be positive or refusal" % where)
        else:
            issue_coverage[str(issue)].add(polarity)
            requirement_coverage[str(requirement)].add(polarity)
        if backend not in EXPECTED_BACKENDS:
            errors.append("%s backend must be serial or mpi" % where)
        identity = (kind, row.get("nodeid", row.get("test_regex")))
        identities[identity] += 1
        if kind == "pytest":
            if backend != "serial":
                errors.append("%s ordinary pytest rows require backend = 'serial'" % where)
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
            has_mpi_registration = bool(
                cpp_suites[target].get("mpi_nproc")
                or cpp_suites[target].get("mpi_variants")
                or cpp_suites[target].get("mpi_rank_parity")
            )
            if backend == "mpi" and not has_mpi_registration:
                errors.append("%s mpi CTest target %r has no MPI registration" % (where, target))
            try:
                selected = [
                    case for case in _registered_ctest_cases(cpp_suites[target])
                    if re.fullmatch(selector, case)
                ]
            except re.error as exc:
                errors.append("%s has invalid CTest regex %r: %s" % (where, selector, exc))
                continue
            if len(selected) != 1:
                errors.append("%s CTest selector %r resolves %d registered case(s): %s" %
                              (where, selector, len(selected), sorted(selected)))
            elif backend == "serial" and selected[0] in _registered_mpi_ctest_cases(
                    cpp_suites[target]):
                errors.append(
                    "%s serial CTest row selects MPI case %r" % (where, selected[0])
                )
            elif backend == "mpi" and selected[0] not in _registered_mpi_ctest_cases(
                    cpp_suites[target]):
                errors.append(
                    "%s mpi CTest row selects non-MPI case %r" % (where, selected[0])
                )
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


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    kwargs = {"cwd": ROOT, "check": True}
    if environment is not None:
        kwargs["env"] = environment
    subprocess.run(command, **kwargs)


def _pytest_skip_count(report: Path) -> int:
    if not report.is_file():
        raise RuntimeError("M2 pytest did not produce its mandatory JUnit report")
    try:
        root = ET.parse(report).getroot()
    except ET.ParseError as exc:
        raise RuntimeError("M2 pytest produced an invalid JUnit report") from exc
    return len(root.findall(".//skipped"))


def _run_pytest(
    nodeids: list[str], *, environment: dict[str, str] | None = None
) -> None:
    environment = dict(os.environ) if environment is None else dict(environment)
    environment["POPS_REQUIRE_NATIVE_TESTS"] = "1"
    environment["POPS_EXACT_PROCESS_NODEIDS"] = "1"
    with tempfile.TemporaryDirectory(prefix="pops-m2-gate-") as temporary:
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
        qualifiers = ["POPS_REQUIRE_NATIVE_TESTS=1"]
        if environment.get("POPS_REQUIRE_MPI_TESTS") == "1":
            qualifiers.append("POPS_REQUIRE_MPI_TESTS=1")
        if environment.get("POPS_NATIVE_DIM"):
            qualifiers.append("POPS_NATIVE_DIM=" + environment["POPS_NATIVE_DIM"])
        print("+", " ".join(qualifiers), " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        skipped = _pytest_skip_count(report)
        if skipped:
            raise RuntimeError(
                "M2 pytest reported %d skipped/xfail proof(s); every proof is mandatory" % skipped
            )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def _run_ctest(
    build_dir: Path,
    target: str,
    selector: str,
    *,
    environment: dict[str, str] | None = None,
) -> None:
    kwargs = {
        "cwd": ROOT,
        "check": True,
        "text": True,
        "capture_output": True,
    }
    if environment is not None:
        kwargs["env"] = environment
    listed = subprocess.run(
        ["ctest", "--test-dir", str(build_dir), "-N", "-R", selector],
        **kwargs,
    )
    match = re.search(r"^Total Tests: (\d+)$", listed.stdout, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("M2 CTest discovery did not report a test total for %r (%s)"
                           % (target, selector))
    selected_count = int(match.group(1))
    if selected_count != 1:
        raise RuntimeError("M2 CTest selector %r for target %r resolved %d tests, expected exactly 1"
                           % (selector, target, selected_count))
    _run(
        ["ctest", "--test-dir", str(build_dir), "--output-on-failure", "-R", selector],
        environment=environment,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--list-ctest-targets", action="store_true")
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--backend", choices=sorted(EXPECTED_BACKENDS))
    parser.add_argument("--dim", type=int, choices=SUPPORTED_NATIVE_DIMENSIONS)
    args = parser.parse_args(argv)

    data, errors = validate_manifest(args.manifest)
    if errors:
        print("M2 gate manifest is incomplete or invalid:", file=sys.stderr)
        for error in errors:
            print(" -", error, file=sys.stderr)
        return 2
    checks = data["check"]
    print("M2 gate source matrix: OK (%d executable, %d deferred)"
          % (len(checks), len(data["deferred"])))
    if args.check_only:
        return 0

    try:
        backend = _selected_backend(args.backend)
        dimension = _selected_native_dimension(args.dim)
    except ValueError as exc:
        parser.error(str(exc))
    selected = _selected_checks(checks, backend=backend, dimension=dimension)
    if not selected:
        parser.error(
            "M2 gate selects no executable rows for backend=%s, dimension=%d"
            % (backend, dimension)
        )
    environment = _required_environment(backend=backend, dimension=dimension)
    if args.list_ctest_targets:
        targets = _required_ctest_targets(
            checks, backend=backend, dimension=dimension
        )
        if not targets:
            print("M2 gate selects no CTest build target", file=sys.stderr)
            return 2
        print("\n".join(targets))
        return 0

    print(
        "M2 gate execution matrix: backend=%s, dimension=%d, selected=%d"
        % (backend, dimension, len(selected))
    )
    nodeids = [row["nodeid"] for row in selected if row["kind"] == "pytest"]
    for chunk in _chunks(nodeids, 24):
        _run_pytest(chunk, environment=environment)
    if not args.python_only:
        for row in sorted(
                (row for row in selected if row["kind"] == "ctest"),
                key=lambda value: (value["target"], value["test_regex"])):
            _run_ctest(
                args.build_dir,
                row["target"],
                row["test_regex"],
                environment=environment,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
