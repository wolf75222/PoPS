#!/usr/bin/env python3
"""Validate and run the bounded ADC-757 prepared-numerics evidence gate."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests/gates/adc757_prepared_numerics.toml"
TEST_MANIFEST = ROOT / "tests/test_manifest.toml"
EXPECTED_REQUIREMENTS = {
    "prepared_local_nonlinear",
    "typed_fallible_evaluation",
    "transactional_recovery_publication",
    "allocation_aware_cell_hot_path",
    "prepared_boundary_publication",
    "post_riemann_boundary_flux",
    "qualified_flux_provider_pack",
    "capability_driven_riemann",
    "mpi_collective_execution",
    "typed_flux_recovery_consumption",
    "runtime_recovery_consumer_publication",
    "analytic_initial_recovery_publication",
    "fallible_primitive_to_conservative_publication",
    "amr_regrid_recovery_publication",
    "amr_restriction_recovery_publication",
    "amr_bootstrap_recovery_publication",
    "amr_history_recovery_publication",
    "physical_boundary_trace_recovery_publication",
    "terminal_source_recovery_publication",
    "type_erased_recovery_method_identity",
    "model_declared_admissibility",
    "prepared_limiter_provider",
    "cell_local_temporal_partition_authority",
    "python_ir_generated_abi_and_restart_parity",
    "host_workspace_reentrancy",
}
EXPECTED_DEFERRED = (
    "remaining_3d_metric_eb_characteristic_and_spatial_provider_matrix",
    "remaining_legacy_recovery_and_boundary_authority_deletion",
    "amr_regrid_migration_and_restart_coherence",
    "gpu_backend_execution",
    "accelerator_stream_partitioning",
    "performance_baselines_and_end_to_end_benchmarks",
    "local_time_and_load_balance_provider_families",
)
GTEST_PATTERN = re.compile(r"\bTEST(?:_F)?\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*\)")


def _cpp_suites() -> dict[str, dict]:
    data = tomllib.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    return {str(row["name"]): row for row in data.get("cpp", {}).get("suite", ())}


def _python_files() -> set[str]:
    data = tomllib.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    files: set[str] = set()
    for suite in data.get("python", {}).get("suite", ()):
        relative_root = suite.get("path")
        if not isinstance(relative_root, str):
            continue
        root = ROOT / relative_root
        if not root.is_dir():
            continue
        files.update(
            source.relative_to(ROOT).as_posix()
            for source in root.rglob("test_*.py")
            if source.is_file()
        )
    return files


def _declared_gtests(suite: dict) -> tuple[dict[str, bool], list[str]]:
    tests: dict[str, bool] = {}
    errors: list[str] = []
    for relative in suite.get("sources", ()):
        source = ROOT / relative
        if not source.is_file():
            errors.append("missing source %s" % relative)
            continue
        text = source.read_text(encoding="utf-8")
        matches = list(GTEST_PATTERN.finditer(text))
        for index, match in enumerate(matches):
            suite_name, test_name = match.groups()
            name = "%s.%s" % (suite_name, test_name)
            if name in tests:
                errors.append("duplicate declared GTest %s" % name)
                continue
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[match.start():end]
            tests[name] = (
                suite_name.startswith("DISABLED_")
                or test_name.startswith("DISABLED_")
                or "GTEST_SKIP" in body
            )
    return tests, errors


def _declared_pytests(relative: str) -> tuple[dict[str, ast.FunctionDef], list[str]]:
    source = ROOT / relative
    if not source.is_file():
        return {}, ["missing source %s" % relative]
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=relative)
    except (OSError, SyntaxError) as exc:
        return {}, ["cannot parse %s: %s" % (relative, exc)]
    tests = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }
    return tests, []


def _pytest_is_skipped(test: ast.FunctionDef) -> bool:
    blocked_decorators = ("pytest.mark.skip", "pytest.mark.skipif", "pytest.mark.xfail")
    if any(
        any(blocked in ast.unparse(decorator) for blocked in blocked_decorators)
        for decorator in test.decorator_list
    ):
        return True
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
        and node.func.attr in {"skip", "xfail"}
        for node in ast.walk(test)
    )


def validate_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[dict, list[str]]:
    """Return the manifest and deterministic source-only validation errors."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return {}, ["cannot read ADC-757 gate manifest %s: %s" % (path, exc)]

    errors: list[str] = []
    expected_fields = {
        "schema_version",
        "gate",
        "issue",
        "evidence_from",
        "deferred",
        "check",
    }
    if set(data) != expected_fields:
        errors.append("manifest fields must be exactly %s" % sorted(expected_fields))
    if data.get("schema_version") != 2:
        errors.append("schema_version must be exactly 2")
    if data.get("gate") != "adc757-prepared-numerics-slice":
        errors.append("gate must be exactly 'adc757-prepared-numerics-slice'")
    if data.get("issue") != "ADC-757":
        errors.append("issue must be exactly ADC-757")
    expected_evidence = [
        "ADC-682",
        "ADC-749",
        "ADC-750",
        "ADC-751",
        "ADC-752",
        "ADC-753",
        "ADC-754",
        "ADC-755",
        "ADC-756",
    ]
    if data.get("evidence_from") != expected_evidence:
        errors.append("evidence_from must be exactly %s" % expected_evidence)
    if data.get("deferred") != list(EXPECTED_DEFERRED):
        errors.append("deferred must enumerate every deliberately unproved family exactly")

    checks = data.get("check")
    if not isinstance(checks, list) or not checks:
        errors.append("manifest must contain [[check]] rows")
        checks = []
    suites = _cpp_suites()
    python_files = _python_files()
    coverage: dict[str, set[str]] = defaultdict(set)
    identities = Counter()
    mpi_checks = 0
    for index, row in enumerate(checks, 1):
        where = "check[%d]" % index
        kind = row.get("kind", "ctest")
        if kind == "pytest":
            expected_row_fields = {"requirement", "polarity", "kind", "path", "test"}
        else:
            expected_row_fields = {"requirement", "polarity", "target", "test_regex"}
        if kind == "mpi_ctest":
            expected_row_fields.update({"kind", "nproc"})
        if set(row) != expected_row_fields:
            errors.append("%s has unknown or missing fields" % where)
            continue
        requirement = row.get("requirement")
        polarity = row.get("polarity")
        target = row.get("target")
        selector = row.get("test_regex")
        if requirement not in EXPECTED_REQUIREMENTS:
            errors.append("%s has unknown requirement %r" % (where, requirement))
        if polarity not in {"positive", "refusal"}:
            errors.append("%s polarity must be positive or refusal" % where)
        else:
            coverage[str(requirement)].add(str(polarity))
        if kind == "pytest":
            relative = row.get("path")
            test_name = row.get("test")
            identity = (kind, relative, test_name)
            identities[identity] += 1
            if relative not in python_files:
                errors.append("%s references unknown Python test file %r" % (where, relative))
                continue
            declared, source_errors = _declared_pytests(str(relative))
            errors.extend("%s: %s" % (where, error) for error in source_errors)
            if test_name not in declared:
                errors.append(
                    "%s references unknown top-level pytest %r in %r"
                    % (where, test_name, relative)
                )
                continue
            if _pytest_is_skipped(declared[str(test_name)]):
                errors.append("%s pytest proof %r is skipped or xfailed" % (where, test_name))
            continue
        identity = (kind, target, selector)
        identities[identity] += 1
        if (
            not isinstance(selector, str)
            or not selector.startswith("^")
            or not selector.endswith("$")
        ):
            errors.append("%s must use one anchored exact CTest regex" % where)
            continue
        if target not in suites:
            errors.append("%s references unknown CTest target %r" % (where, target))
            continue
        suite = suites[target]
        labels = {str(label) for label in suite.get("labels", ())}
        if kind == "mpi_ctest":
            mpi_checks += 1
            nproc = row.get("nproc")
            if "mpi" not in labels:
                errors.append("%s mpi_ctest target %r lacks the mpi label" % (where, target))
            if (
                isinstance(nproc, bool)
                or not isinstance(nproc, int)
                or nproc < 1
                or nproc not in suite.get("mpi_nproc", ())
            ):
                errors.append(
                    "%s nproc must be one exact rank count declared by %r" % (where, target)
                )
            expected_selector = "^%s_np%s$" % (target, nproc)
            if selector != expected_selector:
                errors.append(
                    "%s mpi_ctest selector must be exactly %r" % (where, expected_selector)
                )
            continue
        if kind != "ctest":
            errors.append("%s has unknown check kind %r" % (where, kind))
            continue
        if "mpi" in labels or "gpu" in labels:
            errors.append("%s ordinary CTest claims a deferred MPI/GPU target %r" % (where, target))
        declared, source_errors = _declared_gtests(suites[target])
        errors.extend("%s: %s" % (where, error) for error in source_errors)
        try:
            matches = sorted(name for name in declared if re.fullmatch(selector, name))
        except re.error as exc:
            errors.append("%s has invalid test_regex: %s" % (where, exc))
            continue
        if len(matches) != 1:
            errors.append(
                "%s must resolve to exactly one declared GTest; got %s" % (where, matches)
            )
        elif declared[matches[0]]:
            errors.append("%s selected CTest %r is skipped or disabled" % (where, matches[0]))

    duplicates = sorted(identity for identity, count in identities.items() if count > 1)
    if duplicates:
        errors.append("duplicate executable checks: %s" % duplicates)
    if mpi_checks != 2:
        errors.append(
            "the closed mpi_collective_execution family requires exactly two MPI CTests"
        )
    for requirement in sorted(EXPECTED_REQUIREMENTS):
        missing = {"positive", "refusal"} - coverage[requirement]
        if missing:
            errors.append("%s lacks %s coverage" % (requirement, "/".join(sorted(missing))))
    return data, errors


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
            "ADC-757 proof target %r (%s) is not built in %s" % (target, selector, build_dir)
        )
    command = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--output-on-failure",
        "-R",
        selector,
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _pytest_skip_count(report: Path) -> int:
    if not report.is_file():
        raise RuntimeError("ADC-757 pytest did not produce its mandatory JUnit report")
    try:
        root = ET.parse(report).getroot()
    except ET.ParseError as exc:
        raise RuntimeError("ADC-757 pytest produced an invalid JUnit report") from exc
    return len(root.findall(".//skipped"))


def _run_pytest(relative: str, test_name: str) -> None:
    environment = os.environ.copy()
    environment["POPS_REQUIRE_MPI_TESTS"] = "1"
    environment["POPS_REQUIRE_NATIVE_TESTS"] = "1"
    with tempfile.TemporaryDirectory(prefix="pops-adc757-gate-") as temporary:
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
            "%s::%s" % (relative, test_name),
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
                "ADC-757 pytest reported %d skipped/xfail proof(s); every proof is mandatory"
                % skipped
            )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--closure",
        action="store_true",
        help="require full ADC-757 closure (intentionally refused while deferred remains)",
    )
    args = parser.parse_args(argv)

    data, errors = validate_manifest(args.manifest)
    if errors:
        print("ADC-757 prepared-numerics gate is invalid:", file=sys.stderr)
        for error in errors:
            print(" -", error, file=sys.stderr)
        return 2
    print(
        "ADC-757 prepared-numerics slice: OK "
        "(%d executable proofs, %d explicitly deferred families)"
        % (len(data["check"]), len(data["deferred"]))
    )
    if args.closure:
        print(
            "ADC-757 closure refused: %d required families remain deferred" % len(data["deferred"]),
            file=sys.stderr,
        )
        return 3
    if args.check_only:
        return 0
    checks = sorted(
        data["check"],
        key=lambda value: (
            value.get("kind", "ctest"),
            value.get("target", value.get("path", "")),
            value.get("test_regex", value.get("test", "")),
        ),
    )
    for row in checks:
        if row.get("kind") == "pytest":
            _run_pytest(row["path"], row["test"])
        else:
            _run_ctest(args.build_dir, row["target"], row["test_regex"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
