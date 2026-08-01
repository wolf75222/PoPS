#!/usr/bin/env python3
"""Validate and run the bounded ADC-757 prepared-numerics evidence gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests/gates/adc757_prepared_numerics.toml"
TEST_MANIFEST = ROOT / "tests/test_manifest.toml"
EXPECTED_REQUIREMENTS = {
    "prepared_local_nonlinear",
    "typed_fallible_evaluation",
    "transactional_recovery_publication",
    "allocation_aware_cell_hot_path",
    "prepared_boundary_publication",
    "capability_driven_riemann",
    "mpi_collective_execution",
    "typed_flux_recovery_consumption",
    "runtime_recovery_consumer_publication",
    "model_declared_admissibility",
    "prepared_limiter_provider",
}
EXPECTED_DEFERRED = (
    "remaining_3d_metric_eb_characteristic_and_spatial_provider_matrix",
    "python_ir_generated_abi_and_restart_parity",
    "remaining_legacy_recovery_and_boundary_authority_deletion",
    "amr_regrid_migration_and_restart_coherence",
    "gpu_backend_execution",
    "workspace_reentrancy_and_stream_partitioning",
    "performance_baselines_and_end_to_end_benchmarks",
    "local_time_and_load_balance_provider_families",
)
GTEST_PATTERN = re.compile(r"\bTEST(?:_F)?\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*\)")


def _cpp_suites() -> dict[str, dict]:
    data = tomllib.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    return {str(row["name"]): row for row in data.get("cpp", {}).get("suite", ())}


def _declared_gtests(suite: dict) -> tuple[set[str], list[str]]:
    names: set[str] = set()
    errors: list[str] = []
    for relative in suite.get("sources", ()):
        source = ROOT / relative
        if not source.is_file():
            errors.append("missing source %s" % relative)
            continue
        text = source.read_text(encoding="utf-8")
        if "GTEST_SKIP" in text or "DISABLED_" in text:
            errors.append("%s contains a skip/disabled marker" % relative)
        names.update("%s.%s" % match.groups() for match in GTEST_PATTERN.finditer(text))
    return names, errors


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
    if data.get("schema_version") != 1:
        errors.append("schema_version must be exactly 1")
    if data.get("gate") != "adc757-prepared-numerics-slice":
        errors.append("gate must be exactly 'adc757-prepared-numerics-slice'")
    if data.get("issue") != "ADC-757":
        errors.append("issue must be exactly ADC-757")
    expected_evidence = ["ADC-749", "ADC-750", "ADC-752", "ADC-753", "ADC-754", "ADC-755"]
    if data.get("evidence_from") != expected_evidence:
        errors.append("evidence_from must be exactly %s" % expected_evidence)
    if data.get("deferred") != list(EXPECTED_DEFERRED):
        errors.append("deferred must enumerate every deliberately unproved family exactly")

    checks = data.get("check")
    if not isinstance(checks, list) or not checks:
        errors.append("manifest must contain [[check]] rows")
        checks = []
    suites = _cpp_suites()
    coverage: dict[str, set[str]] = defaultdict(set)
    identities = Counter()
    mpi_checks = 0
    for index, row in enumerate(checks, 1):
        where = "check[%d]" % index
        kind = row.get("kind", "ctest")
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
        identity = (target, selector)
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
        names, source_errors = _declared_gtests(suites[target])
        errors.extend("%s: %s" % (where, error) for error in source_errors)
        try:
            matches = sorted(name for name in names if re.fullmatch(selector, name))
        except re.error as exc:
            errors.append("%s has invalid test_regex: %s" % (where, exc))
            continue
        if len(matches) != 1:
            errors.append(
                "%s must resolve to exactly one declared GTest; got %s" % (where, matches)
            )

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
    for row in sorted(data["check"], key=lambda value: (value["target"], value["test_regex"])):
        _run_ctest(args.build_dir, row["target"], row["test_regex"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
