#!/usr/bin/env python3
"""Audit and run the fail-closed M4 native-runtime/IO conformance matrix."""

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
DEFAULT_MANIFEST = ROOT / "tests/gates/m4_runtime_io.toml"
TEST_MANIFEST = ROOT / "tests/test_manifest.toml"
EXPECTED_ISSUES = tuple("ADC-%d" % number for number in range(679, 688))
REQUIRED_POLARITIES = {
    "component_manifest": {"positive", "refusal"},
    "generated_registry": {"positive", "refusal"},
    "external_package": {"positive", "refusal"},
    "external_flux": {"positive"},
    "external_boundary": {"positive"},
    "external_tagger": {"positive"},
    "external_transfer": {"positive"},
    "external_solver": {"positive"},
    "external_writer": {"positive"},
    "native_interfaces": {"positive", "refusal"},
    "flux_contract": {"positive", "refusal"},
    "platform_execution": {"positive", "refusal"},
    "runtime_instance": {"positive", "refusal"},
    "consumer_graph": {"positive", "refusal"},
    "accepted_publication": {"positive", "refusal"},
    "exact_npz": {"positive", "refusal"},
    "exact_hdf5": {"positive", "refusal"},
    "exact_paraview": {"positive", "refusal"},
    "collective_hdf5": {"positive"},
    "strict_checkpoint": {"positive", "refusal"},
    "diagnostics": {"positive", "refusal"},
    "tamper_capability_abi": {"refusal"},
    "legacy_stepper_retirement": {"positive"},
    "gate_execution": {"positive"},
}
REQUIREMENT_ISSUES = {
    "component_manifest": {"ADC-679"},
    "generated_registry": {"ADC-679"},
    "external_package": {"ADC-680"},
    "external_flux": {"ADC-680"},
    "native_interfaces": {"ADC-681"},
    "external_boundary": {"ADC-681"},
    "external_tagger": {"ADC-681"},
    "flux_contract": {"ADC-682"},
    "platform_execution": {"ADC-683"},
    "runtime_instance": {"ADC-684"},
    "external_transfer": {"ADC-684"},
    "external_writer": {"ADC-685"},
    "consumer_graph": {"ADC-685"},
    "accepted_publication": {"ADC-685"},
    "exact_npz": {"ADC-686"},
    "exact_hdf5": {"ADC-686"},
    "exact_paraview": {"ADC-686"},
    "collective_hdf5": {"ADC-686"},
    "strict_checkpoint": {"ADC-686"},
    "diagnostics": {"ADC-686"},
    "external_solver": {"ADC-687"},
    "legacy_stepper_retirement": {"ADC-687"},
    "gate_execution": {"ADC-687"},
    "tamper_capability_abi": {"ADC-679", "ADC-680", "ADC-683", "ADC-687"},
}
NATIVE_PYTEST_PREFIXES = (
    "tests/python/integration/amr/",
    "tests/python/integration/io/",
    "tests/python/integration/mpi/",
    "tests/python/integration/native_loader/",
    "tests/python/integration/runtime/",
)
_GTEST_DECLARATION = re.compile(
    r"\bTEST(?:_F)?\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\)"
)
_CPP_RAW_STRING_START = re.compile(r'(?:u8|u|U|L)?R"([^\s()\\]{0,16})\(')
_MOCK_FIXTURES = {"monkeypatch", "mocker", "mock", "patch"}
_FORBIDDEN_CALLS = {
    "pytest.skip",
    "pytest.xfail",
    "pytest.importorskip",
    "unittest.mock.patch",
    "mock.patch",
    "require_mpi_or_skip",
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


def _forbidden_python_markers(node: ast.AST) -> list[str]:
    markers: list[str] = []
    for decorator in getattr(node, "decorator_list", ()):
        name = _dotted_name(decorator)
        if name.endswith((".skip", ".skipif", ".xfail")) or name in {
            "skip",
            "skipif",
            "xfail",
        }:
            markers.append(name)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        fixtures = {
            argument.arg
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
        }
        markers.extend("fixture:%s" % name for name in sorted(fixtures & _MOCK_FIXTURES))
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _dotted_name(child.func)
            if name in _FORBIDDEN_CALLS or name.endswith(
                (".importorskip", ".skip", ".xfail", ".mock", ".patch")
            ):
                markers.append(name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            module = (child.module or "") if isinstance(child, ast.ImportFrom) else ""
            names = [alias.name for alias in child.names]
            if module.startswith(("unittest.mock", "pytest_mock")) or any(
                name.startswith(("unittest.mock", "pytest_mock")) for name in names
            ):
                markers.append("mock-import")
        elif isinstance(child, ast.Try):
            for handler in child.handlers:
                caught = _dotted_name(handler.type) if handler.type is not None else ""
                if caught in {"ImportError", "ModuleNotFoundError"}:
                    markers.append("optional-import-fallback")
    return markers


def _has_authenticated_mpi_guard(module: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "tests.python.support.requirements"
        and any(alias.name == "require_mpi_or_skip" for alias in node.names)
        for node in ast.walk(module)
    )


def _ctest_suites() -> dict[str, dict]:
    data = tomllib.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    return {str(row["name"]): row for row in data.get("cpp", {}).get("suite", ())}


def _python_suites() -> tuple[dict, ...]:
    data = tomllib.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    return tuple(data.get("python", {}).get("suite", ()))


def _python_mpi_entrypoints() -> dict[str, int]:
    entries: dict[str, int] = {}
    for suite in _python_suites():
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
    orchestrators: set[str] = set()
    for suite in _python_suites():
        for row in suite.get("mpi_orchestrators", ()):
            if not isinstance(row, dict) or set(row) != {"path"}:
                raise ValueError(
                    "invalid Python MPI orchestrator %r; expected exactly one path field"
                    % row
                )
            path = row["path"]
            if not isinstance(path, str) or not path:
                raise ValueError("invalid Python MPI orchestrator path %r" % path)
            if path in orchestrators:
                raise ValueError("duplicate Python MPI orchestrator %s" % path)
            orchestrators.add(path)
    return orchestrators


def _python_suite_owns(relative: str) -> bool:
    path = Path(relative)
    return any(
        path == Path(str(suite.get("path", "")))
        or Path(str(suite.get("path", ""))) in path.parents
        for suite in _python_suites()
    )


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
    return {
        "%s.%s" % declaration
        for declaration in _GTEST_DECLARATION.findall(_cpp_code_only(source))
    }


def _registered_ctest_cases(target: str, suite: dict) -> set[str]:
    cases: set[str] = set()
    for relative in suite.get("sources", ()):
        source = ROOT / relative
        if source.is_file():
            cases.update(_registered_gtest_cases(source.read_text(encoding="utf-8")))
    for field in ("mpi_nproc", "mpi_rank_parity", "mpi_variants"):
        cases.update(
            "%s_np%d" % (target, nproc)
            for nproc in suite.get(field, ())
            if not isinstance(nproc, bool) and isinstance(nproc, int) and nproc > 0
        )
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
    exact = {
        "^%s$" % re.escape(case)
        for case in _registered_ctest_cases(target, suite)
    }
    if selector not in exact:
        errors.append(
            "%s CTest selector %r is not one exact source-registered case for target %r"
            % (where, selector, target)
        )


def _validate_python_nodeid(
    nodeid: object,
    where: str,
    errors: list[str],
    *,
    mpi_entrypoint: bool = False,
) -> str | None:
    if not isinstance(nodeid, str) or nodeid.count("::") != 1:
        errors.append("%s must contain one exact file::test nodeid" % where)
        return None
    relative, function_name = nodeid.split("::")
    test_path = ROOT / relative
    if not test_path.is_file():
        errors.append("%s references missing test file %s" % (where, relative))
        return None
    if not _python_suite_owns(relative):
        errors.append("%s is not owned by tests/test_manifest.toml" % relative)
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
    module_nodes = [
        node
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    markers = _forbidden_python_markers(function)
    module = ast.Module(body=module_nodes, type_ignores=[])
    module_markers = _forbidden_python_markers(module)
    if mpi_entrypoint and "require_mpi_or_skip" in module_markers:
        if not _has_authenticated_mpi_guard(module):
            errors.append(
                "%s uses an unauthenticated MPI prerequisite guard" % nodeid
            )
        module_markers = [
            marker for marker in module_markers
            if marker != "require_mpi_or_skip"
        ]
    markers.extend(module_markers)
    if markers:
        errors.append(
            "%s is not an unconditional real proof; found %s"
            % (nodeid, sorted(set(markers)))
        )
    return relative


def _validate_deferred(
    data: dict, errors: list[str]
) -> set[tuple[str, str, str]]:
    rows = data.get("deferred")
    if not isinstance(rows, list):
        errors.append("deferred must be an array of explicit gap tables")
        return set()
    gaps: set[tuple[str, str, str]] = set()
    identities = Counter()
    for index, row in enumerate(rows, 1):
        where = "deferred[%d]" % index
        expected = {
            "issue",
            "requirement",
            "polarity",
            "reason",
            "evidence_paths",
        }
        if not isinstance(row, dict) or set(row) != expected:
            errors.append(
                "%s must contain issue/requirement/polarity/reason/evidence_paths"
                % where
            )
            continue
        issue = row.get("issue")
        requirement = row.get("requirement")
        polarity = row.get("polarity")
        reason = row.get("reason")
        evidence_paths = row.get("evidence_paths")
        if issue not in EXPECTED_ISSUES:
            errors.append("%s has unknown issue %r" % (where, issue))
        if requirement not in REQUIRED_POLARITIES:
            errors.append("%s has unknown requirement %r" % (where, requirement))
        elif issue not in REQUIREMENT_ISSUES[requirement]:
            errors.append(
                "%s requirement %r cannot be deferred under %r"
                % (where, requirement, issue)
            )
        if polarity not in {"positive", "refusal"}:
            errors.append("%s polarity must be positive or refusal" % where)
        elif (
            requirement in REQUIRED_POLARITIES
            and polarity not in REQUIRED_POLARITIES[requirement]
        ):
            errors.append(
                "%s requirement %r has no %r polarity"
                % (where, requirement, polarity)
            )
        if not isinstance(reason, str) or len(reason.strip()) < 20:
            errors.append("%s requires a precise non-empty reason" % where)
        if not isinstance(evidence_paths, list) or not evidence_paths:
            errors.append("%s requires at least one evidence path" % where)
        else:
            for relative in evidence_paths:
                if not isinstance(relative, str) or not relative:
                    errors.append("%s has an invalid evidence path %r" % (where, relative))
                elif not (ROOT / relative).exists():
                    errors.append(
                        "%s gap evidence path no longer exists: %s" % (where, relative)
                    )
        identity = (str(issue), str(requirement), str(polarity))
        identities[identity] += 1
        gaps.add(identity)
    duplicates = sorted(identity for identity, count in identities.items() if count > 1)
    if duplicates:
        errors.append("duplicate deferred gaps: %s" % duplicates)
    return gaps


def audit_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[dict, list[str]]:
    """Return source-only structural errors without pretending deferred gaps are closed."""
    errors: list[str] = []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return {}, ["cannot read M4 gate manifest %s: %s" % (path, exc)]

    if data.get("schema_version") != 1:
        errors.append("schema_version must be exactly 1")
    if data.get("gate") != "m4-runtime-io":
        errors.append("gate must be exactly 'm4-runtime-io'")
    if set(data) != {"schema_version", "gate", "issues", "deferred", "check"}:
        errors.append("manifest fields must be schema_version/gate/issues/deferred/check")
    if data.get("issues") != list(EXPECTED_ISSUES):
        errors.append("issues must list ADC-679..ADC-687 exactly once")

    deferred_gaps = _validate_deferred(data, errors)
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
        mpi_entrypoints = _python_mpi_entrypoints()
        mpi_orchestrators = _python_mpi_orchestrators()
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        errors.append("cannot read Python MPI ownership: %s" % exc)
        mpi_entrypoints = {}
        mpi_orchestrators = set()

    for index, row in enumerate(checks, 1):
        where = "check[%d]" % index
        base = {"issue", "requirement", "polarity", "kind", "target"}
        kind = row.get("kind") if isinstance(row, dict) else None
        expected = (
            base | {"nodeid", "nproc"}
            if kind == "mpi_python"
            else base | ({"nodeid"} if kind == "pytest" else {"test_regex"})
        )
        if not isinstance(row, dict) or set(row) != expected:
            errors.append("%s has unknown or missing fields: %s" % (where, sorted(row)))
            continue

        issue = row.get("issue")
        requirement = row.get("requirement")
        polarity = row.get("polarity")
        target = row.get("target")
        if issue not in EXPECTED_ISSUES:
            errors.append("%s has unknown issue %r" % (where, issue))
        if requirement not in REQUIRED_POLARITIES:
            errors.append("%s has unknown requirement %r" % (where, requirement))
        elif issue not in REQUIREMENT_ISSUES[requirement]:
            errors.append(
                "%s requirement %r cannot be attributed to %r"
                % (where, requirement, issue)
            )
        if kind != "ctest" and target != requirement:
            errors.append(
                "%s target must equal its exact requirement %r" % (where, requirement)
            )
        if polarity not in {"positive", "refusal"}:
            errors.append("%s polarity must be positive or refusal" % where)
        else:
            issue_coverage[str(issue)].add(polarity)
            requirement_coverage[str(requirement)].add(polarity)

        identity = (kind, row.get("nodeid", row.get("test_regex")))
        identities[identity] += 1
        if kind == "pytest":
            relative = _validate_python_nodeid(row.get("nodeid"), where, errors)
            if (
                relative is not None
                and relative.startswith("tests/python/integration/mpi/")
                and relative not in mpi_orchestrators
            ):
                errors.append(
                    "%s is not a manifest-owned serial MPI orchestrator" % relative
                )
            if (
                polarity == "positive"
                and relative is not None
                and relative.startswith(NATIVE_PYTEST_PREFIXES)
            ):
                native_positive_issues.add(str(issue))
        elif kind == "mpi_python":
            relative = _validate_python_nodeid(
                row.get("nodeid"),
                where,
                errors,
                mpi_entrypoint=True,
            )
            nproc = row.get("nproc")
            if isinstance(nproc, bool) or not isinstance(nproc, int) or nproc < 1:
                errors.append("%s MPI Python row requires a positive integer nproc" % where)
            elif relative is not None:
                expected_nproc = mpi_entrypoints.get(relative)
                if expected_nproc is None:
                    errors.append("%s is not a manifest-owned MPI Python entrypoint" % relative)
                elif expected_nproc != nproc:
                    errors.append(
                        "%s requires nproc=%d, not %d"
                        % (relative, expected_nproc, nproc)
                    )
            if polarity == "positive":
                native_positive_issues.add(str(issue))
        elif kind == "ctest":
            target_name = row.get("target")
            # CTest rows still carry the semantic requirement in target, so the
            # build target is encoded as "requirement@ctest-target".
            if not isinstance(target_name, str) or "@" not in target_name:
                errors.append(
                    "%s CTest target must be requirement@manifest-suite" % where
                )
                continue
            semantic, suite_name = target_name.split("@", 1)
            if semantic != requirement:
                errors.append(
                    "%s CTest target must start with requirement %r" % (where, requirement)
                )
            suite = cpp_suites.get(suite_name)
            if suite is None:
                errors.append("%s references unknown CTest target %r" % (where, suite_name))
                continue
            _validate_exact_ctest_selector(
                row.get("test_regex"), suite_name, suite, where, errors
            )
            for relative in suite.get("sources", ()):
                source = ROOT / relative
                if not source.is_file():
                    errors.append(
                        "%s target %r has missing source %s"
                        % (where, suite_name, relative)
                    )
                else:
                    text = source.read_text(encoding="utf-8")
                    if "DISABLED_" in text:
                        errors.append(
                            "%s target %r contains a disabled test" % (where, suite_name)
                        )
            if polarity == "positive":
                native_positive_issues.add(str(issue))
        else:
            errors.append("%s kind must be pytest, mpi_python, or ctest" % where)

    duplicates = sorted(identity for identity, count in identities.items() if count > 1)
    if duplicates:
        errors.append("duplicate executable checks: %s" % duplicates)
    for issue in EXPECTED_ISSUES:
        missing = {"positive", "refusal"} - issue_coverage[issue]
        unresolved = {
            polarity
            for polarity in missing
            if not any(
                deferred_issue == issue and deferred_polarity == polarity
                for deferred_issue, _requirement, deferred_polarity in deferred_gaps
            )
        }
        if unresolved:
            errors.append(
                "%s lacks %s coverage"
                % (issue, "/".join(sorted(unresolved)))
            )
        if issue not in native_positive_issues:
            errors.append("%s lacks a mandatory native positive proof" % issue)
    for requirement, required in sorted(REQUIRED_POLARITIES.items()):
        missing = required - requirement_coverage[requirement]
        unresolved = {
            polarity
            for polarity in missing
            if not any(
                deferred_requirement == requirement
                and deferred_polarity == polarity
                for _issue, deferred_requirement, deferred_polarity in deferred_gaps
            )
        }
        if unresolved:
            errors.append(
                "%s lacks %s coverage"
                % (requirement, "/".join(sorted(unresolved)))
            )
    return data, errors


def validate_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[dict, list[str]]:
    """Fail closed when even one structurally valid M4 requirement is deferred."""
    data, errors = audit_manifest(path)
    if errors:
        return data, errors
    for row in data["deferred"]:
        errors.append(
            "%s/%s/%s remains deferred: %s"
            % (
                row["issue"],
                row["requirement"],
                row["polarity"],
                row["reason"],
            )
        )
    return data, errors


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def _required_environment() -> dict[str, str]:
    environment = os.environ.copy()
    if environment.get("POPS_NATIVE_DIM") != "2":
        raise RuntimeError(
            "M4 execution requires launcher-provided POPS_NATIVE_DIM=2; "
            "native dimension inference is forbidden"
        )
    environment["POPS_REQUIRE_MPI_TESTS"] = "1"
    environment["POPS_REQUIRE_NATIVE_TESTS"] = "1"
    root = str(ROOT)
    inherited = environment.get("PYTHONPATH", "")
    python_path = [root]
    python_path.extend(
        entry
        for entry in inherited.split(os.pathsep)
        if entry and entry != root
    )
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    return environment


def _mpi_python_command(mpi_exec: str, nproc: int, relative: str) -> list[str]:
    if shutil.which(mpi_exec) is None:
        raise RuntimeError("required MPI launcher %r is unavailable" % mpi_exec)
    bootstrap = (
        "from pops._native_selector import select_native_dimension; "
        "select_native_dimension(2); "
        "import runpy, sys; "
        "runpy.run_path(sys.argv[1], run_name='__main__')"
    )
    return [
        mpi_exec,
        "-n",
        str(nproc),
        sys.executable,
        "-c",
        bootstrap,
        str(ROOT / relative),
    ]


def _junit_skip_count(report: Path, producer: str) -> int:
    if not report.is_file():
        raise RuntimeError("M4 %s did not produce its mandatory JUnit report" % producer)
    try:
        root = ET.parse(report).getroot()
    except ET.ParseError as exc:
        raise RuntimeError("M4 %s produced an invalid JUnit report" % producer) from exc
    return len(root.findall(".//skipped"))


def _run_required_pytest(nodeids: list[str]) -> None:
    environment = _required_environment()
    with tempfile.TemporaryDirectory(prefix="pops-m4-gate-") as temporary:
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
        skipped = _junit_skip_count(report, "pytest")
        if skipped:
            raise RuntimeError(
                "M4 pytest reported %d skipped/xfail proof(s); every proof is mandatory"
                % skipped
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
            "M4 CTest target %r (%s) is not built in %s"
            % (target, selector, build_dir)
        )
    with tempfile.TemporaryDirectory(prefix="pops-m4-ctest-") as temporary:
        report = Path(temporary) / "ctest.xml"
        command = [
            "ctest",
            "--test-dir",
            str(build_dir),
            "--output-on-failure",
            "--output-junit",
            str(report),
            "-R",
            selector,
        ]
        print("+", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        skipped = _junit_skip_count(report, "CTest")
        if skipped:
            raise RuntimeError(
                "M4 CTest %r reported %d skipped proof(s)" % (selector, skipped)
            )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)


def _required_ctest_targets(checks: Iterable[dict]) -> tuple[str, ...]:
    """Return the exact native build targets needed by the selected CTest proofs."""
    return tuple(
        sorted(
            {
                row["target"].split("@", 1)[1]
                for row in checks
                if row["kind"] == "ctest"
            }
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--audit-only",
        action="store_true",
        help="verify exact evidence and explicit gaps without claiming M4 closure",
    )
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument(
        "--list-ctest-targets",
        action="store_true",
        help="print the exact native targets required by a closed manifest",
    )
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-mpi")
    parser.add_argument("--mpi-exec", default="mpiexec")
    args = parser.parse_args(argv)

    if args.audit_only:
        data, errors = audit_manifest(args.manifest)
    else:
        data, errors = validate_manifest(args.manifest)
    if errors:
        print("M4 gate is incomplete or invalid:", file=sys.stderr)
        for error in errors:
            print(" -", error, file=sys.stderr)
        return 2

    checks = data["check"]
    if args.list_ctest_targets:
        targets = _required_ctest_targets(checks)
        if not targets:
            print("M4 gate selects no CTest build target", file=sys.stderr)
            return 2
        print("\n".join(targets))
        return 0
    print(
        "M4 gate source matrix: %s (%d executable, %d deferred)"
        % (
            (
                "AUDITED OPEN"
                if data["deferred"]
                else "AUDITED CLOSED"
            )
            if args.audit_only
            else "CLOSED",
            len(checks),
            len(data["deferred"]),
        )
    )
    if args.audit_only or args.check_only:
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
            env=_required_environment(),
        )
    if not args.python_only:
        for row in sorted(
            (row for row in checks if row["kind"] == "ctest"),
            key=lambda value: (value["target"], value["test_regex"]),
        ):
            _semantic, target = row["target"].split("@", 1)
            _run_ctest(args.build_dir, target, row["test_regex"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
