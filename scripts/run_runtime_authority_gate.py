#!/usr/bin/env python3
"""Audit and run the closed ADC-700/702/720 runtime-authority gate.

The manifest is an executable evidence ledger.  A row is accepted only when it points at a
manifest-owned, source-registered test with one exact nodeid or CTest selector.  Runtime rows
are launched with the lane's authenticated native/MPI/OpenMP environment; no prerequisite is
allowed to turn into a skip or xfail.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
import importlib.util
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
DEFAULT_MANIFEST = ROOT / "tests/gates/runtime_authority.toml"
TEST_MANIFEST = ROOT / "tests/test_manifest.toml"

EXPECTED_ISSUES = ("ADC-700", "ADC-702", "ADC-720")

# The runtime-authority gate is the final consumer of these two earlier ledgers.  It does not
# silently replace their executable proofs: their manifests must remain present, closed, and
# explicitly cover the temporal/reflux issues which this gate composes.  The MPI execution lane
# invokes each runner below after the final rows, so source validation and executable composition
# remain separate operations.
COMPOSED_GATE_SPECS = (
    (
        "tests/gates/m2_temporal_execution.toml",
        "scripts/run_m2_gate.py",
        "m2-temporal-execution",
        ("ADC-648",) + tuple("ADC-%d" % number for number in range(661, 669)),
        65,
        frozenset({"ADC-665", "ADC-666"}),
    ),
    (
        "tests/gates/m3_amr_multilayout.toml",
        "scripts/run_m3_gate.py",
        "m3-amr-multilayout",
        tuple("ADC-%d" % number for number in range(672, 679)),
        57,
        frozenset({"ADC-677"}),
    ),
)

# The names are deliberately semantic rather than implementation-file names.  Each requirement
# is nevertheless bound to one issue below, which prevents a nearby proof from silently taking
# ownership of a different ticket.
REQUIRED_POLARITIES = {
    "abi_identity": {"positive", "refusal"},
    "solve_outcome": {"positive", "refusal"},
    "common_services": {"positive", "refusal"},
    "schedule_parity": {"positive", "refusal"},
    "cell_local_ordinary": {"positive", "refusal"},
    "strict_restart": {"positive", "refusal"},
    "hot_path_allocation": {"positive", "refusal"},
    "legacy_context_barrier": {"positive", "refusal"},
    "legacy_symbol_barrier": {"positive", "refusal"},
    "legacy_fragment_barrier": {"positive", "refusal"},
    "pending_marker_barrier": {"positive", "refusal"},
    "transaction_authority": {"positive", "refusal"},
    "prepared_installation": {"positive", "refusal"},
    "lowering_refusal": {"refusal"},
    "gate_execution": {"positive", "refusal"},
}
REQUIREMENT_ISSUES = {
    "abi_identity": {"ADC-700"},
    "solve_outcome": {"ADC-700"},
    "common_services": {"ADC-702"},
    "schedule_parity": {"ADC-700", "ADC-702"},
    "cell_local_ordinary": {"ADC-720"},
    "strict_restart": {"ADC-702", "ADC-720"},
    "hot_path_allocation": {"ADC-702", "ADC-720"},
    "legacy_context_barrier": {"ADC-700"},
    "legacy_symbol_barrier": {"ADC-700"},
    "legacy_fragment_barrier": {"ADC-702"},
    "pending_marker_barrier": {"ADC-702"},
    "transaction_authority": {"ADC-702"},
    "prepared_installation": {"ADC-700"},
    "lowering_refusal": {"ADC-700"},
    "gate_execution": set(EXPECTED_ISSUES),
}
EXPECTED_BACKENDS = {"serial", "mpi", "openmp"}
EXPECTED_ALLOCATIONS = {"none", "required"}
EXPECTED_CHECK_COUNT = 58
REQUIRED_LOWERING_REFUSAL_NODEIDS = frozenset(
    {
        "tests/python/unit/codegen/test_scheduler_codegen.py::test_scratch_skip_refuses_unprepared_stale_state",
        "tests/python/unit/codegen/test_scheduler_codegen.py::test_on_end_refuses_to_lower",
    }
)
CXX_SOURCE_GLOBS = (
    "*.cpp",
    "*.cc",
    "*.cxx",
    "*.cu",
    "*.hpp",
    "*.hh",
    "*.hxx",
    "*.cuh",
    "*.inc",
)
CXX_SUFFIXES = frozenset(pattern[1:] for pattern in CXX_SOURCE_GLOBS)

PROGRAM_AUTHORITY_FILES = (
    "include/pops/runtime/program/program_abi.hpp",
    "include/pops/runtime/program/program_loader.hpp",
    "include/pops/runtime/program/owned_program_installation.hpp",
    "include/pops/runtime/program/program_persistent_value_store.hpp",
    "include/pops/runtime/program/program_execution_services.hpp",
    "include/pops/runtime/program/program_runtime_state.hpp",
    "include/pops/runtime/program/program_transaction.hpp",
)
PUBLIC_AMR_ROOT = "include/pops/runtime/program/program_execution_services_amr.hpp"
RETIRED_AUTHORITY_FILES = (
    "include/pops/runtime/program/program_service_bundle.hpp",
)
DISPATCHER_SOURCES = (
    "src/runtime/system/system.cpp",
    "src/runtime/amr/amr_system.cpp",
)
LEGACY_CONTEXT_TOKENS = (
    "ProgramContext",
    "AmrProgramContext",
    "program_context.hpp",
    "amr_program_context.hpp",
)
LEGACY_SYMBOL_TOKENS = (
    "SystemDriver",
    "SystemCoupler",
    "program_driver_",
    "seal_program_preparation_host",
)
LEGACY_PROGRAM_AMR_SYMBOL_TOKENS = (
    # These are the historical split Program ABI entry/routes/boundary spellings.  Keep this an
    # explicit closed list: native brick package ABIs have their own ``*_amr`` symbols and are not
    # part of the Program cutover.
    "pops_install_program_amr",
    "pops_register_program_provider_routes_amr",
    "pops_program_dt_bound_amr",
    "pops_install_field_boundaries_amr",
)
LEGACY_INSTALLER_TOKENS = (
    # Only split Program entry/boundary installers belong to this barrier.  The similarly named
    # external_install_* helpers are native brick-package ABI and intentionally remain out of it.
    "pops_install_program_amr",
    "pops_install_field_boundaries_amr",
)
# These are the retired C++ facade/state routes.  They are intentionally kept separate from the
# native-brick ABI list above: a native brick may expose its own ``*_amr`` symbols, but no runtime
# Program is allowed to publish a second step, hierarchy, restart, or unverified installer route.
LEGACY_PUBLIC_INSTALLER_TOKENS = (
    "install_program_step",
    "install_program_hierarchy_refresh",
    "install_program_history_remap_accepted",
    "install_program_restart_hooks",
    "install_unverified_step",
)
FORBIDDEN_PENDING_MARKERS = ("pending:checkpointed_hierarchy_cache",)
LEGACY_CHECKPOINT_MARKERS = (
    # Historical wire magics are intentionally explicit; migration tests may retain their bytes
    # under tests/, but production runtime sources may not publish or dispatch them.
    "POPSAST4",
    "POPSAND4",
    "POPSAUX1",
    "POPSHYS1",
)
ABI_VERSION_PATTERN = re.compile(r"\bkProgramInstallAbiVersion\s*=\s*5[uUlL]*\b")
LEGACY_ABI_VERSION_PATTERN = re.compile(
    r"\bkProgramInstallAbiVersion\s*=\s*(?!5[uUlL]*\b)\d+[uUlL]*\b"
)
PROGRAM_EXECUTION_PRIMARY_PATTERN = re.compile(
    r"\btemplate\s*<\s*int\s+Dim\s*>\s*"
    r"class\s+ProgramExecutionServices\s*(?:final\s*)?(?::[^{}]*)?\{"
)
PROGRAM_EXECUTION_TEMPLATE_PATTERN = re.compile(
    r"\btemplate\s*<\s*(?P<parameters>[^>]*)>\s*"
    r"(?:friend\s+)?class\s+ProgramExecutionServices\b"
)
PROGRAM_EXECUTION_SECOND_PARAMETER_PATTERN = re.compile(
    r"\bProgramExecutionServices\s*<[^>]*,"
)
DISPATCHER_PATTERN = re.compile(r"\bdispatch_cadence_step\s*\(")
DIRECT_DISPATCH_BYPASS_PATTERN = re.compile(
    r"\bprogram_?(?:\.|->)\s*step_\s*\("
)
AMR_RUNTIME_STEP_BYPASS_PATTERN = re.compile(
    r"\b(?:AmrRuntime(?:\s*<[^>{};\n]+>)?\s*::\s*step|"
    r"(?:amr_)?runtime_?\s*(?:\.|->)\s*step)\s*\("
)
CACHE_NODE_ID_ONLY_PATTERN = re.compile(
    r"(?:\bstd::map\s*<\s*int\s*,\s*CacheSlot\b|"
    r"\bstd::unordered_map\s*<\s*int\s*,\s*CacheSlot\b|"
    r"\b(?:node_ids|declare_slot|cache_nodes|program_cache_nodes)\s*\(|"
    r"\bcache_(?:should_update|store_scratch|restore_scratch|accumulate_dt|effective_dt)\s*"
    r"\(\s*(?:int|std::int64_t)\s+(?:node_id|value_id)\b)"
)
DIRECT_PROVIDER_CONSTRUCTOR_PATTERN = re.compile(
    r"\bProgramExecutionServices\s*\(\s*(?:(?:::)?pops::)?"
    r"(?:System|AmrSystem)\s*<"
)
DIRECT_PROVIDER_FACTORY_PATTERN = re.compile(
    r"\bmake_program_execution_provider\s*\(\s*(?:(?:::)?pops::)?"
    r"(?:System|AmrSystem)\s*<"
)
PREPARATION_PROVIDER_FACTORY_PATTERN = re.compile(
    r"\bmake_program_execution_provider\s*(?:<[^;{}()]*>)?\s*\(\s*"
    r"const\s+ProgramPreparationHostRef\s*&"
)
PREPARATION_PROVIDER_CALL_PATTERN = re.compile(
    r"\bmake_program_execution_provider\s*(?:<[^;{}()]*>)?\s*\(\s*"
    r"host\s*->\s*preparation\b"
)
DIRECT_PROVIDER_CALL_PATTERN = re.compile(
    r"\bmake_program_execution_provider\s*(?:<[^;{}()]*>)?\s*\(\s*"
    r"(?!host\s*->\s*preparation\b)[^)]*\)"
)
# Compatibility spellings removed by the final transaction/resource-plan cutover.  These are
# intentionally scoped below to the owning source/class; names such as ``freeze`` and ``read`` are
# valid in unrelated authoring/IO code and must not become repository-wide false positives.
RETIRED_TRANSACTION_METHOD_NAMES = (
    "add_participant",
    "try_add_participant",
    "freeze",
    "frozen",
    "frozen_effect_capacity",
    "begin_step",
)
RETIRED_TRANSACTION_ENUM_NAMES = {
    "ProgramTransactionPhase": (
        "Unbound",
        "Snapshot",
        "Candidate",
        "SolveGuardEffectPrepare",
        "HiddenPublish",
        "CompensableEffects",
        "AtomicSeal",
        "IrreversibleFinalize",
        "Accepted",
        "RolledBack",
        "FailStop",
    ),
    "ProgramTransactionFailure": (
        "None",
        "NotBound",
        "AlreadyActive",
        "Registration",
        "Budget",
        "Snapshot",
        "Candidate",
        "Solve",
        "Guard",
        "EffectPrepare",
        "HiddenPublish",
        "Compensation",
        "AtomicSeal",
        "Finalize",
        "FailStop",
    ),
}
RETIRED_PLAN_ALIAS_NAMES = frozenset(
    {
        "PersistentValueKey",
        "PersistentValue",
        "PersistentPlan",
        "ResourcePlan",
        "ResourcePlanEntry",
        "ResourceKey",
        "lower_resource_plan",
        "lower_persistent_plan",
        "resource_plan_lowering",
        "persistent_resource_plan",
    }
)
TRANSACTION_ALIAS_TARGET_PATTERN = re.compile(
    r"\busing\s+(?P<alias>[A-Za-z_]\w*)\s*=\s*"
    r"(?P<target>(?:[A-Za-z_]\w*\s*::\s*)*ProgramTransaction"
    r"(?:Phase|Failure|Fault|Budget|Consensus|ParticipantOps|Registry|Transaction)"
    r"(?:\s*::\s*[A-Za-z_]\w*)?)\b"
)
TRANSACTION_TYPEDEF_ALIAS_PATTERN = re.compile(
    r"\btypedef\s+(?P<target>(?:[A-Za-z_]\w*\s*::\s*)*ProgramTransaction"
    r"(?:Phase|Failure|Fault|Budget|Consensus|ParticipantOps|Registry|Transaction)"
    r"(?:\s*::\s*[A-Za-z_]\w*)?)\s+(?P<alias>[A-Za-z_]\w*)\s*;"
)
INSTALLATION_ALIAS_TARGET_PATTERN = re.compile(
    r"\busing\s+(?P<alias>[A-Za-z_]\w*)\s*=\s*"
    r"(?P<target>(?:[A-Za-z_]\w*\s*::\s*)*"
    r"(?:ProgramResourcePlan(?:Entry)?|ProgramPersistentValueKey|"
    r"ProgramInstallationTables(?:\s*::\s*[A-Za-z_]\w*)?))\b"
)
INSTALLATION_TYPEDEF_ALIAS_PATTERN = re.compile(
    r"\btypedef\s+(?P<target>(?:[A-Za-z_]\w*\s*::\s*)*"
    r"(?:ProgramResourcePlan(?:Entry)?|ProgramPersistentValueKey|"
    r"ProgramInstallationTables(?:\s*::\s*[A-Za-z_]\w*)?))\s+(?P<alias>[A-Za-z_]\w*)\s*;"
)
LEGACY_RESOURCE_TUPLE_USING_PATTERN = re.compile(
    r"\busing\s+(?P<alias>[A-Za-z_]\w*)\s*=\s*"
    r"(?:std\s*::\s*)?tuple\s*<",
    re.IGNORECASE,
)
LEGACY_RESOURCE_TUPLE_TYPEDEF_PATTERN = re.compile(
    r"\btypedef\s+(?:std\s*::\s*)?tuple\s*<[^;]*>\s*"
    r"(?P<alias>[A-Za-z_]\w*)\s*;",
    re.IGNORECASE,
)
PYTHON_PLAN_ALIAS_ASSIGNMENT_PATTERN = re.compile(
    r"(?m)^\s*(?P<alias>[A-Za-z_]\w*)\s*=\s*"
    r"(?P<target>(?:ProgramResourcePlan(?:Entry)?|ProgramPersistentValueKey|"
    r"lower_program_resource_plan))\s*$"
)
PYTHON_PLAN_TUPLE_ALIAS_PATTERN = re.compile(
    r"(?m)^\s*(?P<alias>[A-Za-z_]\w*(?:resource|persistent|plan|value|key)[A-Za-z_0-9]*)"
    r"\s*(?::[^=]+)?=\s*(?:tuple|Tuple)\b"
)
CANONICAL_CHECKPOINT_MAGIC_TOKENS = {
    "include/pops/runtime/program/amr_program_checkpoint.hpp":
    "'P', 'O', 'P', 'S', 'A', 'N', 'D', '5'",
    "include/pops/runtime/system/auxiliary_checkpoint.hpp":
    "'P', 'O', 'P', 'S', 'A', 'U', 'X', '2'",
    "include/pops/runtime/amr/persistent_tagging_state.hpp":
    "'P', 'O', 'P', 'S', 'H', 'Y', 'S', '2'",
}
CANONICAL_CHECKPOINT_VERSION_TOKENS = {
    "python/pops/_generated_release_contract.py": {
        "UNIFORM_CHECKPOINT_PAYLOAD_VERSION": 9,
        "AMR_CHECKPOINT_PAYLOAD_VERSION": 12,
    },
}
REQUIRED_AUTHORITY_PATTERNS = {
    "include/pops/runtime/program/owned_program_installation.hpp": {
        "PreparedProgramInstallation": re.compile(r"\bclass\s+PreparedProgramInstallation\b"),
    },
    "include/pops/runtime/program/program_transaction.hpp": {
        "ProgramTransaction": re.compile(r"\bclass\s+ProgramTransaction\b"),
        "AcceptedReadLease": re.compile(r"\bclass\s+AcceptedReadLease\b"),
        "atomic_seal": re.compile(r"\batomic_seal\s*\("),
        "acquire_read": re.compile(r"\bacquire_read\s*\("),
    },
    "include/pops/runtime/program/program_runtime_state.hpp": {
        "PreparedArtifactPublication": re.compile(r"\bclass\s+PreparedArtifactPublication\b"),
        "install_prepared_artifact": re.compile(r"\binstall_prepared_artifact\s*\("),
        "atomic publication": re.compile(
            r"\bpublish_prepared_artifact_\s*\([^)]*\)\s*noexcept"
        ),
    },
    "include/pops/runtime/program/program_execution_services.hpp": {
        "SolveOutcome": re.compile(r"\bSolveOutcome\b"),
    },
}
ALLOCATION_PROOF_SOURCE = "tests/cpp/unit/runtime/test_program_execution_services_contract.cpp"
ALLOCATION_PROOF_CASE = "ProgramExecutionServicesContract.GeneratedScratchIsPersistentExactAndNonAliasing"
ALLOCATION_PROOF_ROW = {
    "issue": "ADC-702",
    "requirement": "hot_path_allocation",
    "polarity": "positive",
    "kind": "ctest",
    "target": "hot_path_allocation@test_program_execution_services_contract",
    "allocation": "required",
    "test_regex": "^ProgramExecutionServicesContract\\.GeneratedScratchIsPersistentExactAndNonAliasing$",
}
GTEST_DECLARATION = re.compile(
    r"\bTEST(?:_F)?\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\)"
)
CPP_RAW_STRING_START = re.compile(r'(?:u8|u|U|L)?R"([^\s()\\]{0,16})\(')
FULL_NODEID = re.compile(r"^[^:\n]+::[A-Za-z_][A-Za-z0-9_]*$")
MOCK_FIXTURES = {"monkeypatch", "mocker", "mock", "patch"}
FORBIDDEN_CALLS = {
    "pytest.skip",
    "pytest.xfail",
    "pytest.importorskip",
    "unittest.mock.patch",
    "mock.patch",
    "require_native_or_skip",
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
        markers.extend("fixture:%s" % name for name in sorted(fixtures & MOCK_FIXTURES))
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _dotted_name(child.func)
            if name in FORBIDDEN_CALLS or name.endswith(
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
        elif isinstance(child, (ast.Assign, ast.AnnAssign)):
            targets = child.targets if isinstance(child, ast.Assign) else (child.target,)
            if any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in targets):
                value = ast.unparse(child.value)
                if ".skip" in value or ".xfail" in value:
                    markers.append("pytestmark:%s" % value)
    return markers


def _has_authenticated_mpi_guard(module: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "tests.python.support.requirements"
        and any(alias.name == "require_mpi_or_skip" for alias in node.names)
        for node in ast.walk(module)
    )


def _python_suites() -> tuple[dict, ...]:
    data = tomllib.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    return tuple(data.get("python", {}).get("suite", ()))


def _python_suite_owns(relative: str) -> bool:
    path = Path(relative)
    return any(
        path == Path(str(suite.get("path", "")))
        or Path(str(suite.get("path", ""))) in path.parents
        for suite in _python_suites()
    )


def _python_mpi_entrypoints() -> dict[str, int]:
    entries: dict[str, int] = {}
    for suite in _python_suites():
        for row in suite.get("mpi_entrypoints", ()):
            if not isinstance(row, dict):
                raise ValueError("invalid Python MPI entrypoint %r" % row)
            path = row.get("path")
            nproc = row.get("nproc")
            if (
                not isinstance(path, str)
                or not path
                or isinstance(nproc, bool)
                or not isinstance(nproc, int)
                or nproc < 1
            ):
                raise ValueError("invalid Python MPI entrypoint %r" % row)
            if path in entries:
                raise ValueError("duplicate Python MPI entrypoint %s" % path)
            entries[path] = nproc
    return entries


def _python_mpi_orchestrators() -> set[str]:
    """Return the manifest-owned Python drivers which create their own MPI worlds.

    These files are intentionally distinct from ``mpi_entrypoints``: an orchestrator must run as
    one serial pytest process so that it can terminate a capture world before starting a restart
    world with a different rank count.  Treating it as an ordinary MPI entrypoint would keep one
    communicator alive and would erase the topology-change proof.
    """
    orchestrators: set[str] = set()
    for suite in _python_suites():
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


def _cpp_suites() -> dict[str, dict]:
    data = tomllib.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    return {str(row["name"]): row for row in data.get("cpp", {}).get("suite", ())}


def _cpp_masked(source: str, *, mask_literals: bool) -> str:
    """Mask C++ comments and optionally literals while preserving source line numbers."""
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
        raw = CPP_RAW_STRING_START.match(source, index)
        if raw is not None:
            terminator = ")" + raw.group(1) + '"'
            end = source.find(terminator, raw.end())
            end = size if end < 0 else end + len(terminator)
            if mask_literals:
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
            if mask_literals:
                mask(index, end)
            index = end
            continue
        index += 1
    return "".join(code)


def _cpp_code_only(source: str) -> str:
    """Mask comments and literals while preserving newlines for C++ syntax scans."""
    return _cpp_masked(source, mask_literals=True)


def _cpp_without_comments(source: str) -> str:
    """Mask comments but retain literals used for authenticated exported symbol names."""
    return _cpp_masked(source, mask_literals=False)


def _python_semantic_tokens(source: str, path: Path) -> tuple[str, ...]:
    """Return identifiers and string constants from a Python source module, excluding comments."""
    tree = ast.parse(source, filename=str(path))
    docstring_nodes: set[int] = set()
    for owner in ast.walk(tree):
        if not isinstance(owner, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if owner.body and isinstance(owner.body[0], ast.Expr):
            value = owner.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                docstring_nodes.add(id(value))
    tokens: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = getattr(node, "id", None) or getattr(node, "attr", None) or getattr(node, "name", None)
            if isinstance(name, str):
                tokens.append(name)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstring_nodes
        ):
            tokens.append(node.value)
    return tuple(tokens)


def _registered_gtest_cases(source: str) -> set[str]:
    return {
        "%s.%s" % declaration
        for declaration in GTEST_DECLARATION.findall(_cpp_code_only(source))
    }


def _validate_execution_service_architecture(errors: list[str]) -> None:
    """Enforce the one generic execution-service root and its canonical cadence entry point."""
    source_paths = tuple(path for path in _authority_sources() if path.is_file())
    source_texts = {
        path: path.read_text(encoding="utf-8")
        for path in source_paths
    }
    source_code = {path: _cpp_code_only(text) for path, text in source_texts.items()}

    amr_directory = ROOT / "include/pops/runtime/program"
    fragment_paths = set(amr_directory.rglob("*.inc"))
    if fragment_paths:
        errors.append(
            "runtime authority forbids AMR runtime fragments (.inc): %s"
            % sorted(fragment.name for fragment in fragment_paths)
        )
    amr_root = ROOT / PUBLIC_AMR_ROOT
    if amr_root.is_file():
        errors.append(
            "%s is a forbidden public AMR execution-services root; use the generic "
            "ProgramExecutionServices<Dim> authority" % PUBLIC_AMR_ROOT
        )

    primary_locations = [
        path
        for path, code in source_code.items()
        if PROGRAM_EXECUTION_PRIMARY_PATTERN.search(code) is not None
    ]
    expected_root = ROOT / "include/pops/runtime/program/program_execution_services.hpp"
    if len(primary_locations) != 1:
        errors.append(
            "runtime authority requires exactly one template<int Dim> class ProgramExecutionServices"
            " (found %d)" % len(primary_locations)
        )
    elif primary_locations[0] != expected_root:
        errors.append(
            "the generic ProgramExecutionServices primary must live in %s"
            % expected_root.relative_to(ROOT)
        )

    for path, code in source_code.items():
        if "AmrProgramExecutionAdapter" in source_texts[path]:
            errors.append(
                "%s retains forbidden AmrProgramExecutionAdapter" % _display_source_path(path)
            )
        for declaration in PROGRAM_EXECUTION_TEMPLATE_PATTERN.finditer(code):
            parameters = declaration.group("parameters")
            if "," in parameters:
                errors.append(
                    "%s declares ProgramExecutionServices with a second template parameter"
                    % _display_source_path(path)
                )
        if PROGRAM_EXECUTION_SECOND_PARAMETER_PATTERN.search(code) is not None:
            errors.append(
                "%s uses a second ProgramExecutionServices template parameter or specialization"
                % _display_source_path(path)
            )

    for relative in RETIRED_AUTHORITY_FILES:
        retired = ROOT / relative
        if retired.is_file():
            errors.append("retired Program authority remains present: %s" % relative)

    runtime_state = ROOT / "include/pops/runtime/program/program_runtime_state.hpp"
    runtime_state_code = source_code.get(runtime_state)
    if runtime_state_code is None:
        errors.append("runtime authority is missing ProgramRuntimeState source")
    elif len(DISPATCHER_PATTERN.findall(runtime_state_code)) != 1:
        errors.append(
            "ProgramRuntimeState must define exactly one canonical dispatch_cadence_step dispatcher"
        )

    for relative in DISPATCHER_SOURCES:
        path = ROOT / relative
        if not path.is_file():
            errors.append("runtime authority dispatcher source is missing: %s" % relative)
            continue
        code = _cpp_code_only(path.read_text(encoding="utf-8"))
        if DISPATCHER_PATTERN.search(code) is None:
            errors.append("%s does not enter the canonical dispatch_cadence_step" % relative)

    bypass_paths = {
        ROOT / relative for relative in (*DISPATCHER_SOURCES, "src/runtime/system/system_io.cpp")
    }
    for directory in (ROOT / "src/runtime", ROOT / "include/pops/runtime"):
        if directory.is_dir():
            bypass_paths.update(
                path
                for suffix in CXX_SOURCE_GLOBS
                for path in directory.rglob(suffix)
            )
    for path in sorted(bypass_paths):
        if not path.is_file():
            continue
        code = _cpp_code_only(path.read_text(encoding="utf-8"))
        if DIRECT_DISPATCH_BYPASS_PATTERN.search(code) is not None:
            errors.append(
                "%s bypasses ProgramRuntimeState::dispatch_cadence_step"
                % _display_source_path(path)
            )


def _validate_detached_provider_architecture(errors: list[str]) -> None:
    """Require the DSO provider to be constructed only from the sealed host preparation image.

    A facade pointer is a live mutable authority.  Allowing either a public
    ``ProgramExecutionServices(System*/AmrSystem*)`` constructor or a matching factory overload
    lets a candidate bypass the prepared image and reintroduces publication-time mutation.  The
    source gate therefore checks the C++ declarations/definitions and generated Python strings
    independently; comments and Python docstrings are excluded by the existing semantic views.
    """
    sources = tuple(path for path in _production_sources() if path.is_file())
    preparation_factory_count = 0
    preparation_call_paths: set[Path] = set()
    for path in sources:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(
                "cannot read provider-construction authority source %s: %s"
                % (_display_source_path(path), exc)
            )
            continue

        display = _display_source_path(path)
        if path.suffix in CXX_SUFFIXES:
            code = _cpp_code_only(text)
            direct_constructor = DIRECT_PROVIDER_CONSTRUCTOR_PATTERN.search(code)
            if direct_constructor is not None:
                errors.append(
                    "%s exposes a direct ProgramExecutionServices(System*/AmrSystem*) "
                    "constructor; DSO providers must use ProgramPreparationHostRef"
                    % display
                )
            direct_factory = DIRECT_PROVIDER_FACTORY_PATTERN.search(code)
            if direct_factory is not None:
                errors.append(
                    "%s exposes a direct make_program_execution_provider(System*/AmrSystem*) "
                    "factory; DSO providers must use ProgramPreparationHostRef"
                    % display
                )
            preparation_factory_count += len(
                PREPARATION_PROVIDER_FACTORY_PATTERN.findall(code)
            )
            if PREPARATION_PROVIDER_CALL_PATTERN.search(code) is not None:
                preparation_call_paths.add(path)
            continue
        if path.suffix != ".py":
            continue

        try:
            semantic_tokens = _python_semantic_tokens(text, path)
        except (OSError, SyntaxError) as exc:
            errors.append(
                "cannot parse provider-construction authority source %s: %s"
                % (display, exc)
            )
            continue
        for token in semantic_tokens:
            if DIRECT_PROVIDER_CALL_PATTERN.search(token) is not None:
                errors.append(
                    "%s emits a direct make_program_execution_provider facade call; "
                    "generated DSO code must pass host->preparation" % display
                )
            if PREPARATION_PROVIDER_CALL_PATTERN.search(token) is not None:
                preparation_call_paths.add(path)

    if preparation_factory_count != 1:
        errors.append(
            "runtime authority requires exactly one preparation-image provider factory "
            "(found %d)" % preparation_factory_count
        )
    for relative in (
        "python/pops/codegen/program_codegen.py",
        "python/pops/codegen/program_emit_amr.py",
    ):
        path = ROOT / relative
        if not path.is_file() or path not in preparation_call_paths:
            errors.append(
                "%s must construct its DSO provider from host->preparation"
                % relative
            )


def _validate_retired_compatibility_aliases(errors: list[str]) -> None:
    """Reject compatibility aliases removed by the final transaction/plan cutover.

    This is intentionally a source witness, not a broad identifier blacklist.  ``freeze`` and
    ``read`` remain valid in ordinary Python/IO code; only declarations in the transaction
    registry, transaction enum spellings, resource-plan type aliases, and the old numeric slot
    fallback are prohibited here.
    """

    transaction_relative = "include/pops/runtime/program/program_transaction.hpp"
    transaction_path = ROOT / transaction_relative
    if transaction_path.is_file():
        try:
            code = _cpp_code_only(transaction_path.read_text(encoding="utf-8"))
        except OSError as exc:
            errors.append("cannot read compatibility-barrier source %s: %s" % (transaction_relative, exc))
            code = ""
        if code:
            retired_methods = [
                name
                for name in RETIRED_TRANSACTION_METHOD_NAMES
                if re.search(r"\b%s\s*\(" % re.escape(name), code)
            ]
            if retired_methods:
                errors.append(
                    "%s retains retired transaction compatibility methods %s"
                    % (transaction_relative, retired_methods)
                )

            retired_enum_spellings: list[str] = []
            for enum_name, members in RETIRED_TRANSACTION_ENUM_NAMES.items():
                for member in members:
                    if re.search(
                        r"\b%s\s*::\s*%s\b" % (re.escape(enum_name), re.escape(member)),
                        code,
                    ):
                        retired_enum_spellings.append("%s::%s" % (enum_name, member))
            if retired_enum_spellings:
                errors.append(
                    "%s retains non-canonical transaction enum spellings %s"
                    % (transaction_relative, retired_enum_spellings)
                )

            using_aliases = [
                "%s=%s" % (match.group("alias"), " ".join(match.group("target").split()))
                for match in TRANSACTION_ALIAS_TARGET_PATTERN.finditer(code)
            ]
            using_aliases.extend(
                "%s=%s" % (match.group("alias"), " ".join(match.group("target").split()))
                for match in TRANSACTION_TYPEDEF_ALIAS_PATTERN.finditer(code)
            )
            if using_aliases:
                errors.append(
                    "%s retains retired transaction type aliases %s"
                    % (transaction_relative, using_aliases)
                )

            # ``read``/``provisional_read`` are legitimate names on AcceptedReadLease and on
            # private erased accessors.  The removed aliases were public methods of the registry;
            # inspect only that class's public section to avoid rejecting those canonical helpers.
            registry = re.search(
                r"\bclass\s+ProgramTransactionRegistry\s+final\s*\{(?P<body>.*?)(?=\n\s*private\s*:)",
                code,
                re.DOTALL,
            )
            if registry is not None:
                registry_aliases = [
                    name
                    for name in ("read", "provisional_read")
                    if re.search(r"\b%s\s*\(" % name, registry.group("body"))
                ]
                if registry_aliases:
                    errors.append(
                        "%s retains retired registry read aliases %s"
                        % (transaction_relative, registry_aliases)
                    )

    installation_relative = "include/pops/runtime/program/owned_program_installation.hpp"
    installation_path = ROOT / installation_relative
    if installation_path.is_file():
        try:
            code = _cpp_code_only(installation_path.read_text(encoding="utf-8"))
        except OSError as exc:
            errors.append("cannot read compatibility-barrier source %s: %s" % (installation_relative, exc))
            code = ""
        if code:
            aliases = [
                "%s=%s" % (match.group("alias"), " ".join(match.group("target").split()))
                for match in INSTALLATION_ALIAS_TARGET_PATTERN.finditer(code)
            ]
            aliases.extend(
                "%s=%s" % (match.group("alias"), " ".join(match.group("target").split()))
                for match in INSTALLATION_TYPEDEF_ALIAS_PATTERN.finditer(code)
            )
            tuple_aliases = [
                match.group("alias")
                for match in LEGACY_RESOURCE_TUPLE_USING_PATTERN.finditer(code)
            ]
            tuple_aliases.extend(
                match.group("alias")
                for match in LEGACY_RESOURCE_TUPLE_TYPEDEF_PATTERN.finditer(code)
            )
            if aliases:
                errors.append(
                    "%s retains retired resource-plan type aliases %s"
                    % (installation_relative, aliases)
                )
            if tuple_aliases:
                errors.append(
                    "%s retains a legacy resource tuple alias %s"
                    % (installation_relative, tuple_aliases)
                )

    plan_relative = "python/pops/codegen/program_persistent_plan.py"
    plan_path = ROOT / plan_relative
    if not plan_path.is_file():
        return
    try:
        plan_source = plan_path.read_text(encoding="utf-8")
        plan_tree = ast.parse(plan_source, filename=str(plan_path))
    except (OSError, SyntaxError) as exc:
        errors.append("cannot parse compatibility-barrier source %s: %s" % (plan_relative, exc))
        return

    plan_aliases: list[str] = []
    tuple_aliases: list[str] = []
    numeric_fallbacks: list[str] = []

    def assignment_targets(node: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        return tuple(target.id for target in targets if isinstance(target, ast.Name))

    canonical_plan_symbols = {
        "ProgramPersistentValueKey",
        "ProgramResourcePlan",
        "ProgramResourcePlanEntry",
        "lower_program_resource_plan",
    }
    for node in ast.walk(plan_tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = assignment_targets(node)
        value = node.value
        if value is None:
            continue
        target_symbol = _dotted_name(value)
        for target in targets:
            if target in RETIRED_PLAN_ALIAS_NAMES and target_symbol in canonical_plan_symbols:
                plan_aliases.append("%s=%s" % (target, target_symbol))
            rendered = ast.unparse(value)
            if (
                target in RETIRED_PLAN_ALIAS_NAMES
                or re.search(r"(?:resource|persistent|plan|value|key).*tuple", target, re.I)
            ) and re.fullmatch(r"(?:typing\.)?Tuple(?:\[.*\])?|tuple(?:\[.*\])?", rendered):
                tuple_aliases.append(target)

    persistent_slot = next(
        (
            node
            for node in ast.walk(plan_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "persistent_slot"
        ),
        None,
    )
    if persistent_slot is not None:
        for node in ast.walk(persistent_slot):
            if not isinstance(node, ast.Call):
                continue
            function_name = _dotted_name(node.func)
            if function_name == "int" and node.args:
                argument = node.args[0]
                if (
                    isinstance(argument, ast.Attribute)
                    and argument.attr == "id"
                ) or (
                    isinstance(argument, ast.Call)
                    and _dotted_name(argument.func) == "getattr"
                    and len(argument.args) >= 2
                    and isinstance(argument.args[1], ast.Constant)
                    and argument.args[1].value == "id"
                ):
                    numeric_fallbacks.append("int(value.id)")
            if "_by_value_id" in function_name:
                numeric_fallbacks.append(function_name)
            if function_name.endswith("slot_for_value") and any(
                isinstance(argument, ast.Attribute) and argument.attr == "id"
                for argument in node.args
            ):
                numeric_fallbacks.append("slot_for_value(value.id)")

    if plan_aliases:
        errors.append("%s retains retired lowering aliases %s" % (plan_relative, plan_aliases))
    if tuple_aliases:
        errors.append("%s retains a legacy persistent tuple alias %s" % (plan_relative, tuple_aliases))
    if numeric_fallbacks:
        errors.append(
            "%s retains a numeric persistent_slot fallback %s"
            % (plan_relative, numeric_fallbacks)
        )


def _registered_ctest_cases(target: str, suite: dict) -> set[str]:
    mpi_counts = tuple(suite.get("mpi_nproc", ())) + tuple(suite.get("mpi_variants", ()))
    is_no_discover_mpi = bool(suite.get("mpi_nproc")) or bool(suite.get("mpi_rank_parity"))
    cases: set[str] = set()
    if not is_no_discover_mpi:
        for relative in suite.get("sources", ()):
            source = ROOT / relative
            if source.is_file():
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
    selector: object, target: str, suite: dict, where: str, errors: list[str]
) -> str | None:
    if not isinstance(selector, str) or not selector:
        errors.append("%s CTest row requires a non-empty test_regex" % where)
        return None
    cases = _registered_ctest_cases(target, suite)
    exact = {"^%s$" % re.escape(case) for case in cases}
    if selector not in exact:
        errors.append(
            "%s CTest selector %r is not one exact source-registered case for target %r"
            % (where, selector, target)
        )
        return None
    return selector[1:-1].replace("\\.", ".")


def _validate_required_lowering_refusal_rows(
    checks: Iterable[dict], errors: list[str]
) -> None:
    """Pin the two unconditional scheduler refusals that protect the lowering boundary."""
    rows = [
        row
        for row in checks
        if isinstance(row, dict) and row.get("requirement") == "lowering_refusal"
    ]
    nodeids = {row.get("nodeid") for row in rows}
    expected = set(REQUIRED_LOWERING_REFUSAL_NODEIDS)
    if nodeids != expected:
        errors.append(
            "lowering_refusal must pin exactly the source-registered nodeids %s (found %s)"
            % (sorted(expected), sorted(nodeids, key=str))
        )
    required_fields = {
        "polarity": "refusal",
        "kind": "pytest",
        "target": "lowering_refusal",
        "backend": "serial",
        "allocation": "none",
    }
    for index, row in enumerate(rows, 1):
        mismatches = [
            "%s=%r" % (field, row.get(field))
            for field, value in required_fields.items()
            if row.get(field) != value
        ]
        if mismatches:
            errors.append(
                "lowering_refusal[%d] has non-canonical proof fields: %s"
                % (index, ", ".join(mismatches))
            )


def _registered_gtest_case_body(source: str, case: str) -> str | None:
    """Return one exact source-registered GTest body, preserving line-oriented diagnostics."""
    matches = list(GTEST_DECLARATION.finditer(_cpp_code_only(source)))
    for index, match in enumerate(matches):
        selected = "%s.%s" % match.groups()
        if selected != case:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        return source[match.start() : end]
    return None


def _gtest_case_has_skip(suite: dict, case: str) -> bool:
    """Return whether the selected GTest body itself is disabled or skips.

    A suite may contain a deliberately MPI-only case with ``GTEST_SKIP`` while its ordinary
    cases remain mandatory.  Scanning the whole translation unit would therefore reject a valid
    exact selector; inspect only the source-registered case selected by the manifest row.
    """
    for relative in suite.get("sources", ()):
        source = ROOT / relative
        if not source.is_file():
            continue
        text = source.read_text(encoding="utf-8")
        body = _registered_gtest_case_body(text, case)
        if body is not None:
            matches = list(GTEST_DECLARATION.finditer(_cpp_code_only(text)))
            selected_match = next(
                match for match in matches if "%s.%s" % match.groups() == case
            )
            return (
                "GTEST_SKIP" in body
                or selected_match.group(1).startswith("DISABLED_")
                or selected_match.group(2).startswith("DISABLED_")
            )
    return False


def _validate_python_nodeid(
    nodeid: object,
    where: str,
    errors: list[str],
    *,
    backend: str,
) -> str | None:
    if not isinstance(nodeid, str) or FULL_NODEID.fullmatch(nodeid) is None:
        errors.append("%s must contain one exact file::test nodeid" % where)
        return None
    relative, function_name = nodeid.split("::")
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        errors.append("%s escapes the repository through its nodeid" % where)
        return None
    test_path = ROOT / relative
    if not test_path.is_file():
        errors.append("%s references missing test file %s" % (where, relative))
        return None
    if not _python_suite_owns(relative):
        errors.append("%s is not owned by tests/test_manifest.toml" % relative)
    try:
        tree = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
    except (OSError, SyntaxError) as exc:
        errors.append("%s cannot parse %s: %s" % (where, relative, exc))
        return None
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    function = functions.get(function_name)
    if function is None:
        errors.append("%s references missing test function %s" % (where, nodeid))
        return None
    markers = _forbidden_python_markers(function)
    module_nodes = [
        node
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    module = ast.Module(body=module_nodes, type_ignores=[])
    module_markers = _forbidden_python_markers(module)
    if "require_mpi_or_skip" in markers:
        if backend != "mpi" or not _has_authenticated_mpi_guard(module):
            markers.append("unauthenticated require_mpi_or_skip")
        markers = [marker for marker in markers if marker != "require_mpi_or_skip"]
    if "require_mpi_or_skip" in module_markers:
        if backend != "mpi" or not _has_authenticated_mpi_guard(module):
            markers.append("unauthenticated require_mpi_or_skip")
        module_markers = [
            marker for marker in module_markers if marker != "require_mpi_or_skip"
        ]
    markers.extend(module_markers)
    if markers:
        errors.append(
            "%s is not an unconditional mandatory proof; found %s"
            % (nodeid, sorted(set(markers)))
        )
    return relative


def _authority_sources() -> tuple[Path, ...]:
    directory = ROOT / "include/pops/runtime/program"
    # The ownership list names the required ABI/service surfaces, while the barrier covers every
    # source-like file below this boundary.  A retired context, symbol, pending marker, public AMR
    # root, or include fragment must not hide in a detail subdirectory.
    return tuple(
        sorted(
            path
            for suffix in ("*.hpp", "*.inc")
            for path in directory.rglob(suffix)
            if path.is_file()
        )
    )


def _production_sources() -> tuple[Path, ...]:
    """Return all source files in the runtime authority's production ownership roots."""
    roots = (
        ROOT / "include/pops/runtime",
        ROOT / "src/runtime",
        ROOT / "python/pops",
    )
    suffixes = (*CXX_SOURCE_GLOBS, "*.py")
    paths: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for suffix in suffixes:
            paths.update(path for path in root.rglob(suffix) if path.is_file())
    return tuple(sorted(paths))


def _display_source_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _scan_production_barriers(
    errors: list[str], source_paths: Iterable[Path], *, reject_any_pending: bool = False
) -> None:
    """Scan production source roots with syntax-aware views and explicit retired spellings."""
    for path in sorted(set(source_paths)):
        if not path.is_file():
            continue
        display = _display_source_path(path)
        lower_stem = path.stem.lower()
        if path.suffix == ".inc" and path.is_relative_to(
            ROOT / "include/pops/runtime/program"
        ):
            errors.append("runtime authority forbids AMR runtime fragments (.inc): %s" % display)
        if lower_stem in {"program_context", "amr_program_context"} or lower_stem.startswith(
            ("program_context_", "amr_program_context_")
        ):
            errors.append("retired ProgramContext source remains present: %s" % display)

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append("cannot read production authority source %s: %s" % (display, exc))
            continue

        if path.suffix in CXX_SUFFIXES:
            code = _cpp_code_only(text)
            semantic = _cpp_without_comments(text)
        elif path.suffix == ".py":
            try:
                tokens = _python_semantic_tokens(text, path)
            except (OSError, SyntaxError) as exc:
                errors.append("cannot parse production authority source %s: %s" % (display, exc))
                continue
            code = "\n".join(tokens)
            semantic = code
        else:
            continue

        contexts = [
            token
            for token in LEGACY_CONTEXT_TOKENS
            if re.search(r"\b%s\b" % re.escape(token), semantic)
        ]
        symbols = [
            token
            for token in (*LEGACY_SYMBOL_TOKENS, *LEGACY_PROGRAM_AMR_SYMBOL_TOKENS)
            if re.search(r"\b%s\b" % re.escape(token), semantic)
        ]
        public_installers = [
            token
            for token in LEGACY_PUBLIC_INSTALLER_TOKENS
            if re.search(r"\b%s\b" % re.escape(token), semantic)
        ]
        installers = [
            token
            for token in LEGACY_INSTALLER_TOKENS
            if re.search(r"\b%s\b" % re.escape(token), semantic)
        ]
        pending = [marker for marker in FORBIDDEN_PENDING_MARKERS if marker in semantic]
        if reject_any_pending and "pending:" in semantic and not pending:
            pending.append("pending:<any>")
        checkpoint = [marker for marker in LEGACY_CHECKPOINT_MARKERS if marker in semantic]
        if contexts:
            errors.append("%s retains retired context names %s" % (display, contexts))
        if symbols:
            errors.append("%s retains retired runtime symbols %s" % (display, symbols))
        if public_installers:
            errors.append(
                "%s retains retired public Program installer routes %s"
                % (display, public_installers)
            )
        if installers:
            errors.append("%s retains retired split installers %s" % (display, installers))
        if pending:
            errors.append("%s contains forbidden pending marker(s) %s" % (display, pending))
        if checkpoint:
            errors.append("%s retains legacy checkpoint/magic marker(s) %s" % (display, checkpoint))
        if LEGACY_ABI_VERSION_PATTERN.search(code) is not None:
            errors.append("%s retains a legacy Program install ABI version" % display)
        if AMR_RUNTIME_STEP_BYPASS_PATTERN.search(code) is not None:
            errors.append("%s bypasses the Program authority through AmrRuntime::step" % display)
        if CACHE_NODE_ID_ONLY_PATTERN.search(code) is not None:
            errors.append("%s retains a node-id-only scheduler cache" % display)


def _validate_required_authority_contract(errors: list[str]) -> None:
    """Require the concrete immutable-installation/transaction/publication protocol surfaces."""
    for relative, patterns in REQUIRED_AUTHORITY_PATTERNS.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append("runtime authority is missing required contract source: %s" % relative)
            continue
        try:
            code = _cpp_code_only(path.read_text(encoding="utf-8"))
        except OSError as exc:
            errors.append("cannot read required contract source %s: %s" % (relative, exc))
            continue
        missing = [name for name, pattern in patterns.items() if pattern.search(code) is None]
        if missing:
            errors.append("%s is missing required authority protocol(s): %s" % (relative, missing))

    for relative, signature in CANONICAL_CHECKPOINT_MAGIC_TOKENS.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append("runtime authority is missing canonical checkpoint source: %s" % relative)
            continue
        try:
            code = _cpp_without_comments(path.read_text(encoding="utf-8"))
        except OSError as exc:
            errors.append("cannot read canonical checkpoint source %s: %s" % (relative, exc))
            continue
        if signature not in code:
            errors.append("%s is missing its canonical checkpoint/magic signature" % relative)

    for relative, versions in CANONICAL_CHECKPOINT_VERSION_TOKENS.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append("runtime authority is missing checkpoint version source: %s" % relative)
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append("cannot read checkpoint version source %s: %s" % (relative, exc))
            continue
        for name, expected in versions.items():
            pattern = re.compile(
                r"\b%s\s*=\s*%d\b" % (re.escape(name), expected)
            )
            if pattern.search(source) is None:
                errors.append(
                    "%s must publish %s = %d for the strict restart contract"
                    % (relative, name, expected)
                )


def _validate_source_barriers(errors: list[str]) -> None:
    missing = [relative for relative in PROGRAM_AUTHORITY_FILES if not (ROOT / relative).is_file()]
    if missing:
        errors.append("runtime authority source ownership is incomplete: %s" % missing)
    _validate_execution_service_architecture(errors)
    _validate_detached_provider_architecture(errors)
    _validate_retired_compatibility_aliases(errors)
    authority_sources = tuple(path for path in _authority_sources() if path.is_file())
    abi_sources = authority_sources
    if sum(
        len(ABI_VERSION_PATTERN.findall(_cpp_code_only(path.read_text(encoding="utf-8"))))
        for path in abi_sources
    ) != 1:
        errors.append("runtime authority must expose exactly one ABI version declaration (v5)")
    _scan_production_barriers(errors, authority_sources, reject_any_pending=True)
    authority_set = set(authority_sources)
    _scan_production_barriers(
        errors,
        (path for path in _production_sources() if path not in authority_set),
    )
    _validate_required_authority_contract(errors)

    # These source files are the ABI/service ownership witnesses consumed by the gate.  A missing
    # manifest row is a source-ownership failure even when CMake happens to discover the file.
    try:
        suites = _cpp_suites()
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError) as exc:
        errors.append("cannot read C++ source ownership: %s" % exc)
        return
    owned_sources = {
        relative
        for suite in suites.values()
        for relative in suite.get("sources", ())
    }
    witness_sources = [
        "tests/cpp/unit/runtime/test_program_host_descriptor.cpp",
        "tests/cpp/unit/runtime/test_program_transaction.cpp",
    ]
    for relative in witness_sources:
        if relative not in owned_sources:
            errors.append("tests/test_manifest.toml does not own %s" % relative)


def _validate_zero_allocation_proof(checks: Iterable[dict], errors: list[str]) -> None:
    """Require one unconditional, source-registered before/after allocator proof."""
    matching_rows = [
        row
        for row in checks
        if isinstance(row, dict)
        and all(row.get(key) == value for key, value in ALLOCATION_PROOF_ROW.items())
    ]
    if len(matching_rows) != 1:
        errors.append(
            "runtime authority requires exactly one allocation proof row %s (found %d)"
            % (ALLOCATION_PROOF_CASE, len(matching_rows))
        )

    source = ROOT / ALLOCATION_PROOF_SOURCE
    if not source.is_file():
        errors.append("runtime authority allocation proof source is missing: %s" % ALLOCATION_PROOF_SOURCE)
        return
    try:
        source_text = source.read_text(encoding="utf-8")
        body = _registered_gtest_case_body(source_text, ALLOCATION_PROOF_CASE)
    except (OSError, UnicodeError) as exc:
        errors.append("cannot read runtime authority allocation proof source: %s" % exc)
        return
    if body is None:
        errors.append("runtime authority allocation proof case is not source-registered: %s" % ALLOCATION_PROOF_CASE)
        return
    code = _cpp_code_only(body)
    required = (
        "install_execution_lane(sim",
        "sim.set_program_block_map(",
        "allocation_event_stats()",
        "before_reuse",
        "after_reuse",
        "EXPECT_EQ(after_reuse.fab_calls, before_reuse.fab_calls)",
        "EXPECT_EQ(after_reuse.fab_bytes, before_reuse.fab_bytes)",
        "ctx.rhs_scratch(41, 0, state)",
    )
    missing = [token for token in required if token not in code]
    if missing:
        errors.append("runtime authority allocation proof is missing counter witness(es): %s" % missing)
    bind = code.find("install_execution_lane(sim")
    block_map = code.find("sim.set_program_block_map(")
    warmup = code.find("ctx.rhs_scratch(41, 0, state)")
    counter = code.find("before_reuse")
    reused = code.find("ctx.rhs_scratch(41, 0, state)", warmup + 1)
    after = code.find("after_reuse")
    if (
        bind < 0
        or block_map < 0
        or warmup < 0
        or counter < 0
        or reused < 0
        or after < 0
        or not bind < block_map < warmup < counter < reused < after
    ):
        errors.append(
            "runtime authority allocation proof must bind/prep and warm scratch before the "
            "before/after reuse counters"
        )
    try:
        owned_sources = {
            relative
            for suite in _cpp_suites().values()
            for relative in suite.get("sources", ())
        }
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError) as exc:
        errors.append("cannot read C++ source ownership for allocation proof: %s" % exc)
    else:
        if ALLOCATION_PROOF_SOURCE not in owned_sources:
            errors.append(
                "tests/test_manifest.toml does not own runtime authority allocation proof source: %s"
                % ALLOCATION_PROOF_SOURCE
            )


def _validate_composed_gate_closures(errors: list[str]) -> None:
    """Attest that the M2/M3 manifests owning adjacent plan issues remain closed."""
    for (
        relative,
        runner_relative,
        expected_gate,
        expected_issues,
        expected_count,
        native_positive_issues,
    ) in COMPOSED_GATE_SPECS:
        path = ROOT / relative
        try:
            text = path.read_text(encoding="utf-8")
            data = tomllib.loads(text)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            errors.append("cannot read composed gate %s: %s" % (relative, exc))
            continue
        if data.get("schema_version") != 1:
            errors.append("composed gate %s must use schema_version = 1" % relative)
        if data.get("gate") != expected_gate:
            errors.append("composed gate %s has unexpected gate name" % relative)
        if data.get("deferred") != []:
            errors.append("composed gate %s must keep deferred = []" % relative)
        if "pending:" in text.lower():
            errors.append("composed gate %s contains a pending marker" % relative)
        checks = data.get("check")
        if not isinstance(checks, list) or not checks:
            errors.append("composed gate %s has no executable checks" % relative)
            continue
        declared = data.get("issues")
        if declared != list(expected_issues):
            errors.append(
                "composed gate %s does not declare required issues %s exactly"
                % (relative, list(expected_issues))
            )
        if len(checks) != expected_count:
            errors.append(
                "composed gate %s must contain exactly %d checks, found %d"
                % (relative, expected_count, len(checks))
            )
        for issue in expected_issues:
            rows = [row for row in checks if isinstance(row, dict) and row.get("issue") == issue]
            polarities = {row.get("polarity") for row in rows}
            if not rows:
                errors.append("composed gate %s has no rows for %s" % (relative, issue))
                continue
            if {"positive", "refusal"} - polarities:
                errors.append(
                    "composed gate %s lacks positive/refusal closure for %s"
                    % (relative, issue)
                )
            if issue in native_positive_issues and not any(
                row.get("polarity") == "positive" and row.get("kind") in {"ctest", "mpi_python"}
                for row in rows
            ):
                errors.append("composed gate %s lacks a native positive for %s" % (relative, issue))

        runner_path = ROOT / runner_relative
        try:
            module_name = "pops_composed_gate_" + expected_gate.replace("-", "_")
            spec = importlib.util.spec_from_file_location(module_name, runner_path)
            if spec is None or spec.loader is None:
                raise ImportError("cannot create module specification")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            validator = getattr(module, "validate_manifest")
            _validated, composed_errors = validator(path)
        except (AttributeError, ImportError, OSError, TypeError) as exc:
            errors.append("cannot load composed gate validator %s: %s" % (runner_relative, exc))
        else:
            errors.extend("composed gate %s: %s" % (relative, error) for error in composed_errors)


def _normalise_allocation(value: object) -> str | None:
    # Keep the manifest schema closed: allocation is an explicit lane-authentication policy, not a
    # truthy convenience flag whose meaning could drift between source and runtime execution.
    return value if isinstance(value, str) else None


def audit_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[dict, list[str]]:
    """Return deterministic source-only errors for the closed runtime-authority matrix."""
    errors: list[str] = []
    try:
        manifest_text = path.read_text(encoding="utf-8")
        data = tomllib.loads(manifest_text)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return {}, ["cannot read runtime authority gate manifest %s: %s" % (path, exc)]

    if set(data) != {"schema_version", "gate", "issues", "deferred", "check"}:
        errors.append("manifest fields must be schema_version/gate/issues/deferred/check")
    if data.get("schema_version") != 1:
        errors.append("schema_version must be exactly 1")
    if data.get("gate") != "runtime-authority":
        errors.append("gate must be exactly 'runtime-authority'")
    if data.get("issues") != list(EXPECTED_ISSUES):
        errors.append("issues must list ADC-700, ADC-702, ADC-720 exactly once")
    if data.get("deferred") != []:
        errors.append("closed runtime authority gate requires deferred = []")
    if any("pending" in str(value).lower() for value in data.get("deferred", ())):
        errors.append("closed runtime authority manifest cannot contain a pending marker")
    if "pending:" in manifest_text.lower():
        errors.append("closed runtime authority manifest cannot contain a pending marker")
    _validate_composed_gate_closures(errors)
    _validate_source_barriers(errors)

    checks = data.get("check")
    if not isinstance(checks, list) or not checks:
        errors.append("manifest must contain [[check]] rows")
        checks = []
    elif len(checks) != EXPECTED_CHECK_COUNT:
        errors.append(
            "closed runtime authority manifest requires exactly %d executable rows (found %d)"
            % (EXPECTED_CHECK_COUNT, len(checks))
        )
    _validate_zero_allocation_proof(checks, errors)

    identities = Counter()
    issue_coverage: dict[str, set[str]] = defaultdict(set)
    requirement_coverage: dict[str, set[str]] = defaultdict(set)
    native_positive_issues: set[str] = set()
    try:
        cpp_suites = _cpp_suites()
    except (OSError, tomllib.TOMLDecodeError, ValueError, KeyError, TypeError) as exc:
        errors.append("cannot read C++ source ownership manifest: %s" % exc)
        cpp_suites = {}
    try:
        mpi_entrypoints = _python_mpi_entrypoints()
    except (OSError, tomllib.TOMLDecodeError, ValueError, KeyError, TypeError) as exc:
        errors.append("cannot read Python MPI entrypoint ownership: %s" % exc)
        mpi_entrypoints = {}
    try:
        mpi_orchestrators = _python_mpi_orchestrators()
    except (OSError, tomllib.TOMLDecodeError, ValueError, KeyError, TypeError) as exc:
        errors.append("cannot read Python MPI orchestrator ownership: %s" % exc)
        mpi_orchestrators = set()

    for index, row in enumerate(checks, 1):
        where = "check[%d]" % index
        if not isinstance(row, dict):
            errors.append("%s must be a table" % where)
            continue
        kind = row.get("kind")
        base = {"issue", "requirement", "polarity", "kind", "target", "backend", "allocation"}
        expected = base | (
            {"nodeid", "nproc"}
            if kind == "mpi_python"
            else {"nodeid"}
            if kind in {"pytest", "mpi_orchestrator"}
            else {"test_regex"}
        )
        if set(row) != expected:
            errors.append("%s has unknown or missing fields: %s" % (where, sorted(row)))
            continue

        issue = row.get("issue")
        requirement = row.get("requirement")
        polarity = row.get("polarity")
        backend = row.get("backend")
        allocation = _normalise_allocation(row.get("allocation"))
        if issue not in EXPECTED_ISSUES:
            errors.append("%s has unknown issue %r" % (where, issue))
        if requirement not in REQUIRED_POLARITIES:
            errors.append("%s has unknown requirement %r" % (where, requirement))
        elif issue not in REQUIREMENT_ISSUES[requirement]:
            errors.append(
                "%s requirement %r cannot be attributed to %r" % (where, requirement, issue)
            )
        if polarity not in {"positive", "refusal"}:
            errors.append("%s polarity must be positive or refusal" % where)
        else:
            issue_coverage[str(issue)].add(polarity)
            requirement_coverage[str(requirement)].add(polarity)
        if backend not in EXPECTED_BACKENDS:
            errors.append("%s backend must be serial, mpi, or openmp" % where)
        if allocation not in EXPECTED_ALLOCATIONS:
            errors.append("%s allocation must be 'none' or 'required'" % where)
        if kind not in {"pytest", "mpi_python", "mpi_orchestrator", "ctest"}:
            errors.append("%s kind must be pytest, mpi_python, mpi_orchestrator, or ctest" % where)
            continue
        if kind != "ctest" and row.get("target") != requirement:
            errors.append("%s target must equal its exact requirement %r" % (where, requirement))
        identity = (kind, row.get("nodeid", row.get("test_regex")))
        identities[identity] += 1

        if kind == "pytest":
            if backend == "mpi":
                errors.append("%s ordinary pytest rows cannot claim the mpi backend" % where)
            relative = _validate_python_nodeid(
                row.get("nodeid"), where, errors, backend=str(backend)
            )
            if polarity == "positive" and relative is not None and not relative.startswith("tests/python/architecture/"):
                native_positive_issues.add(str(issue))
        elif kind == "mpi_python":
            if backend != "mpi":
                errors.append("%s mpi_python rows require backend = 'mpi'" % where)
            relative = _validate_python_nodeid(
                row.get("nodeid"), where, errors, backend="mpi"
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
                        "%s requires nproc=%d, not %d" % (relative, expected_nproc, nproc)
                    )
            if polarity == "positive":
                native_positive_issues.add(str(issue))
        elif kind == "mpi_orchestrator":
            if backend != "mpi":
                errors.append("%s mpi_orchestrator rows require backend = 'mpi'" % where)
            relative = _validate_python_nodeid(
                row.get("nodeid"), where, errors, backend="mpi"
            )
            if relative is not None and relative not in mpi_orchestrators:
                errors.append(
                    "%s is not a manifest-owned serial MPI orchestrator" % relative
                )
            if polarity == "positive":
                native_positive_issues.add(str(issue))
        else:
            target = row.get("target")
            selector = row.get("test_regex")
            if not isinstance(target, str) or target.count("@") != 1:
                errors.append("%s CTest target must be requirement@manifest-suite" % where)
                continue
            semantic, suite_name = target.split("@", 1)
            if semantic != requirement:
                errors.append("%s CTest target must start with requirement %r" % (where, requirement))
            suite = cpp_suites.get(suite_name)
            if suite is None:
                errors.append("%s references unknown CTest target %r" % (where, suite_name))
                continue
            labels = {str(label) for label in suite.get("labels", ())}
            has_mpi_registration = bool(
                suite.get("mpi_nproc") or suite.get("mpi_variants") or suite.get("mpi_rank_parity")
            )
            if backend == "mpi" and "mpi" not in labels and not has_mpi_registration:
                errors.append("%s mpi CTest target %r has no MPI registration" % (where, suite_name))
            if backend == "serial" and "mpi" in labels:
                errors.append("%s serial CTest row selects an MPI target %r" % (where, suite_name))
            if backend == "openmp":
                errors.append("%s openmp CTest rows are not supported; use an OpenMP Python row" % where)
            selected_case = _validate_exact_ctest_selector(selector, suite_name, suite, where, errors)
            if selected_case is not None and _gtest_case_has_skip(suite, selected_case):
                errors.append(
                    "%s selected CTest case %r is skipped or disabled" % (where, selected_case)
                )
            for relative in suite.get("sources", ()):
                source = ROOT / relative
                if not source.is_file():
                    errors.append("%s target %r has missing source %s" % (where, suite_name, relative))
                else:
                    text = source.read_text(encoding="utf-8")
                    if "DISABLED_" in text and selected_case is None:
                        errors.append("%s target %r contains an undiscoverable disabled marker" % (where, suite_name))
            if polarity == "positive":
                native_positive_issues.add(str(issue))

    _validate_required_lowering_refusal_rows(checks, errors)

    duplicates = sorted(identity for identity, count in identities.items() if count > 1)
    if duplicates:
        errors.append("duplicate executable checks: %s" % duplicates)
    for issue in EXPECTED_ISSUES:
        missing = {"positive", "refusal"} - issue_coverage[issue]
        if missing:
            errors.append("%s lacks %s coverage" % (issue, "/".join(sorted(missing))))
        if issue not in native_positive_issues:
            errors.append("%s lacks a mandatory native positive proof" % issue)
    for requirement, required in sorted(REQUIRED_POLARITIES.items()):
        missing = required - requirement_coverage[requirement]
        if missing:
            errors.append("%s lacks %s coverage" % (requirement, "/".join(sorted(missing))))
    return data, errors


def validate_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[dict, list[str]]:
    """Fail closed if any source, row, or deferred declaration is invalid."""
    return audit_manifest(path)


def _required_environment(
    rows: Iterable[dict] = (), *, backend: str = "serial"
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["POPS_REQUIRE_NATIVE_TESTS"] = "1"
    if backend == "mpi":
        environment["POPS_REQUIRE_MPI_TESTS"] = "1"
    required_allocations = any(row.get("allocation") == "required" for row in rows)
    if required_allocations:
        if environment.get("POPS_RUNTIME_AUTHORITY_ALLOCATION") != "1":
            raise RuntimeError(
                "allocation-authenticated rows require POPS_RUNTIME_AUTHORITY_ALLOCATION=1"
            )
        environment["POPS_REQUIRE_HOT_ALLOCATION_FREE"] = "1"
    if backend == "openmp":
        if environment.get("POPS_RUNTIME_AUTHORITY_OPENMP") != "1":
            raise RuntimeError("OpenMP rows require POPS_RUNTIME_AUTHORITY_OPENMP=1")
        try:
            threads = int(environment.get("OMP_NUM_THREADS", "0"))
        except ValueError as exc:
            raise RuntimeError("OpenMP rows require a positive OMP_NUM_THREADS") from exc
        if threads < 1:
            raise RuntimeError("OpenMP rows require a positive OMP_NUM_THREADS")
        kokkos_root = environment.get("POPS_KOKKOS_ROOT") or environment.get("Kokkos_ROOT")
        if not kokkos_root or not Path(kokkos_root).is_dir():
            raise RuntimeError("OpenMP rows require an existing POPS_KOKKOS_ROOT or Kokkos_ROOT")
    root = str(ROOT)
    inherited = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        [root, *(entry for entry in inherited.split(os.pathsep) if entry and entry != root)]
    )
    return environment


def _junit_skip_count(report: Path, producer: str) -> int:
    if not report.is_file():
        raise RuntimeError("runtime authority %s did not produce its mandatory JUnit report" % producer)
    try:
        root = ET.parse(report).getroot()
    except ET.ParseError as exc:
        raise RuntimeError("runtime authority %s produced an invalid JUnit report" % producer) from exc
    return len(root.findall(".//skipped"))


def _run_required_pytest(nodeids: list[str], *, environment: dict[str, str]) -> None:
    with tempfile.TemporaryDirectory(prefix="pops-runtime-authority-pytest-") as temporary:
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
        print("+", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        skipped = _junit_skip_count(report, "pytest")
        if skipped:
            raise RuntimeError(
                "runtime authority pytest reported %d skipped/xfail proof(s)" % skipped
            )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)


def _run(command: list[str], *, environment: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _validated_native_dimension(dimension: int) -> int:
    """Return one supported native dimension, rejecting programmatic bypasses of argparse."""
    if isinstance(dimension, bool) or dimension not in (1, 2, 3):
        raise ValueError("native dimension must be exactly 1, 2, or 3")
    return dimension


def _require_build_native_dimension(build_dir: Path, dimension: int) -> None:
    """Fail closed unless the CTest build is configured for the selected native dimension."""
    dimension = _validated_native_dimension(dimension)
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        raise RuntimeError(
            "runtime authority CTest requires %s with POPS_NATIVE_DIM=%d"
            % (cache, dimension)
        )
    matches = re.findall(
        r"^POPS_NATIVE_DIM:[^=]+=(.+)$",
        cache.read_text(encoding="utf-8", errors="replace"),
        flags=re.MULTILINE,
    )
    if len(matches) != 1:
        raise RuntimeError(
            "runtime authority CTest build %s must define POPS_NATIVE_DIM exactly once"
            % build_dir
        )
    try:
        configured_dimension = _validated_native_dimension(int(matches[0].strip()))
    except ValueError as exc:
        raise RuntimeError(
            "runtime authority CTest build %s has invalid POPS_NATIVE_DIM=%r"
            % (build_dir, matches[0].strip())
        ) from exc
    if configured_dimension != dimension:
        raise RuntimeError(
            "runtime authority CTest build %s has POPS_NATIVE_DIM=%d, but --dim=%d"
            % (build_dir, configured_dimension, dimension)
        )


def _mpi_python_command(
    mpi_exec: str,
    nproc: int,
    relative: str,
    *,
    dimension: int,
) -> list[str]:
    dimension = _validated_native_dimension(dimension)
    if shutil.which(mpi_exec) is None:
        raise RuntimeError("required MPI launcher %r is unavailable" % mpi_exec)
    bootstrap = (
        "from pops._native_selector import select_native_dimension; "
        "import runpy, sys; "
        "select_native_dimension(int(sys.argv[1])); "
        "runpy.run_path(sys.argv[2], run_name='__main__')"
    )
    return [
        mpi_exec,
        "-n",
        str(nproc),
        sys.executable,
        "-c",
        bootstrap,
        str(dimension),
        str(ROOT / relative),
    ]


def _run_ctest(
    build_dir: Path,
    target: str,
    selector: str,
    *,
    environment: dict[str, str],
) -> None:
    listed = subprocess.run(
        ["ctest", "--test-dir", str(build_dir), "-N", "-R", selector],
        cwd=ROOT,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    if "Total Tests: 0" in listed.stdout or "Test #" not in listed.stdout:
        raise RuntimeError(
            "runtime authority CTest target %r (%s) is not built in %s"
            % (target, selector, build_dir)
        )
    with tempfile.TemporaryDirectory(prefix="pops-runtime-authority-ctest-") as temporary:
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
        _run(command, environment=environment)
        skipped = _junit_skip_count(report, "CTest")
        if skipped:
            raise RuntimeError(
                "runtime authority CTest %r reported %d skipped proof(s)" % (selector, skipped)
            )


def _required_ctest_targets(checks: Iterable[dict], *, backend: str = "mpi") -> tuple[str, ...]:
    selected = _selected_checks(checks, backend=backend)
    return tuple(
        sorted(
            {
                row["target"].split("@", 1)[1]
                for row in selected
                if row["kind"] == "ctest"
            }
        )
    )


def _selected_checks(checks: Iterable[dict], *, backend: str) -> list[dict]:
    if backend == "mpi":
        return [row for row in checks if row.get("backend") in {"serial", "mpi"}]
    return [row for row in checks if row.get("backend") == backend]


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


@contextmanager
def _temporary_environment(environment: dict[str, str]) -> Iterable[None]:
    """Run a composed gate with the same authenticated environment as this gate."""
    previous = os.environ.copy()
    os.environ.clear()
    os.environ.update(environment)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def _load_composed_runner(runner_relative: str, module_name: str):
    runner_path = ROOT / runner_relative
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot create module specification for %s" % runner_relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_composed_gate(
    relative: str,
    runner_relative: str,
    *,
    build_dir: Path,
    mpi_exec: str,
    environment: dict[str, str],
    python_only: bool,
    ctest_only: bool,
    dimension: int,
) -> None:
    """Execute one complete adjacent gate through its own runner, not just its validator."""
    module_name = "pops_runtime_composed_" + Path(runner_relative).stem
    module = _load_composed_runner(runner_relative, module_name)
    manifest = ROOT / relative
    with _temporary_environment(environment):
        if ctest_only:
            data, errors = module.validate_manifest(manifest)
            if errors:
                raise RuntimeError(
                    "%s gate manifest is incomplete or invalid: %s"
                    % (relative, "; ".join(errors))
                )
            rows = sorted(
                (row for row in data["check"] if row["kind"] == "ctest"),
                key=lambda value: (value["target"], value["test_regex"]),
            )
            for row in rows:
                module._run_ctest(
                    build_dir,
                    row["target"],
                    row["test_regex"],
                )
            return

        # M2/M3 do not expose a native-dimension flag.  They inherit the authenticated
        # POPS_NATIVE_DIM environment established by this runner, including their MPI children.
        _validated_native_dimension(dimension)
        argv = ["--manifest", str(manifest), "--build-dir", str(build_dir)]
        if Path(runner_relative).name == "run_m3_gate.py":
            argv.extend(("--mpi-exec", mpi_exec))
        if python_only:
            argv.append("--python-only")
        print("+ composed gate %s %s %s" % (relative, runner_relative, " ".join(argv)), flush=True)
        status = module.main(argv)
        if status != 0:
            raise RuntimeError("composed gate %s returned status %s" % (relative, status))


def _run_composed_gates(
    checks: Iterable[dict],
    *,
    backend: str,
    build_dir: Path,
    mpi_exec: str,
    environment: dict[str, str],
    python_only: bool,
    ctest_only: bool,
    dimension: int,
) -> None:
    """Run M2, M3, and M3's topology-changing orchestrator on the authenticated MPI lane."""
    if backend != "mpi":
        return

    for relative, runner_relative, _gate, _issues, _count, _native in COMPOSED_GATE_SPECS:
        _run_composed_gate(
            relative,
            runner_relative,
            build_dir=build_dir,
            mpi_exec=mpi_exec,
            environment=environment,
            python_only=python_only,
            ctest_only=ctest_only,
            dimension=dimension,
        )

    if ctest_only:
        return

    m3_module = _load_composed_runner("scripts/run_m3_gate.py", "pops_runtime_composed_m3_orchestrator")
    m3_manifest = ROOT / "tests/gates/m3_amr_multilayout.toml"
    m3_data, m3_errors = m3_module.validate_manifest(m3_manifest)
    if m3_errors:
        raise RuntimeError(
            "M3 orchestrator ownership manifest is incomplete or invalid: %s"
            % "; ".join(m3_errors)
        )
    owned_orchestrators = set(m3_module._python_mpi_orchestrators())
    m3_nodeids = {
        row.get("nodeid")
        for row in m3_data["check"]
        if row.get("kind") == "pytest"
    }
    orchestrator_rows = [
        row for row in checks if row.get("kind") == "mpi_orchestrator"
    ]
    for row in orchestrator_rows:
        relative = row["nodeid"].split("::", 1)[0]
        if relative not in owned_orchestrators:
            raise RuntimeError(
                "runtime authority orchestrator %s is not owned by the M3 manifest" % relative
            )
        if row["nodeid"] not in m3_nodeids:
            raise RuntimeError(
                "runtime authority orchestrator %s is not an executable M3 row" % row["nodeid"]
            )
        print("+ M3 runner executes MPI orchestrator row %s" % row["nodeid"], flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--list-ctest-targets", action="store_true")
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument("--ctest-only", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-mpi")
    parser.add_argument("--dim", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--mpi-exec", default="mpiexec")
    parser.add_argument("--backend", choices=sorted(EXPECTED_BACKENDS), default="serial")
    parser.add_argument("--openmp", action="store_true", help="select authenticated OpenMP rows")
    args = parser.parse_args(argv)
    if args.python_only and args.ctest_only:
        parser.error("--python-only and --ctest-only are mutually exclusive")
    backend = "openmp" if args.openmp else args.backend
    if args.openmp and args.backend != "serial":
        parser.error("--openmp cannot be combined with --backend")
    dimension = _validated_native_dimension(args.dim)

    if args.audit_only:
        data, errors = audit_manifest(args.manifest)
    else:
        data, errors = validate_manifest(args.manifest)
    if errors:
        print("Runtime authority gate is incomplete or invalid:", file=sys.stderr)
        for error in errors:
            print(" -", error, file=sys.stderr)
        return 2

    checks = data["check"]
    if args.list_ctest_targets:
        targets = _required_ctest_targets(checks, backend=backend)
        if not targets:
            print("runtime authority gate selects no CTest build target", file=sys.stderr)
            return 2
        print("\n".join(targets))
        return 0

    selected = _selected_checks(checks, backend=backend)
    print(
        "Runtime authority gate source matrix: %s (%d executable, %d deferred; backend=%s, selected=%d)"
        % (
            "AUDITED CLOSED" if args.audit_only else "CLOSED",
            len(checks),
            len(data["deferred"]),
            backend,
            len(selected),
        )
    )
    if args.check_only or args.audit_only:
        return 0

    if not args.python_only:
        _require_build_native_dimension(args.build_dir, dimension)
    environment = _required_environment(selected, backend=backend)
    environment["POPS_NATIVE_DIM"] = str(dimension)
    if backend == "mpi":
        # The topology-changing M3 orchestrator chooses its launcher from POPS_MPIEXEC.  Keep
        # that choice identical to the launcher used by this final gate instead of allowing a
        # second ambient mpiexec selection inside the capture/restart driver.
        environment["POPS_MPIEXEC"] = args.mpi_exec
    if not args.ctest_only:
        nodeids = [row["nodeid"] for row in selected if row["kind"] == "pytest"]
        for chunk in _chunks(nodeids, 24):
            _run_required_pytest(chunk, environment=environment)
        for row in selected:
            if row["kind"] != "mpi_python":
                continue
            relative = row["nodeid"].split("::", 1)[0]
            _run(
                _mpi_python_command(
                    args.mpi_exec,
                    row["nproc"],
                    relative,
                    dimension=dimension,
                ),
                environment=environment,
            )
    if not args.python_only:
        for row in sorted(
            (row for row in selected if row["kind"] == "ctest"),
            key=lambda value: (value["target"], value["test_regex"]),
        ):
            suite_name = row["target"].split("@", 1)[1]
            _run_ctest(
                args.build_dir,
                suite_name,
                row["test_regex"],
                environment=environment,
            )
    _run_composed_gates(
        selected,
        backend=backend,
        build_dir=args.build_dir,
        mpi_exec=args.mpi_exec,
        environment=environment,
        python_only=args.python_only,
        ctest_only=args.ctest_only,
        dimension=dimension,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
