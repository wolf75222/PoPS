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
    "resource_plan_ceiling": {"positive", "refusal"},
    "lowering_refusal": {"refusal"},
    "gate_execution": {"positive", "refusal"},
}
REQUIREMENT_ISSUES = {
    "abi_identity": {"ADC-700"},
    "solve_outcome": {"ADC-700", "ADC-702"},
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
    "resource_plan_ceiling": {"ADC-702"},
    "lowering_refusal": {"ADC-700"},
    "gate_execution": set(EXPECTED_ISSUES),
}
EXPECTED_BACKENDS = {"serial", "mpi", "openmp"}
EXPECTED_ALLOCATIONS = {"none", "required"}
SUPPORTED_NATIVE_DIMENSIONS = (1, 2, 3)
EXPECTED_CHECK_COUNT = 73
REQUIRED_LOWERING_REFUSAL_NODEIDS = frozenset(
    {
        "tests/python/unit/codegen/test_scheduler_codegen.py::test_scratch_skip_refuses_unprepared_stale_state",
        "tests/python/unit/codegen/test_scheduler_codegen.py::test_on_end_refuses_to_lower",
    }
)
REQUIRED_SEMANTIC_BARRIER_NODEIDS = {
    "legacy_context_barrier": frozenset(
        {
            "tests/python/architecture/test_program_only_temporal_facades.py::test_system_temporal_facades_dispatch_only_through_an_installed_program",
            "tests/python/architecture/test_program_only_temporal_facades.py::test_historical_block_scheduler_is_not_an_installed_temporal_authority",
        }
    ),
    "legacy_symbol_barrier": frozenset(
        {
            "tests/python/architecture/test_no_legacy_runtime_routes.py::test_program_has_one_runtime_branch_spelling",
            "tests/python/architecture/test_program_only_temporal_facades.py::test_production_has_no_second_amr_time_engine",
        }
    ),
    "legacy_fragment_barrier": frozenset(
        {
            "tests/python/architecture/test_amr_program_support_parity.py::test_context_include_parser_authenticates_both_delimiters_and_hidden_fragments",
            "tests/python/architecture/test_program_only_temporal_facades.py::test_static_system_assembler_is_retired_from_the_final_runtime_surface",
        }
    ),
    "pending_marker_barrier": frozenset(
        {
            "tests/python/architecture/test_amr_program_support_parity.py::test_parser_finds_only_explicit_known_deferrals",
            "tests/python/architecture/test_amr_program_support_parity.py::test_context_sensitive_routes_report_green_or_pending_from_resolved_hierarchy",
        }
    ),
}
REQUIRED_SEMANTIC_BARRIER_POLARITIES = {
    "legacy_context_barrier": {
        "positive": "tests/python/architecture/test_program_only_temporal_facades.py::test_system_temporal_facades_dispatch_only_through_an_installed_program",
        "refusal": "tests/python/architecture/test_program_only_temporal_facades.py::test_historical_block_scheduler_is_not_an_installed_temporal_authority",
    },
    "legacy_symbol_barrier": {
        "positive": "tests/python/architecture/test_no_legacy_runtime_routes.py::test_program_has_one_runtime_branch_spelling",
        "refusal": "tests/python/architecture/test_program_only_temporal_facades.py::test_production_has_no_second_amr_time_engine",
    },
    "legacy_fragment_barrier": {
        "positive": "tests/python/architecture/test_amr_program_support_parity.py::test_context_include_parser_authenticates_both_delimiters_and_hidden_fragments",
        "refusal": "tests/python/architecture/test_program_only_temporal_facades.py::test_static_system_assembler_is_retired_from_the_final_runtime_surface",
    },
    "pending_marker_barrier": {
        "positive": "tests/python/architecture/test_amr_program_support_parity.py::test_parser_finds_only_explicit_known_deferrals",
        "refusal": "tests/python/architecture/test_amr_program_support_parity.py::test_context_sensitive_routes_report_green_or_pending_from_resolved_hierarchy",
    },
}
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
    "RuntimeProgramContext",
    "program_context.hpp",
    "amr_program_context.hpp",
    "runtime_program_context.hpp",
)
# Native bricks have a deliberately separate C ABI. Keep the precise exported spellings here
# rather than relying on an incidental mismatch with the retired Program-route patterns below:
# a future broadening of those patterns must not reject one of these independently versioned
# brick contracts. Conversely, a new brick-looking Program split route is not grandfathered.
NATIVE_BRICK_ABI_SYMBOLS = frozenset(
    (
        "pops_brick_manifest",
        "pops_brick_nvars",
        "pops_brick_nproviders",
        "pops_brick_residual_v5",
        "pops_brick_install_system_v7",
        "pops_brick_install_amr_v5",
        "pops_brick_model_identity",
        "pops_brick_kokkos_backend",
        "pops_brick_kokkos_version",
    )
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
LEGACY_PROGRAM_QUALIFIED_INSTALL_ABI_PATTERN = re.compile(
    # `pops_install_program` is the sole ABI-v5 Program entrypoint.  Any target/version suffix
    # recreates a second install ABI; native-brick `pops_brick_install_amr_v5` is distinct.
    r"\bpops_install_program(?:_(?:amr|system|uniform|v[0-9]+))+\b"
)
# A retired Program installer may have carried a generator/backend qualifier before its target
# suffix (for example ``pops_install_program_cuda_amr``).  Keep this anchored to the exact
# ``pops_install_program`` family so native brick exports such as ``pops_brick_install_amr_v5``
# remain outside the barrier.
LEGACY_PROGRAM_INSTALL_TARGET_VARIANT_PATTERN = re.compile(
    r"\bpops_install_program(?:_[A-Za-z0-9]+)*(?:_(?:amr|system|uniform|v[0-9]+))"
    r"(?:_[A-Za-z0-9]+)*\b"
)
# The former System/AMR split also had unversioned route, time-bound and boundary exports.  Keep
# these Program-prefixed spellings distinct from the native-brick ``pops_register_provider_routes``
# ABI, and reject only the known split families (including an explicit target/version suffix).
LEGACY_PROGRAM_SPLIT_SYMBOL_TOKENS = (
    "pops_program_route_manifest",
    "pops_program_routes",
    "pops_program_dt_bound",
    "pops_program_dt",
    "pops_program_boundaries",
    "pops_install_field_boundaries",
    "pops_install_program_boundaries",
)
LEGACY_PROGRAM_SPLIT_ABI_PATTERN = re.compile(
    # The first cutover used both the long manifest/bound spellings and the shorter
    # routes/dt/boundaries exports.  All of these names published a second Program ABI; keep
    # their roots closed while allowing historical target/backend/version suffix chains.
    r"\b(?:pops_program_(?:route_manifest|routes|dt_bound|dt|boundaries)|"
    r"pops_install_(?:field|program)_boundaries|pops_register_program_provider_routes)"
    # Historical generators appended target, backend, and version qualifiers (for example
    # ``_amr_v7``, ``_v2_system``, or ``_cuda_legacy``).  The base names are retired ABI families,
    # so consume any identifier suffix while retaining a word boundary.  Native
    # ``pops_register_provider_routes_amr`` has no ``program_`` component and is not matched.
    r"(?:_[A-Za-z0-9]+)*\b"
)
# A retired split export sometimes carried generator-specific qualifiers before its AMR target
# (for example ``pops_program_dt_bound_cuda_amr``).  The base names remain deliberately closed;
# only the suffix chain ending in the retired AMR target is open, so unrelated Program symbols do
# not become false positives.
LEGACY_PROGRAM_SPLIT_AMR_VARIANT_PATTERN = re.compile(
    r"\b(?:pops_program_(?:route_manifest|routes|dt_bound|dt|boundaries)|"
    r"pops_install_(?:field|program)_boundaries|pops_register_program_provider_routes)"
    r"(?:_[A-Za-z][A-Za-z0-9]*)*_amr\b"
)
# Historical generators also emitted a target-specific Program export that did not retain one of
# the three names above (for example ``pops_install_program_flux_amr_v2``).  Require ``program``
# as its own underscore-delimited component so the generic barrier does not reject the native
# brick exports ``pops_register_provider_routes_amr`` or ``pops_brick_install_amr_v5``.
LEGACY_PROGRAM_GENERIC_AMR_PATTERN = re.compile(
    r"\bpops_(?:[A-Za-z0-9]+_)*program(?:_[A-Za-z0-9]+)*_amr"
    r"(?:_[A-Za-z0-9]+)*\b",
    re.IGNORECASE,
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
# ``install_program_amr`` was a facade-local Program installer rather than a native-brick ABI.
# Match its bounded suffix family separately from the cell-temporal routes below so a later
# target/version spelling cannot recreate the retired AMR authority.
LEGACY_PROGRAM_AMR_INSTALLER_PATTERN = re.compile(
    r"\b(?:pops_)?(?:install|register)_program_amr(?:_[A-Za-z0-9]+)*\s*\(",
    re.IGNORECASE,
)
LEGACY_PROGRAM_AMR_INSTALLER_NAME_PATTERN = re.compile(
    r"(?:pops_)?(?:install|register)_program_amr(?:_[A-Za-z0-9]+)*",
    re.IGNORECASE,
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
LEGACY_CHECKPOINT_VERSION_PATTERNS = (
    re.compile(
        r"\b(?:k|[A-Za-z_]*)(?:Uniform|uniform)[A-Za-z_]*(?:Checkpoint|checkpoint)"
        r"[A-Za-z_]*(?:Version|version)\s*=\s*8[uUlL]*\b"
    ),
    re.compile(
        r"\b(?:k|[A-Za-z_]*)(?:Amr|AMR|amr)[A-Za-z_]*(?:Checkpoint|checkpoint)"
        r"[A-Za-z_]*(?:Version|version)\s*=\s*11[uUlL]*\b"
    ),
    re.compile(
        r"\b(?:k|[A-Za-z_]*)(?:Checkpoint|checkpoint)[A-Za-z_]*(?:Version|version)"
        r"\s*=\s*(?:8|11)[uUlL]*\b"
    ),
)
LEGACY_CHECKPOINT_CHAR_ARRAY_PATTERNS = {
    marker: re.compile(
        r"(?:['\"]" + r"['\"],\s*['\"]".join(re.escape(char) for char in marker) + r"['\"])",
        re.IGNORECASE,
    )
    for marker in LEGACY_CHECKPOINT_MARKERS
}
ABI_VERSION_PATTERN = re.compile(
    r"\bkProgramInstallAbiVersion\s*(?:=|\{)\s*5[uUlL]*\b"
)
LEGACY_ABI_VERSION_PATTERN = re.compile(
    r"\bkProgramInstallAbiVersion\s*(?:=|\{)\s*(?!5[uUlL]*\b)\d+[uUlL]*\b"
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
DISPATCHER_DEFINITION_PATTERN = re.compile(
    r"(?m)^\s*(?:inline\s+)?(?:void|bool|auto|[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s+"
    r"(?:[A-Za-z_]\w*::)?dispatch_cadence_step\s*\([^;{}]*\)\s*(?:const\s*)?\{"
)
PARALLEL_AUTHORITY_TABLE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:parallel|Parallel)[A-Za-z0-9_]*(?:table|Table)(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])(?:table|Table)[A-Za-z0-9_]*(?:parallel|Parallel)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
PARALLEL_AUTHORITY_DISPATCH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:parallel|Parallel)[A-Za-z0-9_]*(?:dispatch|Dispatch)(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])(?:dispatch|Dispatch)[A-Za-z0-9_]*(?:parallel|Parallel)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
PARALLEL_AUTHORITY_TABLE_NAME_PATTERN = re.compile(
    r"(?:parallel|Parallel)[A-Za-z0-9_]*(?:table|Table)|"
    r"(?:table|Table)[A-Za-z0-9_]*(?:parallel|Parallel)",
    re.IGNORECASE,
)
PARALLEL_AUTHORITY_DISPATCH_NAME_PATTERN = re.compile(
    r"(?:parallel|Parallel)[A-Za-z0-9_]*(?:dispatch|Dispatch)|"
    r"(?:dispatch|Dispatch)[A-Za-z0-9_]*(?:parallel|Parallel)",
    re.IGNORECASE,
)
# A secondary Program authority can be renamed without retaining the historical ``parallel``
# qualifier.  These are deliberately exact generic spellings; do not turn this into a broad
# ``program_*`` ban because candidate metadata and ordinary authoring helpers use that namespace.
PROGRAM_AUTHORITY_TABLE_PATTERN = re.compile(
    r"\b(?:ProgramDispatchTable|program_dispatch_table|program_table)"
    r"(?:_[A-Za-z0-9]+)*\b"
)
PROGRAM_AUTHORITY_DISPATCH_PATTERN = re.compile(
    r"\bprogram_dispatch(?:_[A-Za-z0-9]+)*\b"
)
PROGRAM_AUTHORITY_TABLE_NAME_PATTERN = re.compile(
    r"(?:ProgramDispatchTable|program_dispatch_table|program_table)"
    r"(?:_[A-Za-z0-9]+)*"
)
PROGRAM_AUTHORITY_DISPATCH_NAME_PATTERN = re.compile(
    r"program_dispatch(?:_[A-Za-z0-9]+)*"
)
# One candidate object is deliberately allowed to carry a local dispatch table while it is being
# prepared.  These exact spellings are not a second published authority: their names encode the
# candidate ownership.  Do not widen this exception to suffix variants, which would let a second
# versioned dispatcher hide behind a candidate-looking name.
CANONICAL_CANDIDATE_AUTHORITY_NAMES = frozenset(
    {
        "ProgramDispatchTableCandidate",
        "program_dispatch_candidate",
        "program_dispatch_table_candidate",
        "program_table_candidate",
    }
)
DIRECT_DISPATCH_BYPASS_PATTERN = re.compile(
    r"\bprogram_?(?:\.|->)\s*step_\s*\("
)
AMR_RUNTIME_STEP_BYPASS_PATTERN = re.compile(
    r"\b(?:"
    r"(?:Amr|AMR)(?:Runtime|Engine|LevelRuntime|LevelEngine|SubcyclingEngine)"
    r"(?:\s*<[^>{};\n]+>)?\s*::\s*"
    r"(?:step|step_level|advance|advance_level|advance_macro_step)|"
    r"(?:amr_(?:runtime|engine|level_runtime|level_engine|subcycling_engine)|"
    r"(?:level|subcycling)_engine|level_runtime|(?:amr_)?runtime)_?\s*(?:\.|->)\s*"
    r"(?:step|step_level|advance|advance_level|advance_macro_step)"
    r"|(?:Amr|AMR)?(?:Runtime|Engine|LevelRuntime|LevelEngine|SubcyclingEngine)"
    r"(?:\s*<[^>{};\n]+>)?\s*::\s*"
    r"(?:step|step_level|advance|advance_level|advance_macro_step)"
    r"|engine\s*(?:\.|->)\s*"
    r"(?:step|step_level|advance|advance_level|advance_macro_step)"
    r")\s*\("
)
AMR_RUNTIME_ALIAS_PATTERN = re.compile(
    r"\busing\s+(?P<alias>[A-Za-z_]\w*)\s*=\s*"
    r"(?:(?:::)?[A-Za-z_]\w*::)*AmrRuntime(?:\s*<[^;{}\n]+>)?\s*;|"
    r"\btypedef\s+(?:(?:::)?[A-Za-z_]\w*::)*AmrRuntime(?:\s*<[^;{}\n]+>)?\s+"
    r"(?P<typedef_alias>[A-Za-z_]\w*)\s*;"
)
PYTHON_AMR_RUNTIME_OWNER_NAMES = frozenset(
    {
        "amr_runtime",
        "amr_engine",
        "amr_level_runtime",
        "amr_level_engine",
        "amr_subcycling_engine",
        "level_runtime",
        "level_engine",
        "subcycling_engine",
        "amrruntime",
        "amrengine",
        "amrlevelruntime",
        "amrlevelengine",
        "amrsubcyclingengine",
        "runtime",
        "engine",
    }
)
PYTHON_AMR_RUNTIME_STEP_NAMES = frozenset(
    {"step", "step_level", "advance", "advance_level", "advance_macro_step"}
)
CELL_LOCAL_INSTALLER_PATTERN = re.compile(
    r"\b(?:pops_)?(?:install|register)_[A-Za-z0-9_]*cell[_-]?local"
    r"(?:[_-][A-Za-z0-9]+)*\s*\(|"
    r"\b(?:pops_)?(?:cell[_-]?local)_[A-Za-z0-9_]*(?:install|register)"
    r"(?:[_-][A-Za-z0-9]+)*\s*\(|"
    r"\b(?:pops_)?(?:install|register)_(?:program_)?cell"
    r"(?:[_-]?(?:local|temporal))?"
    r"(?:[_-][A-Za-z0-9]+)*\s*\(|"
    r"\b(?:pops_)?cell[_-]?(?:local|temporal)_[A-Za-z0-9_]*(?:install|register)"
    r"(?:[_-][A-Za-z0-9]+)*\s*"
    r"\(",
    re.IGNORECASE,
)
CELL_LOCAL_INSTALLER_NAME_PATTERN = re.compile(
    r"(?:pops_)?(?:install|register)_[A-Za-z0-9_]*cell[_-]?local"
    r"(?:[_-][A-Za-z0-9]+)*|"
    r"(?:pops_)?(?:cell[_-]?local)_[A-Za-z0-9_]*(?:install|register)"
    r"(?:[_-][A-Za-z0-9]+)*|"
    r"(?:pops_)?(?:install|register)_(?:program_)?cell"
    r"(?:[_-]?(?:local|temporal))?"
    r"(?:[_-][A-Za-z0-9]+)*|"
    r"(?:pops_)?cell[_-]?(?:local|temporal)_[A-Za-z0-9_]*(?:install|register)"
    r"(?:[_-][A-Za-z0-9]+)*",
    re.IGNORECASE,
)
CACHE_NODE_ID_ONLY_PATTERN = re.compile(
    r"(?:\bstd::map\s*<\s*int\s*,\s*CacheSlot\b|"
    r"\bstd::unordered_map\s*<\s*int\s*,\s*CacheSlot\b|"
    r"\b(?:node_ids|declare_slot|cache_nodes|program_cache_nodes)\s*\(|"
    r"\bcache_(?:should_update|store_scratch|restore_scratch|accumulate_dt|effective_dt)\s*"
    r"\(\s*(?:int|std::int64_t)\s+(?:node_id|value_id)\b)"
)
# Python checkpoint/resource-plan code still needs to inspect a legacy ``cache_nodes`` input in
# order to reject it.  The source barrier therefore checks AST publications and cache-owned
# mappings, rather than every occurrence of the key or of the legitimate control-graph ``node_id``
# label.
LEGACY_CACHE_WIRE_KEYS = frozenset(("cache_nodes", "cache_node_id", "node_id"))
PYTHON_SERIALIZATION_NAME_HINTS = frozenset(
    (
        "capture",
        "checkpoint",
        "dump",
        "encode",
        "persist",
        "serialize",
        "serialise",
        "snapshot",
        "wire",
        "write",
    )
)
PYTHON_WIRE_CONTAINER_NAME_HINTS = frozenset(
    ("checkpoint", "data", "out", "payload", "record", "serialized", "serialised", "snapshot", "state", "wire")
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
    source_paths = tuple(
        path for path in _runtime_source_paths() if path.suffix in CXX_SUFFIXES
    )
    source_texts = {
        path: path.read_text(encoding="utf-8")
        for path in source_paths
    }
    source_code = {path: _cpp_code_only(text) for path, text in source_texts.items()}

    fragment_paths = {
        path
        for path in _runtime_source_paths()
        if path.suffix == ".inc" and path.is_file()
    }
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
    public_amr_name = Path(PUBLIC_AMR_ROOT).name
    for path in source_paths:
        if path.name == public_amr_name and path != amr_root:
            errors.append(
                "%s is a forbidden public AMR execution-services root; use the generic "
                "ProgramExecutionServices<Dim> authority" % _display_source_path(path)
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
    elif (
        len(DISPATCHER_PATTERN.findall(runtime_state_code)) != 1
        or len(DISPATCHER_DEFINITION_PATTERN.findall(runtime_state_code)) != 1
    ):
        errors.append(
            "ProgramRuntimeState must define exactly one canonical dispatch_cadence_step dispatcher"
        )

    dispatcher_definitions = sum(
        len(DISPATCHER_DEFINITION_PATTERN.findall(code)) for code in source_code.values()
    )
    if dispatcher_definitions != 1:
        errors.append(
            "runtime authority must expose exactly one dispatch_cadence_step definition "
            "across all runtime production roots (found %d)" % dispatcher_definitions
        )

    for relative in DISPATCHER_SOURCES:
        path = ROOT / relative
        if not path.is_file():
            errors.append("runtime authority dispatcher source is missing: %s" % relative)
            continue
        code = _cpp_code_only(path.read_text(encoding="utf-8"))
        if DISPATCHER_PATTERN.search(code) is None:
            errors.append("%s does not enter the canonical dispatch_cadence_step" % relative)

    bypass_paths = set(source_paths)
    bypass_paths.update(
        ROOT / relative for relative in (*DISPATCHER_SOURCES, "src/runtime/system/system_io.cpp")
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


def _validate_required_semantic_barrier_rows(
    checks: Iterable[dict], errors: list[str]
) -> None:
    """Pin each semantic legacy barrier to its authenticated source-registered nodeids."""
    for requirement, expected in REQUIRED_SEMANTIC_BARRIER_NODEIDS.items():
        rows = [
            row
            for row in checks
            if isinstance(row, dict) and row.get("requirement") == requirement
        ]
        actual = {row.get("nodeid") for row in rows}
        if actual != set(expected):
            errors.append(
                "%s must pin exactly the source-registered nodeids %s (found %s)"
                % (requirement, sorted(expected), sorted(actual, key=str))
            )
        actual_polarities = {(row.get("polarity"), row.get("nodeid")) for row in rows}
        expected_polarities = set(REQUIRED_SEMANTIC_BARRIER_POLARITIES[requirement].items())
        if actual_polarities != expected_polarities:
            errors.append(
                "%s must pin each polarity to its exact source-registered nodeid "
                "(expected %s, found %s)"
                % (
                    requirement,
                    sorted(expected_polarities),
                    sorted(actual_polarities, key=str),
                )
            )
        for index, row in enumerate(rows, 1):
            mismatches = [
                "%s=%r" % (field, row.get(field))
                for field, value in {
                    "kind": "pytest",
                    "backend": "serial",
                    "allocation": "none",
                }.items()
                if row.get(field) != value
            ]
            if mismatches:
                errors.append(
                    "%s[%d] has non-canonical proof fields: %s"
                    % (requirement, index, ", ".join(mismatches))
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
        ROOT / "python/bindings",
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


def _runtime_source_paths() -> tuple[Path, ...]:
    """Return the union of every runtime production root and the authority ownership view."""
    return tuple(
        sorted(
            {
                path
                for path in (*_authority_sources(), *_production_sources())
                if path.is_file()
            }
        )
    )


def _display_source_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _is_retired_program_context_filename(path: Path) -> bool:
    """Recognize retired ProgramContext headers, including historical separator spellings."""
    # Preserve camel-case word boundaries before lowercasing: RuntimeProgramContext.hpp and
    # RuntimeProgramContextDetail.hpp are the same retired family as
    # runtime_program_context*.hpp, while runtime_program_contextual.hpp remains unrelated.
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", path.stem)
    normalized = re.sub(r"[-.]+", "_", normalized).lower()
    # Keep the suffix boundary explicit: ``program_contextual.hpp`` is an ordinary, unrelated
    # name, while detail/generated headers such as ``runtime-program-context-detail.hpp`` are
    # still part of the retired context family.
    return re.fullmatch(
        r"(?:(?:amr|runtime)_)*program_context(?:_.+)?", normalized
    ) is not None or re.fullmatch(r"runtimeprogramcontext(?:v[0-9]+)?", normalized) is not None


def _is_canonical_python_program_facade_step(
    node: ast.Call, path: Path, parents: dict[int, ast.AST]
) -> bool:
    """Permit the one public installed-Program facade, and no generic ``engine.step`` escape.

    ``step_adaptive`` is authoring convenience code whose native engine has already received an
    installed Program.  It is not a second AMR runtime.  Keep its exception structural and exact:
    a same-named helper elsewhere, another method, another receiver, or another argument shape is
    still a bypass.  This is intentionally stricter than a path- or owner-wide allowlist.
    """
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        return False
    if relative != "python/pops/runtime/_cadence_install.py":
        return False
    if (
        not isinstance(node.func, ast.Attribute)
        or not isinstance(node.func.value, ast.Name)
        or node.func.value.id != "engine"
        or node.func.attr != "step"
        or len(node.args) != 1
        or node.keywords
        or not isinstance(node.args[0], ast.Name)
        or node.args[0].id != "dt"
    ):
        return False
    current = parents.get(id(node))
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if current.name != "step_adaptive":
                return False
            if not current.body or not isinstance(current.body[0], ast.Expr):
                return False
            docstring = current.body[0].value
            return (
                isinstance(docstring, ast.Constant)
                and isinstance(docstring.value, str)
                and "through the installed Program" in docstring.value
            )
        current = parents.get(id(current))
    return False


def _python_amr_runtime_step_bypass(source: str, path: Path) -> bool:
    """Find direct Python AMR engine/per-level stepping without matching text in prose."""
    tree = ast.parse(source, filename=str(path))
    parents = {
        id(child): owner
        for owner in ast.walk(tree)
        for child in ast.iter_child_nodes(owner)
    }
    owner_names = set(PYTHON_AMR_RUNTIME_OWNER_NAMES)
    # A direct alias of a recognized fallback runtime is just as authoritative as the original
    # receiver.  Follow only plain name/attribute assignments; arbitrary calls or expressions
    # are intentionally outside this lexical barrier to avoid guessing ownership.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value_name = _dotted_name(node.value).rsplit(".", 1)[-1].lower().rstrip("_")
            if value_name not in owner_names:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if isinstance(target, ast.Name):
                    alias = target.id.lower().rstrip("_")
                    if alias not in owner_names:
                        owner_names.add(alias)
                        changed = True
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = _dotted_name(node.func.value)
        owner_leaf = owner.rsplit(".", 1)[-1].lower().rstrip("_")
        method = node.func.attr.lower()
        if (
            owner_leaf in owner_names
            and method in PYTHON_AMR_RUNTIME_STEP_NAMES
            and not _is_canonical_python_program_facade_step(node, path, parents)
        ):
            return True
    return False


def _cpp_amr_runtime_step_bypass(code: str) -> bool:
    """Find direct AMR stepping, including explicit aliases of ``AmrRuntime`` only."""
    if AMR_RUNTIME_STEP_BYPASS_PATTERN.search(code) is not None:
        return True
    aliases = {
        match.group("alias") or match.group("typedef_alias")
        for match in AMR_RUNTIME_ALIAS_PATTERN.finditer(code)
    }
    return any(
        re.search(
            r"\b%s(?:\s*<[^>{};\n]+>)?\s*::\s*"
            r"(?:step|step_level|advance|advance_level|advance_macro_step)\s*\("
            % re.escape(alias),
            code,
        )
        is not None
        for alias in aliases
    )


def _python_legacy_checkpoint_findings(source: str, path: Path) -> list[str]:
    """Find legacy checkpoint bytes/versions in Python without trusting comments/docstrings."""
    tree = ast.parse(source, filename=str(path))
    docstring_nodes: set[int] = set()
    for owner in ast.walk(tree):
        if not isinstance(owner, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if owner.body and isinstance(owner.body[0], ast.Expr):
            value = owner.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                docstring_nodes.add(id(value))
    findings: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstring_nodes
        ):
            findings.update(
                marker for marker in LEGACY_CHECKPOINT_MARKERS if marker in node.value
            )
        if isinstance(node, (ast.List, ast.Tuple)):
            chars = "".join(
                element.value
                for element in node.elts
                if isinstance(element, ast.Constant)
                and isinstance(element.value, str)
                and len(element.value) == 1
            )
            findings.update(
                marker for marker in LEGACY_CHECKPOINT_MARKERS if marker in chars
            )
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if not isinstance(value, ast.Constant) or not isinstance(value.value, int):
                continue
            if value.value not in {8, 11}:
                continue
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            if any(
                "checkpoint" in name.lower() and "version" in name.lower()
                for name in names
            ):
                findings.add("checkpoint-version-%d" % value.value)
    return sorted(findings)


def _python_legacy_abi_findings(source: str, path: Path) -> list[str]:
    """Find non-v5 Program install ABI assignments in Python production sources."""
    tree = ast.parse(source, filename=str(path))
    findings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, int):
            continue
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if any("programinstallabiversion" in name.lower() for name in names):
            if value.value != 5:
                findings.add("ProgramInstallAbiVersion=%d" % value.value)
    return sorted(findings)


def _python_legacy_cache_wire_findings(source: str, path: Path) -> list[str]:
    """Find old cache wire keys only where Python constructs or writes serialized data.

    ``cache_nodes`` remains a deliberately supported *input* to the fail-closed legacy-shape
    check.  Likewise, ``node_id`` is used throughout the temporal control graph.  Looking for
    either token in the semantic-token stream would therefore reject valid guards and metadata;
    this AST pass limits the barrier to returned/assigned/serialized mappings and cache-owned
    node records.  A node-id-only mapping is considered legacy authority data when its enclosing
    operation is serialization-like or its owner is explicitly cache-specific.
    """
    source_lower = source.lower()
    if "cache_nodes" not in source_lower and "cache_node_id" not in source_lower and not (
        "node_id" in source_lower
        and (
            "cache" in source_lower
            or any(hint in source_lower for hint in PYTHON_SERIALIZATION_NAME_HINTS)
        )
    ):
        return []
    tree = ast.parse(source, filename=str(path))
    parents: dict[int, ast.AST] = {}
    for owner in ast.walk(tree):
        for child in ast.iter_child_nodes(owner):
            parents[id(child)] = owner

    def string_value(node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def target_names(node: ast.AST | None) -> tuple[str, ...]:
        if isinstance(node, ast.Name):
            return (node.id,)
        if isinstance(node, ast.Attribute):
            return (node.attr,)
        if isinstance(node, ast.Subscript):
            return target_names(node.value)
        return ()

    def ancestors(node: ast.AST) -> Iterable[ast.AST]:
        current = parents.get(id(node))
        while current is not None:
            yield current
            current = parents.get(id(current))

    def enclosing_function_names(node: ast.AST) -> tuple[str, ...]:
        return tuple(
            owner.name
            for owner in ancestors(node)
            if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef))
        )

    def has_serialization_name(node: ast.AST) -> bool:
        names: list[str] = list(enclosing_function_names(node))
        for owner in ancestors(node):
            if isinstance(owner, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = owner.targets if isinstance(owner, ast.Assign) else [owner.target]
                if isinstance(owner, ast.AugAssign):
                    targets = [owner.target]
                for target in targets:
                    names.extend(target_names(target))
            elif isinstance(owner, ast.Call):
                call_name = _dotted_name(owner.func).lower()
                leaf = call_name.rsplit(".", 1)[-1]
                if leaf in {"dump", "dumps", "encode", "save", "serialize", "serialise", "write"}:
                    return True
                if "checkpoint" in call_name or "serialize" in call_name or "serialise" in call_name:
                    return True
        lowered = " ".join(names).lower()
        return any(hint in lowered for hint in PYTHON_SERIALIZATION_NAME_HINTS)

    def has_wire_container_name(node: ast.AST) -> bool:
        names: list[str] = []
        for owner in ancestors(node):
            if isinstance(owner, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = owner.targets if isinstance(owner, ast.Assign) else [owner.target]
                if isinstance(owner, ast.AugAssign):
                    targets = [owner.target]
                for target in targets:
                    names.extend(target_names(target))
            elif isinstance(owner, ast.Call) and isinstance(owner.func, ast.Attribute):
                names.extend(target_names(owner.func.value))
        names.extend(enclosing_function_names(node))
        lowered = " ".join(names).lower()
        return any(hint in lowered for hint in PYTHON_WIRE_CONTAINER_NAME_HINTS)

    def is_error_payload(node: ast.AST) -> bool:
        """Allow old-shape values used only as payloads of an intentional rejection."""
        return any(isinstance(owner, ast.Raise) for owner in ancestors(node))

    def is_wire_mapping(node: ast.Dict) -> bool:
        for owner in ancestors(node):
            if isinstance(owner, ast.Return):
                return True
            if isinstance(owner, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = owner.targets if isinstance(owner, ast.Assign) else [owner.target]
                if isinstance(owner, ast.AugAssign):
                    targets = [owner.target]
                if any(target_names(target) for target in targets):
                    return has_wire_container_name(node) or has_serialization_name(node)
            if isinstance(owner, ast.Call):
                call_name = _dotted_name(owner.func).lower()
                leaf = call_name.rsplit(".", 1)[-1]
                if leaf in {"dump", "dumps", "encode", "save", "serialize", "serialise", "write"}:
                    return True
                if leaf == "update":
                    return has_wire_container_name(node)
        return has_serialization_name(node)

    def cache_context(node: ast.AST) -> bool:
        names: list[str] = list(enclosing_function_names(node))
        for owner in ancestors(node):
            if isinstance(owner, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = owner.targets if isinstance(owner, ast.Assign) else [owner.target]
                if isinstance(owner, ast.AugAssign):
                    targets = [owner.target]
                for target in targets:
                    names.extend(target_names(target))
            elif isinstance(owner, ast.Call) and isinstance(owner.func, ast.Attribute):
                names.extend(target_names(owner.func.value))
        # A cache-owned mapping is itself an authority boundary even before it reaches a
        # serializer.  This catches a staged ``{\"node_id\": ...}`` cache plan while leaving
        # ordinary control-graph node metadata and explicit legacy-shape rejection guards alone.
        return "cache" in " ".join(names).lower()

    def mapping_keys(node: ast.Dict) -> set[str]:
        return {
            value
            for key in node.keys
            if (value := string_value(key)) is not None
        }

    legacy_cache_keys = LEGACY_CACHE_WIRE_KEYS - {"node_id"}
    findings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = mapping_keys(node)
            if is_error_payload(node):
                continue
            cache_keys = {key for key in keys if key in legacy_cache_keys}
            # An explicit legacy cache key is itself a wire publication.  Do not require a
            # serializer-looking container name: generated code may assign the mapping to a
            # neutral temporary before handing it to the ABI.  Rejection guards use membership
            # tests and therefore do not enter this branch.
            if cache_keys:
                findings.update(cache_keys)
                if "node_id" in keys:
                    findings.add("node_id")
            elif "node_id" in keys and not legacy_cache_keys.intersection(keys):
                if has_serialization_name(node) or cache_context(node):
                    findings.add("node_id")
        elif isinstance(node, ast.Call):
            call_name = _dotted_name(node.func).lower()
            if call_name.rsplit(".", 1)[-1] == "dict":
                keyword_keys = {
                    keyword.arg
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
                # ``dict(...)`` is a common generated spelling for the JSON/checkpoint payload;
                # an explicit cache_nodes key is sufficient evidence even when the temporary
                # container has an otherwise neutral name.  Keep node_id-only construction scoped
                # to a cache/serialization context just like literal mappings above.
                if is_error_payload(node):
                    continue
                cache_keys = {
                    key for key in keyword_keys if key in legacy_cache_keys
                }
                if cache_keys:
                    findings.update(cache_keys)
                    if "node_id" in keyword_keys:
                        findings.add("node_id")
                elif "node_id" in keyword_keys and (
                    has_serialization_name(node) or cache_context(node)
                ):
                    findings.add("node_id")
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "setdefault":
                if (
                    not is_error_payload(node)
                    and node.args
                    and string_value(node.args[0]) in LEGACY_CACHE_WIRE_KEYS - {"node_id"}
                ):
                    findings.add(string_value(node.args[0]))
                continue
            if node.func.attr == "__setitem__":
                if (
                    not is_error_payload(node)
                    and node.args
                    and string_value(node.args[0]) in LEGACY_CACHE_WIRE_KEYS - {"node_id"}
                ):
                    findings.add(string_value(node.args[0]))
                continue
            if node.func.attr != "update":
                continue
            for argument in node.args:
                if not isinstance(argument, ast.Dict):
                    continue
                keys = mapping_keys(argument)
                if is_error_payload(argument):
                    continue
                findings.update(
                    key for key in keys if key in LEGACY_CACHE_WIRE_KEYS - {"node_id"}
                )
                if "node_id" in keys and (
                    has_serialization_name(argument)
                    or cache_context(argument)
                    or legacy_cache_keys.intersection(keys)
                ):
                    findings.add("node_id")
            for keyword in node.keywords:
                if is_error_payload(node):
                    continue
                if keyword.arg in legacy_cache_keys:
                    findings.add(keyword.arg)
                elif keyword.arg == "node_id" and (
                    has_serialization_name(node)
                    or cache_context(node)
                    or has_wire_container_name(node)
                ):
                    findings.add(keyword.arg)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if isinstance(node, ast.AugAssign):
                targets = [node.target]
            for target in targets:
                if not isinstance(target, ast.Subscript):
                    continue
                key = string_value(target.slice)
                if is_error_payload(node):
                    continue
                if key in legacy_cache_keys:
                    # A subscript assignment is an unambiguous wire write even when the
                    # temporary/container is neutrally named.
                    findings.add(key)
                elif key == "node_id" and (
                    has_serialization_name(target) or cache_context(target)
                ):
                    findings.add(key)
    return sorted(findings)


def _legacy_checkpoint_findings(
    text: str, path: Path, semantic: str
) -> list[str]:
    findings = [marker for marker in LEGACY_CHECKPOINT_MARKERS if marker in semantic]
    if path.suffix in CXX_SUFFIXES:
        findings.extend(
            "encoded:%s" % marker
            for marker, pattern in LEGACY_CHECKPOINT_CHAR_ARRAY_PATTERNS.items()
            if pattern.search(semantic) is not None
        )
        if any(pattern.search(semantic) is not None for pattern in LEGACY_CHECKPOINT_VERSION_PATTERNS):
            findings.append("checkpoint-version-8-or-11")
    elif path.suffix == ".py":
        findings.extend(_python_legacy_checkpoint_findings(text, path))
    return sorted(set(findings))


def _scan_production_barriers(
    errors: list[str], source_paths: Iterable[Path], *, reject_any_pending: bool = False
) -> None:
    """Scan production source roots with syntax-aware views and explicit retired spellings."""
    for path in sorted(set(source_paths)):
        if not path.is_file():
            continue
        display = _display_source_path(path)
        if path.suffix == ".inc":
            errors.append("runtime authority forbids AMR runtime fragments (.inc): %s" % display)
        if _is_retired_program_context_filename(path):
            errors.append("retired ProgramContext source remains present: %s" % display)

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append("cannot read production authority source %s: %s" % (display, exc))
            continue

        if path.suffix in CXX_SUFFIXES:
            code = _cpp_code_only(text)
            semantic = _cpp_without_comments(text)
            semantic_tokens: tuple[str, ...] = ()
        elif path.suffix == ".py":
            try:
                tokens = _python_semantic_tokens(text, path)
            except (OSError, SyntaxError) as exc:
                errors.append("cannot parse production authority source %s: %s" % (display, exc))
                continue
            code = "\n".join(tokens)
            semantic = code
            semantic_tokens = tokens
        else:
            continue

        contexts = [
            token
            for token in LEGACY_CONTEXT_TOKENS
            if re.search(r"\b%s\b" % re.escape(token), semantic)
        ]
        symbols = [
            token
            for token in (
                *LEGACY_SYMBOL_TOKENS,
                *LEGACY_PROGRAM_AMR_SYMBOL_TOKENS,
                *LEGACY_PROGRAM_SPLIT_SYMBOL_TOKENS,
            )
            if re.search(r"\b%s\b" % re.escape(token), semantic)
        ]
        symbols.extend(LEGACY_PROGRAM_SPLIT_ABI_PATTERN.findall(semantic))
        symbols.extend(LEGACY_PROGRAM_SPLIT_AMR_VARIANT_PATTERN.findall(semantic))
        symbols.extend(LEGACY_PROGRAM_GENERIC_AMR_PATTERN.findall(semantic))
        symbols.extend(LEGACY_PROGRAM_QUALIFIED_INSTALL_ABI_PATTERN.findall(semantic))
        symbols.extend(LEGACY_PROGRAM_INSTALL_TARGET_VARIANT_PATTERN.findall(semantic))
        symbols = [symbol for symbol in symbols if symbol not in NATIVE_BRICK_ABI_SYMBOLS]
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
        checkpoint = _legacy_checkpoint_findings(text, path, semantic)
        cell_local_installers = [
            match.group(0).strip()
            for match in CELL_LOCAL_INSTALLER_PATTERN.finditer(code)
        ]
        if path.suffix == ".py":
            cell_local_installers.extend(
                token
                for token in semantic_tokens
                if CELL_LOCAL_INSTALLER_NAME_PATTERN.fullmatch(token) is not None
            )
        program_amr_installers = [
            match.group(0).strip()
            for match in LEGACY_PROGRAM_AMR_INSTALLER_PATTERN.finditer(code)
        ]
        if path.suffix == ".py":
            program_amr_installers.extend(
                token
                for token in semantic_tokens
                if LEGACY_PROGRAM_AMR_INSTALLER_NAME_PATTERN.fullmatch(token) is not None
            )
        parallel_tables = [
            match.group(0)
            for match in PARALLEL_AUTHORITY_TABLE_PATTERN.finditer(code)
        ]
        parallel_dispatches = [
            match.group(0)
            for match in PARALLEL_AUTHORITY_DISPATCH_PATTERN.finditer(code)
        ]
        program_tables = [
            match.group(0)
            for match in PROGRAM_AUTHORITY_TABLE_PATTERN.finditer(code)
        ]
        program_dispatches = [
            match.group(0)
            for match in PROGRAM_AUTHORITY_DISPATCH_PATTERN.finditer(code)
        ]
        if path.suffix == ".py":
            parallel_tables.extend(
                token
                for token in semantic_tokens
                if PARALLEL_AUTHORITY_TABLE_NAME_PATTERN.fullmatch(token) is not None
            )
            parallel_dispatches.extend(
                token
                for token in semantic_tokens
                if PARALLEL_AUTHORITY_DISPATCH_NAME_PATTERN.fullmatch(token) is not None
            )
            program_tables.extend(
                token
                for token in semantic_tokens
                if PROGRAM_AUTHORITY_TABLE_NAME_PATTERN.fullmatch(token) is not None
            )
            program_dispatches.extend(
                token
                for token in semantic_tokens
                if PROGRAM_AUTHORITY_DISPATCH_NAME_PATTERN.fullmatch(token) is not None
            )
        program_tables = [
            token
            for token in program_tables
            if token not in CANONICAL_CANDIDATE_AUTHORITY_NAMES
        ]
        program_dispatches = [
            token
            for token in program_dispatches
            if token not in CANONICAL_CANDIDATE_AUTHORITY_NAMES
        ]
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
        if cell_local_installers:
            errors.append(
                "%s retains forbidden cell-local installer route(s) %s"
                % (display, sorted(set(cell_local_installers)))
            )
        if program_amr_installers:
            errors.append(
                "%s retains retired Program AMR installer route(s) %s"
                % (display, sorted(set(program_amr_installers)))
            )
        if parallel_tables:
            errors.append(
                "%s retains a parallel runtime authority table (%s)"
                % (display, sorted(set(parallel_tables)))
            )
        if parallel_dispatches:
            errors.append(
                "%s retains a parallel runtime authority dispatch (%s)"
                % (display, sorted(set(parallel_dispatches)))
            )
        if program_tables:
            errors.append(
                "%s retains a secondary Program runtime authority table (%s)"
                % (display, sorted(set(program_tables)))
            )
        if program_dispatches:
            errors.append(
                "%s retains a secondary Program runtime authority dispatch (%s)"
                % (display, sorted(set(program_dispatches)))
            )
        if LEGACY_ABI_VERSION_PATTERN.search(code) is not None:
            errors.append("%s retains a legacy Program install ABI version" % display)
        if path.suffix == ".py" and _python_legacy_abi_findings(text, path):
            errors.append("%s retains a legacy Program install ABI version" % display)
        amr_step_bypass = (
            _python_amr_runtime_step_bypass(text, path)
            if path.suffix == ".py"
            else _cpp_amr_runtime_step_bypass(code)
        )
        if amr_step_bypass:
            errors.append(
                "%s bypasses the Program authority through AmrRuntime::step or a direct "
                "AMR engine/per-level step" % display
            )
        if CACHE_NODE_ID_ONLY_PATTERN.search(code) is not None:
            errors.append("%s retains a node-id-only scheduler cache" % display)
        if path.suffix == ".py":
            cache_wire = _python_legacy_cache_wire_findings(text, path)
            if cache_wire:
                errors.append(
                    "%s retains a node-id-only scheduler cache wire (%s)"
                    % (display, cache_wire)
                )


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
    production_sources = tuple(path for path in _production_sources() if path.is_file())
    runtime_sources = tuple(sorted(set((*authority_sources, *production_sources))))
    abi_sources = tuple(path for path in runtime_sources if path.suffix in CXX_SUFFIXES)
    if sum(
        len(ABI_VERSION_PATTERN.findall(_cpp_code_only(path.read_text(encoding="utf-8"))))
        for path in abi_sources
    ) != 1:
        errors.append(
            "runtime authority must expose exactly one ABI version declaration (v5) "
            "across all runtime production roots"
        )
    _scan_production_barriers(errors, authority_sources, reject_any_pending=True)
    authority_set = set(authority_sources)
    _scan_production_barriers(
        errors,
        (path for path in production_sources if path not in authority_set),
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


def _validate_row_dimensions(
    value: object, where: str, errors: list[str]
) -> tuple[int, ...] | None:
    """Validate one optional row dimension qualifier and return its canonical values."""
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
    _validate_required_semantic_barrier_rows(checks, errors)

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
    environment["POPS_EXACT_PROCESS_NODEIDS"] = "1"
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
    nodeid: str,
    *,
    dimension: int,
) -> list[str]:
    """Build an MPI command that invokes exactly one manifest-owned file::function nodeid."""
    dimension = _validated_native_dimension(dimension)
    if not isinstance(nodeid, str) or FULL_NODEID.fullmatch(nodeid) is None:
        raise ValueError("MPI Python execution requires an exact file::function nodeid")
    relative, function_name = nodeid.split("::", 1)
    candidate = (ROOT / relative).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("MPI Python nodeid must stay inside the repository") from exc
    if candidate.suffix != ".py":
        raise ValueError("MPI Python nodeid must select a Python source file")
    if shutil.which(mpi_exec) is None:
        raise RuntimeError("required MPI launcher %r is unavailable" % mpi_exec)
    bootstrap = (
        "from pops._native_selector import select_native_dimension\n"
        "from pathlib import Path\n"
        "import runpy, sys\n"
        "select_native_dimension(int(sys.argv[1]))\n"
        "nodeid = sys.argv[3]\n"
        "relative, separator, function_name = nodeid.partition('::')\n"
        "expected_relative = Path(sys.argv[2]).resolve().relative_to(Path.cwd()).as_posix()\n"
        "if not separator or function_name != %r or relative != expected_relative:\n"
        "    raise SystemExit('MPI Python nodeid/path mismatch: ' + nodeid)\n"
        "module = runpy.run_path(sys.argv[2], run_name='__runtime_authority_node__')\n"
        "test = module.get(function_name)\n"
        "if not callable(test):\n"
        "    raise SystemExit('MPI Python nodeid does not resolve to a function: ' + nodeid)\n"
        "test()\n"
        "raise SystemExit(int(module.get('_fails', 0)))\n" % function_name
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
        nodeid,
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


def _required_ctest_targets(
    checks: Iterable[dict], *, backend: str = "mpi", dimension: int
) -> tuple[str, ...]:
    selected = _selected_checks(checks, backend=backend, dimension=dimension)
    return tuple(
        sorted(
            {
                row["target"].split("@", 1)[1]
                for row in selected
                if row["kind"] == "ctest"
            }
        )
    )


def _selected_checks(
    checks: Iterable[dict], *, backend: str, dimension: int
) -> list[dict]:
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
        if row.get("backend") == backend and supports_dimension(row)
    ]


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
    dimension = _validated_native_dimension(dimension)
    composed_environment = dict(environment)
    composed_environment["POPS_NATIVE_DIM"] = str(dimension)
    with _temporary_environment(composed_environment):
        if ctest_only:
            data, errors = module.validate_manifest(manifest)
            if errors:
                raise RuntimeError(
                    "%s gate manifest is incomplete or invalid: %s"
                    % (relative, "; ".join(errors))
                )
            rows = sorted(
                (
                    row
                    for row in data["check"]
                    if row["kind"] == "ctest"
                    and (
                        "dimensions" not in row
                        or dimension in row["dimensions"]
                    )
                ),
                key=lambda value: (value["target"], value["test_regex"]),
            )
            for row in rows:
                module._run_ctest(
                    build_dir,
                    row["target"],
                    row["test_regex"],
                )
            return

        # M2 inherits the authenticated environment; M3 also receives an explicit dimension so
        # its row filtering remains deterministic when called as a composed runner.
        argv = ["--manifest", str(manifest), "--build-dir", str(build_dir)]
        if Path(runner_relative).name == "run_m3_gate.py":
            argv.extend(("--dim", str(dimension), "--mpi-exec", mpi_exec))
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
    parser.add_argument("--dim", type=int, choices=SUPPORTED_NATIVE_DIMENSIONS)
    parser.add_argument("--mpi-exec", default="mpiexec")
    parser.add_argument("--backend", choices=sorted(EXPECTED_BACKENDS), default="serial")
    parser.add_argument("--openmp", action="store_true", help="select authenticated OpenMP rows")
    args = parser.parse_args(argv)
    if args.python_only and args.ctest_only:
        parser.error("--python-only and --ctest-only are mutually exclusive")
    backend = "openmp" if args.openmp else args.backend
    if args.openmp and args.backend != "serial":
        parser.error("--openmp cannot be combined with --backend")

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
    if args.check_only or args.audit_only:
        print(
            "Runtime authority gate source matrix: %s (%d executable, %d deferred; "
            "backend=%s, selected=all)"
            % (
                "AUDITED CLOSED" if args.audit_only else "CLOSED",
                len(checks),
                len(data["deferred"]),
                backend,
            )
        )
        return 0

    try:
        dimension = _selected_native_dimension(args.dim)
    except ValueError as exc:
        parser.error(str(exc))

    if args.list_ctest_targets:
        targets = _required_ctest_targets(checks, backend=backend, dimension=dimension)
        if not targets:
            print("runtime authority gate selects no CTest build target", file=sys.stderr)
            return 2
        print("\n".join(targets))
        return 0

    selected = _selected_checks(checks, backend=backend, dimension=dimension)
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
            _run(
                _mpi_python_command(
                    args.mpi_exec,
                    row["nproc"],
                    row["nodeid"],
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
