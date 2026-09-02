"""Explicit native Programs for low-level Python runtime tests.

These helpers are deliberately low-level runtime fixtures, not Program-authoring evidence or a
runtime compatibility layer.  A helper builds an ABI-v5 ``problem.so`` with an explicit block
identity table, prepares a real ``ProgramExecutionServices`` owner, and only then lets a test
exercise a spatial/runtime seam.

The forward-Euler bridge is the only such seam appropriate for a spatial/runtime test that does not
need a temporal method.  Projection and coupled-source splitting are opt-in by block
identity/registration; semantic time-method tests use typed symbolic Programs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from pops.codegen.cache import (
    _artifact_cache_lock,
    _artifact_cache_staging_path,
    _dsl_optflags,
    pops_cache_dir,
)
from pops.codegen.compile_link_flags import deterministic_program_link_flags
from pops.codegen.toolchain import (
    _check_headers_match_module,
    _probe_cxx_std,
    _run_compile,
    loader_cxx_std,
    pops_include,
    pops_loader_build_flags,
)
from pops.runtime._system import AmrSystem, System


def _coupling_application(block_count: int, enabled: bool, *, target: str) -> str:
    if not enabled:
        return ""
    candidates = ["        {%d, &next_%d}" % (block, block) for block in range(block_count)]
    arguments = "dt"
    if target == "amr_system":
        arguments = "%s, %s, %s, dt" % (
            json.dumps("pops.test.euler.coupled-source.graph/final"),
            json.dumps("pops.test.euler.coupled-source.rate/final"),
            json.dumps("pops.test.euler.coupled-source.application/final"),
        )
    return "      ctx.apply_coupling_operators(%s, {\n%s\n      });" % (
        arguments,
        ",\n".join(candidates),
    )


def _forward_euler_body(
    block_count: int,
    projection_indices: tuple[int, ...],
    *,
    apply_couplings: bool,
    target: str,
) -> str:
    declarations = []
    requests = []
    combinations = []
    projections = []
    commits = []
    for block in range(block_count):
        declarations.extend(
            (
                "      pops::MultiFab<pops::kNativeDimension>& state_%d = ctx.state(%d);"
                % (block, block),
                "      pops::MultiFab<pops::kNativeDimension>& rhs_%d = "
                "ctx.rhs_scratch(%d, 0, state_%d);" % (block, 1000 + block, block),
            )
        )
        requests.append(
            "        {%d, &state_%d, &rhs_%d, %d, 0}" % (block, block, block, 3000 + block)
        )
        combinations.extend(
            (
                "      pops::MultiFab<pops::kNativeDimension>& next_%d = "
                "ctx.scratch_state(%d, 0, state_%d);" % (block, 2000 + block, block),
                "      ctx.lincomb(next_%d, pops::Real(1), state_%d, dt, rhs_%d, dt, "
                "{{0, 1, 1}}, {{1, 1, 1}});" % (block, block, block),
            )
        )
        commits.append("        {&state_%d, &next_%d}" % (block, block))
    for block in projection_indices:
        projections.append("      ctx.apply_projection(%d, next_%d);" % (block, block))
    return "\n".join(
        (
            "      ctx.set_stage_time(0, 1);",
            "      (void)pops::consume_solve_outcome(ctx.solve_fields());",
            *declarations,
            "      ctx.rhs_group(4000, {\n%s\n      });" % ",\n".join(requests),
            *combinations,
            _coupling_application(block_count, apply_couplings, target=target),
            *projections,
            "      ctx.commit_many({\n%s\n      });" % ",\n".join(commits),
        )
    )


def _make_test_field_solve_explicitly_coarse(body: str) -> str:
    """Qualify the legacy test-only AMR field cadence without claiming a fine solve."""
    generic = "      (void)pops::consume_solve_outcome(ctx.solve_fields());"
    if body.count(generic) != 1:
        raise AssertionError("explicit test Program must contain one default field solve")
    return body.replace(
        generic,
        "      if (ctx.level() == 0)\n"
        "        (void)pops::consume_solve_outcome(\n"
        "            ctx.solve_default_field_on_coarse_level());",
    )


def _source(
    *,
    target: str,
    block_names: tuple[str, ...],
    projection_indices: tuple[int, ...],
    coupled_sources: bool,
    identity: str,
) -> str:
    body = _forward_euler_body(
        len(block_names),
        projection_indices,
        apply_couplings=coupled_sources,
        target=target,
    )
    if target == "amr_system":
        body = _make_test_field_solve_explicitly_coarse(body)
    from pops.runtime.routes import route_registry_signature

    block_strings = "\n".join(
        'static constexpr char kBlockName%d[] = %s;' % (index, json.dumps(name))
        for index, name in enumerate(block_names)
    )
    block_rows = ",\n    ".join(
        '{{kBlockName%d, static_cast<std::uint64_t>(sizeof(kBlockName%d) - 1)}}' % (index, index)
        for index in range(len(block_names))
    )
    coupled_identity_size = 0
    if coupled_sources:
        coupled_identity_size = sum(
            len(value)
            for value in (
                "pops.test.euler.coupled-source.graph/final",
                "pops.test.euler.coupled-source.rate/final",
                "pops.test.euler.coupled-source.application/final",
            )
        )
    flux_table = ""
    if target == "amr_system":
        flux_rows = ",\n    ".join(
            "{UINT64_C(1), UINT64_C(1), UINT64_C(%d), UINT64_C(%d)}"
            % (1 if coupled_sources else 0, coupled_identity_size)
            for _ in block_names
        )
        flux_table = """
static const ProgramFluxBudgetRecord kFluxBudgets[] = {
    %s
};
""" % flux_rows
    candidate_capabilities = (
        "kProgramCapabilityHierarchy | kProgramCapabilitySchedules | "
        "kProgramCapabilityCellTemporal | kProgramCapabilityPersistentValues | "
        "kProgramCapabilityTransactions"
        if target == "amr_system"
        else "kProgramCapabilitySchedules | kProgramCapabilityPersistentValues | kProgramCapabilityTransactions"
    )
    candidate_services = (
        "kProgramServiceState | kProgramServiceFields | kProgramServiceSpatial | "
        "kProgramServiceHierarchy | kProgramServiceHistory | kProgramServiceClock | "
        "kProgramServiceReduction | kProgramServiceTransaction | kProgramServicePersistentValues"
        if target == "amr_system"
        else "kProgramServiceState | kProgramServiceFields | kProgramServiceSpatial | "
        "kProgramServiceHistory | kProgramServiceClock | kProgramServiceReduction | "
        "kProgramServiceTransaction | kProgramServicePersistentValues"
    )
    if target == "system":
        step_builder = """\
    state->step = [ctx_owner = state->ctx_owner](double dt) {
      auto& ctx = *ctx_owner;
      ctx.begin_step(dt);
%s
    };""" % body
        lifecycle = ""
    else:
        step_builder = """\
    state->step = [ctx_owner = state->ctx_owner](double macro_dt) {
      ctx_owner->advance_hierarchy(macro_dt, [ctx_owner](double dt) {
        auto& ctx = *ctx_owner;
%s
      });
    };""" % body
        lifecycle = """
    descriptor.hierarchy_refresh = &candidate_hierarchy_refresh;
    descriptor.history_remap_accepted = &candidate_history_remap;
    descriptor.restart_regrid_preflight = &candidate_restart_preflight;
    descriptor.restart_regrid = &candidate_restart_regrid;
    descriptor.restart_resync = &candidate_restart_resync;
    descriptor.create_accepted_snapshot = &candidate_accepted_snapshot;"""
    return """\
#if !defined(POPS_RUNTIME_SHARED_EXCEPTION_ABI)
#error "test Programs require the shared runtime exception ABI consumer contract"
#endif
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/program/program_abi.hpp>
#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>

namespace {
using namespace pops::runtime::program;

void write_diagnostic(ProgramInstallDiagnostic* diagnostic, ProgramInstallErrorCode code,
                      const char* message) noexcept {
  if (diagnostic == nullptr) return;
  diagnostic->code = code;
  std::size_t size = 0;
  if (message != nullptr)
    for (; size + 1 < sizeof(diagnostic->message) && message[size] != '\\0'; ++size)
      diagnostic->message[size] = message[size];
  diagnostic->message[size] = '\\0';
}

struct CandidateState final {
  std::shared_ptr<ProgramExecutionServices<pops::kNativeDimension>> ctx_owner;
  std::function<void(double)> step;
};

void candidate_step(void* opaque, double dt) {
  static_cast<CandidateState*>(opaque)->step(dt);
}
void candidate_destroy(void* opaque) noexcept { delete static_cast<CandidateState*>(opaque); }
%s
%s
static const ProgramBlockRecord kBlocks[] = {
    %s
};
%s
static constexpr char kArtifactIdentity[] = %s;
static constexpr char kProgramName[] = "pops.test.euler";
static constexpr char kAbiKey[] = POPS_ABI_KEY_LITERAL;
static constexpr char kRouteManifest[] = %s;
static constexpr char kBoundaryManifest[] = "pops.boundary.manifest.v1";
static constexpr char kResourceManifest[] = "pops.persistent-resource.manifest.v1";
static constexpr char kCheckpointIdentity[] = "pops.checkpoint.identity.v1";

bool candidate_prepare(void* opaque, const ProgramHostDescriptor* host,
                       ProgramInstallDiagnostic* diagnostic) noexcept {
  if (opaque == nullptr || host == nullptr || diagnostic == nullptr ||
      !valid_program_host_descriptor(*host) ||
      host->native_dimension != static_cast<std::uint32_t>(pops::kNativeDimension) ||
      host->runtime_kind != ProgramRuntimeKind::%s ||
      host->execution_lane != ProgramExecutionLane::host) {
    write_diagnostic(diagnostic, ProgramInstallErrorCode::invalid_host_descriptor,
                     "explicit test Program received an incompatible host");
    return false;
  }
  auto* state = static_cast<CandidateState*>(opaque);
  if (state->ctx_owner || state->step) {
    write_diagnostic(diagnostic, ProgramInstallErrorCode::artifact_rejected,
                     "explicit test Program candidate was prepared twice");
    return false;
  }
  try {
    state->ctx_owner = make_program_execution_provider<pops::kNativeDimension>(host->preparation);
    state->ctx_owner->configure_primary_clock("pops.test.clock.macro");
%s
    return true;
  } catch (...) {
    write_diagnostic(diagnostic, ProgramInstallErrorCode::artifact_rejected,
                     "explicit test Program preparation failed");
    return false;
  }
}

}  // namespace

extern "C" pops::runtime::program::ProgramInstallAbiProbe
pops_program_install_abi_probe_v5() noexcept {
  return pops::runtime::program::make_program_install_abi_probe();
}

extern "C" bool pops_install_program(const ProgramHostDescriptor* host,
                                      ProgramCandidateDescriptor* candidate,
                                      ProgramInstallDiagnostic* diagnostic) noexcept {
  if (host == nullptr || candidate == nullptr || diagnostic == nullptr ||
      !valid_program_host_descriptor(*host) ||
      host->native_dimension != static_cast<std::uint32_t>(pops::kNativeDimension) ||
      host->runtime_kind != ProgramRuntimeKind::%s ||
      host->execution_lane != ProgramExecutionLane::host) {
    write_diagnostic(diagnostic, ProgramInstallErrorCode::invalid_host_descriptor,
                     "explicit test Program received an incompatible host");
    return false;
  }
  *candidate = {};
  try {
    auto state = std::make_unique<CandidateState>();
    ProgramCandidateDescriptor descriptor{};
    descriptor.struct_size = static_cast<std::uint32_t>(sizeof(ProgramCandidateDescriptor));
    descriptor.abi_version = kProgramInstallAbiVersion;
    descriptor.native_dimension = static_cast<std::uint32_t>(pops::kNativeDimension);
    descriptor.runtime_kind = ProgramRuntimeKind::%s;
    descriptor.provided_capability_bits = %s;
    descriptor.required_capability_bits = %s;
    descriptor.required_service_bits = %s;
    descriptor.program_name = {kProgramName, sizeof(kProgramName) - 1};
    descriptor.artifact_identity = {kArtifactIdentity, sizeof(kArtifactIdentity) - 1};
    descriptor.abi_key = {kAbiKey, sizeof(kAbiKey) - 1};
    descriptor.route_manifest = {kRouteManifest, sizeof(kRouteManifest) - 1};
    descriptor.boundary_manifest = {kBoundaryManifest, sizeof(kBoundaryManifest) - 1};
    descriptor.persistent_resource_manifest = {kResourceManifest, sizeof(kResourceManifest) - 1};
    descriptor.checkpoint_identity = {kCheckpointIdentity, sizeof(kCheckpointIdentity) - 1};
    descriptor.blocks = {kBlocks, %d, sizeof(ProgramBlockRecord)};
%s
    descriptor.maximum_bytes = sizeof(CandidateState);
    descriptor.context = state.get();
    descriptor.prepare = &candidate_prepare;
    descriptor.step = &candidate_step;
    descriptor.destroy = &candidate_destroy;
%s
    if (!valid_program_candidate_descriptor(descriptor)) {
      write_diagnostic(diagnostic, ProgramInstallErrorCode::invalid_candidate,
                       "explicit test Program produced an invalid candidate");
      return false;
    }
    *candidate = descriptor;
    (void)state.release();
    return true;
  } catch (...) {
    write_diagnostic(diagnostic, ProgramInstallErrorCode::artifact_rejected,
                     "explicit test Program candidate construction failed");
    return false;
  }
}
""" % (
        "" if target == "system" else """\
void candidate_hierarchy_refresh(void* opaque) {
  static_cast<CandidateState*>(opaque)->ctx_owner->refresh_accepted_hierarchy({});
}
void candidate_history_remap(void* opaque, const void* descriptor) {
  if (descriptor == nullptr) throw std::invalid_argument("null history remap");
  static_cast<CandidateState*>(opaque)->ctx_owner->accept_history_remap(
      *static_cast<const AmrProgramHistoryRemapDescriptor*>(descriptor));
}
void candidate_restart_preflight(void* opaque) {
  static_cast<CandidateState*>(opaque)->ctx_owner->preflight_restart_regrid();
}
void candidate_restart_regrid(void* opaque) {
  static_cast<CandidateState*>(opaque)->ctx_owner->restart_regrid();
}
void candidate_restart_resync(void* opaque) {
  static_cast<CandidateState*>(opaque)->ctx_owner->resync_after_restart();
}
AcceptedProgramExecutionServicesSnapshot* candidate_accepted_snapshot(void* opaque) {
  return static_cast<CandidateState*>(opaque)->ctx_owner->create_accepted_context_snapshot().release();
}
""",
        block_strings,
        block_rows,
        flux_table,
        json.dumps(identity),
        json.dumps(route_registry_signature()),
        "amr" if target == "amr_system" else "uniform",
        step_builder,
        "amr" if target == "amr_system" else "uniform",
        "amr" if target == "amr_system" else "uniform",
        candidate_capabilities,
        candidate_capabilities,
        candidate_services,
        len(block_names),
        "    descriptor.flux_budgets = {kFluxBudgets, %d, sizeof(ProgramFluxBudgetRecord)};"
        % len(block_names)
        if target == "amr_system"
        else "",
        lifecycle,
    )


def _compile_program(source: str) -> str:
    include = pops_include()
    header_signature = _check_headers_match_module(include)
    compiler, compile_flags, link_flags = pops_loader_build_flags()
    standard = _probe_cxx_std(compiler, loader_cxx_std())
    link_flags = deterministic_program_link_flags(link_flags)
    identity_material = json.dumps(
        {
            "source": source,
            "compiler": compiler,
            "standard": standard,
            "header_signature": header_signature,
            "compile_flags": compile_flags,
            "link_flags": link_flags,
            "optflags": _dsl_optflags(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(identity_material).hexdigest()
    cache = Path(pops_cache_dir()) / "test-programs"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / ("%s.so" % digest)
    with _artifact_cache_lock(destination):
        if destination.exists():
            return str(destination)
        staging = Path(_artifact_cache_staging_path(destination))
        try:
            with tempfile.TemporaryDirectory(prefix="pops-test-program-") as tmp:
                cpp = Path(tmp) / "program.cpp"
                cpp.write_text(source, encoding="utf-8")
                if sys.platform == "win32":  # pragma: no cover - production compiler is POSIX-only
                    raise RuntimeError("explicit Python test Programs are unavailable on Windows")
                command = [
                    compiler,
                    "-shared",
                    "-fPIC",
                    "-std=" + standard,
                    *_dsl_optflags(),
                    '-DPOPS_HEADER_SIG="%s"' % header_signature,
                    *compile_flags,
                    "-I",
                    include,
                    str(cpp),
                    "-o",
                    str(staging),
                    *link_flags,
                ]
                _run_compile(command, "explicit Python test Program")
            os.replace(staging, destination)
        finally:
            try:
                staging.unlink()
            except FileNotFoundError:
                pass
    return str(destination)


def _install_forward_euler_bridge(
    runtime: Any,
    *,
    project_blocks: tuple[str, ...] = (),
    coupled_sources: bool = False,
) -> str:
    if not isinstance(runtime, (System, AmrSystem)):
        raise TypeError("runtime must be a pops.runtime System or AmrSystem")
    block_names = tuple(runtime.block_names())
    if not block_names or any(not isinstance(name, str) or not name for name in block_names):
        raise ValueError("an explicit test Program requires non-empty runtime block identities")
    if len(set(block_names)) != len(block_names):
        raise ValueError("runtime block identities must be unique")
    unknown = tuple(name for name in project_blocks if name not in block_names)
    if unknown:
        raise ValueError("projection names unknown runtime blocks: %s" % (unknown,))
    projection_indices = tuple(block_names.index(name) for name in project_blocks)
    target = "amr_system" if isinstance(runtime, AmrSystem) else "system"
    seed = json.dumps(
        {
            "schema": "pops-test-explicit-program-v2",
            "method": "euler",
            "target": target,
            "blocks": block_names,
            "project": projection_indices,
            "coupled_sources": coupled_sources,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    identity = hashlib.sha256(seed.encode()).hexdigest()
    library = _compile_program(
        _source(
            target=target,
            block_names=block_names,
            projection_indices=projection_indices,
            coupled_sources=coupled_sources,
            identity=identity,
        )
    )
    runtime.install_program(library)
    return library


def install_forward_euler_program(
    runtime: Any,
    *,
    project_blocks: tuple[str, ...] = (),
    coupled_sources: bool = False,
) -> str:
    """Install an explicit, operator-free forward-Euler Program on ``runtime``.

    ``project_blocks`` and ``coupled_sources`` are explicit because projection and source splitting
    are Program nodes, not hidden block-time policies.
    """
    return _install_forward_euler_bridge(
        runtime,
        project_blocks=project_blocks,
        coupled_sources=coupled_sources,
    )


__all__ = [
    "install_forward_euler_program",
]
