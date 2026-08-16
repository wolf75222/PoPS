"""Explicit native Programs for low-level Python runtime tests.

These helpers are deliberately low-level runtime fixtures, not Program-authoring evidence or a
runtime compatibility layer.  A helper builds an ABI-checked ``problem.so`` with an explicit block
identity table, installs a real ``ProgramContext`` or ``AmrProgramContext``, and only then lets a
test exercise a spatial/runtime seam.

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

from pops.codegen._compile_emit import _emit_route_manifest
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


def _block_identity_exports(block_names: tuple[str, ...]) -> str:
    cases = "".join(
        "    case %d: return %s;\n" % (index, json.dumps(name))
        for index, name in enumerate(block_names)
    )
    return (
        'extern "C" int pops_program_block_count() { return %d; }\n'
        'extern "C" const char* pops_program_block_name(int index) {\n'
        "  switch (index) {\n"
        "%s"
        '    default: return "";\n'
        "  }\n"
        "}\n"
    ) % (len(block_names), cases)


def _empty_module_metadata_exports() -> str:
    """Emit the mandatory authenticated metadata families for an operator-free bridge Program."""
    string_exports = "".join(
        'extern "C" const char* pops_module_%s(int) { return ""; }\n' % name
        for name in (
            "operator_name",
            "operator_kind",
            "operator_signature",
            "operator_requirements",
            "operator_owner",
            "state_space_name",
            "state_space_owner",
            "field_space_name",
            "field_space_owner",
        )
    )
    return (
        'extern "C" int pops_module_operator_count() { return 0; }\n'
        'extern "C" int pops_module_state_space_count() { return 0; }\n'
        'extern "C" int pops_module_field_space_count() { return 0; }\n' + string_exports
    )


def _coupling_application(block_count: int, enabled: bool, *, target: str) -> str:
    if not enabled:
        return ""
    candidates = ["        {%d, &next_%d}" % (block, block) for block in range(block_count)]
    arguments = "dt"
    if target == "amr_system":
        arguments = "pops_program_hash(), %s, %s, dt" % (
            json.dumps("pops.test.euler.coupled-source.rate/final"),
            json.dumps("pops.test.euler.coupled-source.application/final"),
        )
    return "      ctx.apply_coupling_operators(%s, {\n%s\n      });" % (
        arguments,
        ",\n".join(candidates),
    )


def _amr_budget_exports(block_count: int, coupled_sources: bool, identity: str) -> str:
    rate_identity = "pops.test.euler.coupled-source.rate/final"
    application_identity = "pops.test.euler.coupled-source.application/final"
    coupling_identity_characters = (
        len(identity) + len(rate_identity) + len(application_identity) if coupled_sources else 0
    )
    return """\
extern "C" bool pops_program_has_flux_expression() { return true; }
extern "C" int pops_program_flux_expression_budget_count() { return %d; }
extern "C" std::uint64_t pops_program_flux_rhs_basis_bound(int block) {
  return block >= 0 && block < %d ? UINT64_C(%d) : UINT64_C(0);
}
extern "C" std::uint64_t pops_program_flux_coefficient_term_bound(int block) {
  return block >= 0 && block < %d ? UINT64_C(1) : UINT64_C(0);
}
extern "C" std::uint64_t pops_program_interface_coupling_application_bound() {
  return UINT64_C(%d);
}
extern "C" std::uint64_t pops_program_interface_coupling_identity_character_bound() {
  return UINT64_C(%d);
}
extern "C" int pops_program_checkpoint_history_count() { return 0; }
extern "C" const char* pops_program_checkpoint_history_name(int) { return ""; }
extern "C" int pops_program_checkpoint_history_owner(int) { return 0; }
extern "C" const char* pops_program_checkpoint_history_state_identity(int) { return ""; }
extern "C" const char* pops_program_checkpoint_history_space_identity(int) { return ""; }
extern "C" const char* pops_program_checkpoint_history_clock_identity(int) { return ""; }
extern "C" const char* pops_program_checkpoint_history_interpolation_identity(int) { return ""; }
extern "C" int pops_program_checkpoint_history_depth(int) { return 0; }
extern "C" int pops_program_checkpoint_history_components(int) { return 0; }
extern "C" int pops_program_checkpoint_logical_clock_count() { return 1; }
extern "C" const char* pops_program_checkpoint_logical_clock_identity(int clock) {
  return clock == 0 ? "pops.test.clock.macro" : "";
}
extern "C" const char* pops_program_checkpoint_temporal_provider_identity() {
  return "pops.temporal-partition.global@1";
}
extern "C" std::uint64_t pops_program_checkpoint_temporal_cell_capacity() {
  return UINT64_C(0);
}
extern "C" std::uint64_t pops_program_checkpoint_temporal_cells_per_topology_cell() {
  return UINT64_C(0);
}
""" % (
        block_count,
        block_count,
        1,
        block_count,
        1 if coupled_sources else 0,
        coupling_identity_characters,
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
    common = """\
#if !defined(POPS_RUNTIME_SHARED_EXCEPTION_ABI)
#error "test Programs require the shared runtime exception ABI consumer contract"
#endif
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/program/program_context.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <cstdint>
#include <limits>
#include <memory>

extern "C" const char* pops_program_abi_key() { return POPS_ABI_KEY_LITERAL; }
%s
extern "C" const char* pops_program_name() { return "pops.test.%s"; }
extern "C" const char* pops_program_hash() { return "%s"; }
extern "C" int pops_program_operator_authority_count() { return 0; }
extern "C" std::uint64_t pops_program_operator_authority_word(int, int) {
  return UINT64_C(0);
}
%s
%s
%s
extern "C" bool pops_program_has_dt_bound() { return false; }
extern "C" pops::Real %s(%s*, pops::Real) {
  return std::numeric_limits<pops::Real>::infinity();
}
""" % (
        _emit_route_manifest("pops_program_route_manifest"),
        "euler",
        identity,
        _block_identity_exports(block_names),
        _empty_module_metadata_exports(),
        _amr_budget_exports(len(block_names), coupled_sources, identity)
        if target == "amr_system"
        else "",
        "pops_program_dt_bound_amr" if target == "amr_system" else "pops_program_dt_bound",
        (
            "pops::AmrSystem<pops::kNativeDimension>"
            if target == "amr_system"
            else "pops::System<pops::kNativeDimension>"
        ),
    )
    if target == "system":
        install = (
            """\
extern "C" void pops_install_program(pops::System<pops::kNativeDimension>* sys) {
  auto ctx_owner = pops::runtime::program::make_program_execution_provider(sys);
  ctx_owner->configure_primary_clock("pops.test.clock.macro");
  ctx_owner->install([=](double dt) {
    auto& ctx = *ctx_owner;
    ctx.begin_step(dt);
%s
  });
}
"""
            % body
        )
    else:
        install = (
            """\
extern "C" void pops_install_program_amr(pops::AmrSystem<pops::kNativeDimension>* sys) {
  auto ctx_owner = pops::runtime::program::make_program_execution_provider(sys);
  ctx_owner->configure_primary_clock("pops.test.clock.macro");
  ctx_owner->install([=](double macro_dt) {
    ctx_owner->advance_hierarchy(macro_dt, [=](double dt) {
      auto& ctx = *ctx_owner;
%s
    });
  }, ctx_owner);
}
"""
            % body
        )
    return common + install


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
