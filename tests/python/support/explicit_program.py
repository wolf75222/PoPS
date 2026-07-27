"""Explicit native Programs for low-level Python runtime tests.

These helpers are deliberately test infrastructure, not a runtime compatibility layer.  A helper
builds an ordinary ABI-checked ``problem.so`` with an explicit block identity table, installs a real
``ProgramContext`` or ``AmrProgramContext``, and only then lets a test call the temporal facade.

The forward-Euler bridge is useful for tests whose subject is a spatial/runtime capability rather
than Program authoring.  Explicit-method tests use the separately named SSPRK2/SSPRK3 installers
below, so their tableau remains authored and visible instead of being inferred from a block adapter.
Projection and coupled-source splitting are opt-in by block identity/registration; implicit-solve
tests still need purpose-built Program primitives.
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
        'extern "C" int pops_module_field_space_count() { return 0; }\n'
        + string_exports
    )


def _coupling_application(block_count: int, enabled: bool) -> str:
    if not enabled:
        return ""
    candidates = [
        "        {%d, &next_%d}" % (block, block)
        for block in range(block_count)
    ]
    return "      ctx.apply_coupling_operators(dt, {\n%s\n      });" % ",\n".join(
        candidates
    )


def _forward_euler_body(
    block_count: int,
    projection_indices: tuple[int, ...],
    *,
    apply_couplings: bool,
) -> str:
    declarations = []
    requests = []
    combinations = []
    projections = []
    commits = []
    for block in range(block_count):
        declarations.extend(
            (
                "      pops::MultiFab& state_%d = ctx.state(%d);" % (block, block),
                "      pops::MultiFab& rhs_%d = "
                "ctx.rhs_scratch(%d, 0, state_%d);" % (block, 1000 + block, block),
            )
        )
        requests.append(
            "        {%d, &state_%d, &rhs_%d, %d, 0}"
            % (block, block, block, 3000 + block)
        )
        combinations.extend(
            (
                "      pops::MultiFab& next_%d = "
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
            "      (void)ctx.solve_fields();",
            *declarations,
            "      ctx.rhs_group(4000, {\n%s\n      });" % ",\n".join(requests),
            *combinations,
            _coupling_application(block_count, apply_couplings),
            *projections,
            "      ctx.commit_many({\n%s\n      });" % ",\n".join(commits),
        )
    )


def _state_declarations(block_count: int) -> list[str]:
    return [
        "      pops::MultiFab& state_%d = ctx.state(%d);" % (block, block)
        for block in range(block_count)
    ]


def _rhs_stage(
    block_count: int,
    *,
    stage: int,
    numerator: int,
    denominator: int,
    state_prefix: str,
) -> list[str]:
    declarations = [
        "      pops::MultiFab& rhs_%d_%d = ctx.rhs_scratch(%d, 0, %s_%d);"
        % (stage, block, 10000 + stage * block_count + block, state_prefix, block)
        for block in range(block_count)
    ]
    requests = [
        "        {%d, &%s_%d, &rhs_%d_%d, %d, 0}"
        % (
            block,
            state_prefix,
            block,
            stage,
            block,
            30000 + stage * block_count + block,
        )
        for block in range(block_count)
    ]
    return [
        "      ctx.set_stage_time(%d, %d);" % (numerator, denominator),
        *declarations,
        "      ctx.rhs_group(%d, {\n%s\n      });"
        % (40000 + stage, ",\n".join(requests)),
    ]


def _projection_and_commit(
    block_count: int,
    projection_indices: tuple[int, ...],
    *,
    apply_couplings: bool,
) -> list[str]:
    projections = [
        "      ctx.apply_projection(%d, next_%d);" % (block, block)
        for block in projection_indices
    ]
    commits = [
        "        {&state_%d, &next_%d}" % (block, block)
        for block in range(block_count)
    ]
    return [
        _coupling_application(block_count, apply_couplings),
        *projections,
        "      ctx.commit_many({\n%s\n      });" % ",\n".join(commits),
    ]


def _ssprk2_body(
    block_count: int,
    projection_indices: tuple[int, ...],
    *,
    apply_couplings: bool,
) -> str:
    stage_one = []
    result = []
    for block in range(block_count):
        stage_one.extend(
            (
                "      pops::MultiFab& stage1_%d = "
                "ctx.scratch_state(%d, 0, state_%d);"
                % (block, 20000 + block, block),
                "      ctx.lincomb(stage1_%d, pops::Real(1), state_%d, dt, rhs_0_%d, dt, "
                "{{0, 1, 1}}, {{1, 1, 1}});" % (block, block, block),
            )
        )
        result.extend(
            (
                "      pops::MultiFab& endpoint1_%d = "
                "ctx.scratch_state(%d, 0, state_%d);"
                % (block, 21000 + block, block),
                "      ctx.lincomb(endpoint1_%d, pops::Real(1), stage1_%d, dt, rhs_1_%d, "
                "dt, {{0, 1, 1}}, {{1, 1, 1}});" % (block, block, block),
                "      pops::MultiFab& next_%d = ctx.scratch_state(%d, 0, state_%d);"
                % (block, 22000 + block, block),
                "      ctx.lincomb(next_%d, pops::Real(1) / pops::Real(2), state_%d, "
                "pops::Real(1) / pops::Real(2), endpoint1_%d, dt, "
                "{{0, 1, 2}}, {{0, 1, 2}});" % (block, block, block),
            )
        )
    return "\n".join(
        (
            *_state_declarations(block_count),
            "      (void)ctx.solve_fields();",
            *_rhs_stage(
                block_count,
                stage=0,
                numerator=0,
                denominator=1,
                state_prefix="state",
            ),
            *stage_one,
            *_rhs_stage(
                block_count,
                stage=1,
                numerator=1,
                denominator=1,
                state_prefix="stage1",
            ),
            *result,
            *_projection_and_commit(
                block_count,
                projection_indices,
                apply_couplings=apply_couplings,
            ),
        )
    )


def _ssprk3_body(
    block_count: int,
    projection_indices: tuple[int, ...],
    *,
    apply_couplings: bool,
) -> str:
    first_stage = []
    second_stage = []
    result = []
    for block in range(block_count):
        first_stage.extend(
            (
                "      pops::MultiFab& stage1_%d = "
                "ctx.scratch_state(%d, 0, state_%d);"
                % (block, 20000 + block, block),
                "      ctx.lincomb(stage1_%d, pops::Real(1), state_%d, dt, rhs_0_%d, dt, "
                "{{0, 1, 1}}, {{1, 1, 1}});" % (block, block, block),
            )
        )
        second_stage.extend(
            (
                "      pops::MultiFab& endpoint1_%d = "
                "ctx.scratch_state(%d, 0, state_%d);"
                % (block, 21000 + block, block),
                "      ctx.lincomb(endpoint1_%d, pops::Real(1), stage1_%d, dt, rhs_1_%d, "
                "dt, {{0, 1, 1}}, {{1, 1, 1}});" % (block, block, block),
                "      pops::MultiFab& stage2_%d = "
                "ctx.scratch_state(%d, 0, state_%d);"
                % (block, 22000 + block, block),
                "      ctx.lincomb(stage2_%d, pops::Real(3) / pops::Real(4), state_%d, "
                "pops::Real(1) / pops::Real(4), endpoint1_%d, dt, "
                "{{0, 3, 4}}, {{0, 1, 4}});" % (block, block, block),
            )
        )
        result.extend(
            (
                "      pops::MultiFab& endpoint2_%d = "
                "ctx.scratch_state(%d, 0, state_%d);"
                % (block, 23000 + block, block),
                "      ctx.lincomb(endpoint2_%d, pops::Real(1), stage2_%d, dt, rhs_2_%d, "
                "dt, {{0, 1, 1}}, {{1, 1, 1}});" % (block, block, block),
                "      pops::MultiFab& next_%d = ctx.scratch_state(%d, 0, state_%d);"
                % (block, 24000 + block, block),
                "      ctx.lincomb(next_%d, pops::Real(1) / pops::Real(3), state_%d, "
                "pops::Real(2) / pops::Real(3), endpoint2_%d, dt, "
                "{{0, 1, 3}}, {{0, 2, 3}});" % (block, block, block),
            )
        )
    return "\n".join(
        (
            *_state_declarations(block_count),
            "      (void)ctx.solve_fields();",
            *_rhs_stage(
                block_count,
                stage=0,
                numerator=0,
                denominator=1,
                state_prefix="state",
            ),
            *first_stage,
            *_rhs_stage(
                block_count,
                stage=1,
                numerator=1,
                denominator=1,
                state_prefix="stage1",
            ),
            *second_stage,
            *_rhs_stage(
                block_count,
                stage=2,
                numerator=1,
                denominator=2,
                state_prefix="stage2",
            ),
            *result,
            *_projection_and_commit(
                block_count,
                projection_indices,
                apply_couplings=apply_couplings,
            ),
        )
    )


def _source(
    *,
    target: str,
    block_names: tuple[str, ...],
    projection_indices: tuple[int, ...],
    coupled_sources: bool,
    identity: str,
    method: str,
) -> str:
    if method in {"euler", "imex_source_free"}:
        body = _forward_euler_body(
            len(block_names),
            projection_indices,
            apply_couplings=coupled_sources,
        )
    elif method == "ssprk2":
        body = _ssprk2_body(
            len(block_names),
            projection_indices,
            apply_couplings=coupled_sources,
        )
    elif method == "ssprk3":
        body = _ssprk3_body(
            len(block_names),
            projection_indices,
            apply_couplings=coupled_sources,
        )
    else:  # pragma: no cover - private callers validate the method
        raise ValueError("unsupported explicit test Program %r" % method)
    common = """\
#if !defined(POPS_RUNTIME_SHARED_EXCEPTION_ABI)
#error "test Programs require the shared runtime exception ABI consumer contract"
#endif
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/program/program_context.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
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
extern "C" bool pops_program_has_dt_bound() { return false; }
extern "C" pops::Real pops_program_dt_bound(
    pops::runtime::program::ProgramContext*, pops::Real) {
  return std::numeric_limits<pops::Real>::infinity();
}
""" % (
        _emit_route_manifest("pops_program_route_manifest"),
        method,
        identity,
        _block_identity_exports(block_names),
        _empty_module_metadata_exports(),
    )
    if target == "system":
        install = """\
extern "C" void pops_install_program(void* system) {
  auto context = std::make_shared<pops::runtime::program::ProgramContext>(system);
  context->configure_primary_clock("pops.test.clock.macro");
  context->install([context](double dt) {
    pops::runtime::program::ProgramContext& ctx = *context;
    ctx.begin_step(dt);
%s
  });
}
""" % body
    else:
        install = """\
extern "C" void pops_install_program_amr(void* system) {
  auto context = std::make_shared<pops::runtime::program::AmrProgramContext>(system);
  context->configure_primary_clock("pops.test.clock.macro");
  context->install([context](double macro_dt) {
    context->advance_hierarchy(macro_dt, [context](double dt) {
      pops::runtime::program::AmrProgramContext& ctx = *context;
%s
    });
  }, context);
}
""" % body
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


def _install_explicit_program(
    runtime: Any,
    *,
    method: str,
    project_blocks: tuple[str, ...] = (),
    coupled_sources: bool = False,
) -> str:
    if method not in {"euler", "ssprk2", "ssprk3", "imex_source_free"}:
        raise ValueError(
            "method must be 'euler', 'ssprk2', 'ssprk3', or 'imex_source_free'"
        )
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
            "method": method,
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
            method=method,
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
    return _install_explicit_program(
        runtime,
        method="euler",
        project_blocks=project_blocks,
        coupled_sources=coupled_sources,
    )


def install_ssprk2_program(
    runtime: Any,
    *,
    project_blocks: tuple[str, ...] = (),
    coupled_sources: bool = False,
) -> str:
    """Install an SSPRK2/Heun Program, optionally followed by coupled-source splitting."""
    return _install_explicit_program(
        runtime,
        method="ssprk2",
        project_blocks=project_blocks,
        coupled_sources=coupled_sources,
    )


def install_ssprk3_program(
    runtime: Any,
    *,
    project_blocks: tuple[str, ...] = (),
    coupled_sources: bool = False,
) -> str:
    """Install an SSPRK3 Program, optionally followed by coupled-source splitting."""
    return _install_explicit_program(
        runtime,
        method="ssprk3",
        project_blocks=project_blocks,
        coupled_sources=coupled_sources,
    )


def install_source_free_imex_program(runtime: Any) -> str:
    """Install the explicit Program to which an IMEX split reduces when its source is identically zero."""
    return _install_explicit_program(runtime, method="imex_source_free")


__all__ = [
    "install_forward_euler_program",
    "install_source_free_imex_program",
    "install_ssprk2_program",
    "install_ssprk3_program",
]
