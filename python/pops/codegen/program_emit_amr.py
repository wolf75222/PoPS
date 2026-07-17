"""pops.codegen.program_emit_amr -- the AMR install-entry emitter (epic ADC-511 / ADC-508, Spec 6).

Split out of :mod:`pops.codegen.program_codegen` so that module stays under the Spec-4 500-line
budget. ``_emit_amr_install`` is the only public name; ``program_codegen`` re-imports it and calls
it from ``emit_cpp_program`` when ``target='amr_system'``.
"""
from __future__ import annotations

from typing import Any


def _emit_amr_install(program: Any, target: Any, prelude: Any, body: Any,
                      hierarchy_bodies: Any = None, dt_bound_body: str | None = None) -> str:
    """C++ source of the AMR install entry the .so exports (epic ADC-511 / ADC-508, Spec 6).

    ``target='system'`` emits NOTHING (a System-only .so carries only ``pops_install_program``).
    ``target='amr_system'`` emits ``pops_install_program_amr``, the entry ``AmrSystem::install_program``
    resolves (it dlopens the .so, validates the ABI key + section-24 requirements, binds the blocks by
    name, seeds the runtime params, then calls this). It constructs an ``AmrProgramContext`` (the AMR
    counterpart of ``ProgramContext``, a DUCK-TYPED structural mirror) over the ``AmrSystem`` and installs
    the recursively subcycled per-level macro-step driver: the IDENTICAL lowered ``{body}`` -- the
    one ``pops_install_program`` runs on ``System`` -- wrapped in an explicit level-clock scheduler.
    Its install-time prelude is materialized once per native level, not once per hierarchy: each
    closure therefore owns fields/workspaces with the exact level layout. A topology-epoch or
    process-local materialization-generation change rematerializes the bundles before the next
    advance. The body references only the variable ``ctx``
    (never the type), so it compiles against ``AmrProgramContext``'s method surface exactly as against
    ``ProgramContext``'s.

    Shape: one macro-step recursively advances each child on its declared parent/child clock relation,
    with exact stage abscissae and mandatory temporal interpolation from parent old/new snapshots, then
    synchronizes finest-first by conservative reflux followed by average-down. The
    body's head-of-step ``ctx.solve_fields()`` fires EXACTLY ONCE per macro-step (a level-0 / not-yet-solved
    guard inside the context), so the coarse Poisson is OncePerStep and injected to every level -- parity
    with the native AMR cadence. The C/F interface is now conservative to round-off: the per-level effective
    flux is captured through the Program's own linear combination and routed through the native
    ``route_reflux`` at level sync (ADC-639), so mass/momentum/energy are conserved across the interface on a
    genuinely multilevel run; a coarse-only / flat Program stays bit-identical."""
    if target != "amr_system":
        return ""
    if hierarchy_bodies is None:
        phase_fields = '    std::function<void(double)> step;\n'
        phase_initializers = (
            '      [=](double dt) {\n'
            '        pops::runtime::program::AmrProgramContext& ctx = *ctx_owner;\n'
            '        (void)dt;\n' + body + '\n'
            '      }\n')
        installed_driver = (
            '    auto _advance_level = [&](double level_dt) {\n'
            '      _refresh_level_programs();\n'
            '      _level_programs->at(static_cast<std::size_t>(ctx.level())).step(level_dt);\n'
            '    };\n'
            '    ctx.advance_hierarchy(dt, _advance_level);\n')
    else:
        gather, solve, publish = hierarchy_bodies
        phase_fields = (
            '    std::function<void(double)> step;\n'
            '    std::function<void(double)> gather;\n'
            '    std::function<void(double)> solve;\n'
            '    std::function<void(double)> publish;\n')
        phase_initializers = (
            '      [=](double dt) {\n'
            '        pops::runtime::program::AmrProgramContext& ctx = *ctx_owner;\n'
            '        (void)dt;\n' + body + '\n'
            '      },\n'
            '      [=](double dt) {\n'
            '        pops::runtime::program::AmrProgramContext& ctx = *ctx_owner;\n'
            '        (void)dt;\n' + gather + '\n'
            '      },\n'
            '      [=](double dt) {\n'
            '        pops::runtime::program::AmrProgramContext& ctx = *ctx_owner;\n'
            '        (void)dt;\n' + solve + '\n'
            '      },\n'
            '      [=](double dt) {\n'
            '        pops::runtime::program::AmrProgramContext& ctx = *ctx_owner;\n'
            '        (void)dt;\n' + publish + '\n'
            '      }\n')
        installed_driver = (
            '    auto _advance_hierarchy = [&](double hierarchy_dt) {\n'
            '      _refresh_level_programs();\n'
            '      const int _nlev = ctx.nlev();\n'
            '      if (!ctx.has_refined_hierarchy()) {\n'
            '        for (int _k = 0; _k < _nlev; ++_k) {\n'
            '          ctx.set_level(_k);\n'
            '          _level_programs->at(static_cast<std::size_t>(_k)).step(hierarchy_dt);\n'
            '        }\n'
            '      } else {\n'
            '        // Gather every level before the unique hierarchy-scoped solve.\n'
            '        for (int _k = 0; _k < _nlev; ++_k) {\n'
            '          ctx.set_level(_k);\n'
            '          _level_programs->at(static_cast<std::size_t>(_k)).gather(hierarchy_dt);\n'
            '        }\n'
            '        ctx.set_level(0);\n'
            '        _level_programs->front().solve(hierarchy_dt);\n'
            '        // The composite solution is complete before any level reconstructs or commits.\n'
            '        for (int _k = 0; _k < _nlev; ++_k) {\n'
            '          ctx.set_level(_k);\n'
            '          _level_programs->at(static_cast<std::size_t>(_k)).publish(hierarchy_dt);\n'
            '        }\n'
            '      }\n'
            '    };\n'
            '    ctx.advance_synchronized_hierarchy(dt, _advance_hierarchy);\n')

    # Every generated prelude allocation is layout-bound. Materialize one complete closure bundle
    # per level before the first advance, and rebuild the set exactly once after a topology epoch or
    # process-local materialization-generation change. This is deliberately generic: scalar scratch,
    # condensed coefficients, matrix-free
    # apply captures, prepared problems and Krylov workspaces all follow the same lifetime protocol.
    level_resources = (
        '  struct _PopsAmrLevelProgram {\n' + phase_fields + '  };\n'
        '  auto _make_level_program = [ctx_owner]() {\n'
        '    pops::runtime::program::AmrProgramContext& ctx = *ctx_owner;\n'
        + prelude + '\n'
        '    return _PopsAmrLevelProgram{\n' + phase_initializers + '    };\n'
        '  };\n'
        '  auto _level_programs = std::make_shared<std::vector<_PopsAmrLevelProgram>>();\n'
        '  auto _level_program_epoch = std::make_shared<std::uint64_t>(\n'
        '      std::numeric_limits<std::uint64_t>::max());\n'
        '  auto _level_program_generation = std::make_shared<std::uint64_t>(\n'
        '      std::numeric_limits<std::uint64_t>::max());\n'
        '  auto _refresh_level_programs = [=]() {\n'
        '    pops::runtime::program::AmrProgramContext& ctx = *ctx_owner;\n'
        '    const std::uint64_t epoch = ctx.program_resource_topology_epoch();\n'
        '    const std::uint64_t generation = ctx.program_resource_topology_generation();\n'
        '    const int levels = ctx.nlev();\n'
        '    if (levels <= 0)\n'
        '      throw std::runtime_error("AMR Program resource refresh requires at least one level");\n'
        '    if (*_level_program_epoch == epoch &&\n'
        '        *_level_program_generation == generation &&\n'
        '        _level_programs->size() == static_cast<std::size_t>(levels))\n'
        '      return;\n'
        '    const int saved_level = ctx.level();\n'
        '    const int restored_level =\n'
        '        saved_level >= 0 && saved_level < levels ? saved_level : 0;\n'
        '    _level_programs->clear();\n'
        '    _level_programs->reserve(static_cast<std::size_t>(levels));\n'
        '    try {\n'
        '      for (int level = 0; level < levels; ++level) {\n'
        '        ctx.set_level(level);\n'
        '        _level_programs->emplace_back(_make_level_program());\n'
        '      }\n'
        '    } catch (...) {\n'
        '      ctx.set_level(restored_level);\n'
        '      throw;\n'
        '    }\n'
        '    ctx.set_level(restored_level);\n'
        '    *_level_program_epoch = epoch;\n'
        '    *_level_program_generation = generation;\n'
        '  };\n'
        '  _refresh_level_programs();\n')
    return (
        '\n#include <pops/runtime/program/amr_program_context.hpp>  // AmrProgramContext (the AMR driver, ADC-508)\n'
        '// AMR install entry (epic ADC-511 / ADC-508, Spec 6): the target=\'amr_system\' counterpart\n'
        '// of pops_install_program. AmrSystem::install_program resolves + calls it after binding the\n'
        '// blocks by name and seeding the runtime params. It constructs an AmrProgramContext (the AMR\n'
        '// mirror of ProgramContext) and installs the explicit parent/child clock driver: the SAME\n'
        '// lowered body is recursively subcycled, temporally interpolated and conservatively synced.\n'
        'extern "C" void pops_install_program_amr(void* sys) {\n'
        '  auto ctx_owner = std::make_shared<pops::runtime::program::AmrProgramContext>(sys);\n'
        '  pops::runtime::program::AmrProgramContext& ctx = *ctx_owner;\n'
        + level_resources +
        '\n  ctx.install([=](double dt) {\n'
        '    pops::runtime::program::AmrProgramContext& ctx = *ctx_owner;\n'
        '    _refresh_level_programs();\n'
        + installed_driver +
        '  });\n'
        '}\n'
        '// AMR counterpart of pops_program_dt_bound. The generated module owns the concrete\n'
        '// AmrProgramContext type; the runtime loader passes only its stable AmrSystem facade.\n'
        '// The body is the identical read-only scalar IR used by the uniform Program ABI.\n'
        'extern "C" pops::Real pops_program_dt_bound_amr(void* sys, pops::Real cfl) {\n'
        '  pops::runtime::program::AmrProgramContext ctx(sys);\n'
        '  (void)ctx; (void)cfl;\n'
        + (dt_bound_body or '    return std::numeric_limits<pops::Real>::infinity();') + '\n'
        '}\n')
