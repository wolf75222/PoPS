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
    one ``pops_install_program`` runs on ``System`` -- wrapped in an explicit level-clock scheduler. The
    body references
    only the variable ``ctx`` (never the type), so it compiles against ``AmrProgramContext``'s method
    surface exactly as against ``ProgramContext``'s.

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
    level_driver = ""
    if hierarchy_bodies is not None:
        gather, solve, publish = hierarchy_bodies
        level_driver = (
            '    if (!ctx.has_refined_hierarchy()) {\n'
            '      for (int _k = 0; _k < _nlev; ++_k) {\n'
            '        ctx.set_level(_k);\n' + body + '\n      }\n'
            '    } else {\n'
            '      // Gather every level before the unique hierarchy-scoped solve.\n'
            '      for (int _k = 0; _k < _nlev; ++_k) {\n'
            '        ctx.set_level(_k);\n' + gather + '\n      }\n'
            '      ctx.set_level(0);\n' + solve + '\n'
            '      // The composite solution is complete before any level reconstructs or commits.\n'
            '      for (int _k = 0; _k < _nlev; ++_k) {\n'
            '        ctx.set_level(_k);\n' + publish + '\n      }\n'
            '    }\n')
    return (
        '\n#include <pops/runtime/program/amr_program_context.hpp>  // AmrProgramContext (the AMR driver, ADC-508)\n'
        '// AMR install entry (epic ADC-511 / ADC-508, Spec 6): the target=\'amr_system\' counterpart\n'
        '// of pops_install_program. AmrSystem::install_program resolves + calls it after binding the\n'
        '// blocks by name and seeding the runtime params. It constructs an AmrProgramContext (the AMR\n'
        '// mirror of ProgramContext) and installs the explicit parent/child clock driver: the SAME\n'
        '// lowered body is recursively subcycled, temporally interpolated and conservatively synced.\n'
        'extern "C" void pops_install_program_amr(void* sys) {\n'
        '  pops::runtime::program::AmrProgramContext ctx(sys);\n'
        + prelude +
        '\n  ctx.install([=](double dt) {\n'
        + (
            '    auto _advance_level = [&](double dt) {\n'
            + body +
            '\n    };\n'
            '    ctx.advance_hierarchy(dt, _advance_level);\n'
            if hierarchy_bodies is None else
            '    auto _advance_hierarchy = [&](double dt) {\n'
            '      const int _nlev = ctx.nlev();\n'
            + level_driver +
            '    };\n'
            '    ctx.advance_synchronized_hierarchy(dt, _advance_hierarchy);\n'
        ) +
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
