"""pops.codegen.program_emit_amr -- the AMR install-entry emitter (epic ADC-511 / ADC-508, Spec 6).

Split out of :mod:`pops.codegen.program_codegen` so that module stays under the Spec-4 500-line
budget. ``_emit_amr_install`` is the only public name; ``program_codegen`` re-imports it and calls
it from ``emit_cpp_program`` when ``target='amr_system'``.
"""


def _emit_amr_install(program, target, prelude, body):
    """C++ source of the AMR install entry the .so exports (epic ADC-511 / ADC-508, Spec 6).

    ``target='system'`` emits NOTHING (a System-only problem.so carries only ``pops_problem_install``).
    ``target='amr_system'`` emits ``pops_problem_install_amr``, the native AMR compiled-problem loader
    resolves (it dlopens the .so, validates the ABI key + section-24 requirements, binds the blocks by
    name, seeds the runtime params, then calls this). It constructs an ``AmrProgramContext`` (the AMR
    counterpart of ``ProgramContext``, a DUCK-TYPED structural mirror) over the ``AmrSystem`` and installs
    the SYNCHRONOUS per-level macro-step driver (epic ADC-508): the IDENTICAL lowered ``{body}`` -- the
    one ``pops_problem_install`` runs on ``System`` -- wrapped in a per-level loop. The body references
    only the variable ``ctx`` (never the type), so it compiles against ``AmrProgramContext``'s method
    surface exactly as against ``ProgramContext``'s.

    Shape (v1, SYNCHRONOUS, NON-subcycled): one macro-step regrids at its head (engine cadence), then
    advances EVERY level with the SAME dt by running the body once per level (``ctx.set_level(k)``), then
    couples the levels (``ctx.couple_levels()`` = fine->coarse average_down). The body's head-of-step
    ``ctx.solve_fields()`` fires EXACTLY ONCE per macro-step (a level-0 / not-yet-solved guard inside the
    context), so the coarse Poisson is OncePerStep and injected to every level -- parity with the native
    AMR cadence. Berger-Oliger subcycling + conservative reflux under a Program are DEFERRED (documented);
    the per-stage fine-level field re-solve falls back to the injected aux (exact for the SSPRK2 parity
    Program, whose coupling is frozen across the RK stages)."""
    if target != "amr_system":
        return ""
    return (
        '\n#include <pops/runtime/program/amr_program_context.hpp>  // AmrProgramContext (the AMR driver, ADC-508)\n'
        '// AMR install entry (epic ADC-511 / ADC-508, Spec 6): the target=\'amr_system\' counterpart\n'
        '// of pops_problem_install. The native AMR compiled-problem loader resolves + calls it after binding the\n'
        '// blocks by name and seeding the runtime params. It constructs an AmrProgramContext (the AMR\n'
        '// mirror of ProgramContext) and installs the SYNCHRONOUS per-level macro-step driver: the SAME\n'
        '// lowered body, wrapped in a per-level loop (the body references only ctx, never the type, so\n'
        '// it compiles against AmrProgramContext exactly as against ProgramContext). v1 is synchronous\n'
        '// (same dt every level + average_down between); Berger-Oliger subcycling/reflux are deferred.\n'
        'extern "C" void pops_problem_install_amr(void* sys) {\n'
        '  pops::runtime::program::AmrProgramContext ctx(sys);\n'
        + prelude +
        '\n  auto generated_program_body = [=](auto& ctx, double dt) {\n'
        '    (void)dt;\n'
        + body +
        '\n  };\n'
        '  ctx.install([=](double dt) {\n'
        '    ctx.reset_step();                       // clear the once-per-step solve_fields guard\n'
        '    ctx.regrid_if_due(ctx.macro_step());    // head-of-step union-tags regrid (engine cadence)\n'
        '    const int _nlev = ctx.nlev();\n'
        '    for (int _k = 0; _k < _nlev; ++_k) {\n'
        '      ctx.set_level(_k);                     // the body addresses block b at the CURRENT level\n'
        '      GeneratedProgram::step(ctx, dt, generated_program_body);\n'
        '\n    }\n'
        '    ctx.couple_levels();                     // (B) fine->coarse average_down (v1: no reflux)\n'
        '  });\n'
        '}\n'
        'extern "C" void pops_install_program_amr(void* sys) { pops_problem_install_amr(sys); }\n')
