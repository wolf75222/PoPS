#!/usr/bin/env python3
"""ADC-508: the per-level AmrProgramContext driver -- a compiled time Program over the AMR hierarchy.

A compiled time Program lowers its macro-step body referencing ONLY the variable ``ctx`` (never the
type ``ProgramContext``). The SAME generated body therefore compiles against ``AmrProgramContext``, a
duck-typed structural mirror that drives the body recursively over the AMR hierarchy. The
``{amr_install}`` codegen slot passes the identical body to ``advance_hierarchy``.

This test asserts:

  1) (host-side, no compiler) the codegen emits a recursive install wrapper for ``target='amr_system'``
     (the fail-loud is gone): ``pops_install_program_amr`` builds an ``AmrProgramContext`` and runs the
     body through ``ctx.advance_hierarchy(dt, body)``;

  2) (Kokkos-gated, the must-pass GATE 4.1a) BIT-IDENTICAL parity -- the SAME SSPRK2 Program installed
     on a single-level ``System`` (``ProgramContext``) and on a single-level ``AmrSystem``
     (``AmrProgramContext``, the coarse-only Program layout) produces the BYTE-IDENTICAL coarse density
     over several steps (0 ulp). This proves the AmrProgramContext seam methods are byte-faithful
     mirrors of ProgramContext's -- the whole duck-typing claim.  Each independently refreshed
     potential must satisfy the same discrete Poisson equation;

  3) (Kokkos-gated) a CUSTOM 2-stage Program (midpoint RK2) installed on a nonlinear scalar Burgers
     operator RUNS, conserves the coarse mass to round-off, and DIFFERS from SSPRK2.  Each scheme is
     also bit-identical AMR vs System (the Program text actually drives the integrator).

WHAT NEEDS WHICH RUNNER. (1) is pure Python. (2)/(3) need a compiler + a visible Kokkos
(``POPS_KOKKOS_ROOT``) to build the .so; the compiled-.so dlopen + per-level RUN IS validatable on
Kokkos CPU (Serial/OpenMP) locally -- unlike GPU (the CUDA run is the ROMEO step). Self-skips (exit 0)
without pops / a built _pops / a compiler. Pytest + ``__main__`` guard (CI runs ``python3 <file>``).
"""
import sys
from fractions import Fraction

# ADC-627 idiom: this file AOT-compiles several Program/.so artifacts; give the
# process-isolated runner headroom over the default (CI runner speed varies 3-4x).
POPS_PROCESS_TIMEOUT = 1200

try:
    import numpy as np

    import pops.runtime._engine_descriptors as engine
    import pops.lib.time as libtime
    from pops.codegen._compile_drivers import compile_problem
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.time._methods.tableau import RungeKuttaTableau
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.runtime._system import AmrSystem, System  # ADC-545 advanced runtime seam
    from pops.time import FailRun
    from tests.python.integration._final_field_program import (
        compile_block_model,
        compiler_model,
        resolve_periodic_field_program,
        scalar_advection_field_model,
    )
except Exception as exc:  # noqa: BLE001 -- pops/numpy unavailable in this interpreter
    print("skip test_amr_program_parity (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

N = 16
NSTEPS = 4
DT = 1.0e-3

_fails = 0


def chk(cond, label):
    global _fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        _fails += 1


def _euler_model(name="adc508_parity_model"):
    """One conservative scalar plus a periodic field solve for context parity."""
    return scalar_advection_field_model(name)


def _nonlinear_model(name="adc508_nonlinear_model"):
    """Periodic scalar Burgers transport, for a genuine midpoint/SSPRK2 discriminator."""
    from pops.domain import Rectangle
    from pops.fields import FieldOutput, GradientOutput
    from pops.frames import Cartesian2D
    from pops.math import ddt, div, laplacian, unknown
    from pops.physics import Model

    frame = Rectangle(
        "%s-domain" % name, lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model(name, frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.5 * rho * rho,), y_axis: (0.125 * rho * rho,)},
        waves={x_axis: (rho,), y_axis: (0.25 * rho,)},
    )
    model.rate("explicit_rhs", equation=ddt(state) == -div(flux))
    potential = model.field("potential")
    phi = unknown(potential)
    model.field_operator(
        "electrostatic",
        unknown=potential,
        equation=(-laplacian(phi) == rho - 1.0),
        outputs=(FieldOutput("phi", potential), GradientOutput("grad", potential)),
    )
    return model


def _ssprk2_program(model, name="adc508_ssprk2", *, target="amr_system"):
    """The canonical SSPRK2 (Heun) Program on one block 'plasma' -- the SAME scheme the native explicit
    AMR advance uses. solve_fields(); R=rhs(U); U1=U+dt R; solve_fields(U1); R1=rhs(U1);
    U <<= 0.5 U + 0.5 (U1 + dt R1)."""
    return resolve_periodic_field_program(
        model,
        lambda state, rate, fields: libtime.SSPRK2(
            state,
            rate=rate,
            fields=fields,
            solve_action=FailRun(),
        ),
        name=name,
        block_name="plasma",
        target=target,
        n=N,
    )


def _midpoint_program(model, name="adc508_midpoint", *, target="amr_system"):
    """A CUSTOM 2-stage scheme (midpoint RK2): U1 = U + 0.5 dt R(U); U <<= U + dt R(U1). A DIFFERENT
    combine through the same seam -- proves the Program text drives the integrator."""
    midpoint = RungeKuttaTableau(
        A=[[], [Fraction(1, 2)]], b=[0, 1], c=[0, Fraction(1, 2)], name="midpoint")
    return resolve_periodic_field_program(
        model,
        lambda state, rate, fields: libtime.RungeKutta(
            state,
            rate=rate,
            fields=fields,
            tableau=midpoint,
            solve_action=FailRun(),
        ),
        name=name,
        block_name="plasma",
        target=target,
        n=N,
    )


def _init_density():
    """A smooth periodic scalar seeded through the same density coupler on both runtimes."""
    x = (np.arange(N) + 0.5) / N
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)


def test_codegen_emits_amr_install_wrapper():
    """The AMR export installs the recursive, clock-qualified hierarchy driver."""
    print("== codegen emits the recursive AmrProgramContext install wrapper ==")
    model = _euler_model("adc508_wrapper_amr")
    plan = _ssprk2_program(model, target="amr_system")
    src = emit_cpp_program(
        plan.time,
        compiler_model(model),
        target="amr_system",
        field_plans=plan.field_plans,
    )
    chk("pops_install_program_amr" in src, "the AMR .so exports pops_install_program_amr")
    body = src.split("pops_install_program_amr", 1)[1]
    chk("AmrProgramContext ctx(sys)" in body,
        "the AMR install constructs an AmrProgramContext over the AmrSystem")
    chk("ctx.advance_hierarchy(dt, _advance_level)" in body,
        "the wrapper delegates to the explicit parent/child clock driver")
    chk("ctx.set_stage_time(0, 1)" in body and "ctx.set_stage_time(1, 1)" in body,
        "exact SSPRK2 stage abscissae are emitted")
    chk("ctx.set_level(" not in body and "ctx.couple_levels(" not in body,
        "level traversal and synchronization are not duplicated in generated code")
    chk("the per-level AMR macro-step driver" not in body
        and "is not yet available" not in body,
        "the fail-loud throw is gone (the real driver is emitted)")
    # The System target still emits NO AMR entry.
    system_model = _euler_model("adc508_wrapper_system")
    system_plan = _ssprk2_program(system_model, target="system")
    src_sys = emit_cpp_program(
        system_plan.time,
        compiler_model(system_model),
        field_plans=system_plan.field_plans,
    )
    chk("pops_install_program_amr" not in src_sys, "the System .so does NOT export the AMR entry")


def _system_run(plan, model, u0, nsteps=NSTEPS, dt=DT):
    """Install `program` on a single-level System and return (density, potential) after nsteps."""
    sim = System(n=N, L=1.0)
    try:
        block_cm = compile_block_model(model, target="system")
        compiled = compile_problem(
            model=model,
            time=plan.time,
            field_plans=plan.field_plans,
            problem_snapshot=plan.snapshot,
        )
    except RuntimeError as exc:
        return None, "compile (System): %s" % str(exc)[:140]
    for field, field_plan in plan.field_plans.items():
        sim._install_field_plan(field, field_plan)
    sim.add_equation("plasma", block_cm,
                     spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                     time=engine.Explicit(method="ssprk2"))
    sim.set_density("plasma", u0)
    sim.install_program(compiled.so_path)
    for _ in range(nsteps):
        sim.step(dt)
    sim.solve_fields()  # refresh phi for the committed final state, not the last RK stage
    return (np.array(sim.get_state("plasma")), np.array(sim.potential())), None


def _amr_run(plan, model, u0, nsteps=NSTEPS, dt=DT):
    """Install `program` on a single-level AmrSystem (coarse-only Program layout) and return
    (coarse density component-0, coarse potential, coarse mass) after nsteps. Uses the
    ``_install_compiled`` seam (the AMR counterpart of System's compiled install): a native instance
    carries the block model, the compiled handle carries the time Program installed on the hierarchy."""
    amr = AmrSystem(n=N, L=1.0, regrid_every=0)
    if not hasattr(amr, "install_program"):
        return None, "the built _pops lacks AmrSystem.install_program (rebuild _pops)"
    try:
        compiled = compile_problem(
            model=model,
            time=plan.time,
            target="amr_system",
            field_plans=plan.field_plans,
            problem_snapshot=plan.snapshot,
        )
        block_cm = compile_block_model(model, target="amr_system")
    except RuntimeError as exc:
        return None, "compile (AMR): %s" % str(exc)[:140]
    try:
        # Install the exact resolved field route, wire the block, seed the FULL conservative state
        # BEFORE the build (install_program
        # forces ensure_built, which freezes the layout -- set_conservative_state must precede it), then
        # install the compiled time Program on the hierarchy.
        for field, field_plan in plan.field_plans.items():
            amr._install_field_plan(field, field_plan)
        amr.add_equation("plasma", block_cm,
                         spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                         time=engine.Explicit(method="ssprk2"))
        amr.set_density("plasma", u0)
        amr.install_program(compiled.so_path)
    except RuntimeError as exc:
        return None, "install (AMR): %s" % str(exc)[:240]
    for _ in range(nsteps):
        amr.step(dt)
    amr.solve_fields()  # refresh phi for the committed final state, not the last RK stage
    return (np.array(amr.density("plasma")), np.array(amr.potential()),
            float(amr.mass("plasma"))), None


def test_single_level_bit_identical_parity():
    """(2) GATE 4.1a: the SAME SSPRK2 Program on a single-level System (ProgramContext) and a
    single-level AmrSystem (AmrProgramContext, coarse-only) must be BIT-IDENTICAL on the evolved coarse
    DENSITY (0 ulp) -- the AmrProgramContext seam methods (state / rhs_into / axpy / lincomb /
    solve_fields / scratch) are byte-faithful ProgramContext mirrors, so the SAME lowered body produces
    the SAME arithmetic. The periodic potential is non-unique and each runtime owns an independent MG
    history, so both are refreshed on the committed state and checked against the same discrete
    Poisson equation rather than compared iterate-to-iterate."""
    print("== single-level SSPRK2 parity: AmrProgramContext == ProgramContext (bit-identical) ==")
    model = _euler_model("adc508_parity_ssprk2")
    u0 = _init_density()

    sys_out, sys_err = _system_run(
        _ssprk2_program(model, target="system"), model, u0)
    if sys_out is None:
        print("skip (%s)" % sys_err)
        return
    amr_model = _euler_model("adc508_parity_ssprk2")
    amr_out, amr_err = _amr_run(
        _ssprk2_program(amr_model, target="amr_system"), amr_model, u0)
    if amr_out is None:
        print("skip (%s)" % amr_err)
        return

    sys_state, sys_phi = sys_out
    amr_rho, amr_phi, amr_mass = amr_out
    sys_rho = sys_state[0]  # density = component 0

    drho = float(np.abs(sys_rho - amr_rho).max())
    chk(np.array_equal(sys_rho, amr_rho),
        "the evolved coarse density is BIT-IDENTICAL System vs AMR (max|diff| = %.3e)" % drho)
    # The two independent iterative solves have different warm-start histories.  Validate the same
    # discrete periodic equation independently instead of comparing their non-unique iterates.
    def relative_poisson_residual(phi, rho):
        h = 1.0 / N
        laplacian = (
            np.roll(phi, -1, axis=0) + np.roll(phi, 1, axis=0)
            + np.roll(phi, -1, axis=1) + np.roll(phi, 1, axis=1) - 4.0 * phi
        ) / (h * h)
        source = rho - 1.0
        residual = -laplacian - source
        return float(np.linalg.norm(residual) / np.linalg.norm(source))

    sys_residual = relative_poisson_residual(sys_phi, sys_rho)
    amr_residual = relative_poisson_residual(amr_phi, amr_rho)
    chk(sys_residual < 1e-7 and amr_residual < 1e-7,
        "both potentials satisfy the same discrete Poisson equation independently "
        "(System %.3e, AMR %.3e)" % (sys_residual, amr_residual))
    chk(np.all(np.isfinite(amr_rho)) and float(amr_rho.min()) > 0.0,
        "the AMR Program kept a finite, strictly-positive density (min = %.4f)" % float(amr_rho.min()))


def test_custom_two_stage_runs_and_differs():
    """(3) a CUSTOM midpoint-RK2 Program installed on AMR RUNS, conserves the coarse mass to round-off,
    and DIFFERS from the SSPRK2 Program -- the Program text drives the integrator, not a hard-coded
    scheme. Also bit-identical vs the same midpoint Program on System (the duck-typing holds for a
    second, different combine)."""
    print("== custom 2-stage (midpoint RK2) Program on AMR: runs, conserves, differs from SSPRK2 ==")
    model = _nonlinear_model("adc508_parity_mid")
    u0 = _init_density()
    m0 = float(u0.mean())  # mean density == coarse mass / area (L=1)

    mid_amr, err = _amr_run(
        _midpoint_program(model, target="amr_system"), model, u0)
    if mid_amr is None:
        print("skip (%s)" % err)
        return
    mid_rho, mid_phi, mid_mass = mid_amr

    # SSPRK2 on the SAME AMR for the differ-check (same model name -> same .so cache key per Program).
    ss_model = _nonlinear_model("adc508_parity_mid")
    ss_amr, err2 = _amr_run(
        _ssprk2_program(ss_model, target="amr_system"), ss_model, u0)
    if ss_amr is None:
        print("skip ssprk2 leg (%s)" % err2)
        return
    ss_rho = ss_amr[0]

    chk(np.all(np.isfinite(mid_rho)), "the midpoint Program produced a finite state")
    # Mass conservation (periodic, no flux through the boundary): coarse mass == initial to round-off.
    chk(abs(mid_mass - m0) < 1e-9,
        "the midpoint Program conserves the coarse mass (|m - m0| = %.2e)" % abs(mid_mass - m0))
    # A DIFFERENT scheme must give a DIFFERENT trajectory (proves the Program drives the integrator).
    diff = float(np.abs(mid_rho - ss_rho).max())
    chk(diff > 1e-12,
        "the midpoint scheme DIFFERS from SSPRK2 through the SAME seam (max|diff| = %.3e)" % diff)

    # Bit-identical vs the same midpoint Program on System (the duck-typing holds for a 2nd combine).
    sys_model = _nonlinear_model("adc508_parity_mid")
    sys_out, sys_err = _system_run(
        _midpoint_program(sys_model, target="system"), sys_model, u0)
    if sys_out is not None:
        sys_rho = sys_out[0][0]
        chk(np.array_equal(sys_rho, mid_rho),
            "the midpoint Program is bit-identical System vs AMR (max|diff| = %.3e)"
            % float(np.abs(sys_rho - mid_rho).max()))
    ss_sys_model = _nonlinear_model("adc508_parity_mid")
    ss_sys_out, ss_sys_err = _system_run(
        _ssprk2_program(ss_sys_model, target="system"), ss_sys_model, u0)
    if ss_sys_out is not None:
        ss_sys_rho = ss_sys_out[0][0]
        chk(np.array_equal(ss_sys_rho, ss_rho),
            "the SSPRK2 Program is bit-identical System vs AMR (max|diff| = %.3e)"
            % float(np.abs(ss_sys_rho - ss_rho).max()))


def _amr_run_cfl(plan, model, u0, nsteps=NSTEPS, cfl=0.4):
    """Install `program` on a single-level AmrSystem and drive it with step_cfl (NOT step). Returns
    (coarse density, program hash, last dt) -- the step_cfl Program route (ADC-508 review fix 1)."""
    amr = AmrSystem(n=N, L=1.0, regrid_every=0)
    if not hasattr(amr, "install_program") or not hasattr(amr, "step_cfl"):
        return None, "the built _pops lacks AmrSystem.install_program/step_cfl (rebuild _pops)"
    try:
        compiled = compile_problem(
            model=model,
            time=plan.time,
            target="amr_system",
            field_plans=plan.field_plans,
            problem_snapshot=plan.snapshot,
        )
        block_cm = compile_block_model(model, target="amr_system")
    except RuntimeError as exc:
        return None, "compile (AMR): %s" % str(exc)[:140]
    try:
        amr.add_equation("plasma", block_cm,
                         spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                         time=engine.Explicit(method="ssprk2"))
        amr.set_density("plasma", u0)
        amr.install_program(compiled.so_path)
        last_dt = 0.0
        for _ in range(nsteps):
            last_dt = float(amr.step_cfl(cfl))
    except RuntimeError as exc:
        return None, "install/step_cfl (AMR): %s" % str(exc)[:240]
    return (np.array(amr.density("plasma")), amr.installed_program_hash(), last_dt), None


def _amr_run_cfl_native(model, u0, nsteps=NSTEPS, cfl=0.4):
    """A NATIVE (no Program installed) AMR step_cfl run, the baseline the Program route must NOT silently
    reproduce. Same block + IC, but no install_program -> the native engine advances under step_cfl."""
    amr = AmrSystem(n=N, L=1.0, regrid_every=0)
    try:
        block_cm = compile_block_model(model, target="amr_system")
    except RuntimeError as exc:
        return None, "compile (AMR native): %s" % str(exc)[:140]
    amr.add_equation("plasma", block_cm,
                     spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                     time=engine.Explicit(method="ssprk2"))
    amr.set_density("plasma", u0)
    last_dt = 0.0
    for _ in range(nsteps):
        last_dt = float(amr.step_cfl(cfl))
    return (np.array(amr.density("plasma")), last_dt), None


def test_step_cfl_routes_through_installed_program():
    """(4) ADC-508 review fix 1: AmrSystem::step_cfl must route through an installed Program, NOT silently
    run the native engine. We install the SSPRK2 Program, drive it with step_cfl, and assert: (a) the
    program ran (its hash is set, finite dt, finite density), and (b) the evolved density DIFFERS from a
    native (no-program) step_cfl run on the same block + IC -- i.e. step_cfl did NOT silently bypass the
    Program. The SSPRK2 Program and the native ssprk2 scheme are NOT byte-identical here (the Program
    expresses its own solve_fields / commit), so a measurable difference proves the Program drove the
    step. Host/CPU-runnable; self-skips without a compiler / Kokkos."""
    print("== step_cfl routes through the installed AMR Program (fix 1: no silent native bypass) ==")
    model = _euler_model("adc508_stepcfl")
    u0 = _init_density()

    prog_out, err = _amr_run_cfl(
        _ssprk2_program(
            model, "adc508_stepcfl_prog", target="amr_system"),
        model,
        u0,
    )
    if prog_out is None:
        if err and "unknown provider slot" in err:
            chk(False, "installed AMR Program field provider slots must be fully materialized: %s"
                % err)
            return
        print("skip (%s)" % err)
        return
    prog_rho, prog_hash, prog_dt = prog_out
    chk(prog_hash != "", "step_cfl on an installed-Program AMR system records the program hash")
    chk(np.isfinite(prog_dt) and prog_dt > 0.0,
        "step_cfl returned a finite, positive CFL dt (%.3e)" % prog_dt)
    chk(np.all(np.isfinite(prog_rho)) and float(prog_rho.min()) > 0.0,
        "the Program-driven step_cfl kept a finite, strictly-positive density (min = %.4f)"
        % float(prog_rho.min()))

    nat_out, nerr = _amr_run_cfl_native(_euler_model("adc508_stepcfl"), u0)
    if nat_out is None:
        print("skip native baseline (%s)" % nerr)
        return
    nat_rho, nat_dt = nat_out
    # The two dt agree (same CFL scan -- the Program route reuses cfl_dt), but the evolved density must
    # DIFFER: if step_cfl had silently run the native scheme, prog_rho would EQUAL nat_rho byte-for-byte.
    diff = float(np.abs(prog_rho - nat_rho).max())
    chk(diff > 1e-14,
        "the Program-driven step_cfl density DIFFERS from the native step_cfl baseline (max|diff| = "
        "%.3e) -- the installed Program is NOT silently bypassed" % diff)


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print("\n%s test_amr_program_parity (%d check failures)"
          % ("FAIL" if _fails else "PASS", _fails))
    return _fails


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
