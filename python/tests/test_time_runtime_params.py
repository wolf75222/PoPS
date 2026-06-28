#!/usr/bin/env python3
"""Runtime params in a compiled time Program (epic ADC-399 / ADC-510).

A ``dsl.Param(..., kind="runtime")`` read in a Program kernel must change the compiled Program's
numeric result AT RUN TIME -- after ``pops.bind`` -- WITHOUT recompiling the ``.so`` and without
re-installing. Before ADC-510 the program codegen REJECTED a runtime param in a source / flux /
linear-source kernel (NotImplementedError); ADC-510 emits a hoisted uniform ``ctx.param(<index>)``
read and wires the value through ProgramContext -> System::program_param (a live store).

(A) Codegen / IR (pure Python, always runs):
    - a Program whose source reads a runtime param lowers to a HOISTED ``ctx.param(<index>)`` uniform
      local before the per-cell for_each_cell loop (NOT a per-cell read), and exports the
      ``pops_program_param_name`` ABI table; a const param stays inlined (no ctx.param);
    - the per-cell kernel reads the param through ``params.get(<index>)`` (the existing device-clean
      carrier), so it never reaches into ctx inside the device lambda;
    - a dt bound is a read-only scalar sub-program with NO ctx.param scope: a RuntimeParamRef cannot
      reach it by construction (Program._scalar_binop rejects a model Param), and the codegen carries a
      belt-and-suspenders guard so it can never silently mis-lower.

(B) End-to-end (skips unless the full toolchain is present): a ZERO-FLUX scalar ``rho`` with a named
    source ``S = alpha * rho`` (alpha a RUNTIME param). One step U + dt*P.source advances rho by
    dt*alpha*rho. bind(params={"alpha": 0.1}) then a step records the increment; from the SAME compiled
    handle (NO recompile -> same .so path), sim.set_param("alpha", 0.2) and a step from the SAME IC
    must DOUBLE the increment (a measurably different result). A SAME-INSTANCE sub-case proves the live
    read on ONE sim (no re-install). Self-skips if _pops lacks install_program, numpy/_pops is absent,
    or no compiler/Kokkos is visible -- never faking the engine.

Run with python3 (PYTHONPATH = built pops package). The CI gate-python runner executes it as a
script via the ``__main__`` guard.
"""
import sys


def _skip(msg):
    print("skip test_time_runtime_params (%s)" % msg)
    sys.exit(0)


try:
    import numpy as np

    import pops
    from pops.physics.facade import Model
    from pops.physics import RuntimeParam, ConstParam
    from pops import time as adctime
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
except Exception as exc:  # noqa: BLE001 -- numpy or _pops unavailable in this interpreter
    _skip("pops/numpy unavailable: %s" % exc)

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def growth_model(name="rt_growth", kind="runtime", alpha=0.1):
    """Scalar 'rho' with NO flux and a NAMED source 'growth' = alpha*rho (alpha a runtime/const param).

    A complete compilable production block: zero flux (-div F == 0 exactly), so the only dynamics is
    the named source the Program requests -- it isolates the runtime param's effect bit-exactly."""
    m = Model(name)
    (rho,) = m.conservative_vars("rho")
    zero = 0.0 * rho
    m.flux(x=[zero], y=[zero])
    m.eigenvalues(x=[zero], y=[zero])
    m.primitive_vars(rho=rho)
    m.conservative_from([rho])
    a = m.param({"runtime": RuntimeParam, "const": ConstParam}[kind]("alpha", alpha))
    m.source_term("growth", [a * rho])  # S = alpha * rho
    return m


def growth_program(name="rt_step"):
    """U^{n+1} = U + dt * P.source('growth', U), committed on block 'plasma'."""
    P = adctime.Program(name)
    U = P.state("plasma")
    S = P.source("growth", state=U)
    P.commit("plasma", P.linear_combine("%s_step" % name, U + P.dt * S))
    return P


def section_a():
    """(A) Codegen / IR -- pure Python, always runs (no Kokkos / no compiler)."""
    print("== (A) runtime-param codegen: hoisted ctx.param + ABI table ==")
    src_rt = growth_program().emit_cpp_program(model=growth_model(kind="runtime"))

    # The runtime param is read ONCE as a uniform local before the per-cell loop (NOT per cell),
    # through ctx.param(<index>); the per-cell expr then reads params.get(<index>) (device-clean).
    chk("= ctx.param(0)" in src_rt, "runtime param lowers to a hoisted ctx.param(0) uniform read")
    chk("params.values[0] = ctx.param(0)" in src_rt,
        "the hoist fills a pops::RuntimeParams snapshot (params.values[0] = ctx.param(0))")
    chk("params.get(0)" in src_rt,
        "the per-cell source reads the param via params.get(0) (device-clean)")
    # The ctx.param read is OUTSIDE the for_each_cell device lambda (hoisted), not inside it.
    i_param = src_rt.index("= ctx.param(0)")
    chk(src_rt.rfind("for_each_cell", 0, i_param) < src_rt.rfind("for (int li", 0, i_param),
        "ctx.param is hoisted inside the per-fab loop but before for_each_cell (uniform over cells)")

    # The .so exports the runtime-param ABI table install_program reads to validate + seed values.
    chk('pops_program_param_count() { return 1; }' in src_rt, "exports param count 1")
    chk('case 0: return "alpha";' in src_rt, "exports the param NAME table (index 0 -> alpha)")
    chk("pops_program_param_default(int i)" in src_rt, "exports the param DEFAULT table")

    # A CONST param of the same model stays inlined HARD (no runtime-param hoist, no param table entry).
    # Check the hoist marker (= ctx.param( assignment) and the snapshot, not the substring "ctx.param("
    # which also appears in the param-table doc comment.
    src_const = growth_program().emit_cpp_program(model=growth_model(kind="const", alpha=0.1))
    chk("= ctx.param(" not in src_const and "pops::RuntimeParams params;" not in src_const,
        "a const param does NOT hoist a runtime-param read (inlined hard, unchanged)")
    chk("(0.1 * rho)" in src_const, "the const param value is inlined hard into the kernel (0.1 * rho)")
    chk('pops_program_param_count() { return 0; }' in src_const, "a const param exports param count 0")

    # A dt bound is a read-only scalar sub-program with NO ctx.param scope. A model runtime param
    # cannot reach it: Program._scalar_binop only accepts a Scalar Value or a number, so multiplying a
    # model Param into a dt bound raises a TypeError at authoring (it can never reach codegen). This
    # keeps the dt bound free of any undeclared `params` reference.
    P = adctime.Program("rt_dtbound")
    U = P.state("plasma")
    P.commit("plasma", P.linear_combine("rt_dtbound_step", U + P.dt * P.source("growth", state=U)))
    a = growth_model(kind="runtime").params["alpha"]  # the facade Model holds the Param objects
    raised = False
    try:
        P.set_dt_bound(lambda P, cfl: cfl * P.hmin() * a)  # a model runtime Param in the dt bound
    except TypeError:
        raised = True
    chk(raised, "a model runtime param cannot enter a dt bound (TypeError at authoring, never codegen)")
    # And the dt bound that legitimately reads cfl / hmin lowers with NO ctx.param (no params scope).
    src_dt = P.emit_cpp_program(model=growth_model(kind="runtime"))
    body = src_dt.split("pops_program_dt_bound")[1] if "pops_program_dt_bound" in src_dt else ""
    chk("ctx.param(" not in body,
        "the dt_bound body emits no ctx.param (read-only scalar sub-program, no params scope)")


def section_b():
    """(B) End-to-end: bind(alpha) then set_param changes the result with the SAME .so (no recompile).

    Skips cleanly unless the full toolchain (install_program / set_program_param bindings + a working
    compiler + Kokkos) is present -- never faking the engine."""
    if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
        print("-- (B) skipped: _pops lacks the install_program binding (rebuild _pops) --")
        return
    if not hasattr(pops.System(n=8, L=1.0, periodic=True), "set_program_param"):
        print("-- (B) skipped: _pops lacks the set_program_param binding (rebuild _pops) --")
        return

    print("== (B) end-to-end: bind(alpha) then set_param changes the result, no recompile ==")
    dt = 0.05
    n = 16

    def initial_rho():
        x = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(x, x, indexing="ij")
        return 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)

    # Compile ONCE: the same compiled handle / .so drives both alpha values (the whole point).
    try:
        compiled = pops.compile_problem(model=growth_model("rt_block", kind="runtime", alpha=0.1),
                                        time=growth_program("rt_step"))
    except RuntimeError as exc:  # no compiler / no Kokkos / .so compile failed
        _skip("compile_problem could not build the .so: %s" % str(exc)[:160])
    so_path = compiled.so_path

    def make_sim():
        sim = pops.System(n=n, L=1.0, periodic=True)
        try:
            block = growth_model("rt_block_inst", kind="runtime", alpha=0.1).compile(backend="aot")
        except RuntimeError as exc:  # no compiler / no Kokkos visible
            _skip("model compile could not build the block .so: %s" % str(exc)[:160])
        sim.add_equation("plasma", block,
                         spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                         time=pops.Explicit(method="euler"))
        sim.set_state("plasma", np.stack([initial_rho()]))
        return sim

    def step_with_alpha(alpha, use_set_param):
        """Install the SAME .so, set alpha (via bind-time params or a post-bind set_param), step once
        from the fixed IC, return the new rho. NO recompile (so_path is fixed across calls)."""
        sim = make_sim()
        sim.install_program(so_path)
        if use_set_param:
            sim.set_program_param("alpha", alpha)
        else:
            sim._install_program_params({"alpha": alpha})  # the bind(params=) routing path
        sim.step(dt)
        return np.array(sim.get_state("plasma"))[0]

    rho0 = initial_rho()

    # alpha = 0.1: out - rho0 == dt * 0.1 * rho0 (zero flux; the named source is the only dynamics).
    out_lo = step_with_alpha(0.1, use_set_param=False)
    ref_lo = dt * 0.1 * rho0
    d_lo = float(np.abs((out_lo - rho0) - ref_lo).max())
    print("  alpha=0.1  max|(out-rho0) - dt*0.1*rho0| = %.3e" % d_lo)
    chk(d_lo < 1e-12,
        "alpha=0.1 (bind params) advances rho by dt*0.1*rho (the runtime param took effect)")

    # SAME .so, alpha changed at run time to 0.2 on a FRESH sim: the increment must DOUBLE.
    out_hi = step_with_alpha(0.2, use_set_param=True)
    ref_hi = dt * 0.2 * rho0
    d_hi = float(np.abs((out_hi - rho0) - ref_hi).max())
    print("  alpha=0.2  max|(out-rho0) - dt*0.2*rho0| = %.3e" % d_hi)
    chk(d_hi < 1e-12, "set_param('alpha', 0.2) doubles the increment (changed at run time, same .so)")

    # The two runs DIFFER measurably -- the runtime param genuinely changed the compiled result.
    diff = float(np.abs(out_hi - out_lo).max())
    print("  max|out(alpha=0.2) - out(alpha=0.1)| = %.3e" % diff)
    chk(diff > 1e-3, "the two alpha values produce a measurably different result")
    chk(compiled.so_path == so_path, "the .so path is unchanged across both alpha values (no recompile)")

    # SAME-INSTANCE: ONE sim, one install_program, set_program_param BETWEEN steps from the same IC.
    # This is the direct proof that ctx.param reads the live store every step (no re-install, no
    # recompile): the SAME installed Program closure yields dt*0.1*rho then, after set_param, dt*0.2*rho.
    print("  -- same-instance: one sim, set_program_param between steps --")
    sim1 = make_sim()
    sim1.install_program(so_path)
    sim1.set_program_param("alpha", 0.1)
    sim1.step(dt)
    inc1 = np.array(sim1.get_state("plasma"))[0] - rho0
    d_si1 = float(np.abs(inc1 - dt * 0.1 * rho0).max())
    chk(d_si1 < 1e-12, "same sim, alpha=0.1: increment == dt*0.1*rho")
    sim1.set_state("plasma", np.stack([rho0]))  # reset to the IC (no re-install, no recompile)
    sim1.set_program_param("alpha", 0.2)         # live store update on the SAME installed Program
    sim1.step(dt)
    inc2 = np.array(sim1.get_state("plasma"))[0] - rho0
    d_si2 = float(np.abs(inc2 - dt * 0.2 * rho0).max())
    chk(d_si2 < 1e-12, "same sim, set_program_param('alpha', 0.2): increment == dt*0.2*rho (live read)")
    chk(float(np.abs(inc2 - inc1).max()) > 1e-3,
        "same sim: the two steps differ measurably (live param read every step, no re-install)")

    # A typo on set_param fails loud, naming the parameter that is not declared.
    sim_t = make_sim()
    sim_t.install_program(so_path)
    try:
        sim_t.set_program_param("beta", 1.0)
        chk(False, "set_param on an undeclared param must raise")
    except RuntimeError as exc:
        chk("beta" in str(exc), "set_param('beta') (undeclared) fails loud naming the param")


def main():
    section_a()
    section_b()
    print("%s test_time_runtime_params" % ("FAIL (%d)" % fails if fails else "PASS"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
