#!/usr/bin/env python3
"""Spec 6 sec.11 (ADC-508): the compiled time-Program install seam on AmrSystem.

``AmrSystem::install_program(so_path)`` is the AMR counterpart of ``System::install_program``: it
dlopens a generated ``problem.so`` (compiled with ``target='amr_system'`` -> the .so exports
``pops_install_program_amr``), checks its ABI key, runs the section-24 requirement validation (block
instance / solver), binds the Program blocks BY NAME, seeds the per-PROGRAM-block ``RuntimeParams``
from the .so metadata, then installs the macro-step body. ``pops.bind`` routes a compiled Program and
its qualified ``params=`` onto these AMR seams; its typed StepStrategy is carried by the Program.

The hierarchy DRIVER (``AmrProgramContext``) has LANDED: the generated
``pops_install_program_amr`` constructs an ``AmrProgramContext`` and installs the recursive,
clock-qualified parent/child advance. This test asserts:

  1) the codegen emits ``pops_install_program_amr`` for ``target='amr_system'`` (building an
     AmrProgramContext + recursive hierarchy driver) and NOT for the System default (host-side check);
  2) the AMR install SEAM (``set_program_cadence`` / ``set_program_params``) is reachable on a built
     ``_pops`` and validates its arguments (cadence >= 1; an unseeded program block rejects a set);
  3) (Kokkos-gated) ``compile_problem(..., target='amr_system')`` builds the .so and
     ``AmrSystem.install_program`` INSTALLS it and one per-level macro-step RUNS. The bit-identical
     parity vs System is test_amr_program_parity; the CUDA run is the ROMEO step.

WHAT NEEDS WHICH RUNNER. (1) is pure Python (any interpreter with pops importable). (2) needs a built
``_pops`` (CI serial-python is enough -- no Kokkos run). (3) needs a compiler + a visible Kokkos
(POPS_KOKKOS_ROOT) to build the .so (CI-Kokkos), and the actual per-level RUN of an installed AMR
Program is ROMEO (the AmrProgramContext driver). Self-skips (exit 0) without pops / a built _pops /
a compiler. Pytest + __main__ guard (CI runs ``python3 <file>``).
"""
import sys

try:
    import numpy as np

    import pops
    from pops import time as adctime
    from pops.physics.facade import Model
    from pops.ir.ops import sqrt
    from pops.runtime.system import AmrSystem  # ADC-545 advanced runtime seam
    from tests.python.support.typed_program import program_states
except Exception as exc:  # noqa: BLE001 -- pops/numpy unavailable in this interpreter
    print("skip test_amr_install_program (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

N = 16


def chk(cond, label):
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    assert cond, label


def _euler_model(name="adc508_amr_model"):
    """A compressible Euler block (no required aux), elliptic_rhs = rho so a field solve is present.
    The PHYSICAL model the Program lowers + the block model the AMR instance carries."""
    GAMMA = 1.4
    m = Model(name)
    rho, rhou, rhov, E = m.conservative_vars("rho", "rho_u", "rho_v", "E")
    u, v = rhou / rho, rhov / rho
    p = (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v))
    pu, pv, pp = m.primitive("u", u), m.primitive("v", v), m.primitive("p", p)
    H = (E + pp) / rho
    c = sqrt(GAMMA * pp / rho)
    m.flux(x=[rhou, rhou * pu + pp, rhou * pv, rho * H * pu],
           y=[rhov, rhov * pu, rhov * pv + pp, rho * H * pv])
    m.eigenvalues(x=[pu - c, pu, pu + c], y=[pv - c, pv, pv + c])
    m.primitive_vars(rho, pu, pv, pp)
    m.conservative_from([rho, rho * pu, rho * pv,
                         pp / (GAMMA - 1.0) + 0.5 * rho * (pu * pu + pv * pv)])
    m.gamma(GAMMA)
    m.elliptic_rhs(rho)
    m.rate_operator("explicit_rhs", flux=True)
    return m


def _lie_program(model, name="adc508_amr_prog"):
    """A single-block Lie step on 'plasma' (solve_fields then a Forward-Euler commit)."""
    P = adctime.Program(name)
    _case, states = program_states(P, model, ("plasma",))
    temporal = states["plasma"]
    u = temporal.n
    fields = P.solve_fields(u)
    r = P._rhs_legacy(state=u, fields=fields)
    P.commit(temporal.next,
             P.value("u1", u + P.dt * r, at=temporal.next.point))
    return P


def _two_block_program(model, name="adc508_amr_2block"):
    """A TWO-block Lie Program (states 'plasma' and 'plasma2'), each a Forward-Euler step. The Program
    binds 2 blocks -> the AMR install must FAIL LOUD (v1 single-block-AMR-Program limit, ADC-508 fix 2)."""
    P = adctime.Program(name)
    _case, states = program_states(P, model, ("plasma", "plasma2"))
    for blk in ("plasma", "plasma2"):
        temporal = states[blk]
        u = temporal.n
        fields = P.solve_fields(u)
        r = P._rhs_legacy(state=u, fields=fields)
        P.commit(temporal.next,
                 P.value("u1_%s" % blk, u + P.dt * r,
                                  at=temporal.next.point))
    return P


def test_codegen_emits_amr_install_export():
    """(1) host-side: emit_cpp_program(target='amr_system') emits pops_install_program_amr; the System
    default does NOT. The AMR export builds an AmrProgramContext and runs the body per level (the
    per-level driver has landed, ADC-508)."""
    print("== codegen emits pops_install_program_amr only for target='amr_system' ==")
    model = _euler_model()
    prog = _lie_program(model)
    src_sys = prog.emit_cpp_program(model=model)
    src_amr = prog.emit_cpp_program(model=model, target="amr_system")
    chk("pops_install_program(" in src_sys, "the System .so exports pops_install_program")
    chk("pops_install_program_amr" not in src_sys, "the System .so does NOT export the AMR entry")
    chk("pops_install_program_amr" in src_amr, "the AMR .so exports pops_install_program_amr")
    amr_entry = src_amr.split("pops_install_program_amr", 1)[1]
    chk("AmrProgramContext ctx(sys)" in amr_entry
        and "ctx.advance_hierarchy(dt, _advance_level)" in amr_entry,
        "the AMR install entry builds the recursive clock-qualified hierarchy driver")
    # bad target rejected
    try:
        prog.emit_cpp_program(model=model, target="bogus")
        chk(False, "an unknown target must raise")
    except ValueError as exc:
        chk("target" in str(exc), "unknown target is rejected with a clear message")


def test_amr_program_seam_validates_arguments():
    """(2) needs a built _pops (no Kokkos run): the AMR program seam is reachable and validates its
    arguments. set_program_cadence rejects substeps/stride < 1; set_program_params rejects an unseeded
    program block (no compiled Program installed -> no block has runtime params)."""
    print("== AmrSystem program seam validates its arguments (no Kokkos run) ==")
    amr = AmrSystem(n=N, L=1.0)
    if not hasattr(amr, "set_program_cadence"):
        print("skip (_pops lacks the AMR program seam; rebuild _pops)")
        return
    chk(amr.installed_program_hash() == "", "no program installed -> empty hash")
    for bad in ((0, 1), (1, 0), (-1, 2)):
        try:
            amr.set_program_cadence(*bad)
            chk(False, "set_program_cadence%r should raise (>= 1 required)" % (bad,))
        except (ValueError, RuntimeError) as exc:
            chk(">= 1" in str(exc) or "substeps" in str(exc) or "stride" in str(exc),
                "cadence %r rejected with a clear message" % (bad,))
    amr.set_program_cadence(2, 3)  # valid -> no raise
    chk(True, "a valid cadence (2, 3) is accepted")
    try:
        amr.set_program_params(0, [1.0])  # no program installed -> block 0 not seeded
        chk(False, "set_program_params on an unseeded block should raise")
    except (RuntimeError, IndexError, ValueError) as exc:
        chk("runtime parameter" in str(exc) or "no runtime" in str(exc) or "program block" in str(exc),
            "an unseeded program block rejects a set")


def test_amr_install_program_end_to_end_kokkos():
    """(3) Kokkos-gated: compile_problem(target='amr_system') builds the .so and
    AmrSystem.install_program installs it. The per-level AmrProgramContext driver (ADC-508) has landed,
    so the install SUCCEEDS and one step RUNS over the hierarchy (the WHOLE loader path -- dlopen + ABI
    key + name binding + param seed + the per-level macro-step -- is wired). Self-skips without a
    compiler / Kokkos. The CUDA run is the ROMEO step; bit-identical parity is test_amr_program_parity."""
    print("== AMR install_program installs + runs the per-level driver ==")
    amr = AmrSystem(n=N, L=1.0, regrid_every=0)
    if not hasattr(amr, "install_program"):
        print("skip (_pops lacks AmrSystem.install_program; rebuild _pops)")
        return
    m = _euler_model()
    try:
        compiled = pops.codegen.compile_problem(
            model=m, time=_lie_program(m), target="amr_system")
        block_cm = m.compile(backend="production", target="amr_system")
    except RuntimeError as exc:
        print("skip (no Kokkos to build the AMR .so: %s)" % str(exc)[:120])
        return

    x = (np.arange(N) + 0.5) / N
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    try:
        amr.add_equation("plasma", block_cm, spatial=pops.FiniteVolume(),
                         time=pops.Explicit(method="ssprk2"))
        amr.set_density("plasma", rho)
        amr.install_program(compiled.so_path)
        amr.step(1e-3)  # the per-level macro-step runs over the hierarchy (AmrProgramContext)
        chk("plasma" in amr.block_names(), "the instance was bound and the AMR program installed")
        chk(amr.installed_program_hash() != "", "the installed program hash is recorded")
        chk(np.all(np.isfinite(np.array(amr.density("plasma")))),
            "one per-level macro-step ran and kept a finite coarse density")
        print("OK  AMR program installed + one step ran end to end (AmrProgramContext)")
    except RuntimeError as exc:
        msg = str(exc)
        # Only an environmental skip (no compiler/Kokkos at runtime) is acceptable now; the driver lands.
        if "Kokkos" in msg or "dlopen" in msg or "compile" in msg:
            print("skip (AMR .so could not load/run in this environment: %s)" % msg[:140])
            return
        raise


def test_multi_block_amr_program_install_fails_loud():
    """(4) ADC-508 fix 2: a Program binding MORE THAN ONE block must FAIL LOUD at install on AMR -- the
    v1 per-level AmrProgramContext driver wires a single block only. Compile a 2-block Program
    (target='amr_system'), add BOTH blocks, install -> a clear RuntimeError naming the v1 limit + the
    alternatives (native AMR route / System). Kokkos-gated (needs a compiler to build the 2-block .so);
    self-skips otherwise. The 2-block .so carries pops_program_block_count == 2, the signal the C++
    install_program guard checks."""
    print("== a multi-block AMR Program install fails loud (fix 2: v1 single-block-AMR limit) ==")
    amr = AmrSystem(n=N, L=1.0, regrid_every=0)
    if not hasattr(amr, "install_program"):
        print("skip (_pops lacks AmrSystem.install_program; rebuild _pops)")
        return
    m = _euler_model("adc508_2block_model")
    try:
        compiled = pops.codegen.compile_problem(
            model=m, time=_two_block_program(m), target="amr_system")
        block_cm = m.compile(backend="production", target="amr_system")
    except RuntimeError as exc:
        print("skip (no Kokkos to build the 2-block AMR .so: %s)" % str(exc)[:120])
        return

    x = (np.arange(N) + 0.5) / N
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    try:
        # Add BOTH blocks so the name-binding loop passes and the guard (not the name bind) is what fires.
        for blk in ("plasma", "plasma2"):
            amr.add_equation(blk, block_cm, spatial=pops.FiniteVolume(),
                             time=pops.Explicit(method="ssprk2"))
            amr.set_density(blk, rho)
    except RuntimeError as exc:
        print("skip (could not add the two AMR blocks: %s)" % str(exc)[:120])
        return
    try:
        amr.install_program(compiled.so_path)
        chk(False, "a 2-block AMR Program install must raise (v1 single-block limit)")
    except RuntimeError as exc:
        msg = str(exc)
        if "Kokkos" in msg or "dlopen" in msg:
            print("skip (the 2-block AMR .so could not load: %s)" % msg[:120])
            return
        chk("multi-block AMR Program is not supported in v1" in msg
            or ("binds" in msg and "blocks" in msg and "v1" in msg),
            "the v1 multi-block-AMR fail-loud message fired: %s" % msg[:200])
        chk("System" in msg or "ADC-503" in msg or "native" in msg.lower(),
            "the message points at the alternative (native AMR route / System)")


def test_amr_program_context_fail_loud_stubs_exist():
    """(4) ADC-508 fix 3: the AmrProgramContext header declares fail-loud stubs for the 15 ctx ops the
    codegen can emit (Schur / named-flux / scheduler-cache) but the v1 AMR path does not wire, so a
    Schur/scheduled Program lowers to a .so that COMPILES (the member exists) and FAILS LOUD at run
    rather than a raw 'no member named X' compile error. Host-side source assertion (no compiler): each
    op is present in the header and routes to a throw. Cross-checked against the codegen emitters: every
    ctx.<op> a program_emit_*.py can write either has a real AmrProgramContext method or a fail-loud
    stub -- no ctx op is missing from the AMR seam."""
    print("== AmrProgramContext declares fail-loud stubs for the deferred ctx ops (fix 3) ==")
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    hdr_path = os.path.join(here, "..", "..", "..", "..", "include", "pops", "runtime", "program",
                            "amr_program_context.hpp")
    if not os.path.exists(hdr_path):
        print("skip (header not found at %s; running from an installed wheel?)" % hdr_path)
        return
    hdr = open(hdr_path).read()
    # ADC-633 WIRED the condensed-Schur ops on the hierarchy (assemble_schur_coeffs /
    # apply_laplacian_coeff / schur_explicit_flux / assemble_schur_rhs / schur_reconstruct /
    # schur_energy) -- they are gone from BOTH the header stubs and the codegen emission (routed
    # through the generic condensed ops + solve_linear_matfree), so they are no longer deferred.
    # The genuinely-deferred remainder is the named-flux and scheduler-cache surface.
    deferred = ["neg_div_flux_into",
                "cache_should_update", "cache_store_aux", "cache_restore_aux", "cache_store_scratch",
                "cache_restore_scratch", "cache_accumulate_dt", "cache_effective_dt", "scheduler_error"]
    for op in deferred:
        chk(op in hdr, "AmrProgramContext declares a stub for ctx.%s" % op)
    chk("deferred_op(" in hdr or "is not wired on the AMR Program path" in hdr,
        "the deferred stubs route through a fail-loud helper (not wired on the AMR Program path)")
    # Every ctx.<op> the codegen emits must resolve on the AMR seam (a real method OR a deferred stub):
    # no raw 'no member named X' compile error for an AMR-target Program.
    import glob
    emit = ""
    for f in glob.glob(os.path.join(here, "..", "..", "..", "..", "python", "pops", "codegen",
                                    "program_emit_*.py")):
        emit += open(f).read()
    import re
    emitted = set(re.findall(r"ctx\.([a-z_]+)", emit))
    missing = [op for op in emitted if op not in hdr]
    chk(not missing,
        "every codegen-emitted ctx op resolves on AmrProgramContext (missing: %s)" % missing)


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print("FAIL %s: %s" % (fn.__name__, exc))
    print("\n%d/%d test functions passed" % (len(fns) - failed, len(fns)))
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
