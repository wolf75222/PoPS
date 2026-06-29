#!/usr/bin/env python3
"""Spec 6 sec.11 (ADC-508): the compiled-problem install seam on AmrSystem.

``AmrSystem::install_problem(so_path)`` is the AMR counterpart of ``System::install_problem``: it
dlopens a generated ``problem.so`` (compiled from ``layout=AMR(...)`` -> the .so exports
``pops_install_program_amr``), checks its ABI key, runs the section-24 requirement validation (block
instance / solver), binds the Program blocks BY NAME, seeds the per-PROGRAM-block ``RuntimeParams``
from the .so metadata, then installs the macro-step body. ``sim.install(compiled, ...)`` routes the
compiled problem plus ``params=`` and ``cadence=`` onto these AMR seams.

The per-level macro-step DRIVER (``AmrProgramContext``) has LANDED (ADC-508): the generated
``pops_install_program_amr`` constructs an ``AmrProgramContext`` and installs the SYNCHRONOUS per-level
macro-step (the identical lowered body wrapped in a per-level loop). This test asserts:

  1) the codegen emits ``pops_install_program_amr`` for ``layout=AMR(...)`` (building an
     AmrProgramContext + a per-level loop) and NOT for the System default (host-side string check);
  2) the AMR install SEAM (``set_program_cadence`` / ``set_program_params``) is reachable on a built
     ``_pops`` and validates its arguments (cadence >= 1; an unseeded program block rejects a set);
  3) (Kokkos-gated) ``compile_problem(..., layout=AMR(...))`` builds the .so and
     ``AmrSystem.install_problem`` INSTALLS it and one per-level macro-step RUNS. The bit-identical
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
    from pops.codegen import Production
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import AMR
    from pops.physics.facade import Model
    from pops.ir.ops import sqrt
    from pops.physics.model import RuntimeParam
except Exception as exc:  # noqa: BLE001 -- pops/numpy unavailable in this interpreter
    print("skip test_amr_install_program (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

N = 16


def _amr_layout():
    return AMR(CartesianMesh(n=N, L=1.0, periodic=True))


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


def _lie_program(name="adc508_amr_prog"):
    """A single-block Lie step on 'plasma' (solve_fields then a Forward-Euler commit)."""
    P = adctime.Program(name)
    u = P.state("plasma")
    fields = P._legacy_solve_fields(u)
    r = P._legacy_rhs(state=u, fields=fields)
    P.commit("plasma", P.linear_combine("u1", u + P.dt * r))
    return P


def _two_block_program(name="adc508_amr_2block"):
    """A TWO-block Lie Program (states 'plasma' and 'plasma2'), each a Forward-Euler step. The Program
    binds 2 blocks -> the AMR install must FAIL LOUD (v1 single-block-AMR-Program limit, ADC-508 fix 2)."""
    P = adctime.Program(name)
    for blk in ("plasma", "plasma2"):
        u = P.state(blk)
        fields = P._legacy_solve_fields(u)
        r = P._legacy_rhs(state=u, fields=fields)
        P.commit(blk, P.linear_combine("u1_%s" % blk, u + P.dt * r))
    return P


def test_codegen_emits_amr_install_export():
    """(1) host-side: emit_cpp_program(layout=AMR(...)) emits pops_install_program_amr; the System
    default does NOT. The AMR export builds an AmrProgramContext and runs the body per level (the
    per-level driver has landed, ADC-508)."""
    print("== codegen emits pops_install_program_amr only for layout=AMR(...) ==")
    prog = _lie_program()
    src_sys = prog.emit_cpp_program(model=_euler_model())
    src_amr = prog.emit_cpp_program(model=_euler_model(), layout=_amr_layout())
    chk("pops_install_program(" in src_sys, "the System .so exports pops_install_program")
    chk("pops_install_program_amr" not in src_sys, "the System .so does NOT export the AMR entry")
    chk("pops_install_program_amr" in src_amr, "the AMR .so exports pops_install_program_amr")
    amr_entry = src_amr.split("pops_install_program_amr", 1)[1]
    chk("AmrProgramContext ctx(sys)" in amr_entry and "ctx.set_level(" in amr_entry,
        "the AMR install entry builds an AmrProgramContext and runs the body per level (ADC-508)")
    # public target= rejected
    try:
        prog.emit_cpp_program(model=_euler_model(), target="bogus")
        chk(False, "a public target= kwarg must raise")
    except TypeError as exc:
        chk("target" in str(exc), "public target= is rejected by the signature")


def test_amr_program_seam_validates_arguments():
    """(2) needs a built _pops (no Kokkos run): the AMR program seam is reachable and validates its
    arguments. set_program_cadence rejects substeps/stride < 1; set_program_params rejects an unseeded
    program block (no compiled Program installed -> no block has runtime params)."""
    print("== AmrSystem program seam validates its arguments (no Kokkos run) ==")
    amr = pops.AmrSystem(n=N, L=1.0)
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
    AmrSystem.install_problem installs it. The per-level AmrProgramContext driver (ADC-508) has landed,
    so the install SUCCEEDS and one step RUNS over the hierarchy (the WHOLE loader path -- dlopen + ABI
    key + name binding + param seed + the per-level macro-step -- is wired). Self-skips without a
    compiler / Kokkos. The CUDA run is the ROMEO step; bit-identical parity is test_amr_program_parity."""
    print("== AMR install_problem installs + runs the per-level driver ==")
    amr = pops.AmrSystem(n=N, L=1.0, regrid_every=0)
    if not hasattr(amr, "_install_problem_so"):
        print("skip (_pops lacks AmrSystem._install_problem_so; rebuild _pops)")
        return
    m = _euler_model()
    try:
        compiled = pops.compile_problem(model=m, time=_lie_program(), layout=_amr_layout())
        block_cm = m._compile_for_runtime(backend=Production(), target="amr_system")
    except RuntimeError as exc:
        print("skip (no Kokkos to build the AMR .so: %s)" % str(exc)[:120])
        return

    x = (np.arange(N) + 0.5) / N
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    try:
        amr._add_equation("plasma", block_cm, spatial=pops.FiniteVolume(),
                         time=pops.Explicit.ssprk2())
        amr.set_density("plasma", rho)
        amr._install_problem_so(compiled.so_path)
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


def test_multi_block_amr_program_install_runs():
    """(4) Spec 6: a Program binding MORE THAN ONE block must install and run on AMR. Compile a
    2-block Program through the public layout=AMR route, add BOTH native AMR blocks, install, and step.
    This locks the clean-break rule: multi-block AMR is not a documented public API that raises because
    plumbing is missing; the C++/codegen/runtime path must be real."""
    print("== a multi-block AMR Program installs and runs ==")
    amr = pops.AmrSystem(n=N, L=1.0, regrid_every=0)
    if not hasattr(amr, "_install_problem_so"):
        print("skip (_pops lacks AmrSystem._install_problem_so; rebuild _pops)")
        return
    m = _euler_model("adc508_2block_model")
    try:
        compiled = pops.compile_problem(model=m, time=_two_block_program(), layout=_amr_layout())
        block_cm = m._compile_for_runtime(backend=Production(), target="amr_system")
    except RuntimeError as exc:
        print("skip (no Kokkos to build the 2-block AMR .so: %s)" % str(exc)[:120])
        return

    x = (np.arange(N) + 0.5) / N
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    try:
        # Add BOTH blocks so the name-binding loop passes and the guard (not the name bind) is what fires.
        for blk in ("plasma", "plasma2"):
            amr._add_equation(blk, block_cm, spatial=pops.FiniteVolume(),
                             time=pops.Explicit.ssprk2())
            amr.set_density(blk, rho)
    except RuntimeError as exc:
        print("skip (could not add the two AMR blocks: %s)" % str(exc)[:120])
        return
    try:
        amr._install_problem_so(compiled.so_path)
        amr.step(1.0e-3)
    except RuntimeError as exc:
        msg = str(exc)
        if "Kokkos" in msg or "dlopen" in msg or "compile" in msg:
            print("skip (the 2-block AMR .so could not load/run in this environment: %s)" % msg[:140])
            return
        raise
    chk(amr.installed_program_hash() != "", "the installed multi-block AMR Program hash is recorded")
    for blk in ("plasma", "plasma2"):
        chk(np.all(np.isfinite(np.array(amr.density(blk)))),
            "multi-block AMR Program advanced %s with finite density" % blk)


def test_amr_program_context_real_seams_exist():
    """(5) Spec 6: the AmrProgramContext header declares the ctx ops the codegen can emit as real AMR
    seams. Historical fail-loud helpers for Schur / named-flux / scheduler-cache must be gone."""
    print("== AmrProgramContext declares real seams for emitted ctx ops ==")
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    hdr_path = os.path.join(here, "..", "..", "include", "pops", "runtime", "program",
                            "amr_program_context.hpp")
    if not os.path.exists(hdr_path):
        print("skip (header not found at %s; running from an installed wheel?)" % hdr_path)
        return
    hdr = open(hdr_path).read()
    deferred = ["assemble_schur_coeffs", "apply_laplacian_coeff", "schur_explicit_flux",
                "assemble_schur_rhs", "schur_reconstruct", "schur_energy", "neg_div_flux_into",
                "cache_should_update", "cache_store_aux", "cache_restore_aux", "cache_store_scratch",
                "cache_restore_scratch", "cache_accumulate_dt", "cache_effective_dt", "scheduler_error"]
    for op in deferred:
        chk(op in hdr, "AmrProgramContext declares ctx.%s" % op)
    chk("deferred_op(" not in hdr and "is not wired on the AMR Program path" not in hdr,
        "historical AMR deferred stubs are gone")
    # Every ctx.<op> the codegen emits must resolve on the AMR seam (a real method OR a deferred stub):
    # no raw 'no member named X' compile error for an AMR-target Program.
    import glob
    emit = ""
    for f in glob.glob(os.path.join(here, "..", "..", "python", "pops", "codegen",
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
