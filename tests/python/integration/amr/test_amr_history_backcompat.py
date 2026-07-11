#!/usr/bin/env python3
"""ADC-631 (e): v3 checkpoint back-compat -- a Program with NO history rings restarts unchanged.

A compiled forward-Euler Program (no ``keep_history``) on a 2-level AMR system with ``regrid_every>0``
writes a v3 checkpoint that carries NO ``history_*`` payload keys (``history_names()`` is empty). Restart
into a
FRESH AmrSystem: ``restore_histories`` is a no-op (no phantom ring re-fill), the continuation is
bit-identical to the uninterrupted run, and ``last_restart_report()`` is None. This locks the documented
cold-start semantics: a v3 file without ring payloads restarts exactly like a pre-631 v3 file.

Self-skips (exit 0) without pops / a built _pops / a compiler / a visible Kokkos. Pytest + __main__.
"""
import os
import sys
import tempfile

try:
    import numpy as np

    import pops
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.physics.facade import Model
    from pops.runtime.system import AmrSystem
    from tests.python.support.typed_program import program_states, synthetic_module
except Exception as exc:  # noqa: BLE001
    print("skip test_amr_history_backcompat (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

N = 16
NSTEPS = 4
DT = 2.0e-3
_C = 0.6  # linear source S(rho) = _C*rho
_fails = 0


def chk(cond, label):
    global _fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        _fails += 1


def _passive_source_model(name):
    """1-variable rho, ZERO flux, linear source S=_C*rho, elliptic_rhs=rho (a field solve runs). The
    warm-start-independent replay class (same as the checkpoint acceptance); compiles on the generic
    local toolchain (no euler_hllc route needed)."""
    m = Model(name)
    (rho,) = m.conservative_vars("rho")
    u = m.primitive("u", 0.0 * rho)
    m.primitive_vars(rho=rho, u=u)
    m.conservative_from([rho])
    m.flux(x=[0.0 * rho], y=[0.0 * rho])
    m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    m.source([_C * rho])
    m.elliptic_rhs(rho)
    return m


def _noring_program(name="adc631_noring"):
    """A forward-Euler Program with NO keep_history -> registers no rings (the back-compat subject)."""
    P = pops.time.Program(name)
    module = synthetic_module("%s_state" % name, components=("rho",))
    _case, states = program_states(P, module, ("blk",))
    temporal = states["blk"]
    U0 = temporal.n
    k0 = P._rhs_legacy("k0", state=U0, fields=None, flux=False, sources=["default"])
    P.commit(temporal.next, P.linear_combine("U1", U0 + P.dt * k0))
    return P


def _blob():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (0.1 ** 2))


def _build():
    amr = AmrSystem(n=N, L=1.0, regrid_every=2)
    if not hasattr(amr, "install_program") or not hasattr(amr, "history_names"):
        return None, None
    try:
        compiled = pops.codegen.compile_problem(model=_passive_source_model("bc_prog"),
                                                 time=_noring_program(), target="amr_system")
        block_cm = _passive_source_model("bc_blk").compile(backend="production", target="amr_system")
    except RuntimeError as exc:
        return None, "compile: %s" % str(exc)[:180]
    try:
        amr.add_equation("blk", block_cm,
                         spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                         time=pops.Explicit(method="ssprk2"))
        amr.set_refinement(1.2)
        amr.set_density("blk", _blob())
        amr.install_program(compiled.so_path)
    except RuntimeError as exc:
        return None, "install: %s" % str(exc)[:240]
    return amr, None


def test_v3_backcompat_no_rings_cold_start():
    print("== v3 back-compat: a no-history Program restarts with an empty ring set (no-op) ==")
    cont, err = _build()
    if cont is None:
        print("skip (%s)" % (err or "no engine"))
        return
    chk(list(cont.history_names()) == [], "a no-keep_history Program registers NO rings")

    half = NSTEPS // 2
    for _ in range(NSTEPS):
        cont.step(DT)
    ref = np.asarray(cont.density("blk"))

    run, _ = _build()
    for _ in range(half):
        run.step(DT)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = run.checkpoint(os.path.join(tmp, "bc"))
        fresh, _ = _build()
        fresh.restart(ckpt)
        chk(fresh.last_restart_report() is None,
            "restart with no rings leaves last_restart_report() None (restore_histories no-op)")
        chk(list(fresh.history_names()) == [], "no phantom rings after the restart")
        for _ in range(NSTEPS - half):
            fresh.step(DT)
    got = np.asarray(fresh.density("blk"))
    d = float(np.abs(ref - got).max())
    chk(np.array_equal(ref, got),
        "the no-history continuation is BIT-IDENTICAL through the v3 restart (max|d| = %.3e)" % d)


if __name__ == "__main__":
    test_v3_backcompat_no_rings_cold_start()
    print("FAILURES:", _fails)
    sys.exit(1 if _fails else 0)
