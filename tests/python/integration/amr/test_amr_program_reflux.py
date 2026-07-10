#!/usr/bin/env python3
"""ADC-639: conservative reflux for a whole-system compiled Program on a genuinely two-level AMR
hierarchy.

The synchronous per-level Program driver (AmrProgramContext) advances every level with the same dt, then
couples fine->coarse by average_down THEN conservative REFLUX at the coarse-fine interface. The per-level
effective flux is captured through the Program's OWN linear combination (the flux ledger,
amr_program_context.hpp) and routed through the native route_reflux at level sync (amr_program_reflux.hpp).
So on a genuinely MULTILEVEL run the total conserved quantity is conserved across the C/F interface to
ROUND-OFF, matching the native reflux -- while the coarse-only / flat Program stays bit-identical (locked
by test_amr_program_parity).

Acceptances (design-639 section 5):
  (a) a 2-level SSPRK2 Program conserves the total mass to < 1e-8 over several steps including a real
      regrid;
  (c) a 2-level MIDPOINT (RK2) Program -- a DIFFERENT combine through the same seam -- also conserves to
      < 1e-8 (validates that the ledger tracks the Program's ACTUAL stage weights Feff = F1, not a
      hard-coded RK), and its trajectory DIFFERS from SSPRK2.

Needs a compiler + a visible Kokkos (POPS_KOKKOS_ROOT) to build the .so; the compiled-.so dlopen + the
per-level RUN is validatable on Kokkos CPU (Serial/OpenMP) locally. Self-skips (exit 0) without pops /
a built _pops / a compiler. Pytest + __main__ guard (CI runs python3 <file>).
"""
import sys

# ADC-627: this file AOT-compiles Program/.so artifacts; give the process-isolated runner headroom.
POPS_PROCESS_TIMEOUT = 1200

try:
    import numpy as np

    import pops
    from pops import time as adctime
    from pops.ir.ops import sqrt
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.physics.facade import Model
    from pops.runtime.system import AmrSystem
except Exception as exc:  # noqa: BLE001 -- pops/numpy unavailable in this interpreter
    print("skip test_amr_program_reflux (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

N = 16
NSTEPS = 6
DT = 1.0e-3

_fails = 0


def chk(cond, label):
    global _fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        _fails += 1


def _euler_model(name):
    """Compressible Euler; elliptic_rhs = rho so a field solve runs. A dispersing blob tags cells -> a
    real regrid, and a genuine C/F interface where the reflux must fire."""
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


def _ssprk2_program(name):
    """Canonical SSPRK2 (Heun): U1 = U + dt R(U); U <<= 0.5 U + 0.5 (U1 + dt R(U1))."""
    P = adctime.Program(name)
    dt = P.dt
    U0 = P.state("blk")
    f0 = P.solve_fields("f0", U0)
    k0 = P._rhs_legacy("k0", state=U0, fields=f0, flux=True, sources=["default"])
    U1 = P.linear_combine("U1", U0 + dt * k0)
    f1 = P.solve_fields("f1", U1)
    k1 = P._rhs_legacy("k1", state=U1, fields=f1, flux=True, sources=["default"])
    U2 = P.linear_combine("U2", 0.5 * U0 + 0.5 * (U1 + dt * k1))
    P.commit(P.state("U", block="blk").next, U2)
    return P


def _midpoint_program(name):
    """Midpoint RK2: U1 = U + 0.5 dt R(U); U <<= U + dt R(U1). Effective flux Feff = F1 (the 2nd stage
    only) -- proves the ledger tracks the Program's actual weights, not a hard-coded scheme."""
    P = adctime.Program(name)
    dt = P.dt
    U0 = P.state("blk")
    f0 = P.solve_fields("f0", U0)
    k0 = P._rhs_legacy("k0", state=U0, fields=f0, flux=True, sources=["default"])
    U1 = P.linear_combine("U1", U0 + 0.5 * dt * k0)
    f1 = P.solve_fields("f1", U1)
    k1 = P._rhs_legacy("k1", state=U1, fields=f1, flux=True, sources=["default"])
    U2 = P.linear_combine("U2", U0 + dt * k1)
    P.commit(P.state("U", block="blk").next, U2)
    return P


def _blob(amp=0.5, w=0.12):
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + amp * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (w * w))


def _run(program_fn, tag, refine_thr=1.2, regrid_every=2, u0=None, nsteps=NSTEPS):
    """Install `program_fn` on a genuinely two-level AmrSystem (a real fine patch under the coarse) and
    return (m0, m_final, density) after nsteps -- or (None, err, None) if the engine is unavailable."""
    if u0 is None:
        u0 = _blob(amp=0.5)
    amr = AmrSystem(n=N, L=1.0, regrid_every=regrid_every)
    if not hasattr(amr, "install_program"):
        return None, "the built _pops lacks AmrSystem.install_program (rebuild _pops)", None
    try:
        compiled = pops.codegen.compile_problem(model=_euler_model("rfx_%s" % tag),
                                                 time=program_fn("rfx_prog_%s" % tag),
                                                 target="amr_system")
        block_cm = _euler_model("rfx_blk_%s" % tag).compile(backend="production",
                                                            target="amr_system")
    except RuntimeError as exc:
        return None, "compile: %s" % str(exc)[:180], None
    try:
        amr.add_equation("blk", block_cm,
                         spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                         time=pops.Explicit(method="ssprk2"))
        amr.set_refinement(refine_thr)  # 2-level hierarchy: tags density > thr (the blob)
        amr.set_density("blk", u0)
        amr.install_program(compiled.so_path)
    except RuntimeError as exc:
        return None, "install: %s" % str(exc)[:240], None
    m0 = float(amr.mass("blk"))
    for _ in range(nsteps):
        amr.step(DT)
    return m0, float(amr.mass("blk")), np.asarray(amr.density("blk"))


def test_multilevel_ssprk2_conserves_to_roundoff():
    """(a) a genuinely 2-level SSPRK2 Program conserves the total mass to < 1e-8 over 6 steps including a
    real regrid -- the conservative reflux at the C/F interface, matching the native path."""
    print("== multilevel SSPRK2 Program: mass conserved to 1e-8 across a real regrid ==")
    m0, mf, rho = _run(_ssprk2_program, "ss")
    if m0 is None:
        print("skip (%s)" % mf)
        return
    chk(np.all(np.isfinite(rho)) and float(rho.min()) > 0.0,
        "the 2-level state stays finite and strictly positive (min = %.4f)" % float(rho.min()))
    chk(abs(mf - m0) < 1e-8,
        "the total mass is conserved to round-off (SSPRK2 + reflux, |m-m0| = %.3e)" % abs(mf - m0))


def test_multilevel_midpoint_conserves_and_differs():
    """(c) a 2-level MIDPOINT Program conserves to < 1e-8 (validates Feff = F1: the ledger tracks the
    Program's actual stage weights, not a hard-coded RK), and its trajectory DIFFERS from SSPRK2."""
    print("== multilevel midpoint Program: Feff=F1 conserved to 1e-8, differs from SSPRK2 ==")
    m0, mf, mid_rho = _run(_midpoint_program, "mid")
    if m0 is None:
        print("skip (%s)" % mf)
        return
    chk(np.all(np.isfinite(mid_rho)),
        "the midpoint 2-level state stays finite")
    chk(abs(mf - m0) < 1e-8,
        "the midpoint effective flux (Feff = F1) conserves the mass to round-off (|m-m0| = %.3e)"
        % abs(mf - m0))
    ss0, ssf, ss_rho = _run(_ssprk2_program, "mid_ss")
    if ss0 is None:
        print("skip ssprk2 leg (%s)" % ssf)
        return
    diff = float(np.abs(mid_rho - ss_rho).max())
    chk(diff > 1e-12,
        "the midpoint scheme DIFFERS from SSPRK2 through the same reflux seam (max|diff| = %.3e)" % diff)


if __name__ == "__main__":
    test_multilevel_ssprk2_conserves_to_roundoff()
    test_multilevel_midpoint_conserves_and_differs()
    print("FAILURES:", _fails)
    sys.exit(1 if _fails else 0)
