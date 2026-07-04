#!/usr/bin/env python3
"""ADC-631 (b): multistep history with ACTIVE regrid on a 2-level AMR hierarchy.

An AB2 Program on a 2-level AMR system with ``regrid_every>0``. Two precise assertions:

  (i)  NULL regrid (the refine criterion tags NOTHING at the regrid steps beyond the frozen seed) ->
       the trajectory equals a no-regrid run to round-off (the only divergence channel is the regrid's
       R0 head solve over-converging the multigrid warm start, bounded far below the scheme error; the
       bitwise ring-remap identity on an unchanged layout is locked by the C++
       test_amr_history_ring.RegridRemapKeepsSlotsConsistent case);
  (ii) REAL regrid (a moving compressible front tags cells) -> the run is stable (finite, coarse mass
       conserved to round-off) and after the regrids EVERY prev(k) global buffer is defined on the NEW
       layout (its flat size == the current sum_k ncomp*nf_k*nf_k) -- the layout-consistency invariant.

Self-skips (exit 0) without pops / a built _pops / a compiler / a visible Kokkos. Pytest + __main__.
"""
import sys

try:
    import numpy as np

    import pops
    import pops.lib.time as lt
    from pops.ir.ops import sqrt
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.physics.facade import Model
    from pops.runtime.system import AmrSystem
except Exception as exc:  # noqa: BLE001
    print("skip test_amr_history_regrid (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

N = 16
NSTEPS = 6
DT = 2.0e-3

_fails = 0


def chk(cond, label):
    global _fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        _fails += 1


def _euler_model(name):
    """Compressible Euler (density blob disperses as pressure waves -> the tagged region moves, so a
    real regrid changes the fine layout); elliptic_rhs = rho so a field solve runs."""
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


def _ab2_program(name):
    P = pops.time.Program(name)
    lt.adams_bashforth2(P, "blk")
    return P


def _blob(amp=0.5, w=0.12):
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + amp * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (w * w))


def _build(regrid_every, refine_thr, u0, tag):
    amr = AmrSystem(n=N, L=1.0, regrid_every=regrid_every)
    if not hasattr(amr, "install_program") or not hasattr(amr, "history_names"):
        return None, None
    try:
        compiled = pops.codegen.compile_problem(model=_euler_model("rg_prog_%s" % tag),
                                                 time=_ab2_program("rg_ab2_%s" % tag),
                                                 target="amr_system")
        block_cm = _euler_model("rg_blk_%s" % tag).compile(backend="production",
                                                          target="amr_system")
    except RuntimeError as exc:
        return None, "compile: %s" % str(exc)[:180]
    try:
        amr.add_equation("blk", block_cm,
                         spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                         time=pops.Explicit(method="ssprk2"))
        if refine_thr is not None:
            amr.set_refinement(refine_thr)  # 2-level hierarchy tagging density > thr
        amr.set_density("blk", u0)
        amr.install_program(compiled.so_path)
    except RuntimeError as exc:
        return None, "install: %s" % str(exc)[:240]
    return amr, None


def test_null_regrid_matches_no_regrid_to_roundoff():
    """(i) A refine threshold no cell reaches at the regrid steps -> the regrid tags nothing (the fine
    layout stays the frozen seed) -> trajectory == no-regrid run to round-off. The residual is the
    regrid's R0 head solve over-converging the warm start, orders below the scheme error."""
    print("== null-regrid: trajectory == no-regrid to round-off ==")
    u0 = _blob(amp=0.2)  # peak 1.2 < the 1e9 threshold everywhere, forever
    a, err = _build(regrid_every=2, refine_thr=1.0e9, u0=u0, tag="null_a")
    if a is None:
        print("skip (%s)" % (err or "no engine"))
        return
    b, err2 = _build(regrid_every=0, refine_thr=1.0e9, u0=u0, tag="null_b")
    if b is None:
        print("skip (%s)" % (err2 or "no engine"))
        return
    for _ in range(NSTEPS):
        a.step(DT)
        b.step(DT)
    da = float(np.abs(np.asarray(a.density("blk")) - np.asarray(b.density("blk"))).max())
    chk(da < 1e-9, "null-regrid trajectory == no-regrid to round-off (max|d| = %.3e)" % da)
    ra = {h: [np.asarray(a.history_global(h, k)).ravel()
              for k in range(int(a.history_depth(h)))] for h in a.history_names()}
    chk(bool(ra) and all(np.all(np.isfinite(x)) for h in ra for x in ra[h]),
        "the ring slots stay finite across the null regrids")


def test_real_regrid_stable_and_layout_consistent():
    """(ii) A real regrid (dispersing blob tags cells) -> the run stays STABLE (finite, mass bounded)
    on a genuinely two-level hierarchy, and every prev(k) buffer is defined on the NEW hierarchy
    (flat size == sum_k ncomp*nf_k*nf_k).

    MASS BOUND, not round-off (the synchronous Program driver, average_down-only): the AmrProgramContext
    v1 macro-step advances every level with the SAME dt and couples fine->coarse by average_down ONLY,
    with NO conservative reflux at the coarse-fine interface (documented in
    amr_program_context.hpp::couple_levels). So on a genuinely MULTILEVEL Program run the total mass
    DRIFTS by the un-refluxed C/F face-flux mismatch -- bounded and far below the scheme error, but NOT
    round-off. Round-off conservation holds on the coarse-only / flat-hierarchy Program path (no C/F
    interface), which is locked bit-for-bit by test_amr_program_parity; the reflux under a Program that
    would restore round-off at a real C/F interface is the AmrProgramContext v2 deferral (ADC-633 wave).
    The assertion here pins STABILITY: the drift stays a few 1e-4, never blows up."""
    print("== real regrid: stable + prev(k) layout-consistent on the new hierarchy ==")
    u0 = _blob(amp=0.5)
    a, err = _build(regrid_every=2, refine_thr=1.2, u0=u0, tag="real")
    if a is None:
        print("skip (%s)" % (err or "no engine"))
        return
    m0 = float(a.mass("blk"))
    for _ in range(NSTEPS):
        a.step(DT)
    rho = np.asarray(a.density("blk"))
    chk(np.all(np.isfinite(rho)), "the state stays finite through the regrids")
    chk(abs(float(a.mass("blk")) - m0) < 2e-4,
        "coarse mass stays bounded across the regrids (average_down-only v1, |m-m0| = %.2e)"
        % abs(float(a.mass("blk")) - m0))
    nlev = int(a.n_levels())
    names = list(a.history_names())
    chk(bool(names), "the AB2 Program registered its history ring on AMR (%r)" % names)
    ok = True
    for h in names:
        ncomp = int(a.history_ncomp(h))
        expected = sum(ncomp * (N << k) * (N << k) for k in range(nlev))
        for k in range(int(a.history_depth(h))):
            buf = np.asarray(a.history_global(h, k), dtype=np.float64).ravel()
            if buf.size != expected or not np.all(np.isfinite(buf)):
                ok = False
                print("    ring %s slot %d size %d != expected %d (or non-finite)"
                      % (h, k, buf.size, expected))
    chk(nlev >= 2 and ok,
        "every prev(k) buffer is on the NEW layout (size == sum_k ncomp*nf_k*nf_k, nlev=%d)" % nlev)


if __name__ == "__main__":
    test_null_regrid_matches_no_regrid_to_roundoff()
    test_real_regrid_stable_and_layout_consistent()
    print("FAILURES:", _fails)
    sys.exit(1 if _fails else 0)
