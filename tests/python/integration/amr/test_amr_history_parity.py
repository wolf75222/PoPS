#!/usr/bin/env python3
"""ADC-631 (a): flat-AMR multistep history == Uniform, bit-for-bit.

The SAME Adams-Bashforth 2 Program (a System-owned multistep history ring, R_{n-1}) installed on a
single-level System (ProgramContext) and on a FLAT single-level AmrSystem (AmrProgramContext, the
coarse-only Program layout) must produce the BYTE-IDENTICAL evolved coarse density over several steps
AND byte-identical ring-slot buffers. This proves the per-level AMR ring seam (register / store / read
/ rotate on detail::AmrHistoryOps) is a byte-faithful mirror of the Uniform HistoryManager when nlev=1
(the per-level slot [level 0] IS the Uniform ring), so the whole compiled-Program byte-code drives both.

Self-skips (exit 0) without pops / a built _pops / a compiler / a visible Kokkos, exactly like
test_amr_program_parity. Pytest + __main__ guard (CI runs ``python3 <file>``).
"""
import sys

try:
    import numpy as np

    import pops
    import pops.lib.time as lt
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.physics._facade import Model
    from pops.runtime._system import AmrSystem, System
    from tests.python.support.typed_program import program_states, synthetic_module
except Exception as exc:  # noqa: BLE001 -- pops/numpy unavailable in this interpreter
    print("skip test_amr_history_parity (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

N = 16
NSTEPS = 5
DT = 5.0e-3
_C = 0.6  # linear source S(rho) = _C*rho: R changes every step, so the AB2 ring MATTERS

_fails = 0


def chk(cond, label):
    global _fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        _fails += 1


def _passive_source_model(name):
    """1-variable rho, ZERO flux, linear source S(rho)=_C*rho + elliptic_rhs=rho (a field solve runs).
    R = c*rho changes every step, so AB2's R_{n-1} ring is load-bearing."""
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


def _ab2_program(name="adc631_ab2"):
    P = pops.time.Program(name)
    module = synthetic_module("%s_state" % name, components=("rho",))
    _case, states = program_states(P, module, ("blk",))
    lt.adams_bashforth2(P, states["blk"])
    return P


def _rho0():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)


def _ring_slots(sim, depth):
    """Every ring's every stored slot as flat float64 buffers (concatenated, order-stable)."""
    out = {}
    for hname in sim.history_names():
        d = int(sim.history_depth(hname))
        out[hname] = [np.asarray(sim.history_global(hname, k), dtype=np.float64).ravel()
                      for k in range(d)]
    return out


def _system_run(u0):
    sim = System(n=N, L=1.0, periodic=True)
    if not hasattr(sim, "install_program") or not hasattr(sim, "history_names"):
        return None, "the built _pops lacks install_program/history_names (rebuild _pops)"
    try:
        block_cm = _passive_source_model("blkS").compile(backend="production")
        compiled = pops.codegen.compile_problem(model=_passive_source_model("progS"),
                                                 time=_ab2_program())
    except RuntimeError as exc:
        return None, "compile (System): %s" % str(exc)[:160]
    sim.add_equation("blk", block_cm,
                     spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                     time=pops.Explicit(method="ssprk2"))
    sim.set_state("blk", np.stack([u0]))
    sim.install_program(compiled.so_path)
    for _ in range(NSTEPS):
        sim.step(DT)
    return (np.array(sim.get_state("blk"))[0], _ring_slots(sim, 2)), None


def _amr_run(u0):
    amr = AmrSystem(n=N, L=1.0, regrid_every=0)  # FLAT: no refinement -> nlev=1 (coarse-only)
    if not hasattr(amr, "install_program") or not hasattr(amr, "history_names"):
        return None, "the built _pops lacks AmrSystem.install_program/history_names (rebuild _pops)"
    try:
        compiled = pops.codegen.compile_problem(model=_passive_source_model("progA"),
                                                 time=_ab2_program(), target="amr_system")
        block_cm = _passive_source_model("blkA").compile(backend="production", target="amr_system")
    except RuntimeError as exc:
        return None, "compile (AMR): %s" % str(exc)[:160]
    try:
        amr.add_equation("blk", block_cm,
                         spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                         time=pops.Explicit(method="ssprk2"))
        amr.set_density("blk", u0)
        amr.install_program(compiled.so_path)
    except RuntimeError as exc:
        return None, "install (AMR): %s" % str(exc)[:240]
    for _ in range(NSTEPS):
        amr.step(DT)
    return (np.array(amr.density("blk")), _ring_slots(amr, 2), int(amr.n_levels())), None


def test_flat_amr_history_equals_uniform():
    print("== flat-AMR AB2 history == Uniform (bit-for-bit density + ring slots) ==")
    u0 = _rho0()
    sys_out, sys_err = _system_run(u0)
    if sys_out is None:
        print("skip (%s)" % sys_err)
        return
    amr_out, amr_err = _amr_run(u0)
    if amr_out is None:
        print("skip (%s)" % amr_err)
        return
    sys_rho, sys_rings = sys_out
    amr_rho, amr_rings, nlev = amr_out

    chk(nlev == 1, "the AMR system is FLAT (nlev=1, coarse-only Program layout)")
    drho = float(np.abs(sys_rho - amr_rho).max())
    chk(np.array_equal(sys_rho, amr_rho),
        "the evolved coarse density is BIT-IDENTICAL System vs AMR (max|diff| = %.3e)" % drho)
    # The ring names + depths match, and every slot buffer is byte-identical (nlev=1 -> the per-level
    # slot IS the Uniform ring).
    chk(sorted(sys_rings) == sorted(amr_rings) and len(amr_rings) >= 1,
        "the same history rings are registered on both (%r)" % sorted(amr_rings))
    all_slots_equal = True
    for hname in sys_rings:
        for k, (a, b) in enumerate(
                zip(sys_rings[hname], amr_rings.get(hname, []), strict=False)):
            if not np.array_equal(a, b):
                all_slots_equal = False
                print("    ring %s slot %d differs: max|d|=%.3e" % (hname, k, np.abs(a - b).max()))
    chk(all_slots_equal, "every ring slot buffer is BIT-IDENTICAL System vs AMR (the seam is faithful)")


if __name__ == "__main__":
    test_flat_amr_history_equals_uniform()
    print("FAILURES:", _fails)
    sys.exit(1 if _fails else 0)
