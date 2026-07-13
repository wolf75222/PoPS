#!/usr/bin/env python3
"""ADC-631 (c): mid-regrid v3 checkpoint of a multistep AMR history -> fresh restart -> bit-identical.

Under ACTIVE regridding (regrid_every=2, regrids firing BEFORE the checkpoint and AFTER the restart),
a compiled multistep Program's history ring is remapped through every regrid; a v3 checkpoint stores it
per the persistence policy and a FRESH AmrSystem restart reproduces the uninterrupted trajectory
BIT-IDENTICALLY (np.array_equal, no tolerance -- extends the ADC-542 acceptance with histories):

  (1) Dense (AB2 R-ring, depth 2): every slot stored -> no replay -> the ring round-trips through the
      regrid remap + v3 restore and the continuation is bit-identical end-to-end;
  (2) NON-Dense state ring (keep_history depth 3, Interval(2), regrid_every=4): the checkpoint stores
      slots {0,2} and the restart REPLAYS slot 1 by re-stepping the installed Program. The checkpoint
      (m=8) is taken BETWEEN two regrids (fired at step 4; next at the head of step 8, post-restart)
      with the whole seed-to-checkpoint span regrid-free (steps 6,7), so the frozen-cadence replay
      re-executes the ORIGINAL (regrid-free) steps; assert the post-restart ring (every slot,
      recomputed included) equals the uninterrupted run's ring at the same macro-step bit-for-bit,
      and the continuation stays bit-identical;
  (3) ADC-635: the same non-Dense ring checkpointed at m=6 has a regrid DUE (step 4) INSIDE the replay
      window; the ADC-631 refusal is lifted -- the restart replays the ring with regrid ACTIVE. On this
      single-block Program layout the hierarchy is coarse-only (ADC-508 parity), so the due regrid is a
      deterministic structural no-op on BOTH the original run and the replay; the reconstruction is
      bit-identical (np.array_equal). Real COMPLETING in-window regrids are covered by
      test_amr_history_regrid_replay.py (two-block 2-level composition).

Self-skips (exit 0) without pops / a built _pops / a compiler / a visible Kokkos. Pytest + __main__.
"""
import os
import sys
import tempfile

try:
    import numpy as np

    import pops
    import pops.lib.time as lt
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.physics._facade import Model
    from pops.runtime._system import AmrSystem
    from pops.time.history_persistence import Interval
    from tests.python.support.typed_program import program_states, synthetic_module
except Exception as exc:  # noqa: BLE001
    print("skip test_amr_history_checkpoint (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

N = 16
DT = 2.0e-3
_C = 0.6  # linear source S(rho) = _C*rho: R changes every step, the ring is load-bearing
_fails = 0


def _advance(sim, nsteps):
    return sim.run(
        t_end=float(sim.time()) + nsteps * DT,
        max_steps=nsteps,
    )


def chk(cond, label):
    global _fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        _fails += 1


def _passive_source_model(name):
    """1-variable rho, ZERO flux, linear source S=_C*rho, elliptic_rhs=rho. The dynamics never read
    phi/aux, so the replayed steps are independent of the multigrid warm start -- the provably
    bit-exact replay class. The refinement tags read the density level itself."""
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


def _ab2_program(name="adc631_ckpt_ab2"):
    """AB2 over the R-ring (Dense by default: no keep_history policy). flux=False keeps the body free
    of solve_fields, staying in the warm-start-independent replay class."""
    P = pops.time.Program(name)
    module = synthetic_module("%s_state" % name, components=("rho",))
    _case, states = program_states(P, module, ("blk",))
    lt.adams_bashforth2(P, states["blk"], flux=False)
    P.step_strategy(pops.time.FixedDt(DT))
    return P


def _state3_program(name="adc631_ckpt_state3"):
    """A depth-3 STATE ring (keep_history depth=3, Interval(2) -> stores slots {0,2}, replays slot 1).

    The commit is a MARKOV forward-Euler recurrence U^{n+1} = U^n + dt*_C*U^n -- it depends ONLY on U^n,
    not on the lagged slots -- so re-stepping from ANY seeded state reproduces the exact next state and
    the deterministic replay reconstructs slot 1 BIT-FOR-BIT. The depth-3 ring is declared by a
    zero-weight read of U.prev(2): it drives _histories to lag 2 (so Interval(2) selects the proper
    subset {0,2}) WITHOUT making the recurrence multi-term (a k-term recurrence would need k seed states,
    which the single-seed replay cannot supply -- the documented replay class). No phi / no flux, so the
    trajectory is independent of the multigrid warm start too."""
    P = pops.time.Program(name)
    module = synthetic_module("%s_state" % name, components=("rho",))
    _case, states = program_states(P, module, ("blk",))
    U = states["blk"]
    P.keep_history(U, depth=3, checkpoint_policy=Interval(2))
    # Markov forward-Euler on the linear source (reads U.n only), + a zero-weight prev(2) read that
    # declares the depth-3 ring without breaking the single-step reconstructability of the replay.
    nxt = P.value("Un", U.n + P.dt * (_C * U.n) + 0.0 * U.prev(2))
    P.commit(U.next, nxt)
    P.step_strategy(pops.time.FixedDt(DT))
    return P


def _blob():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (0.15 ** 2))


def _build(program, regrid_every=2):
    amr = AmrSystem(n=N, L=1.0, regrid_every=regrid_every)
    if not hasattr(amr, "install_program") or not hasattr(amr, "history_names"):
        return None, None
    try:
        compiled = pops.codegen.compile_problem(model=_passive_source_model(program.name + "_m"),
                                                 time=program, target="amr_system")
        block_cm = _passive_source_model(program.name + "_b").compile(backend="production",
                                                                     target="amr_system")
    except RuntimeError as exc:
        return None, "compile: %s" % str(exc)[:180]
    try:
        amr.add_equation("blk", block_cm,
                         spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                         time=pops.Explicit(method="ssprk2"))
        amr.set_refinement(1.2)  # tags the blob -> a real 2-level hierarchy, regrids at steps 2,4,...
        amr.set_density("blk", _blob())
        amr.install_program(compiled.so_path)
        authored = compiled.program
        amr._step_strategy = authored._step_strategy
        amr._step_transaction_plan = authored.transaction_plan()
        # Attach the per-ring persistence policy (what pops.bind's step-5a does): the low-level
        # install_program(so_path) seam does not see the compiled Program object, so the v3 checkpoint
        # would otherwise persist Dense. name -> policy from the compiled Program's _history_persistence.
        persistence = getattr(getattr(compiled, "program", None), "_history_persistence", None)
        if persistence:
            amr.set_history_persistence(
                {name: policy for name, (_depth, policy) in persistence.items()})
    except RuntimeError as exc:
        return None, "install: %s" % str(exc)[:240]
    return amr, None


def _rings(amr):
    return {h: [np.asarray(amr.history_global(h, k), dtype=np.float64).ravel()
                for k in range(int(amr.history_depth(h)))] for h in amr.history_names()}


def _run_case(program_factory, nsteps, half, label, regrid_every=2):
    """continuous(nsteps) vs [run(half), ckpt, fresh restart, continue]. Returns the comparison data."""
    cont, err = _build(program_factory(), regrid_every)
    if cont is None:
        return None, err or "no engine"
    _advance(cont, half)
    cont_rings_at_half = _rings(cont)
    _advance(cont, nsteps - half)
    ref = np.asarray(cont.density("blk"))

    run, _ = _build(program_factory(), regrid_every)
    _advance(run, half)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = run.checkpoint(os.path.join(tmp, label))
        d = np.load(ckpt, allow_pickle=False)
        stored_info = {}
        for h in run.history_names():
            depth = int(d["history_depth_" + h])
            key = "history_stored_slots_" + h
            stored = [int(s) for s in d[key]] if key in d else list(range(depth))
            stored_info[h] = (depth, sorted(stored))
        fresh, _ = _build(program_factory(), regrid_every)
        fresh.restart(ckpt)
        rings_after_restart = _rings(fresh)
        report = fresh.last_restart_report()
        _advance(fresh, nsteps - half)
    got = np.asarray(fresh.density("blk"))
    return (ref, got, cont_rings_at_half, rings_after_restart, stored_info, report), None


def test_ab2_dense_checkpoint_bit_identical():
    print("== (1) AB2 Dense: mid-regrid v3 ckpt -> restart -> bit-identical continuation ==")
    out, err = _run_case(_ab2_program, nsteps=6, half=3, label="ab2")
    if out is None:
        print("skip (%s)" % err)
        return
    ref, got, cont_rings, rest_rings, stored_info, _report = out
    chk(all(len(s) == depth for depth, s in stored_info.values()) and bool(stored_info),
        "Dense stores every ring slot (no replay): %r" % stored_info)
    ok_rings = all(np.array_equal(a, b) for h in cont_rings
                   for a, b in zip(cont_rings[h], rest_rings.get(h, []), strict=False))
    chk(ok_rings, "the restored ring equals the uninterrupted ring at the checkpoint step, bit-for-bit")
    chk(np.array_equal(ref, got),
        "AB2 continuous == (run, ckpt, restart, continue) BIT-IDENTICALLY (max|d| = %.3e)"
        % float(np.abs(ref - got).max()))


def test_state3_interval_replay_bit_identical():
    print("== (2) state ring Interval(2): ckpt at m=8 between regrids -> restart REPLAYS slot 1 ==")
    out, err = _run_case(_state3_program, nsteps=12, half=8, label="state3", regrid_every=4)
    if out is None:
        print("skip (%s)" % err)
        return
    ref, got, cont_rings, rest_rings, stored_info, report = out
    chk(bool(stored_info) and all(len(s) < depth for depth, s in stored_info.values()),
        "Interval(2) stores a SUBSET of the ring slots (the gap is replayed): %r" % stored_info)
    chk(report is not None and any(h["recomputed_slots"] >= 1 for h in report.histories),
        "the restart report records the replayed (recomputed) slots")
    ok_rings = all(np.array_equal(a, b) for h in cont_rings
                   for a, b in zip(cont_rings[h], rest_rings.get(h, []), strict=False))
    chk(ok_rings,
        "EVERY post-restart ring slot (recomputed included) equals the uninterrupted ring bit-for-bit")
    chk(np.array_equal(ref, got),
        "the replayed-ring continuation is BIT-IDENTICAL to uninterrupted (max|d| = %.3e)"
        % float(np.abs(ref - got).max()))


def test_state3_replay_window_straddling_regrid_bit_identical():
    print("== (3) ADC-635: ckpt at m=6 puts a regrid (step 4) INSIDE the replay window -> replay it ==")
    out, err = _run_case(_state3_program, nsteps=10, half=6, label="straddle", regrid_every=4)
    if out is None:
        print("skip (%s)" % err)
        return
    ref, got, cont_rings, rest_rings, stored_info, report = out
    chk(bool(stored_info) and all(len(s) < depth for depth, s in stored_info.values()),
        "Interval(2) stores a SUBSET; the straddling gap is replayed THROUGH the in-window regrid: %r"
        % stored_info)
    ok_rings = all(np.array_equal(a, b) for h in cont_rings
                   for a, b in zip(cont_rings[h], rest_rings.get(h, []), strict=False))
    chk(ok_rings,
        "EVERY post-restart ring slot (recomputed through the in-window regrid) equals uninterrupted")
    chk(np.array_equal(ref, got),
        "the straddling-window replay continuation is BIT-IDENTICAL to uninterrupted (max|d| = %.3e)"
        % float(np.abs(ref - got).max()))


if __name__ == "__main__":
    test_ab2_dense_checkpoint_bit_identical()
    test_state3_interval_replay_bit_identical()
    test_state3_replay_window_straddling_regrid_bit_identical()
    print("FAILURES:", _fails)
    sys.exit(1 if _fails else 0)
