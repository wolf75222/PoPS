#!/usr/bin/env python3
"""ADC-635: replay AMR history rings THROUGH in-window regrids -> bit-identical restart.

ADC-631 refused a selective (non-Dense) restart whose replay window straddled a head-of-step regrid:
the stored ring slots are remapped onto the checkpoint hierarchy, and a frozen one-shot re-step could
not reproduce the chained interpolation. ADC-635 lifts that refusal. The replay now re-steps with
regrid ACTIVE, driving the facade cursor so the ORIGINAL in-window regrid schedule fires and each
recomputed slot rides the same incremental remap chain the stored anchors rode. A coherence guard
refuses off-schedule regrid completions and a corrupted schedule fingerprint.

The composition is TWO blocks: "blk" (program-driven, ring, blob above the tag threshold) plus a
FROZEN background block "bg" (below threshold, never tagged, untouched by the Program). A single-block
compiled Program deliberately builds a coarse-only hierarchy (the ADC-508 parity layout) where
regrid() is a structural no-op; the second block keeps the historical 2-level seed so the in-window
regrids genuinely COMPLETE and remap the ring. The frozen bg block never advances, so its replay-time
state equals its original-era state at every step and the tag union stays deterministic.

The cases (all np.array_equal, no tolerance; a FRESH AmrSystem restart):

  (a) IN-WINDOW REGRID, one due step in the replay window: the regrid COMPLETES during the replay
      (asserted via the replay's fired schedule), the reconstructed ring slots AND the post-restart
      continuation are bit-identical to the uninterrupted run.
  (b) MULTIPLE in-window regrids in one replay window (a wide gap spanning >= 2 due steps): same.
  (c) NO-REGRID non-regression: a clean-window selective checkpoint still round-trips bit-identically
      (guards the new cursor driving against an off-by-one that would spuriously fire a regrid).
  (d) INVERTED GUARD: corrupting the recorded regrid-schedule fingerprint makes the restart raise the
      hard coherence error; an uncorrupted file restarts clean.

The model is the warm-start-independent passive-source class (a pointwise linear recurrence: the
provably bit-exact replay class ADC-631 documents).
Self-skips (exit 0) without pops / a built _pops / a compiler / a visible Kokkos. Pytest + __main__.
"""
import os
import sys
import tempfile

try:
    import numpy as np

    import pops
    import pops.runtime._engine_descriptors as engine
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.physics._facade import Model
    from pops.runtime._system import AmrSystem
    from pops.time._history.persistence import Interval
    from tests.python.support.typed_program import program_states, synthetic_module
except Exception as exc:  # noqa: BLE001
    print("skip test_amr_history_regrid_replay (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

N = 16
DT = 2.0e-3
_C = 0.6  # linear source S(rho) = _C*rho: the ring is load-bearing (R changes every step)
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
    phi/aux (warm-start-independent -> the provably bit-exact replay class); the tags read the density."""
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


def _state_ring_program(depth, k, name):
    """A depth-`depth` STATE ring (keep_history, Interval(k) -> stores a subset, replays the gaps).

    A MARKOV forward-Euler recurrence U^{n+1} = U^n + dt*_C*U^n (depends ONLY on U^n) so re-stepping
    from any seeded state reproduces the next state bit-for-bit -- the single-seed replay reconstructs
    the missing slots exactly. A zero-weight prev(depth-1) read declares the ring depth without making
    the recurrence multi-term (a k-term recurrence would need k seed states the single-seed replay
    cannot supply -- the documented replay class)."""
    P = pops.time.Program(name)
    module = synthetic_module("%s_state" % name, components=("rho",))
    _case, states = program_states(P, module, ("blk",))
    U = states["blk"]
    P.keep_history(U, depth=depth, checkpoint_policy=Interval(k))
    nxt = P.value("Un", U.n + P.dt * (_C * U.n) + 0.0 * U.prev(depth - 1))
    P.commit(U.next, nxt)
    P.step_strategy(pops.time.FixedDt(DT))
    return P


def _blob():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (0.15 ** 2))


def _build(program, regrid_every):
    amr = AmrSystem(n=N, L=1.0, regrid_every=regrid_every)
    if not hasattr(amr, "install_program") or not hasattr(amr, "history_names"):
        return None, "no engine"
    try:
        compiled = pops.codegen.compile_problem(model=_passive_source_model(program.name + "_m"),
                                                 time=program, target="amr_system")
        block_cm = _passive_source_model(program.name + "_b").compile(backend="production",
                                                                      target="amr_system")
        bg_cm = _passive_source_model(program.name + "_bg").compile(backend="production",
                                                                    target="amr_system")
    except RuntimeError as exc:
        return None, "compile: %s" % str(exc)[:180]
    try:
        amr.add_equation("blk", block_cm,
                         spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                         time=engine.Explicit(method="ssprk2"))
        # The FROZEN background block: >= 2 blocks keeps the historical 2-level seed (a single-block
        # Program builds a coarse-only hierarchy where regrid() is a structural no-op). The Program
        # never commits "bg", so it stays frozen at its flat sub-threshold density (never tagged).
        amr.add_equation("bg", bg_cm,
                         spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                         time=engine.Explicit(method="ssprk2"))
        amr.set_refinement(1.2)  # tags the blk blob only -> real 2-level layouts at the cadence
        amr.set_density("blk", _blob())
        amr.set_density("bg", np.full((N, N), 0.5))  # flat, below threshold: tags nothing, ever
        amr.install_program(compiled.so_path)
        authored = compiled.program
        amr._step_strategy = authored._step_strategy
        amr._step_transaction_plan = authored.transaction_plan()
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


def _run_case(program_factory, nsteps, half, label, regrid_every):
    """continuous(nsteps) vs [run(half), ckpt, FRESH restart, continue]. Returns the comparison data."""
    cont, err = _build(program_factory(), regrid_every)
    if cont is None:
        return None, err
    if int(cont.n_levels()) < 2:
        return None, "hierarchy is single-level (regrid is a structural no-op); 2 levels required"
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
            fp = "history_regrid_steps_" + h
            fingerprint = [int(s) for s in d[fp]] if fp in d else None
            stored_info[h] = (depth, sorted(stored), fingerprint)
        fresh, _ = _build(program_factory(), regrid_every)  # a FRESH AmrSystem (fresh install_program)
        fresh.restart(ckpt)
        # The in-window regrids the replay actually COMPLETED (ADC-635 facade accessor; empty on a
        # clean window). Read right after restart, before any post-restart step.
        fired = (sorted(int(s) for s in fresh._s.last_replay_regrid_steps())
                 if hasattr(fresh._s, "last_replay_regrid_steps") else None)
        rings_after_restart = _rings(fresh)
        report = fresh.last_restart_report()
        _advance(fresh, nsteps - half)
    got = np.asarray(fresh.density("blk"))
    return (ref, got, cont_rings_at_half, rings_after_restart, stored_info, report, fired), None


def _assert_bit_identical(out, label, want_fired):
    """@p want_fired: the exact fired schedule the replay must report (None = don't assert it)."""
    ref, got, cont_rings, rest_rings, stored_info, report, fired = out
    chk(bool(stored_info) and all(len(s) < depth for depth, s, _ in stored_info.values()),
        "%s stores a SUBSET of the ring (the gap is replayed): %r"
        % (label, {h: (v[0], v[1]) for h, v in stored_info.items()}))
    if want_fired is not None:
        chk(fired == want_fired,
            "the replay COMPLETED the in-window regrids %r (fingerprint %r)"
            % (fired, {h: v[2] for h, v in stored_info.items()}))
    ok_rings = all(np.array_equal(a, b) for h in cont_rings
                   for a, b in zip(cont_rings[h], rest_rings.get(h, []), strict=False))
    chk(ok_rings, "every post-restart ring slot (recomputed included) equals uninterrupted bit-for-bit")
    chk(np.array_equal(ref, got),
        "%s continuation is BIT-IDENTICAL to uninterrupted (max|d| = %.3e)"
        % (label, float(np.abs(ref - got).max())))


def test_a_in_window_regrid_bit_identical():
    print("== (a) one in-window regrid in the replay window -> bit-identical ring + continuation ==")
    # depth 3, Interval(2) -> stored {0,2}; ckpt at m=6, regrid_every=4: the re-step producing slot 1
    # runs at cursor 4 (a due regrid) -- INSIDE the replay window; the 2-level blob tags make it COMPLETE.
    out, err = _run_case(lambda: _state_ring_program(3, 2, "adc635_a"), nsteps=10, half=6,
                         label="a", regrid_every=4)
    if out is None:
        print("skip (%s)" % err)
        return
    _assert_bit_identical(out, "in-window-regrid", want_fired=[4])


def test_b_multiple_in_window_regrids_bit_identical():
    print("== (b) multiple in-window regrids in one replay window -> bit-identical ==")
    # depth 5, Interval(4) -> stored {0,4}; ckpt at m=8, regrid_every=2: the gap 0..4 re-steps at
    # cursors 4,5,6,7 (producing slots 3,2,1,0) -> due regrids at BOTH cursor 4 and cursor 6.
    out, err = _run_case(lambda: _state_ring_program(5, 4, "adc635_b"), nsteps=12, half=8,
                         label="b", regrid_every=2)
    if out is None:
        print("skip (%s)" % err)
        return
    _, _, _, _, stored_info, _, _ = out
    chk(any(fp is not None and len(fp) >= 2 for _, _, fp in stored_info.values()),
        "the replay window records >= 2 in-window regrids: %r"
        % {h: v[2] for h, v in stored_info.items()})
    _assert_bit_identical(out, "multi-in-window-regrid", want_fired=[4, 6])


def test_c_clean_window_non_regression():
    print("== (c) no-regrid non-regression: a clean replay window still round-trips bit-identically ==")
    # depth 3, Interval(2) -> stored {0,2}; ckpt at m=8, regrid_every=4: the re-steps run at cursors 6,7
    # (no due regrid) -- the ADC-631 clean-window case, an empty fingerprint AND an empty fired schedule.
    out, err = _run_case(lambda: _state_ring_program(3, 2, "adc635_c"), nsteps=12, half=8,
                         label="c", regrid_every=4)
    if out is None:
        print("skip (%s)" % err)
        return
    _, _, _, _, stored_info, _, _ = out
    chk(all(fp is not None and len(fp) == 0 for _, _, fp in stored_info.values()),
        "the clean-window fingerprint is EMPTY (no in-window regrid): %r"
        % {h: v[2] for h, v in stored_info.items()})
    _assert_bit_identical(out, "clean-window", want_fired=[])


def test_d_corrupted_fingerprint_refused():
    print("== (d) inverted guard: a corrupted regrid-schedule fingerprint fails the restart LOUD ==")
    run, err = _build(_state_ring_program(3, 2, "adc635_d"), regrid_every=4)
    if run is None:
        print("skip (%s)" % err)
        return
    _advance(run, 6)  # ckpt at m=6: one in-window regrid at cursor 4
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = run.checkpoint(os.path.join(tmp, "d"))
        d = dict(np.load(ckpt, allow_pickle=False))
        # An uncorrupted file restarts clean.
        clean, _ = _build(_state_ring_program(3, 2, "adc635_d"), regrid_every=4)
        ok = True
        try:
            clean.restart(ckpt)
        except Exception as exc:  # noqa: BLE001
            ok = False
            print("    (unexpected clean-restart failure: %s)" % str(exc)[:120])
        chk(ok, "the uncorrupted checkpoint restarts clean")
        # Corrupt the recorded fingerprint of the ring (claim a regrid at a step that is not due).
        fp_key = next((k for k in d if k.startswith("history_regrid_steps_")), None)
        chk(fp_key is not None, "the checkpoint carries a history_regrid_steps_ fingerprint")
        if fp_key is None:
            return
        d[fp_key] = np.asarray([999], dtype=np.int64)
        bad = os.path.join(tmp, "d_bad.npz")
        np.savez_compressed(bad, **d)
        fresh, _ = _build(_state_ring_program(3, 2, "adc635_d"), regrid_every=4)
        raised = ""
        try:
            fresh.restart(bad)
        except (ValueError, RuntimeError) as exc:
            raised = str(exc)
    chk("regrid" in raised and ("inconsistent" in raised or "corrupted" in raised),
        "the corrupted fingerprint is REFUSED loud (got: %s)" % (raised[:140] or "<none>"))


if __name__ == "__main__":
    test_a_in_window_regrid_bit_identical()
    test_b_multiple_in_window_regrids_bit_identical()
    test_c_clean_window_non_regression()
    test_d_corrupted_fingerprint_refused()
    print("FAILURES:", _fails)
    sys.exit(1 if _fails else 0)
