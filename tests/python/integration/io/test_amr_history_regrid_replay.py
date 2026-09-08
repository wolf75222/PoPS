#!/usr/bin/env python3
"""Two-block AMR selective history: exact frozen-grid replay and active-remap refusal.

The original N16 two-level composition has both "blk" and "bg" advancing through affine dynamics.
The original depth3/depth5 active-regrid graphs request new fine coverage that retained state rings
cannot currently populate. They remain explicit refusal/rollback controls with unchanged geometry,
depth, schedules and step budgets. This is an M6.4/M7 limitation, not qualified live regrid replay.
Separately declared frozen-grid 1:1 runs preserve the full restart, omitted-slot and corrupted-fingerprint
oracles. Frozen-grid 2:1 selective restart is a separate refusal/rollback control.
All comparisons are bit-exact; every fresh runtime binds the same genuine compiled artifact.
"""
import os
import tempfile
import json

from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)


# Full two-block matrix compiles nine distinct artifacts; retain a bounded process budget.
POPS_PROCESS_TIMEOUT = 600

_native_missing = missing_native_compile_requirement(repo_include(), default_cxx())
if _native_missing:
    require_native_or_skip("test_amr_history_regrid_replay: %s" % _native_missing)

try:
    import numpy as np

    import pops
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.math import ddt, div
    from pops.physics import Model
    from pops.time._history.persistence import Interval
    from tests.python.integration.io.test_amr_history_checkpoint import (
        _accepted_image, _assert_accepted_image_equal,
        _assert_multirate_restart_refusal, _assert_new_coverage_refusal, _bind_checkpoint_artifact,
        _compile_checkpoint_program, _program_states,
    )
except Exception as exc:  # noqa: BLE001
    require_native_or_skip(
        "test_amr_history_regrid_replay cannot import pops/numpy: %s" % exc)

N = 16
DT = 2.0e-3
_C = 0.6  # linear source S(rho) = _C*rho: the ring is load-bearing (R changes every step)


def _advance(sim, nsteps):
    return sim.run(
        t_end=float(sim.time()) + nsteps * DT,
        max_steps=nsteps,
    )


def chk(cond, label):
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    assert cond, label


def _passive_source_model(name):
    """One scalar, zero flux and linear source S=_C*rho; refinement tags read its density.

    The field-free dynamics retain the deterministic affine replay class.
    """
    frame = Rectangle(
        "%s-domain" % name, lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model(name, frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
        waves={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
    )
    source = model.source("growth", on=state, value=(_C * rho,))
    model.rate("source_rate", equation=ddt(state) == -div(flux) + source)
    return model


def _state_ring_program(model, depth, k, name):
    """A depth-`depth` STATE ring (keep_history, Interval(k) -> stores a subset, replays the gaps).

    A strictly affine recurrence U^{n+1} = U^n + dt*_C*U^n (depends only on U^n and dt, with no
    RHS/field/operator context) makes re-stepping from any seeded state reproduce the next state
    bit-for-bit. A zero-weight prev(depth-1) read declares the ring depth without making the
    recurrence multi-term (a k-term recurrence would need k seed states the single-seed replay
    cannot supply -- the documented replay class)."""
    P = pops.time.Program(name)
    _case, states = _program_states(P, model, ("blk", "bg"))
    U = states["blk"]
    background = states["bg"]
    # The helper argument is the physical slot count; keep_history authoring takes max lag.
    P.keep_history(U, depth=depth - 1, checkpoint_policy=Interval(k))
    nxt = P.value(
        "Un",
        U.n + P.dt * _C * U.n + 0.0 * U.prev(depth - 1),
        at=U.next.point,
    )
    P.commit(U.next, nxt)
    background_next = P.value(
        "background_next",
        background.n + P.dt * _C * background.n,
        at=background.next.point,
    )
    P.commit(background.next, background_next)
    P.step_strategy(pops.time.FixedDt(DT))
    return P, _case


def _blob():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (0.15 ** 2))


_COMPILED_CASES = {}


def _build(program_factory, regrid_every, *, temporal_ratio=2):
    key = (program_factory, regrid_every, temporal_ratio)
    initials = {"blk": _blob(), "bg": np.full((N, N), 0.5)}
    if key not in _COMPILED_CASES:
        model = _passive_source_model("history_regrid_model")
        _COMPILED_CASES[key] = _compile_checkpoint_program(
            model, program_factory(model), regrid_every=regrid_every, initials=initials,
            temporal_ratio=temporal_ratio,
        )
    return _bind_checkpoint_artifact(_COMPILED_CASES[key], initials), None


def _rings(amr):
    return {
        (h, int(level)): [
            np.asarray(amr.history_global(h, level, k), dtype=np.float64).ravel().copy()
            for k in range(int(amr.history_depth(h)))
        ]
        for h in amr.history_names()
        for level in amr.history_levels(h)
    }


def _rings_equal(first, second):
    return first.keys() == second.keys() and all(
        len(first[name]) == len(second[name])
        and all(np.array_equal(left, right)
                for left, right in zip(first[name], second[name], strict=True))
        for name in first)


def _run_case(program_factory, nsteps, half, label, regrid_every, *, temporal_ratio):
    """continuous(nsteps) vs [run(half), ckpt, FRESH restart, continue]. Returns the comparison data."""
    cont, err = _build(program_factory, regrid_every, temporal_ratio=temporal_ratio)
    assert cont is not None, err
    assert int(cont.n_levels()) >= 2, \
        "history regrid replay requires a real two-level hierarchy"
    _advance(cont, half)
    cont_rings_at_half = _rings(cont)
    _advance(cont, nsteps - half)
    ref = np.asarray(cont.density("blk"))

    run, _ = _build(program_factory, regrid_every, temporal_ratio=temporal_ratio)
    _advance(run, half)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = run.checkpoint(os.path.join(tmp, label))
        d = np.load(ckpt, allow_pickle=False)
        temporal = json.loads(str(d["temporal_restart_state"]))
        accepted_dt = float.fromhex(
            temporal["controller_state"]["last_accepted_dt"])
        stored_info = {}
        for h in run.history_names():
            depth = int(d["history_depth_" + h])
            levels = [int(level) for level in d["history_levels_" + h]]
            for level in levels:
                slot_dts = np.asarray(
                    d["history_slot_dt_%s_level_%d" % (h, level)], dtype=np.float64
                ).reshape(-1)
                expected_dts = np.full(depth, DT / (temporal_ratio ** level), dtype=np.float64)
                if depth > 1:
                    expected_dts[1] = accepted_dt / (temporal_ratio ** level)
                assert np.array_equal(slot_dts, expected_dts), (
                    "AMR history slot_dt must remain the level-qualified macro dt "
                    "(level=%d, got %r)" % (level, slot_dts.tolist())
                )
            key = "history_stored_slots_" + h
            stored = [int(s) for s in d[key]] if key in d else list(range(depth))
            fp = "history_regrid_steps_" + h
            fingerprint = [int(s) for s in d[fp]] if fp in d else None
            requested = [int(s) for s in d["history_requested_stored_slots_" + h]]
            mode = str(d["history_storage_mode_" + h])
            stored_info[h] = (
                depth, sorted(requested), sorted(stored), mode, fingerprint)
        fresh, _ = _build(program_factory, regrid_every, temporal_ratio=temporal_ratio)  # a FRESH AmrSystem (fresh install_program)
        fresh.restart(ckpt)
        # Dense safety promotion and clean-window replay must both complete no structural regrid.
        # Read the native guard evidence immediately after restart.
        fired = (sorted(int(s) for s in fresh._s.last_replay_regrid_steps())
                 if hasattr(fresh._s, "last_replay_regrid_steps") else None)
        rings_after_restart = _rings(fresh)
        report = fresh.last_restart_report()
        _advance(fresh, nsteps - half)
    got = np.asarray(fresh.density("blk"))
    _assert_accepted_image_equal(_accepted_image(cont), _accepted_image(fresh))
    return (ref, got, cont_rings_at_half, rings_after_restart, stored_info, report, fired), None


def _assert_bit_identical(out, label, *, want_mode, want_fingerprint):
    ref, got, cont_rings, rest_rings, stored_info, report, fired = out
    chk(bool(stored_info) and all(mode == want_mode
                                 for _, _, _, mode, _ in stored_info.values()),
        "%s resolved storage mode is %s: %r" % (label, want_mode, stored_info))
    chk(all(fp == want_fingerprint for _, _, _, _, fp in stored_info.values()),
        "%s authenticated regrid fingerprint is %r" % (label, want_fingerprint))
    chk(all(requested == stored and len(stored) < depth
            for depth, requested, stored, _, _ in stored_info.values()),
        "%s retains selective policy storage on the frozen hierarchy" % label)
    chk(report is not None and any(row["recomputed_slots"] > 0
                                  for row in report.histories),
        "%s report records native replay of omitted slots" % label)
    chk(fired == [], "%s restart completes no structural regrid (got %r)" % (label, fired))
    ok_rings = _rings_equal(cont_rings, rest_rings)
    chk(ok_rings, "every post-restart ring slot (recomputed included) equals uninterrupted bit-for-bit")
    chk(np.array_equal(ref, got),
        "%s continuation is BIT-IDENTICAL to uninterrupted (max|d| = %.3e)"
        % (label, float(np.abs(ref - got).max())))


def test_original_active_regrid_graphs_refuse_without_mutation():
    # Preserve each original graph's physical slots, anchor spacing, program name, regrid cadence,
    # full step budget and intended checkpoint. No graph is silently frozen or shortened.
    for depth, interval, name, nsteps, half, cadence in (
        (3, 2, "adc635_a", 10, 6, 4),
        (5, 4, "adc635_b", 12, 8, 2),
        (3, 2, "adc635_c", 12, 8, 4),
        (3, 2, "adc635_d", 6, 6, 4),
    ):
        def factory(model, depth=depth, interval=interval, name=name):
            return _state_ring_program(model, depth, interval, name)

        live, _ = _build(factory, regrid_every=cadence)
        _assert_new_coverage_refusal(live, nsteps=nsteps, label="%s-ckpt%d" % (name, half))


def test_c_frozen_hierarchy_interval_replay_bit_identical():
    print("== (c) no-regrid non-regression: a clean replay window still round-trips bit-identically ==")
    # Explicitly frozen positive: same two blocks, N16, depth3, 12 steps and checkpoint at 8.
    out, err = _run_case(lambda model: _state_ring_program(model, 3, 2, "adc635_c"), nsteps=12, half=8,
                         label="c-frozen-1to1", regrid_every=0, temporal_ratio=1)
    assert out is not None, err
    _, _, _, _, stored_info, _, _ = out
    chk(all(fp is not None and len(fp) == 0
            for _, _, _, _, fp in stored_info.values()),
        "the clean-window fingerprint is EMPTY (no in-window regrid): %r"
        % {h: v[4] for h, v in stored_info.items()})
    _assert_bit_identical(
        out, "clean-window", want_mode="policy", want_fingerprint=[])


def test_b_frozen_multiple_anchor_gaps_bit_identical():
    out, err = _run_case(lambda model: _state_ring_program(model, 5, 4, "adc635_b_frozen"),
                         nsteps=12, half=8, label="b-frozen-1to1", regrid_every=0, temporal_ratio=1)
    assert out is not None, err
    _assert_bit_identical(out, "frozen-five-slot", want_mode="policy", want_fingerprint=[])
    assert all(row["recomputed_slots"] == 3 for row in out[5].histories)


def test_two_block_multirate_selective_restart_refuses_without_mutation():
    for depth, interval, name, half in ((3, 2, "adc635_c", 8), (5, 4, "adc635_b", 8)):
        def factory(model, depth=depth, interval=interval, name=name):
            return _state_ring_program(model, depth, interval, name)

        _assert_multirate_restart_refusal(
            lambda factory=factory: _build(factory, regrid_every=0, temporal_ratio=2)[0],
            half=half, label="%s-2to1" % name,
        )


def test_d_corrupted_fingerprint_refused():
    print("== (d) inverted guard: a corrupted regrid-schedule fingerprint fails the restart LOUD ==")
    def factory(model):
        return _state_ring_program(model, 3, 2, "adc635_d")

    run, err = _build(factory, regrid_every=0, temporal_ratio=1)
    assert run is not None, err
    _advance(run, 6)  # frozen-grid control isolates authenticated fingerprint validation
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = run.checkpoint(os.path.join(tmp, "d"))
        d = dict(np.load(ckpt, allow_pickle=False))
        # An uncorrupted file restarts clean.
        clean, _ = _build(factory, regrid_every=0, temporal_ratio=1)
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
        # The intended oracle is the authenticated cadence/fingerprint guard, not the outer payload
        # digest. Remove the old seal and re-seal the deliberately mutated payload with the same real
        # bound/run identities before asking a fresh identical runtime to restart it.
        from pops.runtime._checkpoint_manifest import (
            IDENTITY_KEY,
            MANIFEST_KEY,
            seal_checkpoint_payload,
        )
        del d[MANIFEST_KEY]
        del d[IDENTITY_KEY]
        seal_checkpoint_payload(run, d, runtime_kind="amr")
        bad = os.path.join(tmp, "d_bad.npz")
        np.savez_compressed(bad, **d)
        fresh, _ = _build(factory, regrid_every=0, temporal_ratio=1)
        before = _accepted_image(fresh)
        raised = ""
        try:
            fresh.restart(bad)
        except (ValueError, RuntimeError) as exc:
            raised = str(exc)
        _assert_accepted_image_equal(before, _accepted_image(fresh))
    chk("regrid fingerprint [999]" in raised and "differs from manifest cadence []" in raised,
        "the corrupted fingerprint is REFUSED loud (got: %s)" % (raised[:140] or "<none>"))


def main():
    test_original_active_regrid_graphs_refuse_without_mutation()
    test_c_frozen_hierarchy_interval_replay_bit_identical()
    test_b_frozen_multiple_anchor_gaps_bit_identical()
    test_d_corrupted_fingerprint_refused()
    test_two_block_multirate_selective_restart_refuses_without_mutation()
    print("PASS test_amr_history_regrid_replay")


if __name__ == "__main__":
    main()
