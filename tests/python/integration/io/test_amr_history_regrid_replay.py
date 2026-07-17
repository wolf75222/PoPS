#!/usr/bin/env python3
"""AMR history persistence across regrid windows is exact, explicit and bit-identical.

A selective replay window that straddles a head-of-step regrid cannot reconstruct values exactly from
anchors already remapped to the checkpoint hierarchy. The resolved checkpoint plan therefore promotes
that checkpoint instance to authenticated ``dense_regrid_safety`` storage. The authoring policy remains
visible as the requested slots; the effective slots, promotion mode and regrid schedule are persisted.
Clean hierarchy windows retain selective storage and native deterministic replay.

The composition is TWO explicitly program-driven blocks: "blk" (ring, blob above the tag threshold)
plus a background block "bg" (below threshold, never tagged). Both states advance through the same
exact conservative rate and therefore have complete Program block/flux-ledger routes. The second block
keeps a real two-level hierarchy so scheduled regrids genuinely change/remap it; this prevents a
coarse-only structural no-op from making an unsafe replay appear valid. Its smooth sub-threshold state
keeps the tag union deterministic.

The cases (all np.array_equal, no tolerance; a FRESH AmrSystem restart):

  (a) one in-window regrid promotes effective storage to dense; no replay fires;
  (b) multiple in-window regrids produce the same explicit dense safety promotion;
  (c) NO-REGRID non-regression: a clean-window selective checkpoint still round-trips bit-identically
      and actually replays the omitted slot;
  (d) INVERTED GUARD: corrupting the recorded regrid-schedule fingerprint makes the restart raise the
      hard coherence error; an uncorrupted file restarts clean.

The model is the warm-start-independent passive-source class (a pointwise linear recurrence: the
provably bit-exact replay class ADC-631 documents).
Missing native prerequisites are explicit local skips and required-lane failures. Pytest + __main__.
"""
import os
import tempfile
import hashlib
import json

from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)


_native_missing = missing_native_compile_requirement(repo_include(), default_cxx())
if _native_missing:
    require_native_or_skip("test_amr_history_regrid_replay: %s" % _native_missing)

try:
    import numpy as np

    import pops
    from pops.codegen._compile_drivers import compile_problem
    import pops.runtime._engine_descriptors as engine
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.math import ddt, div
    from pops.physics import Model
    from pops.runtime._system import AmrSystem
    from pops.time._history.persistence import Interval
    from tests.python.integration._final_field_program import compile_block_model
    from tests.python.support.typed_program import program_states
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
    """1-variable rho, ZERO flux, linear source S=_C*rho, elliptic_rhs=rho. The dynamics never read
    phi/aux (warm-start-independent -> the provably bit-exact replay class); the tags read the density."""
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

    A MARKOV forward-Euler recurrence U^{n+1} = U^n + dt*_C*U^n (depends ONLY on U^n) so re-stepping
    from any seeded state reproduces the next state bit-for-bit -- the single-seed replay reconstructs
    the missing slots exactly. A zero-weight prev(depth-1) read declares the ring depth without making
    the recurrence multi-term (a k-term recurrence would need k seed states the single-seed replay
    cannot supply -- the documented replay class)."""
    P = pops.time.Program(name)
    _case, states = program_states(P, model, ("blk", "bg"))
    U = states["blk"]
    background = states["bg"]
    rate = model.module.operator_handle("source_rate")
    # The helper argument is the physical slot count; keep_history authoring takes max lag.
    P.keep_history(U, depth=depth - 1, checkpoint_policy=Interval(k))
    nxt = P.value(
        "Un",
        U.n + P.dt * rate(U.n) + 0.0 * U.prev(depth - 1),
        at=U.next.point,
    )
    P.commit(U.next, nxt)
    background_next = P.value(
        "background_next",
        background.n + P.dt * rate(background.n),
        at=background.next.point,
    )
    P.commit(background.next, background_next)
    P.step_strategy(pops.time.FixedDt(DT))
    return P


def _blob():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (0.15 ** 2))


def _complete_native_bind(amr, compiled, initials, *, regrid_every):
    """Freeze the native history fixture at the real accepted bind boundary."""
    from pops.identity import make_identity
    from pops.model.bind_schema import BindSchema
    from pops.runtime._bound_snapshot import BoundSnapshot
    from tests.python.support.native_execution_context import (
        compiled_problem_execution_context,
    )

    authored = compiled.program
    authored = getattr(authored, "program", authored)
    context = compiled_problem_execution_context(compiled, target="amr_system")
    amr._execution_context = context
    evidence = {}
    for name, value in sorted(initials.items()):
        array = np.ascontiguousarray(value, dtype=np.float64)
        evidence[name] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "content_sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
        }
    snapshot = BoundSnapshot(
        semantic_identity=compiled.semantic_identity,
        artifact_identity=compiled.artifact_identity,
        layout={"kind": "amr", "cells": [N, N], "regrid_every": regrid_every},
        blocks=[{"name": name} for name in initials],
        field_plans={},
        step_transaction=authored.transaction_plan().to_data(),
        params=[],
        aux_evidence={},
        initial_evidence=evidence,
        bind_schema_identity=make_identity(
            "bind-schema", BindSchema().to_dict()),
        execution_context=context.to_data(),
    )
    amr._temporal_restart_state.configure_program(
        authored.temporal_manifest(), time=amr.time(), macro_step=amr.macro_step())
    amr._finalize_bind(snapshot)


def _build(program_factory, regrid_every):
    amr = AmrSystem(n=N, L=1.0, regrid_every=regrid_every)
    amr.set_temporal_relations([2], [1], ["integral_only"])
    if not hasattr(amr, "install_program") or not hasattr(amr, "history_names"):
        require_native_or_skip(
            "test_amr_history_regrid_replay requires install_program/history_names bindings")
    model = _passive_source_model("history_regrid_model")
    program = program_factory(model)
    compiled = compile_problem(model=model, time=program, target="amr_system")
    block_cm = compile_block_model(model, target="amr_system")
    bg_cm = compile_block_model(model, target="amr_system")
    amr.add_equation("blk", block_cm,
                     spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                     time=engine.Explicit(method="ssprk2"))
    # The explicit background block keeps the historical 2-level seed (a single-block Program builds
    # a coarse-only hierarchy where regrid() is a structural no-op). It is part of the same compiled
    # Program and advances through the exact zero-transport conservative rate; its smooth density
    # remains below the tag threshold throughout these short runs.
    amr.add_equation("bg", bg_cm,
                     spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                     time=engine.Explicit(method="ssprk2"))
    amr.set_refinement(1.2)  # tags the blk blob only -> real 2-level layouts at the cadence
    initials = {"blk": _blob(), "bg": np.full((N, N), 0.5)}
    amr.set_density("blk", initials["blk"])
    amr.set_density("bg", initials["bg"])  # smooth and below threshold throughout the run
    amr.install_program(compiled.so_path)
    authored = compiled.program
    amr._step_strategy = authored._step_strategy
    amr._step_transaction_plan = authored.transaction_plan()
    persistence = getattr(getattr(compiled, "program", None), "_history_persistence", None)
    if persistence:
        amr.set_history_persistence(
            {name: policy for name, (_depth, policy) in persistence.items()})
    _complete_native_bind(amr, compiled, initials, regrid_every=regrid_every)
    return amr, None


def _rings(amr):
    return {h: [np.asarray(amr.history_global(h, k), dtype=np.float64).ravel()
                for k in range(int(amr.history_depth(h)))] for h in amr.history_names()}


def _rings_equal(first, second):
    return first.keys() == second.keys() and all(
        len(first[name]) == len(second[name])
        and all(np.array_equal(left, right)
                for left, right in zip(first[name], second[name], strict=True))
        for name in first)


def _run_case(program_factory, nsteps, half, label, regrid_every):
    """continuous(nsteps) vs [run(half), ckpt, FRESH restart, continue]. Returns the comparison data."""
    cont, err = _build(program_factory, regrid_every)
    assert cont is not None, err
    assert int(cont.n_levels()) >= 2, \
        "history regrid replay requires a real two-level hierarchy"
    _advance(cont, half)
    cont_rings_at_half = _rings(cont)
    _advance(cont, nsteps - half)
    ref = np.asarray(cont.density("blk"))

    run, _ = _build(program_factory, regrid_every)
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
            slot_dts = np.asarray(
                d["history_slot_dt_" + h], dtype=np.float64).reshape(-1)
            expected_dts = np.full(depth, DT, dtype=np.float64)
            if depth > 1:
                expected_dts[1] = accepted_dt
            assert np.array_equal(slot_dts, expected_dts), (
                "AMR history slot_dt must remain the primary-clock macro dt "
                "across every fine-level substep (got %r)" % slot_dts.tolist()
            )
            key = "history_stored_slots_" + h
            stored = [int(s) for s in d[key]] if key in d else list(range(depth))
            fp = "history_regrid_steps_" + h
            fingerprint = [int(s) for s in d[fp]] if fp in d else None
            requested = [int(s) for s in d["history_requested_stored_slots_" + h]]
            mode = str(d["history_storage_mode_" + h])
            stored_info[h] = (
                depth, sorted(requested), sorted(stored), mode, fingerprint)
        fresh, _ = _build(program_factory, regrid_every)  # a FRESH AmrSystem (fresh install_program)
        fresh.restart(ckpt)
        # Dense safety promotion and clean-window replay must both complete no structural regrid.
        # Read the native guard evidence immediately after restart.
        fired = (sorted(int(s) for s in fresh._s.last_replay_regrid_steps())
                 if hasattr(fresh._s, "last_replay_regrid_steps") else None)
        rings_after_restart = _rings(fresh)
        report = fresh.last_restart_report()
        _advance(fresh, nsteps - half)
    got = np.asarray(fresh.density("blk"))
    return (ref, got, cont_rings_at_half, rings_after_restart, stored_info, report, fired), None


def _assert_bit_identical(out, label, *, want_mode, want_fingerprint):
    ref, got, cont_rings, rest_rings, stored_info, report, fired = out
    chk(bool(stored_info) and all(mode == want_mode
                                 for _, _, _, mode, _ in stored_info.values()),
        "%s resolved storage mode is %s: %r" % (label, want_mode, stored_info))
    chk(all(fp == want_fingerprint for _, _, _, _, fp in stored_info.values()),
        "%s authenticated regrid fingerprint is %r" % (label, want_fingerprint))
    if want_mode == "dense_regrid_safety":
        chk(all(len(requested) < depth and len(stored) == depth
                for depth, requested, stored, _, _ in stored_info.values()),
            "%s preserves selective intent but stores every slot safely" % label)
        chk(report is not None and all(
            row["storage_mode"] == want_mode
            and row["requested_slots"] < row["stored_slots"]
            and row["recomputed_slots"] == 0
            for row in report.histories),
            "%s report exposes the dense safety promotion and zero replay" % label)
    else:
        chk(all(requested == stored and len(stored) < depth
                for depth, requested, stored, _, _ in stored_info.values()),
            "%s retains selective policy storage on the clean hierarchy window" % label)
        chk(report is not None and any(row["recomputed_slots"] > 0
                                      for row in report.histories),
            "%s report records the native replay of omitted slots" % label)
    chk(fired == [], "%s restart completes no structural regrid (got %r)" % (label, fired))
    ok_rings = _rings_equal(cont_rings, rest_rings)
    chk(ok_rings, "every post-restart ring slot (recomputed included) equals uninterrupted bit-for-bit")
    chk(np.array_equal(ref, got),
        "%s continuation is BIT-IDENTICAL to uninterrupted (max|d| = %.3e)"
        % (label, float(np.abs(ref - got).max())))


def test_a_in_window_regrid_bit_identical():
    print("== (a) one in-window regrid -> explicit dense safety storage + bit-identical restart ==")
    # depth 3, Interval(2) -> stored {0,2}; ckpt at m=6, regrid_every=4: the re-step producing slot 1
    # runs at cursor 4 (a due regrid) -- INSIDE the replay window; the 2-level blob tags make it COMPLETE.
    out, err = _run_case(lambda model: _state_ring_program(model, 3, 2, "adc635_a"), nsteps=10, half=6,
                         label="a", regrid_every=4)
    assert out is not None, err
    _assert_bit_identical(
        out, "in-window-regrid", want_mode="dense_regrid_safety", want_fingerprint=[4])


def test_b_multiple_in_window_regrids_bit_identical():
    print("== (b) multiple in-window regrids -> explicit dense safety storage ==")
    # depth 5, Interval(4) -> stored {0,4}; ckpt at m=8, regrid_every=2: the gap 0..4 re-steps at
    # cursors 4,5,6,7 (producing slots 3,2,1,0) -> due regrids at BOTH cursor 4 and cursor 6.
    out, err = _run_case(lambda model: _state_ring_program(model, 5, 4, "adc635_b"), nsteps=12, half=8,
                         label="b", regrid_every=2)
    assert out is not None, err
    _, _, _, _, stored_info, _, _ = out
    chk(any(fp is not None and len(fp) >= 2
            for _, _, _, _, fp in stored_info.values()),
        "the storage plan records >= 2 in-window regrids: %r"
        % {h: v[4] for h, v in stored_info.items()})
    _assert_bit_identical(
        out, "multi-in-window-regrid", want_mode="dense_regrid_safety",
        want_fingerprint=[4, 6])


def test_c_clean_window_non_regression():
    print("== (c) no-regrid non-regression: a clean replay window still round-trips bit-identically ==")
    # depth 3, Interval(2) -> stored {0,2}; ckpt at m=8, regrid_every=4: the re-steps run at cursors 6,7
    # (no due regrid) -- the ADC-631 clean-window case, an empty fingerprint AND an empty fired schedule.
    out, err = _run_case(lambda model: _state_ring_program(model, 3, 2, "adc635_c"), nsteps=12, half=8,
                         label="c", regrid_every=4)
    assert out is not None, err
    _, _, _, _, stored_info, _, _ = out
    chk(all(fp is not None and len(fp) == 0
            for _, _, _, _, fp in stored_info.values()),
        "the clean-window fingerprint is EMPTY (no in-window regrid): %r"
        % {h: v[4] for h, v in stored_info.items()})
    _assert_bit_identical(
        out, "clean-window", want_mode="policy", want_fingerprint=[])


def test_d_corrupted_fingerprint_refused():
    print("== (d) inverted guard: a corrupted regrid-schedule fingerprint fails the restart LOUD ==")
    def factory(model):
        return _state_ring_program(model, 3, 2, "adc635_d")

    run, err = _build(factory, regrid_every=4)
    assert run is not None, err
    _advance(run, 6)  # ckpt at m=6: one in-window regrid at cursor 4
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = run.checkpoint(os.path.join(tmp, "d"))
        d = dict(np.load(ckpt, allow_pickle=False))
        # An uncorrupted file restarts clean.
        clean, _ = _build(factory, regrid_every=4)
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
        fresh, _ = _build(factory, regrid_every=4)
        raised = ""
        try:
            fresh.restart(bad)
        except (ValueError, RuntimeError) as exc:
            raised = str(exc)
    chk("regrid" in raised and ("inconsistent" in raised or "corrupted" in raised),
        "the corrupted fingerprint is REFUSED loud (got: %s)" % (raised[:140] or "<none>"))


def main():
    test_a_in_window_regrid_bit_identical()
    test_b_multiple_in_window_regrids_bit_identical()
    test_c_clean_window_non_regression()
    test_d_corrupted_fingerprint_refused()
    print("PASS test_amr_history_regrid_replay")


if __name__ == "__main__":
    main()
