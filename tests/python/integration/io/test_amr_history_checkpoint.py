#!/usr/bin/env python3
"""ADC-631 (c): mid-regrid v3 checkpoint of a multistep AMR history -> fresh restart -> bit-identical.

Under ACTIVE regridding (regrid_every=2, regrids firing BEFORE the checkpoint and AFTER the restart),
a compiled multistep Program's history ring is remapped through every regrid; a v3 checkpoint stores it
per the persistence policy and a FRESH AmrSystem restart reproduces the uninterrupted trajectory
BIT-IDENTICALLY (np.array_equal, no tolerance -- extends the ADC-542 acceptance with histories):

  (1) Dense (AB2 R-ring, depth 2): every slot stored -> no replay -> the ring round-trips through the
      regrid remap + v3 restore and the continuation is bit-identical end-to-end;
  (2) NON-Dense 3-slot state ring (max lag 2, Interval(2), regrid_every=4): the checkpoint stores
      slots {0,2} and the restart REPLAYS slot 1 by re-stepping the installed Program. The checkpoint
      (m=8) is taken BETWEEN two regrids (fired at step 4; next at the head of step 8, post-restart)
      with the whole seed-to-checkpoint span regrid-free (steps 6,7), so the frozen-cadence replay
      re-executes the ORIGINAL (regrid-free) steps; assert the post-restart ring (every slot,
      recomputed included) equals the uninterrupted run's ring at the same macro-step bit-for-bit,
      and the continuation stays bit-identical;
  (3) the same non-Dense authoring policy checkpointed at m=6 has a regrid due at step 4 inside its
      replay window. The resolved checkpoint plan explicitly promotes effective storage to
      ``dense_regrid_safety`` while retaining the requested slots and schedule as authenticated
      metadata; restart performs no unsafe replay and remains bit-identical.
  (4) a 5-slot Interval(2) ring stores anchors {0,2,4}; replay reconstructs the two independent gaps
      from their exact older anchors and publishes slots by logical index. This catches shifted-ring
      implementations that trust the Program's internal rotate order instead of the accepted state.

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
    require_native_or_skip("test_amr_history_checkpoint: %s" % _native_missing)

try:
    import numpy as np

    import pops
    import pops.runtime._engine_descriptors as engine
    import pops.lib.time as lt
    from pops.codegen._compile_drivers import compile_problem
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.math import ddt, div
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.physics import Model
    from pops.problem import Case
    from pops.runtime._system import AmrSystem
    from pops.time._history.persistence import Interval
    from tests.python.integration._final_field_program import (
        compile_block_model,
    )
    from tests.python.support.typed_program import program_states, state_handle
except Exception as exc:  # noqa: BLE001
    require_native_or_skip(
        "test_amr_history_checkpoint cannot import pops/numpy: %s" % exc)

N = 16
DT = 2.0e-3
_C = 0.6  # linear source S(rho) = _C*rho: R changes every step, the ring is load-bearing
def _advance(sim, nsteps):
    return sim.run(
        t_end=float(sim.time()) + nsteps * DT,
        max_steps=nsteps,
    )


def chk(cond, label):
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    assert cond, label


def _passive_source_model(name):
    """One scalar with zero transport and a linear source ``S = _C * rho``.

    The field-free dynamics make checkpoint replay independent of solver warm starts, which is the
    provably bit-exact replay class. Refinement tags read the density level itself.
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


def _ab2_program(model, name="adc631_ckpt_ab2"):
    """AB2 over the R-ring (Dense by default: no keep_history policy). flux=False keeps the body free
    of solve_fields, staying in the warm-start-independent replay class."""
    module = model.module
    case = Case("%s-case" % name)
    state = case.block("blk", module)[state_handle(module)]
    P = lt.AdamsBashforth(
        state, rate=module.operator_handle("source_rate"), order=2)
    P.step_strategy(pops.time.FixedDt(DT))
    return P


def _state3_program(model, name="adc631_ckpt_state3"):
    """A 3-slot STATE ring (max lag 2, Interval(2) -> stores slots {0,2}, replays slot 1).

    The commit is the strictly affine recurrence U^{n+1} = U^n + dt*_C*U^n -- it depends only on U^n,
    with no RHS/field/operator context -- so re-stepping from any seeded state reproduces the exact
    next state and replay reconstructs slot 1 bit-for-bit. The 3-slot ring is declared by a
    zero-weight read of U.prev(2): it drives _histories to lag 2 (so Interval(2) selects the proper
    subset {0,2}) WITHOUT making the recurrence multi-term (a k-term recurrence would need k seed states,
    which the single-seed replay cannot supply -- the documented replay class). No phi / no flux, so the
    trajectory is independent of the multigrid warm start too."""
    P = pops.Program(name)
    _case, states = program_states(P, model, ("blk",))
    U = states["blk"]
    P.keep_history(U, depth=2, checkpoint_policy=Interval(2))
    # Strictly affine growth (reads U.n only), + a zero-weight prev(2) read that declares the 3-slot
    # ring without breaking the single-step reconstructability of the replay.
    nxt = P.value(
        "Un", U.n + P.dt * _C * U.n + 0.0 * U.prev(2), at=U.next.point)
    P.commit(U.next, nxt)
    P.step_strategy(pops.time.FixedDt(DT))
    return P


def _state5_program(model, name="adc631_ckpt_state5"):
    """A 5-slot strictly affine ring with two independently replayed Interval(2) gaps."""
    P = pops.Program(name)
    _case, states = program_states(P, model, ("blk",))
    U = states["blk"]
    P.keep_history(U, depth=4, checkpoint_policy=Interval(2))
    nxt = P.value(
        "Un", U.n + P.dt * _C * U.n + 0.0 * U.prev(4), at=U.next.point)
    P.commit(U.next, nxt)
    P.step_strategy(pops.time.FixedDt(DT))
    return P


def _blob():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (0.15 ** 2))


def _complete_native_bind(amr, compiled, initial, *, regrid_every):
    """Freeze this deliberately native integration fixture with its exact compile/bind evidence.

    The test exercises the native AMR history store directly, but run/checkpoint are accepted-state
    operations and therefore require the same authenticated lifecycle boundary as ``pops.bind``.
    This helper creates that boundary from the real compiled identities and payload digest; it does
    not weaken the production guard or invent a compatibility path.
    """
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
    array = np.ascontiguousarray(initial, dtype=np.float64)
    snapshot = BoundSnapshot(
        semantic_identity=compiled.semantic_identity,
        artifact_identity=compiled.artifact_identity,
        layout={"kind": "amr", "cells": [N, N], "regrid_every": regrid_every},
        blocks=[{"name": "blk"}],
        field_plans={},
        step_transaction=authored.transaction_plan().to_data(),
        params=[],
        aux_evidence={},
        initial_evidence={
            "blk": {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "content_sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
            }
        },
        bind_schema_identity=make_identity(
            "bind-schema", BindSchema().to_dict()),
        execution_context=context.to_data(),
    )
    amr._temporal_restart_state.configure_program(
        authored.temporal_manifest(), time=amr.time(), macro_step=amr.macro_step())
    amr._finalize_bind(snapshot)


def _build(program_factory, regrid_every=2):
    amr = AmrSystem(n=N, L=1.0, regrid_every=regrid_every)
    # One explicit clock relation for the resolved two-level hierarchy. Spatial refinement never
    # doubles as an implicit time-subcycling authority.
    amr.set_temporal_relations([2], [1], ["integral_only"])
    if not hasattr(amr, "install_program") or not hasattr(amr, "history_names"):
        require_native_or_skip(
            "test_amr_history_checkpoint requires install_program/history_names bindings")
    model = _passive_source_model("%s_model" % program_factory.__name__.lstrip("_"))
    program = program_factory(model)
    compiled = compile_problem(
        model=model, time=program, target="amr_system")
    block_cm = compile_block_model(model, target="amr_system")
    amr.add_equation("blk", block_cm,
                     spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                     time=engine.Explicit(method="ssprk2"))
    amr.set_refinement(1.2)  # tags the blob -> a real 2-level hierarchy, regrids at steps 2,4,...
    initial = _blob()
    amr.set_density("blk", initial)
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
    _complete_native_bind(amr, compiled, initial, regrid_every=regrid_every)
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


def _run_case(program_factory, nsteps, half, label, regrid_every=2):
    """continuous(nsteps) vs [run(half), ckpt, fresh restart, continue]. Returns the comparison data."""
    cont, err = _build(program_factory, regrid_every)
    assert cont is not None, err
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
                # Slot 1 is the just-accepted macro step after the terminal ring rotation.  A
                # run-to-target controller may clip that step by one ulp, so authenticate it against
                # the checkpoint's exact temporal authority rather than a decimal test constant.
                expected_dts[1] = accepted_dt
            assert np.array_equal(slot_dts, expected_dts), (
                "AMR history slot_dt must be the primary-clock macro dt for every "
                "level-coherent slot (got %r)" % slot_dts.tolist()
            )
            key = "history_stored_slots_" + h
            stored = [int(s) for s in d[key]] if key in d else list(range(depth))
            requested = [int(s) for s in d["history_requested_stored_slots_" + h]]
            mode = str(d["history_storage_mode_" + h])
            fp_key = "history_regrid_steps_" + h
            fingerprint = [int(s) for s in d[fp_key]] if fp_key in d else None
            stored_info[h] = (
                depth, sorted(requested), sorted(stored), mode, fingerprint)
        fresh, _ = _build(program_factory, regrid_every)
        fresh.restart(ckpt)
        rings_after_restart = _rings(fresh)
        report = fresh.last_restart_report()
        _advance(fresh, nsteps - half)
    got = np.asarray(fresh.density("blk"))
    return (ref, got, cont_rings_at_half, rings_after_restart, stored_info, report), None


def test_ab2_dense_checkpoint_bit_identical():
    print("== (1) AB2 Dense: mid-regrid v3 ckpt -> restart -> bit-identical continuation ==")
    out, err = _run_case(_ab2_program, nsteps=6, half=3, label="ab2")
    assert out is not None, err
    ref, got, cont_rings, rest_rings, stored_info, _report = out
    chk(all(len(stored) == depth and requested == stored and mode == "policy"
            for depth, requested, stored, mode, _ in stored_info.values())
        and bool(stored_info),
        "Dense stores every ring slot (no replay): %r" % stored_info)
    ok_rings = _rings_equal(cont_rings, rest_rings)
    chk(ok_rings, "the restored ring equals the uninterrupted ring at the checkpoint step, bit-for-bit")
    chk(np.array_equal(ref, got),
        "AB2 continuous == (run, ckpt, restart, continue) BIT-IDENTICALLY (max|d| = %.3e)"
        % float(np.abs(ref - got).max()))


def test_state3_interval_replay_bit_identical():
    print("== (2) state ring Interval(2): ckpt at m=8 between regrids -> restart REPLAYS slot 1 ==")
    out, err = _run_case(_state3_program, nsteps=12, half=8, label="state3", regrid_every=4)
    assert out is not None, err
    ref, got, cont_rings, rest_rings, stored_info, report = out
    chk(bool(stored_info) and all(
        requested == stored and len(stored) < depth and mode == "policy" and fp == []
        for depth, requested, stored, mode, fp in stored_info.values()),
        "Interval(2) stores a SUBSET of the ring slots (the gap is replayed): %r" % stored_info)
    chk(report is not None and any(h["recomputed_slots"] >= 1 for h in report.histories),
        "the restart report records the replayed (recomputed) slots")
    ok_rings = _rings_equal(cont_rings, rest_rings)
    chk(ok_rings,
        "EVERY post-restart ring slot (recomputed included) equals the uninterrupted ring bit-for-bit")
    chk(np.array_equal(ref, got),
        "the replayed-ring continuation is BIT-IDENTICAL to uninterrupted (max|d| = %.3e)"
        % float(np.abs(ref - got).max()))


def test_state3_replay_window_straddling_regrid_bit_identical():
    print("== (3) ckpt at m=6 straddles regrid step 4 -> explicit dense safety storage ==")
    out, err = _run_case(_state3_program, nsteps=10, half=6, label="straddle", regrid_every=4)
    assert out is not None, err
    ref, got, cont_rings, rest_rings, stored_info, report = out
    chk(bool(stored_info) and all(
        len(requested) < depth and len(stored) == depth
        and mode == "dense_regrid_safety" and fp == [4]
        for depth, requested, stored, mode, fp in stored_info.values()),
        "Interval(2) intent is promoted to dense_regrid_safety for the straddling window: %r"
        % stored_info)
    chk(report is not None and all(
        h["storage_mode"] == "dense_regrid_safety"
        and h["requested_slots"] < h["stored_slots"]
        and h["recomputed_slots"] == 0
        for h in report.histories),
        "the restart report exposes the safety promotion and zero replay")
    ok_rings = _rings_equal(cont_rings, rest_rings)
    chk(ok_rings,
        "EVERY post-restart ring slot restored densely equals uninterrupted")
    chk(np.array_equal(ref, got),
        "the straddling-window continuation is BIT-IDENTICAL to uninterrupted (max|d| = %.3e)"
        % float(np.abs(ref - got).max()))


def test_state5_multiple_anchor_gaps_replay_by_index_bit_identical():
    print("== (4) 5-slot Interval(2): two anchor gaps replay by exact logical index ==")
    # At m=11 with cadence 6, cursors 7..10 form a stable replay window between the completed
    # regrid at 6 and the next due regrid at 12. Continuation crosses that next real regrid.
    out, err = _run_case(
        _state5_program, nsteps=15, half=11, label="state5", regrid_every=6)
    assert out is not None, err
    ref, got, cont_rings, rest_rings, stored_info, report = out
    chk(bool(stored_info) and all(
        requested == [0, 2, 4] and stored == [0, 2, 4]
        and depth == 5 and mode == "policy" and fp == []
        for depth, requested, stored, mode, fp in stored_info.values()),
        "Interval(2) retains three exact anchors around two gaps: %r" % stored_info)
    chk(report is not None and all(
        history["recomputed_slots"] == 2 for history in report.histories),
        "restart reports exactly the two omitted logical slots")
    chk(_rings_equal(cont_rings, rest_rings),
        "both reconstructed gaps equal the uninterrupted ring by index, bit-for-bit")
    chk(np.array_equal(ref, got),
        "multi-gap continuation remains BIT-IDENTICAL through the next real regrid")


def main():
    test_ab2_dense_checkpoint_bit_identical()
    test_state3_interval_replay_bit_identical()
    test_state3_replay_window_straddling_regrid_bit_identical()
    test_state5_multiple_anchor_gaps_replay_by_index_bit_identical()
    print("PASS test_amr_history_checkpoint")


if __name__ == "__main__":
    main()
