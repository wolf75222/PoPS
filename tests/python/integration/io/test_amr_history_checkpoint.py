#!/usr/bin/env python3
"""Authenticated AMR checkpoint/restart and unsupported history-remap rollback.

The public Case -> compile -> bind lifecycle supplies every state route and the exact checkpoint
resource budget. The original N16 two-level geometry is retained throughout. Original active-regrid graphs retain
their 2:1 temporal relation; separately labeled frozen-grid replay positives use 1:1 level clocks.
AB2 Dense history supports active regridding and bit-identical full restart. Three- and five-slot
Interval state histories support affine replay on a frozen 1:1 hierarchy, including exact omitted slots.
Frozen 2:1 selective restart is explicitly refused with exact rollback: whole-macro replay cannot
reconstruct one child-level lag without an additional per-level reconstruction contract.
Their original active-regrid graphs are separate refusal/rollback controls: projecting retained
state histories onto newly refined cells is deferred to M6.4/M7. Dense Balance restart and held
variable-dt cadence windows retain separate positive coverage; Interval + Balance is refused.
All state/history comparisons are bit-exact. Missing native prerequisites are explicit local skips
and required-lane failures. Pytest + __main__.
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

# This process-isolated acceptance file performs six native AMR/checkpoint
# scenarios plus the explicit refusal of selective diagnostic replay.  A cold CI runner can legitimately exceed the 300 s suite-wide
# default while compiling their distinct Program shapes; keep the exception
# local and bounded instead of weakening every Python process test.
POPS_PROCESS_TIMEOUT = 600


_native_missing = missing_native_compile_requirement(repo_include(), default_cxx())
if _native_missing:
    require_native_or_skip("test_amr_history_checkpoint: %s" % _native_missing)

try:
    import numpy as np

    import pops
    import pops.lib.time as lt
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.math import ddt, div
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.physics import Model
    from pops.problem import Case
    from pops.time._history.persistence import Dense, Interval
    from tests.python.support.typed_program import state_handle
except Exception as exc:  # noqa: BLE001
    require_native_or_skip("test_amr_history_checkpoint cannot import pops/numpy: %s" % exc)

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
    frame = Rectangle("%s-domain" % name, lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
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


def _program_states(program, model, names):
    """Retain the formula Model as the compiler authority for each Case block."""
    case = Case("%s-program-case" % program.name)
    declaration = state_handle(model.module)
    blocks = {name: case.block(name, model) for name in names}
    return case, {name: program.state(block[declaration]) for name, block in blocks.items()}


def _ab2_program(model, name="adc631_ckpt_ab2"):
    """AB2 over the R-ring (Dense by default: no keep_history policy). flux=False keeps the body free
    of solve_fields, staying in the warm-start-independent replay class."""
    module = model.module
    case = Case("%s-case" % name)
    state = case.block("blk", model)[state_handle(module)]
    P = lt.AdamsBashforth(state, rate=module.operator_handle("source_rate"), order=2)
    P.step_strategy(pops.time.FixedDt(DT))
    return P, case


def _state3_program(
    model,
    name="adc631_ckpt_state3",
    *,
    step_strategy=None,
    balance_replay_proof=False,
    checkpoint_policy=None,
):
    """A 3-slot STATE ring (max lag 2, Interval(2) -> stores slots {0,2}, replays slot 1).

    The commit is the strictly affine recurrence U^{n+1} = U^n + dt*_C*U^n -- it depends only on U^n,
    with no RHS/field/operator context -- so re-stepping from any seeded state reproduces the exact
    next state and replay reconstructs slot 1 bit-for-bit. The 3-slot ring is declared by a
    zero-weight read of U.prev(2): it drives _histories to lag 2 (so Interval(2) selects the proper
    subset {0,2}) WITHOUT making the recurrence multi-term (a k-term recurrence would need k seed states,
    which the single-seed replay cannot supply -- the documented replay class). No phi / no flux, so the
    trajectory is independent of the multigrid warm start too."""
    P = pops.Program(name)
    _case, states = _program_states(P, model, ("blk",))
    U = states["blk"]
    P.keep_history(
        U, depth=2,
        checkpoint_policy=Interval(2) if checkpoint_policy is None else checkpoint_policy,
    )
    # Strictly affine growth (reads U.n only), + a zero-weight prev(2) read that declares the 3-slot
    # ring without breaking the single-step reconstructability of the replay.
    nxt = P.value("Un", U.n + P.dt * _C * U.n + 0.0 * U.prev(2), at=U.next.point)
    ledger = None
    if balance_replay_proof:
        from pops.diagnostics import BalanceLedger
        total = P.sum(U.n)
        ledger = BalanceLedger("amr-selective-replay")
        P.record_balance(
            ledger,
            storage_change=total,
            outward_boundary_flux=0.0 * total,
            sources=0.0 * total,
            reflux=0.0 * total,
            projection=0.0 * total,
        )
    P.commit(U.next, nxt)
    P.step_strategy(pops.time.FixedDt(DT) if step_strategy is None else step_strategy)
    return P, _case, ledger


def _state3_balance_program(model):
    return _state3_program(
        model,
        name="adc686_ckpt_state3_balance",
        balance_replay_proof=True,
        checkpoint_policy=Dense(),
    )


def _state5_program(model, name="adc631_ckpt_state5"):
    """A 5-slot strictly affine ring with two independently replayed Interval(2) gaps."""
    P = pops.Program(name)
    _case, states = _program_states(P, model, ("blk",))
    U = states["blk"]
    P.keep_history(U, depth=4, checkpoint_policy=Interval(2))
    nxt = P.value("Un", U.n + P.dt * _C * U.n + 0.0 * U.prev(4), at=U.next.point)
    P.commit(U.next, nxt)
    P.step_strategy(pops.time.FixedDt(DT))
    return P, _case


def _blob():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (0.15**2))


def _compile_checkpoint_program(model, authored, *, regrid_every, initials, program_cadence=None,
                                temporal_ratio=2):
    """Compile the full Case that owns the Program, hierarchy and checkpoint resources."""
    from pops.amr import (
        AMRClockRelation, AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging,
        AMRTransfer, Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, PatchLayout, Tag,
    )
    from pops.codegen import Production
    from pops.initial import InitialCondition
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer
    from pops.lib.initial import BindArray
    from pops.math import ValueExpr
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.projection import ConservativeCellAverage
    from pops.params import RuntimeParam

    program, case, *extra = authored
    ledger = extra[0] if extra else None
    if program_cadence is not None:
        program.cadence(substeps=program_cadence[0], stride=program_cadence[1])
    state = next(iter(model.states.values()))
    rate = model.operators["source_rate"]
    flux = model.fluxes["transport"]
    transfer = AMRTransfer()
    threshold = case.param(RuntimeParam("refine_threshold", default=1.2))
    instances = {}
    rules = []
    for name in initials:
        block = case.blocks()[name]
        instance = block[state]
        instances[name] = instance
        numerics = DiscretizationPlan()
        numerics.rates.add(
            rate,
            FiniteVolume(flux=flux, variables=variables.Conservative(state),
                         reconstruction=FirstOrder(), riemann=Rusanov()),
        )
        case.numerics(numerics, block=block)
        case.initials.add(InitialCondition(
            state=instance, value=BindArray(), projection=ConservativeCellAverage()))
        transfer.state(instance, StateTransfer())
        rules.append(Tag(ValueExpr(instance) > ValueExpr(threshold)))
    case.program(program)
    if ledger is not None:
        from pops.diagnostics import Balance
        from pops.output import ConsumerGraph, NPZ, ParallelMode, ScientificOutput
        from pops.runtime_environment import runtime_environment_report

        from pops._native_selector import select_native_dimension, selected_native_module

        if selected_native_module(required=False) is None:
            select_native_dimension(2)
        communicator = runtime_environment_report().get("communicator")
        if communicator == "serial":
            output_mode = ParallelMode.SERIAL
        elif communicator == "MPI_COMM_WORLD":
            output_mode = ParallelMode.ROOT
        else:
            raise RuntimeError("Balance fixture needs a proved native communicator")
        schedule = pops.time.every(2, clock=program.clock)
        case.consumers(ConsumerGraph.from_consumers((ScientificOutput(
            format=NPZ(mode=output_mode), schedule=schedule,
            diagnostics=(Balance(ledger, block=case.blocks()["blk"], cadence=schedule),),
            target="balance_due",
        ),)))
    layout = AMR(
        grid=CartesianGrid(frame=model.frame, cells=(N, N),
                           periodic=PeriodicAxes(model.frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=tuple(rules) + (Buffer(cells=1),),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=(AMRRegrid.frozen() if regrid_every == 0 else
                AMRRegrid(schedule=pops.time.every(regrid_every, clock=program.clock))),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, temporal_ratio),)),
        patch_layout=PatchLayout(distribute_coarse=False),
    )
    resolved = pops.resolve(
        pops.validate(case), layout=layout, backend=Production(),
        compile_options={"include": repo_include(), "cxx": default_cxx()},
    )
    return pops.compile(resolved), instances


def _bind_checkpoint_artifact(compiled, initials):
    """Bind the same artifact into a fresh runtime with its exact InstallPlan budget."""
    from tests.python.support.native_execution_context import artifact_execution_context

    artifact, instances = compiled
    runtime = pops.bind(
        artifact,
        resources={"execution_context": artifact_execution_context(artifact)},
        initial_values={
            instances[name]: np.ascontiguousarray(values[None, ...], dtype=np.float64)
            for name, values in initials.items()
        },
    )
    # These tests deliberately exercise the native checkpoint archive and history APIs. The
    # executor already owns the exact InstallPlan-derived budget; retain its public owner too.
    amr = runtime._executor
    amr._fixture_runtime = runtime
    assert amr._checkpoint_resource_budget is runtime._checkpoint_resource_budget
    return amr


_COMPILED_CASES = {}


def _build(program_factory, regrid_every=2, program_cadence=None, *, temporal_ratio=2):
    key = (program_factory, regrid_every, program_cadence, temporal_ratio)
    initials = {"blk": _blob()}
    if key not in _COMPILED_CASES:
        model = _passive_source_model("%s_model" % program_factory.__name__.lstrip("_"))
        _COMPILED_CASES[key] = _compile_checkpoint_program(
            model, program_factory(model), regrid_every=regrid_every,
            initials=initials, program_cadence=program_cadence, temporal_ratio=temporal_ratio,
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
        and all(
            np.array_equal(left, right)
            for left, right in zip(first[name], second[name], strict=True)
        )
        for name in first
    )


def _accepted_image(amr):
    """Deep, exact image of every accepted state, history, clock and hierarchy route."""
    from pops.runtime._amr_checkpoint_contract import contract_for, restart_topology_image

    return {
        "state": {
            (name, level): np.asarray(amr._s.block_level_state_global(name, level)).copy()
            for name in amr.block_names() for level in range(amr.n_levels())
        },
        "rings": _rings(amr),
        "history_dt": {
            (name, level, slot): float(amr._s.history_slot_dt(name, level, slot)).hex()
            for name in amr.history_names() for level in amr.history_levels(name)
            for slot in range(amr.history_depth(name))
        },
        "time": float(amr.time()).hex(),
        "macro_step": amr.macro_step(),
        "last_dt": float(amr._s.program_last_dt()).hex(),
        "patches": tuple(amr.patch_boxes()),
        "topology": restart_topology_image(amr._s),
        "accepted_contract": contract_for(amr._s),
    }


def _assert_accepted_image_equal(before, after):
    assert before.keys() == after.keys()
    assert before["state"].keys() == after["state"].keys()
    assert all(np.array_equal(before["state"][key], after["state"][key])
               for key in before["state"]), "refused regrid changed an accepted block/level state"
    assert _rings_equal(before["rings"], after["rings"]), "refused regrid changed history slots"
    assert all(before[key] == after[key] for key in before if key not in {"state", "rings"}), \
        "refused regrid changed accepted clocks, history dt, hierarchy or provenance"


def _assert_new_coverage_refusal(amr, *, nsteps, label):
    """Run the original live graph until its unsupported new-coverage request, then prove rollback."""
    assert amr.n_levels() == 2 and amr.patch_boxes(), "requires the original refined N16 hierarchy"
    for _ in range(nsteps):
        before = _accepted_image(amr)
        try:
            _advance(amr, 1)
        except ValueError as error:
            assert "deferred history remap supports initialized direct-child AB2 IntegralOnly" in str(error)
            _assert_accepted_image_equal(before, _accepted_image(amr))
            chk(True, "%s: unsupported retained-state remap refuses and rolls back at step %d" %
                (label, amr.macro_step()))
            return
    raise AssertionError("%s completed without the expected explicit new-coverage refusal" % label)


def _assert_multirate_restart_refusal(build, *, half, label):
    """Keep the real 2:1 checkpoint and prove rejection rolls back its provisional restore."""
    run = build()
    _advance(run, half)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = run.checkpoint(os.path.join(tmp, label))
        fresh = build()
        before = _accepted_image(fresh)
        before_temporal = fresh._temporal_restart_state.to_data()
        try:
            fresh.restart(checkpoint)
        except (ValueError, RuntimeError) as error:
            assert "selective replay requires synchronized 1:1 level clocks" in str(error)
            _assert_accepted_image_equal(before, _accepted_image(fresh))
            assert fresh._temporal_restart_state.to_data() == before_temporal
            chk(True, "%s: 2:1 selective replay refuses with exact restart rollback" % label)
        else:
            raise AssertionError("multirate replay published a slot without a reconstruction contract")


def _run_case(program_factory, nsteps, half, label, regrid_every=2, *, temporal_ratio=2):
    """continuous(nsteps) vs [run(half), ckpt, fresh restart, continue]. Returns the comparison data."""
    cont, err = _build(program_factory, regrid_every, temporal_ratio=temporal_ratio)
    assert cont is not None, err
    _advance(cont, half)
    regrids_at_half = cont._s.checkpoint_regrid_count()
    if regrid_every > 0:
        assert regrids_at_half > 0, "the positive must execute regrids before checkpoint"
    cont_rings_at_half = _rings(cont)
    _advance(cont, nsteps - half)
    if regrid_every > 0:
        assert cont._s.checkpoint_regrid_count() > regrids_at_half, \
            "the positive must execute regrids after checkpoint"
    ref = np.asarray(cont.density("blk"))

    run, _ = _build(program_factory, regrid_every, temporal_ratio=temporal_ratio)
    _advance(run, half)
    accepted_program_dt = run._s.program_last_dt()
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = run.checkpoint(os.path.join(tmp, label))
        d = np.load(ckpt, allow_pickle=False)
        temporal = json.loads(str(d["temporal_restart_state"]))
        accepted_dt = float.fromhex(temporal["controller_state"]["last_accepted_dt"])
        stored_info = {}
        for h in run.history_names():
            depth = int(d["history_depth_" + h])
            levels = [int(level) for level in d["history_levels_" + h]]
            for level in levels:
                slot_dts = np.asarray(
                    d["history_slot_dt_%s_level_%d" % (h, level)], dtype=np.float64
                ).reshape(-1)
                # The explicit temporal relation, not spatial refinement, owns fine-level dt.
                temporal_substeps = temporal_ratio ** level
                expected_dts = np.full(depth, DT / temporal_substeps, dtype=np.float64)
                if depth > 1:
                    # Slot 1 is the just-accepted macro step after the terminal ring rotation.  A
                    # run-to-target controller may clip that step by one ulp.
                    expected_dts[1] = accepted_dt / temporal_substeps
                assert np.array_equal(slot_dts, expected_dts), (
                    "AMR history slot_dt must be exact per level (level=%d, got %r)"
                    % (level, slot_dts.tolist())
                )
            key = "history_stored_slots_" + h
            stored = [int(s) for s in d[key]] if key in d else list(range(depth))
            requested = [int(s) for s in d["history_requested_stored_slots_" + h]]
            mode = str(d["history_storage_mode_" + h])
            fp_key = "history_regrid_steps_" + h
            fingerprint = [int(s) for s in d[fp_key]] if fp_key in d else None
            stored_info[h] = (depth, sorted(requested), sorted(stored), mode, fingerprint)
        fresh, _ = _build(program_factory, regrid_every, temporal_ratio=temporal_ratio)
        fresh.restart(ckpt)
        assert fresh._s.program_last_dt() == accepted_program_dt, (
            "history replay must restore the checkpoint's accepted Program dt instead of leaking "
            "one internal replay interval"
        )
        rings_after_restart = _rings(fresh)
        report = fresh.last_restart_report()
        _advance(fresh, nsteps - half)
    got = np.asarray(fresh.density("blk"))
    _assert_accepted_image_equal(_accepted_image(cont), _accepted_image(fresh))
    return (ref, got, cont_rings_at_half, rings_after_restart, stored_info, report), None


def test_ab2_dense_checkpoint_bit_identical():
    print("== (1) AB2 Dense: mid-regrid v5 ckpt -> restart -> bit-identical continuation ==")
    out, err = _run_case(_ab2_program, nsteps=6, half=3, label="ab2")
    assert out is not None, err
    ref, got, cont_rings, rest_rings, stored_info, _report = out
    chk(
        all(
            len(stored) == depth and requested == stored and mode == "policy"
            for depth, requested, stored, mode, _ in stored_info.values()
        )
        and bool(stored_info),
        "Dense stores every ring slot (no replay): %r" % stored_info,
    )
    ok_rings = _rings_equal(cont_rings, rest_rings)
    chk(
        ok_rings,
        "the restored ring equals the uninterrupted ring at the checkpoint step, bit-for-bit",
    )
    chk(
        np.array_equal(ref, got),
        "AB2 continuous == (run, ckpt, restart, continue) BIT-IDENTICALLY (max|d| = %.3e)"
        % float(np.abs(ref - got).max()),
    )


def test_state3_frozen_interval_replay_bit_identical():
    print("== (2) frozen 1:1 state ring Interval(2): ckpt at m=8 -> restart REPLAYS slot 1 ==")
    out, err = _run_case(_state3_program, nsteps=12, half=8, label="state3-frozen-1to1",
                         regrid_every=0, temporal_ratio=1)
    assert out is not None, err
    ref, got, cont_rings, rest_rings, stored_info, report = out
    chk(
        bool(stored_info)
        and all(
            requested == stored and len(stored) < depth and mode == "policy" and fp == []
            for depth, requested, stored, mode, fp in stored_info.values()
        ),
        "Interval(2) stores a SUBSET of the ring slots (the gap is replayed): %r" % stored_info,
    )
    chk(
        report is not None and any(h["recomputed_slots"] >= 1 for h in report.histories),
        "the restart report records the replayed (recomputed) slots",
    )
    ok_rings = _rings_equal(cont_rings, rest_rings)
    chk(
        ok_rings,
        "EVERY post-restart ring slot (recomputed included) equals the uninterrupted ring bit-for-bit",
    )
    chk(
        np.array_equal(ref, got),
        "the replayed-ring continuation is BIT-IDENTICAL to uninterrupted (max|d| = %.3e)"
        % float(np.abs(ref - got).max()),
    )


def test_state3_dense_balance_continuation_and_selective_refusal():
    print("== (2b) Dense Balance checkpoint is exact; selective Balance replay is refused ==")
    out, err = _run_case(
        _state3_balance_program, nsteps=6, half=3, label="state3-balance", regrid_every=0,
    )
    assert out is not None, err
    ref, got, cont_rings, rest_rings, stored_info, report = out
    chk(
        bool(stored_info) and all(
            requested == stored == list(range(depth)) and mode == "policy" and fp is None
            for depth, requested, stored, mode, fp in stored_info.values()
        ),
        "the Balance Program retains every state-history slot under Dense",
    )
    chk(
        report is None or all(h["recomputed_slots"] == 0 for h in report.histories),
        "Dense Balance restart performs no diagnostic replay",
    )
    chk(
        _rings_equal(cont_rings, rest_rings) and np.array_equal(ref, got),
        "Balance checkpoint rings and continued state remain bit-identical",
    )
    model = _passive_source_model("selective-balance-refusal")
    try:
        _compile_checkpoint_program(
            model, _state3_program(model, balance_replay_proof=True),
            regrid_every=0, initials={"blk": _blob()},
        )
    except ValueError as error:
        message = str(error).lower()
        chk("replay" in message and ("reduce" in message or "balance" in message),
            "selective Balance replay is explicitly refused: %s" % error)
    else:
        raise AssertionError("Interval + Balance compiled without a diagnostic replay contract")


def test_frozen_multirate_selective_restart_refuses_without_mutation():
    for factory, half in ((_state3_program, 8), (_state5_program, 11)):
        _assert_multirate_restart_refusal(
            lambda factory=factory: _build(factory, regrid_every=0, temporal_ratio=2)[0],
            half=half, label="%s-2to1" % factory.__name__,
        )


def test_state3_original_active_regrid_refuses_without_mutation():
    # Original clean-window (12/8) and straddling-window (10/6) graphs both request unsupported
    # new fine coverage before their checkpoint. Keep their N16/depth3/regrid4 configurations.
    for nsteps, half in ((12, 8), (10, 6)):
        live, _ = _build(_state3_program, regrid_every=4)
        _assert_new_coverage_refusal(live, nsteps=nsteps, label="state3-%d-ckpt%d" % (nsteps, half))


def test_state5_frozen_multiple_anchor_gaps_replay_by_index_bit_identical():
    print("== (4) frozen 1:1 5-slot Interval(2): two anchor gaps replay by exact logical index ==")
    # The separately declared frozen-grid positive preserves the original depth, steps and anchors.
    out, err = _run_case(_state5_program, nsteps=15, half=11, label="state5-frozen-1to1",
                         regrid_every=0, temporal_ratio=1)
    assert out is not None, err
    ref, got, cont_rings, rest_rings, stored_info, report = out
    chk(
        bool(stored_info)
        and all(
            requested == [0, 2, 4]
            and stored == [0, 2, 4]
            and depth == 5
            and mode == "policy"
            and fp == []
            for depth, requested, stored, mode, fp in stored_info.values()
        ),
        "Interval(2) retains three exact anchors around two gaps: %r" % stored_info,
    )
    chk(
        report is not None
        and all(history["recomputed_slots"] == 2 for history in report.histories),
        "restart reports exactly the two omitted logical slots",
    )
    chk(
        _rings_equal(cont_rings, rest_rings),
        "both reconstructed gaps equal the uninterrupted ring by index, bit-for-bit",
    )
    chk(
        np.array_equal(ref, got),
        "frozen-grid multi-gap continuation remains BIT-IDENTICAL",
    )


def test_state5_original_active_regrid_refuses_without_mutation():
    live, _ = _build(_state5_program, regrid_every=6)
    _assert_new_coverage_refusal(live, nsteps=15, label="state5-15-ckpt11")


def test_amr_variable_dt_stride_checkpoint_closes_like_continuous_run():
    """A held AMR window restarts exactly and stores selective histories densely for safety."""
    from pops.time import ExternalTimeGrid

    steps = tuple(0.01 * index for index in range(1, 13))
    grid = (0.0,) + tuple(sum(steps[:index]) for index in range(1, len(steps) + 1))
    controls = {"amr_checkpoint_test_grid": grid}

    def variable_state3_program(model):
        return _state3_program(
            model,
            name="adc631_ckpt_variable_state3",
            step_strategy=ExternalTimeGrid("amr_checkpoint_test_grid"),
        )

    def fresh():
        system, error = _build(
            variable_state3_program,
            regrid_every=0,
            program_cadence=(1, 3),
        )
        assert system is not None, error
        return system

    def accepted_levels(system):
        return tuple(
            np.asarray(system._s.block_level_state_global("blk", level))
            for level in range(system._s.n_levels())
        )

    continuous = fresh()
    continuous.run(t_end=grid[-1], max_steps=len(steps), controls=controls)

    interrupted = fresh()
    interrupted.run(t_end=grid[9], max_steps=9, controls=controls)
    expected_window_start = interrupted.time()
    interrupted.run(t_end=grid[10], max_steps=1, controls=controls)
    expected_window_dt = interrupted.time() - expected_window_start
    assert interrupted._s.program_cadence_window_steps() == 1
    assert interrupted._s.program_cadence_window_dt() == expected_window_dt
    assert interrupted._s.program_cadence_window_start_time() == expected_window_start

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = interrupted.checkpoint(os.path.join(tmp, "amr_variable_stride"))
        with np.load(checkpoint, allow_pickle=False) as checkpoint_data:
            for history_name in checkpoint_data["history_names"]:
                name = str(history_name)
                assert (
                    str(checkpoint_data["history_storage_mode_" + name]) == "dense_cadence_safety"
                )
                assert len(checkpoint_data["history_stored_slots_" + name]) == int(
                    checkpoint_data["history_depth_" + name]
                )
        restarted = fresh()
        restarted.restart(checkpoint)
        assert restarted._s.program_cadence_window_steps() == 1
        assert restarted._s.program_cadence_window_dt() == expected_window_dt
        assert restarted._s.program_cadence_window_start_time() == expected_window_start
        restarted.run(
            t_end=grid[-1],
            max_steps=len(steps) - 10,
            controls=controls,
        )

    assert restarted.macro_step() == continuous.macro_step() == len(steps)
    assert restarted.time() == continuous.time()
    expected = accepted_levels(continuous)
    actual = accepted_levels(restarted)
    assert len(actual) == len(expected)
    assert all(np.array_equal(got, want) for got, want in zip(actual, expected, strict=True))


def main():
    test_ab2_dense_checkpoint_bit_identical()
    test_state3_frozen_interval_replay_bit_identical()
    test_state3_dense_balance_continuation_and_selective_refusal()
    test_state3_original_active_regrid_refuses_without_mutation()
    test_state5_frozen_multiple_anchor_gaps_replay_by_index_bit_identical()
    test_state5_original_active_regrid_refuses_without_mutation()
    test_frozen_multirate_selective_restart_refuses_without_mutation()
    test_amr_variable_dt_stride_checkpoint_closes_like_continuous_run()
    print("PASS test_amr_history_checkpoint")


if __name__ == "__main__":
    main()
