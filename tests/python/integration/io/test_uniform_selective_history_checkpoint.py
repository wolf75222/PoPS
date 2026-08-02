#!/usr/bin/env python3
"""Real Uniform selective-history checkpoint/restart with variable accepted ``dt``.

This is the single-level counterpart of the AMR history replay proof.  A real compiled Program owns
a five-slot state ring whose ``Interval(2)`` policy persists anchors ``{0, 2, 4}``; restart must
recompute both missing slots, restore the exact last accepted Program ``dt``, and then produce a
bit-identical continuation under a non-constant sequence of accepted step sizes.

Missing native prerequisites are explicit skips.  Once they are present, compilation, installation,
selective checkpointing, replay, and exact continuation are required to succeed.
"""

from __future__ import annotations

import json
import os
import tempfile

from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)


_native_missing = missing_native_compile_requirement(repo_include(), default_cxx())
if _native_missing:
    require_native_or_skip(
        "test_uniform_selective_history_checkpoint: %s" % _native_missing
    )

try:
    import numpy as np

    import pops
    import pops.runtime._engine_descriptors as engine
    from pops.codegen._compile_drivers import compile_problem
    from pops.identity import make_identity
    from pops.model.bind_schema import BindSchema
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.runtime._bound_snapshot import BoundSnapshot
    from pops.runtime._system import System
    from pops.time._history.persistence import Interval
    from tests.python.integration._final_field_program import (
        compile_block_model,
        passive_source_model,
    )
    from tests.python.support.native_execution_context import (
        compiled_problem_execution_context,
    )
    from tests.python.support.typed_program import program_states
except Exception as exc:  # noqa: BLE001
    require_native_or_skip(
        "test_uniform_selective_history_checkpoint cannot import pops/numpy: %s" % exc
    )


N = 12
COEFFICIENT = 0.6
# Dyadic values keep the controller's accepted-time differences exactly equal to the submitted dt.
DT_SEQUENCE = (
    2.0 / 256.0,
    5.0 / 256.0,
    3.0 / 256.0,
    6.0 / 256.0,
    4.0 / 256.0,
    7.0 / 256.0,
    1.0 / 256.0,
    # The first post-restart attempt must reproduce the checkpointed controller exactly. Once that
    # attempt is accepted the following intervals may vary again.
    1.0 / 256.0,
    3.0 / 256.0,
    5.0 / 256.0,
    2.0 / 256.0,
)
CHECKPOINT_STEP = 7


def _program(model):
    """Five-slot affine history plus a sparse Balance producer guarded during replay."""
    from pops.diagnostics import BalanceLedger
    from pops.output._balance_due_contract import (
        BalanceDueConsumer,
        BalanceDueContract,
        BalanceDueRoute,
    )

    program = pops.Program("uniform_selective_state5")
    _case, states = program_states(program, model, ("blk",))
    state = states["blk"]
    program.keep_history(state, depth=4, checkpoint_policy=Interval(2))
    total = program.sum(state.n)
    ledger = BalanceLedger("uniform-selective-replay")
    program.record_balance(
        ledger,
        storage_change=total,
        outward_boundary_flux=0.0 * total,
        sources=0.0 * total,
        reflux=0.0 * total,
        projection=0.0 * total,
    )
    route = ledger.route_identity(state.block)
    balance_due_contract = BalanceDueContract(
        make_identity("consumer-graph", {"test": "uniform-selective-replay"}),
        (
            BalanceDueRoute(
                route,
                (
                    BalanceDueConsumer(
                        make_identity(
                            "consumer-manifest",
                            {"test": "uniform-selective-replay"},
                        ),
                        pops.time.every(2, clock=program.clock),
                    ),
                ),
            ),
        ),
    )
    next_state = program.value(
        "Un",
        state.n
        + program.dt * COEFFICIENT * state.n
        + 0.0 * state.prev(4),
        at=state.next.point,
    )
    program.commit(state.next, next_state)
    program.step_strategy(pops.time.FixedDt(DT_SEQUENCE[0]))
    return program, balance_due_contract


def _initial_state():
    x = (np.arange(N) + 0.5) / N
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.3 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)


def _authorize_bound_runtime(sim, compiled):
    """Complete the authenticated low-level bind used by this native integration fixture."""
    component = compiled.program
    authored = getattr(component, "program", component)
    context = compiled_problem_execution_context(compiled, target="system")
    sim._execution_context = context
    sim._step_strategy = authored._step_strategy
    sim._step_transaction_plan = authored.transaction_plan()
    sim._temporal_restart_state.configure_program(
        authored.temporal_manifest(),
        time=sim.time(),
        macro_step=sim.macro_step(),
    )
    snapshot = BoundSnapshot(
        semantic_identity=compiled.semantic_identity,
        artifact_identity=compiled.artifact_identity,
        layout={"kind": "uniform"},
        blocks=[{"name": "blk"}],
        field_plans={},
        step_transaction=sim._step_transaction_plan.to_data(),
        params=[],
        aux_evidence={},
        initial_evidence={},
        bind_schema_identity=make_identity("bind-schema", BindSchema().to_dict()),
        execution_context=context.to_data(),
    )
    sim._finalize_bind(snapshot)


def _build(compiled, compiled_block, initial):
    sim = System(n=N, L=1.0, periodicity=(True, True))
    sim.add_equation(
        "blk",
        compiled_block,
        spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
        time=engine.Explicit(method="euler"),
    )
    sim.set_state("blk", np.stack([initial]))
    sim.install_program(compiled.so_path)
    persistence = getattr(getattr(compiled, "program", None), "_history_persistence", None)
    assert persistence, "the compiled Program lost its authored history-persistence policy"
    sim.set_history_persistence(
        {name: policy for name, (_depth, policy) in persistence.items()}
    )
    _authorize_bound_runtime(sim, compiled)
    return sim


def _advance(sim, dts):
    for dt in dts:
        sim.step(dt)


def _state(sim):
    return np.asarray(sim.get_state("blk"), dtype=np.float64).copy()


def _rings(sim):
    return {
        name: tuple(
            np.asarray(sim.history_global(name, slot), dtype=np.float64).copy()
            for slot in range(int(sim.history_depth(name)))
        )
        for name in sim.history_names()
    }


def _assert_rings_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        assert len(left[name]) == len(right[name])
        for expected, actual in zip(left[name], right[name], strict=True):
            assert np.array_equal(expected, actual), (
                "history %s changed across selective checkpoint/replay" % name
            )


def test_uniform_interval_history_variable_dt_restart_is_bit_identical():
    model = passive_source_model(
        "uniform_selective_history_model", coefficient=COEFFICIENT
    )
    program, balance_due_contract = _program(model)
    compiled = compile_problem(
        model=model,
        time=program,
        target="system",
        balance_due_contract=balance_due_contract,
    )
    compiled_block = compile_block_model(model, target="system")
    initial = _initial_state()

    continuous = _build(compiled, compiled_block, initial)
    _advance(continuous, DT_SEQUENCE[:CHECKPOINT_STEP])
    continuous_state_at_checkpoint = _state(continuous)
    continuous_rings_at_checkpoint = _rings(continuous)
    continuous_time_at_checkpoint = continuous.time()
    _advance(continuous, DT_SEQUENCE[CHECKPOINT_STEP:])
    expected_state = _state(continuous)
    expected_rings = _rings(continuous)

    interrupted = _build(compiled, compiled_block, initial)
    _advance(interrupted, DT_SEQUENCE[:CHECKPOINT_STEP])
    assert interrupted._s.program_last_dt() == DT_SEQUENCE[CHECKPOINT_STEP - 1]
    from pops.runtime._run_manifest import begin_run
    from pops.runtime._step_strategy import run_control_payload

    begin_run(
        interrupted,
        t_end=interrupted.time(),
        step_transaction=run_control_payload(
            pops.time.FixedDt(DT_SEQUENCE[CHECKPOINT_STEP - 1])
        ),
        max_steps=0,
        output_dir=None,
    )

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = interrupted.checkpoint(os.path.join(tmp, "uniform-selective"))
        with np.load(checkpoint, allow_pickle=False) as payload:
            temporal = json.loads(str(payload["temporal_restart_state"]))
            accepted_dt = float.fromhex(
                temporal["controller_state"]["last_accepted_dt"]
            )
            assert accepted_dt == DT_SEQUENCE[CHECKPOINT_STEP - 1]
            assert float(payload["program_last_dt"]) == accepted_dt

            history_names = tuple(str(name) for name in interrupted.history_names())
            assert len(history_names) == 1
            history_name = history_names[0]
            assert int(payload["history_depth_" + history_name]) == 5
            requested = [
                int(slot)
                for slot in payload[
                    "history_requested_stored_slots_" + history_name
                ]
            ]
            stored = [
                int(slot)
                for slot in payload["history_stored_slots_" + history_name]
            ]
            assert requested == [0, 2, 4]
            assert stored == requested
            assert str(payload["history_storage_mode_" + history_name]) == "policy"
            slot_dts = np.asarray(
                payload["history_slot_dt_" + history_name], dtype=np.float64
            )
            native_slot_dts = np.asarray(
                [
                    interrupted.history_slot_dt(history_name, slot)
                    for slot in range(5)
                ],
                dtype=np.float64,
            )
            assert np.array_equal(slot_dts, native_slot_dts)
            assert len(set(slot_dts.tolist())) > 1

        restarted = _build(compiled, compiled_block, initial)
        restarted.restart(checkpoint)

        assert restarted._s.program_last_dt() == DT_SEQUENCE[CHECKPOINT_STEP - 1]
        assert restarted.macro_step() == CHECKPOINT_STEP
        assert restarted.time() == continuous_time_at_checkpoint
        assert np.array_equal(_state(restarted), continuous_state_at_checkpoint)
        _assert_rings_equal(continuous_rings_at_checkpoint, _rings(restarted))

        report = restarted.last_restart_report()
        assert report is not None
        assert len(report.histories) == 1
        assert report.histories[0]["storage_mode"] == "policy"
        assert report.histories[0]["requested_slots"] == 3
        assert report.histories[0]["stored_slots"] == 3
        assert report.histories[0]["recomputed_slots"] == 2

        _advance(restarted, DT_SEQUENCE[CHECKPOINT_STEP:])

    assert restarted.macro_step() == len(DT_SEQUENCE)
    assert restarted.time() == continuous.time()
    assert restarted._s.program_last_dt() == DT_SEQUENCE[-1]
    assert continuous._s.program_last_dt() == DT_SEQUENCE[-1]
    assert np.array_equal(_state(restarted), expected_state)
    _assert_rings_equal(expected_rings, _rings(restarted))


def _run():
    test_uniform_interval_history_variable_dt_restart_is_bit_identical()
    print(
        "Uniform selective Interval(2) variable-dt checkpoint/restart: "
        "bit-identical continuation"
    )


if __name__ == "__main__":
    _run()
