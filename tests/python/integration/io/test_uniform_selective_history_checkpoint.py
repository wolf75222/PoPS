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
    from pops.diagnostics import Balance, BalanceLedger
    from pops.output import NPZ, ScientificOutput
    from pops.time._history.persistence import Interval
    from tests.python.integration._final_field_program import (
        passive_source_model,
        resolve_periodic_field_program,
    )
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
GRID_ID = "uniform_selective_time_grid"
TIME_GRID = [0.0]
for _dt in DT_SEQUENCE:
    TIME_GRID.append(TIME_GRID[-1] + _dt)
TIME_GRID = tuple(TIME_GRID)
_BALANCE_LEDGER = BalanceLedger("uniform-selective-replay")
_BALANCE_ROUTE: dict[str, str] = {}


def _program(state_instance, _rate, _field):
    """Five-slot affine Balance Program whose omitted anchors are exactly replayable."""
    program = pops.Program("uniform_selective_state5")
    state = program.state(state_instance)
    program.keep_history(state, depth=4, checkpoint_policy=Interval(2))
    next_state = program.value(
        "Un",
        state.n
        + program.dt * COEFFICIENT * state.n
        + 0.0 * state.prev(4),
        at=state.next.point,
    )
    increment = program.value(
        "accepted_increment",
        next_state - state.n,
        at=state.next.point,
    )
    storage_change = program.sum(increment)
    outward_boundary_flux = -program.sum(increment)
    sources = program.sum(increment)
    reflux = program.sum(increment)
    projection = -2.0 * program.sum(increment)
    program.record_balance(
        _BALANCE_LEDGER,
        storage_change=storage_change,
        outward_boundary_flux=outward_boundary_flux,
        sources=sources,
        reflux=reflux,
        projection=projection,
    )
    _BALANCE_ROUTE["token"] = _BALANCE_LEDGER.route_identity(state.block).token
    program.commit(state.next, next_state)
    program.step_strategy(pops.time.ExternalTimeGrid(GRID_ID))
    return program


def _balance_consumers(_case, block, _state, program):
    schedule = pops.time.every(2, clock=program.clock)
    return (
        ScientificOutput(
            format=NPZ(),
            schedule=schedule,
            diagnostics=(
                Balance(
                    _BALANCE_LEDGER,
                    block=block,
                    cadence=schedule,
                ),
            ),
            target="balance_due",
        ),
    )


def _initial_state():
    x = (np.arange(N) + 0.5) / N
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.3 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)


def _build(artifact, initial):
    return pops.bind(
        artifact,
        initial_state={"blk": np.ascontiguousarray(np.stack([initial]))},
    )


def _advance(sim, dts, *, output_dir):
    start = int(sim.macro_step())
    stop = start + len(dts)
    assert tuple(dts) == DT_SEQUENCE[start:stop]
    report = pops.run(
        sim,
        t_end=TIME_GRID[stop],
        max_steps=len(dts),
        output_dir=output_dir,
        **{GRID_ID: TIME_GRID},
    )
    assert report.accepted_steps == len(dts)


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


def _assert_due_balance(sim):
    token = _BALANCE_ROUTE.get("token")
    assert token, "the public Balance consumer lost its exact Program ledger route"
    terms = dict(sim._executor._s._accepted_balance_terms(token))
    required = {
        "storage_change",
        "outward_boundary_flux",
        "sources",
        "reflux",
        "projection",
    }
    assert set(terms) == required
    change = terms["storage_change"]
    assert change > 0.0, "the due Balance reduction must observe non-trivial growth"
    assert terms == {
        "storage_change": change,
        "outward_boundary_flux": -change,
        "sources": change,
        "reflux": change,
        "projection": -2.0 * change,
    }
    residual = (
        terms["storage_change"]
        + terms["outward_boundary_flux"]
        - terms["sources"]
        - terms["reflux"]
        - terms["projection"]
    )
    assert residual == 0.0


def test_uniform_interval_history_variable_dt_restart_is_bit_identical():
    model = passive_source_model(
        "uniform_selective_history_model", coefficient=COEFFICIENT
    )
    resolved = resolve_periodic_field_program(
        model,
        _program,
        name="uniform-selective-history",
        block_name="blk",
        target="system",
        n=N,
        rate_name="source_rate",
        cxx=default_cxx(),
        include=repo_include(),
        strict_restart=True,
        consumer_factory=_balance_consumers,
    )
    artifact = pops.compile(resolved)
    artifact.verify()
    initial = _initial_state()

    with tempfile.TemporaryDirectory() as tmp:
        continuous = _build(artifact, initial)
        _advance(
            continuous,
            DT_SEQUENCE[:2],
            output_dir=os.path.join(tmp, "continuous-due"),
        )
        _assert_due_balance(continuous)
        _advance(
            continuous,
            DT_SEQUENCE[2:CHECKPOINT_STEP],
            output_dir=os.path.join(tmp, "continuous-checkpoint"),
        )
        continuous_state_at_checkpoint = _state(continuous)
        continuous_rings_at_checkpoint = _rings(continuous)
        continuous_time_at_checkpoint = continuous.time()
        _advance(
            continuous,
            DT_SEQUENCE[CHECKPOINT_STEP:],
            output_dir=os.path.join(tmp, "continuous-final"),
        )
        expected_state = _state(continuous)
        expected_rings = _rings(continuous)

        interrupted = _build(artifact, initial)
        _advance(
            interrupted,
            DT_SEQUENCE[:CHECKPOINT_STEP],
            output_dir=os.path.join(tmp, "interrupted"),
        )
        assert (
            interrupted._executor._s.program_last_dt()
            == DT_SEQUENCE[CHECKPOINT_STEP - 1]
        )
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
                    interrupted._executor.history_slot_dt(history_name, slot)
                    for slot in range(5)
                ],
                dtype=np.float64,
            )
            assert np.array_equal(slot_dts, native_slot_dts)
            assert len(set(slot_dts.tolist())) > 1

        restarted = _build(artifact, initial)
        restarted.restart(checkpoint)

        assert (
            restarted._executor._s.program_last_dt()
            == DT_SEQUENCE[CHECKPOINT_STEP - 1]
        )
        assert restarted.macro_step() == CHECKPOINT_STEP
        assert restarted.time() == continuous_time_at_checkpoint
        assert np.array_equal(_state(restarted), continuous_state_at_checkpoint)
        _assert_rings_equal(continuous_rings_at_checkpoint, _rings(restarted))

        report = restarted._executor.last_restart_report()
        assert report is not None
        assert len(report.histories) == 1
        assert report.histories[0]["storage_mode"] == "policy"
        assert report.histories[0]["requested_slots"] == 3
        assert report.histories[0]["stored_slots"] == 3
        assert report.histories[0]["recomputed_slots"] == 2

        _advance(
            restarted,
            DT_SEQUENCE[CHECKPOINT_STEP:],
            output_dir=os.path.join(tmp, "restarted-final"),
        )

    assert restarted.macro_step() == len(DT_SEQUENCE)
    assert restarted.time() == continuous.time()
    assert restarted._executor._s.program_last_dt() == DT_SEQUENCE[-1]
    assert continuous._executor._s.program_last_dt() == DT_SEQUENCE[-1]
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
