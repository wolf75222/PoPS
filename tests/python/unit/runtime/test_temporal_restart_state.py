"""ADC-667 strict Uniform next-attempt checkpoint state."""
from __future__ import annotations

import json

import numpy as np
import pytest

from pops._bootstrap import StepAttemptRejected
from pops.runtime._step_strategy import run_control_payload, run_step_attempt
from pops.runtime._system_io import _SystemIO
from pops.runtime._temporal_restart import TemporalRestartState
from pops.runtime._uniform_restart_preflight import preflight_uniform_restart
from pops.time import FixedDt


class _Native:
    def __init__(self, *, reject=False):
        self.t = 0.0
        self.cursor = 0
        self.reject = reject

    def time(self):
        return self.t

    def macro_step(self):
        return self.cursor

    def step(self, dt):
        if self.reject:
            raise StepAttemptRejected("rejected")
        self.t += dt
        self.cursor += 1


class _Engine:
    def __init__(self, native, state):
        self._s = native
        self._temporal_restart_state = state


def _bound_state(strategy=None):
    state = TemporalRestartState()
    state.begin_run(
        strategy or run_control_payload(FixedDt(0.125)),
        time=0.0, macro_step=0,
    )
    return state


def test_accepted_attempt_advances_cursor_and_round_trips_exact_controller_state():
    native = _Native()
    state = _bound_state()
    run_step_attempt(_Engine(native, state), native, FixedDt(0.125), t_end=1.0)

    payload = state.checkpoint_json(time=native.time(), macro_step=native.macro_step())
    restored = TemporalRestartState.from_json(
        np.array(payload), time=native.time(), macro_step=native.macro_step())
    data = restored.to_data()
    assert data["schedule_cursors"] == {"macro_step": 1}
    assert data["controller_state"]["last_accepted_dt"] == (0.125).hex()
    assert data["transaction_stats"] == {"accepted": 1, "rejected": 0, "failed": 0}
    restored.begin_run(
        run_control_payload(FixedDt(0.125)), time=0.125, macro_step=1)
    with pytest.raises(RuntimeError, match="checkpointed step strategy"):
        restored.begin_run(
            run_control_payload(FixedDt(0.25)), time=0.125, macro_step=1)


def test_rejection_preserves_native_cursor_and_makes_checkpoint_ineligible(tmp_path):
    native = _Native(reject=True)
    state = _bound_state()
    engine = _Engine(native, state)
    with pytest.raises(StepAttemptRejected):
        run_step_attempt(engine, native, FixedDt(0.125), t_end=1.0)

    assert (native.time(), native.macro_step()) == (0.0, 0)
    assert state.transaction_stats == {"accepted": 0, "rejected": 1, "failed": 0}
    target = tmp_path / "must_not_exist.npz"
    with pytest.raises(RuntimeError, match="accepted synchronized"):
        _SystemIO.checkpoint(engine, str(target))
    assert not target.exists()

    native.reject = False
    state.begin_run(run_control_payload(FixedDt(0.125)), time=0.0, macro_step=0)
    run_step_attempt(engine, native, FixedDt(0.125), t_end=1.0)
    assert state.transaction_stats == {"accepted": 1, "rejected": 1, "failed": 0}
    assert state.status == "accepted"


def test_strict_temporal_manifest_refuses_missing_or_unsynchronized_state():
    state = _bound_state()
    payload = json.loads(state.checkpoint_json(time=0.0, macro_step=0))
    payload.pop("event_queue")
    with pytest.raises(ValueError, match="incomplete strict manifest"):
        TemporalRestartState.from_json(json.dumps(payload), time=0.0, macro_step=0)

    payload = json.loads(state.checkpoint_json(time=0.0, macro_step=0))
    payload["synchronized"] = False
    payload["status"] = "rejected"
    with pytest.raises(ValueError, match="not an accepted synchronized point"):
        TemporalRestartState.from_json(json.dumps(payload), time=0.0, macro_step=0)


@pytest.mark.parametrize(
    ("section", "value", "message"),
    [
        ("strategy", {**run_control_payload(FixedDt(0.1)), "extra": True}, "strategy"),
        ("controller_state", {"last_accepted_dt": "0x1p-3", "extra": 0}, "controller"),
        ("event_queue", [{"kind": "output"}], "event"),
        ("transaction_stats", {"accepted": 0, "rejected": -1, "failed": 0}, "statistics"),
    ],
)
def test_strict_temporal_sections_reject_extra_keys_and_invalid_values(section, value, message):
    state = _bound_state()
    payload = json.loads(state.checkpoint_json(time=0.0, macro_step=0))
    payload[section] = value
    with pytest.raises((TypeError, ValueError), match=message):
        TemporalRestartState.from_json(np.array(json.dumps(payload)), time=0.0, macro_step=0)


class _Payload(dict):
    @property
    def files(self):
        return list(self)


def test_uniform_preflight_rejects_incomplete_dynamic_indexes_before_native_restore():
    payload = _Payload({
        "program_hash": np.array("ab" * 32),
        "history_names": np.array([], dtype="U1"),
        "cache_nodes": np.array([], dtype=np.int64),
        "cache_names": np.array([], dtype="U1"),
        "temporal_restart_state": np.array("{}"),
    })
    preflight_uniform_restart(payload)

    payload["history_names"] = np.array(["rhs"])
    with pytest.raises(ValueError, match="history 'rhs'.*incomplete strict manifest"):
        preflight_uniform_restart(payload)
