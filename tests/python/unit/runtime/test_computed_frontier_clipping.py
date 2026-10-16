"""Explicit request capping; accepted time still comes only from the Program."""
import copy
import json

import pytest

from pops.runtime._step_strategy import StepAttemptRejected, prepare_program_run
from pops.runtime._temporal_restart import TemporalRestartState
from pops.time import ComputedDt
from tests.python.unit.runtime.test_step_strategy import _Engine, _Native


def subject(policy, *, start=0., factor=1., rejected=0):
    engine, native, temporal = _Engine(policy), _Native(), TemporalRestartState(time_hex=float(start).hex())
    engine._temporal_restart_state = temporal
    native.t = start
    receipt = {}
    def advance(dt):
        nonlocal rejected
        native.calls.append(dt)
        if rejected:
            rejected -= 1
            raise StepAttemptRejected("step attempt rejected during guard: authored")
        duration = factor * dt
        native.t += duration
        native.cursor += 1
        receipt["pops.frontier.duration"] = duration
    native.step = advance
    engine.program_diagnostics = lambda: receipt
    prepared = prepare_program_run(engine)
    prepared.begin(temporal, time=start, macro_step=0)
    return native, temporal, prepared


def test_nominal_wire_and_request_remain_exactly_legacy():
    legacy = ComputedDt(1., endpoint_ulps=4, max_rejections=1)
    assert legacy.to_data() == {
        "kind": "computed_dt", "schema_version": 1,
        "dt": {"kind": "binary64", "value": "0x1.0000000000000p+0"},
        "endpoint_ulps": 4, "interval_rule": "no_spatial_exchanges",
        "shrink": {"kind": "binary64", "value": "0x1.0000000000000p-1"},
        "max_rejections": 1,
    }
    assert legacy.initial_request(.8, 1.) == 1.
    assert ComputedDt.from_data(legacy.to_data()) == legacy


def test_remaining_policy_is_versioned_closed_and_cannot_be_silently_downgraded():
    policy = ComputedDt(1., clip_to_frontier=True)
    wire = policy.to_data()
    assert wire["schema_version"] == 2 and wire["request_limit"] == "remaining"
    assert ComputedDt.from_data(wire) == policy
    for key, value in (("schema_version", 1), ("request_limit", "nominal"), ("request_limit", True)):
        invalid = dict(wire, **{key: value})
        with pytest.raises(ValueError):
            ComputedDt.from_data(invalid)
    with pytest.raises(TypeError, match="must be bool"):
        ComputedDt(1., clip_to_frontier=1)


@pytest.mark.parametrize("nominal,start,limit", [(1., .875, 1.), (.25, .125, 1.), (.5, 1.25, 1.375)])
def test_request_is_remaining_at_this_entry_and_restart_authenticates_it(nominal, start, limit):
    native, temporal, prepared = subject(ComputedDt(nominal, clip_to_frontier=True), start=start)
    prepared.run_step(native, t_end=limit)
    wanted = min(nominal, limit-start)
    assert native.calls == [wanted]
    assert native.time() == start+wanted
    payload = temporal.checkpoint_json(time=native.time(), macro_step=1)
    restored = TemporalRestartState.from_json(payload, time=native.time(), macro_step=1)
    assert restored.controller_state == temporal.controller_state
    changed = json.loads(payload)
    changed["controller_state"]["program_frontier"]["requested_duration"] = (2*wanted).hex()
    with pytest.raises(ValueError, match="Program|program"):
        TemporalRestartState.from_json(json.dumps(changed), time=native.time(), macro_step=1)


def test_retry_shrinks_capped_request_and_restart_binds_limit_and_rejection_count():
    native, temporal, prepared = subject(ComputedDt(1., max_rejections=1, clip_to_frontier=True),
                                         start=.75, rejected=1)
    report = prepared.run_step(native, t_end=1.)
    assert native.calls == [.25, .125] and report.attempts == 2
    assert native.time() == .875
    payload = temporal.checkpoint_json(time=native.time(), macro_step=1)
    TemporalRestartState.from_json(payload, time=native.time(), macro_step=1)
    for field, value in (("limit", 2.0.hex()), ("rejections", 0)):
        changed = json.loads(payload)
        changed["controller_state"]["program_frontier"][field] = value
        with pytest.raises(ValueError, match="Program|program"):
            TemporalRestartState.from_json(json.dumps(changed), time=native.time(), macro_step=1)


def test_each_new_step_uses_its_own_entry_and_effective_duration():
    native, temporal, prepared = subject(ComputedDt(1., clip_to_frontier=True), factor=.5)
    for expected in (1., .5, .25):
        prepared.run_step(native, t_end=1.)
        assert native.calls[-1] == expected
    assert native.time() == .875
    assert temporal.controller_state["program_frontier"]["duration"] == .125.hex()


def test_clipping_does_not_authorize_a_computed_overshoot():
    native, temporal, prepared = subject(ComputedDt(1., clip_to_frontier=True), start=.875, factor=2.)
    before = copy.deepcopy(temporal.to_data())
    with pytest.raises(RuntimeError, match="crosses.*frontier"):
        prepared.run_step(native, t_end=1.)
    assert native.calls == [.125]
    after = temporal.to_data()
    assert after["clock"] == before["clock"]
    assert after["controller_state"] == before["controller_state"]
    assert after["status"] == "failed" and after["transaction_stats"]["failed"] == 1
