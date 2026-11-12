"""N8 lot « Règle de pas compilée » (design note revision 1.1): the step rule of an Advance in its
compiled control (schema 4) and in its elaboration, without a native run."""
import pytest

import examples.api_v02.models_M01_M14 as early
from pops.time import AdaptiveCFL, FixedDt
from pops.time._program.control import (
    ADVANCE_CONTROL_MEMORY_SCHEMA, ADVANCE_CONTROL_SCHEMA, ADVANCE_CONTROL_STEP_RULE_SCHEMA,
    AdvanceControl,
)
from pops.time._program.elaboration import Advance, Proposal, Retry, StepRule, elaborate
from pops.time._program.traced_method import Candidate, method

DEFAULT = (("finite", "default"),)


def _control(**options):
    values = dict(assessments=DEFAULT, max_attempts=2, max_transforms=0, first_dt=None,
                  step_rule=(0.45, None))
    values.update(options)
    return AdvanceControl(**values)


@pytest.mark.parametrize("max_dt", [None, 0.005])
def test_a_rule_records_schema_four_and_round_trips(max_dt):
    control = _control(step_rule=(0.45, max_dt))
    data = control.to_data()
    assert data["schema"] == ADVANCE_CONTROL_STEP_RULE_SCHEMA == "pops.advance-control@4"
    expected = {"rule": "cfl", "cfl": (0.45).hex()}
    if max_dt is not None:
        expected["max_dt"] = max_dt.hex()
    assert data["proposal"] == expected
    assert set(data) == {"schema", "accept", "retry", "proposal", "outputs", "memory",
                         "trial_failures"}
    assert AdvanceControl.from_data(data) == control


def test_a_control_without_a_rule_keeps_its_record():
    """E1: no existing identity moves; schemas 1 to 3 are written as before."""
    fixed = AdvanceControl(assessments=DEFAULT, max_attempts=1, max_transforms=0, first_dt=0.001)
    assert fixed.step_rule is None
    assert fixed.to_data() == {"schema": ADVANCE_CONTROL_SCHEMA, "accept": [
        {"assess": "finite", "origin": "default"}], "retry": {"max_attempts": 1,
        "max_transforms": 0}, "proposal": {"dt": (0.001).hex()}}
    memory = AdvanceControl(assessments=DEFAULT, max_attempts=1, max_transforms=0,
                            first_dt=0.001, trial_failures=({
                                "site": "s", "guard": "solve", "variant": "SolveFailed",
                                "retry_can_change": False, "classified": True, "lot": None},))
    assert memory.to_data()["schema"] == ADVANCE_CONTROL_MEMORY_SCHEMA
    assert memory.to_data()["proposal"] == {"dt": (0.001).hex()}


@pytest.mark.parametrize("change", ["rule proposal under schema 3", "dt under schema 4",
                                    "another rule", "a stray key", "both"])
def test_a_record_that_mixes_rule_and_dt_is_refused(change):
    data = _control(step_rule=(0.45, 0.005)).to_data()
    if change == "rule proposal under schema 3":
        data["schema"] = ADVANCE_CONTROL_MEMORY_SCHEMA
    elif change == "dt under schema 4":
        data["proposal"] = {"dt": (0.001).hex()}
    elif change == "another rule":
        data["proposal"]["rule"] = "error"
    elif change == "a stray key":
        data["proposal"]["min_dt"] = (0.001).hex()
    else:
        data["proposal"]["dt"] = (0.001).hex()
    with pytest.raises(ValueError, match="another schema|malformed proposal"):
        AdvanceControl.from_data(data)


@pytest.mark.parametrize("options, message", [
    (dict(first_dt=0.001), "no first_dt"),
    (dict(step_rule=(0.0, None)), "cfl"),
    (dict(step_rule=(float("inf"), None)), "cfl"),
    (dict(step_rule=(0.45, -1.0)), "max_dt"),
    (dict(step_rule=(0.45, float("nan"))), "max_dt"),
    (dict(step_rule=(1, None)), "cfl"),
    (dict(step_rule=None), "first_dt"),
])
def test_a_malformed_rule_is_refused(options, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _control(**options)


@pytest.mark.parametrize("arguments", [(0,), (-0.4,), (float("nan"),), (True,), ("0.4",),
                                       (0.4, 0.0), (0.4, float("inf"))])
def test_step_rule_takes_positive_finite_numbers(arguments):
    with pytest.raises((TypeError, ValueError), match="StepRule"):
        StepRule(*arguments)


@method
def _euler(B, p, rate):
    return Candidate(B.state + p.dt * rate(B.state, B.time), B.time + p.dt)


def _parts(monkeypatch):
    seen = {}

    def capture(case, block, state, flow, dt, *, name, memory, record, b6):
        seen.update(case=case, block=block, state=state, flow=flow)
    with monkeypatch.context() as patch:
        patch.setattr(early, "_traced_recorded_ssprk2", capture)
        early.build_m08()
    return seen


@pytest.mark.parametrize("rule", [StepRule(0.45), StepRule(0.45, max_dt=0.005)])
def test_a_rule_elaborates_master_adaptive_cfl_and_its_control(rule, monkeypatch):
    parts = _parts(monkeypatch)
    elaboration = elaborate(Advance(_euler(rate=parts["flow"]), retry=Retry(max_attempts=2)),
                            case=parts["case"], state=parts["block"][parts["state"]],
                            proposal=rule, name="probe")
    strategy = elaboration.program._step_strategy
    assert type(strategy) is AdaptiveCFL
    assert (strategy.cfl, strategy.max_dt) == (rule.cfl, rule.max_dt)
    assert elaboration.control.step_rule == (rule.cfl, rule.max_dt)
    assert elaboration.control.first_dt is None
    ir = elaboration.program._serialize(include_provenance=False)
    assert ir["version"] == 33 and ir["advance_control"] == elaboration.control.to_data()


def test_a_proposal_still_elaborates_a_fixed_step(monkeypatch):
    parts = _parts(monkeypatch)
    elaboration = elaborate(Advance(_euler(rate=parts["flow"])), case=parts["case"],
                            state=parts["block"][parts["state"]], proposal=Proposal(0.001))
    strategy = elaboration.program._step_strategy
    assert type(strategy) is FixedDt and elaboration.control.step_rule is None
    assert elaboration.program._serialize(include_provenance=False)["version"] < 33
