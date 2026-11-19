"""N8 lot « Règle de pas compilée » (design note revision 1.1): the step rule of an Advance in its
compiled control (schema 4) and in its elaboration, without a native run."""
import pytest

import examples.api_v02.models_M01_M14 as early
from pops.time import AdaptiveCFL, FixedDt
from pops.time._program.control import (
    ADVANCE_CONTROL_MEMORY_SCHEMA, ADVANCE_CONTROL_SCHEMA, ADVANCE_CONTROL_STEP_RULE_SCHEMA,
    ADVANCE_CONTROL_TRIAL_VARIANTS_SCHEMA, AdvanceControl,
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
    expected = {"rule": "stability", "fraction": (0.45).hex()}
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
                            first_dt=0.001, memory=(_MEMORY_ROW,))
    assert memory.to_data()["schema"] == ADVANCE_CONTROL_MEMORY_SCHEMA
    assert memory.to_data()["proposal"] == {"dt": (0.001).hex()}
    assert AdvanceControl.from_data(memory.to_data()) == memory
    ruled = _control(memory=(_MEMORY_ROW,))
    assert ruled.to_data()["schema"] == ADVANCE_CONTROL_STEP_RULE_SCHEMA


_MEMORY_ROW = {"path": "m/h", "role": "auxiliary", "type": "t", "writer": "m", "readers": [],
               "ring": "h"}
_SOLVE_ROW = {"site": "m/solve", "guard": "solve", "variant": ["BudgetExhausted", "SolveFailed"],
              "retry_can_change": False, "classified": True, "lot": None}


@pytest.mark.parametrize("step_rule", [None, (0.45, 0.005)])
def test_a_failure_site_records_schema_five_with_either_proposal(step_rule):
    """N9 « La référence sous les solves de relation » §3.6: a control with a failure site records
    schema 5, whose rows list the variants that each site can report, with the proposal of a dt
    or of a rule; it round-trips, and the IR version is 35."""
    first_dt = 0.001 if step_rule is None else None
    control = _control(first_dt=first_dt, step_rule=step_rule, memory=(_MEMORY_ROW,),
                       trial_failures=(_SOLVE_ROW, {
                           "site": "m/load", "guard": "field_evaluation", "variant": [],
                           "retry_can_change": False, "classified": False, "lot": "later"}, {
                           "site": "m/positive", "guard": "admissibility",
                           "variant": ["NotAdmissible"], "retry_can_change": True,
                           "classified": True, "lot": None}))
    data = control.to_data()
    assert data["schema"] == ADVANCE_CONTROL_TRIAL_VARIANTS_SCHEMA == "pops.advance-control@5"
    assert set(data) == {"schema", "accept", "retry", "proposal", "outputs", "memory",
                         "trial_failures"}
    assert ("rule" in data["proposal"]) is (step_rule is not None)
    assert AdvanceControl.from_data(data) == control


@pytest.mark.parametrize("change", [
    "the variant of schema 3", "a solve that lists one variant", "an unclassified solve",
    "a classified site that lists none", "schema 5 relabelled 3", "schema 5 relabelled 4",
    "schema 5 without a site"])
def test_a_failure_site_row_lists_the_variants_of_its_guard(change):
    """A row lists exactly the variants of its guard, classified when it lists one, and only
    schema 5 holds a failure site."""
    data = _control(first_dt=0.001, step_rule=None, trial_failures=(_SOLVE_ROW,)).to_data()
    row = data["trial_failures"][0]
    if change == "the variant of schema 3":
        row["variant"] = "SolveFailed"
    elif change == "a solve that lists one variant":
        row["variant"] = ["SolveFailed"]
    elif change == "an unclassified solve":
        row["classified"] = False
    elif change == "a classified site that lists none":
        row.update(guard="field_evaluation", variant=[])
    elif change == "schema 5 relabelled 3":
        data["schema"] = ADVANCE_CONTROL_MEMORY_SCHEMA
    elif change == "schema 5 relabelled 4":
        data["schema"] = ADVANCE_CONTROL_STEP_RULE_SCHEMA
    else:
        data["trial_failures"] = []
    with pytest.raises(ValueError, match="lists the variants|not canonical|malformed proposal"):
        AdvanceControl.from_data(data)


@pytest.mark.parametrize("change", ["rule proposal under schema 3", "dt under schema 4",
                                    "another rule", "a stray key", "both"])
def test_a_record_that_mixes_rule_and_dt_is_refused(change):
    data = _control(step_rule=(0.45, 0.005)).to_data()
    if change == "rule proposal under schema 3":
        data["schema"] = ADVANCE_CONTROL_MEMORY_SCHEMA
    elif change == "dt under schema 4":
        data["proposal"] = {"dt": (0.001).hex()}
    elif change == "another rule":
        data["proposal"]["rule"] = "cfl"
    elif change == "a stray key":
        data["proposal"]["min_dt"] = (0.001).hex()
    else:
        data["proposal"]["dt"] = (0.001).hex()
    with pytest.raises(ValueError, match="another schema|malformed proposal"):
        AdvanceControl.from_data(data)


@pytest.mark.parametrize("options, message", [
    (dict(first_dt=0.001), "no first_dt"),
    (dict(step_rule=(0.0, None)), "fraction"),
    (dict(step_rule=(float("inf"), None)), "fraction"),
    (dict(step_rule=(0.45, -1.0)), "max_dt"),
    (dict(step_rule=(0.45, float("nan"))), "max_dt"),
    (dict(step_rule=(1, None)), "fraction"),
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


def _rule(max_dt=None):
    """As the slice's library word gives it: the fraction and the builder's strategy."""
    return StepRule(0.45, max_dt, strategy=AdaptiveCFL(0.45, max_dt))


@pytest.mark.parametrize("max_dt", [None, 0.005])
def test_a_rule_elaborates_its_strategy_and_its_control(max_dt, monkeypatch):
    rule = _rule(max_dt)
    parts = _parts(monkeypatch)
    elaboration = elaborate(Advance(_euler(rate=parts["flow"]), retry=Retry(max_attempts=2)),
                            case=parts["case"], state=parts["block"][parts["state"]],
                            proposal=rule, name="probe")
    assert elaboration.program._step_strategy is rule.strategy
    assert elaboration.control.step_rule == (rule.fraction, rule.max_dt) == (0.45, max_dt)
    assert elaboration.control.first_dt is None
    ir = elaboration.program._serialize(include_provenance=False)
    assert ir["advance_control"] == elaboration.control.to_data()
    # M08's flow requests a field, so its failure sites raise the control to schema 5 and the IR
    # to 35 (N9 « La référence sous les solves de relation » §3.6); the rule stays its proposal.
    assert elaboration.control.trial_failures
    assert ir["version"] == 35 and ir["advance_control"]["schema"] == "pops.advance-control@5"
    assert ir["advance_control"]["proposal"]["rule"] == "stability"


def test_a_rule_without_a_strategy_does_not_elaborate(monkeypatch):
    """The core does not name the builder's strategy (CF2): the rule's library word gives it."""
    parts = _parts(monkeypatch)
    rule = StepRule(0.45)
    assert rule.strategy is None and repr(rule) == "StepRule(fraction=0.45, max_dt=None)"
    with pytest.raises(TypeError, match="names no strategy"):
        elaborate(Advance(_euler(rate=parts["flow"])), case=parts["case"],
                  state=parts["block"][parts["state"]], proposal=rule)
    with pytest.raises(TypeError, match="registered StepStrategy"):
        StepRule(0.45, strategy=object())


def test_a_proposal_still_elaborates_a_fixed_step(monkeypatch):
    parts = _parts(monkeypatch)
    elaboration = elaborate(Advance(_euler(rate=parts["flow"])), case=parts["case"],
                            state=parts["block"][parts["state"]], proposal=Proposal(0.001))
    strategy = elaboration.program._step_strategy
    assert type(strategy) is FixedDt and elaboration.control.step_rule is None
    ir = elaboration.program._serialize(include_provenance=False)
    assert ir["advance_control"]["proposal"] == {"dt": (0.001).hex()}
    # No rule: its record is schema 5 for M08's failure sites (§3.6), never the rule's 4.
    assert ir["version"] == 35 and ir["advance_control"]["schema"] == "pops.advance-control@5"
