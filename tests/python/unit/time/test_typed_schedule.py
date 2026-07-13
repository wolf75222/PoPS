"""ADC-663 closed schedule algebra and honest native lowering boundary."""
from __future__ import annotations

import pytest

from typed_program_support import typed_state

from pops.time import Program
from pops.time.points import Clock, StagePoint, TimePoint
from pops.time.schedule import (
    AMRLevel, AcceptedStep, AccumulateDt, Always, Attempt, AtEnd, AtStart,
    ClockTick, Error, Event, EventHandle, Every, Hold, Schedule, Skip, Stage, WallOutput,
    When, Zero,
)


def test_schedule_is_a_closed_typed_product_without_string_selectors():
    clock = Clock("macro")
    schedule = Schedule(Every(AcceptedStep(clock), 4), off=Hold())

    assert schedule.clock == clock
    assert schedule.trigger.n == 4
    assert isinstance(schedule.off, Hold)
    with pytest.raises(TypeError, match="closed Trigger"):
        Schedule("every")
    with pytest.raises(TypeError, match="closed OffPolicy"):
        Schedule(Every(AcceptedStep(clock), 4), off="hold")
    with pytest.raises(ValueError, match="no off-cadence"):
        Schedule(Always(AcceptedStep(clock)), off=Skip())


def test_all_final_domains_triggers_and_off_policies_are_values():
    clock = Clock("macro")
    stage_point = StagePoint("s1", {"main": TimePoint(clock, 0.5)})
    domains = (
        AcceptedStep(clock), Attempt(clock), Stage(clock, stage_point), ClockTick(clock),
        AMRLevel(clock, level=2),
        Event(clock, event=EventHandle(Program("owner").owner_path, "shock")),
        WallOutput(clock),
    )
    triggers = (Always(domains[0]), Every(domains[0], 2), AtStart(domains[0]),
                AtEnd(domains[0]), When(domains[0], True))
    policies = (Hold(), Skip(), Zero(), AccumulateDt(), Error())

    assert len(domains) == 7 and len(triggers) == 5 and len(policies) == 5


def test_domain_rejects_foreign_clock_point_and_stage_requires_stage_point():
    macro, fast = Clock("macro"), Clock("fast")
    with pytest.raises(ValueError, match="different clock"):
        AcceptedStep(macro, at=TimePoint(fast))
    with pytest.raises(TypeError, match="StagePoint"):
        Stage(macro, TimePoint(macro))


def _scheduled_rate(*, off=None, domain_factory=AcceptedStep):
    program = Program("scheduled")
    state = typed_state(program, "fluid", state_name="U")
    rate = program._rhs_legacy(state=state.n, sources=[])
    domain = domain_factory(program.clock)
    schedule = Schedule(Every(domain, 2), off=off)
    rate = program._replace_value(rate, attrs={**rate.attrs, "schedule": schedule})
    final = program.value(
        "final", state.n + program.dt * rate, at=state.next.point)
    program.commit(state.next, final)
    return program, rate


def test_scheduled_read_without_off_policy_fails_before_native_lowering():
    program, _ = _scheduled_rate()
    with pytest.raises(ValueError, match="no explicit OffPolicy"):
        program._check_schedules_lowerable()


def test_typed_schedule_serialization_and_rebuild_are_exact():
    program, rate = _scheduled_rate(off=Zero())
    encoded = program._serialize()["nodes"][rate.id]["attrs"]["schedule"]

    assert encoded["domain"]["type"] == "AcceptedStep"
    assert encoded["trigger"] == {"type": "Every", "n": 2}
    assert encoded["off"] == "Zero"
    rebuilt = program._rebuild(lambda value: True)
    assert rebuilt._serialize(include_provenance=False) == \
        program._serialize(include_provenance=False)


@pytest.mark.parametrize("domain_factory, domain_name", [
    (Attempt, "Attempt"), (ClockTick, "ClockTick"),
    (lambda clock: AMRLevel(clock, level=0), "AMRLevel"),
    (lambda clock: Event(
        clock, event=EventHandle(Program("event_owner").owner_path, "event")), "Event"),
    (WallOutput, "WallOutput"),
])
def test_future_runtime_domains_are_preserved_but_refused_honestly(
        domain_factory, domain_name):
    program, _ = _scheduled_rate(off=Zero(), domain_factory=domain_factory)
    with pytest.raises(NotImplementedError, match=domain_name):
        program._check_schedules_lowerable()
