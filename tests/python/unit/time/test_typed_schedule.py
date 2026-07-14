"""ADC-663 typed schedule algebra and honest native lowering boundary."""
from __future__ import annotations
from pops.codegen.program_codegen import _check_schedules_lowerable

import json

import pytest

from typed_program_support import typed_state

from pops.time import Program
from pops.numerics.terms import Flux
from pops.time.points import Clock, StagePoint, TimePoint
from pops.time import (
    AMRLevel, AcceptedStep, AccumulateDt, Always, Attempt, AtEnd, AtStart,
    ClockTick, Error, Event, EventHandle, Every, Hold, Schedule, Skip, Stage, WallOutput,
    When, Zero,
)


def test_schedule_is_a_typed_product_without_string_selectors():
    clock = Clock("macro")
    schedule = Schedule(Every(AcceptedStep(clock), 4), off=Hold())

    assert schedule.clock == clock
    assert schedule.trigger.n == 4
    assert isinstance(schedule.off, Hold)
    with pytest.raises(TypeError, match="Trigger"):
        Schedule("every")
    with pytest.raises(TypeError, match="OffPolicy"):
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
    rate = program.rhs(state=state.n, terms=[Flux()])
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
        _check_schedules_lowerable(program)


def test_typed_schedule_serialization_and_rebuild_are_exact():
    program, rate = _scheduled_rate(off=Zero())
    encoded = program._serialize()["nodes"][rate.id]["attrs"]["schedule"]

    assert encoded["schema_version"] == 3
    assert encoded["domain"]["type"] == {
        "uri": "pops://time/schedule/domains/accepted-step", "version": 1,
    }
    assert encoded["trigger"]["type"] == {
        "uri": "pops://time/schedule/triggers/every", "version": 1,
    }
    assert encoded["trigger"]["payload"] == {"n": 2}
    assert encoded["off"]["type"] == {
        "uri": "pops://time/schedule/off-policies/zero", "version": 1,
    }
    assert encoded["off"]["payload"] == {}
    rebuilt = program._rebuild(lambda value: True)
    assert rebuilt._serialize(include_provenance=False) == \
        program._serialize(include_provenance=False)


def test_native_clock_stage_and_level_domains_keep_exact_coordinates():
    clock = Clock("macro")
    stage_point = StagePoint("s1", {"main": TimePoint(clock, 0.5)})

    stage = Stage(clock, stage_point).native_schedule_domain(where="test")
    tick = ClockTick(clock).native_schedule_domain(where="test")
    level = AMRLevel(clock, level=2).native_schedule_domain(where="test")

    assert stage.timeline.value == "stage"
    assert stage.clock_id == clock.qualified_id
    assert stage.stage_identity == json.dumps(
        stage_point.to_data(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert tick.timeline.value == "clock_tick"
    assert tick.clock_id == clock.qualified_id
    assert level.timeline.value == "amr_level"
    assert level.clock_id == clock.qualified_id
    assert level.level == 2


@pytest.mark.parametrize("domain_factory, domain_name", [
    (Attempt, "Attempt"),
    (lambda clock: Event(
        clock, event=EventHandle(Program("event_owner").owner_path, "event")), "Event"),
    (WallOutput, "WallOutput"),
])
def test_future_runtime_domains_are_preserved_but_refused_honestly(
        domain_factory, domain_name):
    program, _ = _scheduled_rate(off=Zero(), domain_factory=domain_factory)
    with pytest.raises(NotImplementedError, match=domain_name):
        _check_schedules_lowerable(program)
