#!/usr/bin/env python3
"""Strict native schedule extension protocol and built-in parity."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace


def _skip(message):
    print("skip test_schedule_extension_protocol (%s)" % message)
    sys.exit(0)


try:
    from pops import time as adctime
    from pops.codegen.program_emit_schedule import _lower_schedule_ir
    from pops.runtime._consumer_contracts import ConsumerMoment
    from pops.runtime._consumer_planning import _is_due, _schedule_coordinate
    from pops.time.points import TimePoint
    from pops.time.schedule import (
        ScheduleAction,
        ScheduleDueIR,
        ScheduleDueKind,
        ScheduleLoweringIR,
        ScheduleOffIR,
    )
    from typed_program_support import typed_state
except Exception as exc:  # noqa: BLE001 -- installed package unavailable in this interpreter
    _skip("pops unavailable: %s" % exc)


@dataclass(frozen=True, slots=True)
class AcceptedChannel(adctime.AcceptedStep):
    """Configurable third-party domain sharing the accepted-step native timeline."""

    channel: str = "primary"
    manifest_tag = "test.accepted_channel"


@dataclass(frozen=True, slots=True)
class ConfigurableEvery(adctime.Trigger):
    """Configurable third-party trigger built from a supported native cadence primitive."""

    period: int
    label: str = "external"
    manifest_tag = "test.configurable_every"

    def native_schedule_due(self, *, where: str) -> ScheduleDueIR:
        del where
        return ScheduleDueIR(ScheduleDueKind.CACHE_PERIOD, period=self.period)

    def consumer_due(self, coordinate, moment):
        return not moment.at_start and coordinate % self.period == 0


@dataclass(frozen=True, slots=True)
class ThirdPartyHold(adctime.OffPolicy):
    """Example third-party policy composed only from validated native actions."""

    def native_schedule_off(self, *, where: str) -> ScheduleOffIR:
        del where
        return ScheduleOffIR(
            after_due=(ScheduleAction.STORE,),
            off_cadence=(ScheduleAction.RESTORE,),
        )

    manifest_tag = "test.third_party_hold"


@dataclass(frozen=True, slots=True)
class ThirdPartySchedule(adctime.Schedule):
    audit_label: str = "external-schedule"


@dataclass(frozen=True, slots=True)
class ThirdPartyPredicate(adctime.Trigger):
    condition: object
    predicate_name: str = "external-predicate"
    manifest_tag = "test.third_party_predicate"

    def native_schedule_due(self, *, where: str) -> ScheduleDueIR:
        del where
        return ScheduleDueIR(ScheduleDueKind.PROGRAM_PREDICATE, predicate=self.condition)

    def consumer_due(self, coordinate, moment):
        del coordinate
        if moment.at_start:
            return False
        if type(self.condition) is not bool:
            from pops.time.schedule_protocol import UnresolvedScheduleCondition

            raise UnresolvedScheduleCondition(self.condition)
        return self.condition


@dataclass(frozen=True, slots=True)
class MissingNativeProtocol(adctime.Trigger):
    manifest_tag = "test.missing_native_protocol"


@dataclass(frozen=True, slots=True)
class BadDueReturn(adctime.Trigger):
    manifest_tag = "test.bad_due_return"

    def native_schedule_due(self, *, where: str):
        del where
        return {"kind": "cache_period", "period": 3}


@dataclass(frozen=True, slots=True)
class BadOffReturn(adctime.OffPolicy):
    manifest_tag = "test.bad_off_return"

    def native_schedule_off(self, *, where: str):
        del where
        return {"off": "skip"}


@dataclass(frozen=True, slots=True)
class BadScheduleReturn(adctime.Schedule):
    def native_schedule_ir(self, *, where: str):
        del where
        return {"domain": "accepted_step"}


def _node(schedule):
    return SimpleNamespace(
        id=37,
        name="third_party_rhs",
        op="rhs",
        attrs={"schedule": schedule},
    )


def _expect_error(error_type, expected, callback):
    try:
        callback()
    except error_type as exc:
        assert expected in str(exc), str(exc)
    else:
        raise AssertionError("expected %s containing %r" % (error_type.__name__, expected))


def _scratch_program(schedule):
    program = adctime.Program("schedule_extension")
    schedule = schedule(program.clock) if callable(schedule) else schedule
    state = typed_state(program, "ions")
    rate = program._rhs_legacy(state=state, flux=True, sources=["default"])
    rate = program._replace_value(rate, attrs={**rate.attrs, "schedule": schedule})
    endpoint = typed_state(program, "ions", state_name="U").next
    program.commit(endpoint, program.value("U1", state + program.dt * rate, at=endpoint.point))
    return program


def test_third_party_schedule_lowers_without_codegen_registration():
    program = adctime.Program("third_party_schedule_clock")

    def make_schedule(clock):
        return ThirdPartySchedule(
            ConfigurableEvery(AcceptedChannel(clock, channel="diagnostics"), period=3),
            off=ThirdPartyHold(),
            audit_label="round-trip",
        )

    schedule = make_schedule(program.clock)
    built_in = adctime.Schedule(
        adctime.Every(adctime.AcceptedStep(program.clock), 3),
        off=adctime.Hold(),
    )
    assert schedule.native_schedule_ir(where="test") == built_in.native_schedule_ir(where="test")
    assert schedule.needs_cache()

    compiled = _scratch_program(make_schedule)
    compiled._check_schedules_lowerable()
    cpp = compiled.emit_cpp_program(model=None)
    assert "ctx.cache_should_update(" in cpp and ", 3)" in cpp
    assert "ctx.cache_store_scratch(" in cpp
    assert "ctx.cache_restore_scratch(" in cpp

    manifest = SimpleNamespace(schedule=schedule, qualified_id="test.consumer")
    moment = ConsumerMoment(point=TimePoint(program.clock), accepted_step=6, attempt=0)
    assert _schedule_coordinate(manifest, moment) == 6
    assert _is_due(manifest, moment) is True


def test_configurable_schedule_round_trip_preserves_exact_extension_identity():
    def make_schedule(clock):
        return ThirdPartySchedule(
            ConfigurableEvery(
                AcceptedChannel(clock, channel="checkpoint"), period=5, label="configured"
            ),
            off=ThirdPartyHold(),
            audit_label="identity",
        )

    authored = _scratch_program(make_schedule)
    before = authored._serialize(include_provenance=False)
    rebuilt = authored._rebuild(lambda value: True)
    after = rebuilt._serialize(include_provenance=False)
    rebuilt_schedule = next(
        value.attrs["schedule"]
        for value in rebuilt._values
        if value.attrs.get("schedule") is not None
    )

    assert before == after
    assert type(rebuilt_schedule) is ThirdPartySchedule
    assert type(rebuilt_schedule.domain) is AcceptedChannel
    assert type(rebuilt_schedule.trigger) is ConfigurableEvery
    assert type(rebuilt_schedule.off) is ThirdPartyHold
    assert rebuilt_schedule.audit_label == "identity"
    assert rebuilt_schedule.domain.channel == "checkpoint"
    assert rebuilt_schedule.trigger.period == 5
    encoded = next(
        node["attrs"]["schedule"] for node in after["nodes"] if "schedule" in node["attrs"]
    )
    assert encoded["type"]["qualname"] == "ThirdPartySchedule"
    assert encoded["domain"]["payload"] == {"channel": "checkpoint"}
    assert encoded["trigger"]["payload"] == {"period": 5, "label": "configured"}


def test_third_party_predicate_survives_call_validation_and_rebuild():
    program = adctime.Program("third_party_predicate")
    state = typed_state(program, "plasma")
    condition = program.norm2(state) > 0
    schedule = adctime.Schedule(
        ThirdPartyPredicate(AcceptedChannel(program.clock), condition),
        off=ThirdPartyHold(),
    )
    operator = SimpleNamespace(name="external_rate", capabilities={"cacheable": True})
    program._validate_schedule(operator, schedule, (state,))
    rate = program._rhs_legacy(state=state, flux=True, sources=["default"])
    rate = program._replace_value(rate, attrs={**rate.attrs, "schedule": schedule})
    endpoint = typed_state(program, "plasma", state_name="U").next
    program.commit(
        endpoint,
        program.value("U1", state + program.dt * rate, at=endpoint.point),
    )
    program._check_schedules_lowerable()
    before = program._serialize(include_provenance=False)
    rebuilt = program._rebuild(lambda value: True)
    after = rebuilt._serialize(include_provenance=False)
    rebuilt_schedule = next(
        value.attrs["schedule"]
        for value in rebuilt._values
        if value.attrs.get("schedule") is not None
    )

    assert before == after
    assert type(rebuilt_schedule.trigger) is ThirdPartyPredicate
    assert rebuilt_schedule.trigger.predicate_name == "external-predicate"
    assert rebuilt_schedule.trigger.condition is rebuilt._values[condition.id]
    encoded = next(
        node["attrs"]["schedule"] for node in after["nodes"] if "schedule" in node["attrs"]
    )
    assert encoded["trigger"]["payload"]["condition"] == {"program_value_id": condition.id}


def test_missing_trigger_protocol_fails_closed():
    program = adctime.Program("missing_schedule_protocol")
    _expect_error(
        NotImplementedError,
        "does not implement native_schedule_due",
        lambda: adctime.Schedule(
            MissingNativeProtocol(adctime.AcceptedStep(program.clock)),
            off=adctime.Skip(),
        ),
    )


def test_bad_trigger_return_fails_closed_at_authoring():
    program = adctime.Program("bad_due_return")
    _expect_error(
        TypeError,
        "must return an exact ScheduleDueIR",
        lambda: adctime.Schedule(
            BadDueReturn(adctime.AcceptedStep(program.clock)),
            off=adctime.Skip(),
        ),
    )


def test_bad_nested_ir_return_fails_closed():
    program = adctime.Program("bad_off_return")
    schedule = adctime.Schedule(
        adctime.Every(adctime.AcceptedStep(program.clock), 3),
        off=BadOffReturn(),
    )
    _expect_error(
        TypeError,
        "off must be an exact ScheduleOffIR",
        lambda: _lower_schedule_ir(_node(schedule), schedule),
    )


def test_bad_schedule_return_fails_closed():
    program = adctime.Program("bad_schedule_return")
    schedule = BadScheduleReturn(
        adctime.Every(adctime.AcceptedStep(program.clock), 3),
        off=adctime.Skip(),
    )
    _expect_error(
        TypeError,
        "must return an exact ScheduleLoweringIR",
        lambda: _lower_schedule_ir(_node(schedule), schedule),
    )


def test_builtin_trigger_and_policy_ir_parity():
    program = adctime.Program("builtin_schedule_ir")
    domain = adctime.AcceptedStep(program.clock)
    cases = (
        (adctime.Always(domain), ScheduleDueKind.ALWAYS, None),
        (adctime.Every(domain, 7), ScheduleDueKind.CACHE_PERIOD, 7),
        (adctime.AtStart(domain), ScheduleDueKind.MACRO_STEP_ZERO, None),
    )
    for trigger, kind, period in cases:
        schedule = adctime.Schedule(trigger)
        lowered = schedule.native_schedule_ir(where="parity")
        assert type(lowered) is ScheduleLoweringIR
        assert lowered.due.kind is kind
        assert lowered.due.period == period

    policies = (
        (adctime.Skip(), (), (), ()),
        (adctime.Zero(), (), (), (ScheduleAction.ZERO,)),
        (adctime.Hold(), (), (ScheduleAction.STORE,), (ScheduleAction.RESTORE,)),
        (
            adctime.AccumulateDt(),
            (ScheduleAction.EFFECTIVE_DT,),
            (ScheduleAction.STORE,),
            (ScheduleAction.ACCUMULATE_DT, ScheduleAction.RESTORE),
        ),
        (adctime.Error(), (), (), (ScheduleAction.ERROR,)),
    )
    for policy, before, after, off in policies:
        lowered = adctime.Schedule(adctime.Every(domain, 2), off=policy).native_schedule_ir(
            where="parity"
        )
        assert lowered.off.before_due == before
        assert lowered.off.after_due == after
        assert lowered.off.off_cadence == off


def _run_as_script():
    functions = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for function in functions:
        try:
            function()
            print("  [OK ] %s" % function.__name__)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("  [XX ] %s: %s" % (function.__name__, exc))
    if failures:
        print("FAIL test_schedule_extension_protocol: %d failure(s)" % failures)
        sys.exit(1)
    print("PASS test_schedule_extension_protocol (%d checks)" % len(functions))


if __name__ == "__main__":
    _run_as_script()
