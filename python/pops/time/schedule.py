"""Typed, extensible schedule algebra for temporal Program nodes.

Schedules are values, not mini configuration dictionaries: a :class:`Domain` says which
runtime timeline owns a cadence, a :class:`Trigger` says when it is due, and an
:class:`OffPolicy` says what a consumer observes between due instants.  No public constructor
accepts a kind or policy string.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from pops.params.use_sites import ParamUse, resolve_param_use
from pops.time.points import Clock
from pops.time.schedule_domains import (
    AMRLevel,
    AcceptedStep,
    Attempt,
    ClockTick,
    Domain,
    Event,
    EventHandle,
    Stage,
    WallOutput,
)
from pops.time.schedule_lowering import (
    ScheduleAction,
    ScheduleComment,
    ScheduleDomainIR,
    ScheduleDueIR,
    ScheduleDueKind,
    ScheduleLoweringIR,
    ScheduleOffIR,
    ScheduleTimeline,
)
from pops.time.schedule_protocol import (
    UnresolvedScheduleCondition,
    component_payload as _component_payload,
    manifest_value as _manifest_value,
    map_component as _map_component,
)


@dataclass(frozen=True, slots=True)
class Trigger:
    """Extension interface for typed triggers.

    A third-party trigger must implement :meth:`native_schedule_due` and return the exact
    :class:`ScheduleDueIR` contract.  Mapping onto the closed native due primitives keeps
    backend lowering deterministic and fail-closed.
    """

    domain: Domain
    manifest_tag: ClassVar[str | None] = None
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self) is Trigger:
            raise TypeError("Trigger is abstract; use or subclass a concrete Trigger")
        _component_payload(self, frozenset({"domain"}))
        if not isinstance(self.domain, Domain):
            raise TypeError("%s domain must implement the Domain interface" % type(self).__name__)

    def native_schedule_due(self, *, where: str) -> ScheduleDueIR:
        raise NotImplementedError(
            "schedule trigger %s at %s does not implement native_schedule_due()"
            % (type(self).__name__, where)
        )

    def schedule_params(self) -> dict[str, Any]:
        return self.schedule_payload()

    def is_always(self) -> bool:
        due = self.native_schedule_due(where="Trigger.is_always()")
        if type(due) is not ScheduleDueIR:
            raise TypeError("Trigger.native_schedule_due() must return an exact ScheduleDueIR")
        return due.kind is ScheduleDueKind.ALWAYS

    def map_values(self, mapper: Callable[[Any], Any]) -> Trigger:
        mapped = _map_component(self, mapper, frozenset({"domain"}), domain=self.domain)
        if type(mapped) is not type(self):
            raise TypeError("Trigger.map_values() must preserve the exact extension type")
        return mapped

    def schedule_payload(self) -> dict[str, Any]:
        return _component_payload(self, frozenset({"domain"}))

    def to_schedule_data(self) -> dict[str, Any]:
        tag = type(self).manifest_tag
        if not isinstance(tag, str) or not tag:
            raise TypeError(
                "schedule trigger %s must declare a non-empty manifest_tag" % type(self).__name__
            )
        payload = self.schedule_payload()
        if type(payload) is not dict:
            raise TypeError("Trigger.schedule_payload() must return an exact dict")
        if "type" in payload:
            raise ValueError("Trigger.schedule_payload() uses reserved key 'type'")
        return {"type": tag, **{key: _manifest_value(item) for key, item in payload.items()}}

    def consumer_due(self, coordinate: int, moment: Any) -> bool:
        raise NotImplementedError(
            "schedule trigger %s does not implement consumer_due()" % type(self).__name__
        )


@dataclass(frozen=True, slots=True)
class Always(Trigger):
    manifest_tag: ClassVar[str | None] = "always"

    def native_schedule_due(self, *, where: str) -> ScheduleDueIR:
        del where
        return ScheduleDueIR(ScheduleDueKind.ALWAYS)

    def consumer_due(self, coordinate: int, moment: Any) -> bool:
        del coordinate
        return not moment.at_start


@dataclass(frozen=True, slots=True)
class Every(Trigger):
    n: int
    manifest_tag: ClassVar[str | None] = "every"

    def __post_init__(self) -> None:
        super(Every, self).__post_init__()
        n = resolve_param_use(self.n, ParamUse.SCHEDULE, where="Every(n=)")
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError("Every n must be a positive int")
        object.__setattr__(self, "n", n)

    def native_schedule_due(self, *, where: str) -> ScheduleDueIR:
        del where
        return ScheduleDueIR(ScheduleDueKind.CACHE_PERIOD, period=self.n)

    def schedule_params(self) -> dict[str, Any]:
        return {"n": self.n}

    def consumer_due(self, coordinate: int, moment: Any) -> bool:
        return not moment.at_start and coordinate % self.n == 0


@dataclass(frozen=True, slots=True)
class AtStart(Trigger):
    manifest_tag: ClassVar[str | None] = "at_start"

    def native_schedule_due(self, *, where: str) -> ScheduleDueIR:
        del where
        return ScheduleDueIR(ScheduleDueKind.MACRO_STEP_ZERO)

    def consumer_due(self, coordinate: int, moment: Any) -> bool:
        del coordinate
        return moment.at_start


@dataclass(frozen=True, slots=True)
class AtEnd(Trigger):
    manifest_tag: ClassVar[str | None] = "at_end"

    def native_schedule_due(self, *, where: str) -> ScheduleDueIR:
        del where
        return ScheduleDueIR(ScheduleDueKind.AT_END)

    def consumer_due(self, coordinate: int, moment: Any) -> bool:
        del coordinate
        return not moment.at_start and moment.at_end


@dataclass(frozen=True, slots=True)
class When(Trigger):
    condition: Any
    manifest_tag: ClassVar[str | None] = "when"

    def native_schedule_due(self, *, where: str) -> ScheduleDueIR:
        del where
        return ScheduleDueIR(ScheduleDueKind.PROGRAM_PREDICATE, predicate=self.condition)

    def schedule_params(self) -> dict[str, Any]:
        return {"cond": self.condition}

    def consumer_due(self, coordinate: int, moment: Any) -> bool:
        del coordinate
        if moment.at_start:
            return False
        if type(self.condition) is not bool:
            raise UnresolvedScheduleCondition(self.condition)
        return self.condition


@dataclass(frozen=True, slots=True)
class OffPolicy:
    """Extension interface for the meaning of an off-cadence read."""

    manifest_tag: ClassVar[str | None] = None
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self) is OffPolicy:
            raise TypeError("OffPolicy is abstract; use or subclass a concrete OffPolicy")
        _component_payload(self, frozenset())

    def native_schedule_off(self, *, where: str) -> ScheduleOffIR:
        raise NotImplementedError(
            "schedule off-policy %s at %s does not implement native_schedule_off()"
            % (type(self).__name__, where)
        )

    def needs_cache(self) -> bool:
        plan = self.native_schedule_off(where="cache capability check")
        if type(plan) is not ScheduleOffIR:
            raise TypeError("OffPolicy.native_schedule_off() must return an exact ScheduleOffIR")
        actions = plan.before_due + plan.after_due + plan.off_cadence
        cache_actions = frozenset(
            {
                ScheduleAction.EFFECTIVE_DT,
                ScheduleAction.STORE,
                ScheduleAction.ACCUMULATE_DT,
                ScheduleAction.RESTORE,
            }
        )
        return any(action in cache_actions for action in actions)

    def schedule_payload(self) -> dict[str, Any]:
        return _component_payload(self, frozenset())

    def map_values(self, mapper: Callable[[Any], Any]) -> OffPolicy:
        mapped = _map_component(self, mapper, frozenset())
        if type(mapped) is not type(self):
            raise TypeError("OffPolicy.map_values() must preserve the exact extension type")
        return mapped

    def to_schedule_data(self) -> str | dict[str, Any]:
        tag = type(self).manifest_tag
        if not isinstance(tag, str) or not tag:
            raise TypeError(
                "schedule off-policy %s must declare a non-empty manifest_tag" % type(self).__name__
            )
        payload = self.schedule_payload()
        if type(payload) is not dict:
            raise TypeError("OffPolicy.schedule_payload() must return an exact dict")
        if "type" in payload:
            raise ValueError("OffPolicy.schedule_payload() uses reserved key 'type'")
        if not payload:
            return tag
        return {"type": tag, **{key: _manifest_value(item) for key, item in payload.items()}}


@dataclass(frozen=True, slots=True)
class Hold(OffPolicy):
    manifest_tag: ClassVar[str | None] = "hold"

    def native_schedule_off(self, *, where: str) -> ScheduleOffIR:
        del where
        return ScheduleOffIR(
            after_due=(ScheduleAction.STORE,), off_cadence=(ScheduleAction.RESTORE,)
        )


@dataclass(frozen=True, slots=True)
class Skip(OffPolicy):
    manifest_tag: ClassVar[str | None] = "skip"

    def native_schedule_off(self, *, where: str) -> ScheduleOffIR:
        del where
        return ScheduleOffIR(comment=ScheduleComment.SKIP)


@dataclass(frozen=True, slots=True)
class Zero(OffPolicy):
    manifest_tag: ClassVar[str | None] = "zero"

    def native_schedule_off(self, *, where: str) -> ScheduleOffIR:
        del where
        return ScheduleOffIR(off_cadence=(ScheduleAction.ZERO,))


@dataclass(frozen=True, slots=True)
class AccumulateDt(OffPolicy):
    manifest_tag: ClassVar[str | None] = "accumulate_dt"

    def native_schedule_off(self, *, where: str) -> ScheduleOffIR:
        del where
        return ScheduleOffIR(
            before_due=(ScheduleAction.EFFECTIVE_DT,),
            after_due=(ScheduleAction.STORE,),
            off_cadence=(ScheduleAction.ACCUMULATE_DT, ScheduleAction.RESTORE),
        )


@dataclass(frozen=True, slots=True)
class Error(OffPolicy):
    manifest_tag: ClassVar[str | None] = "error"

    def native_schedule_off(self, *, where: str) -> ScheduleOffIR:
        del where
        return ScheduleOffIR(off_cadence=(ScheduleAction.ERROR,))


@dataclass(frozen=True, slots=True)
class Schedule:
    """One exact typed cadence. ``off`` may be omitted only while its result is never read."""

    trigger: Trigger
    off: OffPolicy | None = None
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        _component_payload(self, frozenset({"trigger", "off"}))
        if not isinstance(self.trigger, Trigger):
            raise TypeError("Schedule trigger must implement the Trigger interface")
        if self.off is not None and not isinstance(self.off, OffPolicy):
            raise TypeError(
                "Schedule off must implement the OffPolicy interface or be None"
            )
        if self.is_always() and self.off is not None:
            raise ValueError("Always has no off-cadence instant; off must be omitted")

    @property
    def domain(self) -> Domain:
        return self.trigger.domain

    @property
    def clock(self) -> Clock:
        return self.domain.clock

    @property
    def params(self) -> dict[str, Any]:
        """Internal generic-IR projection used by graph walkers; not an authoring surface."""
        params = self.trigger.schedule_params()
        if type(params) is not dict:
            raise TypeError("Trigger.schedule_params() must return an exact dict")
        return params

    def is_always(self) -> bool:
        result = self.trigger.is_always()
        if type(result) is not bool:
            raise TypeError("Trigger.is_always() must return an exact bool")
        return result

    def needs_cache(self) -> bool:
        if self.off is None:
            return False
        result = self.off.needs_cache()
        if type(result) is not bool:
            raise TypeError("OffPolicy.needs_cache() must return an exact bool")
        return result

    def native_schedule_ir(self, *, where: str) -> ScheduleLoweringIR:
        """Return the exact native lowering contract implemented by this schedule."""
        domain = self.domain.native_schedule_domain(where=where)
        due = self.trigger.native_schedule_due(where=where)
        off = ScheduleOffIR() if self.off is None else self.off.native_schedule_off(where=where)
        return ScheduleLoweringIR(domain=domain, due=due, off=off)

    def validate_site(self, *, clock: Clock, point: Any = None, where: str = "schedule") -> None:
        if self.clock != clock:
            raise ValueError(
                "%s clock %r does not match evaluation clock %r"
                % (where, self.clock.name, clock.name)
            )
        if self.domain.at is not None and point is not None and self.domain.at != point:
            raise ValueError("%s point does not match its typed domain point" % where)

    def map_values(self, mapper: Callable[[Any], Any]) -> Schedule:
        trigger = self.trigger.map_values(mapper)
        if type(trigger) is not type(self.trigger):
            raise TypeError("Trigger.map_values() must preserve the exact extension type")
        off = None if self.off is None else self.off.map_values(mapper)
        if self.off is not None and type(off) is not type(self.off):
            raise TypeError("OffPolicy.map_values() must preserve the exact extension type")
        mapped = _map_component(
            self, mapper, frozenset({"trigger", "off"}), trigger=trigger, off=off
        )
        if type(mapped) is not type(self):
            raise TypeError("Schedule.map_values() must preserve the exact extension type")
        return mapped

    def schedule_payload(self) -> dict[str, Any]:
        return _component_payload(self, frozenset({"trigger", "off"}))

    def to_data(self) -> dict[str, Any]:
        domain = self.domain.to_schedule_data()
        if type(domain) is not dict or not isinstance(domain.get("type"), str):
            raise TypeError(
                "Domain.to_schedule_data() must return an exact dict with a type string"
            )
        trigger = self.trigger.to_schedule_data()
        if type(trigger) is not dict or not isinstance(trigger.get("type"), str):
            raise TypeError(
                "Trigger.to_schedule_data() must return an exact dict with a type string"
            )
        off = None if self.off is None else self.off.to_schedule_data()
        if off is not None and not (
            (isinstance(off, str) and off)
            or (type(off) is dict and isinstance(off.get("type"), str))
        ):
            raise TypeError(
                "OffPolicy.to_schedule_data() must return a non-empty string or typed dict"
            )
        payload = self.schedule_payload()
        if type(payload) is not dict:
            raise TypeError("Schedule.schedule_payload() must return an exact dict")
        if frozenset({"schema_version", "domain", "trigger", "off"}).intersection(payload):
            raise ValueError("Schedule.schedule_payload() uses a reserved manifest key")
        return {
            "schema_version": 1,
            "domain": domain,
            "trigger": trigger,
            "off": off,
            **{key: _manifest_value(item) for key, item in payload.items()},
        }

    def __repr__(self) -> str:
        return "Schedule(%r%s)" % (self.trigger, ", off=%r" % self.off if self.off else "")


# Ergonomic helpers build only the exact typed algebra; none accepts a selector string.
def always(*, clock: Clock) -> Schedule:
    return Schedule(Always(AcceptedStep(clock)))


def every(n: Any, *, clock: Clock) -> Schedule:
    return Schedule(Every(AcceptedStep(clock), n))


def when(cond: Any, *, clock: Clock) -> Schedule:
    return Schedule(When(AcceptedStep(clock), cond))


def on_start(*, clock: Clock) -> Schedule:
    return Schedule(AtStart(AcceptedStep(clock)))


def on_end(*, clock: Clock) -> Schedule:
    return Schedule(AtEnd(AcceptedStep(clock)))


__all__ = [
    "ScheduleTimeline",
    "ScheduleDueKind",
    "ScheduleAction",
    "ScheduleComment",
    "ScheduleDomainIR",
    "ScheduleDueIR",
    "ScheduleOffIR",
    "ScheduleLoweringIR",
    "Domain",
    "AcceptedStep",
    "Attempt",
    "Stage",
    "ClockTick",
    "AMRLevel",
    "EventHandle",
    "Event",
    "WallOutput",
    "Trigger",
    "Always",
    "Every",
    "AtStart",
    "AtEnd",
    "When",
    "OffPolicy",
    "Hold",
    "Skip",
    "Zero",
    "AccumulateDt",
    "Error",
    "Schedule",
    "always",
    "every",
    "when",
    "on_start",
    "on_end",
]
