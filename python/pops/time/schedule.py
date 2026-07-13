"""Closed, typed schedule algebra for temporal Program nodes.

Schedules are values, not mini configuration dictionaries: a :class:`Domain` says which
runtime timeline owns a cadence, a :class:`Trigger` says when it is due, and an
:class:`OffPolicy` says what a consumer observes between due instants.  No public constructor
accepts a kind or policy string.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pops.model.ownership import OwnerPath
from pops.params.use_sites import ParamUse, resolve_param_use
from pops.time.points import Clock, StagePoint, TimePoint, point_clock


def _clock(value: Any, where: str) -> Clock:
    if type(value) is not Clock:
        raise TypeError("%s must be an exact Clock" % where)
    return value


def _point(value: Any, clock: Clock, where: str) -> TimePoint | StagePoint | None:
    if value is None:
        return None
    if type(value) not in (TimePoint, StagePoint):
        raise TypeError("%s must be an exact TimePoint or StagePoint" % where)
    if point_clock(value, where) != clock:
        raise ValueError("%s belongs to a different clock" % where)
    return value


@dataclass(frozen=True, slots=True)
class Domain:
    """Closed base class for schedule domains; instantiate one of its concrete subclasses."""

    clock: Clock
    at: TimePoint | StagePoint | None = None
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self) is Domain:
            raise TypeError("Domain is closed; use AcceptedStep/Attempt/Stage/... instead")
        clock = _clock(self.clock, "%s clock" % type(self).__name__)
        object.__setattr__(self, "at", _point(self.at, clock, "%s at" % type(self).__name__))


@dataclass(frozen=True, slots=True)
class AcceptedStep(Domain):
    """Cadence indexed only by committed, accepted macro steps."""


@dataclass(frozen=True, slots=True)
class Attempt(Domain):
    """Cadence indexed by step attempts, including rejected attempts."""


@dataclass(frozen=True, slots=True)
class Stage(Domain):
    """Cadence at one exact named stage point."""

    at: StagePoint

    def __post_init__(self) -> None:
        super(Stage, self).__post_init__()
        if type(self.at) is not StagePoint:
            raise TypeError("Stage at must be an exact StagePoint")


@dataclass(frozen=True, slots=True)
class ClockTick(Domain):
    """Cadence on a distinct logical clock tick (multirate runtime domain)."""


@dataclass(frozen=True, slots=True)
class AMRLevel(Domain):
    """Cadence for one AMR hierarchy level."""

    level: int = 0

    def __post_init__(self) -> None:
        super(AMRLevel, self).__post_init__()
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level < 0:
            raise ValueError("AMRLevel level must be a non-negative int")


@dataclass(frozen=True, slots=True)
class EventHandle:
    """Immutable owner-qualified identity of one runtime event channel."""

    owner: OwnerPath
    local_id: str
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner", OwnerPath.coerce(self.owner).canonical())
        if not isinstance(self.local_id, str) or not self.local_id:
            raise ValueError("EventHandle local_id must be a non-empty string")

    def to_data(self) -> dict[str, Any]:
        return {"schema_version": 1, "owner": self.owner.to_data(), "local_id": self.local_id}


@dataclass(frozen=True, slots=True)
class Event(Domain):
    """Cadence driven by an owner-qualified typed runtime event channel."""

    event: EventHandle | None = None

    def __post_init__(self) -> None:
        super(Event, self).__post_init__()
        if type(self.event) is not EventHandle:
            raise TypeError("Event event must be an exact EventHandle")


@dataclass(frozen=True, slots=True)
class WallOutput(Domain):
    """Host wall/output cadence, owned by the later ConsumerGraph runtime."""


@dataclass(frozen=True, slots=True)
class Trigger:
    domain: Domain
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self) is Trigger:
            raise TypeError("Trigger is closed; use Always/Every/AtStart/AtEnd/When")
        if type(self.domain) not in (
                AcceptedStep, Attempt, Stage, ClockTick, AMRLevel, Event, WallOutput):
            raise TypeError("%s domain must be an exact closed Domain" % type(self).__name__)


@dataclass(frozen=True, slots=True)
class Always(Trigger):
    pass


@dataclass(frozen=True, slots=True)
class Every(Trigger):
    n: int

    def __post_init__(self) -> None:
        super(Every, self).__post_init__()
        n = resolve_param_use(self.n, ParamUse.SCHEDULE, where="Every(n=)")
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError("Every n must be a positive int")
        object.__setattr__(self, "n", n)


@dataclass(frozen=True, slots=True)
class AtStart(Trigger):
    pass


@dataclass(frozen=True, slots=True)
class AtEnd(Trigger):
    pass


@dataclass(frozen=True, slots=True)
class When(Trigger):
    condition: Any


@dataclass(frozen=True, slots=True)
class OffPolicy:
    """Closed base class for the meaning of an off-cadence read."""

    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self) is OffPolicy:
            raise TypeError("OffPolicy is closed; use Hold/Skip/Zero/AccumulateDt/Error")


@dataclass(frozen=True, slots=True)
class Hold(OffPolicy):
    pass


@dataclass(frozen=True, slots=True)
class Skip(OffPolicy):
    pass


@dataclass(frozen=True, slots=True)
class Zero(OffPolicy):
    pass


@dataclass(frozen=True, slots=True)
class AccumulateDt(OffPolicy):
    pass


@dataclass(frozen=True, slots=True)
class Error(OffPolicy):
    pass


_DOMAIN_TAGS = {AcceptedStep: "accepted_step", Attempt: "attempt", Stage: "stage",
                ClockTick: "clock_tick", AMRLevel: "amr_level", Event: "event",
                WallOutput: "wall_output"}
_TRIGGER_TAGS = {Always: "always", Every: "every", AtStart: "at_start",
                 AtEnd: "at_end", When: "when"}
_OFF_TAGS = {Hold: "hold", Skip: "skip", Zero: "zero",
             AccumulateDt: "accumulate_dt", Error: "error"}


@dataclass(frozen=True, slots=True)
class Schedule:
    """One exact typed cadence. ``off`` may be omitted only while its result is never read."""

    trigger: Trigger
    off: OffPolicy | None = None
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.trigger) not in _TRIGGER_TAGS:
            raise TypeError("Schedule trigger must be an exact closed Trigger")
        if self.off is not None and type(self.off) not in _OFF_TAGS:
            raise TypeError("Schedule off must be an exact closed OffPolicy or None")
        if type(self.trigger) is Always and self.off is not None:
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
        if type(self.trigger) is Every:
            return {"n": self.trigger.n}
        if type(self.trigger) is When:
            return {"cond": self.trigger.condition}
        return {}

    def is_always(self) -> bool:
        return type(self.trigger) is Always

    def needs_cache(self) -> bool:
        return type(self.off) in (Hold, AccumulateDt)

    def validate_site(self, *, clock: Clock, point: Any = None, where: str = "schedule") -> None:
        if self.clock != clock:
            raise ValueError("%s clock %r does not match evaluation clock %r"
                             % (where, self.clock.name, clock.name))
        if self.domain.at is not None and point is not None and self.domain.at != point:
            raise ValueError("%s point does not match its typed domain point" % where)

    def map_values(self, mapper: Callable[[Any], Any]) -> Schedule:
        trigger = self.trigger
        if type(trigger) is When:
            trigger = When(trigger.domain, mapper(trigger.condition))
        return Schedule(trigger, off=self.off)

    def to_data(self) -> dict[str, Any]:
        domain = {"type": _DOMAIN_TAGS[type(self.domain)], "clock": self.clock.to_data(),
                  "at": self.domain.at.to_data() if self.domain.at is not None else None}
        if type(self.domain) is AMRLevel:
            domain["level"] = self.domain.level
        elif type(self.domain) is Event:
            domain["event"] = self.domain.event.to_data()
        trigger = {"type": _TRIGGER_TAGS[type(self.trigger)]}
        if type(self.trigger) is Every:
            trigger["n"] = self.trigger.n
        elif type(self.trigger) is When:
            trigger["condition"] = self.trigger.condition
        return {"schema_version": 1, "domain": domain, "trigger": trigger,
                "off": _OFF_TAGS[type(self.off)] if self.off is not None else None}

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


__all__ = ["Domain", "AcceptedStep", "Attempt", "Stage", "ClockTick", "AMRLevel",
           "EventHandle", "Event",
           "WallOutput", "Trigger", "Always", "Every", "AtStart", "AtEnd", "When",
           "OffPolicy", "Hold", "Skip", "Zero", "AccumulateDt", "Error", "Schedule",
           "always", "every", "when", "on_start", "on_end"]
