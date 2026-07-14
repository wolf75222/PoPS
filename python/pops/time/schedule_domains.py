"""Typed schedule domains and their native/consumer extension interfaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from typing import Any, ClassVar

from pops.model.ownership import OwnerPath
from pops.time.points import Clock, StagePoint, TimePoint, point_clock
from pops.time.schedule_lowering import ScheduleDomainIR, ScheduleTimeline
from pops.time.schedule_protocol import (
    component_payload,
    manifest_value,
    map_component,
)


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
    """Extension interface for the runtime coordinate owned by a schedule."""

    clock: Clock
    at: TimePoint | StagePoint | None = None
    manifest_tag: ClassVar[str | None] = None
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self) is Domain:
            raise TypeError("Domain is abstract; use or subclass a concrete Domain")
        component_payload(self, frozenset({"clock", "at"}))
        clock = _clock(self.clock, "%s clock" % type(self).__name__)
        object.__setattr__(self, "at", _point(self.at, clock, "%s at" % type(self).__name__))

    def native_schedule_domain(self, *, where: str) -> ScheduleDomainIR:
        raise NotImplementedError(
            "schedule domain %s at %s is typed and preserved, but this runtime only supports "
            "AcceptedStep; Attempt needs StepTransaction, Stage/ClockTick/AMRLevel need ADC-677, "
            "and Event/WallOutput need ConsumerGraph" % (type(self).__name__, where)
        )

    def schedule_payload(self) -> dict[str, Any]:
        return component_payload(self, frozenset({"clock", "at"}))

    def to_schedule_data(self) -> dict[str, Any]:
        tag = type(self).manifest_tag
        if not isinstance(tag, str) or not tag:
            raise TypeError(
                "schedule domain %s must declare a non-empty manifest_tag" % type(self).__name__
            )
        payload = self.schedule_payload()
        if type(payload) is not dict:
            raise TypeError("Domain.schedule_payload() must return an exact dict")
        if frozenset({"type", "clock", "at"}).intersection(payload):
            raise ValueError("Domain.schedule_payload() uses a reserved manifest key")
        return {
            "type": tag,
            "clock": self.clock.to_data(),
            "at": self.at.to_data() if self.at is not None else None,
            **{key: manifest_value(item) for key, item in payload.items()},
        }

    def consumer_coordinate(self, moment: Any) -> int | None:
        raise NotImplementedError(
            "schedule domain %s does not implement consumer_coordinate()" % type(self).__name__
        )

    def consumer_occurrence_evidence(self, moment: Any) -> dict[str, Any]:
        del moment
        return {}

    def map_values(
        self, mapper: Callable[[Any], Any], *, clock: Clock, at: TimePoint | StagePoint | None
    ) -> Domain:
        mapped = map_component(self, mapper, frozenset({"clock", "at"}), clock=clock, at=at)
        if type(mapped) is not type(self):
            raise TypeError("Domain.map_values() must preserve the exact extension type")
        return mapped


@dataclass(frozen=True, slots=True)
class AcceptedStep(Domain):
    """Cadence indexed only by committed, accepted macro steps."""

    manifest_tag: ClassVar[str | None] = "accepted_step"

    def native_schedule_domain(self, *, where: str) -> ScheduleDomainIR:
        del where
        return ScheduleDomainIR(ScheduleTimeline.ACCEPTED_STEP, self.clock.qualified_id)

    def consumer_coordinate(self, moment: Any) -> int | None:
        return moment.accepted_step


@dataclass(frozen=True, slots=True)
class Attempt(Domain):
    """Cadence indexed by step attempts, including rejected attempts."""

    manifest_tag: ClassVar[str | None] = "attempt"

    def consumer_coordinate(self, moment: Any) -> int | None:
        return moment.attempt


@dataclass(frozen=True, slots=True)
class Stage(Domain):
    """Cadence at one exact named stage point."""

    at: StagePoint
    manifest_tag: ClassVar[str | None] = "stage"

    def __post_init__(self) -> None:
        super(Stage, self).__post_init__()
        if type(self.at) is not StagePoint:
            raise TypeError("Stage at must be an exact StagePoint")

    def consumer_coordinate(self, moment: Any) -> int | None:
        return moment.accepted_step if moment.stage == self.at else None

    def native_schedule_domain(self, *, where: str) -> ScheduleDomainIR:
        del where
        return ScheduleDomainIR(
            ScheduleTimeline.STAGE,
            self.clock.qualified_id,
            stage_identity=json.dumps(
                self.at.to_data(), sort_keys=True, separators=(",", ":"), allow_nan=False),
        )

    def consumer_occurrence_evidence(self, moment: Any) -> dict[str, Any]:
        return {"stage": moment.stage.to_data() if moment.stage else None}


@dataclass(frozen=True, slots=True)
class ClockTick(Domain):
    """Cadence on a distinct logical clock tick (multirate runtime domain)."""

    manifest_tag: ClassVar[str | None] = "clock_tick"

    def consumer_coordinate(self, moment: Any) -> int | None:
        return moment.clock_tick

    def native_schedule_domain(self, *, where: str) -> ScheduleDomainIR:
        del where
        return ScheduleDomainIR(ScheduleTimeline.CLOCK_TICK, self.clock.qualified_id)


@dataclass(frozen=True, slots=True)
class AMRLevel(Domain):
    """Cadence for one AMR hierarchy level; subclasses may extend its typed payload."""

    level: int = 0
    manifest_tag: ClassVar[str | None] = "amr_level"

    def __post_init__(self) -> None:
        super(AMRLevel, self).__post_init__()
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level < 0:
            raise ValueError("AMRLevel level must be a non-negative int")

    def consumer_coordinate(self, moment: Any) -> int | None:
        return moment.accepted_step if moment.level == self.level else None

    def native_schedule_domain(self, *, where: str) -> ScheduleDomainIR:
        del where
        return ScheduleDomainIR(
            ScheduleTimeline.AMR_LEVEL, self.clock.qualified_id, level=self.level)

    def consumer_occurrence_evidence(self, moment: Any) -> dict[str, Any]:
        return {"level": moment.level}


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
    manifest_tag: ClassVar[str | None] = "event"

    def __post_init__(self) -> None:
        super(Event, self).__post_init__()
        if type(self.event) is not EventHandle:
            raise TypeError("Event event must be an exact EventHandle")

    def schedule_payload(self) -> dict[str, Any]:
        return {"event": self.event.to_data()}

    def consumer_coordinate(self, moment: Any) -> int | None:
        return moment.accepted_step if self.event in moment.events else None

    def consumer_occurrence_evidence(self, moment: Any) -> dict[str, Any]:
        del moment
        return {"event": self.event.to_data()}


@dataclass(frozen=True, slots=True)
class WallOutput(Domain):
    """Host wall/output cadence, owned by the ConsumerGraph runtime."""

    manifest_tag: ClassVar[str | None] = "wall_output"

    def consumer_coordinate(self, moment: Any) -> int | None:
        return moment.wall_tick


__all__ = [
    "Domain",
    "AcceptedStep",
    "Attempt",
    "Stage",
    "ClockTick",
    "AMRLevel",
    "EventHandle",
    "Event",
    "WallOutput",
]
