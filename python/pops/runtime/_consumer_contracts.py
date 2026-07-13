"""Immutable ConsumerGraph and accepted-side-effect planning values."""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from pops.identity import Identity, make_identity
from pops.model import Handle
from pops.time import EventHandle, Schedule, StagePoint, TimePoint

if TYPE_CHECKING:
    from pops.fields import FieldContext, FieldReadPolicy, LayoutBinding


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TypeError("%s must be non-empty canonical text" % where)
    return value


def _index(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError("%s must be an integer >= 0" % where)
    return value


def _exact_handle(value: Any, kind: str | None, where: str) -> Handle:
    if type(value) is not Handle or not value.is_resolved:
        raise TypeError("%s must be an exact canonical Handle" % where)
    if kind is not None and value.kind != kind:
        raise TypeError("%s Handle kind must be %r" % (where, kind))
    return value


class ConsumerKind(Enum):
    DIAGNOSTIC = "diagnostic"
    SCIENTIFIC_OUTPUT = "scientific_output"
    CHECKPOINT = "checkpoint"
    MONITOR = "monitor"


class ParallelMode(Enum):
    SERIAL = "serial"
    COLLECTIVE = "collective"
    PER_RANK = "per_rank"


class ConsumerFailureAction:
    """Closed failure decision applied to one consumer sample."""

    __pops_ir_immutable__ = True

    def to_data(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FailRun(ConsumerFailureAction):
    def to_data(self) -> dict[str, Any]:
        return {"action": "fail_run"}


@dataclass(frozen=True, slots=True)
class Retry(ConsumerFailureAction):
    max_attempts: int

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) \
                or self.max_attempts < 2:
            raise ValueError("Retry.max_attempts must be an integer >= 2")

    def to_data(self) -> dict[str, Any]:
        return {"action": "retry", "max_attempts": self.max_attempts}


@dataclass(frozen=True, slots=True)
class SkipSampleReported(ConsumerFailureAction):
    def to_data(self) -> dict[str, Any]:
        return {"action": "skip_sample_reported"}


_FAILURE_ACTIONS = (FailRun, Retry, SkipSampleReported)


@dataclass(frozen=True, slots=True)
class ConsumerQuantity:
    """One owner-qualified runtime resource selected by a consumer."""

    reference: Handle
    runtime_resource: str
    layout_id: str
    levels: tuple[int, ...] = ()
    field_context: FieldContext | None = None
    field_policy: FieldReadPolicy | None = None
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        _exact_handle(self.reference, None, "ConsumerQuantity.reference")
        _text(self.runtime_resource, "ConsumerQuantity.runtime_resource")
        _text(self.layout_id, "ConsumerQuantity.layout_id")
        if not isinstance(self.levels, tuple):
            raise TypeError("ConsumerQuantity.levels must be a tuple")
        levels = tuple(_index(value, "ConsumerQuantity.levels[]") for value in self.levels)
        if levels != tuple(sorted(set(levels))):
            raise ValueError("ConsumerQuantity.levels must be sorted and unique")
        if self.field_context is not None:
            from pops.fields import FieldContext

            if type(self.field_context) is not FieldContext:
                raise TypeError("ConsumerQuantity.field_context must be an exact FieldContext")
            if self.reference != self.field_context.operator:
                raise ValueError("ConsumerQuantity field reference and FieldContext disagree")
            if self.field_context.layout.layout.qualified_id != self.layout_id:
                raise ValueError("ConsumerQuantity layout and FieldContext layout disagree")
        if self.field_policy is not None:
            from pops.fields import FieldReadPolicy

            if not isinstance(self.field_policy, FieldReadPolicy):
                raise TypeError("ConsumerQuantity.field_policy must be a FieldReadPolicy")
            if self.field_context is None:
                raise ValueError("a field_policy requires an exact field_context")
        object.__setattr__(self, "identity", make_identity("consumer-quantity", self._payload()))

    def _payload(self) -> dict[str, Any]:
        return {
            "reference": self.reference.canonical_identity(),
            "runtime_resource": self.runtime_resource,
            "layout_id": self.layout_id,
            "levels": list(self.levels),
            "field_context": self.field_context.to_data() if self.field_context else None,
            "field_policy": self.field_policy.to_data() if self.field_policy else None,
        }

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.to_data()}


@dataclass(frozen=True, slots=True)
class ConsumerManifest:
    """Semantic declaration of one distinct ConsumerGraph node."""

    handle: Handle
    kind: ConsumerKind
    quantities: tuple[ConsumerQuantity, ...]
    schedule: Schedule
    target_uri: str
    output_format: str
    parallel_mode: ParallelMode
    dependencies: tuple[Handle, ...] = ()
    failure_action: ConsumerFailureAction = field(default_factory=FailRun)
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        _exact_handle(self.handle, "consumer", "ConsumerManifest.handle")
        if type(self.kind) is not ConsumerKind:
            raise TypeError("ConsumerManifest.kind must be an exact ConsumerKind")
        if not isinstance(self.quantities, tuple) or any(
                type(value) is not ConsumerQuantity for value in self.quantities):
            raise TypeError("ConsumerManifest.quantities must contain exact ConsumerQuantity values")
        quantities = tuple(sorted(self.quantities, key=lambda value: value.identity.token))
        if len({value.identity for value in quantities}) != len(quantities):
            raise ValueError("ConsumerManifest contains duplicate quantities")
        object.__setattr__(self, "quantities", quantities)
        if type(self.schedule) is not Schedule:
            raise TypeError("ConsumerManifest.schedule must be an exact Schedule")
        _text(self.target_uri, "ConsumerManifest.target_uri")
        _text(self.output_format, "ConsumerManifest.output_format")
        if type(self.parallel_mode) is not ParallelMode:
            raise TypeError("ConsumerManifest.parallel_mode must be an exact ParallelMode")
        if not isinstance(self.dependencies, tuple):
            raise TypeError("ConsumerManifest.dependencies must be a tuple")
        dependencies = tuple(sorted(
            (_exact_handle(value, "consumer", "ConsumerManifest.dependencies[]")
             for value in self.dependencies), key=lambda value: value.qualified_id))
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("ConsumerManifest contains duplicate dependencies")
        if self.handle in dependencies:
            raise ValueError("ConsumerManifest cannot depend on itself")
        object.__setattr__(self, "dependencies", dependencies)
        if type(self.failure_action) not in _FAILURE_ACTIONS:
            raise TypeError("ConsumerManifest.failure_action must be FailRun, Retry, or SkipSampleReported")
        object.__setattr__(self, "identity", make_identity("consumer-manifest", self._payload()))

    @property
    def qualified_id(self) -> str:
        return self.handle.qualified_id

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "handle": self.handle.canonical_identity(),
            "kind": self.kind.value,
            "quantities": [value.to_data() for value in self.quantities],
            "schedule": self.schedule.to_data(),
            "target_uri": self.target_uri,
            "output_format": self.output_format,
            "parallel_mode": self.parallel_mode.value,
            "dependencies": [value.canonical_identity() for value in self.dependencies],
            "failure_action": self.failure_action.to_data(),
        }

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.to_data()}


class ConsumerGraph:
    """Canonical DAG; ready-node ties are broken by qualified consumer identity."""

    __slots__ = ("nodes", "topology", "identity", "_by_id", "_sealed")

    def __init__(self, manifests: Iterable[ConsumerManifest]) -> None:
        supplied = tuple(manifests)
        if any(type(value) is not ConsumerManifest for value in supplied):
            raise TypeError("ConsumerGraph requires exact ConsumerManifest values")
        nodes = tuple(sorted(supplied, key=lambda value: value.qualified_id))
        ids = [value.qualified_id for value in nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("ConsumerGraph contains duplicate consumer handles")
        by_id = {value.qualified_id: value for value in nodes}
        indegree = {value.qualified_id: len(value.dependencies) for value in nodes}
        followers: dict[str, list[str]] = {value.qualified_id: [] for value in nodes}
        for value in nodes:
            for dependency in value.dependencies:
                if dependency.qualified_id not in by_id or by_id[dependency.qualified_id].handle != dependency:
                    raise ValueError("ConsumerGraph dependency %s is not an exact graph node" % dependency.qualified_id)
                followers[dependency.qualified_id].append(value.qualified_id)
        ready = [consumer_id for consumer_id, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        topology = []
        while ready:
            current = heapq.heappop(ready)
            topology.append(by_id[current])
            for follower in sorted(followers[current]):
                indegree[follower] -= 1
                if indegree[follower] == 0:
                    heapq.heappush(ready, follower)
        if len(topology) != len(nodes):
            cycle = sorted(consumer_id for consumer_id, degree in indegree.items() if degree)
            raise ValueError("ConsumerGraph contains a dependency cycle: %s" % cycle)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "topology", tuple(topology))
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "identity", make_identity("consumer-graph", self._payload()))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("ConsumerGraph is immutable")
        object.__setattr__(self, name, value)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "nodes": [value.to_data() for value in self.nodes],
            "topology": [value.qualified_id for value in self.topology],
        }

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.to_data()}


@dataclass(frozen=True, slots=True)
class ConsumerMoment:
    """Exact runtime evidence used to evaluate every typed schedule domain."""

    point: TimePoint
    accepted_step: int
    attempt: int
    clock_tick: int = 0
    wall_tick: int = 0
    stage: StagePoint | None = None
    level: int | None = None
    events: tuple[EventHandle, ...] = ()
    layouts: tuple[LayoutBinding, ...] = ()
    at_start: bool = False
    at_end: bool = False

    def __post_init__(self) -> None:
        from pops.fields import LayoutBinding

        if type(self.point) is not TimePoint:
            raise TypeError("ConsumerMoment.point must be an exact TimePoint")
        for name in ("accepted_step", "attempt", "clock_tick", "wall_tick"):
            _index(getattr(self, name), "ConsumerMoment.%s" % name)
        if self.stage is not None and type(self.stage) is not StagePoint:
            raise TypeError("ConsumerMoment.stage must be an exact StagePoint or None")
        if self.level is not None:
            _index(self.level, "ConsumerMoment.level")
        if not isinstance(self.events, tuple) or any(type(value) is not EventHandle for value in self.events):
            raise TypeError("ConsumerMoment.events must contain exact EventHandle values")
        events = tuple(sorted(self.events, key=lambda value: (str(value.owner), value.local_id)))
        if len(set(events)) != len(events):
            raise ValueError("ConsumerMoment.events must be unique")
        object.__setattr__(self, "events", events)
        if not isinstance(self.layouts, tuple) or any(type(value) is not LayoutBinding for value in self.layouts):
            raise TypeError("ConsumerMoment.layouts must contain exact LayoutBinding values")
        layouts = tuple(sorted(self.layouts, key=lambda value: value.layout.qualified_id))
        if len({value.layout.qualified_id for value in layouts}) != len(layouts):
            raise ValueError("ConsumerMoment.layouts contains duplicate layouts")
        object.__setattr__(self, "layouts", layouts)
        if type(self.at_start) is not bool or type(self.at_end) is not bool:
            raise TypeError("ConsumerMoment at_start/at_end must be bool")

    def layout_for(self, layout_id: str) -> LayoutBinding:
        matches = [value for value in self.layouts if value.layout.qualified_id == layout_id]
        if not matches:
            raise KeyError(layout_id)
        return matches[0]

    def to_data(self) -> dict[str, Any]:
        return {
            "point": self.point.to_data(),
            "accepted_step": self.accepted_step,
            "attempt": self.attempt,
            "clock_tick": self.clock_tick,
            "wall_tick": self.wall_tick,
            "stage": self.stage.to_data() if self.stage else None,
            "level": self.level,
            "events": [value.to_data() for value in self.events],
            "layouts": [value.to_data() for value in self.layouts],
            "at_start": self.at_start,
            "at_end": self.at_end,
        }


@dataclass(frozen=True, slots=True)
class ScheduleCursor:
    consumer_id: str
    last_occurrence: str | None = None
    committed_samples: int = 0

    def __post_init__(self) -> None:
        _text(self.consumer_id, "ScheduleCursor.consumer_id")
        if self.last_occurrence is not None:
            _text(self.last_occurrence, "ScheduleCursor.last_occurrence")
        _index(self.committed_samples, "ScheduleCursor.committed_samples")

    def to_data(self) -> dict[str, Any]:
        return {
            "consumer_id": self.consumer_id,
            "last_occurrence": self.last_occurrence,
            "committed_samples": self.committed_samples,
        }


class ConsumerCursorSet:
    __slots__ = ("rows", "_by_id")

    def __init__(self, rows: Iterable[ScheduleCursor] = ()) -> None:
        supplied = tuple(rows)
        if any(type(value) is not ScheduleCursor for value in supplied):
            raise TypeError("ConsumerCursorSet requires exact ScheduleCursor values")
        values = tuple(sorted(supplied, key=lambda value: value.consumer_id))
        if len({value.consumer_id for value in values}) != len(values):
            raise ValueError("ConsumerCursorSet contains duplicate consumer ids")
        object.__setattr__(self, "rows", values)
        object.__setattr__(self, "_by_id", MappingProxyType({value.consumer_id: value for value in values}))

    def for_consumer(self, consumer_id: str) -> ScheduleCursor:
        _text(consumer_id, "consumer_id")
        return self._by_id.get(consumer_id, ScheduleCursor(consumer_id))

    def replace(self, cursor: ScheduleCursor) -> ConsumerCursorSet:
        if type(cursor) is not ScheduleCursor:
            raise TypeError("ConsumerCursorSet.replace requires an exact ScheduleCursor")
        values = {value.consumer_id: value for value in self.rows}
        values[cursor.consumer_id] = cursor
        return ConsumerCursorSet(tuple(values.values()))

    def to_data(self) -> dict[str, Any]:
        return {"schema_version": 1, "rows": [value.to_data() for value in self.rows]}


__all__ = [
    "ConsumerCursorSet", "ConsumerFailureAction", "ConsumerGraph", "ConsumerKind",
    "ConsumerManifest", "ConsumerMoment", "ConsumerQuantity", "FailRun", "ParallelMode",
    "Retry", "ScheduleCursor", "SkipSampleReported",
]
