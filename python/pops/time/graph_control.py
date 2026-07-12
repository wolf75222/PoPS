"""Immutable structured-control records owned by :mod:`pops.time.graph`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from pops.time.graph import (
    ValueRef, _Node, _nonempty, _node_id, _point, _point_clocks,
)
from pops.time.points import Clock, StagePoint, TimePoint


def _node_data(node: Any, **specific: Any) -> dict[str, Any]:
    return {
        "kind": node.kind,
        "node_id": node.node_id,
        "clock": node.clock.to_data(),
        "point": node.point.to_data(),
        **specific,
    }


def _capture_refs(*regions: Any) -> tuple[ValueRef, ...]:
    seen = set()
    refs = []
    for region in regions:
        if type(region) is not Region:
            continue
        for capture in region.captures:
            if capture.value.node_id not in seen:
                seen.add(capture.value.node_id)
                refs.append(capture.value)
    return tuple(refs)


@dataclass(frozen=True, slots=True, init=False)
class RegionCapture:
    """One immutable outer-SSA import made explicit at a structured-region boundary."""

    value: ValueRef
    clock: Clock
    point: TimePoint | StagePoint

    def __init__(self, value: ValueRef, clock: Clock, point: TimePoint | StagePoint) -> None:
        if type(value) is not ValueRef:
            raise TypeError("RegionCapture value must be an exact ValueRef")
        if type(clock) is not Clock:
            raise TypeError("RegionCapture clock must be an exact Clock")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "point", _point(point))
        if _point_clocks(self.point) != frozenset((clock,)):
            raise ValueError("RegionCapture point must use exactly the capture clock")

    def to_data(self) -> dict[str, Any]:
        return {
            "value": self.value.to_data(),
            "clock": self.clock.to_data(),
            "point": self.point.to_data(),
        }


@dataclass(frozen=True, slots=True, init=False)
class Region:
    """Immutable structured subgraph with explicit outer captures and one readable result."""

    name: str
    captures: tuple[RegionCapture, ...]
    nodes: tuple[Any, ...]
    result: ValueRef
    clocks: tuple[Clock, ...]

    def __init__(self, name: str, captures: Any, nodes: Any, result: ValueRef,
                 *, clocks: Any = None) -> None:
        from pops.time.graph import _NODE_TYPES

        object.__setattr__(self, "name", _nonempty(name, where="Region name"))
        frozen_captures = tuple(captures)
        if any(type(capture) is not RegionCapture for capture in frozen_captures):
            raise TypeError("Region captures must contain exact RegionCapture values")
        frozen_nodes = tuple(nodes)
        if any(type(node) not in _NODE_TYPES for node in frozen_nodes):
            raise TypeError("Region nodes must be exact graph node records")
        if type(result) is not ValueRef:
            raise TypeError("Region result must be an exact ValueRef")
        declared = tuple(clocks) if clocks is not None else tuple(dict.fromkeys(
            [capture.clock for capture in frozen_captures]
            + [node.clock for node in frozen_nodes]))
        if any(type(clock) is not Clock for clock in declared):
            raise TypeError("Region clocks must contain exact Clock values")
        if len(set(declared)) != len(declared):
            raise ValueError("Region clocks must be unique")
        object.__setattr__(self, "captures", frozen_captures)
        object.__setattr__(self, "nodes", frozen_nodes)
        object.__setattr__(self, "result", result)
        object.__setattr__(self, "clocks", declared)
        self._validate()

    def _validate(self) -> None:
        from pops.time.graph_validation import validate_nodes

        available: dict[int, Any] = {}
        for capture in self.captures:
            if capture.value.node_id in available:
                raise ValueError("Region capture ids must be unique")
            available[capture.value.node_id] = capture
        validate_nodes(self.nodes, self.clocks, available, where="Region %r" % self.name)
        source = available.get(self.result.node_id)
        if source is None:
            raise ValueError("Region result must name a capture or readable region node")
        if getattr(source, "readable", True) is False:
            raise ValueError("Region result cannot read a Commit node")

    def to_data(self) -> dict[str, Any]:
        return {
            "kind": "pops.program-region",
            "name": self.name,
            "clocks": [clock.to_data() for clock in self.clocks],
            "captures": [capture.to_data() for capture in self.captures],
            "nodes": [node.to_data() for node in self.nodes],
            "result": self.result.to_data(),
        }


@dataclass(frozen=True, slots=True, init=False)
class Branch(_Node):
    kind: ClassVar[str] = "branch"
    node_id: int
    condition: ValueRef
    state: ValueRef
    when_true: Any
    when_false: Any
    clock: Clock
    point: TimePoint | StagePoint
    name: str

    def __init__(self, node_id: int, condition: ValueRef, when_true: Any,
                 when_false: Any, clock: Clock, point: TimePoint | StagePoint,
                 *, state: ValueRef | None = None, name: str = "branch") -> None:
        from pops.time.graph import ProgramGraph

        if type(condition) is not ValueRef:
            raise TypeError("Branch condition must be an exact ValueRef")
        if state is None:
            state = condition
        if type(state) is not ValueRef:
            raise TypeError("Branch state must be an exact ValueRef")
        if type(when_true) not in (Region, ProgramGraph) \
                or type(when_false) not in (Region, ProgramGraph):
            raise TypeError("Branch arms must be exact immutable Region or ProgramGraph values")
        object.__setattr__(self, "node_id", _node_id(node_id))
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "when_true", when_true)
        object.__setattr__(self, "when_false", when_false)
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "point", _point(point))
        object.__setattr__(self, "name", _nonempty(name, where="Branch name"))

    def references(self) -> tuple[ValueRef, ...]:
        direct = (self.state, self.condition)
        captures = _capture_refs(self.when_true, self.when_false)
        return direct + tuple(ref for ref in captures if ref not in direct)

    def to_data(self) -> dict[str, Any]:
        return _node_data(
            self, state=self.state.to_data(), condition=self.condition.to_data(),
            when_true=self.when_true.to_data(), when_false=self.when_false.to_data(),
            name=self.name)


@dataclass(frozen=True, slots=True, init=False)
class Loop(_Node):
    """Structured ``while`` or fixed-count ``range`` with immutable recursive regions."""

    kind: ClassVar[str] = "loop"
    node_id: int
    loop_kind: str
    initial: ValueRef
    body: Region
    condition: Region | None
    count: int | None
    clock: Clock
    point: TimePoint | StagePoint
    name: str

    def __init__(self, node_id: int, loop_kind: str, initial: ValueRef, body: Region,
                 clock: Clock, point: TimePoint | StagePoint, *, condition: Region | None = None,
                 count: int | None = None, name: str = "loop") -> None:
        if loop_kind not in {"while", "range"}:
            raise ValueError("Loop loop_kind must be 'while' or 'range'")
        if type(initial) is not ValueRef:
            raise TypeError("Loop initial must be an exact ValueRef")
        if type(body) is not Region:
            raise TypeError("Loop body must be an exact immutable Region")
        if condition is not None and type(condition) is not Region:
            raise TypeError("Loop condition must be an exact immutable Region or None")
        if loop_kind == "while" and (condition is None or count is not None):
            raise ValueError("while Loop requires condition and forbids count")
        if loop_kind == "range" and (
                condition is not None or isinstance(count, bool) or not isinstance(count, int)
                or count < 0):
            raise ValueError("range Loop requires a non-negative Python int count and no condition")
        object.__setattr__(self, "node_id", _node_id(node_id))
        object.__setattr__(self, "loop_kind", loop_kind)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "point", _point(point))
        object.__setattr__(self, "name", _nonempty(name, where="Loop name"))

    def references(self) -> tuple[ValueRef, ...]:
        captures = _capture_refs(self.condition, self.body)
        return (self.initial,) + tuple(ref for ref in captures if ref != self.initial)

    def to_data(self) -> dict[str, Any]:
        return _node_data(
            self, loop_kind=self.loop_kind, initial=self.initial.to_data(),
            body=self.body.to_data(),
            condition=self.condition.to_data() if self.condition is not None else None,
            count=self.count, name=self.name)


__all__ = ["Branch", "Loop", "Region", "RegionCapture"]
