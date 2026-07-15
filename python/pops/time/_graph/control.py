"""Immutable structured-control records for the canonical temporal graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, cast

from pops.time._graph.base import (
    CanonicalData,
    Node as _Node,
    ValueRef,
    node_data as _node_data,
    node_id as _node_id,
    nonempty as _nonempty,
    point as _point,
    point_clocks as _point_clocks,
)
from pops.time._graph.nodes import NODE_TYPES
from pops.time._graph.validation import validate_nodes
from pops.time.points import Clock, StagePoint, TimePoint


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
    signature: CanonicalData | None

    def __init__(self, value: ValueRef, clock: Clock, point: TimePoint | StagePoint,
                 *, signature: Any = None) -> None:
        if type(value) is not ValueRef:
            raise TypeError("RegionCapture value must be an exact ValueRef")
        if type(clock) is not Clock:
            raise TypeError("RegionCapture clock must be an exact Clock")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "point", _point(point))
        object.__setattr__(
            self, "signature",
            None if signature is None else CanonicalData(
                signature, where="RegionCapture.signature"))
        if _point_clocks(self.point) != frozenset((clock,)):
            raise ValueError("RegionCapture point must use exactly the capture clock")

    def to_data(self) -> dict[str, Any]:
        data = {
            "value": self.value.to_data(),
            "clock": self.clock.to_data(),
            "point": self.point.to_data(),
        }
        if self.signature is not None:
            data["signature"] = self.signature.to_data()
        return data


@dataclass(frozen=True, slots=True, init=False)
class Region:
    """Immutable structured subgraph with explicit outer captures and one readable result."""

    name: str
    captures: tuple[RegionCapture, ...]
    nodes: tuple[Any, ...]
    result: ValueRef
    clocks: tuple[Clock, ...]
    result_signature: CanonicalData | None

    def __init__(self, name: str, captures: Any, nodes: Any, result: ValueRef,
                 *, clocks: Any = None, result_signature: Any = None) -> None:
        object.__setattr__(self, "name", _nonempty(name, where="Region name"))
        frozen_captures = tuple(captures)
        if any(type(capture) is not RegionCapture for capture in frozen_captures):
            raise TypeError("Region captures must contain exact RegionCapture values")
        frozen_nodes = tuple(nodes)
        if any(type(node) not in (*NODE_TYPES, Branch, Loop) for node in frozen_nodes):
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
        object.__setattr__(
            self, "result_signature",
            None if result_signature is None else CanonicalData(
                result_signature, where="Region.result_signature"))
        self._validate()

    def _validate(self) -> None:
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
        actual_signature = getattr(source, "signature", None)
        if self.result_signature is not None and actual_signature is not None \
                and actual_signature != self.result_signature:
            raise ValueError("Region result does not match its declared result signature")

    def result_source(self) -> Any:
        for source in (*self.captures, *self.nodes):
            source_id = (
                cast(RegionCapture, source).value.node_id
                if type(source) is RegionCapture
                else cast(_Node, source).node_id
            )
            if source_id == self.result.node_id:
                return source
        raise RuntimeError("validated Region lost its result source")

    def to_data(self) -> dict[str, Any]:
        data = {
            "kind": "pops.program-region",
            "name": self.name,
            "clocks": [clock.to_data() for clock in self.clocks],
            "captures": [capture.to_data() for capture in self.captures],
            "nodes": [node.to_data() for node in self.nodes],
            "result": self.result.to_data(),
        }
        if self.result_signature is not None:
            data["result_signature"] = self.result_signature.to_data()
        return data


@dataclass(frozen=True, slots=True, init=False)
class Branch(_Node):
    kind: ClassVar[str] = "branch"
    node_id: int
    condition: ValueRef
    when_true: Any
    when_false: Any
    clock: Clock
    point: TimePoint | StagePoint
    name: str
    result_signature: CanonicalData

    def __init__(self, node_id: int, condition: ValueRef, when_true: Any,
                 when_false: Any, clock: Clock, point: TimePoint | StagePoint,
                 *, name: str = "branch", result_signature: Any) -> None:
        if type(condition) is not ValueRef:
            raise TypeError("Branch condition must be an exact ValueRef")
        if type(when_true) is not Region or type(when_false) is not Region:
            raise TypeError("Branch arms must be exact immutable Region values")
        signature = CanonicalData(result_signature, where="Branch.result_signature")
        if when_true.result_signature != signature or when_false.result_signature != signature:
            raise ValueError("Branch arm result signatures must exactly match the branch signature")
        for arm in (when_true, when_false):
            result_source = arm.result_source()
            if result_source.clock != clock or result_source.point != point:
                raise ValueError(
                    "Branch arm results must share the branch clock and exact point")
        object.__setattr__(self, "node_id", _node_id(node_id))
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "when_true", when_true)
        object.__setattr__(self, "when_false", when_false)
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "point", _point(point))
        object.__setattr__(self, "name", _nonempty(name, where="Branch name"))
        object.__setattr__(self, "result_signature", signature)

    def references(self) -> tuple[ValueRef, ...]:
        direct = (self.condition,)
        captures = _capture_refs(self.when_true, self.when_false)
        return direct + tuple(ref for ref in captures if ref not in direct)

    def to_data(self) -> dict[str, Any]:
        return _node_data(
            self, condition=self.condition.to_data(),
            when_true=self.when_true.to_data(), when_false=self.when_false.to_data(),
            name=self.name, result_signature=self.result_signature.to_data())


@dataclass(frozen=True, slots=True, init=False)
class Loop(_Node):
    """Structured while/range or fixed-ratio child-clock subcycle."""

    kind: ClassVar[str] = "loop"
    node_id: int
    loop_kind: str
    initial: ValueRef
    body: Region
    condition: Region | None
    count: int | None
    parent_clock: Clock | None
    clock: Clock
    point: TimePoint | StagePoint
    name: str

    def __init__(self, node_id: int, loop_kind: str, initial: ValueRef, body: Region,
                 clock: Clock, point: TimePoint | StagePoint, *, condition: Region | None = None,
                 count: int | None = None, parent_clock: Clock | None = None,
                 name: str = "loop") -> None:
        if loop_kind not in {"while", "range", "subcycle"}:
            raise ValueError("Loop loop_kind must be 'while', 'range', or 'subcycle'")
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
        if loop_kind == "subcycle" and (
                condition is not None or isinstance(count, bool) or not isinstance(count, int)
                or count <= 0 or type(parent_clock) is not Clock or parent_clock == clock):
            raise ValueError(
                "subcycle Loop requires a positive count and one distinct exact parent_clock")
        if loop_kind != "subcycle" and parent_clock is not None:
            raise ValueError("only a subcycle Loop may declare parent_clock")
        object.__setattr__(self, "node_id", _node_id(node_id))
        object.__setattr__(self, "loop_kind", loop_kind)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "parent_clock", parent_clock)
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "point", _point(point))
        object.__setattr__(self, "name", _nonempty(name, where="Loop name"))

    def references(self) -> tuple[ValueRef, ...]:
        captures = _capture_refs(self.condition, self.body)
        return (self.initial,) + tuple(ref for ref in captures if ref != self.initial)

    def to_data(self) -> dict[str, Any]:
        data = _node_data(
            self, loop_kind=self.loop_kind, initial=self.initial.to_data(),
            body=self.body.to_data(),
            condition=self.condition.to_data() if self.condition is not None else None,
            count=self.count, name=self.name)
        if self.parent_clock is not None:
            data["parent_clock"] = self.parent_clock.to_data()
        return data


__all__ = ["Branch", "Loop", "Region", "RegionCapture"]
