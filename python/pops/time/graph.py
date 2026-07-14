"""Immutable, canonical SSA ProgramGraph records detached from authoring objects."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from typing import Any, ClassVar

from pops._ir.literals import ScalarLiteral, scalar_data
from pops.model.ownership import OwnerPath
from pops.time.points import Clock, StagePoint, TimePoint


def _nonempty(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("%s must be a non-empty string" % where)
    return value


def _node_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("ProgramGraph node_id must be a non-negative Python int")
    return value


def _strict_data(value: Any, *, where: str) -> Any:
    if isinstance(value, CanonicalData):
        return value.to_data()
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float, Decimal, Fraction, ScalarLiteral)):
        return {"scalar": scalar_data(value)}
    if isinstance(value, Enum):
        return {
            "enum": "%s.%s.%s"
            % (type(value).__module__, type(value).__qualname__, value.name)
        }
    if isinstance(value, OwnerPath):
        return {"owner_path": value.canonical().to_data()}
    if isinstance(value, (Clock, TimePoint, StagePoint)):
        return value.to_data()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("%s mapping keys must be non-empty strings" % where)
        return {
            key: _strict_data(item, where="%s.%s" % (where, key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _strict_data(item, where="%s[%d]" % (where, index))
            for index, item in enumerate(value)
        ]
    canonical = getattr(value, "canonical_identity", None)
    if callable(canonical):
        return _strict_data(canonical(), where=where)
    to_data = getattr(value, "to_data", None)
    if callable(to_data) and getattr(value, "__pops_ir_immutable__", False) is True:
        return _strict_data(to_data(), where=where)
    raise TypeError(
        "%s contains mutable/opaque %s; provide canonical immutable data"
        % (where, type(value).__name__))


@dataclass(frozen=True, slots=True, init=False)
class CanonicalData:
    """Hashable canonical-data snapshot used for semantic node metadata."""

    _json: str

    def __init__(self, value: Any, *, where: str = "ProgramGraph metadata") -> None:
        data = _strict_data(value, where=where)
        object.__setattr__(
            self, "_json", json.dumps(data, sort_keys=True, separators=(",", ":")))

    def to_data(self) -> Any:
        return json.loads(self._json)


@dataclass(frozen=True, slots=True)
class ValueRef:
    """Readable SSA edge. Commit nodes deliberately cannot produce one."""

    node_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _node_id(self.node_id))

    def to_data(self) -> dict[str, int]:
        return {"node_id": self.node_id}


def _refs(values: Any, *, where: str) -> tuple[ValueRef, ...]:
    result = tuple(values)
    if any(type(value) is not ValueRef for value in result):
        raise TypeError("%s must contain exact ValueRef values" % where)
    return result


def _point(value: Any) -> TimePoint | StagePoint:
    if type(value) not in (TimePoint, StagePoint):
        raise TypeError("ProgramGraph node point must be an exact TimePoint or StagePoint")
    return value


def _point_clocks(point: TimePoint | StagePoint) -> frozenset[Clock]:
    if type(point) is TimePoint:
        return frozenset((point.clock,))
    return frozenset(item.clock for item in point.partitions.values())


class _Node:
    kind: ClassVar[str]
    readable: ClassVar[bool] = True

    def references(self) -> tuple[ValueRef, ...]:
        return ()


@dataclass(frozen=True, slots=True, init=False)
class StateRead(_Node):
    kind: ClassVar[str] = "state_read"
    node_id: int
    state: CanonicalData
    clock: Clock
    point: TimePoint | StagePoint
    name: str
    metadata: CanonicalData

    def __init__(self, node_id: int, state: Any, clock: Clock,
                 point: TimePoint | StagePoint, *, name: str = "state",
                 metadata: Any = None) -> None:
        object.__setattr__(self, "node_id", _node_id(node_id))
        object.__setattr__(self, "state", CanonicalData(state, where="StateRead.state"))
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "point", _point(point))
        object.__setattr__(self, "name", _nonempty(name, where="StateRead name"))
        object.__setattr__(
            self,
            "metadata",
            CanonicalData({} if metadata is None else metadata, where="StateRead.metadata"),
        )

    def to_data(self) -> dict[str, Any]:
        return _node_data(
            self,
            state=self.state.to_data(),
            name=self.name,
            metadata=self.metadata.to_data(),
        )


@dataclass(frozen=True, slots=True, init=False)
class ProgramValue(_Node):
    """One named SSA value definition in the immutable graph."""

    kind: ClassVar[str] = "program_value"
    node_id: int
    name: str
    value_type: str
    op: str
    inputs: tuple[ValueRef, ...]
    attrs: CanonicalData
    clock: Clock
    point: TimePoint | StagePoint

    def __init__(self, node_id: int, name: str, value_type: str, op: str,
                 inputs: Any, clock: Clock, point: TimePoint | StagePoint,
                 *, attrs: Any = None) -> None:
        object.__setattr__(self, "node_id", _node_id(node_id))
        object.__setattr__(self, "name", _nonempty(name, where="ProgramValue name"))
        object.__setattr__(
            self, "value_type", _nonempty(value_type, where="ProgramValue value_type"))
        object.__setattr__(self, "op", _nonempty(op, where="ProgramValue op"))
        object.__setattr__(self, "inputs", _refs(inputs, where="ProgramValue inputs"))
        object.__setattr__(
            self, "attrs", CanonicalData({} if attrs is None else attrs, where="ProgramValue.attrs"))
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "point", _point(point))

    def references(self) -> tuple[ValueRef, ...]:
        return self.inputs

    def to_data(self) -> dict[str, Any]:
        return _node_data(
            self, name=self.name, value_type=self.value_type, op=self.op,
            inputs=[value.to_data() for value in self.inputs], attrs=self.attrs.to_data())


@dataclass(frozen=True, slots=True, init=False)
class Unknown(_Node):
    kind: ClassVar[str] = "unknown"
    node_id: int
    name: str
    space: CanonicalData
    clock: Clock
    point: TimePoint | StagePoint

    def __init__(self, node_id: int, name: str, space: Any, clock: Clock,
                 point: TimePoint | StagePoint) -> None:
        object.__setattr__(self, "node_id", _node_id(node_id))
        object.__setattr__(self, "name", _nonempty(name, where="Unknown name"))
        object.__setattr__(self, "space", CanonicalData(space, where="Unknown.space"))
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "point", _point(point))

    def to_data(self) -> dict[str, Any]:
        return _node_data(self, name=self.name, space=self.space.to_data())


@dataclass(frozen=True, slots=True, init=False)
class OperatorCall(_Node):
    kind: ClassVar[str] = "operator_call"
    node_id: int
    operator: CanonicalData
    inputs: tuple[ValueRef, ...]
    clock: Clock
    point: TimePoint | StagePoint
    name: str

    def __init__(self, node_id: int, operator: Any, inputs: Any, clock: Clock,
                 point: TimePoint | StagePoint, *, name: str = "operator_call") -> None:
        object.__setattr__(self, "node_id", _node_id(node_id))
        object.__setattr__(
            self, "operator", CanonicalData(operator, where="OperatorCall.operator"))
        object.__setattr__(self, "inputs", _refs(inputs, where="OperatorCall inputs"))
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "point", _point(point))
        object.__setattr__(self, "name", _nonempty(name, where="OperatorCall name"))

    def references(self) -> tuple[ValueRef, ...]:
        return self.inputs

    def to_data(self) -> dict[str, Any]:
        return _node_data(
            self, operator=self.operator.to_data(),
            inputs=[value.to_data() for value in self.inputs], name=self.name)


@dataclass(frozen=True, slots=True, init=False)
class ResidualEvaluation(_Node):
    """One pure evaluation of a residual operator on an ordered unknown product."""
    kind: ClassVar[str] = "residual_evaluation"
    node_id: int
    operator: CanonicalData
    unknowns: tuple[ValueRef, ...]
    clock: Clock
    point: TimePoint | StagePoint
    name: str
    attrs: CanonicalData

    def __init__(self, node_id: int, operator: Any, unknowns: Any, clock: Clock,
                 point: TimePoint | StagePoint, *, name: str = "residual",
                 attrs: Any = None) -> None:
        refs = _refs(unknowns, where="ResidualEvaluation unknowns")
        if not refs:
            raise ValueError("ResidualEvaluation requires a non-empty unknown product")
        object.__setattr__(self, "node_id", _node_id(node_id))
        object.__setattr__(
            self, "operator", CanonicalData(operator, where="ResidualEvaluation.operator"))
        object.__setattr__(self, "unknowns", refs)
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "point", _point(point))
        object.__setattr__(self, "name", _nonempty(name, where="ResidualEvaluation name"))
        object.__setattr__(self, "attrs", CanonicalData(
            {} if attrs is None else attrs, where="ResidualEvaluation.attrs"))

    def references(self) -> tuple[ValueRef, ...]:
        return self.unknowns

    def to_data(self) -> dict[str, Any]:
        return _node_data(
            self, operator=self.operator.to_data(),
            unknowns=[value.to_data() for value in self.unknowns],
            name=self.name, attrs=self.attrs.to_data())


@dataclass(frozen=True, slots=True, init=False)
class ResidualSolve(_Node):
    """A solve whose residual and initial unknown product are explicit graph references."""
    kind: ClassVar[str] = "residual_solve"
    node_id: int
    residual: ValueRef
    initial: tuple[ValueRef, ...]
    clock: Clock
    point: TimePoint | StagePoint
    name: str
    attrs: CanonicalData

    def __init__(self, node_id: int, residual: ValueRef, initial: Any, clock: Clock,
                 point: TimePoint | StagePoint, *, name: str = "solve_residual",
                 attrs: Any = None) -> None:
        if type(residual) is not ValueRef:
            raise TypeError("ResidualSolve residual must be an exact ValueRef")
        initial_refs = _refs(initial, where="ResidualSolve initial")
        if not initial_refs:
            raise ValueError("ResidualSolve requires a non-empty initial unknown product")
        object.__setattr__(self, "node_id", _node_id(node_id))
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "initial", initial_refs)
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "point", _point(point))
        object.__setattr__(self, "name", _nonempty(name, where="ResidualSolve name"))
        object.__setattr__(self, "attrs", CanonicalData(
            {} if attrs is None else attrs, where="ResidualSolve.attrs"))

    def references(self) -> tuple[ValueRef, ...]:
        return (self.residual, *self.initial)

    def to_data(self) -> dict[str, Any]:
        return _node_data(
            self, residual=self.residual.to_data(),
            initial=[value.to_data() for value in self.initial],
            name=self.name, attrs=self.attrs.to_data())


@dataclass(frozen=True, slots=True, init=False)
class Solve(_Node):
    kind: ClassVar[str] = "solve"
    node_id: int
    unknown: ValueRef
    operator: ValueRef
    rhs: ValueRef
    clock: Clock
    point: TimePoint | StagePoint
    name: str
    attrs: CanonicalData

    def __init__(self, node_id: int, unknown: ValueRef, operator: ValueRef, rhs: ValueRef,
                 clock: Clock, point: TimePoint | StagePoint, *, name: str = "solve",
                 attrs: Any = None) -> None:
        object.__setattr__(self, "node_id", _node_id(node_id))
        for label, value in (("unknown", unknown), ("operator", operator), ("rhs", rhs)):
            if type(value) is not ValueRef:
                raise TypeError("Solve %s must be an exact ValueRef" % label)
            object.__setattr__(self, label, value)
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "point", _point(point))
        object.__setattr__(self, "name", _nonempty(name, where="Solve name"))
        object.__setattr__(
            self, "attrs", CanonicalData({} if attrs is None else attrs, where="Solve.attrs"))

    def references(self) -> tuple[ValueRef, ...]:
        return (self.unknown, self.operator, self.rhs)

    def to_data(self) -> dict[str, Any]:
        return _node_data(
            self, unknown=self.unknown.to_data(), operator=self.operator.to_data(),
            rhs=self.rhs.to_data(), name=self.name, attrs=self.attrs.to_data())

from pops.time.graph_control import Branch, Loop, Region, RegionCapture  # noqa: E402

@dataclass(frozen=True, slots=True, init=False)
class Synchronize(_Node):
    kind: ClassVar[str] = "synchronize"
    node_id: int
    value: ValueRef
    source_clock: Clock
    target_clock: Clock
    relation: CanonicalData
    point: TimePoint | StagePoint
    name: str

    @property
    def clock(self) -> Clock:
        return self.target_clock

    def __init__(self, node_id: int, value: ValueRef, source_clock: Clock,
                 target_clock: Clock, relation: Any, point: TimePoint | StagePoint,
                 *, name: str = "synchronize") -> None:
        if type(value) is not ValueRef:
            raise TypeError("Synchronize value must be an exact ValueRef")
        if source_clock == target_clock:
            raise ValueError("Synchronize requires two distinct clocks")
        object.__setattr__(self, "node_id", _node_id(node_id))
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "source_clock", source_clock)
        object.__setattr__(self, "target_clock", target_clock)
        object.__setattr__(
            self, "relation", CanonicalData(relation, where="Synchronize.relation"))
        object.__setattr__(self, "point", _point(point))
        object.__setattr__(self, "name", _nonempty(name, where="Synchronize name"))

    def references(self) -> tuple[ValueRef, ...]:
        return (self.value,)

    def to_data(self) -> dict[str, Any]:
        return _node_data(
            self, value=self.value.to_data(), source_clock=self.source_clock.to_data(),
            target_clock=self.target_clock.to_data(), relation=self.relation.to_data(),
            name=self.name)


@dataclass(frozen=True, slots=True, init=False)
class Commit(_Node):
    """Write-only state commit. It has no readable ``ValueRef`` result."""

    kind: ClassVar[str] = "commit"
    readable: ClassVar[bool] = False
    node_id: int
    target: CanonicalData
    value: ValueRef
    clock: Clock
    point: TimePoint | StagePoint

    def __init__(self, node_id: int, target: Any, value: ValueRef, clock: Clock,
                 point: TimePoint | StagePoint) -> None:
        if type(value) is not ValueRef:
            raise TypeError("Commit value must be an exact ValueRef")
        if type(target) is ValueRef:
            raise TypeError("Commit target must be write-only endpoint identity, not a ValueRef")
        object.__setattr__(self, "node_id", _node_id(node_id))
        object.__setattr__(self, "target", CanonicalData(target, where="Commit.target"))
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "point", _point(point))

    def references(self) -> tuple[ValueRef, ...]:
        return (self.value,)

    def to_data(self) -> dict[str, Any]:
        return _node_data(self, target=self.target.to_data(), value=self.value.to_data())


_NODE_TYPES = (
    StateRead, ProgramValue, Unknown, OperatorCall, ResidualEvaluation, ResidualSolve,
    Solve, Branch, Loop, Synchronize, Commit,
)



def _node_data(node: Any, **specific: Any) -> dict[str, Any]:
    return {
        "kind": node.kind,
        "node_id": node.node_id,
        "clock": node.clock.to_data(),
        "point": node.point.to_data(),
        **specific,
    }


@dataclass(frozen=True, slots=True, init=False)
class ProgramGraph:
    """Canonical immutable graph accepted by temporal resolve/compile phases."""

    name: str
    clocks: tuple[Clock, ...]
    nodes: tuple[Any, ...]
    graph_hash: str

    def __init__(self, name: str, nodes: Any, *, clocks: Any = None) -> None:
        object.__setattr__(self, "name", _nonempty(name, where="ProgramGraph name"))
        frozen_nodes = tuple(nodes)
        if any(type(node) not in _NODE_TYPES for node in frozen_nodes):
            raise TypeError("ProgramGraph nodes must be exact graph node records")
        declared = tuple(clocks) if clocks is not None else tuple(dict.fromkeys(
            node.clock for node in frozen_nodes))
        if any(type(clock) is not Clock for clock in declared):
            raise TypeError("ProgramGraph clocks must contain exact Clock values")
        if len(set(declared)) != len(declared):
            raise ValueError("ProgramGraph clocks must be unique")
        object.__setattr__(self, "clocks", declared)
        object.__setattr__(self, "nodes", frozen_nodes)
        self._validate()
        payload = json.dumps(self.to_data(), sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "graph_hash", hashlib.sha256(payload.encode()).hexdigest())

    def _validate(self) -> None:
        from pops.time.graph_validation import validate_nodes

        available: dict[int, Any] = {}
        validate_nodes(self.nodes, self.clocks, available, where="ProgramGraph")

    def ref(self, node: Any) -> ValueRef:
        if type(node) not in _NODE_TYPES or not any(candidate is node for candidate in self.nodes):
            raise ValueError("ProgramGraph.ref requires an exact node owned by this graph")
        if not node.readable:
            raise TypeError("Commit is a write-only graph endpoint and has no readable ValueRef")
        return ValueRef(node.node_id)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "pops.program-graph",
            "name": self.name,
            "clocks": [clock.to_data() for clock in self.clocks],
            "nodes": [node.to_data() for node in self.nodes],
        }


__all__ = [
    "Branch", "CanonicalData", "Commit", "Loop", "OperatorCall", "ProgramGraph", "Region",
    "RegionCapture", "ProgramValue", "ResidualEvaluation", "ResidualSolve", "Solve",
    "StateRead", "Synchronize", "Unknown", "ValueRef",
]
