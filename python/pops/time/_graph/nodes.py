"""Immutable node records for the canonical temporal SSA graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from pops.time._graph.base import (
    CanonicalData,
    Node as _Node,
    ValueRef,
    node_data as _node_data,
    node_id as _node_id,
    nonempty as _nonempty,
    point as _point,
    refs as _refs,
)
from pops.time.points import Clock, StagePoint, TimePoint


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
            self,
            "attrs",
            CanonicalData({} if attrs is None else attrs, where="ProgramValue.attrs"),
        )
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


NODE_TYPES = (
    StateRead, ProgramValue, Unknown, OperatorCall, ResidualEvaluation, ResidualSolve,
    Solve, Synchronize, Commit,
)


__all__ = [
    "Commit", "OperatorCall", "ProgramValue", "ResidualEvaluation", "ResidualSolve",
    "Solve", "StateRead", "Synchronize", "Unknown",
]
