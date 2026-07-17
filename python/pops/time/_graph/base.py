"""Foundational immutable values shared by temporal graph records."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from typing import Any, ClassVar, cast

from pops._ir.literals import ScalarLiteral, scalar_data
from pops.model.ownership import OwnerPath
from pops.time.points import Clock, StagePoint, TimePoint


def nonempty(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("%s must be a non-empty string" % where)
    return value


def node_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("ProgramGraph node_id must be a non-negative Python int")
    return value


def strict_data(value: Any, *, where: str) -> Any:
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
            key: strict_data(item, where="%s.%s" % (where, key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            strict_data(item, where="%s[%d]" % (where, index))
            for index, item in enumerate(value)
        ]
    canonical = getattr(value, "canonical_identity", None)
    if callable(canonical):
        return strict_data(canonical(), where=where)
    to_data = getattr(value, "to_data", None)
    if callable(to_data) and getattr(value, "__pops_ir_immutable__", False) is True:
        return strict_data(to_data(), where=where)
    raise TypeError(
        "%s contains mutable/opaque %s; provide canonical immutable data"
        % (where, type(value).__name__))


@dataclass(frozen=True, slots=True, init=False)
class CanonicalData:
    """Hashable canonical-data snapshot used for semantic node metadata."""

    _json: str

    def __init__(self, value: Any, *, where: str = "ProgramGraph metadata") -> None:
        data = strict_data(value, where=where)
        object.__setattr__(
            self, "_json", json.dumps(data, sort_keys=True, separators=(",", ":")))

    def to_data(self) -> Any:
        return json.loads(self._json)


@dataclass(frozen=True, slots=True)
class ValueRef:
    """Readable SSA edge. Commit nodes deliberately cannot produce one."""

    node_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", node_id(self.node_id))

    def to_data(self) -> dict[str, int]:
        return {"node_id": self.node_id}


def refs(values: Any, *, where: str) -> tuple[ValueRef, ...]:
    result = tuple(values)
    if any(type(value) is not ValueRef for value in result):
        raise TypeError("%s must contain exact ValueRef values" % where)
    return result


def point(value: Any) -> TimePoint | StagePoint:
    if type(value) not in (TimePoint, StagePoint):
        raise TypeError("ProgramGraph node point must be an exact TimePoint or StagePoint")
    return value


def point_clocks(value: TimePoint | StagePoint) -> frozenset[Clock]:
    if type(value) is TimePoint:
        return frozenset((cast(TimePoint, value).clock,))
    return frozenset(
        item.clock for item in cast(StagePoint, value).partitions.values()
    )


def node_data(node: Any, **specific: Any) -> dict[str, Any]:
    return {
        "kind": node.kind,
        "node_id": node.node_id,
        "clock": node.clock.to_data(),
        "point": node.point.to_data(),
        **specific,
    }


class Node:
    """Closed internal base protocol for exact graph node records."""

    kind: ClassVar[str]
    readable: ClassVar[bool] = True
    node_id: int

    def references(self) -> tuple[ValueRef, ...]:
        return ()


__all__ = ["CanonicalData", "ValueRef"]
