"""Exact immutable temporal coordinates for Program graphs."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pops._ir.literals import ScalarLiteral, scalar_literal
from pops.model.ownership import OwnerPath


def _name(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("%s must be a non-empty string" % where)
    return value


def _offset(value: Any) -> ScalarLiteral:
    literal = scalar_literal(value)
    if literal.kind not in {"integer", "rational", "decimal", "binary64"}:
        raise TypeError("TimePoint offset must be an exact numeric scalar")
    if literal.unit is not None or literal.target is not None or literal.cpp is not None:
        raise TypeError("TimePoint offset cannot carry a unit, target, or custom C++ spelling")
    return literal


@dataclass(frozen=True, slots=True, init=False)
class Clock:
    """One named logical clock, optionally qualified by a canonical owner."""

    name: str
    owner: OwnerPath | None
    __pops_ir_immutable__ = True

    def __init__(self, name: str, owner: OwnerPath | None = None) -> None:
        object.__setattr__(self, "name", _name(name, where="Clock name"))
        if owner is not None:
            owner = OwnerPath.coerce(owner).canonical()
        object.__setattr__(self, "owner", owner)

    @property
    def qualified_id(self) -> str:
        payload = json.dumps(self.to_data(), sort_keys=True, separators=(",", ":"))
        return "pops.clock.v1::sha256:%s" % hashlib.sha256(payload.encode()).hexdigest()

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "owner": self.owner.to_data() if self.owner is not None else None,
        }


@dataclass(frozen=True, slots=True, init=False)
class TimePoint:
    """Exact coordinate ``clock[n + step] + offset * dt`` before target conversion."""

    clock: Clock
    offset: ScalarLiteral
    step: int
    __pops_ir_immutable__ = True

    def __init__(self, clock: Clock, offset: Any = 0, *, step: int = 0) -> None:
        if type(clock) is not Clock:
            raise TypeError("TimePoint clock must be an exact Clock")
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("TimePoint step must be a Python int")
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "offset", _offset(offset))
        object.__setattr__(self, "step", step)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "clock": self.clock.to_data(),
            "step": self.step,
            "offset": self.offset.to_data(),
        }


@dataclass(frozen=True, slots=True, init=False)
class StagePoint:
    """Named partition coordinates for one stage.

    Partitioned methods may assign different exact times to e.g. ``explicit`` and ``implicit``.
    The unqualified :attr:`time` accessor is therefore defined only when every coordinate agrees.
    """

    name: str
    _coordinates: tuple[tuple[str, TimePoint], ...]
    __pops_ir_immutable__ = True

    def __init__(self, name: str, partitions: Mapping[str, TimePoint]) -> None:
        object.__setattr__(self, "name", _name(name, where="StagePoint name"))
        if not isinstance(partitions, Mapping) or not partitions:
            raise TypeError("StagePoint partitions must be a non-empty mapping")
        coordinates = []
        for partition, point in partitions.items():
            partition = _name(partition, where="StagePoint partition")
            if type(point) is not TimePoint:
                raise TypeError(
                    "StagePoint partition %r must contain an exact TimePoint" % partition)
            coordinates.append((partition, point))
        coordinates.sort(key=lambda item: item[0])
        object.__setattr__(self, "_coordinates", tuple(coordinates))

    @property
    def partitions(self) -> Mapping[str, TimePoint]:
        return MappingProxyType(dict(self._coordinates))

    def time_for(self, partition: str) -> TimePoint:
        partition = _name(partition, where="StagePoint partition")
        try:
            return self.partitions[partition]
        except KeyError:
            raise KeyError(
                "unknown StagePoint partition %r; declared partitions: %s"
                % (partition, [name for name, _ in self._coordinates])) from None

    @property
    def time(self) -> TimePoint:
        first = self._coordinates[0][1]
        if any(point != first for _, point in self._coordinates[1:]):
            raise ValueError(
                "StagePoint %r has ambiguous partition times; use time_for(partition)"
                % self.name)
        return first

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "partitions": {
                name: point.to_data() for name, point in self._coordinates
            },
        }


def point_clock(point: Any, where: str) -> Clock:
    """Return the single logical clock of an exact temporal point."""
    if type(point) is TimePoint:
        return point.clock
    if type(point) is StagePoint:
        clocks = {coordinate.clock for coordinate in point.partitions.values()}
        if len(clocks) != 1:
            raise ValueError("%s: StagePoint partitions do not share one clock" % where)
        return next(iter(clocks))
    raise TypeError("%s: expected an exact TimePoint or StagePoint" % where)


__all__ = ["Clock", "StagePoint", "TimePoint", "point_clock"]
