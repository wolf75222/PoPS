"""Typed semantic contracts for retained temporal histories."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pops.model.handles import Handle
from pops.model.hash_data import canonical_hash_data
from pops.model.ownership import OwnerPath
from pops.time.points import Clock, TimePoint
from pops.time.references import handle_data


@runtime_checkable
class InterpolationCapability(Protocol):
    """Extension protocol describing what a retained history can reconstruct."""

    __pops_history_interpolation__: bool

    def to_data(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class NoInterpolation:
    """Only exact retained samples are available."""

    __pops_history_interpolation__ = True
    __pops_ir_immutable__ = True

    def to_data(self) -> dict[str, Any]:
        return {"kind": "none", "schema_version": 1}


@dataclass(frozen=True, slots=True)
class PiecewiseConstant:
    """Hold the latest valid sample between source-clock ticks."""

    __pops_history_interpolation__ = True
    __pops_ir_immutable__ = True

    def to_data(self) -> dict[str, Any]:
        return {"kind": "piecewise_constant", "schema_version": 1}


@dataclass(frozen=True, slots=True)
class LinearInterpolation:
    """Interpolate between two bracketing accepted samples."""

    __pops_history_interpolation__ = True
    __pops_ir_immutable__ = True

    def to_data(self) -> dict[str, Any]:
        return {"kind": "linear", "schema_version": 1, "minimum_samples": 2}


@dataclass(frozen=True, slots=True, init=False)
class DenseOutput:
    """Method-provided polynomial dense output of one declared order."""

    order: int
    __pops_history_interpolation__ = True
    __pops_ir_immutable__ = True

    def __init__(self, order: int) -> None:
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("DenseOutput order must be a Python int >= 1")
        object.__setattr__(self, "order", order)

    def to_data(self) -> dict[str, Any]:
        return {"kind": "dense_output", "schema_version": 1, "order": self.order}


def interpolation_data(value: Any) -> dict[str, Any]:
    """Return a detached canonical snapshot of one interpolation capability."""
    if not isinstance(value, InterpolationCapability) \
            or value.__pops_history_interpolation__ is not True:
        raise TypeError(
            "history interpolation must implement InterpolationCapability "
            "(__pops_history_interpolation__ and to_data())"
        )
    data = value.to_data()
    if not isinstance(data, dict) or not isinstance(data.get("kind"), str) or not data["kind"]:
        raise TypeError("history interpolation to_data() must contain a non-empty kind")
    canonical = canonical_hash_data(data, where="history interpolation")
    if not isinstance(canonical, dict):
        raise TypeError("history interpolation to_data() must return a mapping")
    return canonical


@dataclass(frozen=True, slots=True, init=False)
class _CanonicalInterpolation:
    """Immutable JSON snapshot of an interpolation extension descriptor."""

    _json: str
    __pops_history_interpolation__ = True
    __pops_ir_immutable__ = True

    def __init__(self, data: dict[str, Any]) -> None:
        canonical = canonical_hash_data(data, where="history interpolation")
        object.__setattr__(
            self,
            "_json",
            json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )

    @property
    def kind(self) -> str:
        return self.to_data()["kind"]

    def to_data(self) -> dict[str, Any]:
        return json.loads(self._json)


@dataclass(frozen=True, slots=True, init=False)
class HistoryValidity:
    """Closed validity interval on exactly one logical source clock."""

    oldest: TimePoint
    newest: TimePoint
    __pops_ir_immutable__ = True

    def __init__(self, oldest: TimePoint, newest: TimePoint) -> None:
        if type(oldest) is not TimePoint or type(newest) is not TimePoint:
            raise TypeError("HistoryValidity endpoints must be exact TimePoint values")
        if oldest.clock != newest.clock:
            raise ValueError("HistoryValidity endpoints must use the same clock")
        old_coordinate = oldest.step + oldest.offset.to_python()
        new_coordinate = newest.step + newest.offset.to_python()
        if old_coordinate > new_coordinate:
            raise ValueError("HistoryValidity oldest endpoint must not follow newest")
        object.__setattr__(self, "oldest", oldest)
        object.__setattr__(self, "newest", newest)

    @property
    def clock(self) -> Clock:
        return self.oldest.clock

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "oldest": self.oldest.to_data(),
            "newest": self.newest.to_data(),
        }


@dataclass(frozen=True, slots=True, init=False)
class HistoryContract:
    """Complete qualified identity and reconstruction contract of one state history."""

    owner: OwnerPath
    state: Handle
    space: Any
    clock: Clock
    validity: HistoryValidity
    interpolation: InterpolationCapability
    depth: int
    __pops_ir_immutable__ = True

    def __init__(
        self,
        *,
        owner: Any,
        state: Handle,
        space: Any,
        clock: Clock,
        validity: HistoryValidity,
        interpolation: Any,
        depth: int,
    ) -> None:
        owner = OwnerPath.coerce(owner).canonical()
        if not isinstance(state, Handle) or state.kind != "state" or not state.is_instance:
            raise TypeError("HistoryContract state must be a block-qualified state Handle")
        if space is not None and getattr(space, "kind", None) != "state":
            raise TypeError("HistoryContract space must be a StateSpace or None")
        if type(clock) is not Clock:
            raise TypeError("HistoryContract clock must be an exact Clock")
        if type(validity) is not HistoryValidity or validity.clock != clock:
            raise ValueError("HistoryContract validity must use the history clock")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("HistoryContract depth must be a Python int >= 1")
        capability = _CanonicalInterpolation(interpolation_data(interpolation))
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "validity", validity)
        object.__setattr__(self, "interpolation", capability)
        object.__setattr__(self, "depth", depth)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "owner": self.owner.to_data(),
            "state": handle_data(self.state),
            "space": self.space.to_data() if self.space is not None else None,
            "clock": self.clock.to_data(),
            "validity": self.validity.to_data(),
            "interpolation": interpolation_data(self.interpolation),
            "depth": self.depth,
        }


__all__ = [
    "DenseOutput",
    "HistoryContract",
    "HistoryValidity",
    "InterpolationCapability",
    "LinearInterpolation",
    "NoInterpolation",
    "PiecewiseConstant",
    "interpolation_data",
]
