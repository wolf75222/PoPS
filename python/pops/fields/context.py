"""Immutable, owner-qualified materialized field contexts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pops.identity import Identity
from pops.model import Handle
from pops.time import Clock, TimePoint

from ._identity import field_identity
from .policies import (
    FieldConsumer,
    FieldReadError,
    FieldReadPolicy,
    HoldLastValue,
    RECOMPUTE_POLICIES,
)


@dataclass(frozen=True, slots=True)
class FieldInput:
    """One qualified dependency and the exact materialized version it read."""

    reference: Handle
    version: Identity
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if not isinstance(self.reference, Handle) or not self.reference.is_resolved:
            raise TypeError("FieldInput reference must be a canonical Handle")
        if not isinstance(self.version, Identity):
            raise TypeError("FieldInput version must be a pops Identity")

    def to_data(self) -> dict[str, Any]:
        return {
            "reference": self.reference.canonical_identity(),
            "version": self.version.token,
        }


@dataclass(frozen=True, slots=True)
class LayoutBinding:
    """Qualified layout identity plus its monotonically changing regrid generation."""

    layout: Handle
    generation: int
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if not isinstance(self.layout, Handle) or not self.layout.is_resolved:
            raise TypeError("LayoutBinding layout must be a canonical Handle")
        if self.layout.kind != "layout":
            raise TypeError("LayoutBinding Handle kind must be 'layout'")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ValueError("LayoutBinding generation must be an integer >= 0")

    def to_data(self) -> dict[str, Any]:
        return {
            "layout": self.layout.canonical_identity(),
            "generation": self.generation,
        }


class FieldMaterialization:
    __pops_ir_immutable__ = True

    def to_data(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Accepted(FieldMaterialization):
    def to_data(self) -> dict[str, Any]:
        return {"state": "accepted"}


@dataclass(frozen=True, slots=True)
class Provisional(FieldMaterialization):
    attempt_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.attempt_id, str) or not self.attempt_id:
            raise ValueError("Provisional attempt_id must be a non-empty string")

    def to_data(self) -> dict[str, Any]:
        return {"state": "provisional", "attempt_id": self.attempt_id}


@dataclass(frozen=True, slots=True, init=False)
class FieldValidity:
    """Explicit valid points avoid assuming continuity between scheduled solves."""

    points: tuple[TimePoint, ...]
    layout: LayoutBinding
    invalid_reason: str | None
    __pops_ir_immutable__ = True

    def __init__(
        self, points: Any, layout: LayoutBinding, *, invalid_reason: str | None = None
    ) -> None:
        point_tuple = tuple(points)
        if any(type(point) is not TimePoint for point in point_tuple):
            raise TypeError("FieldValidity points must contain exact TimePoint values")
        if not isinstance(layout, LayoutBinding):
            raise TypeError("FieldValidity layout must be a LayoutBinding")
        if invalid_reason is not None and (
            not isinstance(invalid_reason, str) or not invalid_reason
        ):
            raise ValueError("FieldValidity invalid_reason must be a non-empty string")
        if invalid_reason is None and not point_tuple:
            raise ValueError("a valid FieldValidity requires at least one TimePoint")
        if len(set(point_tuple)) != len(point_tuple):
            raise ValueError("FieldValidity points must be unique")
        clocks = {point.clock for point in point_tuple}
        if len(clocks) > 1:
            raise ValueError("FieldValidity points must share one Clock")
        object.__setattr__(self, "points", point_tuple)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "invalid_reason", invalid_reason)

    @classmethod
    def valid_at(cls, point: TimePoint, layout: LayoutBinding) -> FieldValidity:
        return cls((point,), layout)

    @classmethod
    def invalid(cls, layout: LayoutBinding, reason: str) -> FieldValidity:
        return cls((), layout, invalid_reason=reason)

    def contains(self, point: TimePoint, layout: LayoutBinding) -> bool:
        return self.invalid_reason is None and layout == self.layout and point in self.points

    def to_data(self) -> dict[str, Any]:
        return {
            "points": [point.to_data() for point in self.points],
            "layout": self.layout.to_data(),
            "invalid_reason": self.invalid_reason,
        }


@dataclass(frozen=True, slots=True)
class UseMaterializedField:
    context_identity: Identity


@dataclass(frozen=True, slots=True)
class UseHeldField:
    context_identity: Identity
    source_point: TimePoint
    requested_point: TimePoint


@dataclass(frozen=True, slots=True)
class RecomputeField:
    context_identity: Identity
    consumer: FieldConsumer
    requested_point: TimePoint
    on_failure: Any


@dataclass(frozen=True, slots=True)
class FieldContext:
    """One exact field materialization; every identity-bearing input is qualified."""

    operator: Handle
    inputs: tuple[FieldInput, ...]
    clock: Clock
    point: TimePoint
    layout: LayoutBinding
    materialization: FieldMaterialization
    validity: FieldValidity
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if not isinstance(self.operator, Handle) or not self.operator.is_resolved:
            raise TypeError("FieldContext operator must be a canonical Handle")
        if self.operator.kind != "field_operator":
            raise TypeError("FieldContext operator Handle kind must be 'field_operator'")
        if any(not isinstance(item, FieldInput) for item in self.inputs):
            raise TypeError("FieldContext inputs must contain FieldInput values")
        references = [item.reference for item in self.inputs]
        if len(references) != len(set(references)):
            raise ValueError("FieldContext inputs must contain each dependency exactly once")
        if type(self.clock) is not Clock or self.clock.owner is None:
            raise TypeError("FieldContext clock must be an owner-qualified exact Clock")
        if type(self.point) is not TimePoint or self.point.clock != self.clock:
            raise ValueError("FieldContext point must use its declared clock")
        if not isinstance(self.layout, LayoutBinding):
            raise TypeError("FieldContext layout must be a LayoutBinding")
        if not isinstance(self.materialization, FieldMaterialization):
            raise TypeError("FieldContext materialization must be Accepted or Provisional")
        if not isinstance(self.validity, FieldValidity):
            raise TypeError("FieldContext validity must be a FieldValidity")
        if self.validity.layout != self.layout:
            raise ValueError("FieldContext validity must target its exact layout generation")
        if self.validity.points and any(
            valid_point.clock != self.clock for valid_point in self.validity.points
        ):
            raise ValueError("FieldContext validity points must use its declared clock")

    @property
    def identity(self) -> Identity:
        return field_identity("field-context", self.to_data())

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "operator": self.operator.canonical_identity(),
            "inputs": [item.to_data() for item in self.inputs],
            "clock": self.clock.to_data(),
            "point": self.point.to_data(),
            "layout": self.layout.to_data(),
            "materialization": self.materialization.to_data(),
            "validity": self.validity.to_data(),
        }

    def inspect(self) -> dict[str, Any]:
        return {**self.to_data(), "identity": self.identity.token}

    def invalidate(self, reason: str) -> FieldContext:
        return replace(self, validity=FieldValidity.invalid(self.layout, reason))

    def accept(self, validity: FieldValidity | None = None) -> FieldContext:
        return replace(
            self,
            materialization=Accepted(),
            validity=validity or FieldValidity.valid_at(self.point, self.layout),
        )

    def resolve_read(
        self,
        consumer: FieldConsumer,
        *,
        at: TimePoint,
        layout: LayoutBinding,
        policy: FieldReadPolicy | None = None,
    ) -> UseMaterializedField | UseHeldField | RecomputeField:
        if not isinstance(consumer, FieldConsumer):
            raise TypeError("field read consumer must be a FieldConsumer")
        if type(at) is not TimePoint or at.clock != self.clock:
            raise ValueError("field read TimePoint must use the FieldContext clock")
        if not isinstance(layout, LayoutBinding):
            raise TypeError("field read layout must be a LayoutBinding")
        current = self.validity.contains(at, layout) and isinstance(self.materialization, Accepted)
        if current:
            return UseMaterializedField(self.identity)
        reason = "field %s is stale/off-schedule for %s at %s on layout generation %d" % (
            self.operator.qualified_id,
            consumer.value,
            at.to_data(),
            layout.generation,
        )
        if policy is None:
            raise FieldReadError(reason + "; provide an explicit typed field read policy")
        if isinstance(policy, HoldLastValue):
            if not isinstance(self.materialization, Accepted):
                policy.on_failure.fail(reason + "; provisional values cannot be held")
            if layout != self.layout:
                policy.on_failure.fail(reason + "; values cannot be held across regrid")
            return UseHeldField(self.identity, self.point, at)
        if isinstance(policy, RECOMPUTE_POLICIES):
            if policy.consumer is not consumer:
                policy.on_failure.fail(
                    reason + "; recompute policy belongs to %s" % policy.consumer.value
                )
            return RecomputeField(self.identity, consumer, at, policy.on_failure)
        raise TypeError("unsupported FieldReadPolicy %s" % type(policy).__name__)


__all__ = [
    "Accepted",
    "FieldContext",
    "FieldInput",
    "FieldMaterialization",
    "FieldValidity",
    "LayoutBinding",
    "Provisional",
    "RecomputeField",
    "UseHeldField",
    "UseMaterializedField",
]
