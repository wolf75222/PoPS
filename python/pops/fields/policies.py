"""Typed stale/off-schedule field-read policies and explicit failure actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class FieldReadError(RuntimeError):
    """A field read is stale, off-schedule, or invalid for its consumer."""


class FieldAttemptRejected(FieldReadError):
    """A provisional step attempt must be rejected because a field cannot be supplied."""


class FieldConsumer(Enum):
    PROGRAM = "program"
    DIAGNOSTIC = "diagnostic"
    OUTPUT = "output"
    TAGGING = "tagging"


class FieldFailureAction:
    """Small extension interface controlling a failed hold/recompute request."""

    __pops_ir_immutable__ = True

    def fail(self, message: str) -> None:
        raise NotImplementedError

    def to_data(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FailFieldRead(FieldFailureAction):
    def fail(self, message: str) -> None:
        raise FieldReadError(message)

    def to_data(self) -> dict[str, Any]:
        return {"action": "fail_field_read"}


@dataclass(frozen=True, slots=True)
class RejectFieldAttempt(FieldFailureAction):
    def fail(self, message: str) -> None:
        raise FieldAttemptRejected(message)

    def to_data(self) -> dict[str, Any]:
        return {"action": "reject_field_attempt"}


class FieldReadPolicy:
    """Closed semantic role with open typed implementations."""

    __pops_ir_immutable__ = True


def _failure(value: Any) -> FieldFailureAction:
    if not isinstance(value, FieldFailureAction):
        raise TypeError("field read policy on_failure must be a FieldFailureAction")
    return value


@dataclass(frozen=True, slots=True)
class HoldLastValue(FieldReadPolicy):
    """Reuse the last accepted value on the same layout only."""

    on_failure: FieldFailureAction

    def __post_init__(self) -> None:
        _failure(self.on_failure)

    def to_data(self) -> dict[str, Any]:
        return {"policy": "hold_last_value", "on_failure": self.on_failure.to_data()}


@dataclass(frozen=True, slots=True)
class RecomputeAtDiagnostic(FieldReadPolicy):
    on_failure: FieldFailureAction
    consumer = FieldConsumer.DIAGNOSTIC

    def __post_init__(self) -> None:
        _failure(self.on_failure)

    def to_data(self) -> dict[str, Any]:
        return {
            "policy": "recompute",
            "consumer": self.consumer.value,
            "on_failure": self.on_failure.to_data(),
        }


@dataclass(frozen=True, slots=True)
class RecomputeAtOutput(FieldReadPolicy):
    on_failure: FieldFailureAction
    consumer = FieldConsumer.OUTPUT

    def __post_init__(self) -> None:
        _failure(self.on_failure)

    def to_data(self) -> dict[str, Any]:
        return {
            "policy": "recompute",
            "consumer": self.consumer.value,
            "on_failure": self.on_failure.to_data(),
        }


@dataclass(frozen=True, slots=True)
class RecomputeAtTagging(FieldReadPolicy):
    on_failure: FieldFailureAction
    consumer = FieldConsumer.TAGGING

    def __post_init__(self) -> None:
        _failure(self.on_failure)

    def to_data(self) -> dict[str, Any]:
        return {
            "policy": "recompute",
            "consumer": self.consumer.value,
            "on_failure": self.on_failure.to_data(),
        }


RECOMPUTE_POLICIES = (
    RecomputeAtDiagnostic,
    RecomputeAtOutput,
    RecomputeAtTagging,
)


__all__ = [
    "FailFieldRead",
    "FieldAttemptRejected",
    "FieldConsumer",
    "FieldFailureAction",
    "FieldReadError",
    "FieldReadPolicy",
    "HoldLastValue",
    "RecomputeAtDiagnostic",
    "RecomputeAtOutput",
    "RecomputeAtTagging",
    "RejectFieldAttempt",
]
