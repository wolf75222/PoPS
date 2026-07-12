"""Structured, immutable validation and backend-support reports for residual systems."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar


def _names(values: Any, where: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value or value.strip() != value for value in result):
        raise ValueError("%s must contain non-empty, trimmed strings" % where)
    if len(set(result)) != len(result):
        raise ValueError("%s contains duplicate names" % where)
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("canonical mappings require non-empty string keys")
        return MappingProxyType({key: _freeze(item) for key, item in sorted(value.items())})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("%s is not canonical report metadata" % type(value).__name__)


def _data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_data(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


class _Report:
    __pops_ir_immutable__: ClassVar[bool] = True
    kind: ClassVar[str]

    def to_data(self) -> dict[str, Any]:
        return {"kind": self.kind, **{
            field.name: _data(getattr(self, field.name)) for field in fields(self)}}


@dataclass(frozen=True, slots=True)
class ResidualReport(_Report):
    kind: ClassVar[str] = "residual_report"
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    facts: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("ResidualReport.valid must be bool")
        object.__setattr__(self, "errors", _names(self.errors, "ResidualReport.errors"))
        object.__setattr__(self, "warnings", _names(self.warnings, "ResidualReport.warnings"))
        object.__setattr__(self, "facts", _freeze(self.facts))
        if self.valid == bool(self.errors):
            raise ValueError("ResidualReport.valid must be exactly equivalent to no errors")


class SupportStatus(str, Enum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SupportReport(_Report):
    kind: ClassVar[str] = "support_report"
    status: SupportStatus | str
    backend: str | None
    missing: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    alternative: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "status", SupportStatus(self.status))
        except (TypeError, ValueError) as exc:
            raise ValueError("SupportReport.status must be a closed SupportStatus") from exc
        if self.backend is not None and (
                not isinstance(self.backend, str) or not self.backend):
            raise ValueError("SupportReport.backend must be a non-empty string or None")
        for name in ("missing", "reasons", "limitations"):
            object.__setattr__(self, name, _names(getattr(self, name), "SupportReport.%s" % name))
        if self.alternative is not None:
            object.__setattr__(self, "alternative", _freeze(self.alternative))
        if self.status is SupportStatus.AVAILABLE:
            if self.backend is None or self.missing or self.reasons or self.limitations:
                raise ValueError("available support needs a backend and cannot contain caveats")
        elif not (self.missing or self.reasons or self.limitations):
            raise ValueError("non-available support must explain its status")

    @property
    def supported(self) -> bool:
        return self.status is SupportStatus.AVAILABLE


__all__ = ["ResidualReport", "SupportReport", "SupportStatus"]
