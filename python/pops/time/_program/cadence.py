"""Immutable macro-step cadence authored by :class:`pops.time.Program`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _positive_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError("%s must be an exact int" % where)
    if value < 1:
        raise ValueError("%s must be >= 1" % where)
    return value


@dataclass(frozen=True, slots=True)
class ProgramCadence:
    """Exact global Program executions within an accepted macro-step window."""

    substeps: int = 1
    stride: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "substeps",
            _positive_int(self.substeps, where="Program cadence substeps"),
        )
        object.__setattr__(
            self,
            "stride",
            _positive_int(self.stride, where="Program cadence stride"),
        )

    @property
    def is_default(self) -> bool:
        return self.substeps == 1 and self.stride == 1

    def to_data(self) -> dict[str, int]:
        return {
            "schema_version": 1,
            "substeps": self.substeps,
            "stride": self.stride,
        }

    @classmethod
    def from_data(cls, data: Any) -> ProgramCadence:
        if type(data) is not dict or set(data) != {
            "schema_version",
            "substeps",
            "stride",
        }:
            raise TypeError("Program cadence data must contain the exact v1 schema")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("Program cadence schema_version must be 1")
        return cls(data["substeps"], data["stride"])


__all__ = ["ProgramCadence"]
