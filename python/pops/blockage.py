"""Versioned classification of a proven API 0.4 blockage (C04).

A validation error is not automatically a scientific blockage. Attach this
record only when the rejecting layer can name the missing relation, realization,
mathematical hypothesis, or domain extension from its own typed contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BlockageClass(str, Enum):
    EXPR = "EXPR"
    IMPL = "IMPL"
    MATH = "MATH"
    SCOPE = "SCOPE"


@dataclass(frozen=True, slots=True)
class Blockage:
    classification: BlockageClass
    phase: str
    source: str
    cause: str
    capability: str | None = None

    def __post_init__(self) -> None:
        if type(self.classification) is not BlockageClass:
            raise TypeError("blockage classification must be a BlockageClass")
        for name in ("phase", "source", "cause"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError("blockage %s must be a non-empty string" % name)
        if self.capability is not None and (type(self.capability) is not str or not self.capability):
            raise ValueError("blockage capability must be a non-empty string or None")

    def to_data(self) -> dict[str, str | int | None]:
        return {
            "schema_version": 1,
            "classification": self.classification.value,
            "phase": self.phase,
            "source": self.source,
            "cause": self.cause,
            "capability": self.capability,
        }


class BlockageValueError(ValueError):
    """Keep ``ValueError`` compatibility while exposing a proven C04 blockage."""

    def __init__(self, message: str, *, blockage: Blockage | None = None) -> None:
        if blockage is not None and type(blockage) is not Blockage:
            raise TypeError("blockage must be an exact Blockage")
        self.blockage = blockage
        super().__init__(message)


class BlockageNotImplementedError(NotImplementedError):
    """Keep ``NotImplementedError`` compatibility for represented but unlowered IR."""

    def __init__(self, message: str, *, blockage: Blockage) -> None:
        if type(blockage) is not Blockage:
            raise TypeError("blockage must be an exact Blockage")
        self.blockage = blockage
        super().__init__(message)


__all__ = ["Blockage", "BlockageClass", "BlockageNotImplementedError", "BlockageValueError"]
