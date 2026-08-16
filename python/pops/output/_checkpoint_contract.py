"""Inert exact-data contracts shared by checkpoint output and live runtimes.

This module intentionally depends on neither runtime implementation nor native extension.
It owns the concrete resource-budget type and canonical envelope member names so output
providers can enforce their already-installed allocation authority without importing
``pops.runtime``.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any


MANIFEST_KEY = "pops_checkpoint_manifest"
IDENTITY_KEY = "pops_restart_identity"


def _capacity(value: Any, *, where: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be an exact integer" % where)
    minimum = 1 if positive else 0
    if value < minimum:
        raise ValueError("%s must be >= %d" % (where, minimum))
    if value > sys.maxsize:
        raise OverflowError("%s exceeds the native addressable range" % where)
    return value


@dataclass(frozen=True, slots=True)
class CheckpointResourceBudget:
    """Trusted allocation envelope installed from one authenticated live runtime."""

    runtime_kind: str
    max_members: int
    max_manifest_characters: int
    max_array_bytes: int
    max_uncompressed_bytes: int
    max_archive_bytes: int
    authority: str

    def __post_init__(self) -> None:
        if self.runtime_kind not in {"uniform", "amr", "multi_layout_uniform"}:
            raise ValueError("checkpoint resource budget has an unsupported runtime kind")
        for name in (
            "max_members",
            "max_manifest_characters",
            "max_array_bytes",
            "max_uncompressed_bytes",
            "max_archive_bytes",
        ):
            _capacity(getattr(self, name), where="checkpoint budget %s" % name, positive=True)
        if self.max_array_bytes > self.max_uncompressed_bytes:
            raise ValueError(
                "checkpoint per-array resource capacity exceeds its aggregate capacity"
            )
        if not isinstance(self.authority, str) or not self.authority:
            raise TypeError("checkpoint resource budget authority must be non-empty text")


def require_checkpoint_resource_budget(owner: Any) -> CheckpointResourceBudget:
    """Return the exact authenticated budget installed on one runtime owner."""
    budget = getattr(owner, "_checkpoint_resource_budget", None)
    if type(budget) is not CheckpointResourceBudget:
        raise RuntimeError("checkpoint decode requires the authenticated live resource budget")
    return budget


__all__ = [
    "CheckpointResourceBudget",
    "IDENTITY_KEY",
    "MANIFEST_KEY",
    "require_checkpoint_resource_budget",
]
