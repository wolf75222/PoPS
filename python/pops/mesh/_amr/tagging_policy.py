"""The policies of AMR tagging: equality, conflict and hysteresis (N12 lot 3c-2b).

A refinement criterion (``refinement_criterion.py``) decides a request per cell; these policies say
what a cell exactly on a strict threshold decides, which request wins when refine and coarsen both
hold, and how many regrids a transition waits. They moved here from the tag graph when the graph was
removed, with their names and identities unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

_SCHEMA_VERSION = 1


class EqualityPolicy(Enum):
    """Action when an indicator is exactly on a strict hysteresis threshold."""

    HOLD = "hold"
    REFINE = "refine"
    COARSEN = "coarsen"


class ConflictPolicy(Enum):
    """Action when refine and coarsen roots both match the same cell."""

    ERROR = "error"
    HOLD = "hold"
    REFINE_WINS = "refine_wins"
    COARSEN_WINS = "coarsen_wins"


@dataclass(frozen=True, slots=True)
class Hysteresis:
    """Temporal stability and equality semantics for refine/coarsen transitions."""

    min_cycles: int
    equality: EqualityPolicy

    def __post_init__(self) -> None:
        if isinstance(self.min_cycles, bool) or not isinstance(self.min_cycles, int) \
                or self.min_cycles < 0:
            raise ValueError("Hysteresis.min_cycles must be an integer >= 0")
        if not isinstance(self.equality, EqualityPolicy):
            raise TypeError("Hysteresis.equality must be an explicit EqualityPolicy")

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "hysteresis_type": "min_cycles",
                "min_cycles": self.min_cycles, "equality": self.equality.value}

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "amr_hysteresis", **self.canonical_identity()}


__all__ = ["ConflictPolicy", "EqualityPolicy", "Hysteresis"]
