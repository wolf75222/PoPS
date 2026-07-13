"""Pure placement descriptors for public physical state declarations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _frame_identity(frame: Any) -> str:
    token = getattr(frame, "canonical_id", None)
    if not isinstance(token, str) or not token:
        raise TypeError("state placement frame must expose a canonical_id")
    if not callable(getattr(frame, "to_dict", None)):
        raise TypeError("state placement frame must implement to_dict()")
    return token


@dataclass(frozen=True, slots=True)
class StatePlacement:
    """Where and how one state family is stored, independent of its variables."""

    frame: Any
    centering: str
    layout: str
    storage: str
    clock: str = "simulation"

    category = "state_placement"

    def __post_init__(self) -> None:
        _frame_identity(self.frame)
        for name in ("centering", "layout", "storage", "clock"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise TypeError("StatePlacement.%s must be canonical non-empty text" % name)

    @property
    def frame_id(self) -> str:
        return _frame_identity(self.frame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "category": self.category,
            "frame": self.frame.to_dict(),
            "frame_id": self.frame_id,
            "centering": self.centering,
            "layout": self.layout,
            "storage": self.storage,
            "clock": self.clock,
        }

    canonical_identity = to_dict
    inspect = to_dict


def CellState(*, frame: Any, storage: str = "multifab",
              clock: str = "simulation") -> StatePlacement:
    """Cell-centred state placement; stencil order and halo depth are intentionally absent."""

    return StatePlacement(
        frame=frame, centering="cell", layout="cell", storage=storage, clock=clock)


__all__ = ["StatePlacement", "CellState"]
