"""Immutable public snapshot of declarations that require a resolved layout."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LayoutSubjects:
    blocks: tuple[Any, ...]
    states: tuple[Any, ...]
    fields: tuple[Any, ...]

    def to_dict(self) -> dict[str, tuple[Any, ...]]:
        return {"blocks": self.blocks, "states": self.states, "fields": self.fields}


__all__ = ["LayoutSubjects"]
