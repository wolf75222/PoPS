"""Typed level selections shared by scientific-output consumers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class LevelSelection:
    """Small immutable protocol for selecting levels from one normalized layout."""

    __pops_ir_immutable__ = True

    def select_levels(self, layout: Any) -> tuple[int, ...]:
        raise NotImplementedError

    def to_data(self) -> dict[str, Any]:
        raise NotImplementedError

    def inspect(self) -> dict[str, Any]:
        return self.to_data()


@dataclass(frozen=True, slots=True)
class AllLevels(LevelSelection):
    def select_levels(self, layout: Any) -> tuple[int, ...]:
        levels = getattr(layout, "levels", None)
        if not isinstance(levels, tuple) or not levels:
            raise TypeError("AllLevels requires a normalized layout with typed levels")
        return tuple(level.index for level in levels)

    def to_data(self) -> dict[str, Any]:
        return {"selection": "all"}


@dataclass(frozen=True, slots=True)
class CoarseOnly(LevelSelection):
    def select_levels(self, layout: Any) -> tuple[int, ...]:
        available = AllLevels().select_levels(layout)
        if 0 not in available:
            raise ValueError("CoarseOnly requires a level-zero layout")
        return (0,)

    def to_data(self) -> dict[str, Any]:
        return {"selection": "coarse"}


@dataclass(frozen=True, slots=True, init=False)
class SelectedLevels(LevelSelection):
    levels: tuple[int, ...]

    def __init__(self, *levels: Any) -> None:
        if not levels:
            raise ValueError("SelectedLevels requires at least one level")
        if any(isinstance(level, bool) or not isinstance(level, int) or level < 0
               for level in levels):
            raise TypeError("SelectedLevels entries must be integers >= 0")
        normalized = tuple(sorted(set(levels)))
        if len(normalized) != len(levels):
            raise ValueError("SelectedLevels entries must be unique")
        object.__setattr__(self, "levels", normalized)

    def select_levels(self, layout: Any) -> tuple[int, ...]:
        available = set(AllLevels().select_levels(layout))
        missing = tuple(level for level in self.levels if level not in available)
        if missing:
            raise ValueError("selected output levels are absent from the resolved layout: %r"
                             % (missing,))
        return self.levels

    def to_data(self) -> dict[str, Any]:
        return {"selection": "selected", "levels": list(self.levels)}


__all__ = ["AllLevels", "CoarseOnly", "LevelSelection", "SelectedLevels"]
