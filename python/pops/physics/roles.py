"""Typed physical roles for conservative-state components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ComponentRole:
    """Closed protocol translated to the native role vocabulary at the IR boundary."""

    __slots__ = ()

    @property
    def native_name(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Density(ComponentRole):
    @property
    def native_name(self) -> str:
        return "Density"


@dataclass(frozen=True, slots=True)
class Momentum(ComponentRole):
    axis: Any

    def __post_init__(self) -> None:
        name = getattr(self.axis, "name", None)
        if name not in ("x", "y", "z"):
            raise TypeError("Momentum axis must be a typed Cartesian x/y/z axis")

    @property
    def native_name(self) -> str:
        return "Momentum" + str(self.axis.name).upper()


__all__ = ["ComponentRole", "Density", "Momentum"]
