"""Small immutable representation descriptors used by physical spaces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Representation:
    """What stored components mean, independent of any conversion implementation."""

    name: str

    category = "representation"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name.strip() != self.name:
            raise TypeError("Representation.name must be canonical non-empty text")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "category": self.category, "name": self.name}

    canonical_identity = to_dict
    inspect = to_dict


def Conservative() -> Representation:
    """Components are conservative densities over their control volumes."""

    return Representation("conservative")


def Primitive() -> Representation:
    """Components are primitive physical variables."""

    return Representation("primitive")


__all__ = ["Representation", "Conservative", "Primitive"]
