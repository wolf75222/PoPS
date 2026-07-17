"""Pure, typed Cartesian coordinate frames.

Frames are semantic authoring descriptors.  They never import the native extension and they do
not contain mesh or execution choices.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from pops.identity import make_identity


_SCHEMA_VERSION = 1


class CartesianDirection(Enum):
    """Closed set of directions carried by :class:`Cartesian2D`."""

    X = "x"
    Y = "y"


@dataclass(frozen=True, slots=True)
class CartesianAxis:
    """One immutable, typed axis of a Cartesian frame."""

    direction: CartesianDirection

    def __post_init__(self) -> None:
        if not isinstance(self.direction, CartesianDirection):
            raise TypeError("CartesianAxis.direction must be a CartesianDirection")

    @property
    def index(self) -> int:
        return 0 if self.direction is CartesianDirection.X else 1

    @property
    def name(self) -> str:
        return self.direction.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "axis_type": "cartesian",
            "direction": self.direction.value,
            "index": self.index,
        }

    canonical_identity = to_dict

    @classmethod
    def from_dict(cls, data: Any) -> CartesianAxis:
        required = {"schema_version", "axis_type", "direction", "index"}
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("CartesianAxis data has an unsupported shape")
        if data["schema_version"] != _SCHEMA_VERSION or data["axis_type"] != "cartesian":
            raise ValueError("CartesianAxis data uses an unsupported schema")
        try:
            result = cls(CartesianDirection(data["direction"]))
        except (TypeError, ValueError) as exc:
            raise ValueError("CartesianAxis direction must be 'x' or 'y'") from exc
        if result.to_dict() != dict(data):
            raise ValueError("CartesianAxis data is not canonical")
        return result


X_AXIS = CartesianAxis(CartesianDirection.X)
Y_AXIS = CartesianAxis(CartesianDirection.Y)


@dataclass(frozen=True, slots=True)
class Cartesian2D:
    """The canonical two-dimensional Cartesian coordinate frame.

    Axes are values, not strings.  Algorithms select them with ``frame.x`` / ``frame.y`` or by
    iterating ``frame.axes``; accepting a runtime string selector here would make typos and
    dimension mismatches visible only during execution.
    """

    dimension: ClassVar[int] = 2

    @property
    def x(self) -> CartesianAxis:
        return X_AXIS

    @property
    def y(self) -> CartesianAxis:
        return Y_AXIS

    @property
    def axes(self) -> tuple[CartesianAxis, CartesianAxis]:
        return (X_AXIS, Y_AXIS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "frame_type": "cartesian_2d",
            "dimension": self.dimension,
            "axes": [axis.to_dict() for axis in self.axes],
        }

    canonical_identity = to_dict

    @property
    def canonical_id(self) -> str:
        return make_identity("frame", self.to_dict(), schema_version=_SCHEMA_VERSION).token

    @classmethod
    def from_dict(cls, data: Any) -> Cartesian2D:
        required = {"schema_version", "frame_type", "dimension", "axes"}
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("Cartesian2D data has an unsupported shape")
        if data["schema_version"] != _SCHEMA_VERSION \
                or data["frame_type"] != "cartesian_2d" or data["dimension"] != 2:
            raise ValueError("Cartesian2D data uses an unsupported schema")
        raw_axes = data["axes"]
        if not isinstance(raw_axes, list):
            raise TypeError("Cartesian2D axes must be a canonical list")
        axes = tuple(CartesianAxis.from_dict(axis) for axis in raw_axes)
        result = cls()
        if axes != result.axes or result.to_dict() != dict(data):
            raise ValueError("Cartesian2D data is not canonical")
        return result


__all__ = [
    "Cartesian2D", "CartesianAxis", "CartesianDirection", "X_AXIS", "Y_AXIS",
]
