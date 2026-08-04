"""Pure, typed Cartesian coordinate frames.

Frames are semantic authoring descriptors.  They never import the native extension and they do
not contain mesh or execution choices.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pops.identity import make_identity


_SCHEMA_VERSION = 1


class CartesianDirection(Enum):
    """Closed physical x/y/z component directions.

    :class:`Cartesian2D` carries only x/y as mesh axes; z remains available to type transverse
    polar components and out-of-plane axial components in a 2.5D model.
    """

    X = "x"
    Y = "y"
    Z = "z"


@dataclass(frozen=True, slots=True)
class CartesianAxis:
    """One immutable, typed axis of a Cartesian frame."""

    direction: CartesianDirection

    def __post_init__(self) -> None:
        if not isinstance(self.direction, CartesianDirection):
            raise TypeError("CartesianAxis.direction must be a CartesianDirection")

    @property
    def index(self) -> int:
        return {
            CartesianDirection.X: 0,
            CartesianDirection.Y: 1,
            CartesianDirection.Z: 2,
        }[self.direction]

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
            raise ValueError("CartesianAxis direction must be 'x', 'y', or 'z'") from exc
        if result.to_dict() != dict(data):
            raise ValueError("CartesianAxis data is not canonical")
        return result


X_AXIS = CartesianAxis(CartesianDirection.X)
Y_AXIS = CartesianAxis(CartesianDirection.Y)
Z_AXIS = CartesianAxis(CartesianDirection.Z)


@dataclass(frozen=True, slots=True)
class Cartesian:
    """One canonical Cartesian frame whose rank is fixed by its authoring domain.

    Numerical code iterates :attr:`axes`; the fixed-rank constructors below are only public value
    constructors and do not select a separate algorithm.  The dimension is persisted in the frame
    identity and later selects exactly one compiled native specialization.
    """

    dimension: int

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise TypeError("Cartesian.dimension must be an exact integer")
        if self.dimension not in (1, 2, 3):
            raise ValueError("Cartesian.dimension must be 1, 2, or 3")

    @property
    def axes(self) -> tuple[CartesianAxis, ...]:
        return (X_AXIS, Y_AXIS, Z_AXIS)[:self.dimension]

    @property
    def x(self) -> CartesianAxis:
        return X_AXIS

    @property
    def y(self) -> CartesianAxis:
        if self.dimension < 2:
            raise AttributeError("a one-dimensional Cartesian frame has no y axis")
        return Y_AXIS

    @property
    def z(self) -> CartesianAxis:
        if self.dimension < 3:
            raise AttributeError("this Cartesian frame has no z axis")
        return Z_AXIS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "frame_type": "cartesian",
            "dimension": self.dimension,
            "axes": [axis.to_dict() for axis in self.axes],
        }

    canonical_identity = to_dict

    @property
    def canonical_id(self) -> str:
        return make_identity("frame", self.to_dict(), schema_version=_SCHEMA_VERSION).token

    @classmethod
    def from_dict(cls, data: Any) -> Cartesian:
        required = {"schema_version", "frame_type", "dimension", "axes"}
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("Cartesian data has an unsupported shape")
        if data["schema_version"] != _SCHEMA_VERSION or data["frame_type"] != "cartesian":
            raise ValueError("Cartesian data uses an unsupported schema")
        raw_axes = data["axes"]
        if not isinstance(raw_axes, list):
            raise TypeError("Cartesian axes must be a canonical list")
        axes = tuple(CartesianAxis.from_dict(axis) for axis in raw_axes)
        result = cls(data["dimension"]) if cls is Cartesian else cls()
        if result.dimension != data["dimension"]:
            raise ValueError(
                "%s data carries Cartesian dimension %r" % (cls.__name__, data["dimension"])
            )
        if axes != result.axes or result.to_dict() != dict(data):
            raise ValueError("Cartesian data is not canonical")
        return result


class Cartesian1D(Cartesian):
    """Public constructor for the rank-one specialization of :class:`Cartesian`."""

    def __init__(self) -> None:
        super().__init__(1)


class Cartesian2D(Cartesian):
    """Public constructor for the rank-two specialization of :class:`Cartesian`."""

    def __init__(self) -> None:
        super().__init__(2)


class Cartesian3D(Cartesian):
    """Public constructor for the rank-three specialization of :class:`Cartesian`."""

    def __init__(self) -> None:
        super().__init__(3)


__all__ = [
    "Cartesian",
    "Cartesian1D",
    "Cartesian2D",
    "Cartesian3D",
    "CartesianAxis",
    "CartesianDirection",
    "X_AXIS",
    "Y_AXIS",
    "Z_AXIS",
]
