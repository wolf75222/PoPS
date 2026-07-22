"""Immutable rectangular domains and their typed geometric boundaries."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import math
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pops.frames import Cartesian2D, CartesianAxis, X_AXIS, Y_AXIS
from pops.identity import make_identity
from pops.identity.semantic import semantic_value

if TYPE_CHECKING:
    from .preview import AnalyticPreviewValue, DomainPreview, GeometryPreviewProvider


_SCHEMA_VERSION = 1


def _identity(domain: str, payload: Any) -> str:
    projected = semantic_value(payload, where="%s identity" % domain)
    return make_identity(domain, projected, schema_version=_SCHEMA_VERSION).token


def _name(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("%s must be non-empty text" % where)
    result = value.strip()
    if "::" in result:
        raise ValueError("%s must not contain the reserved '::' separator" % where)
    return result


def _point(value: Any, *, where: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise TypeError("%s must contain exactly two real coordinates" % where)
    result = []
    for index, coordinate in enumerate(value):
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise TypeError("%s[%d] must be a real number, never bool" % (where, index))
        converted = float(coordinate)
        if not math.isfinite(converted):
            raise ValueError("%s[%d] must be finite" % (where, index))
        result.append(converted)
    return (result[0], result[1])


@dataclass(frozen=True, slots=True)
class DomainTag:
    """A semantic region label, distinct from boundary or runtime selectors."""

    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, where="DomainTag.name"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "tag_type": "domain", "name": self.name}

    @classmethod
    def from_dict(cls, data: Any) -> DomainTag:
        required = {"schema_version", "tag_type", "name"}
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("DomainTag data has an unsupported shape")
        if data["schema_version"] != _SCHEMA_VERSION or data["tag_type"] != "domain":
            raise ValueError("DomainTag data uses an unsupported schema")
        result = cls(data["name"])
        if result.to_dict() != dict(data):
            raise ValueError("DomainTag data is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class RectangleBoundaryNames:
    """Names assigned to the four typed boundaries of a rectangle."""

    x_min: str = "x_min"
    x_max: str = "x_max"
    y_min: str = "y_min"
    y_max: str = "y_max"

    def __post_init__(self) -> None:
        for field in ("x_min", "x_max", "y_min", "y_max"):
            object.__setattr__(self, field, _name(
                getattr(self, field), where="RectangleBoundaryNames.%s" % field))
        values = (self.x_min, self.x_max, self.y_min, self.y_max)
        if len(set(values)) != len(values):
            raise ValueError("Rectangle boundary names must be unique")

    def to_dict(self) -> dict[str, str]:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }

    @classmethod
    def from_dict(cls, data: Any) -> RectangleBoundaryNames:
        required = {"x_min", "x_max", "y_min", "y_max"}
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("RectangleBoundaryNames data has an unsupported shape")
        result = cls(**dict(data))
        if result.to_dict() != dict(data):
            raise ValueError("RectangleBoundaryNames data is not canonical")
        return result


class BoundarySide(Enum):
    LOWER = "lower"
    UPPER = "upper"


@dataclass(frozen=True, slots=True)
class DomainBoundary:
    """One stable, geometric boundary of a domain.

    This is intentionally not a post-resolution ``BoundaryHandle``: it has no Case owner and no
    boundary-condition policy.  Resolution can qualify it later without changing its geometry.
    """

    domain_geometry_id: str
    name: str
    axis: CartesianAxis
    side: BoundarySide
    coordinate: float

    def __post_init__(self) -> None:
        if not isinstance(self.domain_geometry_id, str) or not self.domain_geometry_id:
            raise TypeError("DomainBoundary.domain_geometry_id must be non-empty text")
        object.__setattr__(self, "name", _name(self.name, where="DomainBoundary.name"))
        if not isinstance(self.axis, CartesianAxis):
            raise TypeError("DomainBoundary.axis must be a CartesianAxis")
        if not isinstance(self.side, BoundarySide):
            raise TypeError("DomainBoundary.side must be a BoundarySide")
        coordinate = _point((self.coordinate, 0.0), where="DomainBoundary.coordinate")[0]
        object.__setattr__(self, "coordinate", coordinate)

    @property
    def outward_sign(self) -> int:
        return -1 if self.side is BoundarySide.LOWER else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "boundary_type": "rectangle_face",
            "domain_geometry_id": self.domain_geometry_id,
            "name": self.name,
            "axis": self.axis.to_dict(),
            "side": self.side.value,
            "coordinate": self.coordinate,
            "outward_sign": self.outward_sign,
        }

    canonical_identity = to_dict

    @property
    def canonical_id(self) -> str:
        return _identity("domain-boundary", self.to_dict())

    @classmethod
    def from_dict(cls, data: Any) -> DomainBoundary:
        required = {
            "schema_version", "boundary_type", "domain_geometry_id", "name", "axis", "side",
            "coordinate", "outward_sign",
        }
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("DomainBoundary data has an unsupported shape")
        if data["schema_version"] != _SCHEMA_VERSION \
                or data["boundary_type"] != "rectangle_face":
            raise ValueError("DomainBoundary data uses an unsupported schema")
        try:
            side = BoundarySide(data["side"])
        except (TypeError, ValueError) as exc:
            raise ValueError("DomainBoundary side must be 'lower' or 'upper'") from exc
        result = cls(
            data["domain_geometry_id"], data["name"], CartesianAxis.from_dict(data["axis"]),
            side, data["coordinate"],
        )
        if result.to_dict() != dict(data):
            raise ValueError("DomainBoundary data is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class BoundaryPair:
    axis: CartesianAxis
    lower: DomainBoundary
    upper: DomainBoundary

    def __post_init__(self) -> None:
        if not isinstance(self.axis, CartesianAxis):
            raise TypeError("BoundaryPair.axis must be a CartesianAxis")
        if not isinstance(self.lower, DomainBoundary) or not isinstance(self.upper, DomainBoundary):
            raise TypeError("BoundaryPair endpoints must be DomainBoundary objects")
        if self.lower.axis != self.axis or self.upper.axis != self.axis:
            raise ValueError("BoundaryPair endpoints must lie on its axis")
        if self.lower.side is not BoundarySide.LOWER \
                or self.upper.side is not BoundarySide.UPPER:
            raise ValueError("BoundaryPair endpoints have inconsistent sides")

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis.to_dict(),
            "lower": self.lower.to_dict(),
            "upper": self.upper.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RectangleBoundaries:
    x_min: DomainBoundary
    x_max: DomainBoundary
    y_min: DomainBoundary
    y_max: DomainBoundary

    def __post_init__(self) -> None:
        expected = (
            (self.x_min, X_AXIS, BoundarySide.LOWER),
            (self.x_max, X_AXIS, BoundarySide.UPPER),
            (self.y_min, Y_AXIS, BoundarySide.LOWER),
            (self.y_max, Y_AXIS, BoundarySide.UPPER),
        )
        if any(not isinstance(boundary, DomainBoundary) for boundary, _, _ in expected):
            raise TypeError("RectangleBoundaries entries must be DomainBoundary objects")
        if any(boundary.axis != axis or boundary.side is not side
               for boundary, axis, side in expected):
            raise ValueError("RectangleBoundaries entries have inconsistent orientations")
        if len({boundary.name for boundary, _, _ in expected}) != 4:
            raise ValueError("RectangleBoundaries names must be unique")
        if len({boundary.domain_geometry_id for boundary, _, _ in expected}) != 1:
            raise ValueError("RectangleBoundaries entries must belong to one domain geometry")

    @property
    def all(self) -> tuple[DomainBoundary, DomainBoundary, DomainBoundary, DomainBoundary]:
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    def pair(self, axis: CartesianAxis) -> BoundaryPair:
        if not isinstance(axis, CartesianAxis):
            raise TypeError("RectangleBoundaries.pair requires a CartesianAxis, never a string")
        if axis == X_AXIS:
            return BoundaryPair(axis, self.x_min, self.x_max)
        if axis == Y_AXIS:
            return BoundaryPair(axis, self.y_min, self.y_max)
        raise ValueError("axis does not belong to Cartesian2D")

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_min": self.x_min.to_dict(),
            "x_max": self.x_max.to_dict(),
            "y_min": self.y_min.to_dict(),
            "y_max": self.y_max.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> RectangleBoundaries:
        required = {"x_min", "x_max", "y_min", "y_max"}
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("RectangleBoundaries data has an unsupported shape")
        result = cls(*(DomainBoundary.from_dict(data[name]) for name in (
            "x_min", "x_max", "y_min", "y_max")))
        if result.to_dict() != dict(data):
            raise ValueError("RectangleBoundaries data is not canonical")
        return result


@dataclass(frozen=True, slots=True, init=False)
class Rectangle:
    """A bounded, axis-aligned two-dimensional domain."""

    name: str
    lower: tuple[float, float]
    upper: tuple[float, float]
    boundary_names: RectangleBoundaryNames
    tags: tuple[DomainTag, ...]

    def __init__(self, name: Any, lower: Any, upper: Any, *, boundaries: Any = None) -> None:
        checked_name = _name(name, where="Rectangle.name")
        checked_lower = _point(lower, where="Rectangle.lower")
        checked_upper = _point(upper, where="Rectangle.upper")
        if any(hi <= lo for lo, hi in zip(checked_lower, checked_upper, strict=True)):
            raise ValueError("Rectangle.upper must be strictly greater than lower on every axis")
        if boundaries is None:
            checked_boundaries = RectangleBoundaryNames()
        elif isinstance(boundaries, RectangleBoundaryNames):
            checked_boundaries = boundaries
        else:
            raise TypeError(
                "Rectangle.boundaries must be RectangleBoundaryNames, never a mapping or selector")
        object.__setattr__(self, "name", checked_name)
        object.__setattr__(self, "lower", checked_lower)
        object.__setattr__(self, "upper", checked_upper)
        object.__setattr__(self, "boundary_names", checked_boundaries)
        object.__setattr__(self, "tags", ())

    @classmethod
    def _from_parts(
        cls,
        name: str,
        lower: tuple[float, float],
        upper: tuple[float, float],
        boundary_names: RectangleBoundaryNames,
        tags: tuple[DomainTag, ...],
    ) -> Rectangle:
        result = cls(name, lower, upper, boundaries=boundary_names)
        object.__setattr__(result, "tags", tags)
        return result

    @property
    def extent(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self.lower, self.upper)

    @property
    def lengths(self) -> tuple[float, float]:
        return (self.upper[0] - self.lower[0], self.upper[1] - self.lower[1])

    @property
    def geometry_id(self) -> str:
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "geometry_type": "rectangle",
            "name": self.name,
            "lower": list(self.lower),
            "upper": list(self.upper),
            "boundary_names": self.boundary_names.to_dict(),
        }
        return _identity("domain-geometry", payload)

    @property
    def boundaries(self) -> RectangleBoundaries:
        names = self.boundary_names
        return RectangleBoundaries(
            DomainBoundary(self.geometry_id, names.x_min, X_AXIS, BoundarySide.LOWER,
                           self.lower[0]),
            DomainBoundary(self.geometry_id, names.x_max, X_AXIS, BoundarySide.UPPER,
                           self.upper[0]),
            DomainBoundary(self.geometry_id, names.y_min, Y_AXIS, BoundarySide.LOWER,
                           self.lower[1]),
            DomainBoundary(self.geometry_id, names.y_max, Y_AXIS, BoundarySide.UPPER,
                           self.upper[1]),
        )

    def tag(self, tag: Any) -> Rectangle:
        checked = tag if isinstance(tag, DomainTag) else DomainTag(tag)
        tags = tuple(sorted(set(self.tags + (checked,)), key=lambda item: item.name))
        if tags == self.tags:
            return self
        return type(self)._from_parts(
            self.name, self.lower, self.upper, self.boundary_names, tags)

    def frame(self, coordinates: Any) -> RectangleFrame:
        if not isinstance(coordinates, Cartesian2D):
            raise TypeError("Rectangle.frame requires Cartesian2D, never a string or runtime route")
        return RectangleFrame(self, coordinates)

    def preview(
        self,
        *,
        geometry: GeometryPreviewProvider | None = None,
        field: AnalyticPreviewValue | None = None,
        resolution: int | tuple[int, int] = (256, 256),
    ) -> DomainPreview:
        """Sample this domain with optional analytic-field and implicit-geometry overlays."""

        from .preview import preview_domain

        return preview_domain(
            self, geometry=geometry, field=field, resolution=resolution)

    def show(
        self,
        *,
        geometry: GeometryPreviewProvider | None = None,
        field: AnalyticPreviewValue | None = None,
        resolution: int | tuple[int, int] = (256, 256),
        path: str | PathLike[str] | None = None,
    ) -> Path | None:
        """Show this domain interactively, or save it when ``path`` is provided."""

        return self.preview(
            geometry=geometry, field=field, resolution=resolution).show(path=path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "domain_type": "rectangle",
            "name": self.name,
            "lower": list(self.lower),
            "upper": list(self.upper),
            "boundary_names": self.boundary_names.to_dict(),
            "tags": [tag.to_dict() for tag in self.tags],
        }

    canonical_identity = to_dict

    @property
    def canonical_id(self) -> str:
        return _identity("domain", self.to_dict())

    @classmethod
    def from_dict(cls, data: Any) -> Rectangle:
        required = {
            "schema_version", "domain_type", "name", "lower", "upper", "boundary_names", "tags",
        }
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("Rectangle data has an unsupported shape")
        if data["schema_version"] != _SCHEMA_VERSION or data["domain_type"] != "rectangle":
            raise ValueError("Rectangle data uses an unsupported schema")
        raw_tags = data["tags"]
        if not isinstance(raw_tags, list):
            raise TypeError("Rectangle tags must be a canonical list")
        result = cls._from_parts(
            _name(data["name"], where="Rectangle.name"),
            _point(data["lower"], where="Rectangle.lower"),
            _point(data["upper"], where="Rectangle.upper"),
            RectangleBoundaryNames.from_dict(data["boundary_names"]),
            tuple(DomainTag.from_dict(tag) for tag in raw_tags),
        )
        if any(hi <= lo for lo, hi in zip(result.lower, result.upper, strict=True)):
            raise ValueError("Rectangle.upper must be strictly greater than lower on every axis")
        if tuple(sorted(set(result.tags), key=lambda item: item.name)) != result.tags \
                or result.to_dict() != dict(data):
            raise ValueError("Rectangle data is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class RectangleFrame:
    """A rectangle bound to one concrete coordinate frame."""

    domain: Rectangle
    coordinates: Cartesian2D

    def __post_init__(self) -> None:
        if not isinstance(self.domain, Rectangle):
            raise TypeError("RectangleFrame.domain must be a Rectangle")
        if not isinstance(self.coordinates, Cartesian2D):
            raise TypeError("RectangleFrame.coordinates must be Cartesian2D")

    @property
    def axes(self) -> tuple[CartesianAxis, CartesianAxis]:
        return self.coordinates.axes

    @property
    def x(self) -> CartesianAxis:
        return self.coordinates.x

    @property
    def y(self) -> CartesianAxis:
        return self.coordinates.y

    @property
    def boundaries(self) -> RectangleBoundaries:
        return self.domain.boundaries

    @property
    def lower(self) -> tuple[float, float]:
        return self.domain.lower

    @property
    def upper(self) -> tuple[float, float]:
        return self.domain.upper

    @property
    def lengths(self) -> tuple[float, float]:
        return self.domain.lengths

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "frame_type": "rectangle_cartesian_2d",
            "domain": self.domain.to_dict(),
            "coordinates": self.coordinates.to_dict(),
        }

    canonical_identity = to_dict

    @property
    def canonical_id(self) -> str:
        return _identity("domain-frame", self.to_dict())

    @classmethod
    def from_dict(cls, data: Any) -> RectangleFrame:
        required = {"schema_version", "frame_type", "domain", "coordinates"}
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("RectangleFrame data has an unsupported shape")
        if data["schema_version"] != _SCHEMA_VERSION \
                or data["frame_type"] != "rectangle_cartesian_2d":
            raise ValueError("RectangleFrame data uses an unsupported schema")
        result = cls(Rectangle.from_dict(data["domain"]),
                     Cartesian2D.from_dict(data["coordinates"]))
        if result.to_dict() != dict(data):
            raise ValueError("RectangleFrame data is not canonical")
        return result


__all__ = [
    "BoundaryPair", "BoundarySide", "DomainBoundary", "DomainTag", "Rectangle",
    "RectangleBoundaries", "RectangleBoundaryNames", "RectangleFrame",
]


# The annotations are public and must resolve under typing.get_type_hints(). Importing after the
# rectangle definitions avoids a cycle while keeping the preview layer independent from mesh.
from .preview import AnalyticPreviewValue, DomainPreview, GeometryPreviewProvider  # noqa: E402
