"""Rank-generic bounded Cartesian domains.

The length of ``lower`` and ``upper`` is the sole authoring dimension authority.  No independent
``dim=`` selector can disagree with the geometry, grid or native specialization selected later.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from pops.frames import Cartesian, CartesianAxis
from pops.identity import make_identity
from pops.identity.semantic import semantic_value

from .rectangle import BoundaryPair, BoundarySide, DomainBoundary, DomainTag


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


def _point(value: Any, *, where: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("%s must be a coordinate sequence" % where)
    raw = tuple(value)
    if len(raw) not in (1, 2, 3):
        raise ValueError("%s must contain one, two, or three coordinates" % where)
    result = []
    for index, coordinate in enumerate(raw):
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise TypeError("%s[%d] must be a real number, never bool" % (where, index))
        converted = float(coordinate)
        if not math.isfinite(converted):
            raise ValueError("%s[%d] must be finite" % (where, index))
        result.append(converted)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CartesianBoundaryNames:
    """Canonical lower/upper name pair for every domain axis."""

    pairs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pairs, tuple) or len(self.pairs) not in (1, 2, 3):
            raise TypeError("CartesianBoundaryNames.pairs must have rank 1, 2, or 3")
        normalized = []
        for axis, pair in enumerate(self.pairs):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(
                    "CartesianBoundaryNames.pairs[%d] must be a lower/upper tuple" % axis
                )
            normalized.append(tuple(
                _name(item, where="CartesianBoundaryNames.pairs[%d]" % axis)
                for item in pair
            ))
        flattened = tuple(item for pair in normalized for item in pair)
        if len(set(flattened)) != len(flattened):
            raise ValueError("Cartesian boundary names must be unique")
        object.__setattr__(self, "pairs", tuple(normalized))

    @classmethod
    def defaults(cls, dimension: int) -> CartesianBoundaryNames:
        if dimension not in (1, 2, 3):
            raise ValueError("Cartesian boundary-name dimension must be 1, 2, or 3")
        names = ("x", "y", "z")[:dimension]
        return cls(tuple(("%s_min" % name, "%s_max" % name) for name in names))

    def to_dict(self) -> dict[str, Any]:
        return {"pairs": [list(pair) for pair in self.pairs]}

    @classmethod
    def from_dict(cls, data: Any) -> CartesianBoundaryNames:
        if not isinstance(data, Mapping) or set(data) != {"pairs"} \
                or not isinstance(data["pairs"], list):
            raise TypeError("CartesianBoundaryNames data has an unsupported shape")
        result = cls(tuple(tuple(pair) for pair in data["pairs"]))
        if result.to_dict() != dict(data):
            raise ValueError("CartesianBoundaryNames data is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class CartesianBoundaries:
    """Ordered boundary pairs for one rank-generic Cartesian domain."""

    pairs: tuple[BoundaryPair, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pairs, tuple) or len(self.pairs) not in (1, 2, 3) \
                or any(not isinstance(pair, BoundaryPair) for pair in self.pairs):
            raise TypeError("CartesianBoundaries.pairs must contain 1, 2, or 3 BoundaryPair values")
        if tuple(pair.axis.index for pair in self.pairs) != tuple(range(len(self.pairs))):
            raise ValueError("CartesianBoundaries must follow canonical x,y,z axis order")
        boundaries = self.all
        if len({boundary.name for boundary in boundaries}) != len(boundaries):
            raise ValueError("Cartesian boundary names must be unique")
        if len({boundary.domain_geometry_id for boundary in boundaries}) != 1:
            raise ValueError("Cartesian boundaries must belong to one domain geometry")

    @property
    def all(self) -> tuple[DomainBoundary, ...]:
        return tuple(boundary for pair in self.pairs for boundary in (pair.lower, pair.upper))

    def pair(self, axis: CartesianAxis) -> BoundaryPair:
        if not isinstance(axis, CartesianAxis):
            raise TypeError("CartesianBoundaries.pair requires a CartesianAxis")
        if axis.index >= len(self.pairs) or self.pairs[axis.index].axis != axis:
            raise ValueError("axis does not belong to this Cartesian domain")
        return self.pairs[axis.index]

    def to_dict(self) -> dict[str, Any]:
        return {"pairs": [pair.to_dict() for pair in self.pairs]}

    @classmethod
    def from_dict(cls, data: Any) -> CartesianBoundaries:
        if not isinstance(data, Mapping) or set(data) != {"pairs"} \
                or not isinstance(data["pairs"], list):
            raise TypeError("CartesianBoundaries data has an unsupported shape")
        pairs = []
        for raw in data["pairs"]:
            if not isinstance(raw, Mapping) or set(raw) != {"axis", "lower", "upper"}:
                raise TypeError("Cartesian boundary pair data has an unsupported shape")
            axis = CartesianAxis.from_dict(raw["axis"])
            pairs.append(BoundaryPair(
                axis,
                DomainBoundary.from_dict(raw["lower"]),
                DomainBoundary.from_dict(raw["upper"]),
            ))
        result = cls(tuple(pairs))
        if result.to_dict() != dict(data):
            raise ValueError("CartesianBoundaries data is not canonical")
        return result


@dataclass(frozen=True, slots=True, init=False)
class CartesianDomain:
    """A bounded Cartesian box whose vector rank is its dimension authority."""

    name: str
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    boundary_names: CartesianBoundaryNames
    tags: tuple[DomainTag, ...]

    def __init__(self, name: Any, lower: Any, upper: Any, *, boundaries: Any = None) -> None:
        checked_name = _name(name, where="CartesianDomain.name")
        checked_lower = _point(lower, where="CartesianDomain.lower")
        checked_upper = _point(upper, where="CartesianDomain.upper")
        if len(checked_lower) != len(checked_upper):
            raise ValueError("CartesianDomain lower and upper must have one common rank")
        if any(hi <= lo for lo, hi in zip(checked_lower, checked_upper, strict=True)):
            raise ValueError("CartesianDomain.upper must exceed lower on every axis")
        if boundaries is None:
            checked_boundaries = CartesianBoundaryNames.defaults(len(checked_lower))
        elif isinstance(boundaries, CartesianBoundaryNames):
            checked_boundaries = boundaries
        else:
            raise TypeError("CartesianDomain.boundaries must be CartesianBoundaryNames")
        if len(checked_boundaries.pairs) != len(checked_lower):
            raise ValueError("CartesianDomain boundary names must have the domain rank")
        object.__setattr__(self, "name", checked_name)
        object.__setattr__(self, "lower", checked_lower)
        object.__setattr__(self, "upper", checked_upper)
        object.__setattr__(self, "boundary_names", checked_boundaries)
        object.__setattr__(self, "tags", ())

    @classmethod
    def _from_parts(
        cls,
        name: str,
        lower: tuple[float, ...],
        upper: tuple[float, ...],
        boundary_names: CartesianBoundaryNames,
        tags: tuple[DomainTag, ...],
    ) -> CartesianDomain:
        result = cls(name, lower, upper, boundaries=boundary_names)
        object.__setattr__(result, "tags", tags)
        return result

    @property
    def dimension(self) -> int:
        return len(self.lower)

    @property
    def coordinates(self) -> Cartesian:
        return Cartesian(self.dimension)

    @property
    def extent(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return (self.lower, self.upper)

    @property
    def lengths(self) -> tuple[float, ...]:
        return tuple(hi - lo for lo, hi in zip(self.lower, self.upper, strict=True))

    @property
    def geometry_id(self) -> str:
        return _identity("domain-geometry", {
            "schema_version": _SCHEMA_VERSION,
            "geometry_type": "cartesian_box",
            "name": self.name,
            "lower": list(self.lower),
            "upper": list(self.upper),
            "boundary_names": self.boundary_names.to_dict(),
        })

    @property
    def boundaries(self) -> CartesianBoundaries:
        pairs = []
        for axis, names in zip(self.coordinates.axes, self.boundary_names.pairs, strict=True):
            pairs.append(BoundaryPair(
                axis,
                DomainBoundary(
                    self.geometry_id, names[0], axis, BoundarySide.LOWER, self.lower[axis.index]
                ),
                DomainBoundary(
                    self.geometry_id, names[1], axis, BoundarySide.UPPER, self.upper[axis.index]
                ),
            ))
        return CartesianBoundaries(tuple(pairs))

    def tag(self, tag: Any) -> CartesianDomain:
        checked = tag if isinstance(tag, DomainTag) else DomainTag(tag)
        tags = tuple(sorted(set(self.tags + (checked,)), key=lambda item: item.name))
        if tags == self.tags:
            return self
        return type(self)._from_parts(
            self.name, self.lower, self.upper, self.boundary_names, tags
        )

    def frame(self) -> CartesianDomainFrame:
        return CartesianDomainFrame(self, self.coordinates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "domain_type": "cartesian_box",
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
    def from_dict(cls, data: Any) -> CartesianDomain:
        required = {
            "schema_version", "domain_type", "name", "lower", "upper", "boundary_names", "tags",
        }
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("CartesianDomain data has an unsupported shape")
        if data["schema_version"] != _SCHEMA_VERSION \
                or data["domain_type"] != "cartesian_box" or not isinstance(data["tags"], list):
            raise ValueError("CartesianDomain data uses an unsupported schema")
        result = cls._from_parts(
            _name(data["name"], where="CartesianDomain.name"),
            _point(data["lower"], where="CartesianDomain.lower"),
            _point(data["upper"], where="CartesianDomain.upper"),
            CartesianBoundaryNames.from_dict(data["boundary_names"]),
            tuple(DomainTag.from_dict(tag) for tag in data["tags"]),
        )
        if tuple(sorted(set(result.tags), key=lambda item: item.name)) != result.tags \
                or result.to_dict() != dict(data):
            raise ValueError("CartesianDomain data is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class CartesianDomainFrame:
    """One bounded Cartesian domain bound to its inferred rank-specific frame."""

    domain: CartesianDomain
    coordinates: Cartesian

    def __post_init__(self) -> None:
        if not isinstance(self.domain, CartesianDomain):
            raise TypeError("CartesianDomainFrame.domain must be a CartesianDomain")
        if not isinstance(self.coordinates, Cartesian):
            raise TypeError("CartesianDomainFrame.coordinates must be Cartesian")
        if self.coordinates.dimension != self.domain.dimension:
            raise ValueError("CartesianDomainFrame coordinate and domain ranks differ")

    @property
    def axes(self) -> tuple[CartesianAxis, ...]:
        return self.coordinates.axes

    @property
    def boundaries(self) -> CartesianBoundaries:
        return self.domain.boundaries

    @property
    def lower(self) -> tuple[float, ...]:
        return self.domain.lower

    @property
    def upper(self) -> tuple[float, ...]:
        return self.domain.upper

    @property
    def lengths(self) -> tuple[float, ...]:
        return self.domain.lengths

    @property
    def canonical_id(self) -> str:
        return _identity("domain-frame", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "frame_type": "cartesian_box",
            "domain": self.domain.to_dict(),
            "coordinates": self.coordinates.to_dict(),
        }

    canonical_identity = to_dict

    @classmethod
    def from_dict(cls, data: Any) -> CartesianDomainFrame:
        required = {"schema_version", "frame_type", "domain", "coordinates"}
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("CartesianDomainFrame data has an unsupported shape")
        if data["schema_version"] != _SCHEMA_VERSION or data["frame_type"] != "cartesian_box":
            raise ValueError("CartesianDomainFrame data uses an unsupported schema")
        result = cls(
            CartesianDomain.from_dict(data["domain"]),
            Cartesian.from_dict(data["coordinates"]),
        )
        if result.to_dict() != dict(data):
            raise ValueError("CartesianDomainFrame data is not canonical")
        return result


__all__ = [
    "CartesianBoundaries",
    "CartesianBoundaryNames",
    "CartesianDomain",
    "CartesianDomainFrame",
]
