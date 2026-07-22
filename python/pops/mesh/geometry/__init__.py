"""Typed embedded-geometry and wall descriptors.

Every embedded transport boundary lowers through the same :class:`LevelSet` contract and a typed
:class:`~pops.mesh.masks.TransportMask`. :class:`Disc` and :class:`NoWall` also remain the typed
selectors consumed by the current elliptic wall seam. Native tokens are lowering details and are
never accepted as public authoring input.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

from .._descriptor import MeshDescriptor
from ..masks import TransportMask, lower_transport_mask
from ...descriptors_report import CapabilitySet, RequirementSet
from pops.analytic import (
    ScalarExpr,
    constant,
    coordinates,
    hypot,
    maximum,
    minimum,
)
from pops.frames import Cartesian2D
from pops.params.use_sites import ParamUse, resolve_param_use


_LEVEL_SET_SCHEMA_VERSION = 1


def _geometry_scalar(value: Any, *, where: str) -> float:
    """Resolve a structural geometry coordinate before numeric coercion."""
    raw = resolve_param_use(value, ParamUse.MESH_EXTENT, where=where)
    if isinstance(raw, bool):
        raise TypeError("%s must be a real number, not bool" % where)
    resolved = float(raw)
    if not math.isfinite(resolved):
        raise ValueError("%s must be finite (got %r)" % (where, value))
    return resolved


def _geometry_coordinates(values: Any, *, where: str) -> tuple[float, ...]:
    """Resolve every coordinate independently so no parameter can hide in a tuple."""
    values = resolve_param_use(values, ParamUse.MESH_EXTENT, where=where)
    return tuple(
        _geometry_scalar(value, where="%s[%d]" % (where, index))
        for index, value in enumerate(values)
    )


def _cartesian_coordinates(frame: Any) -> tuple[ScalarExpr, ScalarExpr]:
    """Return analytic coordinates after authenticating a Cartesian authoring frame."""
    coordinate_system = getattr(frame, "coordinates", frame)
    if not isinstance(coordinate_system, Cartesian2D):
        raise TypeError("level-set geometry requires a typed Cartesian2D frame")
    return coordinates(frame)


def _frame_center(frame: Any) -> tuple[float, float]:
    lower = getattr(frame, "lower", None)
    upper = getattr(frame, "upper", None)
    if lower is None or upper is None:
        raise ValueError(
            "Disc(center=None).level_set(frame) requires a bounded frame exposing lower and upper"
        )
    checked_lower = _geometry_coordinates(lower, where="Disc.level_set(frame.lower)")
    checked_upper = _geometry_coordinates(upper, where="Disc.level_set(frame.upper)")
    if len(checked_lower) != 2 or len(checked_upper) != 2:
        raise ValueError("Disc.level_set frame bounds must contain exactly two coordinates")
    if any(high <= low for low, high in zip(checked_lower, checked_upper, strict=True)):
        raise ValueError("Disc.level_set frame upper bounds must be greater than lower bounds")
    return tuple((low + high) * 0.5 for low, high in zip(
        checked_lower, checked_upper, strict=True))  # type: ignore[return-value]


class Geometry(MeshDescriptor):
    """Extensible descriptor interface for an embedded geometry."""

    category = "geometry"

    def __bool__(self) -> bool:
        raise TypeError(
            "Geometry has no Python truth value; compose implicit geometries with &, |, - and ~"
        )

    def __or__(self, other: Any) -> GeometryComposition:
        return union(self, other)

    def __and__(self, other: Any) -> GeometryComposition:
        return intersection(self, other)

    def __sub__(self, other: Any) -> GeometryComposition:
        return difference(self, other)

    def __invert__(self) -> GeometryComposition:
        return complement(self)

    def union(self, *others: Any) -> GeometryComposition:
        return union(self, *others)

    def intersection(self, *others: Any) -> GeometryComposition:
        return intersection(self, *others)

    def difference(self, other: Any) -> GeometryComposition:
        return difference(self, other)

    def complement(self) -> GeometryComposition:
        return complement(self)

    def capabilities(self) -> Any:
        return CapabilitySet({"provides": "level_set"})

    def level_set(self, frame: Any) -> LevelSet:
        """Resolve this geometry to the generic implicit-surface contract."""
        del frame
        raise TypeError(
            "%s does not implement level_set(frame); embedded geometries must lower to a "
            "pops.mesh.geometry.LevelSet" % self.name
        )

    def lower_wall(self) -> Any:
        """Lower this geometry to the native Poisson wall tokens ``(wall, wall_radius)``.

        Only a disc and a no-wall are wired to the native conducting-wall predicate; the base
        geometry is NOT a Poisson wall. Raising keeps a clear message rather than silently
        emitting an inert wall; subclasses that ARE a wall override this.
        """
        raise TypeError(
            "%s cannot be used as a Poisson wall; wall= requires "
            "pops.mesh.geometry.Disc or NoWall"
            % self.name)


class NoWall(Geometry):
    """No conducting wall: the elliptic solve sees the full Cartesian domain."""

    def capabilities(self) -> Any:
        return CapabilitySet({"provides": "level_set", "wall": False})

    def level_set(self, frame: Any) -> LevelSet:
        """Return the all-active level set after authenticating the Cartesian frame."""
        _cartesian_coordinates(frame)
        return LevelSet(constant(-1.0))

    def lower_wall(self) -> Any:
        """Lower to the private native no-wall representation."""
        return ("none", 0.0)


class Disc(Geometry):
    """A disc geometry and the centered circular-wall selector.

    ``center=None`` means the center of the owning domain and is the only form supported by the
    current elliptic wall provider. An explicit center remains valid embedded-geometry metadata but
    :meth:`lower_wall` rejects it instead of silently discarding it. Transport uses this same
    geometry through ``EmbeddedBoundary(Disc(...), transport, boundary_flux)``.
    """

    def __init__(self, center: Any = None, radius: Any = 0.5) -> None:
        self.center = (None if center is None else
                       _geometry_coordinates(center, where="Disc(center=)"))
        if self.center is not None and len(self.center) != 2:
            raise ValueError("Disc: center must contain exactly two coordinates")
        self.radius = _geometry_scalar(radius, where="Disc(radius=)")
        if self.radius <= 0.0:
            raise ValueError("Disc: radius must be > 0 (got %r)" % (self.radius,))

    def options(self) -> dict:
        return {"center": self.center, "radius": self.radius}

    def lower_wall(self) -> Any:
        """Lower to the native conducting-wall tokens ``("circle", radius)``."""
        if self.center is not None:
            raise ValueError(
                "Disc used as a Poisson wall must use center=None (the owning domain center); "
                "an explicit center is not supported by the native wall provider")
        return ("circle", self.radius)

    def level_set(self, frame: Any) -> LevelSet:
        """Bind this disc to ``frame`` with the convention ``phi < 0`` inside."""
        x_value, y_value = _cartesian_coordinates(frame)
        center = self.center if self.center is not None else _frame_center(frame)
        cx, cy = center
        return LevelSet(hypot(x_value - cx, y_value - cy) - self.radius)


class HalfPlane(Geometry):
    """A half-plane wall: a point on the plane + an outward normal."""

    def __init__(self, point: Any = (0.0, 0.0), normal: Any = (1.0, 0.0)) -> None:
        self.point = _geometry_coordinates(point, where="HalfPlane(point=)")
        self.normal = _geometry_coordinates(normal, where="HalfPlane(normal=)")
        if len(self.point) != 2 or len(self.normal) != 2:
            raise ValueError("HalfPlane point and normal must contain exactly two coordinates")
        if self.normal == (0.0, 0.0):
            raise ValueError("HalfPlane normal must be non-zero")

    def options(self) -> dict:
        return {"point": self.point, "normal": self.normal}

    def level_set(self, frame: Any) -> LevelSet:
        """Bind this half-plane to ``frame``; the side opposite the normal is active."""
        x_value, y_value = _cartesian_coordinates(frame)
        px, py = self.point
        nx, ny = self.normal
        return LevelSet((x_value - px) * nx + (y_value - py) * ny)


class LevelSet(Geometry):
    """A generic analytic geometry with active region ``phi(x) < 0``."""

    def __init__(self, expression: Any) -> None:
        if type(expression) is not ScalarExpr:
            raise TypeError("LevelSet expression must be a pops.analytic.ScalarExpr")
        expression.validate()
        if expression.has_parameters:
            raise NotImplementedError(
                "parameterized LevelSet geometry is not supported: uniform layout geometry is "
                "signed during resolve, before bind values exist. Use a parameter-free LevelSet; "
                "support requires a distinct bind-authenticated geometry-plan contract."
            )
        if expression.input_references():
            raise NotImplementedError(
                "input-dependent LevelSet geometry is not supported: geometry installation has "
                "no discrete input table. Use a coordinate-only LevelSet; dynamic geometry "
                "requires a distinct native geometry-update contract."
            )
        self.expression = expression

    def options(self) -> dict:
        return {
            "active_when": "phi<0",
            "expression": self.expression.to_data(),
        }

    @property
    def frame_id(self) -> str | None:
        return self.expression.frame_id

    def level_set(self, frame: Any) -> LevelSet:
        """Authenticate this already-generic level set against its owning frame."""
        frame_id = getattr(frame, "canonical_id", None)
        if not isinstance(frame_id, str) or not frame_id:
            raise TypeError("LevelSet.level_set(frame) requires a canonical physical frame")
        if self.frame_id not in (None, frame_id):
            raise ValueError("embedded LevelSet and supplied layout use different frames")
        return self

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": _LEVEL_SET_SCHEMA_VERSION,
            "geometry_type": "level_set",
            "active_when": "phi<0",
            "expression": self.expression.to_data(),
        }

    canonical_identity = to_data

    @classmethod
    def from_data(cls, data: Any) -> LevelSet:
        required = {"schema_version", "geometry_type", "active_when", "expression"}
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("LevelSet data has an unsupported shape")
        if type(data["schema_version"]) is not int \
                or data["schema_version"] != _LEVEL_SET_SCHEMA_VERSION \
                or data["geometry_type"] != "level_set" \
                or data["active_when"] != "phi<0":
            raise ValueError("LevelSet data uses an unsupported schema or sign convention")
        result = cls(ScalarExpr.from_data(data["expression"]))
        if result.to_data() != dict(data):
            raise ValueError("LevelSet data is not canonical")
        return result

def _geometry(value: Any, *, where: str) -> Geometry:
    if not isinstance(value, Geometry):
        raise TypeError("%s requires Geometry operands" % where)
    return value


def _balanced_level_set_composition(
    geometries: tuple[LevelSet, ...], combine: Any,
) -> LevelSet:
    """Compose an ordered collection without making expression depth linear in its size."""
    expressions = tuple(geometry.expression for geometry in geometries)
    while len(expressions) > 1:
        next_level = [
            combine(expressions[index], expressions[index + 1])
            for index in range(0, len(expressions) - 1, 2)
        ]
        if len(expressions) % 2:
            next_level.append(expressions[-1])
        expressions = tuple(next_level)
    return LevelSet(cast(tuple[ScalarExpr], expressions)[0])


class GeometryComposition(Geometry):
    """Deferred, frame-independent CSG descriptor over ``Geometry.level_set``."""

    _ARITY = {"union": None, "intersection": None, "difference": 2, "complement": 1}

    def __init__(self, operation: Any, operands: Any) -> None:
        if operation not in self._ARITY:
            raise ValueError("unsupported geometry composition %r" % operation)
        checked = tuple(_geometry(item, where=operation) for item in operands)
        arity = self._ARITY[operation]
        if (arity is None and len(checked) < 2) or (arity is not None and len(checked) != arity):
            raise ValueError("geometry %s has invalid arity" % operation)
        self.operation = operation
        self.operands = checked

    def options(self) -> dict[str, Any]:
        return {"operation": self.operation, "operands": self.operands}

    def level_set(self, frame: Any) -> LevelSet:
        resolved = tuple(item.level_set(frame) for item in self.operands)
        if any(type(item) is not LevelSet for item in resolved):
            raise TypeError("Geometry.level_set(frame) must return an exact LevelSet")
        if self.operation == "union":
            return _balanced_level_set_composition(resolved, minimum)
        if self.operation == "intersection":
            return _balanced_level_set_composition(resolved, maximum)
        if self.operation == "difference":
            return LevelSet(maximum(resolved[0].expression, -resolved[1].expression))
        return LevelSet(-resolved[0].expression)


def _flatten_geometry(operation: str, geometries: tuple[Geometry, ...]) -> tuple[Geometry, ...]:
    flattened = []
    for geometry in geometries:
        if type(geometry) is GeometryComposition and geometry.operation == operation:
            flattened.extend(geometry.operands)
        else:
            flattened.append(geometry)
    return tuple(flattened)


def union(first: Any, second: Any, *others: Any) -> GeometryComposition:
    """Build a deferred union ``min(phi_1, ..., phi_n) < 0`` for any Geometry provider."""
    geometries = tuple(
        _geometry(item, where="union") for item in (first, second, *others)
    )
    return GeometryComposition("union", _flatten_geometry("union", geometries))


def intersection(first: Any, second: Any, *others: Any) -> GeometryComposition:
    """Build a deferred intersection ``max(phi_1, ..., phi_n) < 0``."""
    geometries = tuple(
        _geometry(item, where="intersection") for item in (first, second, *others)
    )
    return GeometryComposition(
        "intersection", _flatten_geometry("intersection", geometries),
    )


def difference(left: Any, right: Any) -> GeometryComposition:
    """Build deferred CSG subtraction ``max(phi_left, -phi_right) < 0``."""
    return GeometryComposition(
        "difference",
        (_geometry(left, where="difference"), _geometry(right, where="difference")),
    )


def complement(geometry: Any) -> GeometryComposition:
    """Build the deferred complement ``-phi < 0``."""
    return GeometryComposition("complement", (_geometry(geometry, where="complement"),))


class DiscDomain(MeshDescriptor):
    """A typed DISC TRANSPORT domain (Spec 5 sec.8.16): center + radius + transport mode.

    ``mode`` must implement :class:`pops.mesh.masks.TransportMask`; strings are rejected at
    construction. Inert: the runtime materialises the mask only after validation.
    """

    category = "disc_domain"

    def __init__(self, center: Any = (0.0, 0.0), radius: Any = 0.5, mode: Any = None) -> None:
        self.center = _geometry_coordinates(center, where="DiscDomain(center=)")
        if len(self.center) != 2:
            raise ValueError(
                "DiscDomain: center must contain exactly two coordinates (got %d)"
                % len(self.center))
        self.radius = _geometry_scalar(radius, where="DiscDomain(radius=)")
        if self.radius <= 0.0:
            raise ValueError("DiscDomain: radius must be > 0 (got %r)" % (self.radius,))
        # Default mode = the inert NoMask (full Cartesian transport; only the mask is materialised).
        if mode is None:
            from ..masks import NoMask  # local: avoid importing the class set into this namespace
            mode = NoMask()
        lower_transport_mask(mode)
        self.mode = mode

    def options(self) -> dict:
        return {"center": self.center, "radius": self.radius, "mode": self.mode.name}

    def capabilities(self) -> Any:
        return CapabilitySet({"transport_domain": "disc"})

    def requirements(self) -> Any:
        return self.mode.requirements()

    def available(self, context: Any = None) -> Any:
        """Defer to the chosen transport mode's explainable availability."""
        return self.mode.available(context)

    def lower(self, context: Any = None) -> Any:
        """Lower to the native ``(cx, cy, R, mode_token)`` set_disc_domain arguments."""
        cx, cy = self.center
        return (cx, cy, self.radius, lower_transport_mask(self.mode))


class EmbeddedBoundary(MeshDescriptor):
    """An embedded boundary = geometry + transport metrics + an explicit boundary flux.

    Passed to a layout as
    ``Uniform(mesh, embedded_boundary=EmbeddedBoundary(wall, Staircase(), ZeroFlux()))``.
    Declares it needs embedded-boundary support in the spatial scheme + a compatible
    field/boundary route; the runtime materialises the masked transport.
    """

    category = "mesh_feature"

    def __init__(self, domain: Any, transport: Any, boundary: Any) -> None:
        if not isinstance(domain, Geometry):
            raise TypeError(
                "EmbeddedBoundary: domain must be a pops.mesh.geometry.Geometry descriptor, got %s"
                % type(domain).__name__)
        if not isinstance(transport, TransportMask):
            raise TypeError(
                "EmbeddedBoundary: transport must be a pops.mesh.masks.TransportMask descriptor, "
                "got %s" % type(transport).__name__)
        lower_transport_mask(transport)
        from pops.boundary.embedded import lower_embedded_boundary_flux

        lower_embedded_boundary_flux(boundary)
        self.domain = domain
        self.transport = transport
        self.boundary = boundary

    def options(self) -> dict:
        return {
            "domain": self.domain.name,
            "transport": self.transport.name,
            "boundary": self.boundary.name,
        }

    def semantic_data(self) -> dict[str, Any]:
        """Preserve the complete geometry and transport policy in scientific identity."""
        return {
            "kind": "embedded_boundary",
            "domain": self.domain,
            "transport": self.transport,
            "boundary": self.boundary,
        }

    def level_set(self, frame: Any) -> LevelSet:
        """Resolve any Geometry implementation through one small implicit-surface interface."""
        result = self.domain.level_set(frame)
        if type(result) is not LevelSet:
            raise TypeError("Geometry.level_set(frame) must return an exact LevelSet")
        return result

    def requirements(self) -> Any:
        return RequirementSet({
            "embedded_boundary_support": True,
            "geometry": self.domain.name,
            "transport_mask": self.transport.name,
            "boundary_flux": self.boundary.name,
        })


__all__ = [
    "Geometry", "GeometryComposition", "Disc", "NoWall", "HalfPlane", "LevelSet",
    "DiscDomain",
    "EmbeddedBoundary", "complement", "difference", "intersection", "union",
]
