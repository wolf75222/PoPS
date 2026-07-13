"""Typed embedded-geometry and wall descriptors.

:class:`DiscDomain` owns the complete transport-domain selection: geometry plus a typed
:class:`~pops.mesh.masks.TransportMask`. :class:`Disc` and :class:`NoWall` are the typed wall
selectors consumed by the elliptic runtime seam. Native tokens are lowering details and are never
accepted as public authoring input.
"""
from __future__ import annotations

import math
from typing import Any

from .._descriptor import MeshDescriptor
from ..masks import TransportMask, lower_disc_mode
from ...descriptors_report import CapabilitySet, RequirementSet
from pops.params.use_sites import ParamUse, resolve_param_use


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


class _Boundary(MeshDescriptor):
    """A handle to the boundary of a geometry (target of a field boundary condition)."""

    category = "geometry_boundary"

    def __init__(self, geometry: Any) -> None:
        self.geometry = geometry

    def options(self) -> dict:
        return {"of": self.geometry.name}


class Geometry(MeshDescriptor):
    """Extensible descriptor interface for an embedded geometry."""

    category = "geometry"

    def boundary(self) -> _Boundary:
        """The boundary of this geometry (e.g. ``Dirichlet(on=wall.boundary())``)."""
        return _Boundary(self)

    def capabilities(self) -> Any:
        return CapabilitySet({"provides": "level_set"})

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

    def lower_wall(self) -> Any:
        """Lower to the private native no-wall representation."""
        return ("none", 0.0)


class Disc(Geometry):
    """A disc geometry and the centered circular-wall selector.

    ``center=None`` means the center of the owning domain and is the only form supported by the
    current elliptic wall provider. An explicit center remains valid embedded-geometry metadata but
    :meth:`lower_wall` rejects it instead of silently discarding it. A transport disc with an
    explicit center is represented by :class:`DiscDomain`.
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


class HalfPlane(Geometry):
    """A half-plane wall: a point on the plane + an outward normal."""

    def __init__(self, point: Any = (0.0, 0.0), normal: Any = (1.0, 0.0)) -> None:
        self.point = _geometry_coordinates(point, where="HalfPlane(point=)")
        self.normal = _geometry_coordinates(normal, where="HalfPlane(normal=)")

    def options(self) -> dict:
        return {"point": self.point, "normal": self.normal}


class LevelSet(Geometry):
    """A generic level-set geometry (the wall is {phi(x) == 0})."""

    def __init__(self, expression: Any) -> None:
        self.expression = expression

    def options(self) -> dict:
        return {"expression": getattr(self.expression, "name", repr(self.expression))}


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
        lower_disc_mode(mode)
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
        return (cx, cy, self.radius, lower_disc_mode(self.mode))


class EmbeddedBoundary(MeshDescriptor):
    """An embedded boundary = a geometry + a transport mask (Spec 5 sec.8.16.1).

    Passed to a layout: ``Uniform(mesh, embedded_boundary=EmbeddedBoundary(wall, CutCell()))``.
    Declares it needs embedded-boundary support in the spatial scheme + a compatible
    field/boundary route; the runtime materialises the masked transport.
    """

    category = "mesh_feature"

    def __init__(self, domain: Any, transport: Any) -> None:
        if not isinstance(domain, Geometry):
            raise TypeError(
                "EmbeddedBoundary: domain must be a pops.mesh.geometry.Geometry descriptor, got %s"
                % type(domain).__name__)
        if not isinstance(transport, TransportMask):
            raise TypeError(
                "EmbeddedBoundary: transport must be a pops.mesh.masks.TransportMask descriptor, "
                "got %s" % type(transport).__name__)
        lower_disc_mode(transport)
        self.domain = domain
        self.transport = transport

    def options(self) -> dict:
        return {"domain": self.domain.name, "transport": self.transport.name}

    def requirements(self) -> Any:
        return RequirementSet({"embedded_boundary_support": True,
                               "geometry": self.domain.name, "transport_mask": self.transport.name})


__all__ = [
    "Geometry", "Disc", "NoWall", "HalfPlane", "LevelSet", "DiscDomain",
    "EmbeddedBoundary",
]
