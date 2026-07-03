"""pops.mesh.geometry -- embedded-geometry descriptors (Spec 5 sec.5.9 / sec.8.16.1).

Typed replacements for the string ``set_disc_domain(..., mode=...)`` form: a geometry
object (Disc / HalfPlane / LevelSet) provides a boundary predicate / level set, and
:class:`EmbeddedBoundary` pairs it with a transport mask. Inert descriptors; the runtime
builds the actual cut-cell / staircase geometry after validation.

Spec 5 sec.8.16 makes the disc DOMAIN itself a typed object: :class:`DiscDomain` carries the
circle ``center`` + ``radius`` AND the transport ``mode`` (a :mod:`pops.mesh.masks` descriptor),
replacing ``sim.set_disc_domain(cx, cy, R, mode="cutcell")`` with
``sim.set_disc_domain(DiscDomain(center=(cx, cy), radius=R, mode=CutCell()))``. The disc / Poisson
walls also lower to the legacy native tokens (``wall="circle"`` + ``wall_radius`` for a
:class:`Disc`, ``wall="none"`` for :class:`NoWall`) so a typed ``set_poisson(wall=...)`` is
byte-identical to the historical string form.

The ``geometry -> masks`` import below is an INTRA-mesh edge (same layer), so it does not add a
cross-layer dependency; the package stays inert and runtime-free at module scope.
"""
from .._descriptor import Availability, MeshDescriptor
from ..masks import lower_disc_mode
from ...descriptors_report import CapabilitySet, RequirementSet


class _Boundary(MeshDescriptor):
    """A handle to the boundary of a geometry (target of a field boundary condition)."""

    category = "geometry_boundary"

    def __init__(self, geometry):
        self.geometry = geometry

    def options(self):
        return {"of": self.geometry.name}


class _Geometry(MeshDescriptor):
    category = "geometry"

    def boundary(self):
        """The boundary of this geometry (e.g. ``Dirichlet(on=wall.boundary())``)."""
        return _Boundary(self)

    def capabilities(self):
        return CapabilitySet({"provides": "level_set"})

    def lower_wall(self):
        """Lower this geometry to the native Poisson wall tokens ``(wall, wall_radius)``.

        Only a disc and a no-wall are wired to the native conducting-wall predicate; the base
        geometry is NOT a Poisson wall. Raising keeps a clear message rather than silently
        emitting an inert wall; subclasses that ARE a wall override this.
        """
        raise TypeError(
            "%s cannot be used as a Poisson wall (the Problem field problem's wall= accepts a "
            "pops.mesh.geometry.Disc / NoWall or the legacy 'circle' / 'none' string)"
            % self.name)


class NoWall(_Geometry):
    """No conducting wall: the elliptic solve sees the full Cartesian square (wall='none')."""

    def capabilities(self):
        return CapabilitySet({"provides": "level_set", "wall": False})

    def lower_wall(self):
        """Lower to the native no-wall tokens (byte-identical to ``wall='none'``)."""
        return ("none", 0.0)


class Disc(_Geometry):
    """A disc wall: center + radius (the embedded boundary is the circle).

    As a Poisson wall it lowers to ``wall="circle"`` + ``wall_radius=radius`` (the native
    conducting-wall predicate is centered at (L/2, L/2); the center is carried for the disc
    TRANSPORT domain, cf. :class:`DiscDomain`).
    """

    def __init__(self, center=(0.0, 0.0), radius=0.5):
        self.center = tuple(float(c) for c in center)
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("Disc: radius must be > 0 (got %r)" % (self.radius,))

    def options(self):
        return {"center": self.center, "radius": self.radius}

    def lower_wall(self):
        """Lower to the native conducting-wall tokens ``("circle", radius)``."""
        return ("circle", self.radius)


class HalfPlane(_Geometry):
    """A half-plane wall: a point on the plane + an outward normal."""

    def __init__(self, point=(0.0, 0.0), normal=(1.0, 0.0)):
        self.point = tuple(float(c) for c in point)
        self.normal = tuple(float(c) for c in normal)

    def options(self):
        return {"point": self.point, "normal": self.normal}


class LevelSet(_Geometry):
    """A generic level-set geometry (the wall is {phi(x) == 0})."""

    def __init__(self, expression):
        self.expression = expression

    def options(self):
        return {"expression": getattr(self.expression, "name", repr(self.expression))}


class DiscDomain(MeshDescriptor):
    """A typed DISC TRANSPORT domain (Spec 5 sec.8.16): center + radius + transport mode.

    The typed replacement for ``sim.set_disc_domain(cx, cy, R, mode="cutcell")``::

        from pops.mesh.geometry import DiscDomain
        from pops.mesh.masks import CutCell
        sim.set_disc_domain(DiscDomain(center=(0.5, 0.5), radius=0.4, mode=CutCell()))

    ``mode`` is a :mod:`pops.mesh.masks` descriptor (``NoMask`` / ``Staircase`` / ``CutCell``)
    OR the legacy string (``"none"`` / ``"staircase"`` / ``"cutcell"``); :meth:`lower` returns
    the ``(cx, cy, R, mode_token)`` tuple the native ``set_disc_domain`` consumes, byte-identical
    to the four-argument string form. Inert: the runtime materialises the mask after validation.
    """

    category = "disc_domain"

    def __init__(self, center=(0.0, 0.0), radius=0.5, mode=None):
        self.center = tuple(float(c) for c in center)
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("DiscDomain: radius must be > 0 (got %r)" % (self.radius,))
        # Default mode = the inert NoMask (full Cartesian transport; only the mask is materialised).
        if mode is None:
            from ..masks import NoMask  # local: avoid importing the class set into this namespace
            mode = NoMask()
        self.mode = mode

    def options(self):
        return {"center": self.center, "radius": self.radius,
                "mode": self.mode if isinstance(self.mode, str) else self.mode.name}

    def capabilities(self):
        return CapabilitySet({"transport_domain": "disc"})

    def requirements(self):
        # A cut-cell disc needs embedded-boundary support; surface the mode's own requirements.
        if isinstance(self.mode, str):
            return RequirementSet()
        return self.mode.requirements()

    def available(self, context=None):
        """Defer to the chosen transport mode's availability (a typed mask explains itself)."""
        if isinstance(self.mode, str):
            return Availability.yes()
        return self.mode.available(context)

    def lower(self, context=None):
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

    def __init__(self, domain, transport):
        self.domain = domain
        self.transport = transport

    def options(self):
        return {"domain": self.domain.name, "transport": self.transport.name}

    def requirements(self):
        return RequirementSet({"embedded_boundary_support": True,
                               "geometry": self.domain.name, "transport_mask": self.transport.name})


__all__ = ["Disc", "NoWall", "HalfPlane", "LevelSet", "DiscDomain", "EmbeddedBoundary"]
