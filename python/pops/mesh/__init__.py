"""Typed mesh descriptors and mesh-local implementation contracts.

``pops.mesh`` describes the discrete domain and the objects the runtime materialises. It
contains no physics and no solver. Layout descriptors live in :mod:`pops.layouts`.

The ordinary public Cartesian path has one spelling: a :class:`CartesianGrid` over a framed
:class:`pops.domain.Rectangle`, with periodic topology expressed by :class:`PeriodicAxes`.
:class:`PolarMesh` remains an advanced, currently supported native annulus descriptor; it is not a
second Cartesian authoring path. Adaptive authoring lives at :mod:`pops.amr`; ``pops.mesh._amr`` is
an implementation package and is deliberately not re-exported here.

Other descriptors:

* mesh: :class:`CartesianGrid`; advanced polar mesh: :class:`PolarMesh`;
  aux halo :class:`AuxHalo`;
  boxes :class:`PatchBox` / :class:`BoxLayout`;
* :mod:`pops.mesh.geometry` -- ``Disc`` / ``HalfPlane`` / ``LevelSet`` / ``EmbeddedBoundary``;
* :mod:`pops.mesh.masks` -- ``NoMask`` / ``Staircase`` / ``CutCell``;
* :mod:`pops.mesh.boundaries` -- ``Periodic`` / ``Physical`` / ``FaceBC`` / face selectors.

Objects are inert authoring values; the runtime materialises grids, patches and halos only after
validation and lowering.
"""
from __future__ import annotations

from ._descriptor import MeshDescriptor
from .grid import CartesianGrid, PeriodicAxes
from .polar import PolarMesh
from .aux import AuxHalo
from .boxes import PatchBox, BoxLayout
from .layout_plan import (
    LayoutHandle, LayoutMappingOperation, LayoutMappingPort, LayoutMappingProvider,
    LayoutMappingRequirement, LayoutRepresentation, LayoutSynchronization,
    LayoutPlan, LayoutPlanBuilder, NormalizedGeometry, NormalizedGeometryProvider,
    normalize_layout_plan)
from .layout_mapping import NativeLayoutMapping
from . import geometry, masks, boundaries

__all__ = [
    "CartesianGrid", "PeriodicAxes", "PolarMesh", "AuxHalo", "PatchBox",
    "BoxLayout",
    "MeshDescriptor",
    "LayoutHandle", "LayoutMappingOperation", "LayoutMappingPort", "LayoutMappingProvider",
    "LayoutMappingRequirement", "LayoutRepresentation", "LayoutSynchronization",
    "LayoutPlan", "LayoutPlanBuilder", "NativeLayoutMapping",
    "NormalizedGeometry", "NormalizedGeometryProvider",
    "normalize_layout_plan",
    "geometry", "masks", "boundaries",
]
