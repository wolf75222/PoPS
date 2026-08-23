"""Typed mesh descriptors and mesh-local implementation contracts.

``pops.mesh`` describes the discrete domain and the objects the runtime materialises. It
contains no physics and no solver. Layout descriptors live in :mod:`pops.layouts`.

The ordinary public Cartesian path has one spelling: a :class:`CartesianGrid` over a bounded
Cartesian frame.  :class:`pops.domain.CartesianDomain` infers rank 1, 2 or 3 from its bounds;
periodic topology is expressed by :class:`PeriodicAxes`.
:class:`PolarMesh` remains an inert annular geometry/output descriptor; the exact-ranked native
runtime accepts Cartesian coordinate providers only and refuses it during resolution. Adaptive
authoring lives at :mod:`pops.amr`; ``pops.mesh._amr`` is an implementation package and is
deliberately not re-exported here.

Other descriptors:

* executable mesh: :class:`CartesianGrid`; annular geometry/output: :class:`PolarMesh`;
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
from .grid import CartesianGrid, PeriodicAxes, RegularBlocks
from .polar import PolarMesh
from .aux import AuxHalo
from .boxes import PatchBox, BoxLayout
from .layout_plan import (
    LayoutHandle,
    LayoutMappingOperation,
    LayoutMappingPort,
    LayoutMappingProvider,
    LayoutMappingRequirement,
    LayoutRepresentation,
    LayoutSynchronization,
    LayoutPlan,
    LayoutPlanBuilder,
    NativeSpatialLayout,
    NormalizedGeometry,
    NormalizedGeometryProvider,
    normalize_layout_plan,
)
from .layout_mapping import NativeLayoutMapping
from . import geometry, masks, boundaries

__all__ = [
    "CartesianGrid",
    "PeriodicAxes",
    "RegularBlocks",
    "PolarMesh",
    "AuxHalo",
    "PatchBox",
    "BoxLayout",
    "MeshDescriptor",
    "LayoutHandle",
    "LayoutMappingOperation",
    "LayoutMappingPort",
    "LayoutMappingProvider",
    "LayoutMappingRequirement",
    "LayoutRepresentation",
    "LayoutSynchronization",
    "LayoutPlan",
    "LayoutPlanBuilder",
    "NativeLayoutMapping",
    "NativeSpatialLayout",
    "NormalizedGeometry",
    "NormalizedGeometryProvider",
    "normalize_layout_plan",
    "geometry",
    "masks",
    "boundaries",
]
