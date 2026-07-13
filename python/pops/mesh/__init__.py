"""pops.mesh -- typed mesh / geometry / layout / AMR descriptors (Spec 5 sec.5.9-5.11).

``pops.mesh`` describes the discrete domain and the objects the runtime materialises. It
contains no physics and no solver. It is the sole public home of mesh, layout, AMR and geometry
descriptors:

* meshes: :class:`CartesianMesh`, :class:`PolarMesh`; aux halo :class:`AuxHalo`;
  boxes :class:`PatchBox` / :class:`BoxLayout`;
* :mod:`pops.mesh.layouts` -- ``Uniform`` / ``AMR``;
* :mod:`pops.mesh.amr` -- ``PatchLayout`` / ``RegridEvery`` / ``Refine`` / ``TagUnion`` / ...;
* :mod:`pops.mesh.geometry` -- ``Disc`` / ``HalfPlane`` / ``LevelSet`` / ``EmbeddedBoundary``;
* :mod:`pops.mesh.masks` -- ``NoMask`` / ``Staircase`` / ``CutCell``;
* :mod:`pops.mesh.boundaries` -- ``Periodic`` / ``Physical`` / ``FaceBC`` / face selectors.

Every object is an inert :class:`pops.mesh._descriptor.MeshDescriptor`; the C++ runtime
materialises the actual grids, patches and halos after validation.
"""
from __future__ import annotations

from ._descriptor import Availability, MeshDescriptor
from .cartesian import CartesianMesh
from .grid import CartesianGrid
from .polar import PolarMesh
from .aux import AuxHalo
from .boxes import PatchBox, BoxLayout
from .layout_plan import (
    LayoutHandle, LayoutMappingProvider, LayoutPlan, LayoutPlanBuilder, normalize_layout_plan)
from . import layouts, amr, geometry, masks, boundaries

__all__ = [
    "CartesianGrid", "CartesianMesh", "PolarMesh", "AuxHalo", "PatchBox", "BoxLayout",
    "Availability", "MeshDescriptor",
    "LayoutHandle", "LayoutMappingProvider", "LayoutPlan", "LayoutPlanBuilder",
    "normalize_layout_plan",
    "layouts", "amr", "geometry", "masks", "boundaries",
]
