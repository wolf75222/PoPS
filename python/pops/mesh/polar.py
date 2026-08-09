"""pops.mesh.polar -- the global annular polar mesh descriptor (Spec 5 sec.5.9 / sec.8.16.1).

``PolarMesh`` describes a global ring r in [r_min, r_max] x theta in [0, 2pi), nr x ntheta
cells. theta is PERIODIC, r carries a PHYSICAL boundary. The descriptor remains useful to
normalize annular geometry and scientific-output measures; the final exact-ranked native runtime
accepts Cartesian coordinate providers only and refuses this descriptor before artifact creation.
"""

from __future__ import annotations

import math
from typing import Any

from ._descriptor import MeshDescriptor
from ..descriptors_report import CapabilitySet
from pops.params.use_sites import ParamUse, resolve_param_use

from ._layout_plan_contracts import (
    NormalizedGeometry,
    POLAR_ANNULUS_2D_COORDINATES,
    POLAR_ANNULUS_CELL_AREA,
)


class PolarMesh(MeshDescriptor):
    """GLOBAL ANNULAR POLAR mesh: r in [r_min, r_max] x theta in [0, 2pi), nr x ntheta cells.

    theta is PERIODIC and r carries a PHYSICAL boundary. This is an inert geometry/output
    descriptor, not a native execution route: ``pops.resolve`` rejects its non-Cartesian
    coordinate provider before compilation, and the private ``System(mesh=...)`` bypass has been
    retired. Standalone polar numerical kernels remain available to C++ algorithm tests.

    ``theta_boxes`` records an azimuthal decomposition for deterministic geometry inspection; it
    must divide ``ntheta``. It does not opt the descriptor into native execution.
    """

    category = "mesh"
    axis_names = ("r", "theta")

    def __init__(self, r_min: Any, r_max: Any, nr: Any, ntheta: Any, theta_boxes: Any = 1) -> None:
        self.dim = len(self.axis_names)
        r_min = resolve_param_use(r_min, ParamUse.MESH_EXTENT, where="PolarMesh(r_min=)")
        r_max = resolve_param_use(r_max, ParamUse.MESH_EXTENT, where="PolarMesh(r_max=)")
        nr = resolve_param_use(nr, ParamUse.SHAPE, where="PolarMesh(nr=)")
        ntheta = resolve_param_use(ntheta, ParamUse.SHAPE, where="PolarMesh(ntheta=)")
        theta_boxes = resolve_param_use(
            theta_boxes, ParamUse.MESH_TOPOLOGY, where="PolarMesh(theta_boxes=)"
        )
        if not (r_max > r_min >= 0.0):
            raise ValueError("PolarMesh: requires r_max > r_min >= 0 (ring)")
        # nr >= 3: the radial drift uses a 2nd-order ONE-SIDED stencil at both walls.
        if nr < 3:
            raise ValueError("PolarMesh: nr >= 3 (2nd-order one-sided radial stencil at the walls)")
        if ntheta < 1:
            raise ValueError("PolarMesh: ntheta >= 1")
        tb = int(theta_boxes)
        if tb < 1:
            raise ValueError("PolarMesh: theta_boxes >= 1 (1 = single-box)")
        if tb > int(ntheta):
            raise ValueError(
                "PolarMesh: theta_boxes <= ntheta (at least one azimuthal cell per band)"
            )
        if int(ntheta) % tb != 0:
            raise ValueError("PolarMesh: theta_boxes must DIVIDE ntheta (equal azimuthal bands)")
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.nr = int(nr)
        self.ntheta = int(ntheta)
        self.theta_boxes = tb

    def options(self) -> dict:
        return {
            "r_min": self.r_min,
            "r_max": self.r_max,
            "nr": self.nr,
            "ntheta": self.ntheta,
            "theta_boxes": self.theta_boxes,
        }

    def capabilities(self) -> Any:
        return CapabilitySet(
            {
                "geometry": "polar",
                "dim": self.dim,
                "native_execution": False,
                "scientific_output_geometry": True,
                "amr": False,
            }
        )

    def normalized_geometry(self) -> NormalizedGeometry:
        """Project exact annular coordinates and the physical polar cell-area measure."""
        return NormalizedGeometry(
            coordinate_system=POLAR_ANNULUS_2D_COORDINATES,
            cell_measure=POLAR_ANNULUS_CELL_AREA,
            axis_names=self.axis_names,
            lower=(self.r_min, 0.0),
            upper=(self.r_max, math.tau),
            cells=(self.nr, self.ntheta),
        )

    def native_spatial_data(self) -> dict[str, Any]:
        """Exact annular periodicity and authored azimuthal-band decomposition."""
        band = self.ntheta // self.theta_boxes
        return {
            "schema_version": 1,
            "periodicity": [False, True],
            "centering": "cell",
            "decomposition": {
                "schema_version": 1,
                "kind": "axis_bands",
                "axis": 1,
                "boxes": [
                    {
                        "lower": [0, index * band],
                        "upper_exclusive": [self.nr, (index + 1) * band],
                    }
                    for index in range(self.theta_boxes)
                ],
            },
        }
