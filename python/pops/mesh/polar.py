"""pops.mesh.polar -- the global annular polar mesh descriptor (Spec 5 sec.5.9 / sec.8.16.1).

``PolarMesh`` describes a global ring r in [r_min, r_max] x theta in [0, 2pi), nr x ntheta
cells. theta is PERIODIC, r carries a PHYSICAL boundary. Axis convention: direction 0 =
radial, direction 1 = azimuthal (cf. PolarGeometry / assemble_rhs_polar on the C++ side).
"""
from __future__ import annotations

from typing import Any

from ._descriptor import MeshDescriptor
from ..descriptors_report import CapabilitySet
from pops.params.use_sites import ParamUse, resolve_param_use
from pops.runtime_environment import NATIVE_DIMENSION, validate_dimension


class PolarMesh(MeshDescriptor):
    """GLOBAL ANNULAR POLAR mesh: r in [r_min, r_max] x theta in [0, 2pi), nr x ntheta cells.

    theta is PERIODIC, r carries a PHYSICAL boundary (wall / outlet). The polar path is
    wired into ``System.step`` (polar transport + polar Poisson + aux drift in the local
    basis). Declared capabilities (Spec 5 sec.8.16.1):

    - polar TRANSPORT: scalar ExB AND isothermal fluid; Riemann flux 'rusanov' (all
      transport) AND 'hll' (isothermal fluid only); 'hllc'/'roe' rejected (no polar Euler
      brick);
    - DIRECT polar Poisson (PolarPoissonSolver): single-rank, single-box only;
    - tensorial polar condensed Program (via pops.lib.time.CondensedSchur): multi-rank/multi-box.

    No AMR, no Cartesian<->polar coupling. ``theta_boxes`` splits the transport into
    azimuthal bands (1 = single-box, bit-identical to history; must divide ntheta).
    """

    category = "mesh"

    def __init__(self, r_min: Any, r_max: Any, nr: Any, ntheta: Any, theta_boxes: Any = 1,
                 *, dim: Any = NATIVE_DIMENSION) -> None:
        self.dim = validate_dimension(dim, where="PolarMesh")
        r_min = resolve_param_use(
            r_min, ParamUse.MESH_EXTENT, where="PolarMesh(r_min=)")
        r_max = resolve_param_use(
            r_max, ParamUse.MESH_EXTENT, where="PolarMesh(r_max=)")
        nr = resolve_param_use(nr, ParamUse.SHAPE, where="PolarMesh(nr=)")
        ntheta = resolve_param_use(
            ntheta, ParamUse.SHAPE, where="PolarMesh(ntheta=)")
        theta_boxes = resolve_param_use(
            theta_boxes, ParamUse.MESH_TOPOLOGY, where="PolarMesh(theta_boxes=)")
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
            raise ValueError("PolarMesh: theta_boxes <= ntheta (at least one azimuthal cell per band)")
        if int(ntheta) % tb != 0:
            raise ValueError("PolarMesh: theta_boxes must DIVIDE ntheta (equal azimuthal bands)")
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.nr = int(nr)
        self.ntheta = int(ntheta)
        self.theta_boxes = tb

    def options(self) -> dict:
        return {"r_min": self.r_min, "r_max": self.r_max, "nr": self.nr,
                "ntheta": self.ntheta, "theta_boxes": self.theta_boxes}

    def capabilities(self) -> Any:
        return CapabilitySet({
            "geometry": "polar",
            "dim": self.dim,
            "scalar_transport": True,
            "isothermal_fluid": True,
            "compressible_euler": False,
            "amr": False,
            "direct_poisson_mpi": False,
            "multibox_transport": self.theta_boxes > 1,
        })

    def _apply(self, config: Any) -> None:
        config.geometry = "polar"
        config.nr = self.nr
        config.ntheta = self.ntheta
        config.r_min = self.r_min
        config.r_max = self.r_max
        config.theta_boxes = self.theta_boxes
        config.n = self.nr  # n serves as the default size for the rest of the config (diagnostics)
