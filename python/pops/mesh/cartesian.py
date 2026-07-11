"""pops.mesh.cartesian -- the Cartesian mesh descriptor (Spec 5 sec.5.9).

The CHOICE of geometry lives in a MESH object, not in the scheme: pops.FiniteVolume
stays reconstruction + Riemann flux + variables (no geometry argument). The mesh is
passed to the system via ``pops.runtime.system.System(mesh=...) (advanced seam)``. ``CartesianMesh`` is the implicit
default (square domain, numerics STRICTLY unchanged, bit-identical).
"""
from __future__ import annotations

from typing import Any

from ._descriptor import MeshDescriptor
from ..descriptors_report import CapabilitySet
from pops.params.use_sites import ParamUse, resolve_param_use
from pops.runtime_environment import NATIVE_DIMENSION, validate_dimension


class CartesianMesh(MeshDescriptor):
    """CARTESIAN mesh (implicit default): square domain [0, L]^2, n x n cells.

    ``System(mesh=pops.mesh.CartesianMesh(n, L, periodic))`` (advanced seam) is STRICTLY equivalent
    (bit-identical) to ``System(n=n, L=L, periodic=periodic)``. Provided for symmetry
    with :class:`pops.mesh.PolarMesh` (the geometry choice is explicit on both sides).
    """

    category = "mesh"

    def __init__(self, n: Any = 64, L: Any = 1.0, periodic: Any = True,
                 *, dim: Any = NATIVE_DIMENSION) -> None:
        self.dim = validate_dimension(dim, where="CartesianMesh")
        self.n = int(resolve_param_use(n, ParamUse.SHAPE, where="CartesianMesh(n=)"))
        self.L = float(resolve_param_use(
            L, ParamUse.MESH_EXTENT, where="CartesianMesh(L=)"))
        self.periodic = bool(resolve_param_use(
            periodic, ParamUse.MESH_TOPOLOGY, where="CartesianMesh(periodic=)"))

    def options(self) -> dict:
        return {"n": self.n, "L": self.L, "periodic": self.periodic}

    def capabilities(self) -> Any:
        return CapabilitySet({"geometry": "cartesian", "dim": self.dim,
                              "periodic": self.periodic, "supports_amr": True})

    def _apply(self, config: Any) -> None:
        config.geometry = "cartesian"
        config.n = self.n
        config.L = self.L
        config.periodic = self.periodic
