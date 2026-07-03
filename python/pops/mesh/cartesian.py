"""pops.mesh.cartesian -- the Cartesian mesh descriptor (Spec 5 sec.5.9).

The CHOICE of geometry lives in a MESH object, not in the scheme: pops.FiniteVolume
stays reconstruction + Riemann flux + variables (no geometry argument). The mesh is
passed to the system via ``pops.runtime.system.System(mesh=...) (advanced seam)``. ``CartesianMesh`` is the implicit
default (square domain, numerics STRICTLY unchanged, bit-identical).
"""
from ._descriptor import MeshDescriptor
from ..descriptors_report import CapabilitySet
from pops.runtime_environment import NATIVE_DIMENSION, validate_dimension


class CartesianMesh(MeshDescriptor):
    """CARTESIAN mesh (implicit default): square domain [0, L]^2, n x n cells.

    ``System(mesh=pops.mesh.CartesianMesh(n, L, periodic))`` (advanced seam) is STRICTLY equivalent
    (bit-identical) to ``System(n=n, L=L, periodic=periodic)``. Provided for symmetry
    with :class:`pops.mesh.PolarMesh` (the geometry choice is explicit on both sides).
    """

    category = "mesh"

    def __init__(self, n=64, L=1.0, periodic=True, *, dim=NATIVE_DIMENSION):
        self.dim = validate_dimension(dim, where="CartesianMesh")
        self.n = int(n)
        self.L = float(L)
        self.periodic = bool(periodic)

    def options(self):
        return {"n": self.n, "L": self.L, "periodic": self.periodic}

    def capabilities(self):
        return CapabilitySet({"geometry": "cartesian", "dim": self.dim,
                              "periodic": self.periodic, "supports_amr": True})

    def _apply(self, config):
        config.geometry = "cartesian"
        config.n = self.n
        config.L = self.L
        config.periodic = self.periodic
