"""pops.mesh.cartesian -- the Cartesian mesh descriptor (Spec 5 sec.5.9).

The CHOICE of geometry lives in a MESH object, not in the scheme: pops.FiniteVolume
stays reconstruction + Riemann flux + variables (no geometry argument). The mesh is
passed to the system via ``pops.System(mesh=...)``. ``CartesianMesh`` is the implicit
default (square domain, numerics STRICTLY unchanged, bit-identical).
"""
from ._descriptor import MeshDescriptor


class CartesianMesh(MeshDescriptor):
    """CARTESIAN mesh (implicit default): square domain [0, L]^2, n x n cells.

    ``pops.System(mesh=pops.mesh.CartesianMesh(n, L, periodic))`` is STRICTLY equivalent
    (bit-identical) to ``pops.System(n=n, L=L, periodic=periodic)``. Provided for symmetry
    with :class:`pops.mesh.PolarMesh` (the geometry choice is explicit on both sides).
    """

    category = "mesh"

    def __init__(self, n=64, L=1.0, periodic=True):
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("CartesianMesh(n=): n must be a Python int >= 1; got %r" % (n,))
        if n < 1:
            raise ValueError("CartesianMesh(n=): n must be >= 1; got %d" % n)
        if isinstance(L, bool) or not isinstance(L, (int, float)):
            raise TypeError("CartesianMesh(L=): L must be a positive real number; got %r" % (L,))
        if float(L) <= 0.0:
            raise ValueError("CartesianMesh(L=): L must be > 0; got %r" % (L,))
        if not isinstance(periodic, bool):
            raise TypeError("CartesianMesh(periodic=): periodic must be a bool; got %r"
                            % (periodic,))
        self.n = int(n)
        self.L = float(L)
        self.periodic = periodic

    def options(self):
        return {"n": self.n, "L": self.L, "periodic": self.periodic}

    def capabilities(self):
        return {"geometry": "cartesian", "dim": 2,
                "periodic": self.periodic, "supports_amr": True}

    def _apply(self, config):
        config.geometry = "cartesian"
        config.n = self.n
        config.L = self.L
        config.periodic = self.periodic
