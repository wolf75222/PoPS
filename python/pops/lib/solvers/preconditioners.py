"""pops.lib.solvers.preconditioners -- the preconditioner brick catalog (Spec 3).

Only the geometric-multigrid preconditioner has a native type; identity/jacobi/
block-jacobi have none yet (the polar solver has its own PolarPrecond enum).
"""
from types import SimpleNamespace

from ..descriptors import _native, _planned, _external_descriptor

preconditioners = SimpleNamespace(
    Identity=lambda: _planned("identity", "identity", category="preconditioner"),
    Jacobi=lambda: _planned("jacobi", "jacobi", category="preconditioner"),
    BlockJacobi=lambda: _planned("block_jacobi", "block_jacobi",
                                 category="preconditioner"),
    GeometricMG=lambda **o: _native("geometric_mg", "pops::GeometricMG", "geometric_mg",
                                    category="preconditioner", **o),
    User=lambda brick_id: _external_descriptor(brick_id, expect_category="preconditioner"),
)

__all__ = ["preconditioners"]
