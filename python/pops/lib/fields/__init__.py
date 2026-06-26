"""pops.lib.fields -- the elliptic-field brick catalog (Spec 3).

The default Poisson coupling is solved by pops::GeometricMG (geometric_mg.hpp); there
is no standalone pops::Poisson / Helmholtz / FieldSolver type yet, so those are
catalogued as planned (available=False).
"""
from types import SimpleNamespace

from ..descriptors import _native, _planned

fields = SimpleNamespace(
    Poisson=lambda **o: _planned("poisson", "poisson", category="field", **o),
    Helmholtz=lambda **o: _planned("helmholtz", "helmholtz", category="field", **o),
    EllipticSolve=lambda **o: _planned("elliptic_solve", "elliptic",
                                       category="field", **o),
    GeometricMG=lambda **o: _native("geometric_mg", "pops::GeometricMG", "geometric_mg",
                                    category="field", **o),
)

__all__ = ["fields"]
