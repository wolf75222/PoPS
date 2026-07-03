"""pops.fields.catalog -- the elliptic-field brick catalog (Spec 5 sec.5.5 / criterion 7).

Spec 5 (criterion 7) homes the generic elliptic-field brick catalog under the top-level
:mod:`pops.fields` package, moving it out of the transitional ``pops.lib.fields``. ``pops.lib``
keeps only presets. This catalog is DISTINCT from the typed ``FieldProblem`` authoring surface of
:mod:`pops.fields`: it is a flat namespace of inert :class:`pops.descriptors.BrickDescriptor`
factories (the brick-id metadata the codegen / capability matrix consume).

The default Poisson coupling is solved by ``pops::GeometricMG`` (geometric_mg.hpp); there is no
standalone ``pops::Poisson`` / ``Helmholtz`` / ``FieldSolver`` type yet, so those are catalogued as
planned (``available=False``). The rich typed elliptic-solver descriptor lives in
:class:`pops.solvers.GeometricMG`.
"""
from __future__ import annotations

from types import SimpleNamespace

from pops.descriptors import _native, _planned

fields = SimpleNamespace(
    Poisson=lambda **o: _planned("poisson", "poisson", category="field", **o),
    Helmholtz=lambda **o: _planned("helmholtz", "helmholtz", category="field", **o),
    EllipticSolve=lambda **o: _planned("elliptic_solve", "elliptic",
                                       category="field", **o),
    GeometricMG=lambda **o: _native("geometric_mg", "pops::GeometricMG", "geometric_mg",
                                    category="field", **o),
)

__all__ = ["fields"]
