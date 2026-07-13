"""pops.fields.catalog -- the elliptic-field brick catalog (Spec 5 sec.5.5 / criterion 7).

The generic elliptic-field brick catalog lives under :mod:`pops.fields`. It is DISTINCT from
the physical :class:`pops.fields.FieldOperator` and numerical
:class:`pops.fields.FieldDiscretization`: it is a flat namespace of inert BrickDescriptor
factories (the brick-id metadata the codegen / capability matrix consume).

The default Poisson coupling is solved by ``pops::GeometricMG`` (geometric_mg.hpp); there is no
standalone ``pops::Poisson`` / ``Helmholtz`` / ``FieldSolver`` type yet, so those are catalogued as
planned (``available=False``). The rich typed elliptic-solver descriptor lives in
:class:`pops.solvers.GeometricMG`.
"""

from __future__ import annotations

from types import SimpleNamespace

from pops.descriptors import _native

fields = SimpleNamespace(
    GeometricMG=lambda **o: _native(
        "geometric_mg", "pops::GeometricMG", "geometric_mg", category="field", **o
    ),
)

__all__ = ["fields"]
