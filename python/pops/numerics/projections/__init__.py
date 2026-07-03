"""pops.numerics.projections -- the cell-wise projection brick catalog (Spec 3 / Spec 5).

Positivity is the pops::zhang_shu_scale free function (positivity.hpp); the others have
no native symbol yet (a generated brick or a planned native type).

Spec 5 (sec.4) homes these post-step projection bricks under ``pops.numerics`` (formerly
``pops.lib.operators``).
"""
from __future__ import annotations

from types import SimpleNamespace

from pops.descriptors import _native, _planned, BrickDescriptor

projections = SimpleNamespace(
    positivity=lambda **o: _native("positivity", "pops::zhang_shu_scale", "positivity",
                                   category="projection", **o),
    bound_preserving=lambda **o: _planned("bound_preserving", "bound_preserving",
                                          category="projection", **o),
    conservative_fix=lambda **o: BrickDescriptor(
        "conservative_fix", "generated", category="projection",
        scheme="conservative_fix", options=o or None),
    divergence_cleaning=lambda **o: BrickDescriptor(
        "divergence_cleaning", "generated", category="projection",
        scheme="divergence_cleaning", options=o or None),
)

# Spec 5: expose the projections at module scope.
positivity = projections.positivity
bound_preserving = projections.bound_preserving
conservative_fix = projections.conservative_fix
divergence_cleaning = projections.divergence_cleaning

__all__ = ["projections", "positivity", "bound_preserving",
           "conservative_fix", "divergence_cleaning"]
