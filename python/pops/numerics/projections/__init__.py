"""pops.numerics.projections -- the cell-wise projection brick catalog (Spec 3 / Spec 5).

Positivity is the ``pops::zhang_shu_scale`` native function. Generated conservative and
divergence projections carry executable builders; unavailable placeholders are not exported.

Spec 5 (sec.4) homes these post-step projection bricks under ``pops.numerics`` (formerly
``pops.lib.operators``).
"""
from __future__ import annotations

from types import SimpleNamespace

from pops.descriptors import _native, BrickDescriptor

projections = SimpleNamespace(
    positivity=lambda **o: _native("positivity", "pops::zhang_shu_scale", "positivity",
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
conservative_fix = projections.conservative_fix
divergence_cleaning = projections.divergence_cleaning

__all__ = ["projections", "positivity",
           "conservative_fix", "divergence_cleaning"]
