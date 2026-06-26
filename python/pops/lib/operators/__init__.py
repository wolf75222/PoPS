"""pops.lib.operators -- the cell-wise projection brick catalog (Spec 3).

Positivity is the pops::zhang_shu_scale free function (positivity.hpp); the others have
no native symbol yet (a generated brick or a planned native type).

DEFER (no catalogued source): ``operators.local_source`` / ``operators.local_matrix`` /
``operators.split`` -- not in the Spec-3 catalog, so creating them would invent surface
(see the PR-D blueprint DEFER list).
"""
from types import SimpleNamespace

from ..descriptors import _native, _planned, BrickDescriptor

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

__all__ = ["projections"]
