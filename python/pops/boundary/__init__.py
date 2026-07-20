"""Public boundary authoring authorities.

Geometry names never select implementations here: a typed geometric boundary is associated with
an immutable condition, then ``DiscretizationPlan`` resolves that declaration against the selected
spatial methods and the canonical Case ownership graph.
"""

from .transport import (
    BoundaryStencilRequirement,
    TransportBoundarySet,
)
from .embedded import EmbeddedBoundaryFlux, ZeroFlux

__all__ = [
    "BoundaryStencilRequirement",
    "EmbeddedBoundaryFlux",
    "TransportBoundarySet",
    "ZeroFlux",
]
