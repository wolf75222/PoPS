"""Public object-level adaptive-mesh authoring.

``pops.amr`` contains declarations used to build an adaptive layout.  Pre-implemented transfer
kernels remain in ``pops.lib.amr``; resolved provider contracts remain internal to
``pops.mesh.amr``.
"""
from .authoring import (
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    Buffer,
    Coarsen,
    PriorityOrder,
    ResolvedAMRAuthorities,
    Tag,
)
from .resolution import (
    AMRResolutionContext,
    AMRTaggingResolutionContext,
    ResolvedTaggingAuthority,
    resolve_amr_authorities,
    resolve_tagging,
)
from pops.mesh.amr.transfer import AMRTransfer


__all__ = [
    "AMRExecution",
    "AMRHierarchy",
    "AMRRegrid",
    "AMRResolutionContext",
    "AMRTagging",
    "AMRTaggingResolutionContext",
    "AMRTransfer",
    "Buffer",
    "Coarsen",
    "PriorityOrder",
    "ResolvedAMRAuthorities",
    "ResolvedTaggingAuthority",
    "Tag",
    "resolve_amr_authorities",
    "resolve_tagging",
]
