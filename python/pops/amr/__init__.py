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
    ResolvedAMRAuthorities,
    Tag,
)
from .resolution import (
    AMRLayoutResolver,
    AMRResolutionContext,
    AMRTaggingResolutionContext,
    ResolvedTaggingAuthority,
    resolve_amr_authorities,
    resolve_tagging,
)
from pops.mesh.amr.transfer import AMRTransfer
from pops.mesh.amr.tagging_graph import ConflictPolicy, EqualityPolicy, Hysteresis


__all__ = [
    "AMRExecution",
    "AMRHierarchy",
    "AMRLayoutResolver",
    "AMRRegrid",
    "AMRResolutionContext",
    "AMRTagging",
    "AMRTaggingResolutionContext",
    "AMRTransfer",
    "Buffer",
    "Coarsen",
    "ConflictPolicy",
    "EqualityPolicy",
    "Hysteresis",
    "ResolvedAMRAuthorities",
    "ResolvedTaggingAuthority",
    "Tag",
    "resolve_amr_authorities",
    "resolve_tagging",
]
