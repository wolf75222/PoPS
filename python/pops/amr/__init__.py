"""Public object-level adaptive-mesh authoring.

``pops.amr`` contains declarations used to build an adaptive layout.  Pre-implemented transfer
kernels remain in ``pops.lib.amr``. Resolved plans remain internal to ``pops.mesh._amr``; the small
immutable native materialization IR is public so external providers can implement its protocol.
"""
from .authoring import (
    AMRClockRelation,
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRRemainderPolicy,
    Buffer,
    Coarsen,
    PatchLayout,
    Tag,
)
from pops.mesh._amr.transfer import AMRTransfer
from pops.mesh._amr import IgnoreAMRCriteria
from pops.mesh._amr.tagging_graph import ConflictPolicy, EqualityPolicy, Hysteresis
from .materialization import (
    CanonicalOptions,
    NativeAMRActionKind,
    NativeAMRMaterializationCapabilities,
    NativeAMRMaterializationDescriptor,
    NativeAMRMaterializationKind,
    TransferCapabilities,
)
from .providers import ClusteringProvider, TaggerProvider


__all__ = [
    "AMRClockRelation",
    "AMRExecution",
    "AMRHierarchy",
    "AMRRegrid",
    "AMRTagging",
    "AMRRemainderPolicy",
    "AMRTransfer",
    "Buffer",
    "CanonicalOptions",
    "Coarsen",
    "ClusteringProvider",
    "ConflictPolicy",
    "EqualityPolicy",
    "Hysteresis",
    "IgnoreAMRCriteria",
    "NativeAMRActionKind",
    "NativeAMRMaterializationCapabilities",
    "NativeAMRMaterializationDescriptor",
    "NativeAMRMaterializationKind",
    "PatchLayout",
    "Tag",
    "TaggerProvider",
    "TransferCapabilities",
]
