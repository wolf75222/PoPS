"""Supported provider SDK for native AMR materialization extensions.

The extension implements one method and returns the exact immutable IR exported here.  Runtime
consumers depend on that IR, never on the extension's Python class or inheritance hierarchy.
"""

from pops.mesh._amr.hierarchy import CanonicalOptions
from pops.mesh._amr.transfer import (
    NativeAMRActionKind,
    NativeAMRMaterializationCapabilities,
    NativeAMRMaterializationDescriptor,
    NativeAMRMaterializationKind,
    TransferCapabilities,
)


__all__ = [
    "CanonicalOptions",
    "NativeAMRActionKind",
    "NativeAMRMaterializationCapabilities",
    "NativeAMRMaterializationDescriptor",
    "NativeAMRMaterializationKind",
    "TransferCapabilities",
]
