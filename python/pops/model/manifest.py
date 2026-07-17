"""Public, import-stable manifest surface.

The immutable row types and read-only builders live in focused implementation modules so the
authentication schema, module payload, and compiler-facing builders remain independently auditable.
"""

from ._manifest_builders import (
    build_module_manifest,
    condensed_route_manifest,
    coupling_operator_manifest,
    module_manifest_of,
)
from ._module_manifest import SCHEMA_VERSION, ModuleManifest
from ._operator_manifest import OperatorManifestEntry, OperatorRegistryManifest

__all__ = [
    "ModuleManifest",
    "OperatorManifestEntry",
    "OperatorRegistryManifest",
    "SCHEMA_VERSION",
    "build_module_manifest",
    "condensed_route_manifest",
    "coupling_operator_manifest",
    "module_manifest_of",
]
