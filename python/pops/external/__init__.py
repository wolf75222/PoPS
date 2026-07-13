"""External source packages and authenticated fixed component artifacts.

The sole public path is ``load(...).require(alias, interface=...)``. Source packages are inert
authoring inputs; compiled artifacts and installed instances are distinct authenticated types and
phase registries. A raw shared-library path or historical brick manifest cannot enter this API.
"""
from .packages import (
    ComponentPackageError,
    ExternalComponent,
    ExternalComponentType,
    FixedBinaryPackage,
    SourceComponentPackage,
    build_fixed_binary_manifest,
    build_source_package_manifest,
    load,
)
from .artifacts import (
    CompiledComponentArtifact,
    ComponentRuntimeContract,
    InstalledComponent,
)
from .registries import CompiledArtifactRegistry, SourcePackageRegistry

__all__ = [
    "load", "ComponentPackageError", "SourceComponentPackage", "FixedBinaryPackage",
    "ExternalComponentType", "ExternalComponent", "CompiledComponentArtifact",
    "InstalledComponent", "ComponentRuntimeContract", "SourcePackageRegistry",
    "CompiledArtifactRegistry", "build_source_package_manifest", "build_fixed_binary_manifest",
]
