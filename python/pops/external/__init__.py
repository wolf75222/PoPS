"""External source packages and authenticated fixed component artifacts.

A compiled brick shipped in a ``.so`` / manifest, compatible with the PoPS ABI manifests, is
referenced by a typed :class:`CompiledBrickRef` (manifest + native id), never a free string.
The reference resolves to the typed ``external_cpp`` descriptor with the manifest's
requirements / capabilities, so PoPS can validate compatibility before runtime. The
in-process catalog + the low-level loader live in :mod:`pops.descriptors`; this package is the
typed user surface over them.
The final path is ``load(...).require(alias, interface=...)``.  Source packages are inert authoring
inputs; compiled artifacts and installed instances are distinct types and phase registries.
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

# Historical brick-manifest readers remain internal compatibility implementation.  They are not
# used by the package path above and cannot enter either final phase registry.
from .bricks import CompiledBrickRef, ExternalBrick
from .manifests import (register, register_manifest_file, read_manifest,
                        CompiledManifest)
from .artifact_manifest import (CompiledArtifactManifest, build_compiled_manifest,
                                 check_layout_supported, apply_native_manifest,
                                 load_native_manifest, build_compiled_manifest_from_so)
from pops.descriptors import load_cpp_library, load_compiled_manifest, external

from . import bricks, manifests, artifact_manifest

__all__ = [
    "load", "ComponentPackageError", "SourceComponentPackage", "FixedBinaryPackage",
    "ExternalComponentType", "ExternalComponent", "CompiledComponentArtifact",
    "InstalledComponent", "ComponentRuntimeContract", "SourcePackageRegistry",
    "CompiledArtifactRegistry", "build_source_package_manifest", "build_fixed_binary_manifest",
    "CompiledBrickRef", "ExternalBrick",
    "register", "register_manifest_file", "read_manifest", "CompiledManifest",
    "CompiledArtifactManifest", "build_compiled_manifest", "check_layout_supported",
    "apply_native_manifest", "load_native_manifest", "build_compiled_manifest_from_so",
    "load_cpp_library", "load_compiled_manifest", "external",
    "bricks", "manifests", "artifact_manifest",
]
