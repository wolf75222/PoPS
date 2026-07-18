"""pops.fields.nullspace -- typed nullspace declarations for a field solve (Spec 5 sec.5.5).

A pure-Neumann / fully periodic elliptic operator has a non-trivial nullspace. This module
declares the mathematical kernel only. Selecting a representative solution is a distinct
typed gauge choice in :mod:`pops.fields.gauges`.

Inert descriptors; they compute nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import CapabilitySet

from ._identity import field_identity


def _handle(value: Any, *, where: str, kind: str) -> Any:
    from pops.model import Handle

    if isinstance(value, str) or not isinstance(value, Handle) or not value.is_resolved:
        raise TypeError("%s requires a canonical Handle" % where)
    if value.kind != kind:
        raise TypeError("%s requires Handle.kind=%r" % (where, kind))
    return value


class ConstantNullspace(Descriptor):
    """The constant-function nullspace of a pure-Neumann / periodic elliptic operator.

    Declaring it names the kernel. It neither changes the RHS nor chooses a gauge.
    """

    category = "nullspace"

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_frozen":
            super().__setattr__(key, value)
            return
        raise AttributeError("ConstantNullspace has no configurable fields")

    def options(self) -> dict:
        return {"nullspace": "constant"}

    def to_data(self) -> dict:
        return {"type": type(self).__name__, "options": self.options()}

    def capabilities(self) -> Any:
        return CapabilitySet({"constant_kernel": True, "rhs_projection": False})

    def _program_prepared_nullspace(self) -> tuple[Any, dict[str, Any]]:
        """Bind the builtin through the same registered provider protocol as extensions."""
        from ._prepared_nullspace_registry import constant_prepared_nullspace_provider

        return constant_prepared_nullspace_provider(), {}

    def _prepared_field_nullspace(self) -> tuple[Any, dict[str, Any]]:
        """Bind field installation through its topology-aware provider protocol."""
        from ._prepared_field_nullspace_builtins import constant_field_nullspace_provider

        return constant_field_nullspace_provider(), {}


class PreparedNullspace(Descriptor):
    """A nullspace declaration backed by one registered native preparation provider.

    ``options`` are inert authoring data.  The provider snapshots and validates them when a
    :class:`pops.linalg.LinearProblem` is constructed, then emits a real C++
    ``FieldNullspacePlan`` from its authenticated native component.
    """

    category = "nullspace"

    def __init__(self, provider: Any, **options: Any) -> None:
        from ._prepared_nullspace_registry import (
            PreparedNullspaceProvider,
            prepared_nullspace_provider_by_emitter_id,
        )

        if type(provider) is not PreparedNullspaceProvider:
            raise TypeError("PreparedNullspace requires an exact registered Provider")
        registered = prepared_nullspace_provider_by_emitter_id(provider.emitter_id)
        if registered is not provider:
            raise ValueError("PreparedNullspace provider is not the registered authority")
        self.provider = provider
        self.provider_options = dict(options)

    def options(self) -> dict[str, Any]:
        return dict(self.provider_options)

    def to_data(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "provider": self.provider.authority(),
            "options": self.options(),
        }

    def _program_prepared_nullspace(self) -> tuple[Any, dict[str, Any]]:
        return self.provider, self.options()


class PreparedFieldNullspace(Descriptor):
    """A topology-aware field nullspace backed by one registered native provider."""

    category = "nullspace"

    def __init__(self, provider: Any, **options: Any) -> None:
        from ._prepared_field_nullspace_registry import (
            PreparedFieldNullspaceProvider,
            prepared_field_nullspace_provider_by_resolver_id,
        )

        if type(provider) is not PreparedFieldNullspaceProvider:
            raise TypeError("PreparedFieldNullspace requires an exact registered FieldProvider")
        if prepared_field_nullspace_provider_by_resolver_id(provider.resolver_id) is not provider:
            raise ValueError("PreparedFieldNullspace provider is not the registered authority")
        self.provider = provider
        self.provider_options = dict(options)

    def options(self) -> dict[str, Any]:
        return dict(self.provider_options)

    def to_data(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "provider": self.provider.authority(),
            "options": self.options(),
        }

    def _prepared_field_nullspace(self) -> tuple[Any, dict[str, Any]]:
        return self.provider, self.options()


@dataclass(frozen=True, slots=True)
class ConnectedComponentsManifest:
    handle: Any
    components: tuple[Any, ...]

    def __post_init__(self) -> None:
        _handle(self.handle, where="ConnectedComponentsManifest.handle",
                kind="connected_components_manifest")
        if not isinstance(self.components, tuple) or not self.components:
            raise TypeError("ConnectedComponentsManifest.components must be a non-empty tuple")
        rows = tuple(_handle(row, where="ConnectedComponentsManifest.components",
                             kind="connected_component") for row in self.components)
        if len(rows) != len(set(rows)):
            raise ValueError("connected component identities must be unique")
        object.__setattr__(self, "components", tuple(sorted(
            rows, key=lambda row: row.qualified_id)))

    def to_data(self) -> dict[str, Any]:
        return {"handle": self.handle.canonical_identity(),
                "components": [row.canonical_identity() for row in self.components]}


@dataclass(frozen=True, slots=True)
class NullspaceBasisVector:
    handle: Any
    component: Any

    def __post_init__(self) -> None:
        _handle(self.handle, where="NullspaceBasisVector.handle", kind="nullspace_basis")
        _handle(self.component, where="NullspaceBasisVector.component",
                kind="connected_component")

    def to_data(self) -> dict[str, Any]:
        return {"handle": self.handle.canonical_identity(),
                "component": self.component.canonical_identity()}


@dataclass(frozen=True, slots=True)
class NullspaceBasis:
    manifest: ConnectedComponentsManifest
    vectors: tuple[NullspaceBasisVector, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ConnectedComponentsManifest):
            raise TypeError("NullspaceBasis.manifest must be a ConnectedComponentsManifest")
        if not isinstance(self.vectors, tuple) or any(
                not isinstance(row, NullspaceBasisVector) for row in self.vectors):
            raise TypeError("NullspaceBasis.vectors must contain NullspaceBasisVector objects")
        components = [row.component for row in self.vectors]
        if len(components) != len(set(components)):
            raise ValueError("nullspace basis requires exactly one vector per component")
        if set(components) != set(self.manifest.components):
            raise ValueError("nullspace basis must cover exactly every connected component")
        object.__setattr__(self, "vectors", tuple(sorted(
            self.vectors, key=lambda row: row.component.qualified_id)))

    @property
    def identity(self) -> Any:
        return field_identity("nullspace-basis", self.to_data())

    def to_data(self) -> dict[str, Any]:
        return {"manifest": self.manifest.to_data(),
                "vectors": [row.to_data() for row in self.vectors]}


@dataclass(frozen=True, slots=True)
class RHSCompatibilityEvidence:
    component: Any
    compatible: bool
    witness: Any

    def __post_init__(self) -> None:
        _handle(self.component, where="RHSCompatibilityEvidence.component",
                kind="connected_component")
        if not isinstance(self.compatible, bool):
            raise TypeError("RHSCompatibilityEvidence.compatible must be bool")
        _handle(self.witness, where="RHSCompatibilityEvidence.witness",
                kind="rhs_compatibility_witness")

    def to_data(self) -> dict[str, Any]:
        return {"component": self.component.canonical_identity(),
                "compatible": self.compatible,
                "witness": self.witness.canonical_identity()}


@dataclass(frozen=True, slots=True)
class NullspaceCompatibility:
    basis: NullspaceBasis
    evidence: tuple[RHSCompatibilityEvidence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.basis, NullspaceBasis):
            raise TypeError("NullspaceCompatibility.basis must be a NullspaceBasis")
        if not isinstance(self.evidence, tuple) or any(
                not isinstance(row, RHSCompatibilityEvidence) for row in self.evidence):
            raise TypeError("NullspaceCompatibility.evidence has invalid rows")
        components = [row.component for row in self.evidence]
        if len(components) != len(set(components)) or set(components) != set(
                self.basis.manifest.components):
            raise ValueError("RHS compatibility requires exactly one proof per component")
        incompatible = [row.component.qualified_id for row in self.evidence
                        if not row.compatible]
        if incompatible:
            raise ValueError(
                "RHS is incompatible with nullspace components %s; silent projection is forbidden"
                % sorted(incompatible))
        object.__setattr__(self, "evidence", tuple(sorted(
            self.evidence, key=lambda row: row.component.qualified_id)))

    def to_data(self) -> dict[str, Any]:
        return {"basis": self.basis.to_data(),
                "evidence": [row.to_data() for row in self.evidence],
                "rhs_projection": "forbidden"}


from ._prepared_nullspace_registry import (  # noqa: E402 -- descriptors precede builtin registry
    PreparedNullspaceContracts,
    PreparedNullspaceNativeEmission,
    PreparedNullspaceProvider,
    PreparedNullspaceUse,
    PreparedNullspaceUsePolicy,
    register_prepared_nullspace_provider,
)
from pops.native_components import PreparedNativeComponent  # noqa: E402
from ._prepared_field_nullspace_registry import (  # noqa: E402
    PreparedFieldNullspaceBinding,
    PreparedFieldNullspaceDefaultPolicy,
    PreparedFieldNullspaceFacts,
    PreparedFieldNullspaceProvider,
    PreparedFieldNullspaceResolution,
    PreparedFieldNullspaceResolutionValidator,
    register_prepared_field_nullspace_provider,
    register_prepared_field_nullspace_default_policy,
)
# Register the ready providers at catalog import.  The protocol registry itself remains free of
# concrete nullspace families and native route names.
from . import _prepared_field_nullspace_builtins as _field_nullspace_builtins  # noqa: E402,F401

# The module itself is the nullspace catalog (``pops.fields.nullspace.Prepared(...)``), mirroring
# the solver provider surface without introducing a second registry or a name dispatcher.
Prepared = PreparedNullspace
Provider = PreparedNullspaceProvider
Contracts = PreparedNullspaceContracts
NativeEmission = PreparedNullspaceNativeEmission
Use = PreparedNullspaceUse
UsePolicy = PreparedNullspaceUsePolicy
NativeComponent = PreparedNativeComponent
HeaderOnlyComponent = PreparedNativeComponent.header_only
register = register_prepared_nullspace_provider
FieldPrepared = PreparedFieldNullspace
FieldProvider = PreparedFieldNullspaceProvider
FieldBinding = PreparedFieldNullspaceBinding
FieldDefaultPolicy = PreparedFieldNullspaceDefaultPolicy
FieldFacts = PreparedFieldNullspaceFacts
FieldResolution = PreparedFieldNullspaceResolution
FieldResolutionValidator = PreparedFieldNullspaceResolutionValidator
register_field_provider = register_prepared_field_nullspace_provider
register_field_default_policy = register_prepared_field_nullspace_default_policy


__all__ = [
    "ConnectedComponentsManifest",
    "ConstantNullspace",
    "Contracts",
    "HeaderOnlyComponent",
    "FieldBinding",
    "FieldDefaultPolicy",
    "FieldFacts",
    "FieldPrepared",
    "FieldProvider",
    "FieldResolution",
    "FieldResolutionValidator",
    "NativeComponent",
    "NativeEmission",
    "NullspaceBasis",
    "NullspaceBasisVector",
    "NullspaceCompatibility",
    "Prepared",
    "PreparedNullspace",
    "PreparedFieldNullspace",
    "Provider",
    "RHSCompatibilityEvidence",
    "Use",
    "UsePolicy",
    "register",
    "register_field_provider",
    "register_field_default_policy",
]
