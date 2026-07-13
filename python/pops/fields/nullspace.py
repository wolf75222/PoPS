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


__all__ = [
    "ConnectedComponentsManifest", "ConstantNullspace", "NullspaceBasis",
    "NullspaceBasisVector", "NullspaceCompatibility", "RHSCompatibilityEvidence",
]
