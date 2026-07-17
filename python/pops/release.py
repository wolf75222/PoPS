"""Immutable release and compatibility contract available without the native runtime."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ._generated_release_contract import (
    AMR_CHECKPOINT_PAYLOAD_VERSION,
    CAPABILITY_VOCABULARY_VERSION,
    CHECKPOINT_ENVELOPE_SCHEMA_VERSION,
    COMPONENT_CATALOG_SCHEMA_VERSION,
    COMPONENT_INTERFACE_ABI_VERSION,
    COMPONENT_MANIFEST_SCHEMA_VERSION,
    COMPONENT_REGISTRY_VERSION,
    NATIVE_ABI_VERSION,
    NORMALIZATION_VERSION,
    PACKAGE_VERSION,
    PUBLIC_API_VERSION,
    RELEASE_CONTRACT_SCHEMA_VERSION,
    RELEASE_CONTRACT_SHA256,
    SEMANTIC_IR_VERSION,
    SUPPORTED_MATRIX,
    UNIFORM_CHECKPOINT_PAYLOAD_VERSION,
)


@dataclass(frozen=True, order=True)
class PackageVersion:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: str) -> PackageVersion:
        if not isinstance(value, str):
            raise TypeError("package version must be text")
        parts = value.split(".")
        if len(parts) != 3 or any(not part.isdigit() for part in parts):
            raise ValueError("package version must be exactly major.minor.patch")
        return cls(*(int(part) for part in parts))


def package_compatible(*, requested: str, available: str) -> bool:
    """Return the CMake/package compatibility policy, including the pre-1.0 boundary."""
    need = PackageVersion.parse(requested)
    have = PackageVersion.parse(available)
    if have < need:
        return False
    if need.major == 0:
        return have.major == 0 and have.minor == need.minor
    return have.major == need.major


def contract() -> MappingProxyType[str, Any]:
    """Return the authenticated immutable contract advertised by this package."""
    return MappingProxyType({
        "package_version": PACKAGE_VERSION,
        "release_contract_schema_version": RELEASE_CONTRACT_SCHEMA_VERSION,
        "public_api_version": PUBLIC_API_VERSION,
        "semantic_ir_version": SEMANTIC_IR_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "component_catalog_schema_version": COMPONENT_CATALOG_SCHEMA_VERSION,
        "component_manifest_schema_version": COMPONENT_MANIFEST_SCHEMA_VERSION,
        "component_registry_version": COMPONENT_REGISTRY_VERSION,
        "capability_vocabulary_version": CAPABILITY_VOCABULARY_VERSION,
        "component_interface_abi_version": COMPONENT_INTERFACE_ABI_VERSION,
        "native_abi_version": NATIVE_ABI_VERSION,
        "checkpoint_envelope_schema_version": CHECKPOINT_ENVELOPE_SCHEMA_VERSION,
        "uniform_checkpoint_payload_version": UNIFORM_CHECKPOINT_PAYLOAD_VERSION,
        "amr_checkpoint_payload_version": AMR_CHECKPOINT_PAYLOAD_VERSION,
        "supported_matrix": SUPPORTED_MATRIX,
        "sha256": RELEASE_CONTRACT_SHA256,
    })


__all__ = ["PackageVersion", "contract", "package_compatible"]
