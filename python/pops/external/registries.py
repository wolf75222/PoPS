"""Separate atomic phase registries for source packages and compiled artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from .artifacts import CompiledComponentArtifact
from .packages import SourceComponentPackage


@dataclass(frozen=True, slots=True)
class SourceRegistryRecord:
    component_id: str
    manifest_digest: str
    package_digest: str
    package: SourceComponentPackage


class SourcePackageRegistry:
    """Authoring-phase registry. All exports of a package publish atomically or not at all."""

    def __init__(self) -> None:
        self._records: dict[str, SourceRegistryRecord] = {}
        self._lock = RLock()
        self._frozen = False
        self._revision = 0

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def revision(self) -> int:
        return self._revision

    def register(self, package: SourceComponentPackage) -> SourceComponentPackage:
        if type(package) is not SourceComponentPackage:
            raise TypeError("SourcePackageRegistry accepts exact SourceComponentPackage values")
        incoming = []
        for component_id in package.exports.values():
            manifest = package.manifest(component_id)
            incoming.append(SourceRegistryRecord(
                component_id, manifest.manifest_digest.token, package.identity.token, package))
        with self._lock:
            if self._frozen:
                raise RuntimeError("SourcePackageRegistry is frozen")
            additions = []
            previous_package = None
            for record in incoming:
                previous = self._records.get(record.component_id)
                if previous is None:
                    additions.append(record)
                elif (previous.manifest_digest, previous.package_digest) != (
                        record.manifest_digest, record.package_digest):
                    raise ValueError("source package identity collision for %r" % record.component_id)
                else:
                    previous_package = previous.package
            for record in additions:
                self._records[record.component_id] = record
            if additions:
                self._revision += 1
        return package if additions or previous_package is None else previous_package

    def resolve(self, component_id: str) -> SourceComponentPackage:
        try:
            return self._records[component_id].package
        except KeyError:
            raise KeyError("unknown source component %r" % component_id) from None

    def freeze(self) -> SourcePackageRegistry:
        with self._lock:
            self._frozen = True
        return self


@dataclass(frozen=True, slots=True)
class CompiledRegistryRecord:
    key: tuple[str, str]
    artifact_digest: str
    binary_digest: str
    artifact: CompiledComponentArtifact


class CompiledArtifactRegistry:
    """Post-link registry keyed by component plus exact target platform identity."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], CompiledRegistryRecord] = {}
        self._lock = RLock()
        self._frozen = False
        self._revision = 0

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def revision(self) -> int:
        return self._revision

    def register(self, artifact: CompiledComponentArtifact) -> CompiledComponentArtifact:
        if type(artifact) is not CompiledComponentArtifact:
            raise TypeError("CompiledArtifactRegistry accepts exact CompiledComponentArtifact values")
        artifact.verify()
        key = artifact.component_id, artifact.platform_manifest.identity.token
        incoming = CompiledRegistryRecord(
            key, artifact.artifact_identity.token, artifact.binary_identity.token, artifact)
        with self._lock:
            if self._frozen:
                raise RuntimeError("CompiledArtifactRegistry is frozen")
            previous = self._records.get(key)
            if previous is not None:
                if (previous.artifact_digest, previous.binary_digest) == (
                        incoming.artifact_digest, incoming.binary_digest):
                    return previous.artifact
                raise ValueError("compiled artifact identity collision for %r" % (key,))
            self._records[key] = incoming
            self._revision += 1
        return artifact

    def resolve(self, component_id: str, platform_identity: Any) -> CompiledComponentArtifact:
        token = getattr(platform_identity, "token", platform_identity)
        try:
            return self._records[(component_id, token)].artifact
        except KeyError:
            raise KeyError("unknown compiled component target %r" % ((component_id, token),)) from None

    def freeze(self) -> CompiledArtifactRegistry:
        with self._lock:
            self._frozen = True
        return self


__all__ = [
    "SourceRegistryRecord", "SourcePackageRegistry", "CompiledRegistryRecord",
    "CompiledArtifactRegistry",
]
