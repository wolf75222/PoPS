"""Authenticated component artifacts and the sole atomic installation boundary."""
from __future__ import annotations

import os
import json
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pops.identity import Identity, make_identity
from pops.interfaces import ComponentInterface

from ._package_data import ComponentPackageError
from .packages import FixedBinaryPackage, _binary_identity


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in sorted(value.items())})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _canonical_runtime_json(value: Any) -> str:
    if isinstance(value, Mapping):
        value = {key: _canonical_runtime_json_value(item) for key, item in value.items()}
    else:
        value = _canonical_runtime_json_value(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_runtime_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _canonical_runtime_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_canonical_runtime_json_value(item) for item in value]
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ComponentRuntimeContract:
    """Complete normalized facts consumed after authoring packages have disappeared."""

    component_id: str
    component_type: str
    version: str
    facets: tuple[str, ...]
    signature: Mapping[str, Any]
    reads: tuple[Any, ...]
    writes: tuple[Any, ...]
    requirements: tuple[Any, ...]
    effects: tuple[Any, ...]
    layouts: tuple[Any, ...]
    clocks: tuple[Any, ...]
    determinism: Mapping[str, Any]
    precision: Mapping[str, Any]
    parameters: tuple[Any, ...]
    capabilities: tuple[Any, ...]
    target: Mapping[str, Any]
    restart: Mapping[str, Any]
    conservation: tuple[Any, ...]
    native_interface: Mapping[str, Any]
    entry_points: Mapping[str, str]
    manifest_data: Mapping[str, Any]

    @classmethod
    def from_manifest(cls, manifest: Any) -> ComponentRuntimeContract:
        native = manifest.signature.get("native_interface")
        if not isinstance(native, Mapping):
            raise TypeError("component manifest has no exact native_interface signature")
        return cls(
            manifest.component_id, manifest.component_type, str(manifest.version),
            tuple(manifest.facets), _freeze(manifest.signature), _freeze(manifest.reads),
            _freeze(manifest.writes), _freeze(manifest.requirements), _freeze(manifest.effects),
            _freeze(manifest.layouts), _freeze(manifest.clocks),
            _freeze(manifest.determinism), _freeze(manifest.precision),
            _freeze(manifest.parameters), _freeze(manifest.capabilities),
            _freeze(manifest.target), _freeze(manifest.restart),
            _freeze(manifest.conservation), _freeze(native), _freeze(manifest.entry_points),
            _freeze(manifest.to_data()),
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "version": self.version,
            "facets": list(self.facets),
            "signature": _thaw(self.signature),
            "reads": _thaw(self.reads),
            "writes": _thaw(self.writes),
            "requirements": _thaw(self.requirements),
            "effects": _thaw(self.effects),
            "layouts": _thaw(self.layouts),
            "clocks": _thaw(self.clocks),
            "determinism": _thaw(self.determinism),
            "precision": _thaw(self.precision),
            "parameters": _thaw(self.parameters),
            "capabilities": _thaw(self.capabilities),
            "target": _thaw(self.target),
            "restart": _thaw(self.restart),
            "conservation": _thaw(self.conservation),
            "native_interface": _thaw(self.native_interface),
            "entry_points": _thaw(self.entry_points),
            "manifest": _thaw(self.manifest_data),
        }


def inspect_exported_symbols(path: Any) -> frozenset[str]:
    """Inspect one binary without loading or executing it."""
    binary = os.fspath(path)
    system = platform.system()
    if system == "Darwin":
        command = ["nm", "-gU", binary]
    elif system in ("Linux", "FreeBSD"):
        command = ["nm", "-D", "--defined-only", binary]
    else:
        raise ComponentPackageError(
            "symbol_inspection", binary, "unsupported platform %s" % system)
    if shutil.which(command[0]) is None:
        raise ComponentPackageError(
            "symbol_inspection", binary, "nm is required before artifact installation")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ComponentPackageError(
            "symbol_inspection", binary, result.stderr.strip() or "nm failed")
    symbols = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if fields:
            name = fields[-1]
            symbols.add(name[1:] if system == "Darwin" and name.startswith("_") else name)
    return frozenset(symbols)


def inspect_symbol_bytes(binary: bytes, *, suffix: str) -> frozenset[str]:
    with tempfile.NamedTemporaryFile(suffix=suffix) as stream:
        stream.write(binary)
        stream.flush()
        return inspect_exported_symbols(stream.name)


def _artifact_payload(value: Any) -> dict[str, Any]:
    return {
        "component_id": value.component_id,
        "component_manifest": value.component_manifest.token,
        "source_package": value.source_package.token if value.source_package else None,
        "platform": value.platform_manifest.identity.token,
        "interface": value.interface.to_data(),
        "runtime_contract": value.runtime_contract.to_data(),
        "entry_symbols": dict(value.entry_symbols),
        "binary": value.binary_identity.token,
        "fixed_signature": value.fixed_signature,
    }


@dataclass(frozen=True, slots=True)
class CompiledComponentArtifact:
    """Final audited binary bytes, detached from source-package paths."""

    component_id: str
    component_manifest: Identity
    runtime_contract: ComponentRuntimeContract
    interface: ComponentInterface
    platform_manifest: Any
    entry_symbols: Mapping[str, str]
    binary_identity: Identity
    binary: bytes = field(repr=False, compare=False)
    source_package: Identity | None = None
    fixed_signature: bool = False
    suffix: str = ".so"
    artifact_identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.component_id, str) or not self.component_id:
            raise TypeError("compiled component_id must be non-empty")
        if not isinstance(self.component_manifest, Identity) \
                or self.component_manifest.domain != "component-manifest":
            raise TypeError("component_manifest must be a component-manifest Identity")
        if type(self.runtime_contract) is not ComponentRuntimeContract \
                or self.runtime_contract.component_id != self.component_id:
            raise TypeError("runtime_contract must match the compiled component")
        if type(self.interface) is not ComponentInterface:
            raise TypeError("compiled interface must be an exact ComponentInterface")
        if _thaw(self.runtime_contract.native_interface) != _thaw(self.interface.to_data()):
            raise ValueError("runtime contract native interface differs from compiled interface")
        from pops.runtime._platform_manifest import PlatformManifest
        if type(self.platform_manifest) is not PlatformManifest:
            raise TypeError("compiled platform_manifest must be an exact PlatformManifest")
        entries = dict(sorted(self.entry_symbols.items()))
        if set(entries) != set(self.interface.runtime_entry_points):
            raise ValueError("compiled entry symbols must cover the interface exactly")
        if any(not isinstance(value, str) or not value for value in entries.values()):
            raise TypeError("compiled entry symbols must be non-empty strings")
        if len(set(entries.values())) != len(entries):
            raise ValueError("compiled entry symbols must be unique")
        object.__setattr__(self, "entry_symbols", MappingProxyType(entries))
        if not isinstance(self.binary, bytes) or not self.binary:
            raise TypeError("compiled binary must be non-empty exact bytes")
        if self.binary_identity != _binary_identity(self.binary):
            raise ComponentPackageError(
                "binary_digest", "binary", "compiled bytes do not match binary identity")
        if self.source_package is not None and (
                not isinstance(self.source_package, Identity)
                or self.source_package.domain != "component-package"):
            raise TypeError("source_package must be a component-package Identity or None")
        if self.fixed_signature == (self.source_package is not None):
            raise ValueError("fixed and source-compiled artifact authorities must be disjoint")
        suffix = str(self.suffix)
        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            raise ValueError("artifact suffix must be a simple extension")
        object.__setattr__(self, "suffix", suffix)
        object.__setattr__(self, "artifact_identity", make_identity(
            "component-artifact", _artifact_payload(self)))

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.entry_symbols.values())

    def verify(self) -> None:
        if _binary_identity(self.binary) != self.binary_identity:
            raise ComponentPackageError("binary_digest", "binary", "artifact bytes changed")
        missing = sorted(set(self.symbols) - inspect_symbol_bytes(self.binary, suffix=self.suffix))
        if missing:
            raise ComponentPackageError(
                "symbols", "binary", "artifact does not export %s" % missing)
        if self.artifact_identity != make_identity("component-artifact", _artifact_payload(self)):
            raise ComponentPackageError(
                "artifact_digest", "artifact", "artifact metadata identity changed")

    def to_data(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_manifest": self.component_manifest.to_data(),
            "runtime_contract": self.runtime_contract.to_data(),
            "interface": self.interface.to_data(),
            "platform_manifest": self.platform_manifest.to_data(),
            "entry_symbols": dict(self.entry_symbols),
            "binary_identity": self.binary_identity.to_data(),
            "artifact_identity": self.artifact_identity.to_data(),
            "source_package": None if self.source_package is None else self.source_package.to_data(),
            "fixed_signature": self.fixed_signature,
            "suffix": self.suffix,
        }

    @classmethod
    def from_fixed(cls, package: FixedBinaryPackage, alias: str,
                   *, interface: ComponentInterface) -> CompiledComponentArtifact:
        try:
            component_id = package.exports[alias]
        except KeyError:
            raise KeyError("unknown fixed component alias %r" % alias) from None
        manifest = next(item for item in package.manifests if item.component_id == component_id)
        interface.require_manifest(manifest, source_package=False)
        entries = {name: manifest.entry_points[name] for name in interface.runtime_entry_points}
        suffix = Path(package.binary_path).suffix or ".so"
        artifact = cls(
            component_id=component_id, component_manifest=manifest.manifest_digest,
            runtime_contract=ComponentRuntimeContract.from_manifest(manifest),
            interface=interface, platform_manifest=package.platform_manifest,
            entry_symbols=entries, binary_identity=package.binary_identity,
            binary=package.binary, fixed_signature=True, suffix=suffix)
        artifact.verify()
        return artifact

    def install(self, directory: Any) -> InstalledComponent:
        """Publish verified bytes atomically without clobbering."""
        self.verify()
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=True)
        destination = root / (self.artifact_identity.hexdigest + self.suffix)
        if destination.exists():
            if _binary_identity(destination.read_bytes()) != self.binary_identity:
                raise ComponentPackageError(
                    "install_collision", str(destination),
                    "content-addressed path has other bytes")
        else:
            fd, temporary = tempfile.mkstemp(
                prefix=".pops-component-", suffix=self.suffix, dir=str(root))
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(self.binary)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, 0o755)
                if _binary_identity(Path(temporary).read_bytes()) != self.binary_identity:
                    raise ComponentPackageError(
                        "binary_digest", temporary, "staged install bytes changed")
                missing = sorted(set(self.symbols) - inspect_exported_symbols(temporary))
                if missing:
                    raise ComponentPackageError(
                        "symbols", temporary, "staged binary does not export %s" % missing)
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    if _binary_identity(destination.read_bytes()) != self.binary_identity:
                        raise ComponentPackageError(
                            "install_collision", str(destination),
                            "content-addressed path has other bytes") from None
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        installed = InstalledComponent(
            self.component_id, self.component_manifest, self.runtime_contract,
            self.interface, self.platform_manifest, self.entry_symbols,
            self.binary_identity, self.artifact_identity, destination,
            "fixed" if self.fixed_signature else "source")
        installed.verify()
        return installed


@dataclass(frozen=True, slots=True)
class InstalledComponent:
    """Authenticated installed instance; raw library paths cannot enter runtime plans."""

    component_id: str
    component_manifest: Identity
    runtime_contract: ComponentRuntimeContract
    interface: ComponentInterface
    platform_manifest: Any
    entry_symbols: Mapping[str, str]
    binary_identity: Identity
    artifact_identity: Identity
    path: Path
    origin: str
    native_handle: Any = field(default=None, repr=False, compare=False)

    def verify(self) -> None:
        if not self.path.is_file() or _binary_identity(self.path.read_bytes()) != self.binary_identity:
            raise ComponentPackageError(
                "binary_digest", str(self.path), "installed binary identity mismatch")
        missing = sorted(set(self.entry_symbols.values()) - inspect_exported_symbols(self.path))
        if missing:
            raise ComponentPackageError(
                "symbols", str(self.path), "installed binary does not export %s" % missing)
        if self.native_handle is not None:
            report = self.native_handle.report()
            expected = {
                "component_id": self.component_id,
                "semantic_identity": self.runtime_contract.manifest_data["digests"]["semantic"],
                "manifest_identity": self.component_manifest.token,
                "catalog_sha256": self.interface.to_data()["catalog_sha256"],
                "abi_key": self.platform_manifest.abi.require("component.abi"),
            }
            for name, value in expected.items():
                if report[name] != value:
                    raise ComponentPackageError(
                        "loaded_identity", str(self.path),
                        "native table %s does not match installed metadata" % name)

    def load(self) -> InstalledComponent:
        """Resolve and authenticate the table once through the native loader."""
        self.verify()
        if self.native_handle is not None:
            return self
        try:
            from pops import _pops
        except ImportError as exc:
            raise RuntimeError(
                "installed component loading requires the matching PoPS native module") from exc
        from pops.codegen._native_host import ensure_native_host_global
        ensure_native_host_global(_pops)
        from pops._platform_contracts import validate_component_runtime
        from pops.runtime._platform_manifest import native_runtime_backend
        validate_component_runtime(
            self.platform_manifest, native_runtime_backend(self.platform_manifest))
        loader = getattr(_pops, "_load_component", None)
        if not callable(loader):
            raise RuntimeError(
                "installed component loading requires a native _load_component provider")
        handle = loader(
            str(self.path), self.component_id,
            self.runtime_contract.manifest_data["digests"]["semantic"],
            self.component_manifest.token, self.interface.to_data()["catalog_sha256"],
            self.platform_manifest.abi.require("component.abi"),
            [(self.interface.abi_id, self.interface.version, self.interface.cpp_table)],
            _canonical_runtime_json(self.runtime_contract.manifest_data["parameters"]),
            _canonical_runtime_json(self.runtime_contract.manifest_data["target"]),
        )
        loaded = replace(self, native_handle=handle)
        loaded.verify()
        return loaded

    def to_data(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_manifest": self.component_manifest.to_data(),
            "runtime_contract": self.runtime_contract.to_data(),
            "interface": self.interface.to_data(),
            "platform_manifest": self.platform_manifest.to_data(),
            "entry_symbols": dict(self.entry_symbols),
            "binary_identity": self.binary_identity.to_data(),
            "artifact_identity": self.artifact_identity.to_data(),
            "path": str(self.path),
            "loaded": self.native_handle is not None,
            "provenance": {
                "origin": self.origin,
                "source_uri": self.runtime_contract.manifest_data["uri"],
                "semantic_identity": self.runtime_contract.manifest_data["digests"]["semantic"],
                "manifest_identity": self.runtime_contract.manifest_data["digests"]["manifest"],
            },
        }


__all__ = [
    "ComponentRuntimeContract", "CompiledComponentArtifact", "InstalledComponent",
    "inspect_exported_symbols", "inspect_symbol_bytes",
]
