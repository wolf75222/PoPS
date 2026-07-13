"""Authenticated component artifacts and the sole atomic installation boundary."""
from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
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


@dataclass(frozen=True, slots=True)
class ComponentRuntimeContract:
    """Normalized facts an installed runtime consumes without reopening authoring packages."""

    component_id: str
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
    entry_points: Mapping[str, str]

    @classmethod
    def from_manifest(cls, manifest: Any) -> ComponentRuntimeContract:
        return cls(
            manifest.component_id, tuple(manifest.facets), _freeze(manifest.signature),
            _freeze(manifest.reads), _freeze(manifest.writes), _freeze(manifest.requirements),
            _freeze(manifest.effects), _freeze(manifest.layouts), _freeze(manifest.clocks),
            _freeze(manifest.determinism), _freeze(manifest.precision),
            _freeze(manifest.entry_points))


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
        if not fields:
            continue
        name = fields[-1]
        if system == "Darwin" and name.startswith("_"):
            name = name[1:]
        symbols.add(name)
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
        "entry_symbols": dict(value.entry_symbols),
        "binary": value.binary_identity.token,
        "fixed_signature": value.fixed_signature,
    }


@dataclass(frozen=True, slots=True)
class CompiledComponentArtifact:
    """Final audited binary bytes.  This is neither authoring source nor an installed instance."""

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
        from pops.runtime.platform_manifest import PlatformManifest
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
        expected_binary = _binary_identity(self.binary)
        if self.binary_identity != expected_binary:
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
        found = inspect_symbol_bytes(self.binary, suffix=self.suffix)
        missing = sorted(set(self.symbols) - found)
        if missing:
            raise ComponentPackageError(
                "symbols", "binary", "artifact does not export %s" % missing)
        expected = make_identity("component-artifact", _artifact_payload(self))
        if self.artifact_identity != expected:
            raise ComponentPackageError(
                "artifact_digest", "artifact", "artifact metadata identity changed")

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
        """Publish verified bytes atomically without clobbering, then return the loadable type."""
        self.verify()
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=True)
        destination = root / (self.artifact_identity.hexdigest + self.suffix)
        if destination.exists():
            if _binary_identity(destination.read_bytes()) != self.binary_identity:
                raise ComponentPackageError(
                    "install_collision", str(destination), "content-addressed path has other bytes")
        else:
            fd, temporary = tempfile.mkstemp(prefix=".pops-component-", suffix=self.suffix,
                                             dir=str(root))
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
                    # Same-filesystem link publication is atomic and, unlike os.replace(), cannot
                    # overwrite a competing process's content-addressed destination.
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
            self.interface, self.platform_manifest,
            self.entry_symbols, self.binary_identity, self.artifact_identity, destination)
        installed.verify()
        return installed


@dataclass(frozen=True, slots=True)
class InstalledComponent:
    """Authenticated installed instance. Raw package paths can never masquerade as this value."""

    component_id: str
    component_manifest: Identity
    runtime_contract: ComponentRuntimeContract
    interface: ComponentInterface
    platform_manifest: Any
    entry_symbols: Mapping[str, str]
    binary_identity: Identity
    artifact_identity: Identity
    path: Path

    def verify(self) -> None:
        if not self.path.is_file() or _binary_identity(self.path.read_bytes()) != self.binary_identity:
            raise ComponentPackageError(
                "binary_digest", str(self.path), "installed binary identity mismatch")
        missing = sorted(set(self.entry_symbols.values()) - inspect_exported_symbols(self.path))
        if missing:
            raise ComponentPackageError(
                "symbols", str(self.path), "installed binary does not export %s" % missing)

    def bind(self, interface: ComponentInterface) -> Any:
        self.verify()
        if interface != self.interface:
            raise TypeError("installed component interface mismatch")
        return interface.bind_installed(self)


class NumericalFluxCpuBinding:
    """One interface binding shared by every NumericalFlux component; no component dispatch."""

    __slots__ = ("_artifact", "_flux", "_stability", "_handle")

    def __init__(self, artifact: InstalledComponent) -> None:
        artifact.verify()
        handle = ctypes.CDLL(str(artifact.path))
        flux = getattr(handle, artifact.entry_symbols["numerical_flux"])
        stability = getattr(handle, artifact.entry_symbols["stability_bound"])
        args = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        flux.argtypes, flux.restype = args, ctypes.c_int
        stability.argtypes, stability.restype = args, ctypes.c_int
        self._artifact, self._handle = artifact, handle
        self._flux, self._stability = flux, stability

    def evaluate(self, left: tuple[float, ...], right: tuple[float, ...],
                 normal: tuple[float, float]) -> tuple[tuple[float, ...], float]:
        if not left or len(left) != len(right) or len(normal) != 2:
            raise ValueError("CPU NumericalFlux binding requires equal non-empty states and a 2D normal")
        n = len(left)
        expected = self._artifact.runtime_contract.signature.get("state_components")
        if n != expected:
            raise ValueError("CPU NumericalFlux binding expected %s state components" % expected)
        array = ctypes.c_double * n
        face = (ctypes.c_double * 2)(*normal)
        output = array()
        status = self._flux(array(*left), array(*right), face, output)
        if status != 0:
            raise RuntimeError("external numerical_flux returned status %d" % status)
        speed = ctypes.c_double()
        status = self._stability(array(*left), array(*right), face, ctypes.byref(speed))
        if status != 0:
            raise RuntimeError("external stability_bound returned status %d" % status)
        return tuple(output), speed.value


__all__ = [
    "ComponentRuntimeContract", "CompiledComponentArtifact", "InstalledComponent",
    "NumericalFluxCpuBinding",
    "inspect_exported_symbols", "inspect_symbol_bytes",
]
