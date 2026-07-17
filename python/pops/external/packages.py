"""Content-authenticated source and fixed-binary component packages."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

from pops.identity import Identity, canonical_bytes, make_identity
from pops.interfaces import ComponentInterface
from pops.model import ComponentManifest

from ._package_data import (
    FIXED_KIND,
    PACKAGE_SCHEMA_VERSION,
    PROTOCOL_ABI,
    SOURCE_KIND,
    ComponentPackageError,
    canonical_relative_path,
    content_identity,
    package_identity,
    payload_row,
    read_json,
    read_payload,
    require_identity,
    validate_binary_row,
    validate_payload_row,
    verify_package_identity,
)


_ALIAS = re.compile(r"^[a-z][a-z0-9_]*$")


def _binary_identity(content: bytes) -> Identity:
    return make_identity(
        "binary",
        {"algorithm": "sha256", "content_digest": hashlib.sha256(content).digest(),
         "size": len(content)},
    )


def _components(value: Any) -> tuple[ComponentManifest, ...]:
    if not isinstance(value, list) or not value:
        raise ComponentPackageError("components", "components", "must be a non-empty list")
    result = []
    for index, row in enumerate(value):
        try:
            result.append(ComponentManifest.from_data(row))
        except (TypeError, ValueError) as exc:
            raise ComponentPackageError(
                "component_manifest", "components[%d]" % index, str(exc)) from exc
    ids = [item.component_id for item in result]
    if len(ids) != len(set(ids)):
        raise ComponentPackageError("components", "components", "component IDs must be unique")
    return tuple(result)


def _exports(value: Any, manifests: tuple[ComponentManifest, ...]) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ComponentPackageError("exports", "exports", "must be a non-empty mapping")
    known = {item.component_id for item in manifests}
    result = {}
    for alias, component_id in sorted(value.items()):
        if not isinstance(alias, str) or _ALIAS.fullmatch(alias) is None:
            raise ComponentPackageError("exports", "exports", "aliases must be snake_case IDs")
        if not isinstance(component_id, str) or component_id not in known:
            raise ComponentPackageError(
                "exports", "exports.%s" % alias, "must name a package component ID")
        result[alias] = component_id
    if len(set(result.values())) != len(result):
        raise ComponentPackageError(
            "exports", "exports", "one component cannot be exported under competing aliases")
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class PackagePayload:
    path: str
    kind: str
    identity: Identity
    content: bytes = field(repr=False, compare=False)

    def to_data(self) -> dict[str, Any]:
        return {"path": self.path, "kind": self.kind, "digest": self.identity.token}


@dataclass(frozen=True, slots=True)
class SourceComponentPackage:
    """Verified immutable source/header/IR bytes; never a compiled artifact."""

    manifests: tuple[ComponentManifest, ...]
    exports: Mapping[str, str]
    payloads: tuple[PackagePayload, ...]
    identity: Identity
    manifest_path: Path = field(compare=False, repr=False)
    protocol_abi: str = PROTOCOL_ABI

    @classmethod
    def from_manifest(cls, path: Any) -> SourceComponentPackage:
        manifest_path, row = read_json(path)
        if row["package_kind"] != SOURCE_KIND:
            raise ComponentPackageError(
                "package_kind", "package_kind", "expected a generic source/header/IR package")
        if row["platform"] is not None or row["binary"] is not None:
            raise ComponentPackageError(
                "source_claims_binary", "package", "source packages cannot claim platform/binary facts")
        manifests = _components(row["components"])
        exports = _exports(row["exports"], manifests)
        if not isinstance(row["payloads"], list) or not row["payloads"]:
            raise ComponentPackageError("payloads", "payloads", "source package has no payloads")
        payloads = []
        names = set()
        for index, value in enumerate(row["payloads"]):
            record = validate_payload_row(value, index)
            if record["path"] in names:
                raise ComponentPackageError(
                    "payloads", "payloads", "payload paths must be unique")
            names.add(record["path"])
            relative = canonical_relative_path(record["path"], where="payloads[%d].path" % index)
            content = read_payload(manifest_path.parent, relative)
            expected = content_identity(record["kind"], content)
            supplied = require_identity(
                record["digest"], expected.domain, where="payloads[%d].digest" % index)
            if supplied != expected:
                raise ComponentPackageError(
                    "source_digest", record["path"], "payload bytes do not match the manifest")
            payloads.append(PackagePayload(record["path"], record["kind"], expected, content))
        identity = verify_package_identity(row)
        return cls(manifests, exports, tuple(payloads), identity, manifest_path)

    def manifest(self, component_id: str) -> ComponentManifest:
        for manifest in self.manifests:
            if manifest.component_id == component_id:
                return manifest
        raise KeyError(component_id)

    def require(self, alias: str, *, interface: ComponentInterface) -> ExternalComponentType:
        if type(interface) is not ComponentInterface:
            raise TypeError("interface must be an exact pops.interfaces.ComponentInterface")
        try:
            manifest = self.manifest(self.exports[alias])
        except KeyError:
            raise KeyError("unknown package component alias %r" % alias) from None
        interface.require_manifest(manifest)
        return ExternalComponentType(self, alias, manifest, interface)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": PACKAGE_SCHEMA_VERSION, "package_kind": SOURCE_KIND,
            "protocol_abi": self.protocol_abi,
            "components": [item.to_data() for item in self.manifests],
            "exports": dict(self.exports), "payloads": [item.to_data() for item in self.payloads],
            "platform": None, "binary": None, "package_digest": self.identity.token,
        }


def _parameter_names(manifest: ComponentManifest) -> set[str]:
    names = set()
    for index, row in enumerate(manifest.parameters):
        if not isinstance(row, Mapping) or not isinstance(row.get("name"), str):
            raise ComponentPackageError(
                "parameter_schema", "parameters[%d]" % index, "must carry a string name")
        names.add(row["name"])
    return names


def _freeze_parameter_value(value: Any) -> Any:
    """Detach the canonical component-parameter tree from caller-owned containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({
            key: _freeze_parameter_value(item) for key, item in sorted(value.items())
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_parameter_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_parameter_value(item) for item in value)
    return value


def _thaw_parameter_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_parameter_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_parameter_value(item) for item in value]
    if isinstance(value, frozenset):
        return frozenset(_thaw_parameter_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ExternalComponentType:
    package: SourceComponentPackage = field(repr=False)
    alias: str
    manifest: ComponentManifest
    interface: ComponentInterface

    def __call__(self, **parameters: Any) -> ExternalComponent:
        unknown = sorted(set(parameters) - _parameter_names(self.manifest))
        if unknown:
            raise TypeError("unknown component parameter(s): %s" % ", ".join(unknown))
        try:
            canonical_bytes(parameters)
        except (TypeError, ValueError) as exc:
            raise TypeError("component parameters must be canonical values") from exc
        return ExternalComponent(self, _freeze_parameter_value(parameters))


@dataclass(frozen=True, slots=True)
class ExternalComponent:
    # The package, manifest, interface and canonical parameter mapping are all frozen and
    # content-authenticated.  The authoring/compiler snapshot may therefore retain this exact
    # authority instead of recursively cloning the package graph (which would also lose the
    # intentional identity relationship used by resolve component matching).
    __pops_ir_immutable__: ClassVar[bool] = True

    component_type: ExternalComponentType
    parameters: Mapping[str, Any]

    @property
    def component_manifest(self) -> ComponentManifest:
        return self.component_type.manifest

    @property
    def package_identity(self) -> Identity:
        return self.component_type.package.identity

    def to_data(self) -> dict[str, Any]:
        return {
            "component_id": self.component_manifest.component_id,
            "component_manifest": self.component_manifest.manifest_digest.token,
            "source_package": self.package_identity.token,
            "interface": self.component_type.interface.to_data(),
            "parameters": _thaw_parameter_value(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class FixedBinaryPackage:
    """Authenticated fixed-signature binary bytes; no template/generic claim is permitted."""

    manifests: tuple[ComponentManifest, ...]
    exports: Mapping[str, str]
    platform_manifest: Any
    symbols: tuple[str, ...]
    binary_identity: Identity
    binary: bytes = field(repr=False, compare=False)
    identity: Identity
    manifest_path: Path = field(compare=False, repr=False)
    binary_path: str

    @classmethod
    def from_manifest(cls, path: Any) -> FixedBinaryPackage:
        manifest_path, row = read_json(path)
        if row["package_kind"] != FIXED_KIND:
            raise ComponentPackageError(
                "package_kind", "package_kind", "expected a fixed-binary package")
        if row["payloads"] != []:
            raise ComponentPackageError(
                "fixed_claims_source", "payloads", "fixed binaries cannot claim generic payloads")
        manifests = _components(row["components"])
        exports = _exports(row["exports"], manifests)
        for manifest in manifests:
            _require_fixed_signature(manifest)
        try:
            from pops.runtime._platform_manifest import PlatformManifest
            platform = PlatformManifest.from_data(row["platform"])
        except (TypeError, ValueError) as exc:
            raise ComponentPackageError("target", "platform", str(exc)) from exc
        binary_row = validate_binary_row(row["binary"])
        relative = canonical_relative_path(binary_row["path"], where="binary.path")
        binary = read_payload(manifest_path.parent, relative)
        binary_identity = _binary_identity(binary)
        supplied = require_identity(binary_row["digest"], "binary", where="binary.digest")
        if supplied != binary_identity:
            raise ComponentPackageError(
                "binary_digest", "binary.digest", "binary bytes do not match the manifest")
        symbols = tuple(binary_row["symbols"])
        required = {entry for manifest in manifests for entry in manifest.entry_points.values()}
        missing = sorted(required - set(symbols))
        if missing:
            raise ComponentPackageError(
                "symbols", "binary.symbols", "missing declared entry point(s): %s" % missing)
        _require_platform_matches(manifests, platform)
        identity = verify_package_identity(row)
        return cls(manifests, exports, platform, symbols, binary_identity, binary, identity,
                   manifest_path, str(relative))

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": PACKAGE_SCHEMA_VERSION, "package_kind": FIXED_KIND,
            "protocol_abi": PROTOCOL_ABI,
            "components": [item.to_data() for item in self.manifests],
            "exports": dict(self.exports), "payloads": [],
            "platform": self.platform_manifest.to_data(),
            "binary": {"path": self.binary_path, "digest": self.binary_identity.token,
                       "symbols": list(self.symbols)},
            "package_digest": self.identity.token,
        }


def _require_fixed_signature(manifest: ComponentManifest) -> None:
    signature = manifest.signature
    if not isinstance(signature, Mapping) or signature.get("generic") is not False:
        raise ComponentPackageError(
            "fixed_generic_claim", "signature.generic",
            "a fixed binary must declare generic=false and exact ABI types")
    if any(key in signature for key in ("template_parameters", "type_variables")):
        raise ComponentPackageError(
            "fixed_generic_claim", "signature", "fixed binaries cannot advertise type variables")
    if len(manifest.target["variants"]) != 1:
        raise ComponentPackageError(
            "fixed_target", "target.variants", "a fixed binary must declare one exact target")


def _require_platform_matches(manifests: tuple[ComponentManifest, ...], platform: Any) -> None:
    dimensions = tuple(platform.capabilities["dimensions"].require("platform.dimensions"))
    scalar = platform.precision.compute.require("platform.precision.compute")
    device = platform.device.require("platform.device")
    normalized_device = "cpu" if device in ("host", "cpu") else device
    for manifest in manifests:
        variant = manifest.target["variants"][0]
        if (variant["dimension"] not in dimensions or variant["scalar"] != scalar
                or variant["device"] != normalized_device):
            raise ComponentPackageError(
                "target", "platform", "component target does not match fixed binary platform")


def load(path: Any) -> SourceComponentPackage | FixedBinaryPackage:
    _, row = read_json(path)
    if row["package_kind"] == SOURCE_KIND:
        return SourceComponentPackage.from_manifest(path)
    return FixedBinaryPackage.from_manifest(path)


def build_source_package_manifest(
    *, components: Mapping[str, ComponentManifest], payloads: Mapping[str, tuple[str, bytes]],
) -> dict[str, Any]:
    manifests = tuple(components.values())
    exports = {alias: manifest.component_id for alias, manifest in sorted(components.items())}
    rows = [payload_row(path, kind, content) for path, (kind, content) in sorted(payloads.items())]
    result = {
        "schema_version": PACKAGE_SCHEMA_VERSION, "package_kind": SOURCE_KIND,
        "protocol_abi": PROTOCOL_ABI,
        "components": [item.to_data() for item in manifests], "exports": exports,
        "payloads": rows, "platform": None, "binary": None, "package_digest": "",
    }
    result["package_digest"] = package_identity(result).token
    return result


def build_fixed_binary_manifest(
    *, components: Mapping[str, ComponentManifest], platform: Any,
    binary_path: str, binary: bytes, symbols: tuple[str, ...],
) -> dict[str, Any]:
    from pops.runtime._platform_manifest import PlatformManifest
    if type(platform) is not PlatformManifest:
        raise TypeError("platform must be an exact PlatformManifest")
    manifests = tuple(components.values())
    for manifest in manifests:
        _require_fixed_signature(manifest)
    result = {
        "schema_version": PACKAGE_SCHEMA_VERSION, "package_kind": FIXED_KIND,
        "protocol_abi": PROTOCOL_ABI,
        "components": [item.to_data() for item in manifests],
        "exports": {alias: item.component_id for alias, item in sorted(components.items())},
        "payloads": [], "platform": platform.to_data(),
        "binary": {"path": binary_path, "digest": _binary_identity(binary).token,
                   "symbols": sorted(set(symbols))},
        "package_digest": "",
    }
    result["package_digest"] = package_identity(result).token
    return result


__all__ = [
    "ComponentPackageError", "PackagePayload", "SourceComponentPackage", "FixedBinaryPackage",
    "ExternalComponentType", "ExternalComponent", "build_source_package_manifest",
    "build_fixed_binary_manifest", "load",
]
