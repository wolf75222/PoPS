"""Strict wire helpers for external component packages."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from pops._manifest_protocol import exact_mapping, strict_json_loads
from pops.identity import Identity, make_identity


PACKAGE_SCHEMA_VERSION = 1
SOURCE_KIND = "source"
FIXED_KIND = "fixed_binary"
PROTOCOL_ABI = "pops.component.protocol.v1"
_TOP = frozenset({"schema_version", "package_kind", "protocol_abi", "components", "exports",
                  "payloads", "platform", "binary", "package_digest"})
_PAYLOAD = frozenset({"path", "kind", "digest"})
_BINARY = frozenset({"path", "digest", "symbols"})
_PAYLOAD_KINDS = frozenset({"header", "source", "ir"})


class ComponentPackageError(ValueError):
    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__("[%s] %s: %s" % (code, path, message))
        self.code, self.path = code, path


def exact_package(value: Any) -> Mapping[str, Any]:
    try:
        row = exact_mapping(value, _TOP, where="component package")
    except (TypeError, ValueError) as exc:
        raise ComponentPackageError("manifest_schema", "package", str(exc)) from exc
    if type(row["schema_version"]) is not int or row["schema_version"] != PACKAGE_SCHEMA_VERSION:
        raise ComponentPackageError(
            "manifest_schema", "schema_version",
            "expected exactly %d" % PACKAGE_SCHEMA_VERSION)
    if row["package_kind"] not in (SOURCE_KIND, FIXED_KIND):
        raise ComponentPackageError("package_kind", "package_kind", "unsupported package kind")
    if row["protocol_abi"] != PROTOCOL_ABI:
        raise ComponentPackageError(
            "protocol_abi", "protocol_abi", "unsupported component protocol ABI")
    return row


def read_json(path: Any) -> tuple[Path, Mapping[str, Any]]:
    manifest_path = Path(path).resolve(strict=True)
    try:
        value = strict_json_loads(manifest_path.read_bytes(), where=str(manifest_path))
    except (OSError, TypeError, ValueError) as exc:
        raise ComponentPackageError("manifest_json", str(manifest_path), str(exc)) from exc
    return manifest_path, exact_package(value)


def canonical_relative_path(value: Any, *, where: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ComponentPackageError("package_path", where, "must be a canonical POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ComponentPackageError("package_path", where, "absolute/traversing paths are forbidden")
    return path


def read_payload(root: Path, relative: PurePosixPath) -> bytes:
    candidate = (root / Path(*relative.parts)).resolve(strict=True)
    try:
        candidate.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ComponentPackageError(
            "package_path", str(relative), "resolved path escapes the package root") from exc
    if not candidate.is_file():
        raise ComponentPackageError("package_path", str(relative), "payload is not a regular file")
    return candidate.read_bytes()


def content_identity(kind: str, content: bytes) -> Identity:
    return make_identity(
        "component-%s" % kind,
        {"algorithm": "sha256", "content_digest": hashlib.sha256(content).digest(),
         "size": len(content)},
    )


def require_identity(token: Any, domain: str, *, where: str) -> Identity:
    try:
        identity = Identity.from_token(token)
    except (TypeError, ValueError) as exc:
        raise ComponentPackageError("digest", where, "invalid identity token") from exc
    if identity.domain != domain:
        raise ComponentPackageError(
            "digest", where, "expected pops.%s identity, got pops.%s" % (domain, identity.domain))
    return identity


def package_identity_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in sorted(_TOP - {"package_digest"})}


def package_identity(row: Mapping[str, Any]) -> Identity:
    return make_identity("component-package", package_identity_payload(row))


def verify_package_identity(row: Mapping[str, Any]) -> Identity:
    supplied = require_identity(row["package_digest"], "component-package", where="package_digest")
    expected = package_identity(row)
    if supplied != expected:
        raise ComponentPackageError(
            "package_digest", "package_digest", "manifest content does not match package digest")
    return expected


def dump_manifest(row: Mapping[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def payload_row(path: str, kind: str, content: bytes) -> dict[str, Any]:
    canonical_relative_path(path, where="payload.path")
    if kind not in _PAYLOAD_KINDS:
        raise ComponentPackageError("payload_kind", "payload.kind", "unsupported payload kind")
    return {"path": path, "kind": kind, "digest": content_identity(kind, content).token}


def validate_payload_row(value: Any, index: int) -> Mapping[str, Any]:
    try:
        row = exact_mapping(value, _PAYLOAD, where="payloads[%d]" % index)
    except TypeError as exc:
        raise ComponentPackageError("manifest_schema", "payloads[%d]" % index, str(exc)) from exc
    canonical_relative_path(row["path"], where="payloads[%d].path" % index)
    if row["kind"] not in _PAYLOAD_KINDS:
        raise ComponentPackageError(
            "payload_kind", "payloads[%d].kind" % index, "unsupported payload kind")
    require_identity(row["digest"], "component-%s" % row["kind"],
                     where="payloads[%d].digest" % index)
    return row


def validate_binary_row(value: Any) -> Mapping[str, Any]:
    try:
        row = exact_mapping(value, _BINARY, where="binary")
    except TypeError as exc:
        raise ComponentPackageError("manifest_schema", "binary", str(exc)) from exc
    canonical_relative_path(row["path"], where="binary.path")
    require_identity(row["digest"], "binary", where="binary.digest")
    symbols = row["symbols"]
    if not isinstance(symbols, list) or not symbols or any(
            not isinstance(item, str) or not item for item in symbols):
        raise ComponentPackageError("symbols", "binary.symbols", "must be non-empty strings")
    if symbols != sorted(set(symbols)):
        raise ComponentPackageError("symbols", "binary.symbols", "must be unique and sorted")
    return row


__all__ = [
    "PACKAGE_SCHEMA_VERSION", "SOURCE_KIND", "FIXED_KIND", "PROTOCOL_ABI",
    "ComponentPackageError", "canonical_relative_path", "content_identity", "dump_manifest",
    "exact_package", "package_identity", "payload_row", "read_json", "read_payload",
    "require_identity", "validate_binary_row", "validate_payload_row", "verify_package_identity",
]
