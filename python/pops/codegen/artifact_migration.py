"""Deterministic offline migration of persisted PoPS manifest artifacts.

This module is intentionally absent from every runtime loader.  It never
``dlopen``s a library and never consults the current runtime: facts which are
not present in the source bytes must be supplied explicitly by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any

from pops.model.manifest_data import strict_json_loads


MIGRATION_PROTOCOL = "pops.manifest-migration.external-brick.v1"
CURRENT_BRICK_SCHEMA_VERSION = 3
_ROW_FIELDS = (
    "id", "category", "requirements", "capabilities", "native_id",
    "supported_layouts", "supported_platforms", "params", "options",
    "exported_symbols",
)
_V2_TOP_FIELDS = frozenset(("schema_version", "abi_key", "bricks"))
_V2_ROW_FIELDS = frozenset(_ROW_FIELDS)


@dataclass(frozen=True)
class MigrationReport:
    source_identity: str
    artifact_identity: str
    from_schema: int
    to_schema: int
    changed: bool
    destination: str


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def _identity(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read(path: Any) -> tuple[bytes, dict[str, Any]]:
    with open(os.fspath(path), "rb") as handle:
        raw = handle.read()
    try:
        value = strict_json_loads(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("legacy external-brick manifest is not strict JSON: %s" % error) from error
    if not isinstance(value, dict):
        raise ValueError("legacy external-brick manifest must be a JSON object")
    return raw, value


def _explicit(metadata: dict[str, Any], brick_id: str, field: str) -> Any:
    rows = metadata.get("bricks", {})
    row = rows.get(brick_id, {}) if isinstance(rows, dict) else {}
    if field not in row:
        raise ValueError(
            "cannot migrate brick %r: field %r is not derivable from source bytes; "
            "supply metadata={'bricks': {%r: {%r: ...}}} or rebuild"
            % (brick_id, field, brick_id, field)
        )
    return row[field]


def _migrate_v2(doc: dict[str, Any], metadata: dict[str, Any], source_identity: str) -> dict[str, Any]:
    unknown = sorted(set(doc) - _V2_TOP_FIELDS)
    if unknown:
        raise ValueError("legacy external-brick manifest has unknown field(s) %s" % unknown)
    if not isinstance(doc.get("bricks"), list):
        raise ValueError("legacy external-brick manifest 'bricks' must be a list")
    abi_key = doc.get("abi_key")
    if not isinstance(abi_key, str) or not abi_key:
        abi_key = metadata.get("abi_key")
    if not isinstance(abi_key, str) or not abi_key:
        raise ValueError(
            "cannot migrate manifest: non-empty abi_key is not derivable; supply explicit metadata"
        )
    rows = []
    seen = set()
    for index, source in enumerate(doc["bricks"]):
        if not isinstance(source, dict):
            raise ValueError("legacy external-brick manifest bricks[%d] must be an object" % index)
        unknown_row = sorted(set(source) - _V2_ROW_FIELDS)
        if unknown_row:
            raise ValueError("legacy brick row has unknown field(s) %s" % unknown_row)
        brick_id = source.get("id")
        if not isinstance(brick_id, str) or not brick_id:
            raise ValueError("legacy brick row id must be a non-empty string")
        if brick_id in seen:
            raise ValueError("legacy external-brick manifest contains duplicate id %r" % brick_id)
        seen.add(brick_id)
        row = {}
        for field in _ROW_FIELDS:
            row[field] = source[field] if field in source else _explicit(metadata, brick_id, field)
        rows.append(row)
    return {
        "schema_version": CURRENT_BRICK_SCHEMA_VERSION,
        "abi_key": abi_key,
        "annotations": {
            "x-pops.migration": {
                "protocol": MIGRATION_PROTOCOL,
                "from_schema": 2,
                "source_content_identity": source_identity,
            }
        },
        "bricks": rows,
    }


def migrate_external_brick_manifest(
    source: Any, destination: Any, *, metadata: dict[str, Any] | None = None,
) -> MigrationReport:
    """Migrate one manifest file without loading code or probing the runtime.

    Current canonical input is copied byte-for-byte canonically.  Version 2 is
    accepted only through its exact historical schema; every missing current
    fact must be supplied explicitly.  The source is never modified.
    """
    source_path = os.path.abspath(os.fspath(source))
    destination_path = os.path.abspath(os.fspath(destination))
    raw, doc = _read(source_path)
    source_identity = _identity(raw)
    version = doc.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("manifest schema_version must be an integer")
    if version == CURRENT_BRICK_SCHEMA_VERSION:
        from pops.descriptors import parse_brick_manifest

        parse_brick_manifest(_canonical_bytes(doc))
        migrated = doc
    elif version == 2:
        migrated = _migrate_v2(doc, dict(metadata or {}), source_identity)
        from pops.descriptors import parse_brick_manifest

        parse_brick_manifest(_canonical_bytes(migrated))
    else:
        raise ValueError("no offline external-brick migration from schema %r" % version)
    output = _canonical_bytes(migrated)
    if source_path == destination_path and output != raw:
        raise ValueError("offline migration refuses to overwrite its source artifact")
    directory = os.path.dirname(destination_path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = os.path.join(
        directory, ".%s.tmp-%d" % (os.path.basename(destination_path), os.getpid()))
    with open(temporary, "wb") as handle:
        handle.write(output)
    os.replace(temporary, destination_path)
    artifact_identity = _identity(output)
    return MigrationReport(
        source_identity=source_identity,
        artifact_identity=artifact_identity,
        from_schema=version,
        to_schema=CURRENT_BRICK_SCHEMA_VERSION,
        changed=artifact_identity != source_identity,
        destination=destination_path,
    )


__all__ = [
    "CURRENT_BRICK_SCHEMA_VERSION", "MIGRATION_PROTOCOL", "MigrationReport",
    "migrate_external_brick_manifest",
]
