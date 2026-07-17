"""Closed, versioned envelopes shared by persisted PoPS manifests.

This module is deliberately dependency-free so model, runtime and external
manifest readers can all use the same wire boundary without import cycles.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


MANIFEST_PROTOCOL = "pops.manifest"
ENVELOPE_KEYS = frozenset({"protocol", "kind", "schema_version", "payload"})


def exact_mapping(value: Any, keys: Any, *, where: str) -> Mapping[str, Any]:
    """Require a mapping with exactly ``keys`` and name both kinds of drift."""
    if not isinstance(value, Mapping):
        raise TypeError("%s must be a mapping" % where)
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(repr(key) for key in actual - expected)
        raise TypeError(
            "%s requires exactly %s (missing=%s, unknown=%s)"
            % (where, sorted(expected), missing, unknown)
        )
    return value


def strict_int(value: Any, *, where: str, minimum: int | None = None) -> int:
    """Validate an integer without accepting bool or coercible strings/floats."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be an integer" % where)
    if minimum is not None and value < minimum:
        raise ValueError("%s must be >= %d" % (where, minimum))
    return value


def strict_string(value: Any, *, where: str, nonempty: bool = True) -> str:
    """Validate a string without stringifying foreign values."""
    if not isinstance(value, str) or (nonempty and not value):
        suffix = " a non-empty string" if nonempty else " a string"
        raise TypeError("%s must be%s" % (where, suffix))
    return value


def manifest_envelope(*, kind: str, schema_version: int, payload: Any) -> dict[str, Any]:
    """Build the one exact envelope used by versioned PoPS manifest values."""
    checked_kind = strict_string(kind, where="manifest kind")
    checked_version = strict_int(schema_version, where="manifest schema_version", minimum=1)
    if not isinstance(payload, Mapping):
        raise TypeError("manifest payload must be a mapping")
    return {
        "protocol": MANIFEST_PROTOCOL,
        "kind": checked_kind,
        "schema_version": checked_version,
        "payload": dict(payload),
    }


def parse_manifest_envelope(
    value: Any,
    *,
    kind: str,
    schema_version: int,
    payload_keys: Any = None,
    where: str,
) -> Mapping[str, Any]:
    """Validate an envelope and return its exact payload mapping."""
    row = exact_mapping(value, ENVELOPE_KEYS, where=where)
    protocol = strict_string(row["protocol"], where="%s protocol" % where)
    if protocol != MANIFEST_PROTOCOL:
        raise ValueError(
            "%s protocol %r is unsupported (expected %r)"
            % (where, protocol, MANIFEST_PROTOCOL)
        )
    actual_kind = strict_string(row["kind"], where="%s kind" % where)
    if actual_kind != kind:
        raise ValueError("%s kind %r is unsupported (expected %r)" % (where, actual_kind, kind))
    actual_version = strict_int(row["schema_version"], where="%s schema_version" % where)
    if actual_version != schema_version:
        raise ValueError(
            "unsupported %s schema_version %r (expected %d)"
            % (where, actual_version, schema_version)
        )
    payload = row["payload"]
    if payload_keys is None:
        if not isinstance(payload, Mapping):
            raise TypeError("%s payload must be a mapping" % where)
        return payload
    return exact_mapping(payload, payload_keys, where="%s payload" % where)


def strict_json_loads(text: Any, *, where: str = "manifest JSON") -> Any:
    """Decode JSON while refusing duplicate keys and non-standard constants."""
    if not isinstance(text, (str, bytes, bytearray)):
        raise TypeError("%s must be str, bytes, or bytearray" % where)

    def object_pairs(pairs: Any) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("%s contains duplicate object key %r" % (where, key))
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise ValueError("%s contains non-finite constant %s" % (where, value))

    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=constant)


__all__ = [
    "ENVELOPE_KEYS", "MANIFEST_PROTOCOL", "exact_mapping", "manifest_envelope",
    "parse_manifest_envelope", "strict_int", "strict_json_loads", "strict_string",
]
