"""Strict deterministic CBOR encoding for PoPS identities.

This is deliberately a narrow value language, not a general CBOR serializer.  Identity layers must
project their domain objects to these values before encoding them.  In particular floats, arbitrary
objects and extension hooks are refused, so Python and C++ can produce the same bytes without
depending on ``repr``, JSON formatting, pickle, or process-local implementation details.
"""
from __future__ import annotations

import hashlib
from typing import Any


_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_SET_TAG = b"\xd9\x01\x02"  # RFC 8746 set tag 258 in preferred CBOR serialization.


def canonical_bytes(value: Any) -> bytes:
    """Encode one supported value using strict deterministic CBOR.

    Supported values are ``None``, booleans, signed int64, Unicode strings, bytes, ordered
    lists/tuples, string-keyed dictionaries, and sets/frozensets.  Dictionary keys and set members
    are sorted by ``(encoded length, encoded bytes)``.  Cycles and opaque values fail loudly.
    """
    return _encode(value)


def canonical_sha256(value: Any) -> str:
    """Return the lowercase SHA-256 hex digest of :func:`canonical_bytes`."""
    return hashlib.sha256(_encode_buffer(value)).hexdigest()


def _encode(value: Any) -> bytes:
    return bytes(_encode_buffer(value))


def _diagnostic_path(path) -> str:
    """Materialize a path only on failure, without quadratic prefix copies."""
    parts = []
    while path is not None:
        path, suffix = path
        parts.append(suffix)
    return "$" + "".join(reversed(parts))


def _encode_buffer(value: Any) -> bytearray:
    # Authoring identities can contain deeply nested expression trees. Traverse
    # explicitly: valid depth must not depend on Python's process-wide recursion
    # limit. Sequence/map chunks stream into one buffer instead of repeatedly
    # copying the entire encoded subtree at every ancestor.
    chunks = bytearray()
    active: set[int] = set()
    # Repeated schema keys dominate finite metadata. This is a bounded encoding
    # cache, not a restriction on values: cache misses always encode normally.
    strings: dict[str, bytes] = {}
    pending: list[tuple[str, Any, Any, bytearray]] = [("value", value, None, chunks)]
    while pending:
        action, value, path, output = pending.pop()
        if action == "leave":
            active.remove(id(value))
            continue
        if action == "bytes":
            output.extend(value)
            continue
        if action == "set":
            items = [bytes(parts) for parts in value]
            items.sort(key=lambda item: (len(item), item))
            if any(left == right for left, right in zip(items, items[1:], strict=False)):
                raise ValueError("canonical CBOR set at %s contains duplicate canonical values" % _diagnostic_path(path))
            for item in items:
                output.extend(item)
            continue
        if value is None:
            output.append(0xf6)
        elif value is False:
            output.append(0xf4)
        elif value is True:
            output.append(0xf5)
        elif isinstance(value, int):
            if value < _INT64_MIN or value > _INT64_MAX:
                raise OverflowError("canonical CBOR integer at %s is outside signed int64" % _diagnostic_path(path))
            output.extend(_head(0, value) if value >= 0 else _head(1, -1 - value))
        elif isinstance(value, str):
            encoded = strings.get(value) if type(value) is str else None
            if encoded is None:
                encoded = _encode_string(value, path)
                if type(value) is str and len(strings) < 4096:
                    strings[value] = encoded
            output.extend(encoded)
        elif isinstance(value, bytes):
            output.extend(_head(2, len(value)))
            output.extend(value)
        elif isinstance(value, (list, tuple, dict, set, frozenset)):
            marker = id(value)
            if marker in active:
                raise ValueError("canonical CBOR cannot encode a reference cycle at %s" % _diagnostic_path(path))
            active.add(marker)
            pending.append(("leave", value, path, output))
            if isinstance(value, (list, tuple)):
                output.extend(_head(4, len(value)))
                for index in range(len(value) - 1, -1, -1):
                    pending.append(("value", value[index], (path, "[%d]" % index), output))
            elif isinstance(value, dict):
                entries = []
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise TypeError("canonical CBOR map key at %s must be a string" % _diagnostic_path(path))
                    encoded = strings.get(key) if type(key) is str else None
                    if encoded is None:
                        encoded = _encode_string(key, (path, "{key}"))
                        if type(key) is str and len(strings) < 4096:
                            strings[key] = encoded
                    entries.append((encoded, key, item))
                entries.sort(key=lambda entry: (len(entry[0]), entry[0]))
                output.extend(_head(5, len(entries)))
                for key_bytes, key, item in reversed(entries):
                    pending.append(("value", item, (path, ".%s" % key), output))
                    pending.append(("bytes", key_bytes, path, output))
            else:
                # Set ordering depends on complete member encodings. These
                # members alone need separate buffers until the set closes.
                members = [(item, bytearray()) for item in value]
                output.extend(_SET_TAG)
                output.extend(_head(4, len(members)))
                pending.append(("set", [parts for _, parts in members], path, output))
                for item, parts in reversed(members):
                    pending.append(("value", item, (path, "{item}"), parts))
        elif isinstance(value, float):
            raise TypeError(
                "canonical CBOR refuses float at %s; identity layers must project binary64 "
                "values to float.hex() strings" % _diagnostic_path(path)
            )
        else:
            raise TypeError(
                "canonical CBOR cannot encode opaque %s at %s" % (type(value).__name__, _diagnostic_path(path))
            )
    return chunks


def _encode_string(value: str, path) -> bytes:
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("canonical CBOR string at %s is not valid Unicode" % _diagnostic_path(path)) from exc
    return _head(3, len(encoded)) + encoded


def _head(major: int, argument: int) -> bytes:
    prefix = major << 5
    if argument < 24:
        return bytes((prefix | argument,))
    if argument <= 0xFF:
        return bytes((prefix | 24, argument))
    if argument <= 0xFFFF:
        return bytes((prefix | 25,)) + argument.to_bytes(2, "big")
    if argument <= 0xFFFFFFFF:
        return bytes((prefix | 26,)) + argument.to_bytes(4, "big")
    if argument <= 0xFFFFFFFFFFFFFFFF:
        return bytes((prefix | 27,)) + argument.to_bytes(8, "big")
    raise OverflowError("canonical CBOR length or argument exceeds uint64")


__all__ = ["canonical_bytes", "canonical_sha256"]
