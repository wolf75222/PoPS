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
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _encode(value: Any) -> bytes:
    # Authoring identities can contain deeply nested expression trees. Traverse
    # explicitly: valid depth must not depend on Python's process-wide recursion
    # limit. Sequence/map chunks stream into one buffer instead of repeatedly
    # copying the entire encoded subtree at every ancestor.
    chunks: list[bytes] = []
    active: set[int] = set()
    pending = [("value", value, "$", chunks)]
    while pending:
        action, value, path, output = pending.pop()
        if action == "leave":
            active.remove(id(value))
            continue
        if action == "bytes":
            output.append(value)
            continue
        if action == "set":
            items = [b"".join(parts) for parts in value]
            items.sort(key=lambda item: (len(item), item))
            if any(left == right for left, right in zip(items, items[1:], strict=False)):
                raise ValueError("canonical CBOR set at %s contains duplicate canonical values" % path)
            output.extend(items)
            continue
        if value is None:
            output.append(b"\xf6")
        elif value is False:
            output.append(b"\xf4")
        elif value is True:
            output.append(b"\xf5")
        elif isinstance(value, int):
            if value < _INT64_MIN or value > _INT64_MAX:
                raise OverflowError("canonical CBOR integer at %s is outside signed int64" % path)
            output.append(_head(0, value) if value >= 0 else _head(1, -1 - value))
        elif isinstance(value, str):
            output.append(_encode_string(value, path))
        elif isinstance(value, bytes):
            output.extend((_head(2, len(value)), value))
        elif isinstance(value, (list, tuple, dict, set, frozenset)):
            marker = id(value)
            if marker in active:
                raise ValueError("canonical CBOR cannot encode a reference cycle at %s" % path)
            active.add(marker)
            pending.append(("leave", value, path, output))
            if isinstance(value, (list, tuple)):
                output.append(_head(4, len(value)))
                for index in range(len(value) - 1, -1, -1):
                    pending.append(("value", value[index], "%s[%d]" % (path, index), output))
            elif isinstance(value, dict):
                entries = []
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise TypeError("canonical CBOR map key at %s must be a string" % path)
                    entries.append((_encode_string(key, "%s{key}" % path), key, item))
                entries.sort(key=lambda entry: (len(entry[0]), entry[0]))
                output.append(_head(5, len(entries)))
                for key_bytes, key, item in reversed(entries):
                    pending.append(("value", item, "%s.%s" % (path, key), output))
                    pending.append(("bytes", key_bytes, path, output))
            else:
                # Set ordering depends on complete member encodings. These
                # members alone need separate buffers until the set closes.
                members = [(item, []) for item in value]
                output.extend((_SET_TAG, _head(4, len(members))))
                pending.append(("set", [parts for _, parts in members], path, output))
                for item, parts in reversed(members):
                    pending.append(("value", item, "%s{item}" % path, parts))
        elif isinstance(value, float):
            raise TypeError(
                "canonical CBOR refuses float at %s; identity layers must project binary64 "
                "values to float.hex() strings" % path
            )
        else:
            raise TypeError(
                "canonical CBOR cannot encode opaque %s at %s" % (type(value).__name__, path)
            )
    return b"".join(chunks)


def _encode_string(value: str, path: str) -> bytes:
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("canonical CBOR string at %s is not valid Unicode" % path) from exc
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
