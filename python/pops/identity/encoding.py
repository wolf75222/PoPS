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
    return _encode(value, active=set(), path="$")


def canonical_sha256(value: Any) -> str:
    """Return the lowercase SHA-256 hex digest of :func:`canonical_bytes`."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _encode(value: Any, *, active: set[int], path: str) -> bytes:
    if value is None:
        return b"\xf6"
    if value is False:
        return b"\xf4"
    if value is True:
        return b"\xf5"
    if isinstance(value, int):
        if value < _INT64_MIN or value > _INT64_MAX:
            raise OverflowError("canonical CBOR integer at %s is outside signed int64" % path)
        if value >= 0:
            return _head(0, value)
        return _head(1, -1 - value)
    if isinstance(value, str):
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("canonical CBOR string at %s is not valid Unicode" % path) from exc
        return _head(3, len(encoded)) + encoded
    if isinstance(value, bytes):
        return _head(2, len(value)) + value
    if isinstance(value, (list, tuple)):
        return _encode_container(
            value,
            active=active,
            path=path,
            build=lambda: _head(4, len(value)) + b"".join(
                _encode(item, active=active, path="%s[%d]" % (path, index))
                for index, item in enumerate(value)
            ),
        )
    if isinstance(value, dict):
        return _encode_container(
            value,
            active=active,
            path=path,
            build=lambda: _encode_dict(value, active=active, path=path),
        )
    if isinstance(value, (set, frozenset)):
        return _encode_container(
            value,
            active=active,
            path=path,
            build=lambda: _encode_set(value, active=active, path=path),
        )
    if isinstance(value, float):
        raise TypeError(
            "canonical CBOR refuses float at %s; identity layers must project binary64 "
            "values to float.hex() strings" % path
        )
    raise TypeError(
        "canonical CBOR cannot encode opaque %s at %s" % (type(value).__name__, path)
    )


def _encode_container(value: Any, *, active: set[int], path: str, build: Any) -> bytes:
    marker = id(value)
    if marker in active:
        raise ValueError("canonical CBOR cannot encode a reference cycle at %s" % path)
    active.add(marker)
    try:
        return build()
    finally:
        active.remove(marker)


def _encode_dict(value: dict[Any, Any], *, active: set[int], path: str) -> bytes:
    entries = []
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("canonical CBOR map key at %s must be a string" % path)
        key_bytes = _encode(key, active=active, path="%s{key}" % path)
        item_bytes = _encode(item, active=active, path="%s.%s" % (path, key))
        entries.append((key_bytes, item_bytes))
    entries.sort(key=lambda entry: (len(entry[0]), entry[0]))
    return _head(5, len(entries)) + b"".join(key + item for key, item in entries)


def _encode_set(value: set[Any] | frozenset[Any], *, active: set[int], path: str) -> bytes:
    items = [
        _encode(item, active=active, path="%s{item}" % path)
        for item in value
    ]
    items.sort(key=lambda item: (len(item), item))
    if any(left == right for left, right in zip(items, items[1:], strict=False)):
        raise ValueError("canonical CBOR set at %s contains duplicate canonical values" % path)
    return _SET_TAG + _head(4, len(items)) + b"".join(items)


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
