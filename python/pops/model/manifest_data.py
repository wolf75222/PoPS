"""Immutable JSON-tree helpers shared by model manifest value objects."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def freeze_json(value: Any, *, where: str = "manifest") -> Any:
    """Deep-copy JSON data into immutable tuples/mapping proxies.

    Foreign descriptor values extend manifests through ``to_data()``.  Unknown
    objects are refused because an address-bearing repr is not stable identity.
    """
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError("%s keys must be non-empty strings" % where)
            frozen[key] = freeze_json(item, where="%s.%s" % (where, key))
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(freeze_json(item, where=where) for item in value)
    if isinstance(value, (set, frozenset)):
        items = [freeze_json(item, where=where) for item in value]
        return tuple(
            sorted(
                items,
                key=lambda item: json.dumps(thaw_json(item), sort_keys=True, separators=(",", ":")),
            )
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("%s contains a non-finite float" % where)
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    hook = getattr(value, "to_data", None)
    if callable(hook):
        return freeze_json(hook(), where=where)
    raise TypeError(
        "%s contains non-JSON value %r; implement to_data() on descriptor metadata" % (where, value)
    )


def thaw_json(value: Any) -> Any:
    """Return a detached plain JSON tree from deeply frozen manifest data."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    if isinstance(value, frozenset):
        return [thaw_json(item) for item in sorted(value, key=repr)]
    return value


def require_manifest_id(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("OperatorManifestEntry id must be a non-negative integer")


def require_manifest_name(value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("ModuleManifest name must be a non-empty string")


def strict_json_loads(text: Any) -> Any:
    """Decode one manifest JSON document without JSON's permissive fallbacks.

    Python's default decoder accepts duplicate object keys and the non-standard
    ``NaN`` / ``Infinity`` constants.  Both would make a signed manifest
    ambiguous, so the manifest protocol rejects them before schema validation.
    """
    if not isinstance(text, (str, bytes, bytearray)):
        raise TypeError("manifest JSON must be str, bytes, or bytearray")

    def _object(pairs: Any) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("manifest JSON contains duplicate object key %r" % key)
            result[key] = value
        return result

    def _constant(value: str) -> Any:
        raise ValueError("manifest JSON contains non-finite constant %s" % value)

    return json.loads(text, object_pairs_hook=_object, parse_constant=_constant)


__all__ = [
    "freeze_json",
    "require_manifest_id",
    "require_manifest_name",
    "strict_json_loads",
    "thaw_json",
]
