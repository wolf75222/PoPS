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
        return tuple(sorted(items, key=lambda item: json.dumps(
            thaw_json(item), sort_keys=True, separators=(",", ":"))))
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
        "%s contains non-JSON value %r; implement to_data() on descriptor metadata"
        % (where, value))


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


__all__ = ["freeze_json", "require_manifest_id", "require_manifest_name", "thaw_json"]
