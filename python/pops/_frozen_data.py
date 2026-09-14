"""Private dependency-neutral copying for immutable containers and semantic payloads."""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def freeze_containers(value: Any) -> Any:
    """Detach container storage while preserving non-container authoring leaves.

    This does not validate a canonical payload or invoke leaf ``freeze`` hooks;
    Case and Program own those validation and transaction boundaries.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({
            freeze_containers(key): freeze_containers(item)
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(freeze_containers(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_containers(item) for item in value)
    return value


def freeze_data(value: Any, where: str) -> Any:
    """Recursively freeze canonical non-floating payload data."""
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value
    if isinstance(value, float):
        raise TypeError("%s cannot contain binary floats" % where)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("%s mapping keys must be non-empty strings" % where)
        return MappingProxyType({
            key: freeze_data(item, "%s.%s" % (where, key))
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(freeze_data(item, "%s[]" % where) for item in value)
    raise TypeError("%s contains opaque %s" % (where, type(value).__name__))


def thaw_data(value: Any) -> Any:
    """Return plain mutable containers for one frozen semantic payload."""
    if isinstance(value, Mapping):
        return {key: thaw_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_data(item) for item in value]
    return value


__all__ = ["freeze_containers", "freeze_data", "thaw_data"]
