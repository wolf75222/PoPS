"""Small JSON-value helpers for immutable public manifests.

Manifests cross the authoring/compiled boundary.  Keeping a caller-owned list or
dict in one of them would therefore make an already authenticated artifact
mutable.  These helpers copy a JSON tree into read-only containers and provide
the inverse detached wire view.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def freeze_manifest_json(value: Any, *, where: str) -> Any:
    """Return an immutable, detached copy of a JSON-compatible value."""
    if isinstance(value, Mapping):
        keys = list(value)
        if any(not isinstance(key, str) or not key for key in keys):
            raise TypeError("%s keys must be non-empty strings" % where)
        frozen = {}
        for key in sorted(keys):
            frozen[key] = freeze_manifest_json(value[key], where="%s.%s" % (where, key))
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_manifest_json(item, where=where) for item in value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("%s contains a non-finite float" % where)
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(
        "%s contains non-JSON value of type %s" % (where, type(value).__name__)
    )


def thaw_manifest_json(value: Any) -> Any:
    """Return a detached plain JSON tree from frozen manifest data."""
    if isinstance(value, Mapping):
        return {key: thaw_manifest_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_manifest_json(item) for item in value]
    return value


def canonical_manifest_json(value: Any) -> str:
    """Canonical JSON used for signed manifest identity."""
    return json.dumps(
        thaw_manifest_json(value),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = ["canonical_manifest_json", "freeze_manifest_json", "thaw_manifest_json"]
