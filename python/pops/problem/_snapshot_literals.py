"""Authenticated JSON projection for scalar literal records."""
from __future__ import annotations

import math
from enum import Enum
from typing import Any


def canonical_literal_data(data: Any, *, path: str) -> Any:
    """Canonicalize a JSON-shaped ScalarLiteral view without re-tagging its integers."""
    if isinstance(data, dict):
        if not all(isinstance(key, str) for key in data):
            raise TypeError("ScalarLiteral.to_data() requires string keys at %s" % path)
        return {
            key: canonical_literal_data(item, path="%s.%s" % (path, key))
            for key, item in data.items()
        }
    if isinstance(data, (list, tuple)):
        return [
            canonical_literal_data(item, path="%s[%d]" % (path, index))
            for index, item in enumerate(data)
        ]
    if isinstance(data, float) and not math.isfinite(data):
        raise ValueError("ScalarLiteral.to_data() contains a non-finite float at %s" % path)
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    cls = type(data)
    raise TypeError("ScalarLiteral.to_data() is not JSON-ready at %s (got %s.%s)" % (
        path, cls.__module__, cls.__qualname__))


def canonical_enum_data(value: Enum, *, path: str) -> dict[str, Any]:
    """Encode one exact enum member without introspecting its cyclic class dictionary."""
    cls = type(value)
    qualified = "%s.%s" % (cls.__module__, cls.__qualname__)
    return {"$enum": {
        "type": qualified,
        "member": value.name,
        "value": canonical_literal_data(
            value.value, path="%s<%s.%s>" % (path, qualified, value.name)),
    }}


__all__ = ["canonical_enum_data", "canonical_literal_data"]
