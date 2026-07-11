"""Validation and immutable normalization of brick-library manifest rows."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops._manifest_immutability import freeze_manifest_json


_BRICK_KEYS = frozenset({
    "id", "brick_type", "category", "scheme", "native_id", "available",
    "requirements", "capabilities", "options",
})


def freeze_bricks(bricks: Any) -> tuple:
    """Validate, detach and deterministically order nested brick records."""
    try:
        rows = list(bricks)
    except TypeError:
        raise TypeError("LibraryManifest bricks must be an iterable of mappings") from None
    frozen = []
    ids = []
    for index, row in enumerate(rows):
        where = "LibraryManifest.bricks[%d]" % index
        if not isinstance(row, Mapping):
            raise TypeError("%s must be a mapping" % where)
        if any(not isinstance(key, str) for key in row):
            raise TypeError("%s keys must be strings" % where)
        missing = sorted(_BRICK_KEYS - set(row))
        unknown = sorted(set(row) - _BRICK_KEYS)
        if missing:
            raise ValueError("%s is missing field(s) %s" % (where, missing))
        if unknown:
            raise ValueError("%s has unknown field(s) %s" % (where, unknown))
        if not isinstance(row["id"], str) or not row["id"]:
            raise TypeError("%s.id must be a non-empty string" % where)
        if not isinstance(row["available"], bool):
            raise TypeError("%s.available must be a bool" % where)
        for key in ("brick_type", "category", "native_id"):
            if not isinstance(row[key], str):
                raise TypeError("%s.%s must be a string" % (where, key))
        for key in ("requirements", "capabilities", "options"):
            if not isinstance(row[key], Mapping):
                raise TypeError("%s.%s must be a mapping" % (where, key))
        ids.append(row["id"])
        frozen.append(freeze_manifest_json(row, where=where))
    if len(set(ids)) != len(ids):
        raise ValueError("LibraryManifest bricks contain duplicate ids")
    return tuple(sorted(frozen, key=lambda entry: entry["id"]))


__all__ = ["freeze_bricks"]
