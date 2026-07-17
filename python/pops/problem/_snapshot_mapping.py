"""Order-preserving canonical projection for authoring mappings."""
from __future__ import annotations

import json
from typing import Any


def canonical_mapping(
    value: Any,
    *,
    path: str,
    active: set[int],
    handle_resolver: Any,
    artifact: bool,
    canonical: Any,
) -> Any:
    """Preserve observable order/type and reject ambiguous canonical keys."""
    if type(value) is dict and all(isinstance(key, str) for key in value):
        return {
            key: canonical(
                item,
                path="%s.%s" % (path, key),
                active=active,
                handle_resolver=handle_resolver,
                artifact=artifact,
            )
            for key, item in value.items()
        }
    entries = []
    canonical_keys: set[str] = set()
    for key, item in value.items():
        canonical_key = canonical(
            key, path="%s{key}" % path, active=active,
            handle_resolver=handle_resolver, artifact=artifact)
        key_token = json.dumps(
            canonical_key, sort_keys=False, separators=(",", ":"), allow_nan=False)
        if key_token in canonical_keys:
            raise ValueError(
                "AuthoringSnapshot mapping at %s contains distinct keys with the same "
                "canonical identity" % path)
        canonical_keys.add(key_token)
        entries.append([
            canonical_key,
            canonical(
                item, path="%s{%r}" % (path, key), active=active,
                handle_resolver=handle_resolver, artifact=artifact),
        ])
    return {"$mapping": {
        "type": "%s.%s" % (type(value).__module__, type(value).__qualname__),
        "entries": entries,
    }}


__all__ = ["canonical_mapping"]
