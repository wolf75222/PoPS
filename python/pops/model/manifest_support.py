"""Small builder helpers for immutable module manifests."""
from __future__ import annotations

from typing import Any


def space_name(item: Any) -> Any:
    """Stable display name of a signature item in a manifest row."""
    name = getattr(item, "name", None)
    if name is not None:
        return name
    domain = getattr(item, "domain_name", None)
    range_ = getattr(item, "range_name", None)
    if domain is not None and range_ is not None:
        return "%s->%s" % (domain, range_)
    keys = getattr(item, "keys", None)
    if callable(keys):
        return "RateBundle{%s}" % ", ".join(keys())
    raise TypeError(
        "signature item %s has no stable manifest name; provide name, "
        "domain_name/range_name, or a bundle keys() protocol" % type(item).__name__)


def state_space_row(space: Any) -> Any:
    return {
        "components": list(space.components),
        "roles": dict(getattr(space, "roles", {}) or {}),
        "layout": getattr(space, "layout", "cell"),
        "storage": getattr(space, "storage", "multifab"),
    }


def field_space_row(space: Any) -> Any:
    return {
        "components": list(space.components),
        "layout": getattr(space, "layout", "cell"),
    }


def param_row(param: Any) -> Any:
    """Manifest row containing a lossless parameter default literal."""
    from pops.ir.literals import scalar_literal

    default = getattr(param, "default", getattr(param, "value", None))
    dtype = getattr(param, "dtype", "real")
    literal = scalar_literal(default, target=dtype)
    row = {"default": literal.to_data(), "dtype": str(getattr(dtype, "name", dtype))}
    kind = getattr(param, "kind", None)
    if kind is not None:
        row["kind"] = kind
    return row


def params_utilization(params: Any) -> Any:
    """Runtime-parameter capacity usage stored in the manifest."""
    from pops.physics.aux import max_runtime_params

    limit = max_runtime_params()
    count = sum(1 for row in params.values() if row.get("kind") == "runtime")
    status = "ok" if count < limit else ("at_limit" if count == limit else "exceeded")
    return {"count": count, "limit": limit, "status": status}


def native_routes() -> Any:
    from pops.runtime.routes import (
        ROUTE_REGISTRY_VERSION,
        route_registry_hash,
        route_registry_signature,
    )

    return {
        "version": ROUTE_REGISTRY_VERSION,
        "hash": route_registry_hash(),
        "signature": route_registry_signature(),
    }


def native_catalog() -> Any:
    from pops.runtime.brick_catalog import brick_catalog
    from pops.runtime.routes import ROUTE_REGISTRY_VERSION

    return {
        "version": ROUTE_REGISTRY_VERSION,
        "bricks": [entry["id"] for entry in brick_catalog()],
    }


__all__ = [
    "field_space_row",
    "native_catalog",
    "native_routes",
    "param_row",
    "params_utilization",
    "space_name",
    "state_space_row",
]
