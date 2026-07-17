"""Small builder helpers for immutable module manifests."""
from __future__ import annotations

from collections.abc import Iterable
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
        raw_keys = keys()
        if isinstance(raw_keys, (str, bytes)) or not isinstance(raw_keys, Iterable):
            raise TypeError("signature bundle keys() must return an iterable of strings")
        bundle_keys = tuple(raw_keys)
        if any(not isinstance(key, str) for key in bundle_keys):
            raise TypeError("signature bundle keys() must return only strings")
        return "RateBundle{%s}" % ", ".join(bundle_keys)
    raise TypeError(
        "signature item %s has no stable manifest name; provide name, "
        "domain_name/range_name, or a bundle keys() protocol" % type(item).__name__)


def state_space_row(space: Any) -> Any:
    return {
        "components": list(space.components),
        "roles": dict(getattr(space, "roles", {}) or {}),
        "layout": getattr(space, "layout", "cell"),
        "storage": getattr(space, "storage", "multifab"),
        "representation": space.representation,
        "centering": space.centering,
        "units": list(space.units),
        "frame": space.frame,
        "clock": space.clock,
    }


def field_space_row(space: Any) -> Any:
    return {
        "components": list(space.components),
        "layout": getattr(space, "layout", "cell"),
        "representation": space.representation,
        "centering": space.centering,
        "units": list(space.units),
        "frame": space.frame,
        "clock": space.clock,
    }


def param_row(param: Any) -> Any:
    """Lossless canonical declaration row used by manifests and bind-schema resolution."""
    from pops.params import ParameterDeclaration

    if not isinstance(param, ParameterDeclaration):
        raise TypeError("parameter manifest rows require a canonical ParameterDeclaration")
    return param.bind_data()


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
    from pops.runtime.brick_catalog import brick_catalog, catalog_info
    from pops.runtime.routes import ROUTE_REGISTRY_VERSION

    return catalog_info() | {
        "route_registry_version": ROUTE_REGISTRY_VERSION,
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
