"""Small structural contracts used by AMR descriptors without reversing package layers."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _projection(value: Any, method: str, *, where: str) -> Mapping[str, Any]:
    project = getattr(value, method, None)
    if not callable(project):
        raise TypeError("%s requires %s()" % (where, method))
    data = project()
    if not isinstance(data, Mapping):
        raise TypeError("%s.%s() must return a mapping" % (where, method))
    return data


def canonical_handle(value: Any, *, where: str, kinds: str | frozenset[str]) -> Any:
    """Validate the immutable owner-qualified Handle protocol, not its defining package."""
    expected = frozenset((kinds,)) if isinstance(kinds, str) else kinds
    if getattr(value, "is_resolved", None) is not True:
        raise TypeError("%s requires a canonical owner-qualified Handle" % where)
    identity = _projection(value, "canonical_identity", where=where)
    kind = identity.get("kind")
    if kind not in expected or getattr(value, "kind", None) != kind:
        raise TypeError("%s requires Handle.kind in %r, got %r" % (where, sorted(expected), kind))
    if not isinstance(identity.get("qualified_id"), str) or not identity["qualified_id"]:
        raise TypeError("%s requires a qualified Handle identity" % where)
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError("%s Handle must be immutable and hashable" % where) from exc
    return value


def clock_data(value: Any, *, where: str, require_owner: bool = True) -> Mapping[str, Any]:
    data = _projection(value, "to_data", where=where)
    if not isinstance(data.get("name"), str) or not data["name"]:
        raise TypeError("%s requires a named Clock" % where)
    if require_owner and data.get("owner") is None:
        raise TypeError("%s must be owner-qualified" % where)
    if not isinstance(getattr(value, "qualified_id", None), str):
        raise TypeError("%s requires a stable Clock qualified_id" % where)
    return data


def event_data(value: Any, *, where: str) -> Mapping[str, Any]:
    data = _projection(value, "to_data", where=where)
    if data.get("owner") is None or not isinstance(data.get("local_id"), str):
        raise TypeError("%s requires an owner-qualified EventHandle" % where)
    if getattr(value, "owner", None) is None or getattr(value, "local_id", None) != data["local_id"]:
        raise TypeError("%s EventHandle projection is inconsistent" % where)
    return data


def schedule_data(value: Any, *, where: str) -> Mapping[str, Any]:
    data = _projection(value, "to_data", where=where)
    domain = data.get("domain")
    trigger = data.get("trigger")
    if not isinstance(domain, Mapping) or not isinstance(trigger, Mapping):
        raise TypeError("%s requires a typed Schedule projection" % where)
    clock_data(getattr(value, "clock", None), where="%s.clock" % where)
    if data.get("off") is not None:
        raise ValueError("regrid schedules are event cadences and cannot define an off policy")
    return data


def time_point_data(value: Any, *, where: str) -> Mapping[str, Any]:
    data = _projection(value, "to_data", where=where)
    step = getattr(value, "step", None)
    if isinstance(step, bool) or not isinstance(step, int) or data.get("step") != step:
        raise TypeError("%s requires an exact integer-indexed TimePoint" % where)
    if data.get("clock") != clock_data(getattr(value, "clock", None), where="%s.clock" % where):
        raise TypeError("%s TimePoint clock projection is inconsistent" % where)
    return data


def transaction_data(value: Any, *, where: str) -> Mapping[str, Any]:
    data = _projection(value, "to_data", where=where)
    for name in ("status", "phase", "action"):
        if not isinstance(getattr(value, name, None), str) or data.get(name) != getattr(value, name):
            raise TypeError("%s requires a typed transaction %s" % (where, name))
    return data


__all__ = [
    "canonical_handle",
    "clock_data",
    "event_data",
    "schedule_data",
    "time_point_data",
    "transaction_data",
]
