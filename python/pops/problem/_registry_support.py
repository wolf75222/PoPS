"""Shared, inert helpers for the typed Case registries."""
from __future__ import annotations

from typing import Any


# Distinguishes no ``kind=`` from ``kind=None``. ParamRegistry rejects either explicit form.
def strict_name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("%s must be a non-empty string" % where)
    return value


def identity_atom(value: Any, _seen: Any = None) -> Any:
    """Return a deterministic, hashable identity for one inert declaration value."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    seen = set() if _seen is None else _seen
    qualified_id = getattr(value, "qualified_id", None)
    if isinstance(qualified_id, str) and qualified_id:
        return ("handle", qualified_id)
    object_id = id(value)
    if object_id in seen:
        return ("cycle", type(value).__module__, type(value).__qualname__)
    seen.add(object_id)
    try:
        return _compound_identity_atom(value, seen)
    finally:
        seen.remove(object_id)


def _compound_identity_atom(value: Any, seen: Any) -> Any:
    if isinstance(value, dict):
        entries = [
            (identity_atom(key, seen), identity_atom(item, seen))
            for key, item in value.items()
        ]
        return ("dict", tuple(sorted(entries, key=repr)))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(identity_atom(item, seen) for item in value))
    if isinstance(value, (set, frozenset)):
        entries = [identity_atom(item, seen) for item in value]
        return (type(value).__name__, tuple(sorted(entries, key=repr)))
    public_state = getattr(value, "__dict__", None)
    if isinstance(public_state, dict):
        return (
            "object",
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                sorted(
                    (key, identity_atom(item, seen))
                    for key, item in public_state.items()
                    if not key.startswith("_")
                )
            ),
        )
    return ("typed-value", type(value).__module__, type(value).__qualname__, repr(value))


def descriptor_declaration_key(value: Any) -> Any:
    """Semantic key used to reject duplicate policy/diagnostic declarations."""
    return (
        getattr(value, "category", None),
        getattr(value, "name", type(value).__name__),
        identity_atom(value),
    )


__all__ = ["descriptor_declaration_key", "identity_atom", "strict_name"]
