"""Recursive sealing helpers for compiled artifact records."""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def freeze_artifact_value(value: Any) -> Any:
    """Replace mutable containers by immutable equivalents and seal nested values."""
    if isinstance(value, Mapping):
        return MappingProxyType({
            freeze_artifact_value(key): freeze_artifact_value(item)
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(freeze_artifact_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_artifact_value(item) for item in value)
    freeze = getattr(value, "freeze", None)
    if callable(freeze):
        result = freeze()
        if result is not None and result is not value:
            raise TypeError("artifact member freeze() must return self")
    seal = getattr(value, "_seal", None)
    if callable(seal) and not getattr(value, "_sealed", False):
        seal()
    return value


def seal_attributes(owner: Any, *, skip: Any = ()) -> None:
    """Deep-freeze every stored attribute on @p owner before setting ``_sealed``."""
    ignored = set(skip) | {"_sealed"}
    for name, value in tuple(getattr(owner, "__dict__", {}).items()):
        if name not in ignored:
            object.__setattr__(owner, name, freeze_artifact_value(value))
    object.__setattr__(owner, "_sealed", True)


__all__ = ["freeze_artifact_value", "seal_attributes"]
