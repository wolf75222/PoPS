"""Shared strict helpers for immutable schedule extension components."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any


def component_payload(value: Any, excluded: frozenset[str]) -> dict[str, Any]:
    """Return every declared extension field, without fallback attribute lookup."""
    component_type = type(value)
    params = component_type.__dict__.get("__dataclass_params__")
    slots = component_type.__dict__.get("__slots__")
    if not is_dataclass(value) or params is None or params.frozen is not True or slots is None:
        raise TypeError(
            "schedule extension %s.%s must be declared as @dataclass(frozen=True, slots=True)"
            % (component_type.__module__, component_type.__qualname__)
        )
    return {
        field.name: object.__getattribute__(value, field.name)
        for field in fields(value)
        if field.name not in excluded
    }


def map_component(
    value: Any, mapper: Callable[[Any], Any], excluded: frozenset[str], **fixed: Any
) -> Any:
    updates = {name: mapper(item) for name, item in component_payload(value, excluded).items()}
    updates.update(fixed)
    return replace(value, **updates)


def manifest_value(value: Any) -> Any:
    """Normalize containers while leaving typed extension leaves explicit."""
    if isinstance(value, Mapping):
        return {key: manifest_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [manifest_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((manifest_value(item) for item in value), key=repr)
    return value


def component_identity(value: Any) -> dict[str, str]:
    component_type = type(value)
    return {"module": component_type.__module__, "qualname": component_type.__qualname__}


class UnresolvedScheduleCondition(TypeError):
    """A consumer trigger still carries an authoring-time rather than resolved predicate."""

    def __init__(self, condition: Any) -> None:
        self.condition_type = type(condition).__name__
        super().__init__("consumer schedule condition is unresolved: %s" % self.condition_type)


__all__ = [
    "component_payload",
    "map_component",
    "manifest_value",
    "component_identity",
    "UnresolvedScheduleCondition",
]
