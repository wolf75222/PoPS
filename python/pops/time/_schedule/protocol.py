"""Shared strict helpers for immutable schedule extension components."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any, TypeVar
from urllib.parse import urlsplit


def _identity(uri: Any, version: Any, *, where: str) -> tuple[str, int]:
    if not isinstance(uri, str) or not uri or uri.strip() != uri \
            or any(character.isspace() for character in uri):
        raise TypeError("%s URI must be canonical non-empty text" % where)
    try:
        parsed = urlsplit(uri)
    except ValueError:
        parsed = None
    if parsed is None or not parsed.scheme or not parsed.netloc \
            or parsed.query or parsed.fragment:
        raise ValueError(
            "%s URI must be absolute, namespaced, and contain no query or fragment" % where)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("%s version must be an integer >= 1" % where)
    return uri, version


_ComponentType = TypeVar("_ComponentType", bound=type[Any])


def stable_component_identity(
    uri: str, version: int = 1
) -> Callable[[_ComponentType], _ComponentType]:
    """Pin a builtin schedule component to a versioned semantic URI."""
    declared = _identity(uri, version, where="stable schedule component")

    def decorate(component_type: _ComponentType) -> _ComponentType:
        component_type.__pops_component_identity__ = declared
        return component_type

    return decorate


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


def component_identity(value: Any) -> dict[str, Any]:
    component_type = type(value)
    declared = component_type.__dict__.get("__pops_component_identity__")
    if declared is None:
        if "component_uri" not in component_type.__dict__ \
                or "component_version" not in component_type.__dict__:
            raise TypeError(
                "schedule extension %s.%s must declare its own component_uri and "
                "component_version; Python module paths are not semantic identities"
                % (component_type.__module__, component_type.__qualname__))
        declared = (
            component_type.__dict__["component_uri"],
            component_type.__dict__["component_version"],
        )
    if not isinstance(declared, tuple) or len(declared) != 2:
        raise TypeError("schedule component declares an invalid semantic identity")
    uri, version = _identity(*declared, where="schedule component")
    return {"uri": uri, "version": version}


class UnresolvedScheduleCondition(TypeError):
    """A consumer trigger still carries an authoring-time rather than resolved predicate."""

    def __init__(self, condition: Any) -> None:
        self.condition_type = type(condition).__name__
        super().__init__("consumer schedule condition is unresolved: %s" % self.condition_type)


__all__ = [
    "component_payload",
    "map_component",
    "manifest_value",
    "stable_component_identity",
    "component_identity",
    "UnresolvedScheduleCondition",
]
