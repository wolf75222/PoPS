"""Strict, side-effect-free validation helpers for board authoring.

The blackboard facade is a translation layer.  Values cross its public boundary
before the lower-level DSL mutates registries, so coercing arbitrary objects with
``str()`` / ``bool()`` or validating only after a builder call makes a failed
declaration observable.  This module keeps those trust-boundary checks small and
shared; :func:`atomic_attrs` is reserved for the few compound declarations whose
lower-level builders necessarily perform more than one mutation.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any


def require_name(value: Any, where: str) -> str:
    """Return a non-empty string without ever coercing an arbitrary object."""
    if not isinstance(value, str):
        raise TypeError("%s must be a non-empty string; got %r" % (where, value))
    if not value:
        raise ValueError("%s must be a non-empty string" % where)
    return value


def require_bool(value: Any, where: str) -> bool:
    """Accept only the two Python Boolean values (``'false'`` is not false)."""
    if not isinstance(value, bool):
        raise TypeError("%s must be bool; got %r" % (where, value))
    return value


def normalize_sequence(value: Any, where: str, *, nonempty: bool = False) -> tuple[Any, ...]:
    """Materialize an authoring sequence while refusing string iteration."""
    if isinstance(value, (str, bytes)):
        raise TypeError("%s must be a sequence, not a string" % where)
    try:
        result = tuple(value)
    except TypeError:
        raise TypeError("%s must be an iterable sequence; got %r" % (where, value)) from None
    if nonempty and not result:
        raise ValueError("%s must contain at least one item" % where)
    return result


def normalize_components(value: Any, where: str) -> tuple[str, ...]:
    """Validate one complete, unique state/field component layout."""
    components = normalize_sequence(value, where, nonempty=True)
    for component in components:
        require_name(component, "%s component" % where)
    if len(set(components)) != len(components):
        raise ValueError("%s components must be unique" % where)
    return components


def normalize_roles(roles: Any, components: tuple[str, ...], where: str) -> dict[str, Any]:
    """Validate optional component-role metadata without truth-value coercion."""
    if roles is None:
        return {}
    if not isinstance(roles, Mapping):
        raise TypeError("%s roles must be a mapping or None" % where)
    result = dict(roles)
    unknown = [key for key in result if key not in components]
    if unknown:
        raise ValueError("%s roles reference unknown component(s) %r" % (where, unknown))
    tokens: dict[str, str] = {}
    for component, role in result.items():
        require_name(component, "%s role component" % where)
        if role is not None:
            from .roles import ComponentRole

            if isinstance(role, str):
                raise TypeError(
                    "%s role for %s requires a typed ComponentRole, not a string"
                    % (where, component))
            if not isinstance(role, ComponentRole):
                raise TypeError(
                    "%s role for %s must implement ComponentRole"
                    % (where, component))
            from .roles import native_role_token
            token = native_role_token(role)
            if token in tokens:
                raise ValueError(
                    "%s roles %s and %s collide on native token %r"
                    % (where, tokens[token], component, token))
            tokens[token] = component
    return result


def normalize_string_mapping(value: Any, where: str) -> dict[str, Any]:
    """Copy a mapping whose public keys are strict, non-empty names."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("%s must be a mapping or None" % where)
    result = dict(value)
    for key in result:
        require_name(key, "%s key" % where)
    return result


def _snapshot(value: Any) -> Any:
    """Copy registry containers recursively while retaining immutable leaf objects."""
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, set):
        return {_snapshot(item) for item in value}
    return value


@contextmanager
def atomic_attrs(*attributes: tuple[Any, str]) -> Any:
    """Restore selected top-level containers if a compound builder raises.

    Public values are fully prevalidated first.  This guard covers exceptional
    failures *inside* a lower-level builder after it has started mutating a DSL
    registry (including injected/future builders), without cloning immutable
    expression/operator leaf objects.
    """
    saved = [(obj, name, _snapshot(getattr(obj, name))) for obj, name in attributes]
    try:
        yield
    except BaseException:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)
        raise


__all__ = [
    "atomic_attrs", "normalize_components", "normalize_roles", "normalize_sequence",
    "normalize_string_mapping", "require_bool", "require_name",
]
