"""Structural Handle collection and canonical resolution for immutable Expr graphs."""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def resolve_expr_references(expr: Any, resolver: Any, memo: dict[int, Any]) -> Any:
    """Clone one immutable Expr graph while resolving typed declaration leaves."""
    from .expr import Var

    if isinstance(expr, Var):
        raise TypeError(
            "Expr.resolve_references cannot resolve free-name Var(%r, %r); use "
            "ValueExpr(declaration_handle) so ownership is explicit" % (expr.name, expr.kind)
        )
    object_id = id(expr)
    cached = memo.get(object_id)
    if cached is not None:
        return cached
    clone = object.__new__(type(expr))
    memo[object_id] = clone
    for base in reversed(type(expr).__mro__):
        slots = base.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot in ("__dict__", "__weakref__", "_pops_symbolic_initializing") or not hasattr(
                expr, slot
            ):
                continue
            object.__setattr__(
                clone,
                slot,
                resolve_reference_value(getattr(expr, slot), resolver, memo),
            )
    state = getattr(expr, "__dict__", None)
    if isinstance(state, dict):
        for name, value in state.items():
            object.__setattr__(clone, name, resolve_reference_value(value, resolver, memo))
    object.__setattr__(clone, "_pops_symbolic_initializing", False)
    return clone


def collect_expr_references(expr: Any, references: list[Any], seen: set[int]) -> None:
    """Collect typed Handle leaves structurally, preserving first-seen graph order."""
    object_id = id(expr)
    if object_id in seen:
        return
    seen.add(object_id)
    for base in reversed(type(expr).__mro__):
        slots = base.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot in ("__dict__", "__weakref__", "_pops_symbolic_initializing") or not hasattr(
                expr, slot
            ):
                continue
            collect_reference_value(getattr(expr, slot), references, seen)
    state = getattr(expr, "__dict__", None)
    if isinstance(state, dict):
        for value in state.values():
            collect_reference_value(value, references, seen)


def collect_reference_value(value: Any, references: list[Any], seen: set[int]) -> None:
    """Collect Handle leaves from one expression attribute or container."""
    from .expr import Expr
    from pops.model import Handle

    if isinstance(value, Handle):
        if all(existing is not value for existing in references):
            references.append(value)
        return
    if isinstance(value, Expr):
        collect_expr_references(value, references, seen)
        return
    protocol = getattr(value, "declaration_references", None)
    if callable(protocol):
        for reference in protocol():
            if not isinstance(reference, Handle):
                raise TypeError(
                    "%s.declaration_references() must return only Handle values"
                    % type(value).__name__
                )
            if all(existing is not reference for existing in references):
                references.append(reference)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            collect_reference_value(key, references, seen)
            collect_reference_value(item, references, seen)
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            collect_reference_value(item, references, seen)


def resolve_reference_value(value: Any, resolver: Any, memo: dict[int, Any]) -> Any:
    """Resolve Handle leaves and immutable containers in an expression attribute."""
    from .expr import Expr
    from pops.model import Handle

    if isinstance(value, Handle):
        resolve = resolver if callable(resolver) else getattr(resolver, "resolve", None)
        if not callable(resolve):
            raise TypeError(
                "Expr.resolve_references requires a callable resolver or an object exposing "
                "resolve(handle)"
            )
        result = resolve(value)
        if not isinstance(result, Handle) or not result.is_resolved:
            raise TypeError("Expr reference resolver must return a canonical pops.model.Handle")
        return result
    if isinstance(value, Expr):
        implementation = getattr(type(value), "resolve_references", None)
        if implementation is not Expr.resolve_references:
            resolved = value.resolve_references(resolver)
            if not isinstance(resolved, Expr):
                raise TypeError("%s.resolve_references() must return an Expr" % type(value).__name__)
            return resolved
        return resolve_expr_references(value, resolver, memo)
    protocol = getattr(value, "resolve_references", None)
    if callable(protocol):
        return protocol(resolver)
    if isinstance(value, tuple):
        return tuple(resolve_reference_value(item, resolver, memo) for item in value)
    if isinstance(value, list):
        return [resolve_reference_value(item, resolver, memo) for item in value]
    if isinstance(value, Mapping):
        resolved = {
            resolve_reference_value(key, resolver, memo): resolve_reference_value(
                item, resolver, memo
            )
            for key, item in value.items()
        }
        return MappingProxyType(resolved) if isinstance(value, MappingProxyType) else resolved
    if isinstance(value, frozenset):
        return frozenset(resolve_reference_value(item, resolver, memo) for item in value)
    if isinstance(value, set):
        return {resolve_reference_value(item, resolver, memo) for item in value}
    return value


__all__ = ["collect_expr_references", "resolve_expr_references"]
