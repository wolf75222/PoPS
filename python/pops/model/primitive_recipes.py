"""Exact immutable local primitive recipes retained by the canonical Module."""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pops._ir.expr import Expr, Var
from pops._ir.visitors import _children


def freeze_primitive_recipes(module: Any, recipes: Any) -> Mapping:
    """Validate a detached recipe graph before publishing it on its source owner."""
    if not isinstance(recipes, Mapping):
        raise TypeError("primitive recipes must be a mapping of names to immutable Expr bodies")
    normalized = dict(recipes)
    if any(not isinstance(name, str) or not name or not isinstance(body, Expr)
           for name, body in normalized.items()):
        raise TypeError("primitive recipes require nonempty names and Expr bodies; callbacks are forbidden")
    dependencies = {name: set() for name in normalized}
    for name, body in normalized.items():
        seen = set()
        def inspect(value: Expr, *, seen: set = seen, name: str = name) -> None:
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, Var) and value.kind == "prim" and value.name in normalized:
                dependencies[name].add(value.name)
            for handle in value.declaration_references():
                if handle.owner_path != module.owner_path:
                    raise ValueError("primitive recipe contains a foreign qualified reference")
            for child in _children(value):
                inspect(child)
        inspect(body)
    complete = set()
    active = []
    def visit(name: str) -> None:
        if name in active:
            raise ValueError("primitive recipe cycle: %s" % " -> ".join((*active, name)))
        if name in complete:
            return
        active.append(name)
        for dependency in sorted(dependencies[name]):
            visit(dependency)
        active.pop()
        complete.add(name)
    for name in normalized:
        visit(name)
    return MappingProxyType(normalized)


def resolve_primitive_recipe(module: Any, name: str) -> Expr | None:
    """Fetch a local ``Var(kind='prim')`` recipe from its exact Module authority.

    The caller keeps the owning Module and resolves only that primitive namespace.
    No global names, callback invocation or alternate physical definition is used.
    """
    from .module import Module
    if not isinstance(module, Module):
        raise TypeError("primitive resolution requires the exact source Module")
    if not isinstance(name, str) or not name:
        raise TypeError("primitive resolution requires a nonempty local name")
    return module.primitive_recipes().get(name)


__all__ = ["resolve_primitive_recipe"]
