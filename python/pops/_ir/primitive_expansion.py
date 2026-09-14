"""Expand an owner's primitive recipes without changing typed leaves or DAG sharing."""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from .expr import Expr, Var


def expand_primitive_recipes(expressions, recipes):
    """Expand only Var(kind='prim') recipes supplied by the exact owning model."""
    memo, active = {}, []

    def clone(value):
        if isinstance(value, Var) and value.kind == "prim" and value.name in recipes:
            if value.name in active:
                raise ValueError("primitive recipe cycle: " + " -> ".join((*active, value.name)))
            active.append(value.name)
            result = clone(recipes[value.name])
            active.pop()
            return result
        if isinstance(value, Expr):
            if id(value) in memo:
                return memo[id(value)]
            result = object.__new__(type(value))
            memo[id(value)] = result
            for base in reversed(type(value).__mro__):
                slots = base.__dict__.get("__slots__", ())
                for key in (slots,) if isinstance(slots, str) else slots:
                    if key not in ("__dict__", "__weakref__", "_pops_symbolic_initializing") \
                            and hasattr(value, key):
                        object.__setattr__(result, key, clone(getattr(value, key)))
            for key, child in getattr(value, "__dict__", {}).items():
                object.__setattr__(result, key, clone(child))
            object.__setattr__(result, "_pops_symbolic_initializing", False)
            return result
        if isinstance(value, Mapping):
            return MappingProxyType({key: clone(child) for key, child in value.items()})
        if isinstance(value, (tuple, list)):
            return tuple(clone(child) for child in value)
        return value

    return clone(expressions)
