"""Canonical symbolic coordinates for owner-qualified state spaces."""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def state_component_symbol(space: Any, component: Any) -> str:
    """Return an unambiguous C++-safe symbol for one state-space component.

    Display names remain unrestricted user-facing identifiers.  The executable IR
    needs a distinct local coordinate when several spaces contain the same physical
    component name, so both UTF-8 names are encoded losslessly instead of sanitized
    (which could collide).
    """
    space_name = getattr(space, "name", space)
    if not isinstance(space_name, str) or not space_name:
        raise TypeError("state symbol requires a named StateSpace")
    if not isinstance(component, str) or not component:
        raise TypeError("state symbol component must be a non-empty string")
    return "pops_state_s_%s_c_%s" % (
        space_name.encode("utf-8").hex(), component.encode("utf-8").hex())


def rebind_state_symbols(value: Any, state: Any, spaces: Any) -> Any:
    """Clone an operator body for one single-state backend coordinate system.

    The canonical Module IR keeps state coordinates qualified.  A block kernel
    intentionally sees one StateSpace, whose legacy arithmetic backend uses bare
    local component names.  This target-lowering step is therefore explicit and
    rejects a coordinate owned by a different state instead of aliasing it.
    """
    from pops._ir.expr import Expr, Var

    selected = {
        state_component_symbol(state, component): component
        for component in state.components
    }
    all_symbols = {
        state_component_symbol(space, component): (space.name, component)
        for space in spaces for component in space.components
    }
    memo: dict[int, Any] = {}

    def _clone(item: Any) -> Any:
        if isinstance(item, Var):
            component = selected.get(item.name)
            if component is not None:
                return Var(component, item.kind)
            foreign = all_symbols.get(item.name)
            if foreign is not None:
                raise ValueError(
                    "single-state lowering for %r cannot read component %r owned by StateSpace %r; "
                    "declare a multi-state operator"
                    % (state.name, foreign[1], foreign[0]))
            return item
        if isinstance(item, Expr):
            cached = memo.get(id(item))
            if cached is not None:
                return cached
            clone = object.__new__(type(item))
            memo[id(item)] = clone
            for base in reversed(type(item).__mro__):
                slots = base.__dict__.get("__slots__", ())
                if isinstance(slots, str):
                    slots = (slots,)
                for slot in slots:
                    if slot in ("__dict__", "__weakref__", "_pops_symbolic_initializing") \
                            or not hasattr(item, slot):
                        continue
                    object.__setattr__(clone, slot, _clone(getattr(item, slot)))
            state_data = getattr(item, "__dict__", None)
            if isinstance(state_data, dict):
                for name, child in state_data.items():
                    object.__setattr__(clone, name, _clone(child))
            object.__setattr__(clone, "_pops_symbolic_initializing", False)
            return clone
        if isinstance(item, tuple):
            return tuple(_clone(child) for child in item)
        if isinstance(item, list):
            return [_clone(child) for child in item]
        if isinstance(item, Mapping):
            result = {_clone(key): _clone(child) for key, child in item.items()}
            return MappingProxyType(result) if isinstance(item, MappingProxyType) else result
        if isinstance(item, frozenset):
            return frozenset(_clone(child) for child in item)
        if isinstance(item, set):
            return {_clone(child) for child in item}
        return item

    return _clone(value)


__all__ = ["rebind_state_symbols", "state_component_symbol"]
