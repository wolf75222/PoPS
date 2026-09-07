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


def rebind_state_symbols(value: Any, state: Any, spaces: Any, *, module: Any = None,
                         quantity_handles: Any = ()) -> Any:
    """Clone an operator body for one single-state backend coordinate system.

    The canonical Module IR keeps state coordinates qualified.  A block kernel
    intentionally sees one StateSpace, whose legacy arithmetic backend uses bare
    local component names.  This target-lowering step is therefore explicit and
    rejects a coordinate owned by a different state instead of aliasing it.
    """
    from pops._ir.expr import Expr, Var
    from pops._ir.quantity import QuantityRef

    def physical_type(space: Any) -> Any:
        data = space.to_data()
        return {key: item for key, item in data.items() if key != "name"}

    # The blackboard may expose an explicit scientific state name while its legacy
    # single-state emitter calls that same bound storage U. This is an authenticated
    # alias supplied by the owner, never guessed from a component/display name.
    aliases = tuple(quantity_handles)
    if aliases and module is None:
        raise TypeError("quantity aliases require a Module authority")
    for handle in aliases:
        if handle.kind != "state" or handle.owner_path != module.owner_path \
                or physical_type(handle.space) != physical_type(state):
            raise ValueError("native quantity alias must preserve its exact physical state type")

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
        from pops._ir.application import ApplicationProjection
        if isinstance(item, ApplicationProjection):
            if item.application.effects:
                raise TypeError("effectful application requires a resolved native realization")
            return _clone(item.expression)
        if isinstance(item, QuantityRef):
            if module is None:
                raise TypeError("qualified quantity lowering requires its Module authority")
            if item.handle.kind == "state":
                expected = module.state_handle(state)
                alias = next((handle for handle in aliases if item.handle == handle), None)
                if alias is not None:
                    if item.space != alias.space:
                        raise ValueError("quantity alias cannot change its authenticated physical type")
                elif item.handle != expected or item.space != state:
                    raise ValueError("single-state lowering cannot read a foreign qualified quantity")
                return Var(item.component, "cons")
            expected = module.field_handle(item.space)
            if item.handle != expected:
                raise ValueError("field quantity does not match its registered Module authority")
            return Var(item.component, "aux")
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


def native_formula_view(facade: Any, module: Any, *, quantity_handles: Any = ()) -> Any:
    """Bind a private formula-emitter copy while retaining the untouched authored Module.

    Unlike rebuilding a Module through the generic legacy adapter, this preserves
    primitive conversions, spectral hooks and all existing supported model metadata.
    The clone owns its rebound expression containers; authoring graph nodes and
    declaration identities remain the source authority.
    """
    emitter = _native_formula_model_view(facade._m, module, quantity_handles=quantity_handles)
    view = object.__new__(type(facade))
    vars(view).update(vars(facade))
    object.__setattr__(view, "_m", emitter)
    object.__setattr__(view, "_module_cache", module)
    object.__setattr__(view, "_compile_source_module_hash", module.module_hash())
    return view


def _native_formula_model_view(model: Any, module: Any, *, quantity_handles: Any = ()) -> Any:
    states = module.state_spaces()
    if len(states) != 1:
        raise ValueError("a formula-emitter view requires one explicitly selected state route")
    state = next(iter(states.values()))
    emitter = object.__new__(type(model))
    vars(emitter).update({
        name: rebind_state_symbols(value, state, states.values(), module=module,
                                  quantity_handles=quantity_handles)
        for name, value in vars(model).items()
    })
    object.__setattr__(emitter, "_formula_native_bound", True)
    return emitter


def native_formula_carrier_view(model: Any) -> Any:
    """Bind an authenticated raw formula carrier for one private emission only."""
    if getattr(model, "_formula_native_bound", False):
        return model
    module = getattr(model, "_formula_source_module", None)
    if module is None:
        return model
    from .module import Module
    if not isinstance(module, Module) or model.owner_path != module.owner_path:
        raise ValueError("native formula carrier source does not match its Module authority")
    from pops.codegen.component_provider_packs import (
        require_emitter_provider_carrier, resolve_component_provider_packs,
    )
    require_emitter_provider_carrier(model, where="native formula source")
    view = _native_formula_model_view(model, module)
    # The source witness is object-bound. Reattach the same exact Module pack to
    # authenticate the private target instead of reusing another object's witness.
    resolve_component_provider_packs(module).attach(view)
    return view


def native_input_state_component_symbol(input_index: int, component_index: int) -> str:
    """A local emitter coordinate after an exact input-state binding was checked."""
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in (input_index, component_index)):
        raise ValueError("native state coordinates require non-negative integer indices")
    return "pops_input_%d_component_%d" % (input_index, component_index)


__all__ = ["rebind_state_symbols", "state_component_symbol", "native_formula_view"]
