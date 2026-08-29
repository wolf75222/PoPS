"""GeneratedModule metadata emission for compiled Program artifacts."""
from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, cast

from pops.codegen.program_models import ProgramModelGraph


def program_module_records(program: Any, model: Any = None) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    """Return the owner-qualified records carried by the v5 Program candidate POD.

    Kept beside the legacy text emitter so both paths derive from the same declaration inventory;
    the v5 Program installer no longer needs one ``dlsym`` per record.
    """
    del program
    operators, states, fields = [], [], []
    seen_states, seen_fields = set(), set()
    if type(model) is ProgramModelGraph:
        items = [(owner, emit_model, model.source_modules_by_owner.get(owner) or emit_model)
                 for owner, emit_model in sorted(model.models_by_owner.items(), key=lambda item: str(item[0]))]
    elif model is not None:
        items = [(getattr(model, "owner_path", ""), model, model)]
    else:
        items = []
    for owner, emit_model, declared_module in items:
        canonical = getattr(owner, "canonical", None)
        owner_name = str(canonical() if callable(canonical) else owner)
        if hasattr(emit_model, "operator_registry"):
            registry = emit_model.operator_registry()
            for op in (registry.get(name) for name in registry.names()):
                operators.append((op.name, op.kind, repr(op.signature),
                                  json.dumps({**op.requirements, "kind": op.kind}, sort_keys=True), owner_name))
        for space in _declared_spaces(declared_module, "state_spaces", "state_space"):
            identity = (owner_name, _space_identity(space))
            if identity not in seen_states:
                seen_states.add(identity)
                states.append((space.name, "state-space", "", "{}", owner_name))
        for space in _declared_spaces(declared_module, "field_spaces", "field_space"):
            identity = (owner_name, _space_identity(space))
            if identity not in seen_fields:
                seen_fields.add(identity)
                fields.append((space.name, "field-space", "", "{}", owner_name))
    return tuple(operators), tuple(states), tuple(fields)


def _declared_spaces(authority: Any, plural: str, singular: str) -> tuple[Any, ...]:
    """Return every declared space, preserving its module declaration order."""
    accessor = getattr(authority, plural, None)
    if callable(accessor):
        declared = accessor()
        values_accessor = getattr(declared, "values", None)
        values = values_accessor() if callable(values_accessor) else declared
        return tuple(cast(Iterable[Any], values))
    accessor = getattr(authority, singular, None)
    return (accessor(),) if callable(accessor) else ()


def _space_identity(space: Any) -> Any:
    """Stable structural identity for owner-local metadata deduplication."""
    try:
        hash(space)
    except TypeError:
        payload = space.to_data() if hasattr(space, "to_data") else repr(space)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return space


__all__ = ["program_module_records"]
