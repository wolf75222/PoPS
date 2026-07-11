"""Resolve legacy untyped Program values against a physical model's typed registry."""
from __future__ import annotations

from typing import Any


def _registry(source: Any) -> Any:
    if source is None:
        return None
    registry = source.operator_registry() if hasattr(source, "operator_registry") else source
    return registry if hasattr(registry, "get") and hasattr(registry, "names") else None


def _operator(registry: Any, name: Any) -> Any:
    try:
        return registry.get(name)
    except KeyError:
        return None


def _append_unique(values: list[Any], candidate: Any) -> None:
    if candidate is not None and candidate not in values:
        values.append(candidate)


def _descriptor_spaces(descriptor: Any, states: list[Any], fields: list[Any]) -> None:
    """Collect State/Field descriptors through their small structural protocols."""
    kind = getattr(descriptor, "kind", None)
    if kind == "state":
        _append_unique(states, descriptor)
    elif kind == "field":
        _append_unique(fields, descriptor)
    elif kind == "rate":
        _append_unique(states, getattr(descriptor, "base_space", None))
    for attr in ("domain", "range"):
        nested = getattr(descriptor, attr, None)
        if nested is not None:
            _descriptor_spaces(nested, states, fields)
    items = getattr(descriptor, "items", None)
    if callable(items):
        for _name, nested in items():
            _descriptor_spaces(nested, states, fields)


def _defaults(registry: Any) -> tuple[Any, Any]:
    states: list[Any] = []
    fields: list[Any] = []
    for name in registry.names():
        signature = registry.get(name).signature
        for descriptor in (*signature.inputs, signature.output):
            _descriptor_spaces(descriptor, states, fields)
    state = states[0] if len(states) == 1 else None
    field = None
    default_of_kind = getattr(registry, "default_of_kind", None)
    if callable(default_of_kind):
        try:
            output = default_of_kind("field_operator").signature.output
            field = output if getattr(output, "kind", None) == "field" else None
        except (KeyError, ValueError):
            pass
    if field is None and len(fields) == 1:
        field = fields[0]
    return state, field


def _walk(program: Any) -> Any:
    seen = set()

    def visit(value: Any) -> Any:
        if value.id in seen:
            return
        seen.add(value.id)
        yield value
        for key in ("cond_block", "body_block", "apply_block", "residual_block"):
            for nested in value.attrs.get(key) or ():
                yield from visit(nested)

    for value in program._values:
        yield from visit(value)
    if program._dt_bound is not None:
        for value in program._dt_bound[0]:
            yield from visit(value)


def resolve_program_spaces(program: Any, model: Any) -> Any:
    """Return ``program`` or a structurally typed, graph-consistent copy.

    Legacy shortcut authoring has no model at node-construction time.  At emit,
    the supplied model's registry is authoritative: a unique StateSpace/FieldSpace
    is propagated onto every matching untyped SSA value.  Space remains in the
    canonical IR and hash; this resolves missing type information rather than
    hiding it.  Ambiguous multi-state registries leave values explicit-only.
    """
    if not hasattr(program, "_values") or not hasattr(program, "_rebuild"):
        return program
    registry = _registry(model)
    if registry is None:
        return program
    state_space, field_space = _defaults(registry)

    def field_for(value: Any) -> Any:
        field = value.attrs.get("field")
        if field is not None:
            from pops.time.references import field_name
            name = field_name(field)
        else:
            name = None
        operator = _operator(registry, name) if name is not None else None
        if operator is not None:
            output = operator.signature.output
            if getattr(output, "kind", None) == "field":
                return output
        return field_space

    def operator_for(value: Any) -> Any:
        name = value.attrs.get("linear_source")
        operator = _operator(registry, name) if isinstance(name, str) else None
        return operator.signature.output if operator is not None else None

    def inferred(value: Any) -> Any:
        if value.space is not None:
            return value.space
        if value.vtype == "state":
            return state_space
        if value.vtype == "rhs" and state_space is not None:
            from pops.model.spaces import Rate
            return Rate(state_space)
        if value.vtype == "fields":
            return field_for(value)
        if value.vtype == "operator":
            return operator_for(value)
        return None

    values = tuple(_walk(program))
    if not any(value.space is None and inferred(value) is not None for value in values):
        return program
    resolved = program._rebuild(lambda _value: True, space_of=inferred)
    resolved.bind_operators(model)
    if state_space is not None:
        resolved._state_spaces = {
            state_ref: state_space if space is None else space
            for state_ref, space in resolved._state_spaces.items()}
        resolved._history_spaces = {
            name: state_space if space is None else space
            for name, space in resolved._history_spaces.items()}
    return resolved


__all__ = ["resolve_program_spaces"]
