"""Phase-specific BindSchema value materialization."""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .resolved_bindings import ResolvedBindings


def resolve_compile(schema: Any) -> Mapping[Any, Any]:
    resolved = {slot.handle: slot.default_value() for slot in schema.const_slots}
    sources = {slot.handle: "const" for slot in schema.const_slots}
    _materialize(schema, resolved, sources, phases=frozenset({"compile"}))
    return MappingProxyType(dict(resolved))


def resolve_bind(
    schema: Any, values: Any, *, compile_values: Mapping[Any, Any],
) -> ResolvedBindings:
    values = {} if values is None else values
    if not isinstance(values, Mapping):
        raise TypeError("BindSchema.resolve_bind values must be a ParamHandle mapping")
    if not isinstance(compile_values, Mapping):
        raise TypeError("BindSchema.resolve_bind requires resolve-time compile_values")
    expected = {slot.handle for slot in schema.const_slots} | {
        slot.handle for slot in schema.derived_slots
        if slot.declaration["phase"] == "compile"}
    if set(compile_values) != expected:
        raise ValueError("compile_values do not exactly match the BindSchema compile phase")
    supplied = {}
    for handle, value in values.items():
        canonical = schema._canonical_handle(handle)
        if canonical in supplied:
            raise ValueError("multiple bind entries resolve to the same ParamHandle %s"
                             % canonical.qualified_id)
        slot = schema._by_handle[canonical]
        if slot.kind != "runtime":
            raise TypeError("bind value supplied for %s parameter %s; only RuntimeParam slots "
                            "are settable" % (slot.kind, slot.qid))
        supplied[canonical] = slot.validate_value(value)
    resolved = dict(compile_values)
    sources = {slot.handle: ("const" if slot.kind == "const" else "derived")
               for slot in schema.slots if slot.handle in compile_values}
    missing = []
    for slot in schema.runtime_slots:
        if slot.handle in supplied:
            resolved[slot.handle], sources[slot.handle] = supplied[slot.handle], "override"
        elif slot.has_default:
            resolved[slot.handle], sources[slot.handle] = slot.default_value(), "default"
        else:
            missing.append(slot.qid)
    if missing:
        raise ValueError("missing required RuntimeParam bind value(s): %s" % ", ".join(missing))
    _materialize(schema, resolved, sources, phases=frozenset({"bind"}))
    return ResolvedBindings(schema, resolved, sources)


def _materialize(schema: Any, resolved: dict[Any, Any], sources: dict[Any, str], *,
                 phases: frozenset[str]) -> None:
    evaluated = dict(resolved)

    def dependency_slot(owner: Any, dependency: Mapping[str, Any]) -> Any:
        matches = [candidate for candidate in schema.slots
                   if schema._scope_key(candidate.handle) == schema._scope_key(owner.handle)
                   and candidate.handle.local_id == dependency["name"]
                   and candidate.kind == dependency["param_kind"]]
        if len(matches) != 1:
            raise RuntimeError("BindSchema dependency index is inconsistent")
        return matches[0]

    def materialize(slot: Any) -> Any:
        if slot.handle in evaluated:
            return evaluated[slot.handle]
        env = {}
        for dependency in slot.declaration["depends_on"]:
            target = dependency_slot(slot, dependency)
            value = materialize(target)
            env[target.handle.local_id] = value
            env[target.qid] = value
            if target.handle.declaration_ref is not None:
                env[target.handle.declaration_ref.qualified_id] = value
        evaluated[slot.handle] = slot.evaluate(env)
        return evaluated[slot.handle]

    for slot in schema.derived_slots:
        if slot.declaration["phase"] in phases:
            resolved[slot.handle] = materialize(slot)
            sources[slot.handle] = "derived"


__all__ = ["resolve_bind", "resolve_compile"]
