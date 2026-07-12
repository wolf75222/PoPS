"""GeneratedModule metadata emission for compiled Program artifacts."""
from __future__ import annotations

import json
from typing import Any

from pops.codegen.program_models import ProgramModelGraph


def _declared_spaces(authority: Any, plural: str, singular: str) -> tuple[Any, ...]:
    """Return every declared space, preserving its module declaration order."""
    accessor = getattr(authority, plural, None)
    if callable(accessor):
        declared = accessor()
        values = declared.values() if hasattr(declared, "values") else declared
        return tuple(values)
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


def emit_module_metadata(program: Any, model: Any = None) -> str:
    """Emit the complete owner-qualified GeneratedModule metadata C++ block.

    Kernel models may expose only one representative state/field space. Whole-Program compilation
    therefore inventories spaces from each canonical source Module in ``ProgramModelGraph`` while
    retaining the emit model's operator registry. Spaces are deduplicated only when both their
    canonical owner and structural identity match.
    """
    del program  # metadata is model-derived and deliberately does not perturb Program identity
    ops, states, fields = [], [], []
    op_owners, state_owners, field_owners = [], [], []
    seen_states, seen_fields = set(), set()
    if type(model) is ProgramModelGraph:
        items = [
            (owner, emit_model, model.source_modules_by_owner.get(owner) or emit_model)
            for owner, emit_model in sorted(
                model.models_by_owner.items(), key=lambda item: str(item[0]))
        ]
    elif model is not None:
        items = [(getattr(model, "owner_path", ""), model, model)]
    else:
        items = []
    for owner, emit_model, declared_module in items:
        canonical_owner = owner.canonical() if hasattr(owner, "canonical") else owner
        owner_name = str(canonical_owner)
        if hasattr(emit_model, "operator_registry"):
            registry = emit_model.operator_registry()
            model_ops = [registry.get(name) for name in registry.names()]
            ops.extend(model_ops)
            op_owners.extend([owner_name] * len(model_ops))
        for space in _declared_spaces(declared_module, "state_spaces", "state_space"):
            identity = (owner_name, _space_identity(space))
            if identity not in seen_states:
                seen_states.add(identity)
                states.append(space.name)
                state_owners.append(owner_name)
        for space in _declared_spaces(declared_module, "field_spaces", "field_space"):
            identity = (owner_name, _space_identity(space))
            if identity not in seen_fields:
                seen_fields.add(identity)
                fields.append(space.name)
                field_owners.append(owner_name)

    def table(accessor: Any, values: Any) -> str:
        cases = "".join('    case %d: return %s;\n' % (i, json.dumps(value))
                        for i, value in enumerate(values))
        return ('extern "C" const char* pops_module_%s(int i) {\n'
                '  switch (i) {\n%s    default: return "";\n  }\n}\n' % (accessor, cases))

    def req_json(op: Any) -> str:
        return json.dumps({**op.requirements, "kind": op.kind})

    parts = [
        "// GeneratedModule metadata (Spec 2 / ADC-442): the typed operator registry exposed by\n"
        "// the .so for introspection + install-time validation. OperatorId = the array index.\n"
        "// NOT called from any hot kernel -- operators are inlined at codegen.\n",
        'extern "C" int pops_module_operator_count() { return %d; }\n' % len(ops),
        'extern "C" int pops_module_state_space_count() { return %d; }\n' % len(states),
        'extern "C" int pops_module_field_space_count() { return %d; }\n' % len(fields),
        table("operator_name", [op.name for op in ops]),
        table("operator_kind", [op.kind for op in ops]),
        table("operator_signature", [repr(op.signature) for op in ops]),
        table("operator_requirements", [req_json(op) for op in ops]),
        table("operator_owner", op_owners),
        table("state_space_name", states),
        table("state_space_owner", state_owners),
        table("field_space_name", fields),
        table("field_space_owner", field_owners),
    ]
    return "".join(parts)


__all__ = ["emit_module_metadata"]
