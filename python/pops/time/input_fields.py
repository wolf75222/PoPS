"""Explicit observation of authenticated runtime inputs at one exact state stage."""
from __future__ import annotations

from pops.model import FieldSpace, StateSpace
from pops.model.provider_pack import ComponentKey, ProviderPack
from pops.time.field_context import FieldContext
from pops.time.operator_resolution import resolve_operator_handle
from pops.time._program.value_validation import require_owned


def runtime_input_pack(pack, field, space):
    """Select the whole declared field surface, refusing computed or missing producers."""
    declaration = field.declaration_ref if field.is_instance else field
    owner = str(declaration.owner_path.canonical())
    rows = []
    for component in space.components:
        key = ComponentKey(owner, "field", space.name, component)
        entry = pack.lookup(key)
        if entry.producer != "runtime_input":
            raise ValueError(
                "input_fields requires runtime_input for every claimed component; "
                "%s/%s is produced by %r" % (key.space, component, entry.producer))
        rows.append((key, pack.contract(key), entry))
    return ProviderPack(rows, capacity=pack.capacity)


def input_fields(program, state, *, for_rate, name=None):
    """Observe one rate's unique runtime-input FieldSpace without solving a field."""
    require_owned(program, state, "input_fields")
    if state.vtype != "state" or state.block is None:
        raise TypeError("input_fields requires a block-qualified State value")
    operator = resolve_operator_handle(
        program, for_rate, where="input_fields", values=(state,))
    inputs = operator.signature.inputs
    states = tuple(item for item in inputs if isinstance(item, StateSpace))
    fields = tuple(item for item in inputs if isinstance(item, FieldSpace))
    if states != (state.space,) or len(fields) != 1 or len(inputs) != 2:
        raise ValueError(
            "input_fields requires exactly this StateSpace and one unique FieldSpace input")
    registry = state.block._instance_registry
    registry.canonical_block(state.block)
    model = registry.spec(state.block.local_id)["model"]
    module = getattr(model, "module", model)
    space = fields[0]
    declared_space = module.field_spaces().get(space.name)
    if declared_space != space:
        raise ValueError("input_fields formal field is not declared by the exact source Module")
    field = state.block[module.field_handle(declared_space)]
    from pops.codegen.component_provider_packs import resolve_component_provider_packs
    runtime_input_pack(resolve_component_provider_packs(module).auxiliary, field, space)
    program._validate_scheduled_reads((state,), consumer="input_fields")
    return program._new(
        "fields", "input_fields", (state,), {"field": field, "for_rate": for_rate},
        name, state.block, space=space,
        field_context=FieldContext(field, ((state.block, state.id),), space.components))
