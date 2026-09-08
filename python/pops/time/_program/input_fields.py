"""Program-owned observation of exact runtime input fields."""
from pops.model import FieldSpace, StateSpace
from pops.model.provider_pack import build_provider_pack, compact_auxiliary_provider_pack
from pops.time.field_context import FieldContext
from pops.time.input_fields import runtime_input_pack
from pops.time.operator_resolution import resolve_operator_handle
from .value_validation import require_owned


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
    runtime_input_pack(compact_auxiliary_provider_pack(build_provider_pack(module)), field, space)
    program._validate_scheduled_reads((state,), consumer="input_fields")
    return program._new(
        "fields", "input_fields", (state,), {"field": field, "for_rate": for_rate},
        name, state.block, space=space,
        field_context=FieldContext(field, ((state.block, state.id),), space.components))
