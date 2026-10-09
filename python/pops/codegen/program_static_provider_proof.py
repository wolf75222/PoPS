"""accepted-static-provider-read@1: exact external storage observations for SSP.

The State input authenticates observation stage/storage ownership. It is not a
mathematical producer of an authenticated runtime-input field. Resolved producer
reprojection and all executed publications remain separate required authorities.
"""
from pops.model import FieldSpace, Handle, OperatorHandle, StateSpace
from pops.model.provider_pack import ComponentKey
from pops.time.field_context import FieldContext
from pops.time.input_fields import runtime_input_pack

CONTRACT = "pops.accepted-static-provider-read@1"


def prove_static_provider_read(program, observation, rate, reads, authority, issued):
    from .program_models import ProgramModelGraph
    from pops.fields._program_publication import validate_field_publication
    from .program_field_publication import publication_target_space

    if type(authority) is not ProgramModelGraph:
        raise ValueError("static provider proof requires resolved ProgramModelGraph source authority")
    if observation.op != "input_fields" or observation.vtype != "fields" \
            or len(observation.inputs) != 1 or set(observation.attrs) != {"field", "for_rate"}:
        raise ValueError("static provider observation lacks its exact typed token contract")
    state, = observation.inputs
    if issued.get(observation.id) is not observation or issued.get(state.id) is not state \
            or state is not rate.inputs[0]:
        raise ValueError("static provider observation has detached or different State stage authority")
    if state.vtype != "state" or observation.block != state.block or rate.block != state.block \
            or observation.point != state.point or observation.clock != state.clock \
            or observation.region != state.region or observation.state_ref != state.state_ref:
        raise ValueError("static provider observation changes exact owner/clock/point/storage authority")
    if observation.attrs["for_rate"] != rate.attrs["operator_handle"]:
        raise ValueError("static provider observation belongs to a different consuming rate")
    owner = authority.owner_for_block(state.block)
    module = authority.source_module_for_owner(owner)
    handle = observation.attrs["for_rate"]
    if not isinstance(handle, OperatorHandle):
        raise TypeError("static provider observation requires an exact typed rate handle")
    declaration = handle.declaration_ref if handle.is_instance else handle
    if not isinstance(declaration, OperatorHandle) or declaration.owner_path.canonical() != owner \
            or handle.is_instance and handle.block_ref != state.block:
        raise ValueError("static provider rate has a foreign source declaration/instance owner")
    registry = module.operator_registry()
    if registry.owner_path.canonical() != owner:
        raise ValueError("static provider rate registry has a different source owner")
    name = registry.target_for_handle(declaration.name)
    operator = registry.get(name)
    if handle.registered_operator_name != name or declaration.registered_operator_name != name \
            or handle.kind != operator.kind or handle.signature != operator.signature:
        raise ValueError("static provider rate has a changed source target/kind/signature")
    fields = tuple(space for space in operator.signature.inputs if isinstance(space, FieldSpace))
    states = tuple(space for space in operator.signature.inputs if isinstance(space, StateSpace))
    if states != (state.space,) or fields != (observation.space,) or len(operator.signature.inputs) != 2:
        raise ValueError("static provider observation changes its exact formal State/FieldSpace surface")
    owner = authority.owner_for_block(state.block)
    module = authority.source_module_for_owner(owner)
    field = observation.attrs["field"]
    if not isinstance(field, Handle) or field.kind != "field" or field.block_ref != state.block \
            or field.declaration_ref is None or field.declaration_ref.owner_path.canonical() != owner:
        raise ValueError("static provider observation has a foreign field declaration/instance owner")
    module.declaration_index().authenticate(field.declaration_ref._resolved())
    if module.field_spaces().get(field.declaration_ref.local_id) != observation.space:
        raise ValueError("static provider observation has undeclared component/space authority")
    context = observation.field_context
    if type(context) is not FieldContext or context.field != field \
            or context.stage_sources != ((state.block, state.id),) \
            or context.outputs != tuple(observation.space.components) or rate.field_context != context:
        raise ValueError("static provider observation changes exact FieldContext/read coverage")
    selected = runtime_input_pack(authority.resolved_provider_pack_for_block(state.block),
                                  field, observation.space)
    if not reads <= {key.component for key in selected}:
        raise ValueError("static provider observation does not cover the constitutive read union")
    keys = set(selected)
    for value in issued.values():
        if value.op != "field_publication":
            continue
        bindings = validate_field_publication(value,
            target_space=lambda target: publication_target_space(authority, target))
        for binding in bindings:
            target = binding["target"]
            key = ComponentKey(str(target.declaration_ref.owner_path.canonical()), "field",
                               target.declaration_ref.local_id, binding["component"])
            if target.block_ref == state.block and key in keys:
                raise ValueError("static provider observation reads an executed replacing field publication")
    return True
