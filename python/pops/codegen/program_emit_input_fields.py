"""Publish only the exact runtime-input prerequisites of an explicit observation."""
from pops.codegen.program_emit_kernels import (
    _model_impl, _prepare_provider_values, program_provider_consumer_qid,
)
from pops.time.input_fields import runtime_input_pack


def emit_input_fields(value, var, lines, model, provider_plans, block_index, target):
    if target != "system":
        raise NotImplementedError("input_fields currently requires the Uniform runtime")
    if provider_plans is None:
        raise ValueError("input_fields requires an authenticated Program provider plan")
    pack = getattr(_model_impl(model), "_auxiliary_provider_pack", None)
    if pack is None:
        raise ValueError("input_fields emitter is missing its exact source ProviderPack")
    selected = runtime_input_pack(pack, value.attrs["field"], value.space)
    binding = provider_plans.bind_pack(
        selected, program_provider_consumer_qid(model, value.id, value.block))
    (state,) = value.inputs
    lines.extend(_prepare_provider_values(binding, block_index, var[state.id]))
    # A fields SSA value is an availability/provenance token. The provider pack owns storage.
    var[value.id] = var[state.id]
