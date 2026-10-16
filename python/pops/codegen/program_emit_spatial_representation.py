"""Attach the executed coordinate transformation to its exact solved Field."""
from __future__ import annotations

from typing import Any

from pops.time.value_signatures import value_signature

from .program_emit_kernels import program_provider_consumer_qid
from .program_emit_transform_obligation import emit_closed_transform
from .program_models import model_for_node


def emit_spatial_representation(program: Any, value: Any, variables: Any, model: Any,
                                lines: list[str], block_indices: Any, target: str) -> None:
    # The already-validated residual is the authority for the transformation.
    # An identity coordinate has no pending transformation to retain.
    if value.attrs["physical_mapping"]["accumulation"] is None:
        return
    from pops.time.references import handle_data
    from pops.time._program.serialization import _json_ready

    mapping = _json_ready(value.attrs["physical_mapping"]["accumulation"])
    matches = [node for node in value.attrs["residual_block"]
               if node.op == "local_transform" and node.inputs == (value.attrs["iterate"],)
               and _json_ready(handle_data(node.attrs["operator_handle"])) == mapping]
    if len(matches) != 1:
        raise ValueError("spatial representation requires its exact residual transformation")
    transform = matches[0]
    node_model = model_for_node(model, transform)
    owner = block_indices[value.block]
    stem = "spatial_representation_%d" % value.id
    state, status, mask = stem+"_mapped", stem+"_status", stem+"_mask"
    field = "pops::MultiFab<pops::kNativeDimension>"
    lines += [
        "%s* %s = nullptr;" % (field, state),
        "%s* %s = nullptr;" % (field, status),
        "pops::runtime::program::field_read_detail::phase(ctx.prepared_execution_lane(), [&] {",
        "  %s = &ctx.scratch_state(%d,2,ctx.state(%d));" % (state, value.id, owner),
        "  %s = &ctx.scalar_scratch(%d,1,ctx.state(%d),1,0);" % (status, value.id, owner),
        '}, "spatial representation workspace preparation rejected");',
        "const auto* %s = ctx.pointwise_active_mask(%d, *%s);" % (mask, owner, state),
    ]
    closed, owned = emit_closed_transform(
        model=node_model, name=transform.attrs["transform"], source=variables[value.id],
        output="(*%s)" % state, status="(*%s)" % status, mask=mask, owner=owner,
        signature=value_signature(value), block_indices=block_indices,
        provider_plans=variables.get(("program_provider_plans",)),
        consumer_qid=program_provider_consumer_qid(node_model, transform.id, transform.block),
        stem=stem, capacity=program.representation_workspace_byte_capacity,
        target=target, issue=True, production_id=value.id)
    lines += closed
    lines.append("auto& %s_value = *%s;" % (stem, owned))
    variables[value.id] = stem+"_value"
