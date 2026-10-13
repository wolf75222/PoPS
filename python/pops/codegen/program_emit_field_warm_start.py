"""Seed a qualified native field iterate, without publishing readable field outputs."""
from pops.identity import make_identity
from pops.codegen.cpp_strings import cpp_string_expression
from pops.codegen.program_emit_field_routes import resolved_field_route
from pops.codegen.program_field_observations import validate_resolved_warm_start


def emit_field_warm_start(value, var, lines, *, target, model_graph, field_plans, block_index):
    if target not in ("system", "amr_system"):
        raise NotImplementedError("field warm start requires its qualified Uniform native route")
    field = validate_resolved_warm_start(value, model_graph=model_graph, field_plans=field_plans)
    slot, plan = resolved_field_route(field, field_plans)
    scalar, _fields, _template, _state = value.inputs
    from pops.time._evaluation_point import evaluation_stage_fraction
    stage = evaluation_stage_fraction(value)
    identity = make_identity("program-field-warm-start", {
        "field_plan": plan.identity.to_data(), "effect_id": value.id,
        "source_ids": tuple(source.id for source in value.inputs),
        "physical_space": scalar.space.to_data(),
    }).token
    # The native slot and identity are authenticated artifact data, never storage offsets.
    lines += ["{", "auto warm_start_stage_%d = ctx.stage_evaluation_scope(%d, %d);" % (
        value.id, stage.numerator, stage.denominator),
        "ctx.%s(%s, %d, %s, %d, %s);" % (
            "stage_field_warm_start" if target == "amr_system" else "publish_field_warm_start",
            cpp_string_expression(slot), block_index, var[scalar.id], value.id,
            cpp_string_expression(identity)), "}"]


__all__ = ["emit_field_warm_start"]
