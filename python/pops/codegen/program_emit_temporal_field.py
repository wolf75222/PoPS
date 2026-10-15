"""Composite field assembly at a local point from authored temporal source maps."""
import json
from pops.time._program.temporal_field import validate_temporal_field, is_temporal_field_solve
from pops.time._program.serialization import _json_ready
from .program_emit_history_transfer import emit_history_transfer_callback


def temporal_field_declarations(program, block_indices, *, target):
    from .program_lowerability import all_ops
    from .program_history_identity import history_space_identity
    lines = []
    for solve in all_ops(program):
        if not is_temporal_field_solve(solve):
            continue
        if target != "amr_system":
            raise ValueError("TemporalEvaluation requires the composite AMR execution provider")
        states = solve.inputs[1].inputs
        data = validate_temporal_field(program, states, solve.attrs["temporal_evaluation"])
        for i, (state, row) in enumerate(zip(states, data["inputs"], strict=True)):
            name = row["history"]
            manifest = next(item for item in program.temporal_manifest()["histories"] if item["name"] == name)
            identity = json.dumps(_json_ready(row), sort_keys=True, separators=(",", ":"))
            interpolation = json.dumps(manifest["interpolation"], sort_keys=True, separators=(",", ":"))
            lines.append("ctx.declare_temporal_field_source(%d, %d, %d, %s, %d, %s, %s, %s, %s, %s," % (
                solve.id, i, block_indices[state.block], json.dumps(name), program._histories[name]+1,
                json.dumps(state.state_ref.qualified_id), json.dumps(history_space_identity(program, name)),
                json.dumps(state.clock.qualified_id), json.dumps(interpolation), json.dumps(identity)))
            lines.extend(emit_history_transfer_callback(row["transfer"]))
            lines.append(");")
    return lines


def emit_temporal_field_assembly(value, var, lines, prelude, *, target):
    if (target != "amr_system" or value.op == "field_component"
            or value.attrs.get("temporal_evaluation") is None
            or var.get(("temporal_field_inner",))):
        return False
    from .program_emit_field_problem import hierarchy_field_solve, emit_field_problem_value
    from pops.time._evaluation_point import evaluation_stage_fraction
    solve = hierarchy_field_solve(value)
    data = validate_temporal_field(value.prog, value.inputs, value.attrs["temporal_evaluation"])
    if data != solve.attrs.get("temporal_evaluation"):
        raise ValueError("field temporal assembly and consuming solve disagree")
    stage = evaluation_stage_fraction(value)
    token = "temporal_field_%d" % value.id
    lines.extend(["ctx.set_stage_time(%d, %d);" % (stage.numerator, stage.denominator),
        "const int %s_level = ctx.level();" % token,
        "const double %s_time = ctx.temporal_field_evaluation_time(%d, 0);" % (token, solve.id),
        "ctx.for_each_program_resource_level([&](int) {"])
    nested = dict(var)
    nested[("temporal_field_inner",)] = True
    for index, state in enumerate(value.inputs):
        name = "%s_input_%d" % (token, index)
        lines.append("const auto& %s = ctx.temporal_field_input(%d, %d, %s, %s_level, %s_time);" % (
            name, solve.id, index, var[state.id], token, token))
        nested[state.id] = name
    emit_field_problem_value(value, nested, lines, prelude, target=target)
    lines.append("});")
    # Expose the real provider storage at the caller's restored active level.
    previous = var.get(("hierarchy_field_phase",))
    var[("hierarchy_field_phase",)] = "solve"
    var[("temporal_field_inner",)] = True
    try:
        emit_field_problem_value(value, var, lines, prelude, target=target)
    finally:
        var.pop(("temporal_field_inner",), None)
        if previous is None:
            var.pop(("hierarchy_field_phase",), None)
        else:
            var[("hierarchy_field_phase",)] = previous
    return True
