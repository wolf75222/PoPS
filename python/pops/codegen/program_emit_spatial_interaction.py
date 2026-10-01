"""Direct native spatial interaction; no Python materialization or implicit commit."""
import json

from pops.time._program.spatial_interaction import interaction_contract
from pops.time._program.serialization import _json_ready



def closed_interaction_identity(value):
    return json.dumps({"attrs":_json_ready(value.attrs), "space":_json_ready(value.space),
                       "point":_json_ready(value.point), "source":value.inputs[0].id},
                      sort_keys=True, separators=(",", ":"))

def emit_spatial_interaction(value, variables, lines, *, block_indices, target):
    source, dimension, kernel, budget, components = interaction_contract(value)
    if target not in ("system", "amr_system"):
        raise NotImplementedError("spatial interaction requires a native Uniform/AMR field provider")
    if value.attrs["contract"] == "pops.spatial-interaction@3":
        if target != "amr_system":
            raise NotImplementedError("completed original source snapshot @1 is realized on the composite AMR provider; Uniform needs its own Accept snapshot port")
        from pops.fields._observation_contract import validate_field_observation
        _width, component, solve = validate_field_observation(source)
        owner = value.attrs["closed_field_source"]["owner_block"]
        if owner not in block_indices:
            raise ValueError("closed interaction storage TimeState has no exact emitted block index")
        phase = variables.get(("original_field_hierarchy_phase",))
        name = "closed_interaction_%d" % value.id
        variables[value.id] = name
        if phase == "gather":
            return
        if phase == "solve":
            identity = closed_interaction_identity(value)
            lines += ["static_assert(pops::kNativeDimension == %d, \"closed interaction dimension differs from native geometry\");" % dimension,
                      "const std::array<int, 1> closed_components_%d{%d};" % (value.id, component),
                      "ctx.prepare_closed_original_interaction(%d, %d, %d, %d, closed_components_%d, %dULL, %s," %
                      (value.id, solve.id, source.id, block_indices[owner], value.id, budget, json.dumps(identity)),
                      "  [=] POPS_HD(const pops::RealVector<pops::kNativeDimension>& x, const pops::RealVector<pops::kNativeDimension>& y) -> pops::Real { return %s; });" % kernel]
        elif phase in (None, "observe", "publish"):
            lines.append("const auto& %s = ctx.closed_original_interaction(%d);" % (name, value.id))
        else:
            raise NotImplementedError("unknown closed original source scheduling phase")
        return
    if value.block not in block_indices:
        raise ValueError("spatial interaction has no exact Program block route")
    from pops.time._evaluation_point import evaluation_stage_fraction
    stage = evaluation_stage_fraction(value)
    if source.op != "history" and not 0 <= stage <= 1:
        raise ValueError("direct interaction source is outside its issued temporal window")
    key = value.id
    name = "spatial_interaction_%d" % key
    variables[key] = name
    identity = json.dumps({"attrs": _json_ready(value.attrs), "point": _json_ready(value.point),
                           "space": _json_ready(value.space), "source": source.id},
                          sort_keys=True, separators=(",", ":"))
    if source.op != "history":
        lines.append("ctx.set_stage_time(%d, %d);" % (stage.numerator, stage.denominator))
    lines.append("static_assert(pops::kNativeDimension == %d, \"interaction kernel dimension differs from native geometry\");" % dimension)
    lines.append("const std::array<int, %d> spatial_components_%d{%s};" % (
        len(components), key, ", ".join(map(str, components))))
    method = "spatial_interaction_owned" if target == "amr_system" else "spatial_interaction"
    history_arguments = ""
    if value.attrs["contract"] == "pops.spatial-interaction@2":
        from pops.codegen.program_history_identity import history_space_identity
        seed = value.attrs["history_source"]
        method = "spatial_interaction_history_owned" if target == "amr_system" else "spatial_interaction_history"
        interpolation = json.dumps(_json_ready(source.attrs["history_contract"])["interpolation"],
                                   sort_keys=True, separators=(",", ":"))
        history_arguments = " %s, %d, %s, %s, %s," % (
            json.dumps(seed["history"]), seed["lag"], json.dumps(seed["state"].qualified_id),
            json.dumps(history_space_identity(value.prog, seed["history"])), json.dumps(interpolation))
    prefix = "auto&" if target == "amr_system" else "auto"
    argument = "%d, " % key if target == "amr_system" else ""
    lines.append("%s %s = ctx.%s(%s%d, %s, spatial_components_%d, %dULL, %s, %s, %s,%s" % (
        prefix, name, method, argument, block_indices[value.block], variables[source.id], key, budget,
        json.dumps(identity), json.dumps(source.clock.qualified_id), "true" if value.attrs["source_scope"] == "accepted" else "false", history_arguments))
    lines.append("  [=] POPS_HD(const pops::RealVector<pops::kNativeDimension>& x, const pops::RealVector<pops::kNativeDimension>& y) -> pops::Real { return %s; });" % kernel)
