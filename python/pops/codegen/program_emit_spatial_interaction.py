"""Direct native spatial interaction; no Python materialization or implicit commit."""
import json

from pops.time._program.spatial_interaction import interaction_contract
from pops.time._program.serialization import _json_ready


def emit_spatial_interaction(value, variables, lines, *, block_indices, target):
    source, dimension, kernel, budget, components = interaction_contract(value)
    if target not in ("system", "amr_system"):
        raise NotImplementedError("spatial interaction requires a native Uniform/AMR field provider")
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
    prefix = "auto&" if target == "amr_system" else "auto"
    argument = "%d, " % key if target == "amr_system" else ""
    lines.append("%s %s = ctx.%s(%s%d, %s, spatial_components_%d, %dULL, %s, %s, %s," % (
        prefix, name, method, argument, block_indices[value.block], variables[source.id], key, budget,
        json.dumps(identity), json.dumps(source.clock.qualified_id), "true" if value.attrs["source_scope"] == "accepted" else "false"))
    lines.append("  [=] POPS_HD(const pops::RealVector<pops::kNativeDimension>& x, const pops::RealVector<pops::kNativeDimension>& y) -> pops::Real { return %s; });" % kernel)

