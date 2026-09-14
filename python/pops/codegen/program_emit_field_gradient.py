"""Cell-centred observations of explicitly consumed physical field solutions."""
from __future__ import annotations

from typing import Any

from pops.fields._observation_contract import validate_field_gradient


def emit_field_gradient(value: Any, var: Any, lines: Any, prelude: Any, *, target: str) -> None:
    if target not in ("system", "amr_system") or prelude is None:
        raise NotImplementedError("field gradient requires an authenticated native field route")
    dimension, solve = validate_field_gradient(value)
    source = value.inputs[0]
    if var.get(("field_observation", source.id)) != source.attrs["field_problem_identity"]:
        raise ValueError("field gradient source has no emitted consumed observation authority")
    token, boundary = "field_gradient_%d" % value.id, "field_gradient_boundary_%d" % value.id
    if target == "amr_system":
        selected = source.attrs["component"]
        lines.append("static_assert(pops::kNativeDimension == %d);" % dimension)
        lines.append("auto& %s = ctx.hierarchy_field_scratch(%d, %d, 0, %d, 1);" % (
            token, solve.id, value.id, dimension))
        lines.append("ctx.observe_hierarchy_field_gradient(%d, %s, %d);" % (solve.id, token, selected))
        var[value.id] = token
        var[("field_observation", value.id)] = source.attrs["field_problem_identity"]
        var[("field_pointer", value.id)] = "(&%s)" % token
        return
    prelude.append("static_assert(pops::kNativeDimension == %d, "
                   '"field gradient dimension differs from the resolved geometry");' % dimension)
    prelude.append("auto %s = std::make_shared<pops::MultiFab<pops::kNativeDimension>>("
                   "ctx.alloc_scalar_field(%d, 1));" % (token, dimension))
    prelude.append("auto %s = ctx.prepare_mesh_boundary_session(%s, ctx.prepared_execution_lane());"
                   % (boundary, var[source.id]))
    # The source was copied from this exact solved physical problem. Its own scalar
    # halo can be filled without modifying the packed solution or any species state.
    physical = solve.inputs[1].attrs["physical_boundary"]
    lines.extend([
        "{",
        "  long gradient_boundary_invalid = 0;",
        "  try {",
        "    std::array<pops::elliptic::nd::PhysicalFieldBoundary, 2 * pops::kNativeDimension> physical{};",
        "    physical.fill(pops::elliptic::nd::PhysicalFieldBoundary::%s);" % physical,
        "    pops::elliptic::nd::require_field_boundary(*%s, physical);" % boundary,
        "  } catch (...) { gradient_boundary_invalid = 1; }",
        "  if (pops::all_reduce_max(gradient_boundary_invalid, ctx.prepared_execution_lane()) != 0)",
        '    throw std::invalid_argument("field gradient physical boundary refused collectively");',
        "  ctx.gradient(*%s, %s, *%s);" % (token, var[source.id], boundary),
        "}",
    ])
    var[value.id] = "(*%s)" % token
    var[("field_observation", value.id)] = source.attrs["field_problem_identity"]
    var[("field_pointer", value.id)] = token
