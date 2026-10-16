"""Cell-centred observations of explicitly consumed physical field solutions."""
from __future__ import annotations

from typing import Any
import json

from pops.fields._observation_contract import validate_field_gradient


def emit_field_gradient(value: Any, var: Any, lines: Any, prelude: Any, *, target: str) -> None:
    if target not in ("system", "amr_system") or prelude is None:
        raise NotImplementedError("field gradient requires an authenticated native field route")
    dimension, solve = validate_field_gradient(value)
    source = value.inputs[0]
    if var.get(("field_observation", source.id)) != source.attrs["field_problem_identity"]:
        raise ValueError("field gradient source has no emitted consumed observation authority")
    token, boundary = "field_gradient_%d" % value.id, "field_gradient_boundary_%d" % value.id
    from .program_field_provider_catalog import field_provider_contract
    from .program_emit_field_problem import guard_candidate_allocations

    contract = json.dumps(field_provider_contract(value))
    capacity = source.attrs.get("observation_byte_capacity", 64 * 1024 * 1024)
    if target == "amr_system":
        snapshot = var.get(("field_observation_snapshot", source.id))
        if snapshot is None:
            raise ValueError("field gradient source has no immutable solved snapshot")
        lines.append("static_assert(pops::kNativeDimension == %d);" % dimension)
        lines.append("auto %s = ctx.allocate_hierarchy_continuation_field_observation(%s, UINT64_C(%d));"
                     % (token, contract, capacity))
        lines.append("ctx.observe_hierarchy_field_gradient(%s, *%s);" % (snapshot, token))
        lines.append("ctx.seal_hierarchy_continuation_field_observation(%s, %d, %s, %s);"
                     % (contract, value.id, token, snapshot))
        var[value.id] = "(*%s)" % token
        var[("field_observation", value.id)] = source.attrs["field_problem_identity"]
        var[("field_pointer", value.id)] = token
        return
    prelude.append("static_assert(pops::kNativeDimension == %d, "
                   '"field gradient dimension differs from the resolved geometry");' % dimension)
    input_copy = token + "_input"
    lines.extend(guard_candidate_allocations([
        "auto %s = ctx.allocate_continuation_field_observation(%s, UINT64_C(%d));"
        % (token, contract, capacity),
        "auto %s = ctx.own_structural_field(%s);" % (input_copy, var[source.id]),
    ]))
    # Boundary preparation belongs to this invocation. Filling its private input
    # copy must never mutate the already sealed observation's retained ghosts.
    lines.append("auto %s = ctx.prepare_mesh_boundary_session(*%s, ctx.prepared_execution_lane());"
                 % (boundary, input_copy))
    # The source's exact physical problem supplies the boundary law for this copy.
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
        "  ctx.gradient(*%s, *%s, *%s);" % (token, input_copy, boundary),
        "}",
    ])
    lines.append("ctx.seal_continuation_field_observation(%s, %d, %s);" % (contract, value.id, token))
    var[value.id] = "(*%s)" % token
    var[("field_observation", value.id)] = source.attrs["field_problem_identity"]
    var[("field_pointer", value.id)] = token
