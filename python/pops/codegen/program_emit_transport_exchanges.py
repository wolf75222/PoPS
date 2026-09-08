"""Accepted quadrature over retained, boundary-qualified native transport faces."""

import json


def emit_transport_exchanges(
    faces, operation, occurrence, evaluation, weight, *, program_block=0, active_field=None
):
    # Native hyperbolic FaceField contains the physical face integral already. Preserve
    # its actual measure in the ledger; -div(F) reverses the diffusive incidence sign.
    active = active_field or "ctx.state(%d)" % program_block
    active_name = faces + "_active"
    local_name = faces + "_local"
    return [
        "{",
        "pops::sync_host();",
        "const auto* %s = ctx.pointwise_active_mask(%d, %s);"
        % (active_name, program_block, active),
        "long accepted_mask_layout_error = 0;",
        "if (%s != nullptr) {" % active_name,
        "  if (%s->local_size() != %s.size()) accepted_mask_layout_error = 1;"
        % (active_name, faces),
        "  else for (std::size_t local = 0; local < %s.size(); ++local)" % faces,
        "    if (%s->box(local) != %s[local].cell_box()) accepted_mask_layout_error = 1;"
        % (active_name, faces),
        "}",
        "if (pops::all_reduce_max(accepted_mask_layout_error, ctx.prepared_execution_lane()) != 0)",
        '  throw std::invalid_argument("accepted transport face mask differs from local patches collectively");',
        "for (std::size_t %s = 0; %s < %s.size(); ++%s) {"
        % (local_name, local_name, faces, local_name),
        "  const auto& accepted_faces = %s[%s];" % (faces, local_name),
        "  const auto cells = accepted_faces.cell_box();",
        "  const auto extent = cells.extent();",
        "  const auto face_values = accepted_faces.view();",
        "  const auto active_values = %s == nullptr" % active_name,
        "      ? pops::FieldView<const pops::Real, pops::kNativeDimension>{}",
        "      : std::as_const(*%s).fab(%s).view();" % (active_name, local_name),
        "  for (std::int64_t ordinal = 0; ordinal < cells.numPts(); ++ordinal) {",
        "    auto remainder = ordinal;",
        "    auto cell = cells.lo;",
        "    for (int axis = 0; axis < pops::kNativeDimension; ++axis) {",
        "      cell[axis] += static_cast<int>(remainder % extent[axis]);",
        "      remainder /= extent[axis];",
        "    }",
        "    if (%s != nullptr && active_values(cell,0) < 0.5) continue;" % active_name,
        "    for (int axis = 0; axis < pops::kNativeDimension; ++axis) {",
        "      pops::Real measure = 1;",
        "      for (int tangent = 0; tangent < pops::kNativeDimension; ++tangent)",
        "        if (tangent != axis) measure *= ctx.geometry().spacing(tangent);",
        "      for (int side = 0; side < 2; ++side) {",
        "        auto face = cell; face[axis] += side;",
        '        std::string quadrature = "cell";',
        "        for (int dimension = 0; dimension < pops::kNativeDimension; ++dimension)",
        '          quadrature += ":" + std::to_string(cell[dimension]);',
        '        quadrature += "/axis:" + std::to_string(axis) + "/side:" + std::to_string(side);',
        "        ctx.stage_exchange(pops::runtime::program::ExchangeRecord{%s, %s, %s, quadrature,"
        % (json.dumps(operation), json.dumps(occurrence), json.dumps(evaluation)),
        "            side == 0 ? 1 : -1, measure, face_values.axes[axis](face, 0)/measure, %s, 1});"
        % weight,
        "      }",
        "    }",
        "  }",
        "}",
        "}",
    ]
