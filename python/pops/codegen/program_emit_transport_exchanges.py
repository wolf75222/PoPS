"""Accepted quadrature over retained, boundary-qualified native transport faces."""
import json


def emit_transport_exchanges(faces, operation, occurrence, evaluation, weight):
    # Native hyperbolic FaceField contains the physical face integral already. Preserve
    # its actual measure in the ledger; -div(F) reverses the diffusive incidence sign.
    return [
        "pops::sync_host();",
        "for (const auto& accepted_faces : %s) {" % faces,
        "  const auto cells = accepted_faces.cell_box();",
        "  const auto extent = cells.extent();",
        "  const auto face_values = accepted_faces.view();",
        "  for (std::int64_t ordinal = 0; ordinal < cells.numPts(); ++ordinal) {",
        "    auto remainder = ordinal;",
        "    auto cell = cells.lo;",
        "    for (int axis = 0; axis < pops::kNativeDimension; ++axis) {",
        "      cell[axis] += static_cast<int>(remainder % extent[axis]);",
        "      remainder /= extent[axis];",
        "    }",
        "    for (int axis = 0; axis < pops::kNativeDimension; ++axis) {",
        "      pops::Real measure = 1;",
        "      for (int tangent = 0; tangent < pops::kNativeDimension; ++tangent)",
        "        if (tangent != axis) measure *= ctx.geometry().spacing(tangent);",
        "      for (int side = 0; side < 2; ++side) {",
        "        auto face = cell; face[axis] += side;",
        "        std::string quadrature = \"cell\";",
        "        for (int dimension = 0; dimension < pops::kNativeDimension; ++dimension)",
        "          quadrature += \":\" + std::to_string(cell[dimension]);",
        "        quadrature += \"/axis:\" + std::to_string(axis) + \"/side:\" + std::to_string(side);",
        "        ctx.stage_exchange(pops::runtime::program::ExchangeRecord{%s, %s, %s, quadrature," % (
            json.dumps(operation), json.dumps(occurrence), json.dumps(evaluation)),
        "            side == 0 ? 1 : -1, measure, face_values.axes[axis](face, 0)/measure, %s, 1});" % weight,
        "      }",
        "    }",
        "  }",
        "}",
    ]
