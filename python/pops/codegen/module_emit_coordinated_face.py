"""Coordinated face v1 using the common expression compiler and native tuple."""
from .cpp_writer import _cse_emit


def emit_coordinated_face_members(model):
    kernel = model._path_conservative["kernel"]
    bindings = {symbol.name: "%s[%d]" % (side, k)
                for side in ("left", "right")
                for k, symbol in enumerate(kernel[side + "_symbols"])}
    lines = [
        "  static constexpr int coordinated_face_contract_version = 1;",
        "  POPS_HD bool path_admissible(const State& state) const {",
        "    for (int k = 0; k < n_vars; ++k) if (!std::isfinite(state[k])) return false;",
        "    return true;", "  }",
        "  template <int Axis>",
        "  POPS_HD pops::PathInterfaceResult<n_vars> coordinated_face(",
        "      const State& left, const State& right) const {",
        "    static_assert(Axis >= 0 && Axis < dimension);",
        "    pops::PathInterfaceResult<n_vars> result;",
        "    if (!path_admissible(left) || !path_admissible(right)) {",
        "      result.status = pops::PathStatus::NonFiniteInput;",
        "      result.input_side = path_admissible(left) ? 1 : 0; return result;", "    }",
        "    if (admissibility(left) != pops::nd::StateConversionStatus::Success ||",
        "        admissibility(right) != pops::nd::StateConversionStatus::Success) {",
        "      result.status = pops::PathStatus::DomainFailure; return result;", "    }",
    ]
    width = model.n_vars
    for axis, row in enumerate(kernel["balances"]):
        lines.append("    %s constexpr (Axis == %d) {" % ("if" if axis == 0 else "else if", axis))
        body, values, names = _cse_emit((*row.flux, *row.left, *row.right, row.stability),
            "pops::Real", "      ", materialize_all=True, return_names=True,
            scalar_bindings=bindings)
        lines += body
        check = " || ".join("!std::isfinite(%s)" % value for value in (*names, *values))
        lines += ["      if (%s || %s < pops::Real(0)) {" % (check, values[-1]),
                  "        result.status = pops::PathStatus::NonFiniteResult; return result;", "      }"]
        for index, name in enumerate(("conservative_flux", "left_ncp", "right_ncp")):
            lines += ["      result.%s.values[%d] = %s;" % (name, k, values[index*width+k])
                      for k in range(width)]
        lines += ["      result.speed_bound = %s;" % values[-1], "    }"]
    lines += ["    result.status = pops::PathStatus::Success; return result;", "  }"]
    return lines
