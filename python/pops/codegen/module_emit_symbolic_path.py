"""Emit an authored path body into the existing finite-volume model brick."""
from __future__ import annotations

from .cpp_writer import _cpp_identifier, _cse_emit


def _checked_expressions(expressions, indent):
    lines, values, names = _cse_emit(expressions, "pops::Real", indent,
                                    materialize_all=True, return_names=True)
    if names:
        lines += [indent + "if (%s) {" % " || ".join("!std::isfinite(%s)" % name for name in names),
                  indent + "  result.status = pops::PathStatus::NonFiniteResult; return result;", indent + "}"]
    return lines, values


def emit_symbolic_path_members(model):
    path = model._path_conservative
    kernel = path["kernel"]
    dimension = len(path["covectors"])
    direction = "std::array<pops::Real, %d>" % dimension
    lines = [
        "  POPS_HD bool path_admissible(const State& state) const {",
        "    for (int k = 0; k < n_vars; ++k) if (!std::isfinite(state[k])) return false;",
        "    return true;", "  }",
        "  POPS_HD pops::PathIntegralResult<n_vars> path_integral(",
        "      const State& left, const State& right, const %s& direction) const {" % direction,
        "    pops::PathIntegralResult<n_vars> result;",
        "    if (!path_admissible(left) || !path_admissible(right)) {",
        "      result.status = pops::PathStatus::NonFiniteInput;",
        "      result.input_side = path_admissible(left) ? 1 : 0; return result;", "    }",
        "    for (auto g : direction) if (!std::isfinite(g)) {",
        "      result.status = pops::PathStatus::NonFiniteDirection; return result;", "    }",
    ]
    for side in ("left", "right"):
        lines += ["    const pops::Real %s = %s[%d];" % (_cpp_identifier(symbol.name), side, k)
                  for k, symbol in enumerate(kernel[side + "_symbols"])]
    for axis in range(dimension):
        lines.append("    if (direction[%d] != pops::Real(0)) {" % axis)
        body, values = _checked_expressions((*kernel["integrals"][axis], kernel["speeds"][axis]), "      ")
        lines += body
        lines += ["      const pops::Real bound = %s;" % values[-1],
                  "      if (!std::isfinite(bound) || bound < pops::Real(0)) {",
                  "        result.status = pops::PathStatus::NonFiniteResult; return result;", "      }",
                  "      result.speed_bound += std::abs(direction[%d]) * bound;" % axis]
        lines += ["      result.integral[%d] += direction[%d] * (%s);" % (k, axis, value)
                  for k, value in enumerate(values[:-1])]
        lines.append("    }")
    lines += ["    for (auto value : result.integral) if (!std::isfinite(value)) {",
              "      result.status = pops::PathStatus::NonFiniteResult; return result;", "    }",
              "    result.status = std::isfinite(result.speed_bound) ? pops::PathStatus::Success",
              "                                                      : pops::PathStatus::NonFiniteResult;",
              "    return result;", "  }",
              "  POPS_HD pops::PathFluxResult<n_vars> path_directional_flux(",
              "      const State& state, const %s& direction) const {" % direction,
              "    pops::PathFluxResult<n_vars> result;",
              "    if (!path_admissible(state)) {",
              "      result.status = pops::PathStatus::NonFiniteInput; return result;", "    }",
              "    for (auto g : direction) if (!std::isfinite(g)) {",
              "      result.status = pops::PathStatus::NonFiniteDirection; return result;", "    }"]
    lines += ["    const pops::Real %s = state[%d];" % (_cpp_identifier(name), k)
              for k, name in enumerate(model.cons_names)]
    for axis, expressions in enumerate(model._flux.values()):
        lines.append("    if (direction[%d] != pops::Real(0)) {" % axis)
        body, values = _checked_expressions(expressions, "      ")
        lines += body
        lines += ["      result.flux.values[%d] += direction[%d] * (%s);" % (k, axis, value)
                  for k, value in enumerate(values)]
        lines.append("    }")
    lines += ["    for (auto value : result.flux.values) if (!std::isfinite(value)) {",
              "      result.status = pops::PathStatus::NonFiniteResult; return result;", "    }",
              "    result.status = pops::PathStatus::Success; return result;", "  }"]
    return lines
