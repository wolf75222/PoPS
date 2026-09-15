"""Native model members for an authenticated full Fan–Li15 path operator."""
import json

from .module_emit_helpers import _codegen_exprs


def emit_path_members(model, *, cse, aux_locals):
    path = getattr(model, "_path_conservative", None)
    if path is None:
        return []
    if model.n_vars != 15 or len(path["covectors"]) != 2:
        raise ValueError("native Fan–Li path requires dimension two and fifteen raw moments")
    if model._stab_speed is not None:
        raise ValueError("the Fan–Li path owns its incident-face CFL proposal speed")
    lines = [
        "  static constexpr bool path_conservative = true;",
        "  static constexpr std::string_view path_operator_identity() noexcept {",
        "    return %s;" % json.dumps(path["identity"]),
        "  }",
        "  POPS_HD static constexpr std::array<bool, 4> path_zero_measure_faces() {",
        "    return {%s};" % ", ".join("true" if value else "false" for value in path["zero_measure_faces"]),
        "  }",
        "  template <int Axis, class Providers>",
        "  POPS_HD std::array<pops::Real, 2> path_covector(const Providers& a) const {",
        '    static_assert(Axis >= 0 && Axis < 2, "Fan–Li path direction is outside its frame");',
    ]
    lines += aux_locals()
    for axis, covector in enumerate(path["covectors"]):
        lines.append("    %s constexpr (Axis == %d) {" % ("if" if axis == 0 else "else if", axis))
        statements, expressions = _codegen_exprs(model, covector, cse, indent="      ")
        lines += statements
        lines.append("      return {%s};" % ", ".join(expressions))
        lines.append("    }")
    lines += [
        "  }",
        "  POPS_HD bool path_admissible(const State& U) const {",
        "    pops::Real raw[15]{};",
        "    for (int component = 0; component < 15; ++component) raw[component] = U[component];",
        "    return pops::moments::fan_li15_admissibility(raw) ==",
        "        pops::moments::FanLi15PathStatus::Success;",
        "  }", "",
        "  template <int Axis, class Providers>",
        "  POPS_HD pops::Real stability_speed(const State& U, const Providers& a) const {",
        "    // step_cfl reduces max(axis speed)/h_min; each of two axes has two faces.",
        "    // Actual common-face and canonical subface speeds are checked at every RHS.",
        "    return pops::Real(4) * max_wave_speed<Axis>(U, a);",
        "  }", "",
    ]
    return lines


def emit_path_proposal_speed():
    # The current-state speed proposes a step. The hierarchy RHS barrier separately checks
    # actual source-transformed/predictor common-face and canonical subface speeds.
    return [
        "    const auto g = path_covector<Axis>(a);",
        "    pops::Real raw[15]{};",
        "    for (int component = 0; component < 15; ++component) raw[component] = U[component];",
        "    const auto result = pops::moments::fan_li15_path_integral(raw, raw, g[0], g[1]);",
        "    return result.succeeded() ? result.speed_bound : std::numeric_limits<pops::Real>::quiet_NaN();",
        "  }", "",
    ]
