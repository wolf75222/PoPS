"""Native model members for an authenticated numerical path operator."""
import json

from .module_emit_helpers import _codegen_exprs


def emit_path_members(model, *, cse, aux_locals):
    path = getattr(model, "_path_conservative", None)
    if path is None:
        return []
    dimension = len(path["covectors"])
    kernel = path["kernel"]
    if model._stab_speed is not None:
        raise ValueError("the numerical path owns its incident-face CFL proposal speed")
    lines = []
    if kernel["kind"] == "normalized_moment_path":
        from .moment_path_kernel import emit_moment_path_kernel
        if dimension != 2 or len(kernel["plan"]["indices"]) != model.n_vars:
            raise ValueError("normalized moment kernel and native state shapes differ")
        lines += ["  " + line for line in emit_moment_path_kernel(kernel["plan"], "PathKernel")]
    elif kernel["kind"] == "symbolic_path":
        from .module_emit_symbolic_path import emit_symbolic_path_members
        lines += emit_symbolic_path_members(model)
    else:
        raise NotImplementedError("no native lowering for the declared numerical path kernel")
    lines += [
        "  static constexpr bool path_conservative = true;",
        "  static constexpr std::string_view path_operator_identity() noexcept {",
        "    return %s;" % json.dumps(path["identity"]),
        "  }",
        "  POPS_HD static constexpr std::array<bool, %d> path_zero_measure_faces() {" % (2 * dimension),
        "    return {%s};" % ", ".join("true" if value else "false" for value in path["zero_measure_faces"]),
        "  }",
        "  template <int Axis, class Providers>",
        "  POPS_HD std::array<pops::Real, %d> path_covector(const Providers& a) const {" % dimension,
        '    static_assert(Axis >= 0 && Axis < dimension, "path direction is outside its frame");',
    ]
    lines += aux_locals()
    for axis, covector in enumerate(path["covectors"]):
        lines.append("    %s constexpr (Axis == %d) {" % ("if" if axis == 0 else "else if", axis))
        statements, expressions = _codegen_exprs(model, covector, cse, indent="      ")
        lines += statements
        lines.append("      return {%s};" % ", ".join(expressions))
        lines.append("    }")
    lines.append("  }")
    if kernel["kind"] == "normalized_moment_path":
        lines += [
        "  POPS_HD bool path_admissible(const State& U) const {",
        "    return PathKernel::admissibility(U) == pops::PathStatus::Success;",
        "  }", "",
        "  POPS_HD pops::PathIntegralResult<n_vars> path_integral(const State& left, const State& right,",
        "                              const std::array<pops::Real, 2>& direction) const {",
        "    return PathKernel{}.path_integral(left, right, direction);",
        "  }",
        "  POPS_HD pops::PathFluxResult<n_vars> path_directional_flux(const State& state,",
        "                                      const std::array<pops::Real, 2>& direction) const {",
        "    return PathKernel{}.path_directional_flux(state, direction);",
        "  }",
        ]
    lines += [
        "  template <int Axis, class Providers>",
        "  POPS_HD pops::Real stability_speed(const State& U, const Providers& a) const {",
        "    // step_cfl reduces max(axis speed)/h_min; every axis has two faces.",
        "    // Actual common-face and canonical subface speeds are checked at every RHS.",
        "    return pops::Real(%d) * max_wave_speed<Axis>(U, a);" % (2 * dimension),
        "  }", "",
    ]
    return lines


def emit_path_proposal_speed():
    # The current-state speed proposes a step. The hierarchy RHS barrier separately checks
    # actual source-transformed/predictor common-face and canonical subface speeds.
    # This adapter consumes only U and the provider pack a. Do not inject named
    # scientific locals here: the path methods own those bindings, and a state
    # component named a/g/result is unrelated to these private adapter variables.
    return [
        "    const auto g = path_covector<Axis>(a);",
        "    const auto result = path_integral(U, U, g);",
        "    return result.succeeded() ? result.speed_bound : std::numeric_limits<pops::Real>::quiet_NaN();",
        "  }", "",
    ]
