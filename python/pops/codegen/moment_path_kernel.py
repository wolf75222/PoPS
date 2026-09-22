"""Generate model-owned moment path kernels from Python constitutive lowering data.

The SDK supplies numerical transforms and analytic integration. This emitter
selects the declared closure, regularized rows, coefficients and speed theorem;
none of those physical choices are compiled into a generic spatial operator.
"""
from __future__ import annotations


def emit_fan_li15_kernel(name: str = "PathKernel") -> list[str]:
    from pops.moments.fan_li import fan_li15_native_plan
    return emit_moment_path_kernel(fan_li15_native_plan(), name)


def emit_moment_path_kernel(plan: dict, name: str) -> list[str]:
    """Lower a physical moment plan onto the generic, stable numerical primitives."""
    order = plan["order"]
    if order not in (2, 3, 4) or plan["closure_order"] != order:
        raise ValueError("normalized moment path requires a supported complete Hermite order")
    indices = tuple(plan["indices"])
    canonical = tuple((p, q) for q in range(order + 1) for p in range(order + 1 - q))
    if indices != canonical:
        raise ValueError("normalized moment path requires the exact q-outer raw moment ordering")
    size = len(indices)
    offset, radical = plan["bound_terms"]
    lines = [
        f"struct {name} {{",
        "  using Real = pops::Real;",
        "  using Direction = std::array<Real, 2>;",
        f"  using Recovered = pops::moments::NormalizedRawMoments<{order}>;",
        f"  using Poly = pops::moments::Polynomial<{order}>;",
        "  template <class State>",
        "  POPS_HD static pops::PathStatus recover(const State& input, Recovered& state) {",
        f"    Real raw[{size}];",
        f"    for (int k = 0; k < {size}; ++k) raw[k] = input[k];",
        f"    return pops::moments::recover_raw_moments<{order}>(raw, state);",
        "  }",
        "  template <class State>",
        "  POPS_HD static pops::PathStatus admissibility(const State& input) {",
        "    Recovered state;",
        "    return recover(input, state);",
        "  }",
        "  POPS_HD static Real speed_bound(const Recovered& state, const Direction& g) {",
        f"    return std::sqrt(Real({offset}) + std::sqrt(Real({radical}))) *",
        "        pops::moments::raw_second_moment_norm(state, g[0], g[1]);",
        "  }",
        "  template <class State>",
        f"  POPS_HD pops::PathFluxResult<{size}> path_directional_flux(",
        "      const State& raw, const Direction& g) const {",
        f"    pops::PathFluxResult<{size}> result;",
        "    Recovered state;",
        "    result.status = recover(raw, state);",
        "    if (!result.succeeded()) return result;",
        "    if (!std::isfinite(g[0]) || !std::isfinite(g[1])) {",
        "      result.status = pops::PathStatus::NonFiniteDirection;",
        "      return result;",
        "    }",
        "    if (g[0] == Real(0) && g[1] == Real(0)) return result;",
        f"    Real h[{size}]{{}}, temperature[3], edge[{order + 2}];",
        "    pops::moments::normalized_hermite(state, h, temperature);",
        f"    pops::moments::hermite_raw_edge<{plan['closure_order']}, {order + 1}>(",
        "        state, h, temperature, edge);",
        f"    Real candidate[{size}]{{}};",
    ]
    for slot, (p, q) in enumerate(indices):
        if p + q < order:
            expr = (f"g[0] * raw[{indices.index((p + 1, q))}] + "
                    f"g[1] * raw[{indices.index((p, q + 1))}]")
        else:
            expr = f"state.density * (g[0] * edge[{q}] + g[1] * edge[{q + 1}])"
        lines.append(f"    candidate[{slot}] = {expr};")
    lines += [
        "    for (Real value : candidate) {",
        "      if (!std::isfinite(value)) {",
        "        result.status = pops::PathStatus::NonFiniteResult;",
        "        return result;",
        "      }",
        "    }",
        f"    for (int k = 0; k < {size}; ++k) result.flux.values[k] = candidate[k];",
        "    return result;",
        "  }",
        "  POPS_HD static void path_polynomials(",
        "      const Recovered& left, const Recovered& right, const Direction& g,",
        f"      Poly (&integrands)[{size}], Real (&factors)[{size}]) {{",
        "    using namespace pops::moments;",
        f"    Poly h[{size}], theta[3], u, v;",
        "    normalized_hermite_path(left, right, h, theta, u, v);",
        "    const Poly derivatives[] = {derivative(u), derivative(v),",
        "        derivative(theta[0]), derivative(theta[1]), derivative(theta[2])};",
    ]
    # The same primary multi-index terms build the Python B matrix. Emitting
    # their staged polynomial products preserves the qualified floating order.
    for slot, factor, directions in plan["regularization"]:
        lines.append("    {")
        for axis, (_, terms) in enumerate(directions):
            products = []
            for (p, q), derivative, divisor in terms:
                # Normalized Hermite order m has polynomial degree at most m;
                # d(u,v)/dw are constant and dTheta/dw is affine. Only this
                # proved bounded algebra may use the fixed-capacity primitive.
                if p >= 0 and q >= 0 and p + q <= order:
                    if p + q + (0 if derivative < 2 else 1) > order:
                        raise ValueError("path integrand exceeds its proved polynomial degree")
                product = f"multiply(hermite_coefficient<{order}>(h, {p}, {q}), derivatives[{derivative}])"
                if divisor != 1:
                    product = f"scale({product}, Real(1) / Real({divisor}))"
                products.append(product)
            lines.append(f"      Poly r{axis} = add({products[0]}, {products[1]});")
            for product in products[2:]:
                lines.append(f"      r{axis} = add(r{axis}, {product});")
        lines += [
            f"      integrands[{slot}] = add(scale(r0, Real({directions[0][0]}) * g[0]),",
            f"          scale(r1, Real({directions[1][0]}) * g[1]));",
            f"      factors[{slot}] = Real({factor});",
            "    }",
        ]
    lines += [
        "  }",
        "  template <class State>",
        f"  POPS_HD pops::PathIntegralResult<{size}> path_integral(",
        "      const State& left, const State& right, const Direction& g) const {",
        f"    return pops::moments::integrate_normalized_moment_path<{order}>(left, right, g, *this);",
        "  }",
        "};",
    ]
    return lines


def emit_fan_li15_test_header() -> str:
    """A reproducible model fixture; never an installed production SDK header."""
    return ("// Generated from pops.moments.fan_li by moment_path_kernel.py. Do not edit.\n"
            "// clang-format off\n"
            "#pragma once\n\n"
            "#include <pops/numerics/moments/normalized_moment_path.hpp>\n"
            "#include <pops/numerics/fv/path_flux.hpp>\n"
            "#include <array>\n\nnamespace test_fan_li15 {\n\n"
            + "\n".join(emit_fan_li15_kernel("Kernel"))
            + """

template <class State>
POPS_HD auto flux(const State& raw, pops::Real gx, pops::Real gy) {
  return Kernel{}.path_directional_flux(raw, {gx, gy});
}
template <class State>
POPS_HD auto path_integral(const State& left, const State& right, pops::Real gx, pops::Real gy) {
  return Kernel{}.path_integral(left, right, {gx, gy});
}
template <class State>
POPS_HD auto interface(const State& left, const State& right, pops::Real gx, pops::Real gy) {
  return pops::path_rusanov_interface<15>(Kernel{}, left, right, Kernel::Direction{gx, gy});
}

}  // namespace test_fan_li15
// clang-format on
""")
