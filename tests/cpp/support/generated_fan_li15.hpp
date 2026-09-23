// Generated from pops.moments.fan_li by moment_path_kernel.py. Do not edit.
// clang-format off
#pragma once

#include <pops/numerics/moments/normalized_moment_path.hpp>
#include <pops/numerics/fv/path_flux.hpp>
#include <array>

namespace test_fan_li15 {

struct Kernel {
  using Real = pops::Real;
  using Direction = std::array<Real, 2>;
  using Recovered = pops::moments::NormalizedRawMoments<4>;
  using Poly = pops::moments::Polynomial<4>;
  template <class State>
  POPS_HD static pops::PathStatus recover(const State& input, Recovered& state) {
    Real raw[15];
    for (int k = 0; k < 15; ++k) raw[k] = input[k];
    return pops::moments::recover_raw_moments<4>(raw, state);
  }
  template <class State>
  POPS_HD static pops::PathStatus admissibility(const State& input) {
    Recovered state;
    return recover(input, state);
  }
  POPS_HD static Real speed_bound(const Recovered& state, const Direction& g) {
    return std::sqrt(Real(6) + std::sqrt(Real(10))) *
        pops::moments::raw_second_moment_norm(state, g[0], g[1]);
  }
  template <class State>
  POPS_HD pops::PathFluxResult<15> path_directional_flux(
      const State& raw, const Direction& g) const {
    pops::PathFluxResult<15> result;
    Recovered state;
    result.status = recover(raw, state);
    if (!result.succeeded()) return result;
    if (!std::isfinite(g[0]) || !std::isfinite(g[1])) {
      result.status = pops::PathStatus::NonFiniteDirection;
      return result;
    }
    if (g[0] == Real(0) && g[1] == Real(0)) return result;
    Real h[15]{}, temperature[3], edge[6];
    pops::moments::normalized_hermite(state, h, temperature);
    pops::moments::hermite_raw_edge<4, 5>(
        state, h, temperature, edge);
    Real candidate[15]{};
    candidate[0] = g[0] * raw[1] + g[1] * raw[5];
    candidate[1] = g[0] * raw[2] + g[1] * raw[6];
    candidate[2] = g[0] * raw[3] + g[1] * raw[7];
    candidate[3] = g[0] * raw[4] + g[1] * raw[8];
    candidate[4] = state.density * (g[0] * edge[0] + g[1] * edge[1]);
    candidate[5] = g[0] * raw[6] + g[1] * raw[9];
    candidate[6] = g[0] * raw[7] + g[1] * raw[10];
    candidate[7] = g[0] * raw[8] + g[1] * raw[11];
    candidate[8] = state.density * (g[0] * edge[1] + g[1] * edge[2]);
    candidate[9] = g[0] * raw[10] + g[1] * raw[12];
    candidate[10] = g[0] * raw[11] + g[1] * raw[13];
    candidate[11] = state.density * (g[0] * edge[2] + g[1] * edge[3]);
    candidate[12] = g[0] * raw[13] + g[1] * raw[14];
    candidate[13] = state.density * (g[0] * edge[3] + g[1] * edge[4]);
    candidate[14] = state.density * (g[0] * edge[4] + g[1] * edge[5]);
    for (Real value : candidate) {
      if (!std::isfinite(value)) {
        result.status = pops::PathStatus::NonFiniteResult;
        return result;
      }
    }
    for (int k = 0; k < 15; ++k) result.flux.values[k] = candidate[k];
    return result;
  }
  POPS_HD static void path_polynomials(
      const Recovered& left, const Recovered& right, const Direction& g,
      Poly (&integrands)[15], Real (&factors)[15]) {
    using namespace pops::moments;
    Poly h[15], theta[3], u, v;
    normalized_hermite_path(left, right, h, theta, u, v);
    const Poly derivatives[] = {derivative(u), derivative(v),
        derivative(theta[0]), derivative(theta[1]), derivative(theta[2])};
    {
      Poly r0 = add(multiply(hermite_coefficient<4>(h, 4, 0), derivatives[0]), multiply(hermite_coefficient<4>(h, 5, -1), derivatives[1]));
      r0 = add(r0, scale(multiply(hermite_coefficient<4>(h, 3, 0), derivatives[2]), Real(1) / Real(2)));
      r0 = add(r0, multiply(hermite_coefficient<4>(h, 4, -1), derivatives[3]));
      r0 = add(r0, scale(multiply(hermite_coefficient<4>(h, 5, -2), derivatives[4]), Real(1) / Real(2)));
      Poly r1 = add(multiply(hermite_coefficient<4>(h, 3, 1), derivatives[0]), multiply(hermite_coefficient<4>(h, 4, 0), derivatives[1]));
      r1 = add(r1, scale(multiply(hermite_coefficient<4>(h, 2, 1), derivatives[2]), Real(1) / Real(2)));
      r1 = add(r1, multiply(hermite_coefficient<4>(h, 3, 0), derivatives[3]));
      r1 = add(r1, scale(multiply(hermite_coefficient<4>(h, 4, -1), derivatives[4]), Real(1) / Real(2)));
      integrands[4] = add(scale(r0, Real(5) * g[0]),
          scale(r1, Real(1) * g[1]));
      factors[4] = Real(-24);
    }
    {
      Poly r0 = add(multiply(hermite_coefficient<4>(h, 3, 1), derivatives[0]), multiply(hermite_coefficient<4>(h, 4, 0), derivatives[1]));
      r0 = add(r0, scale(multiply(hermite_coefficient<4>(h, 2, 1), derivatives[2]), Real(1) / Real(2)));
      r0 = add(r0, multiply(hermite_coefficient<4>(h, 3, 0), derivatives[3]));
      r0 = add(r0, scale(multiply(hermite_coefficient<4>(h, 4, -1), derivatives[4]), Real(1) / Real(2)));
      Poly r1 = add(multiply(hermite_coefficient<4>(h, 2, 2), derivatives[0]), multiply(hermite_coefficient<4>(h, 3, 1), derivatives[1]));
      r1 = add(r1, scale(multiply(hermite_coefficient<4>(h, 1, 2), derivatives[2]), Real(1) / Real(2)));
      r1 = add(r1, multiply(hermite_coefficient<4>(h, 2, 1), derivatives[3]));
      r1 = add(r1, scale(multiply(hermite_coefficient<4>(h, 3, 0), derivatives[4]), Real(1) / Real(2)));
      integrands[8] = add(scale(r0, Real(4) * g[0]),
          scale(r1, Real(2) * g[1]));
      factors[8] = Real(-6);
    }
    {
      Poly r0 = add(multiply(hermite_coefficient<4>(h, 2, 2), derivatives[0]), multiply(hermite_coefficient<4>(h, 3, 1), derivatives[1]));
      r0 = add(r0, scale(multiply(hermite_coefficient<4>(h, 1, 2), derivatives[2]), Real(1) / Real(2)));
      r0 = add(r0, multiply(hermite_coefficient<4>(h, 2, 1), derivatives[3]));
      r0 = add(r0, scale(multiply(hermite_coefficient<4>(h, 3, 0), derivatives[4]), Real(1) / Real(2)));
      Poly r1 = add(multiply(hermite_coefficient<4>(h, 1, 3), derivatives[0]), multiply(hermite_coefficient<4>(h, 2, 2), derivatives[1]));
      r1 = add(r1, scale(multiply(hermite_coefficient<4>(h, 0, 3), derivatives[2]), Real(1) / Real(2)));
      r1 = add(r1, multiply(hermite_coefficient<4>(h, 1, 2), derivatives[3]));
      r1 = add(r1, scale(multiply(hermite_coefficient<4>(h, 2, 1), derivatives[4]), Real(1) / Real(2)));
      integrands[11] = add(scale(r0, Real(3) * g[0]),
          scale(r1, Real(3) * g[1]));
      factors[11] = Real(-4);
    }
    {
      Poly r0 = add(multiply(hermite_coefficient<4>(h, 1, 3), derivatives[0]), multiply(hermite_coefficient<4>(h, 2, 2), derivatives[1]));
      r0 = add(r0, scale(multiply(hermite_coefficient<4>(h, 0, 3), derivatives[2]), Real(1) / Real(2)));
      r0 = add(r0, multiply(hermite_coefficient<4>(h, 1, 2), derivatives[3]));
      r0 = add(r0, scale(multiply(hermite_coefficient<4>(h, 2, 1), derivatives[4]), Real(1) / Real(2)));
      Poly r1 = add(multiply(hermite_coefficient<4>(h, 0, 4), derivatives[0]), multiply(hermite_coefficient<4>(h, 1, 3), derivatives[1]));
      r1 = add(r1, scale(multiply(hermite_coefficient<4>(h, -1, 4), derivatives[2]), Real(1) / Real(2)));
      r1 = add(r1, multiply(hermite_coefficient<4>(h, 0, 3), derivatives[3]));
      r1 = add(r1, scale(multiply(hermite_coefficient<4>(h, 1, 2), derivatives[4]), Real(1) / Real(2)));
      integrands[13] = add(scale(r0, Real(2) * g[0]),
          scale(r1, Real(4) * g[1]));
      factors[13] = Real(-6);
    }
    {
      Poly r0 = add(multiply(hermite_coefficient<4>(h, 0, 4), derivatives[0]), multiply(hermite_coefficient<4>(h, 1, 3), derivatives[1]));
      r0 = add(r0, scale(multiply(hermite_coefficient<4>(h, -1, 4), derivatives[2]), Real(1) / Real(2)));
      r0 = add(r0, multiply(hermite_coefficient<4>(h, 0, 3), derivatives[3]));
      r0 = add(r0, scale(multiply(hermite_coefficient<4>(h, 1, 2), derivatives[4]), Real(1) / Real(2)));
      Poly r1 = add(multiply(hermite_coefficient<4>(h, -1, 5), derivatives[0]), multiply(hermite_coefficient<4>(h, 0, 4), derivatives[1]));
      r1 = add(r1, scale(multiply(hermite_coefficient<4>(h, -2, 5), derivatives[2]), Real(1) / Real(2)));
      r1 = add(r1, multiply(hermite_coefficient<4>(h, -1, 4), derivatives[3]));
      r1 = add(r1, scale(multiply(hermite_coefficient<4>(h, 0, 3), derivatives[4]), Real(1) / Real(2)));
      integrands[14] = add(scale(r0, Real(1) * g[0]),
          scale(r1, Real(5) * g[1]));
      factors[14] = Real(-24);
    }
  }
  template <class State>
  POPS_HD pops::PathIntegralResult<15> path_integral(
      const State& left, const State& right, const Direction& g) const {
    return pops::moments::integrate_normalized_moment_path<4>(left, right, g, *this);
  }
};

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
