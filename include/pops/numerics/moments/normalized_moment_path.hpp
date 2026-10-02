#pragma once

#include <pops/numerics/moments/raw_moment_recovery.hpp>
#include <pops/numerics/moments/density_path_arithmetic.hpp>

namespace pops::moments {

/// Analytic straight-raw-state integration after density normalization. The
/// caller supplies the physical polynomial coefficients and the whole-path
/// speed majorant. Neither a moment closure nor a regularization is selected here.
template <int Order, int Degree = Order, class State, class Direction, class Law>
POPS_HD PathIntegralResult<CartesianMomentBasis<Order>::size> integrate_normalized_moment_path(
    const State& raw_left, const State& raw_right, const Direction& direction, const Law& law) {
  constexpr int N = CartesianMomentBasis<Order>::size;
  static_assert(Degree >= 0 && Degree < std::numeric_limits<int>::max(),
                "polynomial capacity must fit the native coefficient index type");
  PathIntegralResult<N> result;
  NormalizedRawMoments<Order> left, right;
  auto status = law.recover(raw_left, left);
  if (status != PathStatus::Success) {
    result.status = status;
    result.input_side = 0;
    return result;
  }
  status = law.recover(raw_right, right);
  if (status != PathStatus::Success) {
    result.status = status;
    result.input_side = 1;
    return result;
  }
  bool zero_direction = true;
  for (Real component : direction) {
    if (!std::isfinite(component)) {
      result.status = PathStatus::NonFiniteDirection;
      return result;
    }
    zero_direction = zero_direction && component == Real(0);
  }
  const auto endpoint_speed = [&](const auto& raw, const auto& recovered) {
    if constexpr (requires { law.speed_bound(raw, recovered, direction); })
      return law.speed_bound(raw, recovered, direction);
    else
      return law.speed_bound(recovered, direction);
  };
  const Real speed_left = endpoint_speed(raw_left, left),
             speed_right = endpoint_speed(raw_right, right);
  const Real speed = speed_left > speed_right ? speed_left : speed_right;
  if (!std::isfinite(speed_left) || !std::isfinite(speed_right) || speed_left < Real(0) ||
      speed_right < Real(0)) {
    result.status = PathStatus::NonFiniteResult;
    return result;
  }
  bool identical = true, reverse = false;
  // One canonical orientation gives exactly antisymmetric evaluation, including
  // equal-density endpoints. Normalized raw density is the first component.
  for (int k = 0; k < N; ++k) {
    // An explicit basis binding may permute physical storage. Canonical
    // comparison is a numerical orientation, not a requirement on that storage.
    const auto component = [&](const auto& raw) {
      if constexpr (requires { law.canonical_component(raw, k); })
        return law.canonical_component(raw, k);
      else
        return raw[k];
    };
    const Real a = component(raw_left), b = component(raw_right);
    if (a != b) {
      identical = false;
      reverse = a < b;
      break;
    }
  }
  if (identical || zero_direction) {
    result.status = PathStatus::Success;
    result.speed_bound = speed;
    return result;
  }
  const auto& first = reverse ? right : left;
  const auto& second = reverse ? left : right;
  Real weight[Degree + 1], density_scale;
  if (!density_path_weights(first.density, second.density, weight, density_scale))
    return result;
  Polynomial<Degree> integrands[N];
  Real factors[N]{};
  law.path_polynomials(first, second, direction, integrands, factors);
  Real candidate[N]{};
  for (int component = 0; component < N; ++component) {
    if (factors[component] == Real(0))
      continue;
    Real sum = Real(0), compensation = Real(0);
    for (int k = 0; k <= Degree; ++k) {
      const Real term = integrands[component].c[k] * weight[k];
      const Real corrected = term - compensation, next = sum + corrected;
      compensation = (next - sum) - corrected;
      sum = next;
    }
    candidate[component] =
        (reverse ? -factors[component] : factors[component]) * sum * density_scale;
    if (!std::isfinite(candidate[component])) {
      result.status = PathStatus::NonFiniteResult;
      return result;
    }
  }
  for (int k = 0; k < N; ++k)
    result.integral[k] = candidate[k];
  result.status = PathStatus::Success;
  result.speed_bound = speed;
  return result;
}

}  // namespace pops::moments
