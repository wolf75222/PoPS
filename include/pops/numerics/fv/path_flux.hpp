#pragma once

#include <pops/numerics/fv/flux_interfaces.hpp>
#include <pops/numerics/fv/path_result.hpp>

#include <cmath>
#include <limits>

namespace pops {

/// Device-callable midpoint with at most one inexact operation for finite IEEE
/// inputs. Unconditional scaling would erase equal subnormal components.
POPS_HD inline Real common_covector_component(Real a, Real b) {
  constexpr Real half_max = std::numeric_limits<Real>::max() / Real(2);
  constexpr Real twice_min = std::numeric_limits<Real>::min() * Real(2);
  const Real aa = std::abs(a), ab = std::abs(b);
  if (aa <= half_max && ab <= half_max)
    return (a + b) / Real(2);
  if (aa < twice_min)
    return a + b / Real(2);
  if (ab < twice_min)
    return a / Real(2) + b;
  return a / Real(2) + b / Real(2);
}

/// First-order Rusanov/DLM tuple for an authored physical path and fixed geometry.
/// The two nonconservative residuals are distinct from the shared flux transfer.
template <int N, class Model, class State, class Direction>
POPS_HD PathInterfaceResult<N> path_rusanov_interface(const Model& model, const State& left,
                                                      const State& right,
                                                      const Direction& direction) {
  PathInterfaceResult<N> result;
  const auto path = model.path_integral(left, right, direction);
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(path)>, PathIntegralResult<N>>);
  result.status = path.status;
  result.input_side = path.input_side;
  if (!path.succeeded())
    return result;
  const auto left_flux = model.path_directional_flux(left, direction);
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(left_flux)>, PathFluxResult<N>>);
  if (!left_flux.succeeded()) {
    result.status = left_flux.status;
    result.input_side = 0;
    return result;
  }
  const auto right_flux = model.path_directional_flux(right, direction);
  if (!right_flux.succeeded()) {
    result.status = right_flux.status;
    result.input_side = 1;
    return result;
  }
  if (!std::isfinite(path.speed_bound) || path.speed_bound < Real(0)) {
    result.status = PathStatus::NonFiniteResult;
    return result;
  }
  Real conservative[N]{}, nonconservative[N]{};
  for (int k = 0; k < N; ++k) {
    const Real dissipative = path.speed_bound == Real(0)
                                 ? Real(0)
                                 : (Real(0.5) * path.speed_bound) * (right[k] - left[k]);
    conservative[k] =
        Real(0.5) * left_flux.flux.values[k] + Real(0.5) * right_flux.flux.values[k] - dissipative;
    nonconservative[k] = Real(-0.5) * path.integral[k];
    if (!std::isfinite(conservative[k]) || !std::isfinite(nonconservative[k])) {
      result.status = PathStatus::NonFiniteResult;
      return result;
    }
  }
  for (int k = 0; k < N; ++k) {
    result.conservative_flux.values[k] = conservative[k];
    result.left_ncp.values[k] = nonconservative[k];
    result.right_ncp.values[k] = nonconservative[k];
  }
  result.speed_bound = path.speed_bound;
  return result;
}

/// A separate numerical policy: an ordinary flux-only consumer cannot discard
/// either nonconservative side contribution. Geometry is shared by both endpoint
/// fluxes, the complete physical path, and its speed certificate.
template <int N>
struct PathRusanovFlux {
  template <int Axis, class Model>
  POPS_HD PathInterfaceResult<N> evaluate_path(
      const Model& model, const typename Model::State& left,
      const BoundFluxProviders<Model>& left_providers, const typename Model::State& right,
      const BoundFluxProviders<Model>& right_providers) const {
    static_assert(path_conservative_model<Model> && Model::n_vars == N);
    static_assert(Axis >= 0 && Axis < Model::dimension);
    auto direction = model.template path_covector<Axis>(left_providers);
    const auto right_direction = model.template path_covector<Axis>(right_providers);
    for (std::size_t k = 0; k < direction.size(); ++k)
      direction[k] = common_covector_component(direction[k], right_direction[k]);
    return path_rusanov_interface<N>(model, left, right, direction);
  }
};

template <int Axis, int N, class Model, int Dim, class LeftStorage, class RightStorage>
POPS_HD PathInterfaceResult<N> evaluate_path_at(
    const PathRusanovFlux<N>& numerical, const Model& model, const typename Model::State& left,
    const LeftStorage& left_providers, const Index<Dim>& left_index,
    const typename Model::State& right, const RightStorage& right_providers,
    const Index<Dim>& right_index) {
  return numerical.template evaluate_path<Axis>(
      model, left, bind_flux_providers_at<Model>(left_providers, left_index), right,
      bind_flux_providers_at<Model>(right_providers, right_index));
}

}  // namespace pops
