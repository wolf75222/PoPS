#pragma once

#include <pops/numerics/fv/flux_interfaces.hpp>
#include <pops/numerics/moments/fan_li15_interface.hpp>

namespace pops {

namespace fan_li15_face_detail {
/// Floating midpoint with at most one inexact operation for finite IEEE inputs.
/// Scaling both inputs unconditionally would erase equal subnormal covectors.
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
}  // namespace fan_li15_face_detail

/// A distinct path API: deliberately not an ordinary NumericalFlux policy.
/// The result contains one conservative transfer and two independent residual
/// contributions. A caller cannot discard the latter via checked_density().
struct FanLi15PathRusanovFlux {
  template <int Axis, class Model>
  POPS_HD moments::FanLi15InterfaceResult evaluate_path(
      const Model& model, const typename Model::State& left,
      const BoundFluxProviders<Model>& left_providers, const typename Model::State& right,
      const BoundFluxProviders<Model>& right_providers) const {
    static_assert(path_conservative_model<Model>);
    static_assert(Model::dimension == 2 && Model::n_vars == 15 && Axis >= 0 && Axis < 2);
    const auto gl = model.template path_covector<Axis>(left_providers);
    const auto gr = model.template path_covector<Axis>(right_providers);
    // The same fixed g drives BOTH endpoint Grad fluxes, the whole raw path and
    // its bound. Separate trace-speed maxima do not establish that path bound.
    const Real gx = fan_li15_face_detail::common_covector_component(gl[0], gr[0]);
    const Real gy = fan_li15_face_detail::common_covector_component(gl[1], gr[1]);
    Real raw_left[15], raw_right[15];
    for (int k = 0; k < 15; ++k) {
      raw_left[k] = left[k];
      raw_right[k] = right[k];
    }
    return moments::fan_li15_rusanov_interface(raw_left, raw_right, gx, gy);
  }
};

template <int Axis, class Model, int Dim, class LeftStorage, class RightStorage>
POPS_HD moments::FanLi15InterfaceResult evaluate_fan_li15_path_at(
    const FanLi15PathRusanovFlux& numerical, const Model& model, const typename Model::State& left,
    const LeftStorage& left_providers, const Index<Dim>& left_index,
    const typename Model::State& right, const RightStorage& right_providers,
    const Index<Dim>& right_index) {
  return numerical.template evaluate_path<Axis>(
      model, left, bind_flux_providers_at<Model>(left_providers, left_index), right,
      bind_flux_providers_at<Model>(right_providers, right_index));
}

}  // namespace pops
