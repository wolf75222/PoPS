/// @file
/// @brief The physical flux of a face whose unit normal is not a logical axis (N17, C1).
///
/// On a mapped configuration served by generated geometry arrays, the face of logical axis a has
/// the physical unit normal n of its area vector. A face law then reads the normal flux
/// F.n = sum_k n_k F_k and a bound of the spectral radius of the normal Jacobian sum_k n_k A_k.
/// The bound below is sum_k |n_k| rho(A_k): for a symmetrizable hyperbolic system one symmetrizer
/// makes every A_k self-adjoint in its inner product, where the spectral radius is the norm, so the
/// norm of the sum is at most the sum of the norms. The view provides nothing else: a face law that
/// reads signed wave speeds, a contact or a dissipation matrix along one axis does not compile
/// against it, so only the laws whose declared requirements are the normal flux, the stability
/// bound and the provider pack are served (the Python resolution refuses the others first).

#pragma once

#include <pops/numerics/fv/flux_interfaces.hpp>

#include <concepts>
#include <limits>

namespace pops {

template <class Model>
struct NormalPhysicalFluxView {
  using State = typename Model::State;
  using ProviderPack = BoundFluxProviders<Model>;
  using Trace = FaceTrace<State, ProviderPack>;
  static constexpr int dimension = physical_model_dimension<Model>;
  static constexpr int n_vars = Model::n_vars;
  static constexpr bool path_conservative = false;
  static_assert(detail::flux_all_axes<Model>(),
                "a physical normal reads the flux and the wave speed of every ranked axis");
  static_assert(!path_conservative_model<Model>,
                "a path-conservative model has no face law on a physical normal");

  Model physical;

  /// F.n in the canonical orientation of the face (toward increasing logical index).
  POPS_HD FluxDensity<State> evaluate(const Trace& trace, const FaceContext& face) const {
    FluxDensity<State> result{};
    for (int axis = 0; axis < dimension; ++axis) {
      FluxDensity<State> along{};
      if constexpr (requires { physical.flux_evaluation(trace.state, trace.providers, axis); }) {
        along = physical.flux_evaluation(trace.state, trace.providers, axis);
      } else {
        along.value =
            detail::model_flux_at_runtime_axis(physical, trace.state, trace.providers, axis);
      }
      if (along.status != EvaluationStatus::kOk)
        return along;
      for (int component = 0; component < n_vars; ++component)
        result.value[component] += face.normal[axis] * along.value[component];
    }
    return result;
  }

  /// sum_k |n_k| rho(A_k); one invalid axis speed makes the bound invalid, whatever its weight.
  POPS_HD StabilityBound stability(const Trace& trace, const FaceContext& face) const {
    Real bound = Real(0);
    for (int axis = 0; axis < dimension; ++axis) {
      const Real speed = detail::model_max_wave_speed_at_runtime_axis(physical, trace.state,
                                                                      trace.providers, axis);
      if (!Kokkos::isfinite(speed) || speed < Real(0))
        return {std::numeric_limits<Real>::quiet_NaN(), StabilityUnit::kLengthPerTime,
                StabilityConvention::kNormalSpectralRadius};
      bound += Kokkos::fabs(face.normal[axis]) * speed;
    }
    return {bound, StabilityUnit::kLengthPerTime, StabilityConvention::kNormalSpectralRadius};
  }
};

/// Evaluate one face law on the physical normal that @p face carries.
template <class Numerical, class Model, int Dim, class LeftStorage, class RightStorage>
POPS_HD FluxEvaluation<typename Model::State> evaluate_numerical_flux_on_normal_at(
    const Numerical& numerical, const Model& model, const typename Model::State& left_state,
    const LeftStorage& left_providers, const Index<Dim>& left_index,
    const typename Model::State& right_state, const RightStorage& right_providers,
    const Index<Dim>& right_index, const FaceContext& face) {
  using Physical = NormalPhysicalFluxView<Model>;
  static_assert(NumericalFlux<Numerical, Physical>,
                "this face law reads more than the normal flux and its normal stability bound");
  const Physical physical{model};
  const auto left =
      make_face_trace<Model>(left_state, bind_flux_providers_at<Model>(left_providers, left_index));
  const auto right = make_face_trace<Model>(
      right_state, bind_flux_providers_at<Model>(right_providers, right_index));
  auto result = numerical(physical, left, right, face);
  if (result.requested_solver != RiemannSolverId::kUnspecified)
    return result;
  constexpr RiemannSolverId solver = [] {
    if constexpr (requires { Numerical::solver_id; })
      return static_cast<RiemannSolverId>(Numerical::solver_id);
    return RiemannSolverId::kExternal;
  }();
  return result.with_single_solver(solver);
}

}  // namespace pops
