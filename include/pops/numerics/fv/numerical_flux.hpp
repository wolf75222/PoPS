/// @file
/// @brief Typed face numerical-flux policies over PhysicalFlux + two FaceTrace values.
///
/// The final contract deliberately contains no overload accepting a complete Model or raw Aux.
/// Physical constitutive evaluation, face numerical coupling, and mesh divergence are three
/// separate interfaces.  All policies return a flux *density* plus a declared stability bound;
/// geometric face measure belongs exclusively to the spatial operator.

#pragma once

#include <pops/numerics/fv/flux_interfaces.hpp>

#include <concepts>

namespace pops {

namespace detail {

template <class State>
POPS_HD StabilityBound max_bound(const StabilityBound& left, const StabilityBound& right) {
  (void)sizeof(State);
  return {left.value > right.value ? left.value : right.value, StabilityUnit::kLengthPerTime,
          StabilityConvention::kNormalSpectralRadius};
}

template <class Policy, class Physical>
POPS_HD FluxEvaluation<typename Physical::State> canonical_evaluation(
    const Policy& policy, const Physical& physical, const typename Physical::Trace& left,
    const typename Physical::Trace& right, const FaceContext& face) {
  const FaceContext canonical = face.canonical_orientation();
  auto result = policy(physical, right, left, canonical);
  result.reverse_orientation();
  return result;
}

template <class>
inline constexpr bool dependent_false = false;

}  // namespace detail

/// Local Lax-Friedrichs/Rusanov flux.
struct RusanovFlux {
  template <PhysicalFlux Physical>
  POPS_HD FluxEvaluation<typename Physical::State> operator()(
      const Physical& physical, const typename Physical::Trace& left,
      const typename Physical::Trace& right, const FaceContext& face) const {
    if (face.orientation == FaceOrientation::kNegative)
      return detail::canonical_evaluation(*this, physical, left, right, face);
    const auto left_density = physical.evaluate(left, face);
    const auto right_density = physical.evaluate(right, face);
    const auto bound = detail::max_bound<typename Physical::State>(physical.stability(left, face),
                                                                   physical.stability(right, face));
    typename Physical::State density{};
    for (int component = 0; component < Physical::n_vars; ++component) {
      density[component] = Real(0.5) * (left_density.value[component] +
                                        right_density.value[component]) -
                           Real(0.5) * bound.value *
                               (right.state[component] - left.state[component]);
    }
    return FluxEvaluation<typename Physical::State>::ok(density, bound);
  }
};

template <PhysicalFlux Physical>
POPS_HD void hll_speeds(const Physical& physical, const typename Physical::Trace& left,
                        const typename Physical::Trace& right, const FaceContext& face,
                        Real& lower, Real& upper)
  requires requires(const Physical& value, const typename Physical::Trace& trace,
                    const FaceContext& context, Real& lo, Real& hi) {
    value.signed_wave_speeds(trace, context, lo, hi);
  }
{
  Real left_lower, left_upper, right_lower, right_upper;
  physical.signed_wave_speeds(left, face, left_lower, left_upper);
  physical.signed_wave_speeds(right, face, right_lower, right_upper);
  lower = left_lower < right_lower ? left_lower : right_lower;
  upper = left_upper > right_upper ? left_upper : right_upper;
}

template <PhysicalFlux Physical>
POPS_HD FluxEvaluation<typename Physical::State> hll_flux_with_speeds(
    const Physical& physical, const typename Physical::Trace& left,
    const typename Physical::Trace& right, const FaceContext& face, Real lower, Real upper) {
  const auto left_density = physical.evaluate(left, face);
  const auto right_density = physical.evaluate(right, face);
  const auto bound = detail::max_bound<typename Physical::State>(physical.stability(left, face),
                                                                 physical.stability(right, face));
  if (lower >= Real(0))
    return FluxEvaluation<typename Physical::State>::ok(left_density.value, bound);
  if (upper <= Real(0))
    return FluxEvaluation<typename Physical::State>::ok(right_density.value, bound);
  typename Physical::State density{};
  const Real inverse = Real(1) / (upper - lower);
  for (int component = 0; component < Physical::n_vars; ++component) {
    density[component] =
        (upper * left_density.value[component] - lower * right_density.value[component] +
         lower * upper * (right.state[component] - left.state[component])) *
        inverse;
  }
  return FluxEvaluation<typename Physical::State>::ok(density, bound);
}

/// Harten-Lax-van Leer two-wave flux.
struct HLLFlux {
  template <PhysicalFlux Physical>
  POPS_HD FluxEvaluation<typename Physical::State> operator()(
      const Physical& physical, const typename Physical::Trace& left,
      const typename Physical::Trace& right, const FaceContext& face) const {
    if (face.orientation == FaceOrientation::kNegative)
      return detail::canonical_evaluation(*this, physical, left, right, face);
    if constexpr (requires(Real& lo, Real& hi) {
                    physical.signed_wave_speeds(left, face, lo, hi);
                  }) {
      Real lower, upper;
      hll_speeds(physical, left, right, face, lower, upper);
      return hll_flux_with_speeds(physical, left, right, face, lower, upper);
    } else {
      static_assert(detail::dependent_false<Physical>,
                    "HLLFlux requires the signed-wave-speed PhysicalFlux interface");
    }
  }
};

template <class Physical>
concept HLLCPhysicalFlux = PhysicalFlux<Physical> &&
    requires(const Physical& physical, const typename Physical::Trace& trace,
             const typename Physical::State& state, const FaceContext& face, Real& lo, Real& hi,
             Real scalar) {
      physical.signed_wave_speeds(trace, face, lo, hi);
      { physical.pressure(state) } -> std::convertible_to<Real>;
      { physical.contact_speed(state, state, scalar, scalar, scalar, scalar, face) } ->
          std::convertible_to<Real>;
      { physical.star_state(state, scalar, scalar, scalar, face) } ->
          std::same_as<typename Physical::State>;
    };

/// Contact-resolving HLLC policy.  Physical structure is supplied by the narrow PhysicalFlux.
struct HLLCFlux {
  template <PhysicalFlux Physical>
  POPS_HD FluxEvaluation<typename Physical::State> operator()(
      const Physical& physical, const typename Physical::Trace& left,
      const typename Physical::Trace& right, const FaceContext& face) const {
    if (face.orientation == FaceOrientation::kNegative)
      return detail::canonical_evaluation(*this, physical, left, right, face);
    if constexpr (HLLCPhysicalFlux<Physical>) {
      Real lower, upper;
      hll_speeds(physical, left, right, face, lower, upper);
      const auto left_density = physical.evaluate(left, face);
      const auto right_density = physical.evaluate(right, face);
      const auto bound = detail::max_bound<typename Physical::State>(
          physical.stability(left, face), physical.stability(right, face));
      if (lower >= Real(0))
        return FluxEvaluation<typename Physical::State>::ok(left_density.value, bound);
      if (upper <= Real(0))
        return FluxEvaluation<typename Physical::State>::ok(right_density.value, bound);
      const Real pressure_left = physical.pressure(left.state);
      const Real pressure_right = physical.pressure(right.state);
      const Real contact = physical.contact_speed(left.state, right.state, pressure_left,
                                                  pressure_right, lower, upper, face);
      typename Physical::State density{};
      if (contact >= Real(0)) {
        const auto star = physical.star_state(left.state, pressure_left, lower, contact, face);
        for (int component = 0; component < Physical::n_vars; ++component)
          density[component] = left_density.value[component] +
                               lower * (star[component] - left.state[component]);
      } else {
        const auto star = physical.star_state(right.state, pressure_right, upper, contact, face);
        for (int component = 0; component < Physical::n_vars; ++component)
          density[component] = right_density.value[component] +
                               upper * (star[component] - right.state[component]);
      }
      return FluxEvaluation<typename Physical::State>::ok(density, bound);
    } else {
      static_assert(detail::dependent_false<Physical>,
                    "HLLCFlux requires pressure, signed wave speeds, contact speed and star state");
    }
  }
};

template <class Physical>
concept RoePhysicalFlux = PhysicalFlux<Physical> &&
    requires(const Physical& physical, const typename Physical::Trace& left,
             const typename Physical::Trace& right, const FaceContext& face) {
      { physical.roe_dissipation(left, right, face) } ->
          std::same_as<typename Physical::State>;
    };

/// Roe-like policy.  Eigenstructure and entropy policy belong to the physical provider.
struct RoeFlux {
  template <PhysicalFlux Physical>
  POPS_HD FluxEvaluation<typename Physical::State> operator()(
      const Physical& physical, const typename Physical::Trace& left,
      const typename Physical::Trace& right, const FaceContext& face) const {
    if (face.orientation == FaceOrientation::kNegative)
      return detail::canonical_evaluation(*this, physical, left, right, face);
    if constexpr (RoePhysicalFlux<Physical>) {
      const auto left_density = physical.evaluate(left, face);
      const auto right_density = physical.evaluate(right, face);
      const auto dissipation = physical.roe_dissipation(left, right, face);
      const auto bound = detail::max_bound<typename Physical::State>(
          physical.stability(left, face), physical.stability(right, face));
      typename Physical::State density{};
      for (int component = 0; component < Physical::n_vars; ++component) {
        density[component] = Real(0.5) *
                                 (left_density.value[component] +
                                  right_density.value[component]) -
                             Real(0.5) * dissipation[component];
      }
      return FluxEvaluation<typename Physical::State>::ok(density, bound);
    } else {
      static_assert(detail::dependent_false<Physical>,
                    "RoeFlux requires the Roe-dissipation PhysicalFlux interface");
    }
  }
};

/// Explicit Euler presets remain distinct route types while sharing the exact generic algorithms.
/// There is one implementation, not two drifting numerical branches.
struct EulerHLLCFlux2D : HLLCFlux {};
struct EulerRoeFlux2D : RoeFlux {};

inline constexpr Real kRoeEntropyFixFraction = Real(0.1);

}  // namespace pops
