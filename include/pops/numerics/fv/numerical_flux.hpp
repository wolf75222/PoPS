/// @file
/// @brief Typed face numerical-flux policies over PhysicalFlux + two FaceTrace values.
///
/// The final contract deliberately contains no overload accepting a complete Model or raw Aux.
/// Physical constitutive evaluation, face numerical coupling, and mesh divergence are three
/// separate interfaces.  All policies return a flux *density* plus a declared stability bound;
/// geometric face measure belongs exclusively to the spatial operator.

#pragma once

#include <pops/numerics/fv/flux_interfaces.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <cmath>
#include <concepts>
#include <limits>

namespace pops {

namespace detail {

POPS_HD inline bool valid_normal_stability_bound(const StabilityBound& bound) {
  return Kokkos::isfinite(bound.value) && bound.value >= Real(0) &&
         bound.unit == StabilityUnit::kLengthPerTime &&
         bound.convention == StabilityConvention::kNormalSpectralRadius;
}

POPS_HD inline bool max_normal_stability_bound(const StabilityBound& left,
                                               const StabilityBound& right,
                                               StabilityBound& result) {
  if (!valid_normal_stability_bound(left) || !valid_normal_stability_bound(right))
    return false;
  result = {left.value > right.value ? left.value : right.value, StabilityUnit::kLengthPerTime,
            StabilityConvention::kNormalSpectralRadius};
  return true;
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

POPS_HD inline bool valid_hll_speed_interval(Real lower, Real upper) {
  return Kokkos::isfinite(lower) && Kokkos::isfinite(upper) && lower <= upper;
}

template <class State>
POPS_HD inline bool finite_state(const State& state) {
  for (int component = 0; component < State::size(); ++component)
    if (!Kokkos::isfinite(state[component]))
      return false;
  return true;
}

/// Union two independently certified signed-wave-speed intervals.  Validate both traces before
/// min/max: IEEE comparisons with NaN are false, so taking the union first could silently discard
/// an invalid left or right trace and manufacture a plausible finite HLL interval.
POPS_HD inline void union_hll_speed_intervals(Real left_lower, Real left_upper, Real right_lower,
                                              Real right_upper, Real& lower, Real& upper) {
  if (!valid_hll_speed_interval(left_lower, left_upper) ||
      !valid_hll_speed_interval(right_lower, right_upper)) {
    lower = upper = std::numeric_limits<Real>::quiet_NaN();
    return;
  }
  lower = left_lower < right_lower ? left_lower : right_lower;
  upper = left_upper > right_upper ? left_upper : right_upper;
}

}  // namespace detail

/// Local Lax-Friedrichs/Rusanov flux.
struct RusanovFlux {
  template <PhysicalFlux Physical>
  POPS_HD FluxEvaluation<typename Physical::State> operator()(const Physical& physical,
                                                              const typename Physical::Trace& left,
                                                              const typename Physical::Trace& right,
                                                              const FaceContext& face) const {
    if (face.orientation == FaceOrientation::kNegative)
      return detail::canonical_evaluation(*this, physical, left, right, face);
    StabilityBound bound{};
    if (!detail::max_normal_stability_bound(physical.stability(left, face),
                                            physical.stability(right, face), bound))
      return FluxEvaluation<typename Physical::State>::reject(
          RiemannFailureCause::kRusanovInvalidStability);
    const auto left_density = physical.evaluate(left, face);
    const auto right_density = physical.evaluate(right, face);
    typename Physical::State density{};
    for (int component = 0; component < Physical::n_vars; ++component) {
      density[component] =
          Real(0.5) * (left_density.value[component] + right_density.value[component]) -
          Real(0.5) * bound.value * (right.state[component] - left.state[component]);
    }
    return FluxEvaluation<typename Physical::State>::ok(density, bound);
  }
};

template <PhysicalFlux Physical>
POPS_HD void hll_speeds(const Physical& physical, const typename Physical::Trace& left,
                        const typename Physical::Trace& right, const FaceContext& face, Real& lower,
                        Real& upper)
  requires requires(const Physical& value, const typename Physical::Trace& trace,
                    const FaceContext& context, Real& lo,
                    Real& hi) { value.signed_wave_speeds(trace, context, lo, hi); }
{
  Real left_lower, left_upper, right_lower, right_upper;
  physical.signed_wave_speeds(left, face, left_lower, left_upper);
  physical.signed_wave_speeds(right, face, right_lower, right_upper);
  detail::union_hll_speed_intervals(left_lower, left_upper, right_lower, right_upper, lower, upper);
}

template <PhysicalFlux Physical>
POPS_HD FluxEvaluation<typename Physical::State> hll_flux_with_speeds(
    const Physical& physical, const typename Physical::Trace& left,
    const typename Physical::Trace& right, const FaceContext& face, Real lower, Real upper) {
  if (!detail::valid_hll_speed_interval(lower, upper))
    return FluxEvaluation<typename Physical::State>::reject(
        RiemannFailureCause::kHllInvalidWaveInterval);
  StabilityBound bound{};
  if (!detail::max_normal_stability_bound(physical.stability(left, face),
                                          physical.stability(right, face), bound))
    return FluxEvaluation<typename Physical::State>::reject(
        RiemannFailureCause::kHllInvalidStability);
  const auto left_density = physical.evaluate(left, face);
  const auto right_density = physical.evaluate(right, face);
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
  POPS_HD FluxEvaluation<typename Physical::State> operator()(const Physical& physical,
                                                              const typename Physical::Trace& left,
                                                              const typename Physical::Trace& right,
                                                              const FaceContext& face) const {
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
concept HLLCPhysicalFlux =
    PhysicalFlux<Physical> &&
    requires(const Physical& physical, const typename Physical::Trace& trace,
             const typename Physical::State& state, const FaceContext& face, Real& lo, Real& hi,
             Real scalar) {
      physical.signed_wave_speeds(trace, face, lo, hi);
      { physical.pressure(state) } -> std::convertible_to<Real>;
      {
        physical.contact_speed(state, state, scalar, scalar, scalar, scalar, face)
      } -> std::convertible_to<Real>;
      {
        physical.star_state(state, scalar, scalar, scalar, face)
      } -> std::same_as<typename Physical::State>;
    };

/// Contact-resolving HLLC policy.  Physical structure is supplied by the narrow PhysicalFlux.
struct HLLCFlux {
  template <PhysicalFlux Physical>
  POPS_HD FluxEvaluation<typename Physical::State> operator()(const Physical& physical,
                                                              const typename Physical::Trace& left,
                                                              const typename Physical::Trace& right,
                                                              const FaceContext& face) const {
    if (face.orientation == FaceOrientation::kNegative)
      return detail::canonical_evaluation(*this, physical, left, right, face);
    if constexpr (HLLCPhysicalFlux<Physical>) {
      Real lower, upper;
      hll_speeds(physical, left, right, face, lower, upper);
      if (!detail::valid_hll_speed_interval(lower, upper))
        return FluxEvaluation<typename Physical::State>::reject(
            RiemannFailureCause::kHllcInvalidWaveInterval);
      StabilityBound bound{};
      if (!detail::max_normal_stability_bound(physical.stability(left, face),
                                              physical.stability(right, face), bound))
        return FluxEvaluation<typename Physical::State>::reject(
            RiemannFailureCause::kHllcInvalidStability);
      const auto left_density = physical.evaluate(left, face);
      const auto right_density = physical.evaluate(right, face);
      if (lower >= Real(0)) {
        if (!detail::finite_state(left_density.value))
          return FluxEvaluation<typename Physical::State>::reject(
              RiemannFailureCause::kHllcNonFinitePhysicalFlux);
        return FluxEvaluation<typename Physical::State>::ok(left_density.value, bound);
      }
      if (upper <= Real(0)) {
        if (!detail::finite_state(right_density.value))
          return FluxEvaluation<typename Physical::State>::reject(
              RiemannFailureCause::kHllcNonFinitePhysicalFlux);
        return FluxEvaluation<typename Physical::State>::ok(right_density.value, bound);
      }
      if (!detail::finite_state(left_density.value) || !detail::finite_state(right_density.value))
        return FluxEvaluation<typename Physical::State>::reject(
            RiemannFailureCause::kHllcNonFinitePhysicalFlux);
      const Real pressure_left = physical.pressure(left.state);
      const Real pressure_right = physical.pressure(right.state);
      if (!Kokkos::isfinite(pressure_left) || !Kokkos::isfinite(pressure_right))
        return FluxEvaluation<typename Physical::State>::reject(
            RiemannFailureCause::kHllcNonFinitePressure);
      const Real contact = physical.contact_speed(left.state, right.state, pressure_left,
                                                  pressure_right, lower, upper, face);
      if (!Kokkos::isfinite(contact))
        return FluxEvaluation<typename Physical::State>::reject(
            RiemannFailureCause::kHllcNonFiniteContact);
      typename Physical::State density{};
      if (contact >= Real(0)) {
        const auto star = physical.star_state(left.state, pressure_left, lower, contact, face);
        if (!detail::finite_state(star))
          return FluxEvaluation<typename Physical::State>::reject(
              RiemannFailureCause::kHllcNonFiniteStarState);
        for (int component = 0; component < Physical::n_vars; ++component)
          density[component] =
              left_density.value[component] + lower * (star[component] - left.state[component]);
      } else {
        const auto star = physical.star_state(right.state, pressure_right, upper, contact, face);
        if (!detail::finite_state(star))
          return FluxEvaluation<typename Physical::State>::reject(
              RiemannFailureCause::kHllcNonFiniteStarState);
        for (int component = 0; component < Physical::n_vars; ++component)
          density[component] =
              right_density.value[component] + upper * (star[component] - right.state[component]);
      }
      if (!detail::finite_state(density))
        return FluxEvaluation<typename Physical::State>::reject(
            RiemannFailureCause::kHllcNonFiniteFlux);
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
                            {
                              physical.roe_dissipation(left, right, face)
                            } -> std::same_as<typename Physical::State>;
                          };

/// Roe-like policy.  Eigenstructure and entropy policy belong to the physical provider.
struct RoeFlux {
  template <PhysicalFlux Physical>
  POPS_HD FluxEvaluation<typename Physical::State> operator()(const Physical& physical,
                                                              const typename Physical::Trace& left,
                                                              const typename Physical::Trace& right,
                                                              const FaceContext& face) const {
    if (face.orientation == FaceOrientation::kNegative)
      return detail::canonical_evaluation(*this, physical, left, right, face);
    if constexpr (RoePhysicalFlux<Physical>) {
      StabilityBound bound{};
      if (!detail::max_normal_stability_bound(physical.stability(left, face),
                                              physical.stability(right, face), bound))
        return FluxEvaluation<typename Physical::State>::reject(
            RiemannFailureCause::kRoeInvalidStability);
      const auto left_density = physical.evaluate(left, face);
      const auto right_density = physical.evaluate(right, face);
      const auto dissipation = physical.roe_dissipation(left, right, face);
      if (!detail::finite_state(dissipation))
        return FluxEvaluation<typename Physical::State>::reject(
            RiemannFailureCause::kRoeNonFiniteDissipation);
      typename Physical::State density{};
      for (int component = 0; component < Physical::n_vars; ++component) {
        density[component] =
            Real(0.5) * (left_density.value[component] + right_density.value[component]) -
            Real(0.5) * dissipation[component];
      }
      if (!detail::finite_state(density))
        return FluxEvaluation<typename Physical::State>::reject(
            RiemannFailureCause::kRoeNonFiniteFlux);
      return FluxEvaluation<typename Physical::State>::ok(density, bound);
    } else {
      static_assert(detail::dependent_false<Physical>,
                    "RoeFlux requires the Roe-dissipation PhysicalFlux interface");
    }
  }
};

}  // namespace pops
