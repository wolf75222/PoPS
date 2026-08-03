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

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>

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
  static constexpr RiemannSolverId solver_id = RiemannSolverId::kRusanov;

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
  static constexpr RiemannSolverId solver_id = RiemannSolverId::kHll;

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
  static constexpr RiemannSolverId solver_id = RiemannSolverId::kHllc;

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
  static constexpr RiemannSolverId solver_id = RiemannSolverId::kRoe;

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

/// Explicit terminal action of a prepared Riemann recovery chain.  It is not a numerical flux and
/// is never evaluated; reaching it preserves the last candidate's typed rejection and prevents
/// publication through the ordinary FluxEvaluation failure path.
struct RejectRiemannRecovery {
  static constexpr RiemannSolverId solver_id = RiemannSolverId::kReject;
};

namespace detail {

template <class Candidate>
consteval RiemannSolverId declared_riemann_solver_id() {
  static_assert(
      requires { Candidate::solver_id; },
      "a prepared Riemann recovery candidate must expose a typed solver_id");
  return static_cast<RiemannSolverId>(Candidate::solver_id);
}

template <class... Candidates>
consteval bool valid_riemann_recovery_chain() {
  constexpr std::array ids{declared_riemann_solver_id<Candidates>()...};
  if constexpr (sizeof...(Candidates) < 2 || sizeof...(Candidates) > 255)
    return false;
  if (ids.back() != RiemannSolverId::kReject)
    return false;
  for (std::size_t index = 0; index + 1 < ids.size(); ++index) {
    if (ids[index] == RiemannSolverId::kUnspecified || ids[index] == RiemannSolverId::kReject)
      return false;
    for (std::size_t previous = 0; previous < index; ++previous)
      if (ids[previous] == ids[index])
        return false;
  }
  return true;
}

template <class Next, class... Rest, PhysicalFlux Physical>
POPS_HD FluxEvaluation<typename Physical::State> continue_riemann_recovery(
    const Physical& physical, const typename Physical::Trace& left,
    const typename Physical::Trace& right, const FaceContext& face,
    const FluxEvaluation<typename Physical::State>& current, RiemannSolverId requested,
    RiemannSolverId last_attempted, std::uint32_t first_recovery_reason, std::uint8_t attempts) {
  if (current.succeeded())
    return current.with_recovery_provenance(requested, last_attempted, last_attempted,
                                            first_recovery_reason, attempts);

  // Retry and fatal outcomes are scheduler decisions, not solver degeneracies.  A prepared chain
  // may recover only a typed candidate rejection; it must never silently downgrade stronger
  // failure semantics.
  if (current.status != EvaluationStatus::kReject)
    return current.with_recovery_provenance(requested, RiemannSolverId::kReject, last_attempted,
                                            first_recovery_reason, attempts);

  if constexpr (std::is_same_v<Next, RejectRiemannRecovery>) {
    static_assert(sizeof...(Rest) == 0,
                  "RejectRiemannRecovery must be the final prepared policy action");
    return current.with_recovery_provenance(requested, RiemannSolverId::kReject, last_attempted,
                                            first_recovery_reason, attempts);
  } else {
    static_assert(NumericalFlux<Next, Physical>,
                  "a prepared Riemann recovery candidate does not satisfy NumericalFlux for the "
                  "selected physical provider");
    constexpr RiemannSolverId next_id = declared_riemann_solver_id<Next>();
    const auto next = Next{}(physical, left, right, face);
    const std::uint32_t recovery_reason =
        first_recovery_reason != 0 ? first_recovery_reason : next.reason_code;
    return continue_riemann_recovery<Rest...>(physical, left, right, face, next, requested, next_id,
                                              recovery_reason,
                                              static_cast<std::uint8_t>(attempts + 1));
  }
}

}  // namespace detail

/// Fixed, allocation-free and device-copyable Riemann recovery chain.
///
/// Candidate types and their order are resolved before a spatial kernel is instantiated.  The hot
/// loop contains no string dispatch, virtual call, callback, exception, heap allocation or hidden
/// substitution.  Only `kReject` advances to the next declared candidate; retry/fatal outcomes
/// remain terminal.  The chain must end explicitly in RejectRiemannRecovery.
template <class... Candidates>
struct PreparedRiemannRecoveryPolicy;

template <class First, class... Rest>
struct PreparedRiemannRecoveryPolicy<First, Rest...> {
  static_assert(detail::valid_riemann_recovery_chain<First, Rest...>(),
                "a prepared Riemann recovery chain must contain unique typed candidates and end "
                "in RejectRiemannRecovery");
  static_assert((std::is_trivially_copyable_v<First> && ... && std::is_trivially_copyable_v<Rest>),
                "prepared Riemann recovery candidates must be device-copyable values");

  static constexpr RiemannSolverId solver_id = First::solver_id;
  static constexpr std::size_t candidate_count = sizeof...(Rest);
  inline static constexpr std::array ordered_solver_ids{
      detail::declared_riemann_solver_id<First>(), detail::declared_riemann_solver_id<Rest>()...};

  template <PhysicalFlux Physical>
  POPS_HD FluxEvaluation<typename Physical::State> operator()(const Physical& physical,
                                                              const typename Physical::Trace& left,
                                                              const typename Physical::Trace& right,
                                                              const FaceContext& face) const {
    static_assert(!std::is_same_v<First, RejectRiemannRecovery>,
                  "a prepared Riemann recovery chain requires a numerical first candidate");
    static_assert(NumericalFlux<First, Physical>,
                  "the requested Riemann candidate does not satisfy NumericalFlux for the "
                  "selected physical provider");
    constexpr RiemannSolverId requested = detail::declared_riemann_solver_id<First>();
    const auto first = First{}(physical, left, right, face);
    const std::uint32_t first_reason = first.succeeded() ? 0 : first.reason_code;
    return detail::continue_riemann_recovery<Rest...>(physical, left, right, face, first, requested,
                                                      requested, first_reason, 1);
  }
};

template <class... Candidates>
POPS_HD constexpr PreparedRiemannRecoveryPolicy<Candidates...> prepare_riemann_recovery_policy() {
  return {};
}

}  // namespace pops
