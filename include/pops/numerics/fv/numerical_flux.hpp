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
  POPS_HD FluxEvaluation<typename Physical::State> operator()(const Physical& physical,
                                                              const typename Physical::Trace& left,
                                                              const typename Physical::Trace& right,
                                                              const FaceContext& face) const {
    if (face.orientation == FaceOrientation::kNegative)
      return detail::canonical_evaluation(*this, physical, left, right, face);
    const auto left_density = physical.evaluate(left, face);
    const auto right_density = physical.evaluate(right, face);
    const auto bound = detail::max_bound<typename Physical::State>(physical.stability(left, face),
                                                                   physical.stability(right, face));
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
  if (!Kokkos::isfinite(lower) || !Kokkos::isfinite(upper) || lower > upper)
    return FluxEvaluation<typename Physical::State>::reject(0x484c4c01u);
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
          density[component] =
              left_density.value[component] + lower * (star[component] - left.state[component]);
      } else {
        const auto star = physical.star_state(right.state, pressure_right, upper, contact, face);
        for (int component = 0; component < Physical::n_vars; ++component)
          density[component] =
              right_density.value[component] + upper * (star[component] - right.state[component]);
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
      const auto left_density = physical.evaluate(left, face);
      const auto right_density = physical.evaluate(right, face);
      const auto dissipation = physical.roe_dissipation(left, right, face);
      const auto bound = detail::max_bound<typename Physical::State>(
          physical.stability(left, face), physical.stability(right, face));
      typename Physical::State density{};
      for (int component = 0; component < Physical::n_vars; ++component) {
        density[component] =
            Real(0.5) * (left_density.value[component] + right_density.value[component]) -
            Real(0.5) * dissipation[component];
      }
      return FluxEvaluation<typename Physical::State>::ok(density, bound);
    } else {
      static_assert(detail::dependent_false<Physical>,
                    "RoeFlux requires the Roe-dissipation PhysicalFlux interface");
    }
  }
};

inline constexpr Real kRoeEntropyFixFraction = Real(0.1);

/// Narrow contract for the explicit canonical Euler presets. Unlike the generic HLLC/Roe routes,
/// these policies own the fixed (rho, m_x, m_y, E) layout and therefore need no contact/star-state
/// or Roe-dissipation hook from the physical model.
template <class Physical>
concept EulerPhysicalFlux2D =
    PhysicalFlux<Physical> && (Physical::n_vars == 4) &&
    requires(const Physical& physical, const typename Physical::State& state) {
      { physical.pressure(state) } -> std::convertible_to<Real>;
    };

template <class Physical>
concept EulerHLLCPhysicalFlux2D =
    EulerPhysicalFlux2D<Physical> &&
    requires(const Physical& physical, const typename Physical::Trace& trace,
             const FaceContext& face, Real& lower,
             Real& upper) { physical.signed_wave_speeds(trace, face, lower, upper); };

/// Explicit canonical 2D Euler HLLC route. The physical model provides flux, pressure and signed
/// outer waves; this policy supplies Toro's contact speed and star states from the fixed Euler
/// layout. It is deliberately independent of HasHLLCStructure, which belongs to generic HLLC.
struct EulerHLLCFlux2D {
  template <EulerHLLCPhysicalFlux2D Physical>
  POPS_HD FluxEvaluation<typename Physical::State> operator()(const Physical& physical,
                                                              const typename Physical::Trace& left,
                                                              const typename Physical::Trace& right,
                                                              const FaceContext& face) const {
    if (face.orientation == FaceOrientation::kNegative)
      return detail::canonical_evaluation(*this, physical, left, right, face);

    const auto& UL = left.state;
    const auto& UR = right.state;
    const int normal = face.axis == 0 ? 1 : 2;
    const int tangent = face.axis == 0 ? 2 : 1;
    const Real rho_left = UL[0], rho_right = UR[0];
    const Real velocity_left = UL[normal] / rho_left;
    const Real velocity_right = UR[normal] / rho_right;
    const Real pressure_left = physical.pressure(UL);
    const Real pressure_right = physical.pressure(UR);
    Real speed_left, speed_right;
    hll_speeds(physical, left, right, face, speed_left, speed_right);
    const auto flux_left = physical.evaluate(left, face);
    const auto flux_right = physical.evaluate(right, face);
    const auto bound = detail::max_bound<typename Physical::State>(physical.stability(left, face),
                                                                   physical.stability(right, face));
    if (speed_left >= Real(0))
      return FluxEvaluation<typename Physical::State>::ok(flux_left.value, bound);
    if (speed_right <= Real(0))
      return FluxEvaluation<typename Physical::State>::ok(flux_right.value, bound);

    const Real contact =
        (pressure_right - pressure_left + rho_left * velocity_left * (speed_left - velocity_left) -
         rho_right * velocity_right * (speed_right - velocity_right)) /
        (rho_left * (speed_left - velocity_left) - rho_right * (speed_right - velocity_right));
    typename Physical::State density{};
    if (contact >= Real(0)) {
      const Real factor = rho_left * (speed_left - velocity_left) / (speed_left - contact);
      typename Physical::State star{};
      star[0] = factor;
      star[normal] = factor * contact;
      star[tangent] = factor * (UL[tangent] / rho_left);
      star[3] =
          factor * (UL[3] / rho_left +
                    (contact - velocity_left) *
                        (contact + pressure_left / (rho_left * (speed_left - velocity_left))));
      for (int component = 0; component < 4; ++component)
        density[component] =
            flux_left.value[component] + speed_left * (star[component] - UL[component]);
    } else {
      const Real factor = rho_right * (speed_right - velocity_right) / (speed_right - contact);
      typename Physical::State star{};
      star[0] = factor;
      star[normal] = factor * contact;
      star[tangent] = factor * (UR[tangent] / rho_right);
      star[3] =
          factor * (UR[3] / rho_right +
                    (contact - velocity_right) *
                        (contact + pressure_right / (rho_right * (speed_right - velocity_right))));
      for (int component = 0; component < 4; ++component)
        density[component] =
            flux_right.value[component] + speed_right * (star[component] - UR[component]);
    }
    return FluxEvaluation<typename Physical::State>::ok(density, bound);
  }
};

/// Explicit canonical ideal-gas 2D Euler Roe route. The complete Roe average, eigenwave
/// decomposition and Harten entropy fix live here; generic Roe remains hook-driven.
struct EulerRoeFlux2D {
  template <EulerPhysicalFlux2D Physical>
  POPS_HD FluxEvaluation<typename Physical::State> operator()(const Physical& physical,
                                                              const typename Physical::Trace& left,
                                                              const typename Physical::Trace& right,
                                                              const FaceContext& face) const {
    if (face.orientation == FaceOrientation::kNegative)
      return detail::canonical_evaluation(*this, physical, left, right, face);

    const auto& UL = left.state;
    const auto& UR = right.state;
    const int normal_component = face.axis == 0 ? 1 : 2;
    const int tangent_component = face.axis == 0 ? 2 : 1;
    const Real rho_left = UL[0], rho_right = UR[0];
    const Real normal_left = UL[normal_component] / rho_left;
    const Real normal_right = UR[normal_component] / rho_right;
    const Real tangent_left = UL[tangent_component] / rho_left;
    const Real tangent_right = UR[tangent_component] / rho_right;
    const Real pressure_left = physical.pressure(UL);
    const Real pressure_right = physical.pressure(UR);
    const Real enthalpy_left = (UL[3] + pressure_left) / rho_left;
    const Real enthalpy_right = (UR[3] + pressure_right) / rho_right;

    const Real sqrt_left = std::sqrt(rho_left);
    const Real sqrt_right = std::sqrt(rho_right);
    const Real denominator = sqrt_left + sqrt_right;
    const Real normal = (sqrt_left * normal_left + sqrt_right * normal_right) / denominator;
    const Real tangent_velocity =
        (sqrt_left * tangent_left + sqrt_right * tangent_right) / denominator;
    const Real enthalpy = (sqrt_left * enthalpy_left + sqrt_right * enthalpy_right) / denominator;
    const Real roe_density = sqrt_left * sqrt_right;
    const Real velocity_squared = normal * normal + tangent_velocity * tangent_velocity;
    const Real gamma_minus_one =
        pressure_left /
        (UL[3] - Real(0.5) * rho_left * (normal_left * normal_left + tangent_left * tangent_left));
    const Real sound_squared = gamma_minus_one * (enthalpy - Real(0.5) * velocity_squared);
    const Real sound = std::sqrt(sound_squared);

    const Real density_jump = rho_right - rho_left;
    const Real pressure_jump = pressure_right - pressure_left;
    const Real normal_jump = normal_right - normal_left;
    const Real tangent_jump = tangent_right - tangent_left;
    const Real amplitude_left =
        (pressure_jump - roe_density * sound * normal_jump) / (Real(2) * sound_squared);
    const Real amplitude_entropy = density_jump - pressure_jump / sound_squared;
    const Real amplitude_shear = roe_density * tangent_jump;
    const Real amplitude_right =
        (pressure_jump + roe_density * sound * normal_jump) / (Real(2) * sound_squared);

    const Real epsilon = kRoeEntropyFixFraction * sound;
    const Real eigenvalue_left = normal - sound;
    const Real eigenvalue_right = normal + sound;
    const Real absolute_left = eigenvalue_left < Real(0) ? -eigenvalue_left : eigenvalue_left;
    const Real absolute_right = eigenvalue_right < Real(0) ? -eigenvalue_right : eigenvalue_right;
    const Real fixed_left =
        absolute_left < epsilon
            ? Real(0.5) * (eigenvalue_left * eigenvalue_left / epsilon + epsilon)
            : absolute_left;
    const Real fixed_entropy = normal < Real(0) ? -normal : normal;
    const Real fixed_right =
        absolute_right < epsilon
            ? Real(0.5) * (eigenvalue_right * eigenvalue_right / epsilon + epsilon)
            : absolute_right;

    const Real dissipation_density = fixed_left * amplitude_left +
                                     fixed_entropy * amplitude_entropy +
                                     fixed_right * amplitude_right;
    const Real dissipation_normal = fixed_left * amplitude_left * (normal - sound) +
                                    fixed_entropy * amplitude_entropy * normal +
                                    fixed_right * amplitude_right * (normal + sound);
    const Real dissipation_tangent =
        fixed_left * amplitude_left * tangent_velocity +
        fixed_entropy * (amplitude_entropy * tangent_velocity + amplitude_shear) +
        fixed_right * amplitude_right * tangent_velocity;
    const Real dissipation_energy =
        fixed_left * amplitude_left * (enthalpy - normal * sound) +
        fixed_entropy * (amplitude_entropy * Real(0.5) * velocity_squared +
                         amplitude_shear * tangent_velocity) +
        fixed_right * amplitude_right * (enthalpy + normal * sound);

    const auto flux_left = physical.evaluate(left, face);
    const auto flux_right = physical.evaluate(right, face);
    const auto bound = detail::max_bound<typename Physical::State>(physical.stability(left, face),
                                                                   physical.stability(right, face));
    typename Physical::State density{};
    density[0] =
        Real(0.5) * (flux_left.value[0] + flux_right.value[0]) - Real(0.5) * dissipation_density;
    density[normal_component] =
        Real(0.5) * (flux_left.value[normal_component] + flux_right.value[normal_component]) -
        Real(0.5) * dissipation_normal;
    density[tangent_component] =
        Real(0.5) * (flux_left.value[tangent_component] + flux_right.value[tangent_component]) -
        Real(0.5) * dissipation_tangent;
    density[3] =
        Real(0.5) * (flux_left.value[3] + flux_right.value[3]) - Real(0.5) * dissipation_energy;
    return FluxEvaluation<typename Physical::State>::ok(density, bound);
  }
};

}  // namespace pops
