#pragma once

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/model/physical_model.hpp>
#include <pops/core/state/state.hpp>
#include <pops/core/state/variables.hpp>
#include <pops/physics/composition/exact_brick_contract.hpp>

#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

/// @file
/// @brief Exact-ranked composition of hyperbolic, source, and elliptic physics bricks.

namespace pops {

namespace composite_detail {

template <class Hyperbolic, class = void>
struct ConservationLawAliases {};

template <class Hyperbolic>
struct ConservationLawAliases<
    Hyperbolic, std::void_t<typename Hyperbolic::Primitive, typename Hyperbolic::Schema>> {
  using Primitive = typename Hyperbolic::Primitive;
  using Schema = typename Hyperbolic::Schema;
};

template <class Hyperbolic, class = void>
struct PrimitiveType {
  using type = typename Hyperbolic::Primitive;
};

template <class Hyperbolic>
struct PrimitiveType<Hyperbolic, std::void_t<typename Hyperbolic::Prim>> {
  using type = typename Hyperbolic::Prim;
};

template <class Hyperbolic>
consteval int hyperbolic_dimension() {
  if constexpr (requires { Hyperbolic::dimension; })
    return static_cast<int>(Hyperbolic::dimension);
  return kNativeDimension;
}

template <class Brick, int Dim>
consteval bool dimension_matches() {
  if constexpr (requires { Brick::dimension; })
    return static_cast<int>(Brick::dimension) == Dim;
  return true;
}

template <int Axis, class Hyperbolic, class Providers>
concept FluxAt =
    requires(const Hyperbolic h, const typename Hyperbolic::State state,
             const Providers& providers) {
      { h.template flux<Axis>(state, providers) } -> std::same_as<typename Hyperbolic::State>;
    } ||
    requires(const Hyperbolic h, const typename Hyperbolic::State state,
             const Providers& providers) {
      { h.flux(state, providers, Axis) } -> std::same_as<typename Hyperbolic::State>;
    } ||
    requires(const Hyperbolic h, const typename Hyperbolic::State state) {
      { h.template flux<Axis>(state) } -> std::same_as<typename Hyperbolic::State>;
    };

template <int Axis, class Hyperbolic, class Providers>
concept MaximumWaveSpeedAt =
    requires(const Hyperbolic h, const typename Hyperbolic::State state,
             const Providers& providers) {
      { h.template max_wave_speed<Axis>(state, providers) } -> std::convertible_to<Real>;
    } ||
    requires(const Hyperbolic h, const typename Hyperbolic::State state,
             const Providers& providers) {
      { h.max_wave_speed(state, providers, Axis) } -> std::convertible_to<Real>;
    } ||
    requires(const Hyperbolic h, const typename Hyperbolic::State state) {
      { h.template max_wave_speed<Axis>(state) } -> std::convertible_to<Real>;
    };

template <int Axis, class Hyperbolic, class Providers>
concept WaveSpeedsAt =
    requires(const Hyperbolic h, const typename Hyperbolic::State state, const Providers& providers,
             Real& lower,
             Real& upper) { h.template wave_speeds<Axis>(state, providers, lower, upper); } ||
    requires(const Hyperbolic h, const typename Hyperbolic::State state, const Providers& providers,
             Real& lower, Real& upper) { h.wave_speeds(state, providers, Axis, lower, upper); } ||
    requires(const Hyperbolic h, const typename Hyperbolic::State state, Real& lower, Real& upper) {
      h.template wave_speeds<Axis>(state, lower, upper);
    };

template <int Axis, class Hyperbolic>
concept ContactSpeedAt =
    requires(const Hyperbolic h, const typename Hyperbolic::State left,
             const typename Hyperbolic::State right, Real scalar) {
      {
        h.template contact_speed<Axis>(left, right, scalar, scalar, scalar, scalar)
      } -> std::convertible_to<Real>;
    } ||
    requires(const Hyperbolic h, const typename Hyperbolic::State left,
             const typename Hyperbolic::State right, Real scalar) {
      {
        h.contact_speed(left, right, scalar, scalar, scalar, scalar, Axis)
      } -> std::convertible_to<Real>;
    };

template <int Axis, class Hyperbolic>
concept StarStateAt =
    requires(const Hyperbolic h, const typename Hyperbolic::State state, Real scalar) {
      {
        h.template hllc_star_state<Axis>(state, scalar, scalar, scalar)
      } -> std::same_as<typename Hyperbolic::State>;
    } || requires(const Hyperbolic h, const typename Hyperbolic::State state, Real scalar) {
      {
        h.hllc_star_state(state, scalar, scalar, scalar, Axis)
      } -> std::same_as<typename Hyperbolic::State>;
    } || requires(const Hyperbolic h, const typename Hyperbolic::State state, Real scalar) {
      {
        h.template star_state<Axis>(state, scalar, scalar, scalar)
      } -> std::same_as<typename Hyperbolic::State>;
    };

template <int Axis, class Hyperbolic, class LeftProviders, class RightProviders>
concept RoeDissipationAt =
    requires(const Hyperbolic h, const typename Hyperbolic::State left,
             const LeftProviders& left_providers, const typename Hyperbolic::State right,
             const RightProviders& right_providers) {
      {
        h.template roe_dissipation<Axis>(left, left_providers, right, right_providers)
      } -> std::same_as<typename Hyperbolic::State>;
    } ||
    requires(const Hyperbolic h, const typename Hyperbolic::State left,
             const LeftProviders& left_providers, const typename Hyperbolic::State right,
             const RightProviders& right_providers) {
      {
        h.roe_dissipation(left, left_providers, right, right_providers, Axis)
      } -> std::same_as<typename Hyperbolic::State>;
    } ||
    requires(const Hyperbolic h, const typename Hyperbolic::State left,
             const typename Hyperbolic::State right) {
      { h.template roe_dissipation<Axis>(left, right) } -> std::same_as<typename Hyperbolic::State>;
    };

template <int Axis, class Hyperbolic, class Providers>
concept StabilitySpeedAt =
    requires(const Hyperbolic h, const typename Hyperbolic::State state,
             const Providers& providers) {
      { h.template stability_speed<Axis>(state, providers) } -> std::convertible_to<Real>;
    } ||
    requires(const Hyperbolic h, const typename Hyperbolic::State state,
             const Providers& providers) {
      { h.stability_speed(state, providers, Axis) } -> std::convertible_to<Real>;
    };

template <int Axis, int Dim, class Hyperbolic, class Providers>
consteval bool hyperbolic_axes_contract() {
  if constexpr (!FluxAt<Axis, Hyperbolic, Providers> ||
                !MaximumWaveSpeedAt<Axis, Hyperbolic, Providers>)
    return false;
  else if constexpr (Axis + 1 < Dim)
    return hyperbolic_axes_contract<Axis + 1, Dim, Hyperbolic, Providers>();
  return true;
}

template <class Hyperbolic, int Dim>
consteval bool hyperbolic_contract() {
  using State = typename Hyperbolic::State;
  using Prim = typename PrimitiveType<Hyperbolic>::type;
  using Providers = ProviderValues<provider_count_for<Hyperbolic, Dim>()>;
  constexpr bool legacy_contract =
      requires(const Hyperbolic h, const State state, const Prim primitive) {
        { Hyperbolic::n_vars } -> std::convertible_to<int>;
        { h.to_primitive(state) } -> std::same_as<Prim>;
        { h.to_conservative(primitive) } -> std::same_as<State>;
        { Hyperbolic::conservative_vars() } -> std::same_as<VariableSet>;
        { Hyperbolic::primitive_vars() } -> std::same_as<VariableSet>;
      };
  constexpr bool conservation_law_contract =
      requires(const Hyperbolic h, const State state, const Prim primitive) {
        typename Hyperbolic::Primitive;
        typename Hyperbolic::Schema;
        h.recover(state);
        h.admissibility(state);
        h.make_conservative(primitive);
      };
  return hyperbolic_axes_contract<0, Dim, Hyperbolic, Providers>() &&
         (legacy_contract || conservation_law_contract);
}

template <class State>
POPS_HD State invalid_state() {
  State result{};
  for (int component = 0; component < State::size(); ++component)
    result[component] = std::numeric_limits<Real>::quiet_NaN();
  return result;
}

}  // namespace composite_detail

template <class Hyperbolic, class Source, class Elliptic>
struct CompositeModel : composite_detail::ConservationLawAliases<Hyperbolic> {
  static constexpr int dimension = composite_detail::hyperbolic_dimension<Hyperbolic>();
  static_assert(dimension >= 1 && dimension <= 3,
                "CompositeModel hyperbolic rank must be 1, 2, or 3");
  static_assert(composite_detail::hyperbolic_contract<Hyperbolic, dimension>(),
                "CompositeModel requires an exact-ranked hyperbolic brick");
  static_assert(composite_detail::dimension_matches<Source, dimension>(),
                "CompositeModel source rank differs from its hyperbolic rank");
  static_assert(composite_detail::dimension_matches<Elliptic, dimension>(),
                "CompositeModel elliptic rank differs from its hyperbolic rank");

  using State = typename Hyperbolic::State;
  using Prim = typename composite_detail::PrimitiveType<Hyperbolic>::type;
  static constexpr int n_vars = Hyperbolic::n_vars;
  static constexpr int characteristic_no_inflow_contract_version = [] {
    if constexpr (requires { Hyperbolic::characteristic_no_inflow_contract_version; })
      return static_cast<int>(Hyperbolic::characteristic_no_inflow_contract_version);
    return 0;
  }();
  static constexpr int characteristic_no_inflow_dimension = [] {
    if constexpr (requires { Hyperbolic::characteristic_no_inflow_dimension; })
      return static_cast<int>(Hyperbolic::characteristic_no_inflow_dimension);
    return 0;
  }();
  static constexpr int characteristic_no_inflow_components = [] {
    if constexpr (requires { Hyperbolic::characteristic_no_inflow_components; })
      return static_cast<int>(Hyperbolic::characteristic_no_inflow_components);
    return 0;
  }();
  static constexpr bool characteristic_no_inflow_conservative = [] {
    if constexpr (requires { Hyperbolic::characteristic_no_inflow_conservative; })
      return static_cast<bool>(Hyperbolic::characteristic_no_inflow_conservative);
    return false;
  }();
  static constexpr int n_providers = [] {
    int width = provider_count_for<Hyperbolic, dimension>();
    if (provider_count_for<Source, dimension>() > width)
      width = provider_count_for<Source, dimension>();
    if (provider_count_for<Elliptic, dimension>() > width)
      width = provider_count_for<Elliptic, dimension>();
    return width;
  }();

  Hyperbolic hyp{};
  Source src{};
  Elliptic ell{};

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept
    requires(physics_contract_detail::ExactPhysicsBrickContract<Hyperbolic> &&
             physics_contract_detail::ExactPhysicsBrickContract<Source> &&
             physics_contract_detail::ExactPhysicsBrickContract<Elliptic>)
  {
    return {"pops.physics.composite-model", 1};
  }

  void serialize_exact_parameters(ExactContractBuilder& contract) const
    requires(physics_contract_detail::ExactPhysicsBrickContract<Hyperbolic> &&
             physics_contract_detail::ExactPhysicsBrickContract<Source> &&
             physics_contract_detail::ExactPhysicsBrickContract<Elliptic>)
  {
    contract.text("pops.physics.composite-model-parameters")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{dimension})
        .scalar(std::int32_t{n_vars})
        .scalar(std::int32_t{n_providers});
    physics_contract_detail::append_exact_brick(contract, "hyperbolic", hyp);
    physics_contract_detail::append_exact_brick(contract, "source", src);
    physics_contract_detail::append_exact_brick(contract, "elliptic", ell);
  }

  template <int Axis, class Providers>
    requires composite_detail::FluxAt<Axis, Hyperbolic, Providers>
  POPS_HD State flux(const State& state, const Providers& providers) const {
    static_assert(Axis >= 0 && Axis < dimension, "CompositeModel flux axis is outside its rank");
    if constexpr (requires { hyp.template flux<Axis>(state, providers); })
      return hyp.template flux<Axis>(state, providers);
    else if constexpr (requires { hyp.flux(state, providers, Axis); })
      return hyp.flux(state, providers, Axis);
    else
      return hyp.template flux<Axis>(state);
  }

  template <int Axis = 0, class Providers>
  POPS_HD State flux_at_runtime_axis(const State& state, const Providers& providers,
                                     int axis) const {
    if (axis == Axis)
      return flux<Axis>(state, providers);
    if constexpr (Axis + 1 < dimension)
      return flux_at_runtime_axis<Axis + 1>(state, providers, axis);
    return composite_detail::invalid_state<State>();
  }

  template <class Providers>
  POPS_HD State flux(const State& state, const Providers& providers, int axis) const {
    return flux_at_runtime_axis(state, providers, axis);
  }

  template <int Axis, class Providers>
    requires composite_detail::MaximumWaveSpeedAt<Axis, Hyperbolic, Providers>
  POPS_HD Real max_wave_speed(const State& state, const Providers& providers) const {
    static_assert(Axis >= 0 && Axis < dimension,
                  "CompositeModel wave-speed axis is outside its rank");
    if constexpr (requires { hyp.template max_wave_speed<Axis>(state, providers); })
      return hyp.template max_wave_speed<Axis>(state, providers);
    else if constexpr (requires { hyp.max_wave_speed(state, providers, Axis); })
      return hyp.max_wave_speed(state, providers, Axis);
    else
      return hyp.template max_wave_speed<Axis>(state);
  }

  template <int Axis = 0, class Providers>
  POPS_HD Real max_wave_speed_at_runtime_axis(const State& state, const Providers& providers,
                                              int axis) const {
    if (axis == Axis)
      return max_wave_speed<Axis>(state, providers);
    if constexpr (Axis + 1 < dimension)
      return max_wave_speed_at_runtime_axis<Axis + 1>(state, providers, axis);
    return std::numeric_limits<Real>::quiet_NaN();
  }

  template <class Providers>
  POPS_HD Real max_wave_speed(const State& state, const Providers& providers, int axis) const {
    return max_wave_speed_at_runtime_axis(state, providers, axis);
  }

  template <class Providers>
  POPS_HD State source(const State& state, const Providers& providers) const {
    return src.apply(state, providers);
  }
  POPS_HD Real elliptic_rhs(const State& state) const { return ell.rhs(state); }
  POPS_HD Prim to_primitive(const State& state) const
    requires requires(const Hyperbolic h, const State value) { h.to_primitive(value); }
  {
    return hyp.to_primitive(state);
  }
  POPS_HD State to_conservative(const Prim& primitive) const
    requires requires(const Hyperbolic h, const Prim value) { h.to_conservative(value); }
  {
    return hyp.to_conservative(primitive);
  }

  POPS_HD auto recover(const State& state) const
    requires requires(const Hyperbolic h, const State value) { h.recover(value); }
  {
    return hyp.recover(state);
  }

  POPS_HD auto admissibility(const State& state) const
    requires requires(const Hyperbolic h, const State value) { h.admissibility(value); }
  {
    return hyp.admissibility(state);
  }

  template <class Primitive>
  POPS_HD auto make_conservative(const Primitive& primitive) const
    requires requires(const Hyperbolic h, const Primitive value) { h.make_conservative(value); }
  {
    return hyp.make_conservative(primitive);
  }
  static VariableSet conservative_vars()
    requires requires { Hyperbolic::conservative_vars(); }
  {
    return Hyperbolic::conservative_vars();
  }
  static VariableSet primitive_vars()
    requires requires { Hyperbolic::primitive_vars(); }
  {
    return Hyperbolic::primitive_vars();
  }

  POPS_HD bool recovery_admissible(const Prim& primitive, int* failing_component) const
    requires requires(const Hyperbolic h, const Prim value, int* component) {
      { h.recovery_admissible(value, component) } -> std::same_as<bool>;
    }
  {
    return hyp.recovery_admissible(primitive, failing_component);
  }

  POPS_HD Real pressure(const State& state) const
    requires requires(const Hyperbolic h, const State value) { h.pressure(value); }
  {
    return hyp.pressure(state);
  }

  template <int Axis, class Providers>
    requires composite_detail::WaveSpeedsAt<Axis, Hyperbolic, Providers>
  POPS_HD void wave_speeds(const State& state, const Providers& providers, Real& lower,
                           Real& upper) const {
    static_assert(Axis >= 0 && Axis < dimension,
                  "CompositeModel signed-wave axis is outside its rank");
    if constexpr (requires { hyp.template wave_speeds<Axis>(state, providers, lower, upper); })
      hyp.template wave_speeds<Axis>(state, providers, lower, upper);
    else if constexpr (requires { hyp.wave_speeds(state, providers, Axis, lower, upper); })
      hyp.wave_speeds(state, providers, Axis, lower, upper);
    else
      hyp.template wave_speeds<Axis>(state, lower, upper);
  }

  template <int Axis = 0, class Providers>
    requires composite_detail::WaveSpeedsAt<Axis, Hyperbolic, Providers>
  POPS_HD void wave_speeds_at_runtime_axis(const State& state, const Providers& providers, int axis,
                                           Real& lower, Real& upper) const {
    if (axis == Axis) {
      wave_speeds<Axis>(state, providers, lower, upper);
      return;
    }
    if constexpr (Axis + 1 < dimension) {
      wave_speeds_at_runtime_axis<Axis + 1>(state, providers, axis, lower, upper);
      return;
    }
    lower = upper = std::numeric_limits<Real>::quiet_NaN();
  }

  template <class Providers>
    requires composite_detail::WaveSpeedsAt<0, Hyperbolic, Providers>
  POPS_HD void wave_speeds(const State& state, const Providers& providers, int axis, Real& lower,
                           Real& upper) const {
    wave_speeds_at_runtime_axis(state, providers, axis, lower, upper);
  }

  template <int Axis>
    requires composite_detail::ContactSpeedAt<Axis, Hyperbolic>
  POPS_HD Real contact_speed(const State& left, const State& right, Real pressure_left,
                             Real pressure_right, Real speed_left, Real speed_right) const {
    static_assert(Axis >= 0 && Axis < dimension,
                  "CompositeModel contact-speed axis is outside its rank");
    if constexpr (requires {
                    hyp.template contact_speed<Axis>(left, right, pressure_left, pressure_right,
                                                     speed_left, speed_right);
                  })
      return hyp.template contact_speed<Axis>(left, right, pressure_left, pressure_right,
                                              speed_left, speed_right);
    else
      return hyp.contact_speed(left, right, pressure_left, pressure_right, speed_left, speed_right,
                               Axis);
  }

  template <int Axis = 0>
    requires composite_detail::ContactSpeedAt<Axis, Hyperbolic>
  POPS_HD Real contact_speed_at_runtime_axis(const State& left, const State& right,
                                             Real pressure_left, Real pressure_right,
                                             Real speed_left, Real speed_right, int axis) const {
    if (axis == Axis)
      return contact_speed<Axis>(left, right, pressure_left, pressure_right, speed_left,
                                 speed_right);
    if constexpr (Axis + 1 < dimension)
      return contact_speed_at_runtime_axis<Axis + 1>(left, right, pressure_left, pressure_right,
                                                     speed_left, speed_right, axis);
    return std::numeric_limits<Real>::quiet_NaN();
  }

  POPS_HD Real contact_speed(const State& left, const State& right, Real pressure_left,
                             Real pressure_right, Real speed_left, Real speed_right, int axis) const
    requires composite_detail::ContactSpeedAt<0, Hyperbolic>
  {
    return contact_speed_at_runtime_axis(left, right, pressure_left, pressure_right, speed_left,
                                         speed_right, axis);
  }

  template <int Axis>
    requires composite_detail::StarStateAt<Axis, Hyperbolic>
  POPS_HD State hllc_star_state(const State& state, Real pressure_value, Real speed,
                                Real contact) const {
    static_assert(Axis >= 0 && Axis < dimension,
                  "CompositeModel HLLC star-state axis is outside its rank");
    if constexpr (requires {
                    hyp.template hllc_star_state<Axis>(state, pressure_value, speed, contact);
                  })
      return hyp.template hllc_star_state<Axis>(state, pressure_value, speed, contact);
    else if constexpr (requires {
                         hyp.hllc_star_state(state, pressure_value, speed, contact, Axis);
                       })
      return hyp.hllc_star_state(state, pressure_value, speed, contact, Axis);
    else
      return hyp.template star_state<Axis>(state, pressure_value, speed, contact);
  }

  template <int Axis = 0>
    requires composite_detail::StarStateAt<Axis, Hyperbolic>
  POPS_HD State hllc_star_state_at_runtime_axis(const State& state, Real pressure_value, Real speed,
                                                Real contact, int axis) const {
    if (axis == Axis)
      return hllc_star_state<Axis>(state, pressure_value, speed, contact);
    if constexpr (Axis + 1 < dimension)
      return hllc_star_state_at_runtime_axis<Axis + 1>(state, pressure_value, speed, contact, axis);
    return composite_detail::invalid_state<State>();
  }

  POPS_HD State hllc_star_state(const State& state, Real pressure_value, Real speed, Real contact,
                                int axis) const
    requires composite_detail::StarStateAt<0, Hyperbolic>
  {
    return hllc_star_state_at_runtime_axis(state, pressure_value, speed, contact, axis);
  }

  template <int Axis, class LeftProviders, class RightProviders>
    requires composite_detail::RoeDissipationAt<Axis, Hyperbolic, LeftProviders, RightProviders>
  POPS_HD State roe_dissipation(const State& left, const LeftProviders& left_providers,
                                const State& right, const RightProviders& right_providers) const {
    static_assert(Axis >= 0 && Axis < dimension, "CompositeModel Roe axis is outside its rank");
    if constexpr (requires {
                    hyp.template roe_dissipation<Axis>(left, left_providers, right,
                                                       right_providers);
                  })
      return hyp.template roe_dissipation<Axis>(left, left_providers, right, right_providers);
    else if constexpr (requires {
                         hyp.roe_dissipation(left, left_providers, right, right_providers, Axis);
                       })
      return hyp.roe_dissipation(left, left_providers, right, right_providers, Axis);
    else
      return hyp.template roe_dissipation<Axis>(left, right);
  }

  template <int Axis = 0, class LeftProviders, class RightProviders>
    requires composite_detail::RoeDissipationAt<Axis, Hyperbolic, LeftProviders, RightProviders>
  POPS_HD State roe_dissipation_at_runtime_axis(const State& left,
                                                const LeftProviders& left_providers,
                                                const State& right,
                                                const RightProviders& right_providers,
                                                int axis) const {
    if (axis == Axis)
      return roe_dissipation<Axis>(left, left_providers, right, right_providers);
    if constexpr (Axis + 1 < dimension)
      return roe_dissipation_at_runtime_axis<Axis + 1>(left, left_providers, right, right_providers,
                                                       axis);
    return composite_detail::invalid_state<State>();
  }

  template <class LeftProviders, class RightProviders>
  POPS_HD State roe_dissipation(const State& left, const LeftProviders& left_providers,
                                const State& right, const RightProviders& right_providers,
                                int axis) const
    requires composite_detail::RoeDissipationAt<0, Hyperbolic, LeftProviders, RightProviders>
  {
    return roe_dissipation_at_runtime_axis(left, left_providers, right, right_providers, axis);
  }

  POPS_HD bool characteristic_no_inflow(const State& interior, const State& reference, int axis,
                                        int outward_sign, State& ghost) const
    requires requires(const Hyperbolic h, const State a, const State b, int d, int side,
                      State& out) { h.characteristic_no_inflow(a, b, d, side, out); }
  {
    return hyp.characteristic_no_inflow(interior, reference, axis, outward_sign, ghost);
  }

  POPS_HD bool characteristic_no_inflow(const State& interior, const State& reference,
                                        const Real* normal, State& ghost) const
    requires requires(const Hyperbolic h, const State a, const State b, const Real* n,
                      State& out) { h.characteristic_no_inflow(a, b, n, out); }
  {
    return hyp.characteristic_no_inflow(interior, reference, normal, ghost);
  }

  POPS_HD State polar_geom_source(const State& state, Real radius) const
    requires(dimension == 2) && requires(const Hyperbolic h, const State value, Real r) {
      h.polar_geom_source(value, r);
    }
  {
    return hyp.polar_geom_source(state, radius);
  }

  template <int Axis, class Providers>
    requires composite_detail::StabilitySpeedAt<Axis, Hyperbolic, Providers>
  POPS_HD Real stability_speed(const State& state, const Providers& providers) const {
    if constexpr (requires { hyp.template stability_speed<Axis>(state, providers); })
      return hyp.template stability_speed<Axis>(state, providers);
    else
      return hyp.stability_speed(state, providers, Axis);
  }

  template <int Axis = 0, class Providers>
    requires composite_detail::StabilitySpeedAt<Axis, Hyperbolic, Providers>
  POPS_HD Real stability_speed_at_runtime_axis(const State& state, const Providers& providers,
                                               int axis) const {
    if (axis == Axis)
      return stability_speed<Axis>(state, providers);
    if constexpr (Axis + 1 < dimension)
      return stability_speed_at_runtime_axis<Axis + 1>(state, providers, axis);
    return std::numeric_limits<Real>::quiet_NaN();
  }

  template <class Providers>
  POPS_HD Real stability_speed(const State& state, const Providers& providers, int axis) const
    requires composite_detail::StabilitySpeedAt<0, Hyperbolic, Providers>
  {
    return stability_speed_at_runtime_axis(state, providers, axis);
  }

  template <class Providers>
  POPS_HD Real stability_dt(const State& state, const Providers& providers) const
    requires requires(const Hyperbolic h, const State value, const Providers& values) {
      h.stability_dt(value, values);
    }
  {
    return hyp.stability_dt(state, providers);
  }

  template <class Providers>
  POPS_HD Real source_frequency(const State& state, const Providers& providers) const
    requires requires(const Source source_brick, const State value, const Providers& values) {
      source_brick.frequency(value, values);
    }
  {
    return src.frequency(state, providers);
  }

  template <class Providers>
  POPS_HD State project(const State& state, const Providers& providers) const
    requires requires(const Hyperbolic h, const State value, const Providers& values) {
      h.project(value, values);
    }
  {
    return hyp.project(state, providers);
  }

  template <class Providers>
  POPS_HD void source_jacobian(const State& state, const Providers& providers,
                               Real (&jacobian)[n_vars][n_vars]) const
    requires requires(const Source source_brick, const State value, const Providers& values,
                      Real (&matrix)[n_vars][n_vars]) {
      source_brick.jacobian(value, values, matrix);
    }
  {
    src.jacobian(state, providers, jacobian);
  }
};

}  // namespace pops
