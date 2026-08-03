/// @file
/// @brief Dimension-generic scalar-advection and ideal-gas Euler conservation laws.

#pragma once

#include <pops/mesh/index/real_vector.hpp>
#include <pops/numerics/spatial/nd/state_schema.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <cmath>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace pops::nd {

namespace conservation_law_detail {

POPS_HD inline bool finite(Real value) {
  return Kokkos::isfinite(value);
}

template <class State>
POPS_HD State invalid_state() {
  State result{};
  for (int component = 0; component < State::size(); ++component)
    result[component] = std::numeric_limits<Real>::quiet_NaN();
  return result;
}

template <class State>
POPS_HD bool finite_state(const State& state) {
  for (int component = 0; component < State::size(); ++component)
    if (!finite(state[component]))
      return false;
  return true;
}

}  // namespace conservation_law_detail

template <int Dim>
class ScalarAdvection {
 public:
  using Schema = ScalarStateSchema<Dim>;
  using State = typename Schema::Conservative;
  using Primitive = typename Schema::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Schema::nvars;

  ScalarAdvection() = default;

  static ScalarAdvection prepare(RealVector<Dim> velocity) {
    for (int axis = 0; axis < Dim; ++axis)
      if (!std::isfinite(static_cast<double>(velocity[axis])))
        throw std::invalid_argument("ND scalar-advection velocity must be finite on every axis");
    return ScalarAdvection(velocity);
  }

  POPS_HD const RealVector<Dim>& velocity() const { return velocity_; }

  POPS_HD StateConversion<Primitive> recover(const State& state) const {
    return {state, conservation_law_detail::finite_state(state)
                       ? StateConversionStatus::Success
                       : StateConversionStatus::NonFiniteState};
  }

  POPS_HD StateConversion<State> make_conservative(const Primitive& primitive) const {
    return {primitive, conservation_law_detail::finite_state(primitive)
                           ? StateConversionStatus::Success
                           : StateConversionStatus::NonFiniteState};
  }

  POPS_HD StateConversionStatus admissibility(const State& state) const {
    return recover(state).status;
  }

  template <int Axis>
  POPS_HD State flux(const State& state) const {
    static_assert(Axis >= 0 && Axis < Dim, "scalar-advection flux axis is outside the dimension");
    return State{velocity_[Axis] * state[Schema::scalar]};
  }

  template <int Axis>
  POPS_HD Real max_wave_speed(const State&) const {
    static_assert(Axis >= 0 && Axis < Dim,
                  "scalar-advection wave-speed axis is outside the dimension");
    return velocity_[Axis] < Real(0) ? -velocity_[Axis] : velocity_[Axis];
  }

  template <int Axis>
  POPS_HD void wave_speeds(const State&, Real& lower, Real& upper) const {
    static_assert(Axis >= 0 && Axis < Dim,
                  "scalar-advection wave-speed axis is outside the dimension");
    lower = upper = velocity_[Axis];
  }

 private:
  POPS_HD explicit constexpr ScalarAdvection(RealVector<Dim> velocity) : velocity_(velocity) {}

  RealVector<Dim> velocity_{};
};

template <int Dim>
class IdealGasEuler {
 public:
  using Schema = EulerStateSchema<Dim>;
  using State = typename Schema::Conservative;
  using Primitive = typename Schema::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Schema::nvars;

  IdealGasEuler() = default;

  static IdealGasEuler prepare(Real gamma) {
    if (!std::isfinite(static_cast<double>(gamma)) || !(gamma > Real(1)))
      throw std::invalid_argument("ND ideal-gas Euler requires a finite gamma greater than one");
    return IdealGasEuler(gamma);
  }

  POPS_HD Real gamma() const { return gamma_; }

  POPS_HD StateConversion<Primitive> recover(const State& conservative) const {
    StateConversion<Primitive> result{};
    if (!conservation_law_detail::finite(gamma_) || !(gamma_ > Real(1))) {
      result.status = StateConversionStatus::InvalidEquationOfState;
      return result;
    }
    if (!conservation_law_detail::finite_state(conservative)) {
      result.status = StateConversionStatus::NonFiniteState;
      return result;
    }

    const Real density = conservative[Schema::density];
    if (!(density > Real(0))) {
      result.status = StateConversionStatus::NonPositiveDensity;
      return result;
    }

    Real kinetic = Real(0);
    result.value[Schema::density] = density;
    for (int axis = 0; axis < Dim; ++axis) {
      const Real velocity = conservative[axis + 1] / density;
      result.value[axis + 1] = velocity;
      kinetic += Real(0.5) * density * velocity * velocity;
    }
    const Real pressure = (gamma_ - Real(1)) * (conservative[Schema::energy] - kinetic);
    if (!conservation_law_detail::finite(kinetic) || !conservation_law_detail::finite(pressure)) {
      result.status = StateConversionStatus::NonFiniteState;
      return result;
    }
    if (!(pressure > Real(0))) {
      result.status = StateConversionStatus::NonPositivePressure;
      return result;
    }
    result.value[Schema::pressure] = pressure;
    result.status = StateConversionStatus::Success;
    return result;
  }

  POPS_HD StateConversion<State> make_conservative(const Primitive& primitive) const {
    StateConversion<State> result{};
    if (!conservation_law_detail::finite(gamma_) || !(gamma_ > Real(1))) {
      result.status = StateConversionStatus::InvalidEquationOfState;
      return result;
    }
    if (!conservation_law_detail::finite_state(primitive)) {
      result.status = StateConversionStatus::NonFiniteState;
      return result;
    }

    const Real density = primitive[Schema::density];
    if (!(density > Real(0))) {
      result.status = StateConversionStatus::NonPositiveDensity;
      return result;
    }
    const Real pressure = primitive[Schema::pressure];
    if (!(pressure > Real(0))) {
      result.status = StateConversionStatus::NonPositivePressure;
      return result;
    }

    result.value[Schema::density] = density;
    Real kinetic = Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      const Real velocity = primitive[axis + 1];
      result.value[axis + 1] = density * velocity;
      kinetic += Real(0.5) * density * velocity * velocity;
    }
    result.value[Schema::energy] = pressure / (gamma_ - Real(1)) + kinetic;
    if (!conservation_law_detail::finite_state(result.value)) {
      result.value = {};
      result.status = StateConversionStatus::NonFiniteState;
      return result;
    }
    result.status = StateConversionStatus::Success;
    return result;
  }

  POPS_HD StateConversionStatus admissibility(const State& state) const {
    return recover(state).status;
  }

  POPS_HD Real pressure(const State& state) const {
    const auto primitive = recover(state);
    return primitive.succeeded() ? primitive.value[Schema::pressure]
                                 : std::numeric_limits<Real>::quiet_NaN();
  }

  template <int Axis>
  POPS_HD State flux(const State& conservative) const {
    static_assert(Axis >= 0 && Axis < Dim, "Euler flux axis is outside the dimension");
    const auto recovered = recover(conservative);
    if (!recovered.succeeded())
      return conservation_law_detail::invalid_state<State>();

    const Primitive& primitive = recovered.value;
    const Real normal_velocity = primitive[Schema::template velocity<Axis>];
    const Real pressure = primitive[Schema::pressure];
    State result{};
    result[Schema::density] = conservative[Schema::template momentum<Axis>];
    for (int momentum_axis = 0; momentum_axis < Dim; ++momentum_axis) {
      result[momentum_axis + 1] = conservative[momentum_axis + 1] * normal_velocity;
      if (momentum_axis == Axis)
        result[momentum_axis + 1] += pressure;
    }
    result[Schema::energy] = (conservative[Schema::energy] + pressure) * normal_velocity;
    return result;
  }

  template <int Axis>
  POPS_HD Real max_wave_speed(const State& conservative) const {
    static_assert(Axis >= 0 && Axis < Dim, "Euler wave-speed axis is outside the dimension");
    const auto primitive = recover(conservative);
    if (!primitive.succeeded())
      return std::numeric_limits<Real>::quiet_NaN();
    const Real velocity = primitive.value[Schema::template velocity<Axis>];
    const Real absolute_velocity = velocity < Real(0) ? -velocity : velocity;
    return absolute_velocity + Kokkos::sqrt(gamma_ * primitive.value[Schema::pressure] /
                                            primitive.value[Schema::density]);
  }

  template <int Axis>
  POPS_HD void wave_speeds(const State& conservative, Real& lower, Real& upper) const {
    static_assert(Axis >= 0 && Axis < Dim, "Euler wave-speed axis is outside the dimension");
    const auto primitive = recover(conservative);
    if (!primitive.succeeded()) {
      lower = upper = std::numeric_limits<Real>::quiet_NaN();
      return;
    }
    const Real velocity = primitive.value[Schema::template velocity<Axis>];
    const Real sound_speed =
        Kokkos::sqrt(gamma_ * primitive.value[Schema::pressure] / primitive.value[Schema::density]);
    lower = velocity - sound_speed;
    upper = velocity + sound_speed;
  }

 private:
  POPS_HD explicit constexpr IdealGasEuler(Real gamma) : gamma_(gamma) {}

  Real gamma_ = Real(1.4);
};

template <int Dim, class Model>
concept ConservationLaw = Dim >= 1 && Dim <= 3 && Model::dimension == Dim && Model::n_vars >= 1 &&
                          std::is_trivially_copyable_v<Model> &&
                          requires(const Model& model, const typename Model::State& state) {
                            typename Model::Schema;
                            typename Model::Primitive;
                            {
                              model.recover(state)
                            } -> std::same_as<StateConversion<typename Model::Primitive>>;
                            { model.admissibility(state) } -> std::same_as<StateConversionStatus>;
                          };

static_assert(ConservationLaw<1, ScalarAdvection<1>>);
static_assert(ConservationLaw<2, ScalarAdvection<2>>);
static_assert(ConservationLaw<3, ScalarAdvection<3>>);
static_assert(ConservationLaw<1, IdealGasEuler<1>>);
static_assert(ConservationLaw<2, IdealGasEuler<2>>);
static_assert(ConservationLaw<3, IdealGasEuler<3>>);

}  // namespace pops::nd
