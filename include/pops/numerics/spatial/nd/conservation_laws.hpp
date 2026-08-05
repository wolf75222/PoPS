/// @file
/// @brief Dimension-generic scalar-advection and ideal-gas Euler conservation laws.

#pragma once

#include <pops/mesh/index/real_vector.hpp>
#include <pops/core/state/variables.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/numerics/spatial/nd/state_schema.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <cmath>
#include <concepts>
#include <array>
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

POPS_HD inline Real entropy_fixed_absolute(Real eigenvalue, Real width) {
  const Real absolute = eigenvalue < Real(0) ? -eigenvalue : eigenvalue;
  if (!(width > Real(0)) || absolute >= width)
    return absolute;
  return Real(0.5) * (eigenvalue * eigenvalue / width + width);
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

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.nd.scalar-advection", 1};
  }

  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(velocity_[axis]);
  }

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

  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"scalar"}, 1, {VariableRole::Scalar}};
  }

  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"scalar"}, 1, {VariableRole::Scalar}};
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

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.nd.ideal-gas-euler", 1};
  }

  void serialize_exact_parameters(ExactContractBuilder& contract) const { contract.scalar(gamma_); }

  IdealGasEuler() = default;

  static IdealGasEuler prepare(Real gamma) {
    if (!std::isfinite(static_cast<double>(gamma)) || !(gamma > Real(1)))
      throw std::invalid_argument("ND ideal-gas Euler requires a finite gamma greater than one");
    return IdealGasEuler(gamma);
  }

  POPS_HD Real gamma() const { return gamma_; }

  static VariableSet conservative_vars() {
    constexpr std::array<const char*, 3> names{"rho_u", "rho_v", "rho_w"};
    constexpr std::array<VariableRole, 3> roles{VariableRole::MomentumX, VariableRole::MomentumY,
                                                VariableRole::MomentumZ};
    VariableSet result{VariableKind::Conservative, {"rho"}, n_vars, {VariableRole::Density}};
    for (int axis = 0; axis < Dim; ++axis) {
      result.names.emplace_back(names[static_cast<std::size_t>(axis)]);
      result.roles.push_back(roles[static_cast<std::size_t>(axis)]);
    }
    result.names.emplace_back("E");
    result.roles.push_back(VariableRole::Energy);
    return result;
  }

  static VariableSet primitive_vars() {
    constexpr std::array<const char*, 3> names{"u", "v", "w"};
    constexpr std::array<VariableRole, 3> roles{VariableRole::VelocityX, VariableRole::VelocityY,
                                                VariableRole::VelocityZ};
    VariableSet result{VariableKind::Primitive, {"rho"}, n_vars, {VariableRole::Density}};
    for (int axis = 0; axis < Dim; ++axis) {
      result.names.emplace_back(names[static_cast<std::size_t>(axis)]);
      result.roles.push_back(roles[static_cast<std::size_t>(axis)]);
    }
    result.names.emplace_back("p");
    result.roles.push_back(VariableRole::Pressure);
    return result;
  }

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

  template <int Axis>
  POPS_HD Real contact_speed(const State& left, const State& right, Real pressure_left,
                             Real pressure_right, Real speed_left, Real speed_right) const {
    static_assert(Axis >= 0 && Axis < Dim, "Euler contact axis is outside the dimension");
    const Real density_left = left[Schema::density];
    const Real density_right = right[Schema::density];
    const Real velocity_left = left[Schema::template momentum<Axis>] / density_left;
    const Real velocity_right = right[Schema::template momentum<Axis>] / density_right;
    return (pressure_right - pressure_left +
            density_left * velocity_left * (speed_left - velocity_left) -
            density_right * velocity_right * (speed_right - velocity_right)) /
           (density_left * (speed_left - velocity_left) -
            density_right * (speed_right - velocity_right));
  }

  template <int Axis>
  POPS_HD State star_state(const State& conservative, Real pressure_value, Real speed,
                           Real contact) const {
    static_assert(Axis >= 0 && Axis < Dim, "Euler star-state axis is outside the dimension");
    const Real density = conservative[Schema::density];
    const Real normal_velocity = conservative[Schema::template momentum<Axis>] / density;
    const Real star_density = density * (speed - normal_velocity) / (speed - contact);
    State result{};
    result[Schema::density] = star_density;
    for (int momentum_axis = 0; momentum_axis < Dim; ++momentum_axis)
      result[momentum_axis + 1] =
          star_density *
          (momentum_axis == Axis ? contact : conservative[momentum_axis + 1] / density);
    result[Schema::energy] =
        star_density * (conservative[Schema::energy] / density +
                        (contact - normal_velocity) *
                            (contact + pressure_value / (density * (speed - normal_velocity))));
    return result;
  }

  template <int Axis>
  POPS_HD State roe_dissipation(const State& left, const State& right) const {
    static_assert(Axis >= 0 && Axis < Dim, "Euler Roe axis is outside the dimension");
    const Real density_left = left[Schema::density];
    const Real density_right = right[Schema::density];
    const Real pressure_left = pressure(left);
    const Real pressure_right = pressure(right);
    const Real root_left = Kokkos::sqrt(density_left);
    const Real root_right = Kokkos::sqrt(density_right);
    const Real denominator = root_left + root_right;
    const Real roe_density = root_left * root_right;
    const Real enthalpy_left = (left[Schema::energy] + pressure_left) / density_left;
    const Real enthalpy_right = (right[Schema::energy] + pressure_right) / density_right;
    const Real enthalpy = (root_left * enthalpy_left + root_right * enthalpy_right) / denominator;

    Real velocity[Dim]{};
    Real velocity_left[Dim]{};
    Real velocity_right[Dim]{};
    Real speed_squared = Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      velocity_left[axis] = left[axis + 1] / density_left;
      velocity_right[axis] = right[axis + 1] / density_right;
      velocity[axis] =
          (root_left * velocity_left[axis] + root_right * velocity_right[axis]) / denominator;
      speed_squared += velocity[axis] * velocity[axis];
    }

    const Real sound_squared = (gamma_ - Real(1)) * (enthalpy - Real(0.5) * speed_squared);
    const Real sound = Kokkos::sqrt(sound_squared);
    const Real density_jump = density_right - density_left;
    const Real pressure_jump = pressure_right - pressure_left;
    const Real normal_jump = velocity_right[Axis] - velocity_left[Axis];
    const Real acoustic_low =
        (pressure_jump - roe_density * sound * normal_jump) / (Real(2) * sound_squared);
    const Real entropy = density_jump - pressure_jump / sound_squared;
    const Real acoustic_high =
        (pressure_jump + roe_density * sound * normal_jump) / (Real(2) * sound_squared);
    const Real normal_velocity = velocity[Axis];
    const Real acoustic_scale = Real(0.1) * sound;
    const Real low_absolute =
        conservation_law_detail::entropy_fixed_absolute(normal_velocity - sound, acoustic_scale);
    const Real contact_absolute = normal_velocity < Real(0) ? -normal_velocity : normal_velocity;
    const Real high_absolute =
        conservation_law_detail::entropy_fixed_absolute(normal_velocity + sound, acoustic_scale);

    State result{};
    result[Schema::density] =
        low_absolute * acoustic_low + contact_absolute * entropy + high_absolute * acoustic_high;
    for (int axis = 0; axis < Dim; ++axis) {
      if (axis == Axis) {
        result[axis + 1] = low_absolute * acoustic_low * (normal_velocity - sound) +
                           contact_absolute * entropy * normal_velocity +
                           high_absolute * acoustic_high * (normal_velocity + sound);
      } else {
        const Real shear = roe_density * (velocity_right[axis] - velocity_left[axis]);
        result[axis + 1] = low_absolute * acoustic_low * velocity[axis] +
                           contact_absolute * (entropy * velocity[axis] + shear) +
                           high_absolute * acoustic_high * velocity[axis];
      }
    }
    Real contact_energy = entropy * Real(0.5) * speed_squared;
    for (int axis = 0; axis < Dim; ++axis)
      if (axis != Axis)
        contact_energy +=
            roe_density * (velocity_right[axis] - velocity_left[axis]) * velocity[axis];
    result[Schema::energy] = low_absolute * acoustic_low * (enthalpy - normal_velocity * sound) +
                             contact_absolute * contact_energy +
                             high_absolute * acoustic_high * (enthalpy + normal_velocity * sound);
    return result;
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
