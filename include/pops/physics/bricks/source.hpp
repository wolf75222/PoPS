#pragma once

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/model/physical_model.hpp>
#include <pops/core/state/state.hpp>
#include <pops/physics/composition/exact_brick_contract.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

/// @file
/// @brief Exact-ranked local source bricks S(U, aux).

namespace pops {

namespace source_detail {

template <int Dim>
struct DefaultGradientProviderSlots;

template <>
struct DefaultGradientProviderSlots<1> {
  using type = ProviderSlots<0>;
};
template <>
struct DefaultGradientProviderSlots<2> {
  using type = ProviderSlots<0, 1>;
};
template <>
struct DefaultGradientProviderSlots<3> {
  using type = ProviderSlots<0, 1, 2>;
};

/// Exact-ranked component map used by source bricks after host-side role binding.
///
/// A source compiled for `Dim` owns exactly `Dim` momentum indices.  There are no dormant y/z
/// fields in lower-rank artifacts and no axis-name branch in device code.
template <int Dim>
struct MomentumComponents {
  static_assert(Dim >= 1 && Dim <= 3, "source momentum components support dimensions 1..3");

  static constexpr int dimension = Dim;
  int values[Dim]{};

  POPS_HD constexpr MomentumComponents() {
    for (int axis = 0; axis < Dim; ++axis)
      values[axis] = axis + 1;
  }

  POPS_HD constexpr int& operator[](int axis) { return values[axis]; }
  POPS_HD constexpr int operator[](int axis) const { return values[axis]; }
};

static_assert(std::is_trivially_copyable_v<MomentumComponents<1>> &&
                  std::is_trivially_copyable_v<MomentumComponents<2>> &&
                  std::is_trivially_copyable_v<MomentumComponents<3>>,
              "ranked source component maps must remain device-copyable");
static_assert(sizeof(MomentumComponents<1>) == sizeof(int) &&
                  sizeof(MomentumComponents<2>) == 2 * sizeof(int) &&
                  sizeof(MomentumComponents<3>) == 3 * sizeof(int),
              "ranked source component maps must not retain inactive axes");

template <int Axis, int Dim, class Force, class State, class Providers>
POPS_HD void apply_gradient_force(const Force& force, const State& state,
                                  const Providers& providers, Real coefficient, State& source,
                                  Real& work) {
  static_assert(Axis >= 0 && Axis < Dim, "source gradient axis is outside the spatial rank");
  const int momentum = force.momentum_components[Axis];
  constexpr int provider_slot = Force::gradient_slots::template slot<Axis>();
  const Real field = -provider_value<provider_slot>(providers);
  source[momentum] = coefficient * state[force.c_rho] * field;
  if constexpr (State::size() == Dim + 2)
    work += coefficient * state[momentum] * field;
  if constexpr (Axis + 1 < Dim)
    apply_gradient_force<Axis + 1, Dim>(force, state, providers, coefficient, source, work);
}

template <class Source>
consteval int declared_dimension() {
  if constexpr (requires { Source::dimension; })
    return static_cast<int>(Source::dimension);
  return 0;
}

}  // namespace source_detail

/// Neutral source. The auxiliary rank is deduced from the exact pointwise carrier.
struct NoSource {
  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.physics.source.none", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder&) const {}

  template <class State, class Providers>
  POPS_HD State apply(const State&, const Providers&) const {
    return State{};
  }
};

/// Electrostatic force `(q/m) rho (-grad phi)` on every momentum axis of `Dim`.
///
/// Canonical states contain density, exactly `Dim` momentum components, and optionally energy.
/// The host role binder may replace every canonical component index before device execution.
template <int Dim, class GradientSlots = typename source_detail::DefaultGradientProviderSlots<Dim>::type>
struct PotentialForceND {
  static_assert(Dim >= 1 && Dim <= 3, "PotentialForceND supports dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;
  using gradient_slots = GradientSlots;
  static_assert(gradient_slots::count == Dim,
                "PotentialForceND requires exactly one explicit gradient slot per axis");
  static constexpr int n_providers = gradient_slots::required_count();
  static constexpr bool requires_energy_role(int state_size) { return state_size == Dim + 2; }

  Real qom = Real(1);
  int c_rho = 0;
  source_detail::MomentumComponents<Dim> momentum_components{};
  int c_E = Dim + 1;

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.physics.source.potential-force-nd", 2};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(std::int32_t{Dim}).scalar(qom).scalar(std::int32_t{c_rho});
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(std::int32_t{momentum_components[axis]});
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(std::int32_t{gradient_slots::values[static_cast<std::size_t>(axis)]});
    contract.scalar(std::int32_t{c_E});
  }

  template <class State, class Providers>
  POPS_HD State apply(const State& state, const Providers& providers) const {
    static_assert(State::size() >= Dim + 1,
                  "PotentialForceND requires density and one momentum component per axis");
    State source{};
    Real work = Real(0);
    source_detail::apply_gradient_force<0, Dim>(*this, state, providers, qom, source, work);
    if constexpr (State::size() == Dim + 2)
      source[c_E] = work;
    return source;
  }
};

/// Gravitational force `rho (-grad phi)` on every momentum axis of `Dim`.
template <int Dim, class GradientSlots = typename source_detail::DefaultGradientProviderSlots<Dim>::type>
struct GravityForceND {
  static_assert(Dim >= 1 && Dim <= 3, "GravityForceND supports dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;
  using gradient_slots = GradientSlots;
  static_assert(gradient_slots::count == Dim,
                "GravityForceND requires exactly one explicit gradient slot per axis");
  static constexpr int n_providers = gradient_slots::required_count();
  static constexpr bool requires_energy_role(int state_size) { return state_size == Dim + 2; }

  int c_rho = 0;
  source_detail::MomentumComponents<Dim> momentum_components{};
  int c_E = Dim + 1;

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.physics.source.gravity-force-nd", 2};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(std::int32_t{Dim}).scalar(std::int32_t{c_rho});
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(std::int32_t{momentum_components[axis]});
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(std::int32_t{gradient_slots::values[static_cast<std::size_t>(axis)]});
    contract.scalar(std::int32_t{c_E});
  }

  template <class State, class Providers>
  POPS_HD State apply(const State& state, const Providers& providers) const {
    static_assert(State::size() >= Dim + 1,
                  "GravityForceND requires density and one momentum component per axis");
    State source{};
    Real work = Real(0);
    source_detail::apply_gradient_force<0, Dim>(*this, state, providers, Real(1), source, work);
    if constexpr (State::size() == Dim + 2)
      source[c_E] = work;
    return source;
  }
};

/// Explicit Lorentz force for an out-of-plane `B_z` field.
///
/// This capability needs the x-y plane. In 3D it rotates the x/y momenta and leaves z unchanged;
/// in 2D the same algebra applies to Cartesian or to the local `(e_r, e_theta)` polar basis.
template <int Dim, int MagneticComponent = 0>
struct MagneticLorentzForceND {
  static_assert(Dim >= 1 && Dim <= 3,
                "MagneticLorentzForceND supports build dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;
  static constexpr bool planar_capability = Dim >= 2;
  static_assert(MagneticComponent >= 0, "magnetic provider slot cannot be negative");
  static constexpr int magnetic_component = MagneticComponent;
  static constexpr int n_providers = magnetic_component + 1;

  Real qom = Real(1);
  source_detail::MomentumComponents<Dim> momentum_components{};

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.physics.source.magnetic-lorentz-force-nd", 2};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(std::int32_t{Dim}).scalar(qom).scalar(std::int32_t{magnetic_component});
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(std::int32_t{momentum_components[axis]});
  }

  template <class State, class Providers>
  POPS_HD State apply(const State& state, const Providers& providers) const {
    static_assert(Dim >= 2,
                  "MagneticLorentzForceND requires an explicit two-axis planar capability");
    static_assert(State::size() >= Dim + 1,
                  "MagneticLorentzForceND requires one momentum component per spatial axis");
    const Real rotation = qom * provider_value<magnetic_component>(providers);
    State source{};
    source[momentum_components[0]] = rotation * state[momentum_components[1]];
    source[momentum_components[1]] = -rotation * state[momentum_components[0]];
    return source;
  }
};

using PotentialForce = PotentialForceND<kNativeDimension>;
using GravityForce = GravityForceND<kNativeDimension>;
using MagneticLorentzForce = MagneticLorentzForceND<kNativeDimension>;

/// Sum of two source bricks over one authenticated auxiliary rank.
template <class A, class B>
struct CompositeSource {
  static constexpr int a_dimension = source_detail::declared_dimension<A>();
  static constexpr int b_dimension = source_detail::declared_dimension<B>();
  static_assert(a_dimension == 0 || b_dimension == 0 || a_dimension == b_dimension,
                "CompositeSource cannot mix source bricks from different spatial ranks");
  static constexpr int dimension =
      a_dimension != 0 ? a_dimension : (b_dimension != 0 ? b_dimension : kNativeDimension);

  A a{};
  B b{};
  static constexpr int n_providers = provider_count_for<A, dimension>() >
                                             provider_count_for<B, dimension>()
                                         ? provider_count_for<A, dimension>()
                                         : provider_count_for<B, dimension>();

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept
    requires(physics_contract_detail::ExactPhysicsBrickContract<A> &&
             physics_contract_detail::ExactPhysicsBrickContract<B>)
  {
    return {"pops.physics.source.composite", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const
    requires(physics_contract_detail::ExactPhysicsBrickContract<A> &&
             physics_contract_detail::ExactPhysicsBrickContract<B>)
  {
    contract.text("pops.physics.composite-source-parameters")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{dimension})
        .scalar(std::int32_t{n_providers});
    physics_contract_detail::append_exact_brick(contract, "left", a);
    physics_contract_detail::append_exact_brick(contract, "right", b);
  }

  template <class State, class Providers>
  POPS_HD State apply(const State& state, const Providers& providers) const {
    return a.apply(state, providers) + b.apply(state, providers);
  }
};

}  // namespace pops
