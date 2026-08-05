#pragma once

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/model/physical_model.hpp>
#include <pops/core/state/state.hpp>
#include <pops/physics/composition/exact_brick_contract.hpp>

#include <cstdint>

/// @file
/// @brief Exact-ranked local source bricks S(U, aux).

namespace pops {

namespace source_detail {

template <int Axis, class Force>
POPS_HD int momentum_component(const Force& force) {
  static_assert(Axis >= 0 && Axis < 3, "source momentum axis is outside the supported rank");
  if constexpr (Axis == 0)
    return force.c_mx;
  else if constexpr (Axis == 1)
    return force.c_my;
  else
    return force.c_mz;
}

template <int Axis, int Dim, class Force, class State>
POPS_HD void apply_gradient_force(const Force& force, const State& state,
                                  const AuxState<Dim>& auxiliary, Real coefficient, State& source,
                                  Real& work) {
  static_assert(Axis >= 0 && Axis < Dim, "source gradient axis is outside the spatial rank");
  const int momentum = momentum_component<Axis>(force);
  const Real field = -auxiliary.template gradient<Axis>();
  source[momentum] = coefficient * state[force.c_rho] * field;
  if constexpr (State::size() == Dim + 2)
    work += coefficient * state[momentum] * field;
  if constexpr (Axis + 1 < Dim)
    apply_gradient_force<Axis + 1>(force, state, auxiliary, coefficient, source, work);
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

  template <class State, int Dim>
  POPS_HD State apply(const State&, const AuxState<Dim>&) const {
    return State{};
  }
};

/// Electrostatic force `(q/m) rho (-grad phi)` on every momentum axis of `Dim`.
///
/// Canonical states contain density, exactly `Dim` momentum components, and optionally energy.
/// The host role binder may replace every canonical component index before device execution.
template <int Dim>
struct PotentialForceND {
  static_assert(Dim >= 1 && Dim <= 3, "PotentialForceND supports dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;
  static constexpr bool requires_energy_role(int state_size) { return state_size == Dim + 2; }

  Real qom = Real(1);
  int c_rho = 0;
  int c_mx = 1;
  int c_my = 2;
  int c_mz = 3;
  int c_E = Dim + 1;

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.physics.source.potential-force-nd", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(std::int32_t{Dim})
        .scalar(qom)
        .scalar(std::int32_t{c_rho})
        .scalar(std::int32_t{c_mx})
        .scalar(std::int32_t{c_my})
        .scalar(std::int32_t{c_mz})
        .scalar(std::int32_t{c_E});
  }

  template <class State>
  POPS_HD State apply(const State& state, const AuxState<Dim>& auxiliary) const {
    static_assert(State::size() >= Dim + 1,
                  "PotentialForceND requires density and one momentum component per axis");
    State source{};
    Real work = Real(0);
    source_detail::apply_gradient_force<0>(*this, state, auxiliary, qom, source, work);
    if constexpr (State::size() == Dim + 2)
      source[c_E] = work;
    return source;
  }
};

/// Gravitational force `rho (-grad phi)` on every momentum axis of `Dim`.
template <int Dim>
struct GravityForceND {
  static_assert(Dim >= 1 && Dim <= 3, "GravityForceND supports dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;
  static constexpr bool requires_energy_role(int state_size) { return state_size == Dim + 2; }

  int c_rho = 0;
  int c_mx = 1;
  int c_my = 2;
  int c_mz = 3;
  int c_E = Dim + 1;

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.physics.source.gravity-force-nd", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(std::int32_t{Dim})
        .scalar(std::int32_t{c_rho})
        .scalar(std::int32_t{c_mx})
        .scalar(std::int32_t{c_my})
        .scalar(std::int32_t{c_mz})
        .scalar(std::int32_t{c_E});
  }

  template <class State>
  POPS_HD State apply(const State& state, const AuxState<Dim>& auxiliary) const {
    static_assert(State::size() >= Dim + 1,
                  "GravityForceND requires density and one momentum component per axis");
    State source{};
    Real work = Real(0);
    source_detail::apply_gradient_force<0>(*this, state, auxiliary, Real(1), source, work);
    if constexpr (State::size() == Dim + 2)
      source[c_E] = work;
    return source;
  }
};

/// Explicit Lorentz force for an out-of-plane `B_z` field.
///
/// This capability needs the x-y plane. In 3D it rotates the x/y momenta and leaves z unchanged;
/// in 2D the same algebra applies to Cartesian or to the local `(e_r, e_theta)` polar basis.
template <int Dim>
struct MagneticLorentzForceND {
  static_assert(Dim >= 1 && Dim <= 3,
                "MagneticLorentzForceND supports build dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;
  static constexpr bool planar_capability = Dim >= 2;
  static constexpr int n_aux = AuxComponentLayout<Dim>::b_z + 1;

  Real qom = Real(1);
  int c_mx = 1;
  int c_my = 2;

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.physics.source.magnetic-lorentz-force-nd", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(std::int32_t{Dim})
        .scalar(qom)
        .scalar(std::int32_t{c_mx})
        .scalar(std::int32_t{c_my});
  }

  template <class State>
  POPS_HD State apply(const State& state, const AuxState<Dim>& auxiliary) const {
    static_assert(Dim >= 2,
                  "MagneticLorentzForceND requires an explicit two-axis planar capability");
    static_assert(State::size() >= Dim + 1,
                  "MagneticLorentzForceND requires one momentum component per spatial axis");
    const Real rotation = qom * auxiliary.B_z;
    State source{};
    source[c_mx] = rotation * state[c_my];
    source[c_my] = -rotation * state[c_mx];
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
  static constexpr int n_aux = aux_comps_for<A, dimension>() > aux_comps_for<B, dimension>()
                                   ? aux_comps_for<A, dimension>()
                                   : aux_comps_for<B, dimension>();

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
        .scalar(std::int32_t{n_aux});
    physics_contract_detail::append_exact_brick(contract, "left", a);
    physics_contract_detail::append_exact_brick(contract, "right", b);
  }

  template <class State>
  POPS_HD State apply(const State& state, const AuxState<dimension>& auxiliary) const {
    return a.apply(state, auxiliary) + b.apply(state, auxiliary);
  }
};

}  // namespace pops
