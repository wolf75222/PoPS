// Exact-ranked AuxState and its device loader share one component authority in 1D, 2D, and 3D.

#include <gtest/gtest.h>

#include <pops/core/state/state.hpp>
#include <pops/mesh/storage/fab.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/spatial/primitives/state_access.hpp>
#include <pops/physics/bricks/hyperbolic.hpp>
#include <pops/physics/bricks/source.hpp>
#include <pops/physics/composition/composite.hpp>
#include <pops/physics/bricks/elliptic.hpp>
#include <pops/runtime/builders/factory/model_factory.hpp>

#include <cstddef>
#include <type_traits>
#include <vector>

using namespace pops;

namespace {

template <int Dim>
struct RankedAuxModel {
  using State = StateVec<1>;
  using Aux = AuxState<Dim>;
  static constexpr int n_vars = 1;

  POPS_HD State flux(const State&, const Aux&, int) const { return {}; }
  POPS_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  POPS_HD State source(const State&, const Aux&) const { return {}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

template <int Dim>
using RankedComposite = CompositeModel<ExBVelocityND<Dim>, NoSource, BackgroundDensity>;

template <int Dim>
using RankedConservationComposite =
    CompositeModel<nd::ScalarAdvection<Dim>, NoSource, BackgroundDensity>;

template <int Dim>
Extent<Dim> unit_extent() {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = 1;
  return result;
}

template <int Dim>
void seed_components(Fab<Dim>& field, Real base) {
  auto host = field.create_host_mirror();
  field.copy_to_host(host);
  const std::size_t component_stride = static_cast<std::size_t>(field.grown_box().numPts());
  for (int component = 0; component < field.ncomp(); ++component)
    host(static_cast<std::size_t>(component) * component_stride) = base + Real(component);
  field.copy_from_host(host);
}

template <int Axis, int Dim>
void expect_gradients(const AuxState<Dim>& auxiliary, Real base) {
  using layout = AuxComponentLayout<Dim>;
  EXPECT_EQ(auxiliary.template gradient<Axis>(),
            base + Real(layout::template gradient_component<Axis>()));
  EXPECT_EQ(auxiliary.template flux_provider<layout::template gradient_component<Axis>()>(),
            auxiliary.template gradient<Axis>());
  if constexpr (Axis + 1 < Dim)
    expect_gradients<Axis + 1>(auxiliary, base);
}

template <int Axis, int Dim>
void marshal_gradients(AuxState<Dim>& auxiliary, const std::vector<double>& marshaled) {
  using layout = AuxComponentLayout<Dim>;
  auxiliary.template gradient<Axis>() =
      marshaled[static_cast<std::size_t>(layout::template gradient_component<Axis>())];
  if constexpr (Axis + 1 < Dim)
    marshal_gradients<Axis + 1>(auxiliary, marshaled);
}

template <int Dim>
void check_device_and_host_marshaling() {
  using layout = AuxComponentLayout<Dim>;
  constexpr int full_width = layout::max_components;
  const Box<Dim> box = Box<Dim>::from_extents(unit_extent<Dim>());
  Fab<Dim> field(box, full_width);
  seed_components(field, Real(100));
  const auto view = static_cast<const Fab<Dim>&>(field).view();
  const AuxState<Dim> device = load_aux<full_width>(view, Index<Dim>{});

  EXPECT_EQ(device.phi, Real(100 + layout::phi));
  expect_gradients<0>(device, Real(100));
  EXPECT_EQ(device.B_z, Real(100 + layout::b_z));
  EXPECT_EQ(device.T_e, Real(100 + layout::t_e));
  EXPECT_EQ(device.template flux_provider<layout::b_z>(), device.B_z);
  EXPECT_EQ(device.template flux_provider<layout::t_e>(), device.T_e);
  EXPECT_EQ(device.template flux_provider<layout::named_begin>(), device.extra[0]);
  for (int slot = 0; slot < kAuxMaxExtra; ++slot)
    EXPECT_EQ(device.extra_field(slot), Real(100 + layout::named_begin + slot));

  std::vector<double> marshaled(static_cast<std::size_t>(full_width));
  for (int component = 0; component < full_width; ++component)
    marshaled[static_cast<std::size_t>(component)] = 100 + component;
  AuxState<Dim> host{};
  host.phi = marshaled[static_cast<std::size_t>(layout::phi)];
  marshal_gradients<0>(host, marshaled);
  host.B_z = marshaled[static_cast<std::size_t>(layout::b_z)];
  host.T_e = marshaled[static_cast<std::size_t>(layout::t_e)];
  for (int slot = 0; slot < kAuxMaxExtra; ++slot)
    host.extra[slot] = marshaled[static_cast<std::size_t>(layout::named_begin + slot)];

  EXPECT_EQ(host.phi, device.phi);
  expect_gradients<0>(host, Real(100));
  EXPECT_EQ(host.B_z, device.B_z);
  EXPECT_EQ(host.T_e, device.T_e);
  for (int slot = 0; slot < kAuxMaxExtra; ++slot)
    EXPECT_EQ(host.extra[slot], device.extra[slot]);
}

template <int Dim>
void check_base_width_ignores_extra_fields() {
  using layout = AuxComponentLayout<Dim>;
  const Box<Dim> box = Box<Dim>::from_extents(unit_extent<Dim>());
  Fab<Dim> field(box, layout::max_components);
  seed_components(field, Real(999));
  const auto view = static_cast<const Fab<Dim>&>(field).view();
  const AuxState<Dim> loaded = load_aux<layout::base_components>(view, Index<Dim>{});

  EXPECT_EQ(loaded.phi, Real(999 + layout::phi));
  expect_gradients<0>(loaded, Real(999));
  EXPECT_EQ(loaded.B_z, Real(0));
  EXPECT_EQ(loaded.T_e, Real(0));
  for (int slot = 0; slot < kAuxMaxExtra; ++slot)
    EXPECT_EQ(loaded.extra[slot], Real(0));
}

template <int Dim>
constexpr bool exact_ranked_storage() {
  return std::is_trivially_copyable_v<AuxState<Dim>> &&
         sizeof(AuxState<Dim>) == sizeof(Real) * AuxComponentLayout<Dim>::max_components &&
         sizeof(AuxState<Dim>::gradients) == sizeof(Real) * Dim;
}

template <int Axis, int Dim>
void seed_force_state(AuxState<Dim>& auxiliary, StateVec<Dim + 2>& state) {
  auxiliary.template gradient<Axis>() = Real(Axis + 1);
  state[Axis + 1] = Real(3 + Axis);
  if constexpr (Axis + 1 < Dim)
    seed_force_state<Axis + 1>(auxiliary, state);
}

template <int Axis, int Dim>
void expect_ranked_force_momenta(const StateVec<Dim + 2>& source, Real density, Real coefficient) {
  EXPECT_EQ(source[Axis + 1], -coefficient * density * Real(Axis + 1));
  if constexpr (Axis + 1 < Dim)
    expect_ranked_force_momenta<Axis + 1, Dim>(source, density, coefficient);
}

template <int Dim>
void check_ranked_gradient_forces() {
  AuxState<Dim> auxiliary{};
  StateVec<Dim + 2> state{};
  state[0] = Real(2);
  seed_force_state<0>(auxiliary, state);
  state[Dim + 1] = Real(17);

  const Real qom = Real(1.5);
  const auto electrostatic = PotentialForceND<Dim>{qom}.apply(state, auxiliary);
  const auto gravity = GravityForceND<Dim>{}.apply(state, auxiliary);
  expect_ranked_force_momenta<0, Dim>(electrostatic, state[0], qom);
  expect_ranked_force_momenta<0, Dim>(gravity, state[0], Real(1));

  Real electrostatic_work = Real(0);
  Real gravity_work = Real(0);
  for (int axis = 0; axis < Dim; ++axis) {
    electrostatic_work -= qom * state[axis + 1] * Real(axis + 1);
    gravity_work -= state[axis + 1] * Real(axis + 1);
  }
  EXPECT_EQ(electrostatic[Dim + 1], electrostatic_work);
  EXPECT_EQ(gravity[Dim + 1], gravity_work);
}

template <int Axis, int Dim>
void expect_cartesian_exb_axes(const ExBVelocityND<Dim>& model, const AuxState<Dim>& auxiliary) {
  const StateVec<1> state{Real(3)};
  Real expected_velocity = Real(0);
  if constexpr (Dim >= 2 && Axis == 0)
    expected_velocity = -auxiliary.template gradient<1>() / model.B0;
  else if constexpr (Dim >= 2 && Axis == 1)
    expected_velocity = auxiliary.template gradient<0>() / model.B0;
  EXPECT_EQ(model.template velocity<Axis>(auxiliary), expected_velocity);
  EXPECT_EQ(model.template flux<Axis>(state, auxiliary)[0], state[0] * expected_velocity);
  EXPECT_EQ(model.flux(state, auxiliary, Axis)[0], state[0] * expected_velocity);
  if constexpr (Axis + 1 < Dim)
    expect_cartesian_exb_axes<Axis + 1>(model, auxiliary);
}

template <int Dim>
void check_ranked_cartesian_exb() {
  AuxState<Dim> auxiliary{};
  auxiliary.template gradient<0>() = Real(2);
  if constexpr (Dim >= 2)
    auxiliary.template gradient<1>() = Real(-4);
  if constexpr (Dim >= 3)
    auxiliary.template gradient<2>() = Real(9);
  expect_cartesian_exb_axes<0>(ExBVelocityND<Dim>{Real(2)}, auxiliary);
}

static_assert(exact_ranked_storage<1>());
static_assert(exact_ranked_storage<2>());
static_assert(exact_ranked_storage<3>());
static_assert(MagneticLorentzForceND<1>::n_aux == AuxComponentLayout<1>::b_z + 1);
static_assert(MagneticLorentzForceND<2>::n_aux == AuxComponentLayout<2>::b_z + 1);
static_assert(MagneticLorentzForceND<3>::n_aux == AuxComponentLayout<3>::b_z + 1);
static_assert(!MagneticLorentzForceND<1>::planar_capability);
static_assert(MagneticLorentzForceND<2>::planar_capability);
static_assert(ExBVelocityPolar::dimension == 2);
static_assert(std::is_same_v<Aux, AuxState<kNativeDimension>>);
static_assert(PhysicalModelFor<RankedAuxModel<1>, 1>);
static_assert(PhysicalModelFor<RankedAuxModel<2>, 2>);
static_assert(PhysicalModelFor<RankedAuxModel<3>, 3>);
static_assert(PhysicalModelFor<SourceFreeModel<RankedAuxModel<1>>, 1>);
static_assert(PhysicalModelFor<SourceFreeModel<RankedAuxModel<2>>, 2>);
static_assert(PhysicalModelFor<SourceFreeModel<RankedAuxModel<3>>, 3>);
static_assert(!PhysicalModelFor<RankedAuxModel<1>, 2>);
static_assert(PhysicalModel<RankedAuxModel<kNativeDimension>>);
static_assert(RankedComposite<1>::dimension == 1);
static_assert(RankedComposite<2>::dimension == 2);
static_assert(RankedComposite<3>::dimension == 3);
static_assert(std::is_same_v<typename RankedComposite<1>::Aux, AuxState<1>>);
static_assert(std::is_same_v<typename RankedComposite<2>::Aux, AuxState<2>>);
static_assert(std::is_same_v<typename RankedComposite<3>::Aux, AuxState<3>>);
static_assert(nd::ConservationLaw<1, RankedConservationComposite<1>>);
static_assert(nd::ConservationLaw<2, RankedConservationComposite<2>>);
static_assert(nd::ConservationLaw<3, RankedConservationComposite<3>>);
static_assert(requires(const RankedConservationComposite<3>& model,
                       const RankedConservationComposite<3>::Primitive& primitive) {
  model.make_conservative(primitive);
});
static_assert(aux_comps_for<RankedAuxModel<1>, 1>() == kAuxBaseCompsFor<1>);
static_assert(aux_comps_for<RankedAuxModel<2>, 2>() == kAuxBaseCompsFor<2>);
static_assert(aux_comps_for<RankedAuxModel<3>, 3>() == kAuxBaseCompsFor<3>);

}  // namespace

TEST(AuxSingleSource, RankedDeviceAndHostMarshalingAgree) {
  check_device_and_host_marshaling<1>();
  check_device_and_host_marshaling<2>();
  check_device_and_host_marshaling<3>();
}

TEST(AuxSingleSource, RankedBaseWidthIgnoresOnlyRealExtras) {
  check_base_width_ignores_extra_fields<1>();
  check_base_width_ignores_extra_fields<2>();
  check_base_width_ignores_extra_fields<3>();
}

TEST(AuxSingleSource, GradientForcesConsumeEveryExactRankedAxis) {
  check_ranked_gradient_forces<1>();
  check_ranked_gradient_forces<2>();
  check_ranked_gradient_forces<3>();
}

TEST(AuxSingleSource, CartesianExBHasOnlyItsAvailablePlanarComponents) {
  check_ranked_cartesian_exb<1>();
  check_ranked_cartesian_exb<2>();
  check_ranked_cartesian_exb<3>();
}

TEST(AuxSingleSource, ThreeDimensionalBzLorentzLeavesZMomentumUntouched) {
  AuxState<3> auxiliary{};
  auxiliary.B_z = Real(2);
  const StateVec<5> state{Real(1), Real(3), Real(-4), Real(7), Real(11)};
  const auto source = MagneticLorentzForceND<3>{Real(0.5)}.apply(state, auxiliary);
  EXPECT_EQ(source[1], Real(-4));
  EXPECT_EQ(source[2], Real(-3));
  EXPECT_EQ(source[3], Real(0));
  EXPECT_EQ(source[4], Real(0));
}

TEST(AuxSingleSource, NativeRoleBinderConnectsEveryCompiledMomentumAxis) {
  VariableSet variables{VariableKind::Conservative, {}, kNativeDimension + 2, {}};
  variables.names.resize(static_cast<std::size_t>(variables.size), "component");
  variables.roles.push_back(VariableRole::Energy);
  variables.roles.push_back(VariableRole::Density);
  if constexpr (kNativeDimension >= 3)
    variables.roles.push_back(VariableRole::MomentumZ);
  if constexpr (kNativeDimension >= 2)
    variables.roles.push_back(VariableRole::MomentumY);
  variables.roles.push_back(VariableRole::MomentumX);

  PotentialForce force{};
  detail::bind_variable_roles(force, variables);
  EXPECT_EQ(force.c_E, 0);
  EXPECT_EQ(force.c_rho, 1);
  EXPECT_EQ(force.c_mx, variables.size - 1);
  if constexpr (kNativeDimension >= 2)
    EXPECT_EQ(force.c_my, variables.size - 2);
  if constexpr (kNativeDimension >= 3)
    EXPECT_EQ(force.c_mz, 2);
}
