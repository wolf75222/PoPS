// Exact-ranked AuxState and its device loader share one component authority in 1D, 2D, and 3D.

#include <gtest/gtest.h>

#include <pops/core/state/state.hpp>
#include <pops/mesh/storage/fab.hpp>
#include <pops/numerics/spatial/primitives/state_access.hpp>

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

static_assert(exact_ranked_storage<1>());
static_assert(exact_ranked_storage<2>());
static_assert(exact_ranked_storage<3>());
static_assert(std::is_same_v<Aux, AuxState<kNativeDimension>>);
static_assert(PhysicalModelFor<RankedAuxModel<1>, 1>);
static_assert(PhysicalModelFor<RankedAuxModel<2>, 2>);
static_assert(PhysicalModelFor<RankedAuxModel<3>, 3>);
static_assert(PhysicalModelFor<SourceFreeModel<RankedAuxModel<1>>, 1>);
static_assert(PhysicalModelFor<SourceFreeModel<RankedAuxModel<2>>, 2>);
static_assert(PhysicalModelFor<SourceFreeModel<RankedAuxModel<3>>, 3>);
static_assert(!PhysicalModelFor<RankedAuxModel<1>, 2>);
static_assert(PhysicalModel<RankedAuxModel<kNativeDimension>>);
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
