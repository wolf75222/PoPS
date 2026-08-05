// POPS_AUX_FIELDS is the single source for the auxiliary component layout. These tests prove that
// the ranked device loader and host marshaling agree for every canonical extra field, and that the
// base-width route leaves extras untouched in 1D, 2D, and 3D.

#include <gtest/gtest.h>

#include <pops/core/state/state.hpp>
#include <pops/mesh/storage/fab.hpp>
#include <pops/numerics/spatial/primitives/state_access.hpp>

#include <cstddef>
#include <vector>

using namespace pops;

namespace {

constexpr int kNExtra = [] {
  int count = 0;
#define POPS_AUX_COUNT(name, index) ++count;
  POPS_AUX_FIELDS(POPS_AUX_COUNT)
#undef POPS_AUX_COUNT
  return count;
}();

constexpr int kFullWidth = [] {
  int width = kAuxBaseComps;
#define POPS_AUX_WIDTH(name, index) width = (index) + 1 > width ? (index) + 1 : width;
  POPS_AUX_FIELDS(POPS_AUX_WIDTH)
#undef POPS_AUX_WIDTH
  return width;
}();

Real aux_member(const Aux& auxiliary, int component) {
  Real value = Real(0);
#define POPS_AUX_GET(name, index) \
  if (component == (index))       \
    value = auxiliary.name;
  POPS_AUX_FIELDS(POPS_AUX_GET)
#undef POPS_AUX_GET
  return value;
}

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

template <int Dim>
void check_device_and_host_marshaling() {
  const Box<Dim> box = Box<Dim>::from_extents(unit_extent<Dim>());
  Fab<Dim> field(box, kFullWidth);
  seed_components(field, Real(100));
  const auto view = static_cast<const Fab<Dim>&>(field).view();
  const Aux device = load_aux<kFullWidth>(view, Index<Dim>{});
  EXPECT_EQ(device.phi, Real(100));
  EXPECT_EQ(device.grad_x, Real(101));
  EXPECT_EQ(device.grad_y, Real(102));
#define POPS_AUX_CHECK_READ(name, index) EXPECT_EQ(aux_member(device, index), Real(100 + (index)));
  POPS_AUX_FIELDS(POPS_AUX_CHECK_READ)
#undef POPS_AUX_CHECK_READ

  const std::size_t cell_count = 1;
  const std::size_t cell = 0;
  std::vector<double> marshaled(static_cast<std::size_t>(kFullWidth));
  for (int component = 0; component < kFullWidth; ++component)
    marshaled[static_cast<std::size_t>(component) * cell_count + cell] = 100 + component;
  Aux host{};
  host.phi = marshaled[cell];
  host.grad_x = marshaled[cell_count + cell];
  host.grad_y = marshaled[2 * cell_count + cell];
#define POPS_AUX_MARSHAL(name, index)                 \
  if (marshaled.size() >= ((index) + 1) * cell_count) \
    host.name = marshaled[(index) * cell_count + cell];
  POPS_AUX_FIELDS(POPS_AUX_MARSHAL)
#undef POPS_AUX_MARSHAL
  EXPECT_EQ(host.phi, device.phi);
  EXPECT_EQ(host.grad_x, device.grad_x);
  EXPECT_EQ(host.grad_y, device.grad_y);
#define POPS_AUX_CHECK_EQUAL(name, index) EXPECT_EQ(host.name, device.name);
  POPS_AUX_FIELDS(POPS_AUX_CHECK_EQUAL)
#undef POPS_AUX_CHECK_EQUAL
}

template <int Dim>
void check_base_width_ignores_extra_fields() {
  const Box<Dim> box = Box<Dim>::from_extents(unit_extent<Dim>());
  Fab<Dim> field(box, kFullWidth);
  seed_components(field, Real(999));
  const auto view = static_cast<const Fab<Dim>&>(field).view();
  const Aux loaded = load_aux<kAuxBaseComps>(view, Index<Dim>{});
  EXPECT_EQ(loaded.phi, Real(999));
  EXPECT_EQ(loaded.grad_x, Real(1000));
  EXPECT_EQ(loaded.grad_y, Real(1001));
#define POPS_AUX_CHECK_ZERO(name, index) EXPECT_EQ(aux_member(loaded, index), Real(0));
  POPS_AUX_FIELDS(POPS_AUX_CHECK_ZERO)
#undef POPS_AUX_CHECK_ZERO
}

}  // namespace

TEST(AuxSingleSource, RankedDeviceAndHostMarshalingAgree) {
  static_assert(kNExtra >= 1);
  static_assert(kFullWidth >= kAuxBaseComps + 1);
  check_device_and_host_marshaling<1>();
  check_device_and_host_marshaling<2>();
  check_device_and_host_marshaling<3>();
}

TEST(AuxSingleSource, RankedBaseWidthIgnoresExtraFields) {
  check_base_width_ignores_extra_fields<1>();
  check_base_width_ignores_extra_fields<2>();
  check_base_width_ignores_extra_fields<3>();
}
