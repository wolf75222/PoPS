// ADC-369: auxiliary providers retain a component-local boundary policy through the exact-rank
// storage registry.  The prepared transport fills the authenticated halo before applying every
// resolved component's physical policy; no shared scratch carrier or legacy dimensional adapter
// participates in this path.

#include <gtest/gtest.h>

#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/system/auxiliary_ghost_fill.hpp>
#include <pops/runtime/system/exact_field_marshaling.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace pops;
using namespace pops::mesh;
using namespace pops::runtime::system;

inline AuxiliaryComponentContract scalar_contract() {
  return {"cell-average", "cell", "unitless", "aux-halo-test", "scalar"};
}

template <int Dim>
Extent<Dim> ones() {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = 1;
  return result;
}

template <int Dim>
AuxiliaryStorageShape<Dim> scalar_shape() {
  AuxiliaryStorageShape<Dim> result;
  for (int axis = 0; axis < Dim; ++axis)
    result.halo[axis] = 1;
  return result;
}

template <int Dim>
Box<Dim> domain() {
  Extent<Dim> extent{};
  for (int axis = 0; axis < Dim; ++axis)
    extent[axis] = 4;
  return Box<Dim>::from_extents(extent);
}

template <int Dim>
Geometry<Dim> geometry(const Box<Dim>& box) {
  RealVector<Dim> lower{};
  RealVector<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = Real(box.length(axis));
  return Geometry<Dim>::from_bounds(box, lower, upper);
}

template <int Dim>
struct AuxiliaryFixture {
  Box<Dim> box = domain<Dim>();
  BoxArray<Dim> layout{std::vector<Box<Dim>>{box}};
  RankSpace<Dim> ranks{Index<Dim>{}, ones<Dim>()};
  Distribution<Dim> distribution = Distribution<Dim>::replicated(layout, ranks);
  AuxiliaryOutput<Dim> shared{{"aux-halo", "input", "carrier", "base"},
                              scalar_contract(),
                              scalar_shape<Dim>(),
                              {AuxiliaryBoundaryPolicy::Kind::first_order_extrapolation, {}}};
  AuxiliaryOutput<Dim> overridden{{"aux-halo", "input", "carrier", "override"},
                                  scalar_contract(),
                                  scalar_shape<Dim>(),
                                  {AuxiliaryBoundaryPolicy::Kind::dirichlet, Real(7)}};
  ExactAuxiliaryRegistry<Dim> registry;
  AuxiliaryStorageGroups<Dim> groups;

  AuxiliaryFixture() {
    registry.add(PreparedAuxiliaryProvider<Dim>{
        "aux-halo-input",
        AuxiliaryProviderKind::input,
        {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
        {shared, overridden},
        {}});
    registry.seal();
    if (registry.storage_groups().size() != 1U)
      throw std::logic_error("auxiliary test fixture expected one resolved storage group");
    const auto& resolved = registry.storage_groups().front();
    groups.groups.emplace(resolved.identity,
                          MultiFab<Dim>(layout, distribution, Index<Dim>{},
                                        static_cast<int>(resolved.component_count), ones<Dim>()));
  }

  MultiFab<Dim>& field() { return *groups.find(registry.storage_groups().front().identity); }

  void seed() {
    const auto shared_address = registry.address_of(shared.key);
    const auto override_address = registry.address_of(overridden.key);
    auto& fab = field().fab(0);
    auto host = fab.create_host_mirror();
    marshaling::for_each_host_index(fab.box(), [&](const Index<Dim>& index, std::size_t) {
      host(marshaling::storage_ordinal(fab, index, static_cast<int>(shared_address.component))) =
          Real(index[0]);
      host(marshaling::storage_ordinal(fab, index, static_cast<int>(override_address.component))) =
          Real(500 + index[0]);
    });
    fab.copy_from_host(host);
  }

  Real value(const AuxiliaryOutput<Dim>& output, const Index<Dim>& index) const {
    const auto address = registry.address_of(output.key);
    const auto& fab = groups.find(address.group)->fab(0);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    return host(marshaling::storage_ordinal(fab, index, static_cast<int>(address.component)));
  }
};

template <int Dim>
void expect_component_local_auxiliary_halos() {
  AuxiliaryFixture<Dim> fixture;
  fixture.seed();
  refresh_auxiliary_group_ghosts(fixture.groups, fixture.registry, fixture.box,
                                 geometry(fixture.box), BoundaryTopology<Dim>::physical());

  Index<Dim> low{};
  low[0] = -1;
  EXPECT_EQ(fixture.value(fixture.shared, low), Real(0));
  EXPECT_EQ(fixture.value(fixture.overridden, low), Real(-486));
  EXPECT_NE(fixture.value(fixture.shared, low), fixture.value(fixture.overridden, low));

  std::array<bool, Dim> periodic{};
  periodic.fill(true);
  fixture.seed();
  refresh_auxiliary_group_ghosts(fixture.groups, fixture.registry, fixture.box,
                                 geometry(fixture.box),
                                 BoundaryTopology<Dim>::axis_periodic(periodic));
  EXPECT_EQ(fixture.value(fixture.shared, low), Real(3));
  EXPECT_EQ(fixture.value(fixture.overridden, low), Real(503));
}

template <int Dim>
std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> extrapolation_faces() {
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> result{};
  result.fill(PhysicalBoundaryFace{PhysicalBoundaryKind::constant_extrapolation});
  return result;
}

}  // namespace

TEST(AuxHalo, ProviderPoliciesAreComponentLocalInOneTwoAndThreeDimensions) {
  expect_component_local_auxiliary_halos<1>();
  expect_component_local_auxiliary_halos<2>();
  expect_component_local_auxiliary_halos<3>();
}

TEST(AuxHalo, InvalidPhysicalComponentRangeIsRejectedInsteadOfSilentlyClamped) {
  const Box<2> box = domain<2>();
  const BoxArray<2> layout(std::vector<Box<2>>{box});
  const auto distribution =
      Distribution<2>::replicated(layout, RankSpace<2>{Index<2>{}, Extent<2>{1, 1}});
  MultiFab<2> field(layout, distribution, Index<2>{}, 2, Extent<2>{1, 1});
  const auto prepared = prepare_physical_boundary(
      box, Extent<2>{1, 1},
      PhysicalBoundaryConditions<2>{BoundaryTopology<2>::physical(), extrapolation_faces<2>(),
                                    RealVector<2>{1, 1}},
      BoundaryScheduleBudget{16});
  EXPECT_THROW(fill_physical_boundary(field, prepared, -1, 1), std::out_of_range);
  EXPECT_THROW(fill_physical_boundary(field, prepared, 2, 1), std::out_of_range);
}

TEST(AuxHalo, RobinAndNonzeroNeumannUseOutwardNormalOnBothFaces) {
  const Box<2> box = Box<2>::from_extents(Extent<2>{4, 3});
  const BoxArray<2> layout(std::vector<Box<2>>{box});
  const auto distribution =
      Distribution<2>::replicated(layout, RankSpace<2>{Index<2>{}, Extent<2>{1, 1}});
  MultiFab<2> field(layout, distribution, Index<2>{}, 1, Extent<2>{1, 1});
  field.set_val(Real(2));

  auto faces = extrapolation_faces<2>();
  for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper})
    faces[static_cast<std::size_t>(Face<2>{0, side}.ordinal())] = {PhysicalBoundaryKind::robin,
                                                                   Real(3), Real(0), Real(1)};
  const auto prepared = prepare_physical_boundary(
      box, Extent<2>{1, 1},
      PhysicalBoundaryConditions<2>{BoundaryTopology<2>::physical(), faces, RealVector<2>{0.25, 1}},
      BoundaryScheduleBudget{16});
  fill_physical_boundary(field, prepared);

  const auto& fab = field.fab(0);
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  EXPECT_NEAR(host(marshaling::storage_ordinal(fab, Index<2>{-1, 1}, 0)), Real(2.75), 1e-12);
  EXPECT_NEAR(host(marshaling::storage_ordinal(fab, Index<2>{4, 1}, 0)), Real(2.75), 1e-12);

  field.set_val(Real(2));
  for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper})
    faces[static_cast<std::size_t>(Face<2>{0, side}.ordinal())] = {PhysicalBoundaryKind::robin,
                                                                   Real(5), Real(2), Real(0.5)};
  const auto mixed = prepare_physical_boundary(
      box, Extent<2>{1, 1},
      PhysicalBoundaryConditions<2>{BoundaryTopology<2>::physical(), faces, RealVector<2>{0.25, 1}},
      BoundaryScheduleBudget{16});
  fill_physical_boundary(field, mixed);
  host = fab.create_host_mirror();
  fab.copy_to_host(host);
  EXPECT_NEAR(host(marshaling::storage_ordinal(fab, Index<2>{-1, 1}, 0)), Real(7.0 / 3.0), 1e-12);
  EXPECT_NEAR(host(marshaling::storage_ordinal(fab, Index<2>{4, 1}, 0)), Real(7.0 / 3.0), 1e-12);
}
