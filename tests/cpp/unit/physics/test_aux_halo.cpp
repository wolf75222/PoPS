// Exact-ranked auxiliary-provider boundary qualification.
//
// Boundary policy belongs to an owner-qualified ComponentKey in the sealed provider registry.
// Native consumers receive a compact ProviderStorageView and ProviderValues pack; neither a
// physical-name convention nor a process-global auxiliary component number crosses that seam.

#include <gtest/gtest.h>

#include <pops/core/state/state.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/layout/rank_space.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/system/auxiliary_ghost_fill.hpp>
#include <pops/runtime/system/exact_field_marshaling.hpp>
#include <pops/runtime/system/provider_storage_binding.hpp>

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace {

using pops::BoundarySide;
using pops::BoundaryTopology;
using pops::Box;
using pops::Extent;
using pops::Face;
using pops::Geometry;
using pops::Index;
using pops::MultiFab;
using pops::PhysicalBoundaryConditions;
using pops::PhysicalBoundaryFace;
using pops::PhysicalBoundaryKind;
using pops::ProviderValues;
using pops::Real;
using pops::RealVector;
using pops::mesh::BoxArray;
using pops::mesh::Distribution;
using pops::mesh::RankSpace;
using pops::runtime::system::AuxiliaryBoundaryPolicy;
using pops::runtime::system::AuxiliaryComponentContract;
using pops::runtime::system::AuxiliaryComponentKey;
using pops::runtime::system::AuxiliaryConsumerProviderPlan;
using pops::runtime::system::AuxiliaryDependency;
using pops::runtime::system::AuxiliaryEvaluationEvent;
using pops::runtime::system::AuxiliaryEvaluationPolicy;
using pops::runtime::system::AuxiliaryFreshness;
using pops::runtime::system::AuxiliaryOutput;
using pops::runtime::system::AuxiliaryProviderKind;
using pops::runtime::system::AuxiliaryStorageGroups;
using pops::runtime::system::AuxiliaryStorageShape;
using pops::runtime::system::ExactAuxiliaryRegistry;
using pops::runtime::system::PreparedAuxiliaryProvider;

template <int Dim>
struct ExactDomain {
  Box<Dim> domain;
  BoxArray<Dim> layout;
  Distribution<Dim> distribution;
  Geometry<Dim> geometry;
  Extent<Dim> ghosts;
};

template <int Dim>
ExactDomain<Dim> exact_domain(int cells) {
  Index<Dim> lower{};
  Index<Dim> upper{};
  Extent<Dim> rank_extent{};
  Extent<Dim> ghosts{};
  RealVector<Dim> physical_lower{};
  RealVector<Dim> physical_upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    upper[axis] = cells - 1;
    rank_extent[axis] = 1;
    ghosts[axis] = 1;
    physical_upper[axis] = Real(1);
  }
  const Box<Dim> domain{lower, upper};
  const BoxArray<Dim> layout(std::vector<Box<Dim>>{domain});
  const auto distribution =
      Distribution<Dim>::replicated(layout, RankSpace<Dim>(Index<Dim>{}, rank_extent));
  return {domain, layout, distribution,
          Geometry<Dim>::from_bounds(domain, physical_lower, physical_upper), ghosts};
}

inline AuxiliaryComponentContract scalar_contract() {
  return {"cell-average", "cell", "unitless", "provider-field", "scalar"};
}

template <int Dim>
AuxiliaryStorageShape<Dim> scalar_shape() {
  AuxiliaryStorageShape<Dim> result;
  for (int axis = 0; axis < Dim; ++axis)
    result.halo[axis] = 1;
  return result;
}

template <int Dim>
AuxiliaryOutput<Dim> output(std::string component, AuxiliaryBoundaryPolicy boundary = {}) {
  return {AuxiliaryComponentKey{"test.aux-halo", "auxiliary", "qualified-values",
                                std::move(component)},
          scalar_contract(), scalar_shape<Dim>(), std::move(boundary)};
}

template <int Dim>
AuxiliaryDependency<Dim> dependency(const AuxiliaryOutput<Dim>& value) {
  return {value.key, value.contract, value.shape};
}

template <int Dim>
AuxiliaryStorageGroups<Dim> allocate_groups(const ExactDomain<Dim>& mesh,
                                            const ExactAuxiliaryRegistry<Dim>& registry) {
  AuxiliaryStorageGroups<Dim> groups;
  for (const auto& group : registry.storage_groups()) {
    Extent<Dim> ghosts{};
    for (int axis = 0; axis < Dim; ++axis)
      ghosts[axis] = group.shape.halo[axis];
    groups.groups.emplace(
        group.identity,
        MultiFab<Dim>(mesh.layout, mesh.distribution, Index<Dim>{},
                      static_cast<int>(group.component_count), ghosts));
  }
  return groups;
}

template <int Dim>
void seed_provider_values(AuxiliaryStorageGroups<Dim>& groups, int cells) {
  for (auto& [_, field] : groups.groups) {
    ASSERT_EQ(field.local_size(), 1U);
    auto& fab = field.fab(0);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    pops::runtime::system::marshaling::for_each_host_index(
        fab.box(), [&](const Index<Dim>& index, std::size_t) {
          Real ordinal = Real(0);
          Real stride = Real(1);
          for (int axis = 0; axis < Dim; ++axis) {
            ordinal += stride * Real(index[axis]);
            stride *= Real(cells);
          }
          for (int component = 0; component < field.ncomp(); ++component)
            host(pops::runtime::system::marshaling::storage_ordinal(fab, index, component)) =
                Real(1000 * component) + ordinal;
        });
    fab.copy_from_host(host);
  }
}

template <int Dim>
Real host_value(const MultiFab<Dim>& field, const Index<Dim>& index, int component) {
  const auto& fab = field.fab(0);
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  return host(pops::runtime::system::marshaling::storage_ordinal(fab, index, component));
}

template <int Dim>
void verifies_component_qualified_override() {
  constexpr int cells = 4;
  const auto mesh = exact_domain<Dim>(cells);
  const auto inherited = output<Dim>("inherited");
  const auto prescribed = output<Dim>(
      "prescribed", {AuxiliaryBoundaryPolicy::Kind::dirichlet, Real(7)});

  ExactAuxiliaryRegistry<Dim> registry;
  registry.add(PreparedAuxiliaryProvider<Dim>{
      "test.aux-halo.inputs", AuxiliaryProviderKind::input,
      AuxiliaryEvaluationPolicy{AuxiliaryEvaluationEvent::initialization,
                                AuxiliaryFreshness::once},
      {inherited, prescribed}, {}});
  // Consumer order deliberately differs from canonical storage order.  The ComponentKeys, not raw
  // group components, must resolve prescribed -> slot 0 and inherited -> slot 1.
  registry.add_consumer_plan(AuxiliaryConsumerProviderPlan<Dim>{
      "test.aux-halo.consumer", {{dependency(prescribed), 0}, {dependency(inherited), 1}}});
  registry.seal();

  auto groups = allocate_groups(mesh, registry);
  ASSERT_EQ(groups.groups.size(), 1U);
  seed_provider_values(groups, cells);
  std::array<bool, Dim> periodic{};
  periodic.fill(false);
  pops::runtime::system::refresh_auxiliary_group_ghosts(
      groups, registry, mesh.domain, mesh.geometry,
      BoundaryTopology<Dim>::axis_periodic(periodic));

  const auto& plan = registry.consumer_plan("test.aux-halo.consumer");
  const auto view = pops::runtime::system::bind_provider_storage_view<Dim, 2>(&plan, &groups, 0);
  for (int axis = 0; axis < Dim; ++axis) {
    Index<Dim> ghost{};
    ghost[axis] = -1;
    const ProviderValues<2> values = pops::load_provider_values<2>(view, ghost);
    const auto inherited_address = registry.address_of(inherited.key);
    const auto prescribed_address = registry.address_of(prescribed.key);
    const MultiFab<Dim>* group = groups.find(inherited_address.group);
    ASSERT_NE(group, nullptr);
    EXPECT_EQ(values[1], host_value(*group, ghost, static_cast<int>(inherited_address.component)));
    EXPECT_EQ(values[0], host_value(*group, ghost, static_cast<int>(prescribed_address.component)));
    EXPECT_EQ(values[1], Real(1000 * inherited_address.component));
    EXPECT_EQ(values[0], Real(14) - Real(1000 * prescribed_address.component));
    EXPECT_NE(values[0], values[1]);
  }
}

TEST(AuxHalo, ComponentQualifiedProviderBoundaryIsExactInOneTwoAndThreeDimensions) {
  verifies_component_qualified_override<1>();
  verifies_component_qualified_override<2>();
  verifies_component_qualified_override<3>();
}

template <int Dim>
void verifies_periodic_topology_has_authority_over_provider_policy() {
  constexpr int cells = 4;
  const auto mesh = exact_domain<Dim>(cells);
  const auto prescribed = output<Dim>(
      "periodic-prescribed", {AuxiliaryBoundaryPolicy::Kind::dirichlet, Real(7)});
  ExactAuxiliaryRegistry<Dim> registry;
  registry.add(PreparedAuxiliaryProvider<Dim>{
      "test.aux-halo.periodic-input", AuxiliaryProviderKind::input,
      {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once}, {prescribed}, {}});
  registry.seal();
  auto groups = allocate_groups(mesh, registry);
  seed_provider_values(groups, cells);
  std::array<bool, Dim> periodic{};
  periodic.fill(true);
  pops::runtime::system::refresh_auxiliary_group_ghosts(
      groups, registry, mesh.domain, mesh.geometry,
      BoundaryTopology<Dim>::axis_periodic(periodic));

  const auto address = registry.address_of(prescribed.key);
  const MultiFab<Dim>* group = groups.find(address.group);
  ASSERT_NE(group, nullptr);
  for (int axis = 0; axis < Dim; ++axis) {
    Index<Dim> ghost{};
    ghost[axis] = -1;
    Real stride = Real(1);
    for (int prior = 0; prior < axis; ++prior)
      stride *= Real(cells);
    EXPECT_EQ(host_value(*group, ghost, static_cast<int>(address.component)),
              Real(1000 * address.component) + Real(cells - 1) * stride);
  }
}

TEST(AuxHalo, PeriodicTopologyRemainsPeriodicForEveryProviderComponentAndDimension) {
  verifies_periodic_topology_has_authority_over_provider_policy<1>();
  verifies_periodic_topology_has_authority_over_provider_policy<2>();
  verifies_periodic_topology_has_authority_over_provider_policy<3>();
}

template <int Dim>
std::size_t boundary_schedule_budget() {
  std::size_t entries = 1;
  for (int axis = 0; axis < Dim; ++axis)
    entries *= 3;
  return entries - 1;
}

template <int Dim>
PhysicalBoundaryConditions<Dim> physical_conditions(PhysicalBoundaryKind kind, Real value,
                                                    Real alpha = Real(0),
                                                    Real beta = Real(1)) {
  std::array<bool, Dim> periodic{};
  periodic.fill(false);
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    spacing[axis] = Real(0.25);
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper})
      faces[static_cast<std::size_t>(Face<Dim>{axis, side}.ordinal())] =
          {kind, value, alpha, beta};
  }
  return {BoundaryTopology<Dim>::axis_periodic(periodic), faces, spacing};
}

template <int Dim>
void verifies_component_range_is_exact() {
  const auto mesh = exact_domain<Dim>(4);
  MultiFab<Dim> field(mesh.layout, mesh.distribution, Index<Dim>{}, 2, mesh.ghosts);
  const auto prepared = pops::prepare_physical_boundary(
      mesh.domain, mesh.ghosts,
      physical_conditions<Dim>(PhysicalBoundaryKind::constant_extrapolation, Real(0)),
      pops::BoundaryScheduleBudget{boundary_schedule_budget<Dim>()});
  EXPECT_THROW(pops::fill_physical_boundary(field, prepared, -1, 1), std::out_of_range);
  EXPECT_THROW(pops::fill_physical_boundary(field, prepared, 2, 1), std::out_of_range);
  EXPECT_THROW(pops::fill_physical_boundary(field, prepared, 1, 2), std::out_of_range);
}

TEST(AuxHalo, InvalidConsumerComponentRangeIsRejectedInEveryDimension) {
  verifies_component_range_is_exact<1>();
  verifies_component_range_is_exact<2>();
  verifies_component_range_is_exact<3>();
}

template <int Dim>
void verifies_outward_normal_affine_laws() {
  const auto mesh = exact_domain<Dim>(4);
  MultiFab<Dim> field(mesh.layout, mesh.distribution, Index<Dim>{}, 1, mesh.ghosts);
  field.set_val(Real(2));
  const auto neumann = pops::prepare_physical_boundary(
      mesh.domain, mesh.ghosts,
      physical_conditions<Dim>(PhysicalBoundaryKind::neumann, Real(3)),
      pops::BoundaryScheduleBudget{boundary_schedule_budget<Dim>()});
  pops::fill_physical_boundary(field, neumann);
  for (int axis = 0; axis < Dim; ++axis) {
    Index<Dim> low{};
    Index<Dim> high{};
    low[axis] = -1;
    high[axis] = 4;
    EXPECT_NEAR(host_value(field, low, 0), Real(2.75), 1e-12);
    EXPECT_NEAR(host_value(field, high, 0), Real(2.75), 1e-12);
  }

  field.set_val(Real(2));
  const auto robin = pops::prepare_physical_boundary(
      mesh.domain, mesh.ghosts,
      physical_conditions<Dim>(PhysicalBoundaryKind::robin, Real(5), Real(2), Real(0.5)),
      pops::BoundaryScheduleBudget{boundary_schedule_budget<Dim>()});
  pops::fill_physical_boundary(field, robin);
  for (int axis = 0; axis < Dim; ++axis) {
    Index<Dim> low{};
    Index<Dim> high{};
    low[axis] = -1;
    high[axis] = 4;
    EXPECT_NEAR(host_value(field, low, 0), Real(7.0 / 3.0), 1e-12);
    EXPECT_NEAR(host_value(field, high, 0), Real(7.0 / 3.0), 1e-12);
  }
}

TEST(AuxHalo, RobinAndNeumannUseTheOutwardNormalInEveryDimension) {
  verifies_outward_normal_affine_laws<1>();
  verifies_outward_normal_affine_laws<2>();
  verifies_outward_normal_affine_laws<3>();
}

}  // namespace
