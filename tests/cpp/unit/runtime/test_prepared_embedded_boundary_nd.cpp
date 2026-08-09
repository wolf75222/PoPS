/// @file
/// @brief Exact 1D/2D/3D and transactional proofs for prepared EB geometry.

#include <gtest/gtest.h>

#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/runtime/system/prepared_embedded_boundary.hpp>

#include <cmath>
#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace {

template <int Dim>
struct SumField {
  pops::FieldView<const pops::Real, Dim> view;

  POPS_HD pops::Real operator()(const pops::Index<Dim>& index) const { return view(index); }
};

template <int Dim>
pops::Extent<Dim> filled_extent(int value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::RealVector<Dim> filled_real(double value) {
  pops::RealVector<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = static_cast<pops::Real>(value);
  return result;
}

template <int Dim>
struct Fixture {
  pops::Box<Dim> domain = pops::Box<Dim>::from_extents(filled_extent<Dim>(8));
  pops::Geometry<Dim> geometry =
      pops::Geometry<Dim>::from_bounds(domain, filled_real<Dim>(0.0), filled_real<Dim>(1.0));
  pops::mesh::BoxArray<Dim> layout{std::vector<pops::Box<Dim>>{domain}};
  pops::mesh::RankSpace<Dim> rank_space{pops::Index<Dim>{}, filled_extent<Dim>(1)};
  pops::mesh::Distribution<Dim> distribution =
      pops::mesh::Distribution<Dim>::replicated(layout, rank_space);
  pops::MultiFab<Dim> prototype{layout, distribution, pops::Index<Dim>{}, 1, pops::Extent<Dim>{}};
  pops::ExecutionLane lane = pops::ExecutionLane::world("test/prepared-eb");
};

template <int Dim>
void prove_staircase() {
  auto world = pops::ExecutionLane::world("test/prepared-eb-serial-preflight");
  if (world.size() != 1)
    GTEST_SKIP() << "serial exact-rank fixture";
  Fixture<Dim> fixture;
  const auto prepared = pops::runtime::system::prepare_embedded_boundary_geometry_collectively(
      {"x", "constant", "sub"}, {0.0, 0.5, 0.0}, fixture.geometry,
      pops::BoundaryTopology<Dim>::physical(), fixture.prototype,
      pops::runtime::system::PreparedEmbeddedBoundaryMode::staircase, pops::EbThresholds{}, 1,
      fixture.lane);
  ASSERT_TRUE(prepared);
  EXPECT_EQ(prepared->mode(), pops::runtime::system::PreparedEmbeddedBoundaryMode::staircase);
  EXPECT_EQ(prepared->generation(), 1U);
  EXPECT_EQ(prepared->digest().size(),
            std::string("pops.prepared-eb-geometry.v1:sha256:").size() + 64U);
  EXPECT_EQ(prepared->phi().layout(), fixture.layout);
  EXPECT_EQ(prepared->active_mask().ghosts(), filled_extent<Dim>(1));
  EXPECT_EQ(prepared->volume_fraction().ghosts(), pops::Extent<Dim>{});

  pops::sync_host();
  const auto phi = prepared->phi().fab(0).view();
  const auto mask = prepared->active_mask().fab(0).view();
  const auto kappa = prepared->volume_fraction().fab(0).view();
  const auto inverse = prepared->inverse_volume_fraction().fab(0).view();
  pops::Index<Dim> active{};
  pops::Index<Dim> inactive{};
  for (int axis = 0; axis < Dim; ++axis) {
    active[axis] = 2;
    inactive[axis] = 6;
  }
  EXPECT_LT(phi(active), pops::Real(0));
  EXPECT_EQ(mask(active), pops::Real(1));
  EXPECT_GT(kappa(active), pops::Real(0));
  EXPECT_GE(inverse(active), pops::Real(1));
  EXPECT_GT(phi(inactive), pops::Real(0));
  EXPECT_EQ(mask(inactive), pops::Real(0));
  EXPECT_EQ(kappa(inactive), pops::Real(0));
  EXPECT_EQ(inverse(inactive), pops::Real(0));

  double active_count = 0.0;
  for (std::size_t local = 0; local < prepared->active_mask().local_size(); ++local)
    active_count +=
        pops::for_each_cell_reduce_sum(prepared->active_mask().box(local),
                                       SumField<Dim>{prepared->active_mask().fab(local).view()});
  std::size_t expected = 4;
  for (int axis = 1; axis < Dim; ++axis)
    expected *= 8;
  EXPECT_EQ(active_count, static_cast<double>(expected));
}

TEST(PreparedEmbeddedBoundaryND, StaircaseIsExactInOneDimension) {
  prove_staircase<1>();
}
TEST(PreparedEmbeddedBoundaryND, StaircaseIsExactInTwoDimensions) {
  prove_staircase<2>();
}
TEST(PreparedEmbeddedBoundaryND, StaircaseIsExactInThreeDimensions) {
  prove_staircase<3>();
}

template <int Dim>
void prove_periodic_halo() {
  auto world = pops::ExecutionLane::world("test/prepared-eb-serial-preflight");
  if (world.size() != 1)
    GTEST_SKIP() << "serial exact-rank fixture";
  Fixture<Dim> fixture;
  std::array<bool, Dim> periodic{};
  periodic[0] = true;
  const auto prepared = pops::runtime::system::prepare_embedded_boundary_geometry_collectively(
      {"x", "constant", "sub"}, {0.0, 0.25, 0.0}, fixture.geometry,
      pops::BoundaryTopology<Dim>::axis_periodic(periodic), fixture.prototype,
      pops::runtime::system::PreparedEmbeddedBoundaryMode::cut_cell, pops::EbThresholds{}, 2,
      fixture.lane);
  pops::sync_host();
  const auto phi = prepared->phi().fab(0).view();
  pops::Index<Dim> low_ghost{};
  pops::Index<Dim> high_valid{};
  pops::Index<Dim> high_ghost{};
  pops::Index<Dim> low_valid{};
  for (int axis = 0; axis < Dim; ++axis) {
    low_ghost[axis] = 3;
    high_valid[axis] = 3;
    high_ghost[axis] = 3;
    low_valid[axis] = 3;
  }
  low_ghost[0] = -1;
  high_valid[0] = 7;
  high_ghost[0] = 8;
  low_valid[0] = 0;
  EXPECT_EQ(phi(low_ghost), phi(high_valid));
  EXPECT_EQ(phi(high_ghost), phi(low_valid));
}

TEST(PreparedEmbeddedBoundaryND, PeriodicHaloUsesExactTopologyInOneDimension) {
  prove_periodic_halo<1>();
}
TEST(PreparedEmbeddedBoundaryND, PeriodicHaloUsesExactTopologyInTwoDimensions) {
  prove_periodic_halo<2>();
}
TEST(PreparedEmbeddedBoundaryND, PeriodicHaloUsesExactTopologyInThreeDimensions) {
  prove_periodic_halo<3>();
}

TEST(PreparedEmbeddedBoundaryND, NonFiniteReplacementRollsBackAcceptedOwner) {
  auto world = pops::ExecutionLane::world("test/prepared-eb-serial-preflight");
  if (world.size() != 1)
    GTEST_SKIP() << "serial exact-rank fixture";
  Fixture<2> fixture;
  std::shared_ptr<const pops::runtime::system::PreparedEmbeddedBoundaryGeometry<2>> accepted;
  pops::runtime::system::replace_prepared_embedded_boundary_geometry_collectively(
      accepted, {"x", "constant", "sub"}, {0.0, 0.5, 0.0}, fixture.geometry,
      pops::BoundaryTopology<2>::physical(), fixture.prototype,
      pops::runtime::system::PreparedEmbeddedBoundaryMode::staircase, pops::EbThresholds{}, 3,
      fixture.lane);
  ASSERT_TRUE(accepted);
  const auto* original = accepted.get();
  const std::string digest = accepted->digest();
  EXPECT_THROW(pops::runtime::system::replace_prepared_embedded_boundary_geometry_collectively(
                   accepted, {"x", "x", "sub", "constant", "div"}, {0.0, 0.0, 0.0, 0.0, 0.0},
                   fixture.geometry, pops::BoundaryTopology<2>::physical(), fixture.prototype,
                   pops::runtime::system::PreparedEmbeddedBoundaryMode::cut_cell,
                   pops::EbThresholds{}, 4, fixture.lane),
               std::domain_error);
  EXPECT_EQ(accepted.get(), original);
  EXPECT_EQ(accepted->digest(), digest);
}

TEST(PreparedEmbeddedBoundaryND, RankDivergentRequestFailsBeforePublication) {
  auto lane = pops::ExecutionLane::world("test/prepared-eb-mismatch");
  if (lane.size() < 2)
    GTEST_SKIP() << "requires at least two MPI ranks";

  constexpr int Dim = 1;
  const auto domain = pops::Box<Dim>::from_extents(filled_extent<Dim>(8));
  const auto geometry =
      pops::Geometry<Dim>::from_bounds(domain, filled_real<Dim>(0.0), filled_real<Dim>(1.0));
  const pops::mesh::BoxArray<Dim> layout{std::vector<pops::Box<Dim>>{domain}};
  pops::Extent<Dim> rank_extent{};
  rank_extent[0] = lane.size();
  const pops::mesh::RankSpace<Dim> ranks{pops::Index<Dim>{}, rank_extent};
  std::vector<pops::Index<Dim>> owners{pops::Index<Dim>{0}};
  const auto distribution = pops::mesh::Distribution<Dim>::partitioned(layout, ranks, owners);
  pops::Index<Dim> local_rank{};
  local_rank[0] = lane.rank();
  const pops::MultiFab<Dim> prototype{layout, distribution, local_rank, 1, pops::Extent<Dim>{}};
  const double split = lane.rank() == 0 ? 0.4 : 0.6;
  EXPECT_THROW((void)pops::runtime::system::prepare_embedded_boundary_geometry_collectively(
                   {"x", "constant", "sub"}, {0.0, split, 0.0}, geometry,
                   pops::BoundaryTopology<Dim>::physical(), prototype,
                   pops::runtime::system::PreparedEmbeddedBoundaryMode::staircase,
                   pops::EbThresholds{}, 5, lane),
               std::runtime_error);
}

TEST(PreparedEmbeddedBoundaryND, DistributedPeriodicHaloUsesOwningExecutionLane) {
  constexpr int Dim = 1;
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test/prepared-eb-periodic-mpi");
  if (lane.size() != 2)
    GTEST_SKIP() << "requires exactly two MPI ranks";

  const auto domain = pops::Box<Dim>::from_extents(filled_extent<Dim>(8));
  const auto geometry =
      pops::Geometry<Dim>::from_bounds(domain, filled_real<Dim>(0.0), filled_real<Dim>(1.0));
  pops::Box<Dim> left = domain;
  pops::Box<Dim> right = domain;
  left.hi[0] = 3;
  right.lo[0] = 4;
  const pops::mesh::BoxArray<Dim> layout{std::vector<pops::Box<Dim>>{left, right}};
  const pops::mesh::RankSpace<Dim> ranks{pops::Index<Dim>{}, filled_extent<Dim>(2)};
  const auto distribution = pops::mesh::Distribution<Dim>::partitioned(
      layout, ranks, {pops::Index<Dim>{0}, pops::Index<Dim>{1}});
  const pops::Index<Dim> local_rank{lane.rank()};
  const pops::MultiFab<Dim> prototype{layout, distribution, local_rank, 1, pops::Extent<Dim>{}};
  std::array<bool, Dim> periodic{true};
  const auto prepared = pops::runtime::system::prepare_embedded_boundary_geometry_collectively(
      {"x", "constant", "sub"}, {0.0, 0.25, 0.0}, geometry,
      pops::BoundaryTopology<Dim>::axis_periodic(periodic), prototype,
      pops::runtime::system::PreparedEmbeddedBoundaryMode::cut_cell, pops::EbThresholds{}, 6, lane);

  ASSERT_EQ(prepared->phi().local_size(), 1U);
  pops::sync_host();
  const auto phi = prepared->phi().fab(0).view();
  if (lane.rank() == 0) {
    EXPECT_DOUBLE_EQ(phi(pops::Index<Dim>{-1}), pops::Real(0.6875));
    EXPECT_DOUBLE_EQ(phi(pops::Index<Dim>{4}), pops::Real(0.3125));
  } else {
    EXPECT_DOUBLE_EQ(phi(pops::Index<Dim>{3}), pops::Real(0.1875));
    EXPECT_DOUBLE_EQ(phi(pops::Index<Dim>{8}), pops::Real(-0.1875));
  }
}

}  // namespace
