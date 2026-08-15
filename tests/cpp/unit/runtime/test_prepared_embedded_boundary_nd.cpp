/// @file
/// @brief Exact 1D/2D/3D and transactional proofs for prepared EB geometry.

#include <gtest/gtest.h>

#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/coordinate_map.hpp>
#include <pops/mesh/geometry/prepared_metric_provider.hpp>
#include <pops/numerics/spatial/embedded_boundary/cut_geometry.hpp>
#include <pops/numerics/spatial/embedded_boundary/operator.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/system/prepared_embedded_boundary.hpp>

#include <cmath>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

template <int Dim>
struct SumField {
  pops::FieldView<const pops::Real, Dim> view;

  POPS_HD pops::Real operator()(const pops::Index<Dim>& index) const { return view(index); }
};

template <int Dim>
struct FillTransportState {
  pops::FieldView<pops::Real, Dim> state;

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    pops::Real value = pops::Real(1);
    for (int axis = 0; axis < Dim; ++axis)
      value += pops::Real(0.1 * (axis + 1)) * static_cast<pops::Real>(cell[axis]);
    state(cell) = value;
  }
};

template <int Dim>
struct CountCutMetricCells {
  pops::FieldView<const pops::Real, Dim> inverse;

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    return inverse(cell) > pops::Real(1) ? pops::Real(1) : pops::Real(0);
  }
};

template <int Dim>
struct CountNonFiniteValues {
  pops::FieldView<const pops::Real, Dim> values;

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    return Kokkos::isfinite(values(cell)) ? pops::Real(0) : pops::Real(1);
  }
};

template <int Dim>
struct MaximumMaskedResidual {
  pops::FieldView<const pops::Real, Dim> residual;
  pops::FieldView<const pops::Real, Dim> active;
  bool select_active = false;

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    if ((active(cell) >= pops::Real(0.5)) != select_active)
      return pops::Real(0);
    return Kokkos::abs(residual(cell));
  }
};

template <int Dim>
struct PoisonActiveInverseVolume {
  pops::FieldView<pops::Real, Dim> inverse;
  pops::FieldView<const pops::Real, Dim> active;

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    if (active(cell) >= pops::Real(0.5))
      inverse(cell) = std::numeric_limits<pops::Real>::quiet_NaN();
  }
};

template <int Dim>
struct MaximumSentinelDifference {
  pops::FieldView<const pops::Real, Dim> values;
  pops::Real sentinel;

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    return Kokkos::abs(values(cell) - sentinel);
  }
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
void prove_ranked_cut_geometry() {
  pops::RealVector<Dim> lower_samples{};
  pops::RealVector<Dim> upper_samples{};
  pops::RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower_samples[axis] = pops::Real(-0.3);
    upper_samples[axis] = pops::Real(-0.3);
    spacing[axis] = pops::Real(0.2 * (axis + 1));
  }
  for (int cut_axis = 0; cut_axis < Dim; ++cut_axis) {
    auto axis_samples = upper_samples;
    axis_samples[cut_axis] = pops::Real(0.1);
    const auto cut = pops::nd::cut_cell_fractions_from_samples<Dim>(pops::Real(-0.1), lower_samples,
                                                                    axis_samples);
    static_assert(decltype(cut)::dimension == Dim);
    for (int axis = 0; axis < Dim; ++axis) {
      EXPECT_DOUBLE_EQ(cut.lower[axis], pops::Real(1));
      EXPECT_DOUBLE_EQ(cut.upper[axis], axis == cut_axis ? pops::Real(0.5) : pops::Real(1));
    }
    EXPECT_DOUBLE_EQ(cut.volume_fraction, pops::Real(0.75));

    const auto stencil = pops::nd::shortley_weller_stencil(cut, spacing);
    pops::Real expected_diagonal = pops::Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      const pops::Real lower_distance = cut.lower[axis] * spacing[axis];
      const pops::Real upper_distance = cut.upper[axis] * spacing[axis];
      const pops::Real span = lower_distance + upper_distance;
      EXPECT_DOUBLE_EQ(stencil.lower[axis], pops::Real(2) / (lower_distance * span));
      EXPECT_DOUBLE_EQ(stencil.upper[axis], pops::Real(2) / (upper_distance * span));
      expected_diagonal += pops::Real(2) / (lower_distance * upper_distance);
    }
    EXPECT_DOUBLE_EQ(stencil.diagonal, expected_diagonal);
  }

  for (int cut_axis = 0; cut_axis < Dim; ++cut_axis) {
    auto grazing_samples = upper_samples;
    grazing_samples[cut_axis] = pops::Real(0.2);
    const auto grazing = pops::nd::cut_cell_fractions_from_samples<Dim>(
        pops::Real(-1.0e-8), lower_samples, grazing_samples);
    EXPECT_DOUBLE_EQ(grazing.upper[cut_axis], pops::kEbCutFractionFloor);
  }
}

TEST(PreparedEmbeddedBoundaryND, CutGeometryUsesOneAxisLoopInOneTwoAndThreeDimensions) {
  prove_ranked_cut_geometry<1>();
  prove_ranked_cut_geometry<2>();
  prove_ranked_cut_geometry<3>();
}

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

template <int Dim>
void prove_sparse_in_domain_ghosts_are_analytic() {
  auto world = pops::ExecutionLane::world("test/prepared-eb-sparse-serial");
  if (world.size() != 1)
    GTEST_SKIP() << "serial exact-rank fixture";

  const auto domain = pops::Box<Dim>::from_extents(filled_extent<Dim>(16));
  const auto geometry =
      pops::Geometry<Dim>::from_bounds(domain, filled_real<Dim>(0.0), filled_real<Dim>(1.0));
  pops::Box<Dim> patch{};
  for (int axis = 0; axis < Dim; ++axis) {
    patch.lo[axis] = 4;
    patch.hi[axis] = 7;
  }
  const pops::mesh::BoxArray<Dim> layout{std::vector<pops::Box<Dim>>{patch}};
  const pops::mesh::RankSpace<Dim> ranks{pops::Index<Dim>{}, filled_extent<Dim>(1)};
  const auto distribution = pops::mesh::Distribution<Dim>::replicated(layout, ranks);
  const pops::MultiFab<Dim> prototype{layout, distribution, pops::Index<Dim>{}, 1,
                                      pops::Extent<Dim>{}};
  const auto prepared = pops::runtime::system::prepare_embedded_boundary_geometry_collectively(
      {"x", "constant", "sub"}, {0.0, 0.5, 0.0}, geometry, pops::BoundaryTopology<Dim>::physical(),
      prototype, pops::runtime::system::PreparedEmbeddedBoundaryMode::staircase,
      pops::EbThresholds{}, 7, world);

  pops::sync_host();
  const auto phi = prepared->phi().fab(0).view();
  pops::Index<Dim> lower{};
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] = 4;
    upper[axis] = 7;
  }
  lower[0] = 3;
  upper[0] = 8;
  EXPECT_DOUBLE_EQ(phi(lower), pops::Real(3.5 / 16.0 - 0.5));
  EXPECT_DOUBLE_EQ(phi(upper), pops::Real(8.5 / 16.0 - 0.5));
}

TEST(PreparedEmbeddedBoundaryND, SparseInDomainGhostsAreAnalyticInOneDimension) {
  prove_sparse_in_domain_ghosts_are_analytic<1>();
}
TEST(PreparedEmbeddedBoundaryND, SparseInDomainGhostsAreAnalyticInTwoDimensions) {
  prove_sparse_in_domain_ghosts_are_analytic<2>();
}
TEST(PreparedEmbeddedBoundaryND, SparseInDomainGhostsAreAnalyticInThreeDimensions) {
  prove_sparse_in_domain_ghosts_are_analytic<3>();
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

template <int Dim>
void prove_prepared_level_set_metric_operator_composition() {
  auto lane = pops::ExecutionLane::world("test/prepared-eb-metric-operator");
  if (lane.size() != 1)
    GTEST_SKIP() << "serial exact-rank execution proof";

  Fixture<Dim> fixture;
  const auto prepared = pops::runtime::system::prepare_embedded_boundary_geometry_collectively(
      {"x", "constant", "sub"}, {0.0, 0.53, 0.0}, fixture.geometry,
      pops::BoundaryTopology<Dim>::physical(), fixture.prototype,
      pops::runtime::system::PreparedEmbeddedBoundaryMode::cut_cell, pops::EbThresholds{}, 7, lane);

  pops::RealVector<Dim> lengths{};
  pops::RealVector<Dim> velocity{};
  for (int axis = 0; axis < Dim; ++axis) {
    lengths[axis] = fixture.geometry.upper()[axis] - fixture.geometry.lower()[axis];
    velocity[axis] = axis == 0 ? pops::Real(0.25) : pops::Real(0);
  }
  const auto metric = pops::prepare_metric_provider(
      fixture.domain, pops::CartesianCoordinateMap<Dim>::make(fixture.geometry.lower(), lengths));
  const auto model = pops::nd::ScalarAdvection<Dim>::prepare(velocity);
  const auto embedded = pops::nd::prepare_embedded_boundary_operator(model, metric);

  pops::MultiFab<Dim> state(fixture.layout, fixture.distribution, pops::Index<Dim>{}, 1,
                            filled_extent<Dim>(1));
  pops::MultiFab<Dim> residual(fixture.layout, fixture.distribution, pops::Index<Dim>{}, 1,
                               pops::Extent<Dim>{});
  pops::for_each_cell(state.fab(0).grown_box(), FillTransportState<Dim>{state.fab(0).view()});
  embedded.assemble_residual(state, *prepared, residual);

  const auto active = std::as_const(prepared->active_mask().fab(0)).view();
  const auto inverse = std::as_const(prepared->inverse_volume_fraction().fab(0)).view();
  const auto result = std::as_const(residual.fab(0)).view();
  EXPECT_GT(pops::for_each_cell_reduce_sum(fixture.domain, CountCutMetricCells<Dim>{inverse}),
            pops::Real(0));
  EXPECT_EQ(pops::for_each_cell_reduce_sum(fixture.domain, CountNonFiniteValues<Dim>{result}),
            pops::Real(0));
  EXPECT_EQ(pops::for_each_cell_reduce_max(fixture.domain,
                                           MaximumMaskedResidual<Dim>{result, active, false}),
            pops::Real(0));
  EXPECT_GT(pops::for_each_cell_reduce_max(fixture.domain,
                                           MaximumMaskedResidual<Dim>{result, active, true}),
            pops::Real(0));

  pops::MultiFab<Dim> invalid_inverse(prepared->inverse_volume_fraction());
  pops::for_each_cell(fixture.domain,
                      PoisonActiveInverseVolume<Dim>{invalid_inverse.fab(0).view(), active});
  constexpr pops::Real sentinel = pops::Real(23);
  residual.set_val(sentinel);
  EXPECT_THROW(
      embedded.assemble_residual(state, prepared->active_mask(), invalid_inverse, residual),
      std::runtime_error);
  EXPECT_EQ(pops::for_each_cell_reduce_max(
                fixture.domain,
                MaximumSentinelDifference<Dim>{std::as_const(residual.fab(0)).view(), sentinel}),
            pops::Real(0));
}

TEST(PreparedEmbeddedBoundaryND,
     AnalyticLevelSetFeedsCartesianCutMetricOperatorTransactionInOneTwoAndThreeDimensions) {
  prove_prepared_level_set_metric_operator_composition<1>();
  prove_prepared_level_set_metric_operator_composition<2>();
  prove_prepared_level_set_metric_operator_composition<3>();
}

TEST(PreparedEmbeddedBoundaryND, DistributedResidualWithoutExecutionLaneFailsBeforePublication) {
  constexpr int Dim = 1;
  const auto domain = pops::Box<Dim>::from_extents(filled_extent<Dim>(4));
  const pops::mesh::BoxArray<Dim> layout{std::vector<pops::Box<Dim>>{domain}};
  const pops::mesh::RankSpace<Dim> ranks{pops::Index<Dim>{}, filled_extent<Dim>(2)};
  const auto distribution = pops::mesh::Distribution<Dim>::replicated(layout, ranks);
  pops::MultiFab<Dim> state{layout, distribution, pops::Index<Dim>{}, 1, filled_extent<Dim>(1)};
  pops::MultiFab<Dim> active{layout, distribution, pops::Index<Dim>{}, 1, filled_extent<Dim>(1)};
  pops::MultiFab<Dim> inverse{layout, distribution, pops::Index<Dim>{}, 1, pops::Extent<Dim>{}};
  pops::MultiFab<Dim> residual{layout, distribution, pops::Index<Dim>{}, 1, pops::Extent<Dim>{}};
  state.set_val(pops::Real(1));
  active.set_val(pops::Real(1));
  inverse.set_val(pops::Real(1));
  constexpr pops::Real sentinel = pops::Real(31);
  residual.set_val(sentinel);

  const auto metric = pops::prepare_metric_provider(
      domain, pops::CartesianCoordinateMap<Dim>::make(filled_real<Dim>(0), filled_real<Dim>(1)));
  const auto model = pops::nd::ScalarAdvection<Dim>::prepare(filled_real<Dim>(0.25));
  const auto embedded = pops::nd::prepare_embedded_boundary_operator(model, metric);
  EXPECT_THROW(embedded.assemble_residual(state, active, inverse, residual), std::logic_error);
  EXPECT_EQ(
      pops::for_each_cell_reduce_max(
          domain, MaximumSentinelDifference<Dim>{std::as_const(residual.fab(0)).view(), sentinel}),
      pops::Real(0));
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
