#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/numerics/elliptic/amr/composite_fac_poisson.hpp>
#include <pops/parallel/comm.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using pops::BoundaryTopology;
using pops::Box;
using pops::EllipticBuildRequest;
using pops::Extent;
using pops::ExecutionLane;
using pops::Geometry;
using pops::Index;
using pops::MultiFab;
using pops::PhysicalBoundaryConditions;
using pops::PhysicalBoundaryFace;
using pops::PhysicalBoundaryKind;
using pops::Real;
using pops::RealVector;
using pops::amr::RefinementRatio;
using pops::elliptic::amr::CompositeFacBuildRequest;
using pops::elliptic::amr::CompositeFacPoisson;
using pops::elliptic::amr::CompositeFacPreparationBudget;
using pops::mesh::BoxArray;
using pops::mesh::BoxArrayValidationBudget;
using pops::mesh::Distribution;
using pops::mesh::RankSpace;

namespace {

template <int Dim>
Extent<Dim> integer_extent(std::int64_t value) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
RealVector<Dim> coordinates(Real value) {
  RealVector<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
PhysicalBoundaryConditions<Dim> constant_dirichlet(const Geometry<Dim>& geometry) {
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  faces.fill(PhysicalBoundaryFace{PhysicalBoundaryKind::dirichlet, Real(1)});
  RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  return PhysicalBoundaryConditions<Dim>{BoundaryTopology<Dim>::physical(), faces, spacing};
}

CompositeFacPreparationBudget budget() {
  CompositeFacPreparationBudget result;
  result.levels = 2;
  result.connections = 1;
  result.parent_child_patch_pairs = 16;
  result.interpolation_regions = 128;
  result.local_scratch_cells = 16'384;
  result.same_level_halo = {
      BoxArrayValidationBudget{16, 256}, 4096, 4096, 64, 16, 1'000'000, 1'000'000, 1'000'000};
  result.parent_gather = {64, 16, 1'000'000, 1'000'000, 1'000'000};
  result.fine_restriction = {64, 16, 1'000'000, 1'000'000, 1'000'000};
  return result;
}

template <int Dim>
CompositeFacBuildRequest<Dim> replicated_coarse_partitioned_fine() {
  Index<Dim> coarse_upper{};
  for (int axis = 0; axis < Dim; ++axis)
    coarse_upper[axis] = 3;
  const Box<Dim> coarse_domain{Index<Dim>{}, coarse_upper};
  const Geometry<Dim> coarse_geometry = Geometry<Dim>::from_bounds(
      coarse_domain, coordinates<Dim>(Real(0)), coordinates<Dim>(Real(1)));
  const BoxArray<Dim> coarse_layout(std::vector<Box<Dim>>{coarse_domain});

  const Geometry<Dim> fine_geometry = coarse_geometry.refine(integer_extent<Dim>(2));
  Box<Dim> fine_left{Index<Dim>{}, fine_geometry.domain().hi};
  Box<Dim> fine_right = fine_left;
  fine_left.hi[0] = 3;
  fine_right.lo[0] = 4;
  const BoxArray<Dim> fine_layout(std::vector<Box<Dim>>{fine_left, fine_right});

  Extent<Dim> rank_extents = integer_extent<Dim>(1);
  rank_extents[0] = 2;
  const RankSpace<Dim> rank_space{Index<Dim>{}, rank_extents};
  Index<Dim> local_rank{};
  local_rank[0] = pops::my_rank();
  Index<Dim> rank_zero{};
  Index<Dim> rank_one{};
  rank_one[0] = 1;
  const Distribution<Dim> coarse_distribution =
      Distribution<Dim>::replicated(coarse_layout, rank_space);
  const Distribution<Dim> fine_distribution = Distribution<Dim>::partitioned(
      fine_layout, rank_space, std::vector<Index<Dim>>{rank_zero, rank_one});
  const BoxArrayValidationBudget layout_budget{2, 1};
  std::array<int, Dim> ratio_components{};
  ratio_components.fill(2);

  EllipticBuildRequest<Dim> coarse{coarse_geometry,
                                   coarse_layout,
                                   coarse_distribution,
                                   local_rank,
                                   constant_dirichlet(coarse_geometry),
                                   Extent<Dim>{},
                                   integer_extent<Dim>(1),
                                   layout_budget};
  EllipticBuildRequest<Dim> fine{fine_geometry,
                                 fine_layout,
                                 fine_distribution,
                                 local_rank,
                                 constant_dirichlet(fine_geometry),
                                 Extent<Dim>{},
                                 integer_extent<Dim>(1),
                                 layout_budget};
  return CompositeFacBuildRequest<Dim>{
      {std::move(coarse), std::move(fine)}, {RefinementRatio<Dim>{ratio_components}}, budget()};
}

template <int Dim>
Real maximum_constant_error(const MultiFab<Dim>& field, const ExecutionLane& lane) {
  Real local_error = Real(0);
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& grown = fab.grown_box();
    const Box<Dim>& valid = fab.box();
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(valid.numPts()); ++ordinal) {
      std::size_t cell = ordinal;
      Index<Dim> index{};
      for (int axis = 0; axis < Dim; ++axis) {
        const std::size_t axis_cells = static_cast<std::size_t>(valid.length(axis));
        index[axis] = valid.lo[axis] + static_cast<int>(cell % axis_cells);
        cell /= axis_cells;
      }
      std::size_t offset = 0;
      std::size_t stride = 1;
      for (int axis = 0; axis < Dim; ++axis) {
        offset += static_cast<std::size_t>(index[axis] - grown.lo[axis]) * stride;
        stride *= static_cast<std::size_t>(grown.length(axis));
      }
      local_error = std::max(local_error, std::abs(host(offset) - Real(1)));
    }
  }
  return static_cast<Real>(pops::all_reduce_max(static_cast<double>(local_error), lane));
}

template <int Dim>
void expect_replicated_coarse_fac() {
  const ExecutionLane lane = ExecutionLane::world(
      "pops.test.composite-fac.replicated-coarse:" + std::to_string(Dim));
  pops::CompositeFacOptions options;
  options.max_iters = 40;
  options.fine_sweeps = 4;
  options.rel_tol = Real(5e-3);
  options.abs_tol = Real(1e-10);
  CompositeFacPoisson<Dim> solver(replicated_coarse_partitioned_fine<Dim>(), options, Real(1));
  EXPECT_TRUE(solver.owns_execution_lane());
  EXPECT_EQ(solver.n_levels(), 2);
  EXPECT_EQ(pops::all_reduce_min(solver.has_remote_same_level_halo() ? 1L : 0L, lane), 1L);
  EXPECT_EQ(pops::all_reduce_min(solver.has_remote_fine_restriction() ? 1L : 0L, lane), 1L);
  for (int level = 0; level < solver.n_levels(); ++level) {
    solver.rhs_level(level).set_val(Real(1));
    solver.phi_level(level).set_val(Real(0));
  }
  const pops::SolveReport report = solver.solve();
  EXPECT_TRUE(report.solved()) << report.reason << " residual=" << report.residual_norm
                               << " reference=" << report.reference_residual_norm;
  EXPECT_LT(report.residual_norm, report.reference_residual_norm);
  EXPECT_LT(maximum_constant_error(solver.phi_level(0), lane), Real(0.08));
  EXPECT_LT(maximum_constant_error(solver.phi_level(1), lane), Real(0.08));
  EXPECT_EQ(pops::all_reduce_min(static_cast<long>(report.iters), lane),
            pops::all_reduce_max(static_cast<long>(report.iters), lane));
}

int run_replicated_coarse_fac(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    EXPECT_EQ(pops::n_ranks(), 2);
    if (pops::n_ranks() == 2)
      expect_replicated_coarse_fac<pops::kNativeDimension>();
    result = ::testing::Test::HasFailure() ? 1 : 0;
  }
  pops::comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_composite_fac_replicated_coarse, SolvesReplicatedCoarsePartitionedFine) {
  EXPECT_EQ(pops::test::RunTestBody(&run_replicated_coarse_fac,
                                    "test_mpi_composite_fac_replicated_coarse"),
            0);
}
