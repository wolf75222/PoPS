#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/numerics/elliptic/amr/composite_fac_poisson.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>
#include <pops/parallel/comm.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using pops::BoundaryTopology;
using pops::Box;
using pops::EllipticBuildRequest;
using pops::Extent;
using pops::ExecutionLane;
using pops::Face;
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
Index<Dim> rank_coordinate(int rank) {
  Index<Dim> result{};
  result[0] = rank;
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
CompositeFacBuildRequest<Dim> make_request_with_ratio(
    const std::array<int, Dim>& ratio_components,
    CompositeFacPreparationBudget preparation = budget()) {
  Index<Dim> coarse_upper{};
  coarse_upper[0] = 7;
  for (int axis = 1; axis < Dim; ++axis)
    coarse_upper[axis] = 3;
  const Box<Dim> coarse_domain{Index<Dim>{}, coarse_upper};
  const Geometry<Dim> coarse_geometry = Geometry<Dim>::from_bounds(
      coarse_domain, coordinates<Dim>(Real(0)), coordinates<Dim>(Real(1)));
  Extent<Dim> ratio_extent{};
  for (int axis = 0; axis < Dim; ++axis)
    ratio_extent[axis] = ratio_components[static_cast<std::size_t>(axis)];
  const Geometry<Dim> fine_geometry = coarse_geometry.refine(ratio_extent);

  Index<Dim> coarse_left_upper = coarse_upper;
  coarse_left_upper[0] = 3;
  Index<Dim> coarse_right_lower{};
  coarse_right_lower[0] = 4;
  const BoxArray<Dim> coarse_layout(std::vector<Box<Dim>>{
      Box<Dim>{Index<Dim>{}, coarse_left_upper},
      Box<Dim>{coarse_right_lower, coarse_upper},
  });

  Index<Dim> fine_first_parent_lower{};
  Index<Dim> fine_first_parent_upper = coarse_upper;
  fine_first_parent_lower[0] = 2;
  fine_first_parent_upper[0] = 3;
  Index<Dim> fine_second_parent_lower{};
  Index<Dim> fine_second_parent_upper = coarse_upper;
  fine_second_parent_lower[0] = 4;
  fine_second_parent_upper[0] = 5;
  const BoxArray<Dim> fine_layout(std::vector<Box<Dim>>{
      pops::refine(Box<Dim>{fine_first_parent_lower, fine_first_parent_upper}, ratio_extent),
      pops::refine(Box<Dim>{fine_second_parent_lower, fine_second_parent_upper}, ratio_extent),
  });

  Extent<Dim> rank_extents = integer_extent<Dim>(1);
  rank_extents[0] = 2;
  const RankSpace<Dim> rank_space{Index<Dim>{}, rank_extents};
  const std::vector<Index<Dim>> coarse_owners{rank_coordinate<Dim>(0), rank_coordinate<Dim>(1)};
  const std::vector<Index<Dim>> fine_owners{rank_coordinate<Dim>(1), rank_coordinate<Dim>(0)};
  const Distribution<Dim> coarse_distribution =
      Distribution<Dim>::partitioned(coarse_layout, rank_space, coarse_owners);
  const Distribution<Dim> fine_distribution =
      Distribution<Dim>::partitioned(fine_layout, rank_space, fine_owners);
  const Index<Dim> local_rank = rank_coordinate<Dim>(pops::my_rank());
  const BoxArrayValidationBudget layout_budget{2, 1};
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
      {std::move(coarse), std::move(fine)}, {RefinementRatio<Dim>{ratio_components}}, preparation};
}

template <int Dim>
CompositeFacBuildRequest<Dim> make_request(CompositeFacPreparationBudget preparation = budget()) {
  std::array<int, Dim> ratio_components{};
  ratio_components.fill(2);
  return make_request_with_ratio<Dim>(ratio_components, preparation);
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
void expect_partitioned_fac() {
  const ExecutionLane lane =
      ExecutionLane::world("pops.test.composite-fac.partitioned:" + std::to_string(Dim));
  pops::CompositeFacOptions options;
  options.max_iters = 40;
  options.fine_sweeps = 4;
  options.rel_tol = Real(5e-3);
  options.abs_tol = Real(1e-10);
  options.coarse_rel_tol = Real(1e-4);
  options.coarse_abs_tol = Real(1e-10);
  options.coarse_cycles = 192;
  CompositeFacPoisson<Dim> solver(make_request<Dim>(), options, Real(1));
  EXPECT_TRUE(solver.owns_execution_lane());
  EXPECT_EQ(pops::all_reduce_min(solver.has_remote_same_level_halo() ? 1L : 0L, lane), 1L);
  EXPECT_EQ(pops::all_reduce_min(solver.has_remote_parent_gather() ? 1L : 0L, lane), 1L);
  EXPECT_EQ(pops::all_reduce_min(solver.has_remote_fine_restriction() ? 1L : 0L, lane), 1L);
  for (int level = 0; level < solver.n_levels(); ++level) {
    solver.rhs_level(level).set_val(Real(1));
    solver.phi_level(level).set_val(Real(0));
  }
  const pops::SolveReport report = solver.solve();
  EXPECT_TRUE(report.solved()) << report.reason << " residual=" << report.residual_norm
                               << " reference=" << report.reference_residual_norm;
  EXPECT_LT(report.residual_norm, report.reference_residual_norm);
  EXPECT_LT(maximum_constant_error(solver.phi_level(0), lane), Real(0.12));
  EXPECT_LT(maximum_constant_error(solver.phi_level(1), lane), Real(0.12));
  EXPECT_EQ(pops::all_reduce_min(static_cast<long>(report.iters), lane),
            pops::all_reduce_max(static_cast<long>(report.iters), lane));
}

template <int Dim>
void expect_exact_rank_ratio_prepares(const std::array<int, Dim>& ratio_components) {
  const ExecutionLane lane =
      ExecutionLane::world("pops.test.composite-fac.ratio:" + std::to_string(Dim));
  CompositeFacPoisson<Dim> solver(make_request_with_ratio<Dim>(ratio_components), {}, Real(1));
  EXPECT_EQ(solver.n_levels(), 2);
}

template <int Dim>
PhysicalBoundaryConditions<Dim> periodic_boundary(const Geometry<Dim>& geometry) {
  std::array<bool, Dim> periodic{};
  periodic.fill(true);
  RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  return PhysicalBoundaryConditions<Dim>{BoundaryTopology<Dim>::axis_periodic(periodic), {},
                                         spacing};
}

template <int Dim>
void fill_periodic_eigenmode(MultiFab<Dim>& rhs, const Geometry<Dim>& geometry) {
  constexpr Real kTwoPi = Real{6.283185307179586476925286766559005768L};
  constexpr Real kPi = Real{3.141592653589793238462643383279502884L};
  Real eigenvalue = Real(0);
  for (int axis = 0; axis < Dim; ++axis) {
    const Real angle = kPi / static_cast<Real>(geometry.domain().length(axis));
    const Real inverse = Real(1) / geometry.spacing(axis);
    eigenvalue += Real(4) * std::sin(angle) * std::sin(angle) * inverse * inverse;
  }
  for (std::size_t local = 0; local < rhs.local_size(); ++local) {
    auto& fab = rhs.fab(local);
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
      Real mode = Real(1);
      for (int axis = 0; axis < Dim; ++axis)
        mode *= std::sin(kTwoPi * geometry.cell_coordinate(axis, index[axis]));
      std::size_t offset = 0;
      std::size_t stride = 1;
      for (int axis = 0; axis < Dim; ++axis) {
        offset += static_cast<std::size_t>(index[axis] - grown.lo[axis]) * stride;
        stride *= static_cast<std::size_t>(grown.length(axis));
      }
      host(offset) = eigenvalue * mode;
    }
    fab.copy_from_host(host);
  }
}

template <int Dim>
Real maximum_mode_error(const MultiFab<Dim>& field, const Geometry<Dim>& geometry,
                        const ExecutionLane& lane) {
  constexpr Real kTwoPi = Real{6.283185307179586476925286766559005768L};
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
      Real mode = Real(1);
      for (int axis = 0; axis < Dim; ++axis)
        mode *= std::sin(kTwoPi * geometry.cell_coordinate(axis, index[axis]));
      std::size_t offset = 0;
      std::size_t stride = 1;
      for (int axis = 0; axis < Dim; ++axis) {
        offset += static_cast<std::size_t>(index[axis] - grown.lo[axis]) * stride;
        stride *= static_cast<std::size_t>(grown.length(axis));
      }
      local_error = std::max(local_error, std::abs(host(offset) - mode));
    }
  }
  return static_cast<Real>(pops::all_reduce_max(static_cast<double>(local_error), lane));
}

template <int Dim>
void expect_mg_distributed_fac() {
  const ExecutionLane lane =
      ExecutionLane::world("pops.test.composite-fac.mg-distributed:" + std::to_string(Dim));
  auto amr_request = make_request<Dim>();
  pops::elliptic::mg::CompositeFacBuildRequest<Dim> request;
  request.levels = std::move(amr_request.levels);
  request.ratios = std::move(amr_request.ratios);
  pops::CompositeFacOptions options;
  options.max_iters = 40;
  options.fine_sweeps = 4;
  options.rel_tol = Real(5e-3);
  options.abs_tol = Real(1e-10);
  options.coarse_rel_tol = Real(1e-4);
  options.coarse_abs_tol = Real(1e-10);
  options.coarse_cycles = 192;
  pops::elliptic::mg::CompositeFacPoisson<Dim> solver(std::move(request), lane, options, Real(1));
  solver.install_nullspace(
      pops::FieldNullspacePlan<Dim>{},
      std::vector<pops::PreparedVectorDistribution<Dim>>(
          static_cast<std::size_t>(solver.n_levels()),
          pops::PreparedVectorDistribution<Dim>::distributed()));
  EXPECT_EQ(pops::all_reduce_min(solver.has_remote_same_level_halo() ? 1L : 0L, lane), 1L);
  EXPECT_EQ(pops::all_reduce_min(solver.has_remote_parent_gather() ? 1L : 0L, lane), 1L);
  EXPECT_EQ(pops::all_reduce_min(solver.has_remote_fine_restriction() ? 1L : 0L, lane), 1L);
  for (int level = 0; level < solver.n_levels(); ++level) {
    solver.rhs_level(level).set_val(Real(1));
    solver.phi_level(level).set_val(Real(0));
  }
  const pops::SolveReport report = solver.solve();
  EXPECT_TRUE(report.solved()) << report.reason << " residual=" << report.residual_norm
                               << " reference=" << report.reference_residual_norm;
  EXPECT_LT(maximum_constant_error(solver.phi_level(0), lane), Real(0.12));
  EXPECT_LT(maximum_constant_error(solver.phi_level(1), lane), Real(0.12));
}

template <int Dim>
void expect_partitioned_singular_nullspace() {
  const ExecutionLane lane =
      ExecutionLane::world("pops.test.composite-fac.singular:" + std::to_string(Dim));
  auto request = make_request<Dim>();
  for (auto& level : request.levels)
    level.boundary = periodic_boundary<Dim>(level.geometry);
  pops::CompositeFacOptions options;
  options.max_iters = 40;
  options.fine_sweeps = 4;
  options.rel_tol = Real(5e-3);
  {
    CompositeFacPoisson<Dim> incompatible(request, options, Real(0));
    incompatible.rhs_level(0).set_val(Real(1));
    incompatible.rhs_level(1).set_val(Real(1));
    const pops::SolveReport report = incompatible.solve();
    EXPECT_EQ(report.status, pops::SolveStatus::kIncompatibleRhs) << report.reason;
  }
  std::vector<Geometry<Dim>> geometries;
  geometries.reserve(request.levels.size());
  for (const auto& level : request.levels)
    geometries.push_back(level.geometry);
  CompositeFacPoisson<Dim> solver(std::move(request), options, Real(0));
  EXPECT_EQ(pops::all_reduce_min(solver.has_remote_parent_gather() ? 1L : 0L, lane), 1L);
  for (int level = 0; level < solver.n_levels(); ++level)
    fill_periodic_eigenmode(solver.rhs_level(level), geometries[static_cast<std::size_t>(level)]);
  const pops::SolveReport report = solver.solve();
  EXPECT_TRUE(report.solved()) << report.reason << " residual=" << report.residual_norm;
  EXPECT_LT(maximum_mode_error(solver.phi_level(0), geometries[0], lane), Real(0.25));
  EXPECT_LT(maximum_mode_error(solver.phi_level(1), geometries[1], lane), Real(0.25));
}

void expect_collective_budget_failure() {
  const ExecutionLane lane = ExecutionLane::world("pops.test.composite-fac.budget");
  CompositeFacPreparationBudget preparation = budget();
  if (pops::my_rank() == 0)
    preparation.local_scratch_cells = 0;
  bool rejected = false;
  std::string message;
  try {
    CompositeFacPoisson<1> solver(make_request<1>(preparation), {}, Real(1));
    (void)solver;
  } catch (const std::exception& error) {
    rejected = true;
    message = error.what();
  }
  EXPECT_EQ(pops::all_reduce_min(rejected ? 1L : 0L, lane), 1L);
  EXPECT_TRUE(pops::all_ranks_agree_exact_ordered_byte_pairs(
      {{std::string_view("partitioned-fac-budget-failure"), std::string_view(message)}}, lane));
}

template <int Dim>
void install_unit_metric(CompositeFacPoisson<Dim>& solver, int level) {
  const auto& phi = solver.phi_level(level);
  MultiFab<Dim> active(phi.layout(), phi.distribution(), phi.local_rank(), 1, Extent<Dim>{});
  MultiFab<Dim> inverse_volume(phi.layout(), phi.distribution(), phi.local_rank(), 1,
                               Extent<Dim>{});
  MultiFab<Dim> aperture_lower(phi.layout(), phi.distribution(), phi.local_rank(), Dim,
                               Extent<Dim>{});
  MultiFab<Dim> aperture_upper(phi.layout(), phi.distribution(), phi.local_rank(), Dim,
                               Extent<Dim>{});
  active.set_val(Real(1));
  inverse_volume.set_val(Real(1));
  aperture_lower.set_val(Real(1));
  aperture_upper.set_val(Real(1));
  solver.install_embedded_boundary(level, active, inverse_volume, aperture_lower, aperture_upper);
}

template <int Dim>
void expect_partitioned_fac_embedded_boundary() {
  const ExecutionLane lane =
      ExecutionLane::world("pops.test.composite-fac.partitioned-eb:" + std::to_string(Dim));
  pops::CompositeFacOptions options;
  options.max_iters = 40;
  options.fine_sweeps = 4;
  options.rel_tol = Real(5e-3);
  options.abs_tol = Real(1e-10);
  options.coarse_rel_tol = Real(1e-4);
  options.coarse_abs_tol = Real(1e-10);
  options.coarse_cycles = 192;
  CompositeFacPoisson<Dim> solver(make_request<Dim>(), options, Real(1));
  EXPECT_EQ(pops::all_reduce_min(solver.has_remote_same_level_halo() ? 1L : 0L, lane), 1L);
  for (int level = 0; level < solver.n_levels(); ++level) {
    install_unit_metric(solver, level);
    solver.rhs_level(level).set_val(Real(1));
    solver.phi_level(level).set_val(Real(0));
  }
  const pops::SolveReport report = solver.solve();
  EXPECT_TRUE(report.solved()) << report.reason << " residual=" << report.residual_norm;
  EXPECT_LT(maximum_constant_error(solver.phi_level(0), lane), Real(0.12));
  EXPECT_LT(maximum_constant_error(solver.phi_level(1), lane), Real(0.12));
}

int run_partitioned_fac_matrix(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    EXPECT_EQ(pops::n_ranks(), 2);
    if (pops::n_ranks() == 2) {
      expect_partitioned_fac<1>();
      expect_partitioned_fac<2>();
      expect_partitioned_fac<3>();
      expect_partitioned_fac_embedded_boundary<2>();
      expect_mg_distributed_fac<1>();
      expect_mg_distributed_fac<2>();
      expect_mg_distributed_fac<3>();
      expect_partitioned_singular_nullspace<1>();
      expect_partitioned_singular_nullspace<2>();
      expect_partitioned_singular_nullspace<3>();
      expect_exact_rank_ratio_prepares<1>({3});
      expect_exact_rank_ratio_prepares<2>({3, 1});
      expect_exact_rank_ratio_prepares<3>({1, 2, 3});
      expect_collective_budget_failure();
    }
    result = ::testing::Test::HasFailure() ? 1 : 0;
  }
  pops::comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_composite_fac_partitioned_nd, RunsPartitionedDimensionalMatrix) {
  EXPECT_EQ(
      pops::test::RunTestBody(&run_partitioned_fac_matrix, "test_mpi_composite_fac_partitioned_nd"),
      0);
}
