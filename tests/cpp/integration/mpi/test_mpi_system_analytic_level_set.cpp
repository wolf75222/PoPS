#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/index/extent.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/layout/rank_space.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/system.hpp>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

template <int Dim>
SystemConfig<Dim> exact_system_config(int cells, bool periodic) {
  SystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = cells;
    config.lower[axis] = Real(0);
    config.upper[axis] = Real(1);
    config.periodicity[axis] = periodic;
  }
  return config;
}

int run_analytic_level_set_collective_preflight(int argc, char** argv) {
  constexpr int Dim = kNativeDimension;
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int rank = my_rank();
  long local_failures = 0;
  const auto require = [&local_failures, rank](bool condition, const char* message) {
    if (!condition) {
      std::cerr << "analytic level-set MPI check failed on rank " << rank << ": " << message
                << '\n';
      ++local_failures;
    }
  };

  require(n_ranks() == 2, "the regression must run with exactly two MPI ranks");

  // A Cartesian System currently owns one global patch, distributed round-robin to rank zero. This
  // probe mirrors that exact layout and proves that only one rank samples the invalid expression;
  // the other rank can reject only through the native collective preflight.
  Extent<Dim> cell_extent{};
  Extent<Dim> process_extent{};
  Extent<Dim> ghosts{};
  for (int axis = 0; axis < Dim; ++axis) {
    cell_extent[axis] = 20;
    process_extent[axis] = axis == 0 ? n_ranks() : 1;
    ghosts[axis] = 1;
  }
  const Box<Dim> domain = Box<Dim>::from_extents(cell_extent);
  const mesh::BoxArray<Dim> ownership_layout(std::vector<Box<Dim>>{domain});
  const mesh::RankSpace<Dim> rank_space(Index<Dim>{}, process_extent);
  const mesh::Distribution<Dim> ownership = mesh::Distribution<Dim>::partitioned(
      ownership_layout, rank_space, {rank_space.coordinate(0)});
  MultiFab<Dim> ownership_probe(ownership_layout, ownership, rank_space.coordinate(my_rank()), 1,
                                ghosts);
  const long local_sampler = ownership_probe.local_size() == 1 ? 1L : 0L;
  require(all_reduce_sum(local_sampler) == 1,
          "the invalid expression must be sampled on exactly one rank");
  require(local_sampler == (rank == 0 ? 1L : 0L),
          "the single Cartesian patch must be owned by rank zero");

  System<Dim> system(exact_system_config<Dim>(20, false));
  system.set_analytic_level_set({"x", "constant", "sub"}, {0.0, 0.5, 0.0}, "staircase", 0.2, 1e-5,
                                0.1);
  const std::vector<double> before = system.embedded_boundary_mask();
  const long active = static_cast<long>(std::count(before.begin(), before.end(), 1.0));
  const long global_active = all_reduce_sum(active);
  const long global_cells = all_reduce_sum(static_cast<long>(before.size()));
  require(global_active > 0 && global_active < global_cells,
          "the committed reference mask must contain active and inactive cells");

  // Both requests are locally valid, but rank one changes the geometry mode and one exact binary64
  // literal. The collective request preflight must reject before replacing the committed mask.
  bool geometry_mismatch_rejected = false;
  std::string geometry_mismatch_message;
  try {
    system.set_analytic_level_set({"x", "constant", "sub"}, {0.0, rank == 0 ? 0.45 : 0.55, 0.0},
                                  rank == 0 ? "staircase" : "cutcell", 0.2, 1e-5, 0.1);
  } catch (const std::runtime_error& error) {
    geometry_mismatch_rejected = true;
    geometry_mismatch_message = error.what();
  }
  require(all_reduce_sum(geometry_mismatch_rejected ? 1L : 0L) == n_ranks(),
          "all ranks must reject a rank-dependent analytic geometry request");
  require(geometry_mismatch_rejected &&
              geometry_mismatch_message.find("differs across MPI ranks") != std::string::npos,
          "rank-dependent geometry rejection must identify exact MPI disagreement");
  const std::vector<double> after_geometry_mismatch = system.embedded_boundary_mask();
  require(all_reduce_sum(after_geometry_mismatch == before ? 0L : 1L) == 0,
          "rank-dependent geometry rejection must preserve the committed mask");

  // (x - x) / 0 is non-finite at every sampled point. Only rank zero owns sample points, yet every
  // rank must receive the same domain_error after the native MPI all-reduce, without deadlock.
  const std::vector<std::string> invalid_ops{"x", "x", "sub", "constant", "div"};
  const std::vector<double> invalid_literals{0.0, 0.0, 0.0, 0.0, 0.0};
  bool rejected = false;
  std::string rejection_message;
  try {
    system.set_analytic_level_set(invalid_ops, invalid_literals, "cutcell", 0.3, 2e-5, 0.2);
  } catch (const std::domain_error& error) {
    rejected = true;
    rejection_message = error.what();
  } catch (const std::exception& error) {
    rejection_message = std::string("wrong exception type: ") + error.what();
  }

  require(all_reduce_sum(rejected ? 1L : 0L) == n_ranks(),
          "all ranks must reject the rank-local non-finite sample");
  require(rejected && rejection_message.find("non-finite") != std::string::npos,
          "every rank must report the analytic finite-value contract");

  const std::vector<double> after = system.embedded_boundary_mask();
  const long local_partial_publication = before == after ? 0L : 1L;
  require(all_reduce_sum(local_partial_publication) == 0,
          "failed replacement must preserve the previously committed global mask on every rank");

  const long failures = all_reduce_sum(local_failures);
  comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_system_analytic_level_set, CollectiveAnalyticRequestsRejectBeforePublication) {
  EXPECT_EQ(pops::test::RunTestBody(&run_analytic_level_set_collective_preflight,
                                    "test_mpi_system_analytic_level_set"),
            0);
}
