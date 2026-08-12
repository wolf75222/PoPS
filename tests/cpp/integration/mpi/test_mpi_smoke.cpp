#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/load_balance.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>

namespace {

template <int Dim>
pops::Extent<Dim> uniform_extent(std::int64_t value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::Index<Dim> rank_coordinate(int rank) {
  pops::Index<Dim> result{};
  result[0] = rank;
  return result;
}

template <int Dim>
void prove_partitioned_storage_and_collectives(int rank_count, int rank) {
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = (axis == 0 ? rank_count * 8 : 8) - 1;
  const pops::Box<Dim> domain{pops::Index<Dim>{}, upper};
  const auto layout = pops::mesh::BoxArray<Dim>::from_domain(domain, uniform_extent<Dim>(4));

  auto rank_extent = uniform_extent<Dim>(1);
  rank_extent[0] = rank_count;
  const pops::mesh::RankSpace<Dim> rank_space{pops::Index<Dim>{}, rank_extent};
  const pops::parallel::LoadBalancePreparationBudget budget{
      layout.size(), static_cast<std::size_t>(rank_count), domain.numPts()};
  const auto plan = pops::parallel::LoadBalanceProvider<Dim>::space_filling_curve().prepare(
      layout, rank_space, budget);
  const auto local_rank = rank_coordinate<Dim>(rank);
  pops::MultiFab<Dim> field(layout, plan.distribution(), local_rank, 1, pops::Extent<Dim>{});
  field.set_val(pops::Real{1});

  EXPECT_EQ(plan.total_weight(), domain.numPts());
  EXPECT_EQ(plan.weights().size(), layout.size());
  EXPECT_EQ(plan.linear_owners().size(), layout.size());
  EXPECT_EQ(plan.distribution().rank_space(), rank_space);
  EXPECT_DOUBLE_EQ(plan.imbalance(), 1.0);
  EXPECT_DOUBLE_EQ(pops::reduce_sum(field), static_cast<pops::Real>(domain.numPts()));
  EXPECT_GT(field.local_size(), 0U);
  EXPECT_EQ(pops::all_reduce_sum(static_cast<long>(field.local_size())),
            static_cast<long>(layout.size()));

  for (const std::size_t global : field.local_global_indices()) {
    EXPECT_EQ(plan.distribution().owner(global), local_rank);
    EXPECT_EQ(plan.linear_owners()[global], static_cast<std::size_t>(rank));
  }
}

int run_mpi_smoke(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    try {
      prove_partitioned_storage_and_collectives<pops::kNativeDimension>(pops::n_ranks(),
                                                                        pops::my_rank());
    } catch (const std::exception& error) {
      std::fprintf(stderr, "rank %d exact-rank MPI smoke failed: %s\n", pops::my_rank(),
                   error.what());
      result = 1;
    }
    result = static_cast<int>(
        pops::all_reduce_max(static_cast<long>(result || ::testing::Test::HasFailure())));
    if (pops::my_rank() == 0 && result == 0)
      std::printf("OK test_mpi_smoke (np=%d dim=%d exact-rank SFC/storage/reduction)\n",
                  pops::n_ranks(), pops::kNativeDimension);
  }
  pops::comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_smoke, NativeDimensionUsesOneExactRankedParallelPath) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_smoke, "test_mpi_smoke"), 0);
}
