#include <gtest/gtest.h>

#include <pops/runtime/amr/prepared_amr_ghost_fill.hpp>

#include <cstddef>
#include <vector>

using namespace pops;
using namespace pops::mesh;
using namespace pops::runtime::amr;

namespace {

std::size_t offset(const Fab<1>& fab, int index) {
  return static_cast<std::size_t>(index - fab.grown_box().lo[0]);
}

void fill_parent(MultiFab<1>& coarse) {
  for (std::size_t local = 0; local < coarse.local_size(); ++local) {
    auto& fab = coarse.fab(local);
    auto host = fab.create_host_mirror();
    for (int i = fab.box().lo[0]; i <= fab.box().hi[0]; ++i)
      host(offset(fab, i)) = static_cast<Real>(i);
    fab.copy_from_host(host);
  }
}

void fill_child(MultiFab<1>& fine) {
  for (std::size_t local = 0; local < fine.local_size(); ++local) {
    auto& fab = fine.fab(local);
    auto host = fab.create_host_mirror();
    for (int i = fab.grown_box().lo[0]; i <= fab.grown_box().hi[0]; ++i)
      host(offset(fab, i)) = fab.box().contains(Index<1>{i}) ? Real(100 + i) : Real(-777);
    fab.copy_from_host(host);
  }
}

Real value(const Fab<1>& fab, int index) {
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  return host(offset(fab, index));
}

AmrGhostFillBudget budget() {
  return AmrGhostFillBudget{CoarseFineGhostScheduleBudget{2, 16, 16, 64, 4, 1024, 1024, 1024},
                            HaloScheduleBudget{{2, 1}, 32, 64, 4, 4, 1024, 1024, 1024}};
}

}  // namespace

TEST(test_mpi_prepared_amr_ghost_fill,
     partitioned_parent_and_swapped_fine_ownership_fill_one_exact_candidate) {
  Kokkos::ScopeGuard kokkos;
  const ExecutionLane control = ExecutionLane::world();
  ASSERT_EQ(control.size(), 3);
  const int rank = control.rank();
  const RankSpace<1> ranks{Index<1>{0}, Extent<1>{3}};

  const Box<1> coarse_domain{Index<1>{0}, Index<1>{7}};
  const BoxArray<1> coarse_layout(
      std::vector<Box<1>>{{Index<1>{0}, Index<1>{3}}, {Index<1>{4}, Index<1>{7}}});
  const Distribution<1> coarse_distribution = Distribution<1>::partitioned(
      coarse_layout, ranks, std::vector<Index<1>>{Index<1>{0}, Index<1>{1}});
  MultiFab<1> coarse(coarse_layout, coarse_distribution, Index<1>{rank}, 1, Extent<1>{0});

  const Box<1> fine_domain{Index<1>{0}, Index<1>{15}};
  const BoxArray<1> fine_layout(
      std::vector<Box<1>>{{Index<1>{4}, Index<1>{7}}, {Index<1>{8}, Index<1>{11}}});
  const Distribution<1> fine_distribution = Distribution<1>::partitioned(
      fine_layout, ranks, std::vector<Index<1>>{Index<1>{1}, Index<1>{0}});
  MultiFab<1> fine(fine_layout, fine_distribution, Index<1>{rank}, 1, Extent<1>{1});
  fill_parent(coarse);
  fill_child(fine);

  ExecutionLane lane =
      ExecutionLane::duplicate_world_collectively("test-mpi-prepared-amr-ghost-fill");
  AmrGhostFillPreparation<1> request{};
  request.fine_level = 1;
  request.coarse_domain = coarse_domain;
  request.fine_domain = fine_domain;
  request.ratio = ::pops::amr::RefinementRatio<1>(2);
  request.topology_generation = 17;
  request.materialization_generation = 23;
  request.field_identity = "state";
  request.budget = budget();
  const auto fill = prepare_amr_ghost_fill(coarse, fine, request, lane);
  EXPECT_EQ(fill.has_remote_parent_jobs(), rank < 2);
  EXPECT_EQ(fill.has_remote_same_level_jobs(), rank < 2);

  runtime::multiblock::BoundaryEvaluationPoint point{};
  point.level = 1;
  fill(fine, point);

  if (rank == 2) {
    EXPECT_EQ(fine.local_size(), 0U);
    return;
  }
  ASSERT_EQ(fine.local_size(), 1U);
  const auto& local = fine.fab(0);
  for (int i = local.grown_box().lo[0]; i <= local.grown_box().hi[0]; ++i) {
    Real expected = Real(100 + i);
    if (!fine_layout[0].contains(Index<1>{i}) && !fine_layout[1].contains(Index<1>{i}))
      expected = static_cast<Real>(i / 2) + (i % 2 == 0 ? Real(-0.25) : Real(0.25));
    EXPECT_DOUBLE_EQ(value(local, i), expected);
  }
}
