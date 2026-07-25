// Native distributed proof of the dynamic AMR active-prefix contract.
//
// Empty collective tags remove both fine levels, a coarse-only step remains valid, and later tags
// regrow the two levels on the same configured hierarchy.  The active depth and patch count must
// agree bit-for-bit across ranks, while mass remains conserved through removal and regrowth.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>

#include "amr_tagging_test_authority.hpp"
#include "amr_transfer_test_authority.hpp"
#include "load_balance_test_authority.hpp"
#include "test_harness.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

int run_dynamic_active_depth(int n, int me, int np) {
  const Geometry geometry{Box2D::from_extents(n, n), Real(0), Real(1), Real(0), Real(1)};
  const BoxArray coarse_boxes = BoxArray::from_domain(geometry.domain, n / 2);
  const Box2D level_one_patch = Box2D{{2, 2}, {n - 3, n - 3}}.refine(2);
  const Box2D level_two_parent{{level_one_patch.lo[0] + 4, level_one_patch.lo[1] + 4},
                               {level_one_patch.hi[0] - 4, level_one_patch.hi[1] - 4}};
  const Box2D level_two_patch = level_two_parent.refine(2);

  auto levels = std::make_shared<std::vector<AmrLevelMP>>();
  levels->push_back(AmrLevelMP{MultiFab(coarse_boxes,
                                        DistributionMapping(coarse_boxes.size(), n_ranks()), 1, 1),
                               nullptr, geometry.dx(), geometry.dy()});
  levels->push_back(
      AmrLevelMP{MultiFab(BoxArray({level_one_patch}), DistributionMapping({0}), 1, 1), nullptr,
                 geometry.dx() / Real(2), geometry.dy() / Real(2)});
  levels->push_back(
      AmrLevelMP{MultiFab(BoxArray({level_two_patch}), DistributionMapping({0}), 1, 1), nullptr,
                 geometry.dx() / Real(4), geometry.dy() / Real(4)});

  for (std::size_t level_index = 0; level_index < levels->size(); ++level_index) {
    MultiFab& values = (*levels)[level_index].U;
    const int scale = 1 << static_cast<int>(level_index);
    for (int local = 0; local < values.local_size(); ++local) {
      Fab2D& fab = values.fab(local);
      const Box2D box = values.box(local);
      for (int j = box.lo[1]; j <= box.hi[1]; ++j)
        for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
          const double x = geometry.x_cell(i / scale);
          const double y = geometry.y_cell(j / scale);
          const double r2 = (x - 0.28) * (x - 0.28) + (y - 0.50) * (y - 0.50);
          fab(i, j, 0) = Real(1.0 + 0.8 * std::exp(-r2 / (0.08 * 0.08)));
        }
    }
  }

  const auto load_balance = test::prepare_test_space_filling_curve_load_balance();
  AmrHierarchyLayout hierarchy = AmrHierarchyLayout::from_levels(*levels, load_balance);

  AmrRuntimeBlock block;
  block.name = "moving";
  block.state_identity = "test://mpi-active-depth/block/moving/state/U";
  block.levels = levels;
  block.advance = [](std::vector<AmrLevelMP>&, const Box2D&, Real, Periodicity, bool,
                     PreparedAmrFillPatchPlan*, PreparedAmrAverageDownPlan*,
                     PreparedAmrAdvanceScratchPlan*) {};
  block.advance_with_temporal_plan =
      [](std::vector<AmrLevelMP>&, const Box2D&, Real, Periodicity, bool,
         const detail::PreparedAmrTemporalPlan&, PreparedAmrFillPatchPlan*,
         PreparedAmrAverageDownPlan*, PreparedAmrAdvanceScratchPlan*) {};
  block.add_elliptic_rhs = [](const MultiFab&, MultiFab&) {};
  block.max_speed = [](const MultiFab&, const MultiFab&) { return Real(0); };
  block.mass = [levels, geometry] {
    Real local_mass = 0;
    const MultiFab& values = levels->front().U;
    const Real cell_volume = geometry.dx() * geometry.dy();
    for (int local = 0; local < values.local_size(); ++local) {
      const ConstArray4 state = values.fab(local).const_array();
      local_mass += for_each_cell_reduce_sum(
          values.box(local),
          [state, cell_volume] POPS_HD(int i, int j) { return state(i, j, 0) * cell_volume; });
    }
    return all_reduce_sum(local_mass);
  };

  std::vector<AmrRuntimeBlock> blocks;
  blocks.push_back(std::move(block));
  AmrRuntime runtime(geometry, std::move(hierarchy), BCRec{}, std::move(blocks),
                     Periodicity{true, true}, /*replicated_coarse=*/false);
  test::install_second_order_amr_transfer_authorities(runtime, 1);
  runtime.set_parent_child_temporal_relations(
      {amr::ParentChildClockRelation(0, 1, amr::Rational(2, 1),
                                     amr::RemainderPolicy::IntegralOnly),
       amr::ParentChildClockRelation(1, 2, amr::Rational(2, 1),
                                     amr::RemainderPolicy::IntegralOnly)});

  const double initial_mass = runtime.mass(0);
  test::install_prepared_threshold_decisions(
      runtime, {{0, 0, Real(1e9), test::PreparedThresholdRelation::Above}},
      {{0, 0, Real(1e9), test::PreparedThresholdRelation::Below}},
      "test::mpi-active-depth-coarsen@1");
  runtime.regrid();
  const bool removed =
      runtime.nlev() == 1 && runtime.max_levels() == 3 && runtime.n_patches() == 0;
  const double removed_mass = runtime.mass(0);

  runtime.step(Real(1e-4));
  test::install_prepared_threshold_decisions(
      runtime, {{0, 0, Real(1.05), test::PreparedThresholdRelation::Above}},
      {{0, 0, Real(1.05), test::PreparedThresholdRelation::Below}},
      "test::mpi-active-depth-regrow@1");
  runtime.regrid();
  const bool regrown =
      runtime.nlev() == 3 && runtime.max_levels() == 3 && runtime.n_patches() > 0;
  const double regrown_mass = runtime.mass(0);

  auto spread = [](double value) {
    return all_reduce_max(value) - (-all_reduce_max(-value));
  };
  const double cross_rank_spread =
      std::fmax(spread(static_cast<double>(runtime.nlev())),
                std::fmax(spread(static_cast<double>(runtime.n_patches())),
                          std::fmax(spread(removed_mass), spread(regrown_mass))));
  const bool conserved = std::fabs(removed_mass - initial_mass) < 1e-10 &&
                         std::fabs(regrown_mass - initial_mass) < 1e-10;
  const long local_failure = removed && regrown && conserved && cross_rank_spread == 0.0 ? 0L : 1L;
  const long failure = all_reduce_max(local_failure);

  if (me == 0) {
    std::printf(
        "AMRDEPTH np=%d | removed=%d regrown=%d | active=%d configured=%d patches=%d | "
        "dm_remove=%.3e dm_regrow=%.3e spread=%.3e\n",
        np, removed ? 1 : 0, regrown ? 1 : 0, runtime.nlev(), runtime.max_levels(),
        runtime.n_patches(), std::fabs(removed_mass - initial_mass),
        std::fabs(regrown_mass - initial_mass), cross_rank_spread);
  }
  return failure == 0 ? 0 : 1;
}

int pops_run_test_mpi_amr_dynamic_active_depth(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int result = run_dynamic_active_depth(/*n=*/16, my_rank(), n_ranks());
  comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_amr_dynamic_active_depth, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_mpi_amr_dynamic_active_depth,
                                    "test_mpi_amr_dynamic_active_depth"),
            0);
}
