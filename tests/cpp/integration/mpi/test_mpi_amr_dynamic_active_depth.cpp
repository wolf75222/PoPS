// Native distributed proof of the dynamic AMR active-prefix contract.
//
// Empty collective tags remove both fine levels, a coarse-only step remains valid, and later tags
// regrow the two levels on the same configured hierarchy.  The active depth and patch count must
// agree bit-for-bit across ranks, while mass remains conserved through removal and regrowth.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/builders/factory/model_factory.hpp>
#include <pops/runtime/config/model_spec.hpp>

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

ModelSpec exb_charge(double q, double B0) {
  ModelSpec spec;
  spec.transport = "exb";
  spec.source = "none";
  spec.elliptic = "charge";
  spec.q = q;
  spec.B0 = B0;
  return spec;
}

std::vector<double> blob(int n, double cx, double cy, double amp, double base, double width) {
  std::vector<double> density(static_cast<std::size_t>(n) * n, base);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n;
      const double y = (j + 0.5) / n;
      const double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      density[static_cast<std::size_t>(j) * n + i] =
          base + amp * std::exp(-r2 / (width * width));
    }
  return density;
}

int run_dynamic_active_depth(int n, int me, int np) {
  AmrBuildParams params;
  params.mesh.n = n;
  params.mesh.L = 1.0;
  params.mesh.distribute_coarse = true;
  params.mesh.coarse_max_grid = n / 2;
  params.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
  params.poisson.bc = BCRec{};
  const detail::SharedAmrLayout layout = detail::make_shared_amr_layout_levels(params, 3);

  std::vector<AmrRuntimeBlock> blocks;
  const std::vector<double> density = blob(n, 0.28, 0.50, 0.8, 1.0, 0.08);
  detail::dispatch_model(exb_charge(0.0, 1.0), [&](auto model) {
    blocks.push_back(detail::dispatch_amr_block(
        model, "minmod", "rusanov", layout, "moving", density,
        /*has_density=*/true, 1.4, 1, false, false, 1));
  });
  blocks.back().state_identity = "test://mpi-active-depth/block/moving/state/U";

  AmrRuntime runtime(layout.geom, layout.runtime_hierarchy(), layout.poisson_bc,
                     std::move(blocks), layout.base_per, layout.replicated_coarse, layout.wall);
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
