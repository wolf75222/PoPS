#include <gtest/gtest.h>

#include <pops/coupling/amr/amr_coupler_mp.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr/bootstrap_transfer_builtins.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/builders/factory/model_factory.hpp>
#include <pops/runtime/config/model_spec.hpp>

#include <pops/mesh/layout/refinement.hpp>

#include <cmath>
#include <optional>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;
using namespace pops::runtime::amr;

namespace {

MultiFab field(const Box2D& box, int ghosts = 0) {
  return MultiFab(BoxArray(std::vector<Box2D>{box}), DistributionMapping(1, n_ranks()), 1,
                  ghosts);
}

SpatialTransferContext context(const Box2D& coarse, int coarse_level = 0,
                               int fine_level = 1) {
  const Box2D fine_domain = coarse.refine(2);
  return SpatialTransferContext{
      coarse_level, fine_level, 1,
      IndexTransform{{coarse.lo[0], coarse.lo[1]},
                     {fine_domain.lo[0], fine_domain.lo[1]}, {2, 2}},
      true};
}

ModelSpec exb_model() {
  ModelSpec spec;
  spec.transport = "exb";
  spec.source = "none";
  spec.elliptic = "charge";
  spec.q = 1.0;
  spec.B0 = 1.0;
  return spec;
}

AmrRuntime bootstrap_runtime(int cells = 8) {
  AmrBuildParams params;
  params.mesh.n = cells;
  params.mesh.L = 1.0;
  params.mesh.regrid_every = 0;
  params.poisson.bc = BCRec{};
  const detail::SharedAmrLayout layout =
      detail::make_shared_amr_layout_levels(params, 1);
  std::vector<AmrRuntimeBlock> blocks;
  detail::dispatch_model(exb_model(), [&](auto model) {
    blocks.push_back(detail::dispatch_amr_block(
        model, "minmod", "rusanov", layout, "transport",
        std::vector<double>(static_cast<std::size_t>(cells) * cells, 1.0),
        true, 1.4, 1, false, false, 1));
  });
  return AmrRuntime(
      layout.geom, layout.runtime_hierarchy(), layout.poisson_bc, std::move(blocks),
      layout.base_per, layout.replicated_coarse, layout.wall);
}

}  // namespace

#if defined(POPS_HAS_KOKKOS)
class KokkosEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { guard_.emplace(); }
  void TearDown() override { guard_.reset(); }

 private:
  std::optional<Kokkos::ScopeGuard> guard_;
};

::testing::Environment* const kKokkosEnv =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
#endif

TEST(test_amr_transfer_properties, CellConservationRestrictionAndNonzeroOrigin) {
  const Box2D coarse_box{{3, 5}, {6, 8}};
  const Box2D fine_box = coarse_box.refine(2);
  MultiFab coarse = field(coarse_box, 1);
  MultiFab fine = field(fine_box, 1);
  MultiFab restricted = field(coarse_box);
  Array4 c = coarse.fab(0).array();
  for (int j = coarse_box.lo[1]; j <= coarse_box.hi[1]; ++j)
    for (int i = coarse_box.lo[0]; i <= coarse_box.hi[0]; ++i)
      c(i, j, 0) = 2.0 + 0.4 * i - 0.3 * j;

  const auto transform = context(coarse_box);
  detail::coupler_conservative_linear_to_fine_mb(
      coarse, fine, transform.index.coarse_origin, transform.index.fine_origin,
      transform.index.refinement_ratio);
  average_down(fine, restricted, 2);
  const ConstArray4 r = restricted.fab(0).const_array();
  for (int j = coarse_box.lo[1]; j <= coarse_box.hi[1]; ++j)
    for (int i = coarse_box.lo[0]; i <= coarse_box.hi[0]; ++i)
      EXPECT_NEAR(r(i, j, 0), c(i, j, 0), 1e-14);
}

TEST(test_amr_transfer_properties, CoupledFaceProlongationCommutesWithDivergence) {
  const Box2D cell_box{{3, 5}, {6, 8}};
  const Box2D coarse_x{{3, 5}, {7, 8}};
  const Box2D coarse_y{{3, 5}, {6, 9}};
  const Box2D fine_x{{6, 10}, {14, 17}};
  const Box2D fine_y{{6, 10}, {13, 18}};
  MultiFab cx = field(coarse_x, 1), cy = field(coarse_y, 1);
  MultiFab fx = field(fine_x, 1), fy = field(fine_y, 1);
  Array4 x = cx.fab(0).array(), y = cy.fab(0).array();
  for (int j = coarse_x.lo[1]; j <= coarse_x.hi[1]; ++j)
    for (int i = coarse_x.lo[0]; i <= coarse_x.hi[0]; ++i)
      x(i, j, 0) = 0.2 * i * i + 0.1 * j;
  for (int j = coarse_y.lo[1]; j <= coarse_y.hi[1]; ++j)
    for (int i = coarse_y.lo[0]; i <= coarse_y.hi[0]; ++i)
      y(i, j, 0) = -0.15 * j * j + 0.05 * i;

  const auto transform = context(cell_box);
  detail::bootstrap_prolong_face_vector(cx, cy, fx, fy, transform);
  const ConstArray4 qx = fx.fab(0).const_array(), qy = fy.fab(0).const_array();
  for (int j = cell_box.lo[1]; j <= cell_box.hi[1]; ++j)
    for (int i = cell_box.lo[0]; i <= cell_box.hi[0]; ++i) {
      const double coarse_div = x(i + 1, j, 0) - x(i, j, 0) +
                                y(i, j + 1, 0) - y(i, j, 0);
      for (int child_y = 0; child_y < 2; ++child_y)
        for (int child_x = 0; child_x < 2; ++child_x) {
          const int fi = 2 * i + child_x, fj = 2 * j + child_y;
          const double fine_div = 2.0 *
              (qx(fi + 1, fj, 0) - qx(fi, fj, 0) +
               qy(fi, fj + 1, 0) - qy(fi, fj, 0));
          EXPECT_NEAR(fine_div, coarse_div, 1e-14);
        }
    }
}

TEST(test_amr_transfer_properties, NodeBilinearIsSecondOrderAtNonzeroOrigin) {
  const Box2D coarse_nodes{{3, 5}, {7, 9}};
  const Box2D fine_nodes{{6, 10}, {14, 18}};
  MultiFab coarse = field(coarse_nodes, 1), fine = field(fine_nodes, 1);
  Array4 c = coarse.fab(0).array();
  for (int j = coarse_nodes.lo[1]; j <= coarse_nodes.hi[1]; ++j)
    for (int i = coarse_nodes.lo[0]; i <= coarse_nodes.hi[0]; ++i)
      c(i, j, 0) = 1.5 + 0.7 * i - 0.2 * j;
  detail::bootstrap_prolong_staggered(
      coarse, fine, TransferCentering::Node,
      context(Box2D{{3, 5}, {6, 8}}));
  const ConstArray4 f = fine.fab(0).const_array();
  for (int j = fine_nodes.lo[1]; j <= fine_nodes.hi[1]; ++j)
    for (int i = fine_nodes.lo[0]; i <= fine_nodes.hi[0]; ++i) {
      const double expected = 1.5 + 0.7 * (0.5 * i) - 0.2 * (0.5 * j);
      EXPECT_NEAR(f(i, j, 0), expected, 1e-14);
    }
}

TEST(test_amr_transfer_properties, CoarseFineAndTemporalUseIndependentPhysicalContexts) {
  const Box2D coarse_box{{3, 5}, {6, 8}};
  const Box2D fine_patch{{8, 12}, {11, 15}};
  MultiFab parent = field(coarse_box), fine = field(fine_patch, 1);
  Array4 p = parent.fab(0).array();
  for (int j = coarse_box.lo[1]; j <= coarse_box.hi[1]; ++j)
    for (int i = coarse_box.lo[0]; i <= coarse_box.hi[0]; ++i)
      p(i, j, 0) = 10.0 * i + j;
  fine.set_val(-7.0);
  auto spatial = prepare_conservative_coarse_fine();
  spatial.coarse_fine(parent, fine, context(coarse_box));
  const ConstArray4 filled = fine.fab(0).const_array();
  EXPECT_DOUBLE_EQ(filled(fine_patch.lo[0] - 1, fine_patch.lo[1], 0),
                   p(3, 6, 0));
  EXPECT_DOUBLE_EQ(filled(fine_patch.lo[0], fine_patch.lo[1], 0), -7.0);

  MultiFab old_value = field(coarse_box), new_value = field(coarse_box);
  MultiFab destination = field(coarse_box);
  Array4 old_data = old_value.fab(0).array(), new_data = new_value.fab(0).array();
  for (int j = coarse_box.lo[1]; j <= coarse_box.hi[1]; ++j)
    for (int i = coarse_box.lo[0]; i <= coarse_box.hi[0]; ++i) {
      old_data(i, j, 0) = i + j;
      new_data(i, j, 0) = i + j + 4.0;
    }
  auto temporal = prepare_linear_time_interpolation();
  temporal.temporal(
      old_value, new_value, destination,
      TemporalTransferContext{{10, 1.0}, {11, 3.0}, {10, 1.5}});
  const ConstArray4 interpolated = destination.fab(0).const_array();
  for (int j = coarse_box.lo[1]; j <= coarse_box.hi[1]; ++j)
    for (int i = coarse_box.lo[0]; i <= coarse_box.hi[0]; ++i)
      EXPECT_DOUBLE_EQ(interpolated(i, j, 0), i + j + 1.0);
  EXPECT_THROW(
      temporal.temporal(
          old_value, new_value, destination,
          TemporalTransferContext{{10, 1.0}, {10, 3.0}, {10, 1.5}}),
      std::runtime_error);
  EXPECT_THROW(
      temporal.temporal(
          old_value, old_value, destination,
          TemporalTransferContext{{10, 1.0}, {11, 3.0}, {10, 1.5}}),
      std::runtime_error);
}

TEST(test_amr_transfer_properties, AnalyticEveryLevelCacheEpochAndL0L1L2Rollback) {
  AmrRuntime runtime = bootstrap_runtime();
  ASSERT_EQ(runtime.nlev(), 1);
  const std::vector<double> baseline = runtime.block_level_state(0, 0);
  runtime.set_bootstrap_threshold_tag(
      0, 0, Real(0.5), "test.tag-provider::component-above");

  for (int failure_level = 0; failure_level <= 2; ++failure_level) {
    runtime.begin_bootstrap_plan();
    EXPECT_GT(runtime.fill_bootstrap_block_constant(0, 0, {7.0}), 0);
    for (int level = 1; level <= failure_level; ++level) {
      runtime.bootstrap_next_level(2);
      EXPECT_GT(runtime.fill_bootstrap_block_constant(0, level, {7.0}), 0);
      const std::vector<double> materialized = runtime.block_level_state(0, level);
      for (double value : materialized)
        if (value != 0.0)
          EXPECT_DOUBLE_EQ(value, 7.0);
    }
    EXPECT_EQ(runtime.nlev(), failure_level + 1);
    runtime.rollback_bootstrap_level();
    EXPECT_EQ(runtime.nlev(), 1);
    EXPECT_EQ(runtime.block_level_state(0, 0), baseline);
  }

  runtime.begin_bootstrap_plan();
  EXPECT_EQ(runtime.invalidate_bootstrap_cache("patch-topology"), 1u);
  EXPECT_TRUE(runtime.rebuild_bootstrap_topology_cache("patch-topology", 0).valid);
  EXPECT_EQ(runtime.invalidate_bootstrap_cache("patch-topology"), 2u);
  const auto& cache = runtime.rebuild_bootstrap_topology_cache("patch-topology", 0);
  EXPECT_TRUE(cache.valid);
  EXPECT_EQ(cache.epoch, 2u);
  runtime.rollback_bootstrap_level();
  EXPECT_THROW(runtime.bootstrap_cache("patch-topology"), std::runtime_error);
}
