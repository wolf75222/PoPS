// Include in test_program_runtime.cpp; exercises the real owner-aware provider.
TEST(ProgramRuntime, ExplicitVectorPairingIncludesEveryComponentAndKeepsLegacyDotZero) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);
  System<kNativeDimension> sim(unit_domain_config<kNativeDimension>(n));
  install_execution_lane(sim, "pops.test.program.dot-all.v1");
  add_gas(sim, 1.4, "none");
  std::vector<double> initial(static_cast<std::size_t>(kGasComponents) * cells, 0.);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    initial[cell] = 1.;
    initial[(kGasComponents - 1) * cells + cell] = 2.5;
  }
  sim.set_state("gas", initial);
  sim.set_program_block_map({0});
  runtime::program::ProgramContext context(&sim);
  auto& accepted = context.state(0);
  auto increment = context.scratch_state_like(accepted);
  increment.set_val(Real(0));
  for (std::size_t patch = 0; patch < increment.local_size(); ++patch) {
    const auto values = increment.fab(patch).view();
    for_each_cell(increment.box(patch), [=] POPS_HD(const Index<kNativeDimension>& cell) {
      values(cell, 0) = Real(-.5);
      values(cell, 1) = Real(1);
    });
  }
  const double legacy_copies = accepted.distribution().replicated()
      ? context.prepared_execution_lane().size() : 1.;
  EXPECT_DOUBLE_EQ(context.dot(0, accepted, increment), -.5 * cells * legacy_copies);
  EXPECT_DOUBLE_EQ(context.dot(0, increment, increment), .25 * cells * legacy_copies);
  EXPECT_DOUBLE_EQ(context.dot_all(0, accepted, increment), -.5 * cells);
  EXPECT_DOUBLE_EQ(context.dot_all(0, increment, increment), 1.25 * cells);
  const double gamma = -2. * context.dot_all(0, accepted, increment) /
      context.dot_all(0, increment, increment);
  EXPECT_DOUBLE_EQ(gamma, .8);
  EXPECT_DOUBLE_EQ(-2. * context.dot(0, accepted, increment) /
      context.dot(0, increment, increment), 4.);
  auto& wrong_width = context.scalar_scratch(873201, 0, accepted, 1, 0);
  const bool rank_zero = context.prepared_execution_lane().rank() == 0;
  EXPECT_ANY_THROW(context.dot_all(0, accepted, rank_zero ? wrong_width : increment));
  auto different_ghosts = accepted.ghosts();
  ++different_ghosts[0];
  MultiFab<kNativeDimension> foreign_ghosts(accepted.layout(), accepted.distribution(),
      accepted.local_rank(), accepted.ncomp(), different_ghosts);
  foreign_ghosts.set_val(Real(1));
  EXPECT_ANY_THROW(context.dot_all(0, accepted, rank_zero ? foreign_ghosts : increment));
  // A one-box System is partitioned to one owner, leaving empty ranks in MPI.
  // Poison an actual owner: setting a scalar on an empty MultiFab writes no DOF.
  const auto& lane = context.prepared_execution_lane();
  std::size_t local_cells = 0;
  for (std::size_t patch = 0; patch < increment.local_size(); ++patch)
    local_cells += static_cast<std::size_t>(increment.box(patch).numPts());
  const long poisoned_rank = all_reduce_max(local_cells != 0 ? lane.rank() : -1L, lane);
  ASSERT_GE(poisoned_rank, 0L);
  ASSERT_GT(all_reduce_sum(lane.rank() == poisoned_rank ? static_cast<long>(local_cells) : 0L,
                          lane), 0L);
  increment.set_val(lane.rank() == poisoned_rank
      ? std::numeric_limits<Real>::quiet_NaN() : Real(1));
  EXPECT_ANY_THROW(context.dot_all(0, increment, increment));
  // An empty participant still enters every vote and contributes no invented
  // nonfinite value. This is distinct from a stored noncontributing replica.
  const long empty_rank = all_reduce_max(local_cells == 0 ? lane.rank() : -1L, lane);
  if (empty_rank >= 0) {
    increment.set_val(lane.rank() == empty_rank
        ? std::numeric_limits<Real>::quiet_NaN() : Real(1));
    EXPECT_DOUBLE_EQ(context.dot_all(0, increment, increment), cells*kGasComponents);
  }
  increment.set_val(Real(1e308));
  EXPECT_ANY_THROW(context.dot_all(0, increment, increment));  // finite inputs, product overflow
  auto unit = context.scratch_state_like(accepted);
  unit.set_val(Real(1));
  EXPECT_ANY_THROW(context.dot_all(0, increment, unit));  // finite products, local sum overflow
}

TEST(ProgramRuntime, ExplicitVectorPairingRefusesDistributedGlobalOverflow) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  comm_init();
  if (n_ranks() < 2)
    GTEST_SKIP() << "Global-only overflow requires multiple contributing owners";
  constexpr int n = 8;
  if (n % n_ranks() != 0)
    GTEST_SKIP() << "Fixture decomposition requires ranks dividing eight";
  System<kNativeDimension> sim(distributed_boundary_domain_config<kNativeDimension>(n));
  install_execution_lane(sim, "pops.test.program.dot-all.global-overflow.v1");
  add_gas(sim, 1.4, "none");
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);
  std::vector<double> initial(kGasComponents * cells, 0.);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    initial[cell] = 1.;
    initial[(kGasComponents - 1) * cells + cell] = 2.5;
  }
  sim.set_state("gas", initial);
  sim.set_program_block_map({0});
  runtime::program::ProgramContext context(&sim);
  const auto& state = context.state(0);
  ASSERT_FALSE(state.distribution().replicated());
  auto left = context.scratch_state_like(state);
  auto right = context.scratch_state_like(state);
  std::size_t local_cells = 0;
  for (std::size_t patch = 0; patch < state.local_size(); ++patch)
    local_cells += static_cast<std::size_t>(state.box(patch).numPts());
  ASSERT_GT(local_cells, 0U);
  // Each rank contributes about .75*DBL_MAX, with finite products/local sums.
  left.set_val(std::numeric_limits<Real>::max() * Real(.75) /
               static_cast<Real>(local_cells * static_cast<std::size_t>(state.ncomp())));
  right.set_val(Real(1));
  EXPECT_TRUE(std::isfinite(pops::dot_owned_active_all_finite_local(left, right, static_cast<const MultiFab<kNativeDimension>*>(nullptr))));
  EXPECT_ANY_THROW(context.dot_all(0, left, right));
}


#include "explicit_amr_program.hpp"
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>

namespace {
struct DotAllAmrScalarModel : pops::nd::ScalarAdvection<kNativeDimension> {
  static constexpr int n_providers = 0;
  POPS_HD State source(const State&, const pops::ProviderValues<0>&) const { return {}; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};
}  // namespace

TEST(ProgramRuntime, ExplicitVectorPairingUsesNativeAmrCoverageAndIgnoresCoveredNaN) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  comm_init();
  constexpr int n = 8;
  AmrSystemConfig<kNativeDimension> config;
  // Materialize the staged coarse values; explicit bootstrap would defer their projection.
  config.explicit_bootstrap = false;
  config.regrid_every = 0;
  config.distribute_coarse = true;
  for (int axis = 0; axis < kNativeDimension; ++axis) {
    config.shape[axis] = n;
    config.periodicity[axis] = true;
    config.coarse_max_grid[axis] = axis == 0 ? 2 : n;
  }
  AmrSystem<kNativeDimension> sim(config);
  pops::test::install_amr_runtime_authority(sim, "pops.test.program.dot-all.amr-coverage.v1");
  sim.set_temporal_relations({1}, {1}, {"integral_only"});
  sim.install_block_state_route("scalar", "pops.test.dot-all/state@1");
  add_compiled_model<kNativeDimension>(sim, "scalar", DotAllAmrScalarModel{}, "minmod", "rusanov",
      "conservative", "explicit", 1.4, 1, 1, {}, {}, 0.,
      static_cast<double>(kWenoEpsilon), false, "pops.test.dot-all/physical-flux@1");
  sim.set_conservative_state("scalar", std::vector<double>(exact_cell_count<kNativeDimension>(n), 2.));
  auto context = runtime::program::make_program_execution_provider(&sim);
  context->install([](double) {}, context);
  sim.set_program_block_map({0});
  using Budget = AmrSystem<kNativeDimension>::PreparedAmrProgramFluxExpressionBlockBudget;
  sim.install_prepared_amr_program_flux_expression_budget(
      "pops.test.dot-all/amr-program@1", std::vector<Budget>(1, Budget{0, 0}), 0, 0);
  context->configure_primary_clock("clock.dot-all");
  // Anchor the actual accepted coarse DOFs before creating the prescribed child.
  // A nonnegative squared error cannot cancel between distributed owners.
  const auto& initial_coarse = context->state(0);
  Real local_initial_error = Real(0);
  for (std::size_t patch = 0; patch < initial_coarse.local_size(); ++patch) {
    const auto values = initial_coarse.fab(patch).view();
    local_initial_error += for_each_cell_reduce_sum(
        initial_coarse.box(patch), [=] POPS_HD(const Index<kNativeDimension>& cell) {
          const Real difference = values(cell, 0) - Real(2);
          return difference * difference;
        });
  }
  ASSERT_DOUBLE_EQ(all_reduce_sum(local_initial_error, context->prepared_execution_lane()), 0.);
  const auto& parent_layout = sim.engine()->hierarchy().layout(0);
  Index<kNativeDimension> upper{};
  for (int axis = 0; axis < kNativeDimension; ++axis)
    upper[axis] = 1;
  const mesh::BoxArray<kNativeDimension> boxes(
      std::vector<Box<kNativeDimension>>{{Index<kNativeDimension>{}, upper}});
  amr::tagging::ClusterOptions<kNativeDimension> options;
  options.min_efficiency = .7;
  options.min_box_size.fill(1);
  options.max_box_size.fill(16);
  options.budget = {16, 256, 8192, 64, 1U << 20};
  amr::tagging::ClusterResultIdentity<kNativeDimension> identity{
      "pops.test.dot-all/cluster", parent_layout.exact_identity(), options, {}, boxes.boxes()};
  std::array<int, kNativeDimension> ratio{};
  ratio.fill(2);
  sim.begin_restart_transaction();
  auto prepared = context->prepare_regrid(0, amr::RefinementRatio<kNativeDimension>(ratio),
      {boxes, std::move(identity)}, {.clustered_parent_layout = {16, 120},
      .fine_layout = {16, 120}, .load_balance = {16, 16, std::numeric_limits<std::int64_t>::max()}});
  const auto& parent = sim.engine()->hierarchy().state(0);
  MultiFab<kNativeDimension> child(prepared.fine_layout()->patches(),
      prepared.fine_layout()->distribution(), parent.local_rank(), parent.ncomp(), parent.ghosts());
  child.set_val(Real(2));
  context->publish_regrid(std::move(prepared), std::move(child));
  ASSERT_EQ(sim.n_levels(), 2);
  const double retained_cells = exact_cell_count<kNativeDimension>(n) -
      exact_cell_count<kNativeDimension>(2) + exact_cell_count<kNativeDimension>(4);
  auto& coarse = context->state(0);
  EXPECT_DOUBLE_EQ(context->dot_all(0, coarse, coarse), 4. * retained_cells);
  auto different_ghosts = coarse.ghosts();
  ++different_ghosts[0];
  MultiFab<kNativeDimension> foreign_ghosts(coarse.layout(), coarse.distribution(),
      coarse.local_rank(), coarse.ncomp(), different_ghosts);
  foreign_ghosts.set_val(Real(2));
  const bool rank_zero = context->prepared_execution_lane().rank() == 0;
  EXPECT_ANY_THROW(context->dot_all(0, coarse, rank_zero ? foreign_ghosts : coarse));
  // A covered coarse value is outside the contraction, even if it is NaN.
  const auto& coverage = sim.prepared_amr_block_level_coverage_mask(0, 0);
  for (std::size_t patch = 0; patch < coarse.local_size(); ++patch) {
    const auto values = coarse.fab(patch).view();
    const auto mask = coverage.fab(patch).view();
    for_each_cell(coarse.box(patch), [=] POPS_HD(const Index<kNativeDimension>& cell) {
      if (mask(cell, 0) < Real(.5))
        values(cell, 0) = std::numeric_limits<Real>::quiet_NaN();
    });
  }
  EXPECT_DOUBLE_EQ(context->dot_all(0, coarse, coarse), 4. * retained_cells);
  sim.rollback_restart_transaction();
}
