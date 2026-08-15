#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"

#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/amr_program_context.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

template <int Dim>
struct AdvectionModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;

  Law law{};

  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.amr-history.scalar-advection", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(law.velocity()[axis]);
  }
  static pops::VariableSet conservative_vars() {
    return {pops::VariableKind::Conservative, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  static pops::VariableSet primitive_vars() {
    return {pops::VariableKind::Primitive, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  POPS_HD pops::nd::StateConversion<Primitive> recover(const State& state) const {
    return law.recover(state);
  }
  POPS_HD pops::nd::StateConversion<State> make_conservative(const Primitive& primitive) const {
    return law.make_conservative(primitive);
  }
  POPS_HD pops::nd::StateConversionStatus admissibility(const State& state) const {
    return law.admissibility(state);
  }
  template <int Axis>
  POPS_HD State flux(const State& state) const {
    return law.template flux<Axis>(state);
  }
  template <int Axis>
  POPS_HD pops::Real max_wave_speed(const State& state) const {
    return law.template max_wave_speed<Axis>(state);
  }
  template <int Axis>
  POPS_HD void wave_speeds(const State& state, pops::Real& lower, pops::Real& upper) const {
    law.template wave_speeds<Axis>(state, lower, upper);
  }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};

template <int Dim>
AdvectionModel<Dim> advection_model() {
  pops::RealVector<Dim> velocity{};
  for (int axis = 0; axis < Dim; ++axis)
    velocity[axis] = pops::Real(axis + 1);
  return {pops::nd::ScalarAdvection<Dim>::prepare(velocity)};
}

template <int Dim>
void install_advection(pops::AmrSystem<Dim>& system) {
  pops::add_compiled_model<Dim>(
      system, "tracer", advection_model<Dim>(), "minmod", "rusanov", "conservative", "explicit",
      static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
      static_cast<double>(pops::kWenoEpsilon), false, "test.amr-history/provider-free");
}

template <int Dim>
std::size_t cell_count(const pops::Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
struct Fixture {
  pops::AmrSystem<Dim> system;
  std::shared_ptr<pops::runtime::program::AmrProgramContext<Dim>> context;

  Fixture() : system(config()) {
    system.install_block_state_route("tracer", "state/tracer");
    install_advection(system);
    system.set_conservative_state("tracer", std::vector<double>(cell_count(config().shape), 1.0));
    (void)system.engine();
    system.set_program_block_map({0});
    context = pops::runtime::program::make_program_execution_provider(&system);
    context->configure_primary_clock("clock.macro");
    context->declare_clock_relation("clock.macro", "clock.fast", 2);
  }

  static pops::AmrSystemConfig<Dim> config() {
    pops::AmrSystemConfig<Dim> result;
    for (int axis = 0; axis < Dim; ++axis)
      result.shape[axis] = 8;
    result.level_count = 2;
    return result;
  }

  void register_history() {
    context->register_history("tracer.rate", 2, 1, 0, "tracer.U", "cell.conservative",
                              "clock.macro", "dense.linear");
  }
};

template <int Dim>
pops::AmrSystemConfig<Dim> two_level_config() {
  pops::AmrSystemConfig<Dim> result;
  result.level_count = 2;
  result.regrid_every = 0;
  result.explicit_bootstrap = true;
  result.transition_ratios.resize(1);
  result.transition_buffers.resize(1);
  result.transition_lookaheads.resize(1);
  for (int axis = 0; axis < Dim; ++axis) {
    result.shape[axis] = 8;
    for (std::size_t transition = 0; transition < 1; ++transition) {
      result.transition_ratios[transition][axis] = 2;
      result.transition_buffers[transition][axis] = 1;
      result.transition_lookaheads[transition][axis] = 1;
    }
  }
  return result;
}

}  // namespace

TEST(test_amr_history_ring, RetainsAndInterpolatesExactRankedState) {
  constexpr int Dim = pops::kNativeDimension;
  Fixture<Dim> fixture;
  fixture.register_history();

  pops::MultiFab<Dim> sample = fixture.context->scratch_state_like(fixture.context->state(0));
  fixture.context->begin_step(0.2);
  sample.set_val(pops::Real(10));
  fixture.context->store_history("tracer.rate", sample, 0);
  fixture.context->rotate_histories("clock.macro");

  fixture.context->begin_step(0.4);
  sample.set_val(pops::Real(20));
  fixture.context->store_history("tracer.rate", sample, 0);
  pops::MultiFab<Dim> interpolated = fixture.context->scratch_state_like(sample);
  interpolated.set_val(pops::Real(-1));
  fixture.context->interpolate_history_linear(interpolated, "tracer.rate", 2, 0, "clock.macro",
                                              "clock.fast", -1, pops::Real(0));

  EXPECT_EQ(pops::reduce_min_local(interpolated), pops::Real(15));
  EXPECT_EQ(pops::reduce_max_local(interpolated), pops::Real(15));
}

TEST(test_amr_history_ring, FacadeTransactionRestoresAcceptedHistoryImage) {
  constexpr int Dim = pops::kNativeDimension;
  Fixture<Dim> fixture;
  fixture.register_history();

  pops::MultiFab<Dim> sample = fixture.context->scratch_state_like(fixture.context->state(0));
  fixture.context->begin_step(0.1);
  sample.set_val(pops::Real(3));
  fixture.context->store_history("tracer.rate", sample, 0);
  fixture.context->rotate_histories("clock.macro");
  ASSERT_EQ(pops::reduce_min_local(fixture.context->history("tracer.rate", 1, 0)), pops::Real(3));

  fixture.system.begin_step_transaction();
  sample.set_val(pops::Real(9));
  fixture.context->store_history("tracer.rate", sample, 0);
  fixture.context->rotate_histories("clock.macro");
  ASSERT_EQ(pops::reduce_max_local(fixture.context->history("tracer.rate", 1, 0)), pops::Real(9));
  fixture.system.rollback_step_transaction();

  EXPECT_EQ(pops::reduce_min_local(fixture.context->history("tracer.rate", 1, 0)), pops::Real(3));
  EXPECT_EQ(pops::reduce_max_local(fixture.context->history("tracer.rate", 1, 0)), pops::Real(3));
}

TEST(test_amr_history_ring, RegisteredHistoryRejectsTopologyPublicationBeforeMutation) {
  constexpr int Dim = pops::kNativeDimension;
  Fixture<Dim> fixture;
  fixture.register_history();
  auto* engine = fixture.system.engine();
  ASSERT_NE(engine, nullptr);

  pops::Index<Dim> lower{};
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] = engine->hierarchy().layout(0).domain().lo[axis] + 2;
    upper[axis] = engine->hierarchy().layout(0).domain().hi[axis] - 2;
  }
  const pops::mesh::BoxArray<Dim> boxes(std::vector<pops::Box<Dim>>{pops::Box<Dim>{lower, upper}});
  pops::amr::tagging::ClusterOptions<Dim> options;
  options.min_efficiency = 0.7;
  for (int axis = 0; axis < Dim; ++axis) {
    options.min_box_size[static_cast<std::size_t>(axis)] = 1;
    options.max_box_size[static_cast<std::size_t>(axis)] = 16;
  }
  options.budget = {16, 256, 8192, 64, 1U << 20};
  pops::amr::tagging::ClusterResultIdentity<Dim> identity{
      "test.amr-history.cluster",
      engine->hierarchy().layout(0).exact_identity(),
      options,
      {},
      boxes.boxes()};
  pops::amr::tagging::ClusterResult<Dim> cluster(boxes, std::move(identity));
  std::array<int, Dim> ratio_components{};
  ratio_components.fill(2);
  const pops::amr::regridding::RegridPreparationBudget budget{
      .clustered_parent_layout = {16, 120},
      .fine_layout = {16, 120},
      .load_balance = {16, 16, std::numeric_limits<std::int64_t>::max()},
  };
  auto prepared = fixture.context->prepare_regrid(
      0, pops::amr::RefinementRatio<Dim>(ratio_components), std::move(cluster), budget);
  ASSERT_TRUE(prepared.fine_layout().has_value());
  pops::MultiFab<Dim> child(
      prepared.fine_layout()->patches(), prepared.fine_layout()->distribution(),
      engine->hierarchy().state(0).local_rank(), engine->hierarchy().state(0).ncomp(),
      engine->hierarchy().state(0).ghosts());

  EXPECT_THROW(fixture.context->publish_regrid(std::move(prepared), std::move(child)),
               std::runtime_error);
  EXPECT_EQ(engine->hierarchy().num_levels(), 1U);
}

TEST(test_amr_history_ring, TwoLevelProgramSynchronizesAtomicallyAndRetries) {
  constexpr int Dim = pops::kNativeDimension;
  const pops::AmrSystemConfig<Dim> config = two_level_config<Dim>();
  pops::AmrSystem<Dim> system(config);
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  system.install_block_state_route("tracer", "state/tracer");
  install_advection(system);
  std::vector<double> initial(cell_count(config.shape), 0.0);
  std::fill_n(initial.begin(), initial.size() / 2, 1.0);
  system.set_conservative_state("tracer", initial);
  pops::test::install_prepared_threshold_union(system, {{"tracer", "u", 0.5}},
                                               "test.amr-history.three-level-tagging@1");

  auto* runtime = system.engine();
  ASSERT_NE(runtime, nullptr);
  pops::Index<Dim> level_one_lo{};
  pops::Index<Dim> level_one_hi{};
  for (int axis = 0; axis < Dim; ++axis) {
    level_one_hi[axis] = axis == 0 ? 7 : 15;
  }
  system.rebuild_hierarchy({pops::AmrPatch<Dim>{1, {level_one_lo, level_one_hi}}}, {0});
  runtime = system.engine();
  ASSERT_EQ(runtime->hierarchy().num_levels(), 2U);
  for (std::size_t level = 1; level < runtime->hierarchy().num_levels(); ++level)
    for (int axis = 0; axis < Dim; ++axis)
      EXPECT_EQ(runtime->hierarchy().layout(level).ratio_from_parent()[axis], 2);
  EXPECT_EQ(system.checkpoint_temporal_relations(),
            (std::vector<std::vector<std::string>>{{"0", "1", "2", "1", "integral_only"}}));
  EXPECT_EQ(system.program_sync_manifest().size(), 1U);
  EXPECT_EQ(system.program_interface_flux_ledger_manifest().size(), 1U);

  system.set_program_block_map({0});
  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("clock.level.0");
  context->declare_clock_relation("clock.level.0", "clock.level.1", 2);

  std::size_t maximum_patches = 0;
  for (std::size_t level = 1; level < runtime->hierarchy().num_levels(); ++level)
    maximum_patches =
        std::max(maximum_patches, runtime->hierarchy().layout(level).patches().size());
  ASSERT_GT(maximum_patches, 0U);
  const std::size_t overlap_pairs = maximum_patches * (maximum_patches - 1U) / 2U;
  const std::array<int, 1> temporal_substeps{2};
  const auto plan =
      context->prepare_subcycling(std::span<const int>(temporal_substeps),
                                  {temporal_substeps.size(), {maximum_patches, overlap_pairs}});
  plan.require_live(*runtime);
  ASSERT_EQ(plan.size(), 1U);
  EXPECT_EQ(plan.transition(0).temporal_substeps(), 2);

  context->for_each_program_resource_level([&](int) {
    context->register_history("tracer.rate", 2, 1, 0, "tracer.U", "cell.conservative",
                              "clock.level.0", "dense.linear");
  });

  std::array<std::vector<double>, 2> accepted;
  for (int level = 0; level < 2; ++level)
    accepted[static_cast<std::size_t>(level)] = system.level_state_global(level);

  std::array<int, 2> rejected_advances{};
  std::vector<int> rejected_order;
  bool inject_failure = true;
  try {
    context->advance_synchronized_hierarchy(0.125, [&](double level_dt) {
      const int level = context->level();
      rejected_order.push_back(level);
      ++rejected_advances[static_cast<std::size_t>(level)];
      context->state(0).set_val(
          pops::Real(10 * (level + 1) + rejected_advances[static_cast<std::size_t>(level)]));
      auto rhs = context->rhs_scratch_like(context->state(0));
      context->rhs_into(0, context->state(0), rhs, 0);
      context->store_history("tracer.rate", context->state(0), 0);
      if (inject_failure && pops::n_ranks() > 1 && pops::my_rank() == 1 && level == 1 &&
          rejected_advances[1] == 1)
        throw std::runtime_error("rank-local synchronized hierarchy rejection");
      if (inject_failure && pops::n_ranks() == 1 && level == 1 && rejected_advances[1] == 1)
        throw std::runtime_error("serial synchronized hierarchy rejection");
      EXPECT_GT(level_dt, 0.0);
    });
  } catch (const std::runtime_error&) {
  }

  EXPECT_EQ(rejected_advances[0], 1);
  EXPECT_EQ(rejected_advances[1], 1);
  EXPECT_EQ(rejected_order, (std::vector<int>{0, 1}));
  for (int level = 0; level < 2; ++level)
    EXPECT_EQ(system.level_state_global(level), accepted[static_cast<std::size_t>(level)]);

  inject_failure = false;
  std::array<int, 2> accepted_advances{};
  std::vector<int> accepted_order;
  context->advance_synchronized_hierarchy(0.125, [&](double level_dt) {
    const int level = context->level();
    accepted_order.push_back(level);
    ++accepted_advances[static_cast<std::size_t>(level)];
    context->state(0).set_val(
        pops::Real(10 * (level + 1) + accepted_advances[static_cast<std::size_t>(level)]));
    auto rhs = context->rhs_scratch_like(context->state(0));
    context->rhs_into(0, context->state(0), rhs, 0);
    context->store_history("tracer.rate", context->state(0), 0);
    EXPECT_GT(level_dt, 0.0);
  });

  EXPECT_EQ(accepted_order, (std::vector<int>{0, 1, 1}));
  EXPECT_EQ(accepted_advances, (std::array<int, 2>{1, 2}));
  const std::vector<double> fine = system.level_state_global(1);
  ASSERT_FALSE(fine.empty());
  EXPECT_EQ(*std::max_element(fine.begin(), fine.end()), 22.0);
  EXPECT_NE(std::find(fine.begin(), fine.end(), 22.0), fine.end());
  const std::vector<double> coarse = system.level_state_global(0);
  ASSERT_FALSE(coarse.empty());
  EXPECT_NE(std::find(coarse.begin(), coarse.end(), 22.0), coarse.end());
  EXPECT_TRUE(std::any_of(coarse.begin(), coarse.end(), [](double value) {
    return value != 11.0 && value != 22.0;
  }));
}
