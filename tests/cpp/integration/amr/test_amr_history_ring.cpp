#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "explicit_amr_program.hpp"

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
  POPS_HD State source(const State&, const pops::ProviderValues<0>&) const { return {}; }
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
    pops::test::install_amr_runtime_authority(system, "test.amr-history.fixture/runtime@1");
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
pops::AmrSystemConfig<Dim> three_level_config() {
  pops::AmrSystemConfig<Dim> result;
  result.level_count = 3;
  result.regrid_every = 0;
  result.transition_ratios.resize(2);
  result.transition_buffers.resize(2);
  result.transition_lookaheads.resize(2);
  for (int axis = 0; axis < Dim; ++axis) {
    result.shape[axis] = 8;
    for (std::size_t transition = 0; transition < 2; ++transition) {
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

TEST(test_amr_history_ring, ThreeLevelProgramFailsClosedWithoutExactFluxExpressionBudget) {
  constexpr int Dim = pops::kNativeDimension;
  const pops::AmrSystemConfig<Dim> config = three_level_config<Dim>();
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "test.amr-history.three-level/runtime@1");
  system.set_temporal_relations({2, 2}, {1, 1}, {"integral_only", "integral_only"});
  system.install_block_state_route("tracer", "state/tracer");
  install_advection(system);
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  pops::test::install_prepared_threshold_union(system, {{"tracer", "u", 0.5}},
                                               "test.amr-history.three-level-tagging@1");

  auto* runtime = system.engine();
  ASSERT_NE(runtime, nullptr);
  ASSERT_EQ(runtime->hierarchy().num_levels(), 3U);
  for (std::size_t level = 1; level < runtime->hierarchy().num_levels(); ++level)
    for (int axis = 0; axis < Dim; ++axis)
      EXPECT_EQ(runtime->hierarchy().layout(level).ratio_from_parent()[axis], 2);
  EXPECT_EQ(system.checkpoint_temporal_relations(),
            (std::vector<std::vector<std::string>>{{"0", "1", "2", "1", "integral_only"},
                                                   {"1", "2", "2", "1", "integral_only"}}));
  EXPECT_TRUE(system.program_sync_manifest().empty());
  EXPECT_TRUE(system.program_interface_flux_ledger_manifest().empty());

  system.set_program_block_map({0});
  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("clock.level.0");
  context->declare_clock_relation("clock.level.0", "clock.level.1", 2);
  context->declare_clock_relation("clock.level.1", "clock.level.2", 2);

  std::size_t maximum_patches = 0;
  for (std::size_t level = 1; level < runtime->hierarchy().num_levels(); ++level)
    maximum_patches =
        std::max(maximum_patches, runtime->hierarchy().layout(level).patches().size());
  ASSERT_GT(maximum_patches, 0U);
  const std::size_t overlap_pairs = maximum_patches * (maximum_patches - 1U) / 2U;
  const std::array<int, 2> temporal_substeps{2, 2};
  const auto plan =
      context->prepare_subcycling(std::span<const int>(temporal_substeps),
                                  {temporal_substeps.size(), {maximum_patches, overlap_pairs}});
  plan.require_live(*runtime);
  ASSERT_EQ(plan.size(), 2U);
  EXPECT_EQ(plan.transition(0).temporal_substeps(), 2);
  EXPECT_EQ(plan.transition(1).temporal_substeps(), 2);
  EXPECT_EQ(plan.transition(0).temporal_substeps() * plan.transition(1).temporal_substeps(), 4);

  std::array<int, 3> level_advances{};
  std::string refusal;
  try {
    context->advance_synchronized_hierarchy(0.125, [&](double) {
      ++level_advances[static_cast<std::size_t>(context->level())];
      context->state(0).set_val(pops::Real(9));
    });
  } catch (const std::logic_error& error) {
    refusal = error.what();
  }

  EXPECT_EQ(refusal, "installed AMR Program has no prepared flux-expression budget");
  EXPECT_EQ(level_advances, (std::array<int, 3>{0, 0, 0}));
  for (std::size_t level = 0; level < runtime->hierarchy().num_levels(); ++level) {
    EXPECT_EQ(pops::reduce_min_local(runtime->hierarchy().state(level)), pops::Real(1));
    EXPECT_EQ(pops::reduce_max_local(runtime->hierarchy().state(level)), pops::Real(1));
  }
}
