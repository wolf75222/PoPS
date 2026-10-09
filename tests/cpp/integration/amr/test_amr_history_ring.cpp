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

  explicit Fixture(bool restart_ready = false) : system(config()) {
    pops::test::install_amr_runtime_authority(system, "test.amr-history.fixture/runtime@1");
    system.install_block_state_route("tracer", "state/tracer");
    install_advection(system);
    system.set_conservative_state("tracer", std::vector<double>(cell_count(config().shape), 1.0));
    (void)system.engine();
    system.set_program_block_map({0});
    // A restart seals the Program's history/flux capacity, so the restart witnesses need
    // a real installed body and its declared budget before registering any history.
    context = restart_ready
                  ? pops::test::install_forward_euler_program_context(system, false, "clock.macro")
                  : pops::runtime::program::make_program_execution_provider(&system);
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

TEST(test_amr_history_ring, RestartRegridImageRejectsPendingStoreAndEndsWithItsTransaction) {
  constexpr int Dim = pops::kNativeDimension;
  Fixture<Dim> fixture(true);
  fixture.register_history();
  auto sample = fixture.context->scratch_state_like(fixture.context->state(0));
  sample.set_val(pops::Real(7));
  const auto epoch = fixture.system.checkpoint_topology_epoch();

  fixture.system.begin_restart_transaction();
  fixture.context->begin_step(0.125);
  fixture.context->store_history("tracer.rate", sample, 0);
  EXPECT_ANY_THROW(fixture.system.begin_restart_regrid_history_sequence());
  fixture.system.rollback_restart_transaction();
  EXPECT_EQ(fixture.system.checkpoint_topology_epoch(), epoch);
  EXPECT_FALSE(fixture.system.history_initialized("tracer.rate", 0));

  // An accepted sample is a valid source. Aborting its restart must release the frozen image;
  // the next restart is a new authority even when its topology epoch is numerically identical.
  fixture.context->begin_step(0.125);
  fixture.context->store_history("tracer.rate", sample, 0);
  fixture.context->rotate_histories("clock.macro");
  fixture.system.begin_restart_transaction();
  ASSERT_NO_THROW(fixture.system.begin_restart_regrid_history_sequence());
  fixture.system.rollback_restart_transaction();
  fixture.system.begin_restart_transaction();
  EXPECT_NO_THROW(fixture.system.begin_restart_regrid_history_sequence());
  fixture.system.commit_restart_transaction();
  fixture.system.finalize_restart_transaction();

  fixture.system.begin_restart_transaction();
  EXPECT_NO_THROW(fixture.system.begin_restart_regrid_history_sequence());
  fixture.system.rollback_restart_transaction();
  EXPECT_EQ(fixture.system.checkpoint_topology_epoch(), epoch);
  EXPECT_EQ(pops::reduce_min_local(fixture.context->history("tracer.rate", 1, 0)), pops::Real(7));
}

TEST(test_amr_history_ring, FrozenRestartImageRefusesLaterStoreRotationAndRestore) {
  constexpr int Dim = pops::kNativeDimension;
  Fixture<Dim> fixture(true);
  fixture.register_history();
  auto sample = fixture.context->scratch_state_like(fixture.context->state(0));
  fixture.context->begin_step(0.125);
  sample.set_val(pops::Real(7));
  fixture.context->store_history("tracer.rate", sample, 0);
  fixture.context->rotate_histories("clock.macro");
  const auto accepted = fixture.system.history_global("tracer.rate", 0, 0);
  const auto fill = fixture.system.history_fill_count("tracer.rate", 0);

  fixture.system.begin_restart_transaction();
  ASSERT_NO_THROW(fixture.system.begin_restart_regrid_history_sequence());
  fixture.context->begin_step(0.125);
  sample.set_val(pops::Real(19));
  EXPECT_THROW(fixture.context->store_history("tracer.rate", sample, 0), std::logic_error);
  EXPECT_THROW(fixture.context->rotate_histories("clock.macro"), std::logic_error);
  EXPECT_THROW(fixture.system.restore_history_fill_count("tracer.rate", 0, fill),
               std::logic_error);
  EXPECT_EQ(fixture.system.history_global("tracer.rate", 0, 0), accepted);
  EXPECT_EQ(fixture.system.history_fill_count("tracer.rate", 0), fill);
  ASSERT_NO_THROW(fixture.system.rollback_restart_transaction());

  // Normal authoring resumes once rollback releases the frozen-image authority.
  fixture.context->begin_step(0.125);
  EXPECT_NO_THROW(fixture.context->store_history("tracer.rate", sample, 0));
  EXPECT_NO_THROW(fixture.context->rotate_histories("clock.macro"));
  EXPECT_EQ(pops::reduce_min_local(fixture.context->history("tracer.rate", 1, 0)), pops::Real(19));
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


namespace {
constexpr int M2Dim = pops::kNativeDimension;
using M2System = pops::AmrSystem<M2Dim>;
using M2Context = pops::runtime::program::AmrProgramContext<M2Dim>;

struct M2AcceptedFacadeImage {
  std::uint64_t epoch;
  std::vector<pops::AmrPatch<M2Dim>> boxes;
  std::vector<std::uint8_t> state, program, history_bits, history_identity;
  std::vector<std::vector<std::string>> clocks;
  std::vector<int> history_fill;
  std::vector<double> history_dt;
  double time;
  int macro_step;
  bool operator==(const M2AcceptedFacadeImage&) const = default;
};

M2AcceptedFacadeImage m2_accepted_facade_image(M2System& system) {
  M2AcceptedFacadeImage image;
  image.epoch = system.checkpoint_topology_epoch();
  image.boxes = system.patch_boxes();
  image.state = system.checkpoint_state_carriers();
  image.program = system.program_accepted_state();
  image.clocks = system.program_clock_manifest();
  image.time = system.time();
  image.macro_step = system.macro_step();
  for (int level : system.history_levels("m2.accepted-history")) {
    image.history_fill.push_back(system.history_fill_count("m2.accepted-history", level));
    const auto identity = system.history_sample_identity("m2.accepted-history", level);
    image.history_identity.insert(image.history_identity.end(), identity.begin(), identity.end());
    for (int slot = 0; slot < system.history_depth("m2.accepted-history"); ++slot) {
      const auto values = system.history_global("m2.accepted-history", level, slot);
      for (const auto byte : std::as_bytes(std::span(values)))
        image.history_bits.push_back(std::to_integer<std::uint8_t>(byte));
      image.history_dt.push_back(system.history_slot_dt("m2.accepted-history", level, slot));
    }
  }
  return image;
}

struct M2FacadeFixture {
  M2System system;
  std::shared_ptr<M2Context> context;
  int reject_level = -1;
  std::array<int, 2> completed_levels{};
  bool coarse_candidate_changed = false;
  bool nonfinite_injected = false;

  static pops::AmrSystemConfig<M2Dim> configuration() {
    auto config = Fixture<M2Dim>::config();
    config.regrid_every = 0;
    for (int axis = 0; axis < M2Dim; ++axis) {
      config.transition_buffers.front()[axis] = 0;
      config.transition_lookaheads.front()[axis] = 0;
    }
    return config;
  }
  M2FacadeFixture() : system(configuration()) {
    pops::test::install_amr_runtime_authority(system, "tests.m2.facade/owned-runtime@1");
    system.set_temporal_relations({2}, {1}, {"integral_only"});
    system.install_block_state_route("tracer", "tests.m2.facade/tracer-state@1");
    install_advection(system);
    system.set_conservative_state("tracer",
                                  std::vector<double>(cell_count(configuration().shape), 1.0));
    pops::test::install_prepared_refine_coarsen_threshold(
        system, {"tracer", "u", .5, pops::test::PreparedThresholdRelation::Above},
        {"tracer", "u", .5, pops::test::PreparedThresholdRelation::Below},
        "tests.m2.facade/prepared-tagging@1");
    if (system.engine()->hierarchy().num_levels() != 2)
      throw std::runtime_error("M2 witness requires a genuinely materialized fine level");
    context = pops::runtime::program::make_program_execution_provider(&system);
    context->configure_primary_clock("clock.macro");
    context->declare_clock_relation("clock.macro", "clock.fast", 2);
    context->install(
        [this](double macro_dt) {
          context->advance_hierarchy(macro_dt, [&](double dt) {
            context->set_stage_time(0, 1);
            auto& state = context->state(0);
            auto& rhs = context->rhs_scratch(9700, 0, state);
            context->rhs_into(0, state, rhs, 9701);
            auto candidate = context->scratch_state_like(state);
            context->copy_grown_component_span(candidate, 0, state, 0, state.ncomp());
            context->axpy(candidate, pops::Real(dt), rhs);
            if (context->level() == reject_level) {
              candidate.set_val(std::numeric_limits<pops::Real>::quiet_NaN());
              nonfinite_injected = true;
            }
            context->commit_many({{&state, &candidate}});
            context->store_history("m2.accepted-history", state, 0);
            context->rotate_histories("clock.macro");
            ++completed_levels.at(static_cast<std::size_t>(context->level()));
            if (context->level() == 0)
              coarse_candidate_changed = pops::norm_inf(rhs) > pops::Real(0);
          });
        },
        context);
    system.set_program_block_map({0});
    using Budget = M2System::PreparedAmrProgramFluxExpressionBlockBudget;
    system.install_prepared_amr_program_flux_expression_budget(
        "tests.m2.facade/physical-advection-budget@1", std::vector<Budget>{{1, 1}}, 0, 0);
    context->for_each_program_resource_level([&](int) {
      context->register_history("m2.accepted-history", 2, 1, 0, "tests.m2.facade/tracer-state@1",
                                "cell.conservative", "clock.macro", "dense.linear");
    });
    system.step(1e-3);
    for (int level : system.history_levels("m2.accepted-history"))
      if (!system.history_initialized("m2.accepted-history", level))
        throw std::runtime_error("M2 accepted image must contain populated real history samples");
    completed_levels.fill(0);
  }
  void change_fine_topology_inside_attempt() {
    std::vector<double> partial(cell_count(configuration().shape), .25);
    std::size_t center = 0, stride = 1;
    for (int axis = 0; axis < M2Dim; ++axis) {
      center += static_cast<std::size_t>(configuration().shape[axis] / 2) * stride;
      stride *= static_cast<std::size_t>(configuration().shape[axis]);
    }
    partial[center] = 1;
    system.set_conservative_state("tracer", partial);
    system.execute_prepared_tagging(0);
    if (!system.regrid_from_prepared_tagging(0))
      throw std::runtime_error("M2 transaction witness did not publish a real changed topology");
  }
};
}  // namespace

TEST(test_amr_history_ring, AcceptedFacadeTransactionCommitsTopologyStateHistoryAndClock) {
  M2FacadeFixture fixture;
  const auto before = m2_accepted_facade_image(fixture.system);
  fixture.system.begin_step_transaction();
  fixture.change_fine_topology_inside_attempt();
  EXPECT_NE(fixture.system.patch_boxes(), before.boxes);
  EXPECT_GT(fixture.system.checkpoint_topology_epoch(), before.epoch);
  fixture.system.step(1e-3);
  EXPECT_GT(fixture.completed_levels[0], 0);
  EXPECT_GT(fixture.completed_levels[1], 0);
  fixture.system.commit_step_transaction();
  fixture.system.finalize_step_transaction();
  const auto committed = m2_accepted_facade_image(fixture.system);
  EXPECT_NE(committed.state, before.state);
  EXPECT_NE(committed.history_bits, before.history_bits);
  EXPECT_NE(committed.history_identity, before.history_identity);
  EXPECT_NE(committed.clocks, before.clocks);
  EXPECT_NE(committed.boxes, before.boxes);
  EXPECT_EQ(committed.macro_step, before.macro_step + 1);
  EXPECT_DOUBLE_EQ(committed.time, before.time + 1e-3);
  fixture.system.begin_step_transaction();
  fixture.system.rollback_step_transaction();
  EXPECT_EQ(m2_accepted_facade_image(fixture.system), committed);
}

TEST(test_amr_history_ring, RejectedFacadeAttemptRestoresTopologyStateHistoryAndClock) {
  M2FacadeFixture fixture;
  const auto before = m2_accepted_facade_image(fixture.system);
  fixture.system.begin_step_transaction();
  fixture.change_fine_topology_inside_attempt();
  ASSERT_NE(fixture.system.patch_boxes(), before.boxes);
  fixture.reject_level = 0;
  std::string refusal;
  try {
    fixture.system.step(1e-3);
    ADD_FAILURE() << "nonfinite native candidate was accepted";
  } catch (const std::exception& error) {
    refusal = error.what();
  }
  EXPECT_NE(refusal.find("prepared ND hyperbolic face evaluation refused publication status=1"),
            std::string::npos);
  RecordProperty("nonfinite_refusal", refusal);
  EXPECT_TRUE(fixture.nonfinite_injected);
  fixture.system.rollback_step_transaction();
  EXPECT_EQ(fixture.system.step_transaction_depth(), 0U);
  EXPECT_EQ(m2_accepted_facade_image(fixture.system), before);
}

TEST(test_amr_history_ring, FineNonFiniteAfterCoarseSuccessRestoresCompleteAcceptedState) {
  M2FacadeFixture fixture;
  const auto before = m2_accepted_facade_image(fixture.system);
  fixture.system.begin_step_transaction();
  fixture.change_fine_topology_inside_attempt();
  ASSERT_NE(fixture.system.patch_boxes(), before.boxes);
  fixture.reject_level = 1;
  std::string refusal;
  try {
    fixture.system.step(1e-3);
    ADD_FAILURE() << "nonfinite native candidate was accepted";
  } catch (const std::exception& error) {
    refusal = error.what();
  }
  EXPECT_NE(refusal.find("prepared ND hyperbolic face evaluation refused publication status=1"),
            std::string::npos);
  RecordProperty("nonfinite_refusal", refusal);
  EXPECT_TRUE(fixture.nonfinite_injected);
  EXPECT_GT(fixture.completed_levels[0], 0);
  EXPECT_TRUE(fixture.coarse_candidate_changed);
  fixture.system.rollback_step_transaction();
  EXPECT_EQ(fixture.system.step_transaction_depth(), 0U);
  EXPECT_EQ(m2_accepted_facade_image(fixture.system), before);
}
