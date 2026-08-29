#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "explicit_amr_program.hpp"

#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/program_execution_services.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

template <int Dim>
using AmrProgramExecutionServices = pops::runtime::program::ProgramExecutionServices<Dim>;

template <int Dim>
concept PublicAmrProgramLifecycleProvider =
    requires(const AmrProgramExecutionServices<Dim>& provider,
             const pops::runtime::program::AmrProgramHistoryRemapDescriptor& descriptor) {
      provider.refresh_accepted_hierarchy();
      provider.accept_history_remap(descriptor);
      provider.preflight_restart_regrid();
      provider.restart_regrid();
      provider.resync_after_restart();
      {
        provider.create_accepted_context_snapshot()
      } -> std::same_as<
          std::unique_ptr<pops::runtime::program::AcceptedProgramExecutionServicesSnapshot>>;
    };

static_assert(PublicAmrProgramLifecycleProvider<pops::kNativeDimension>);

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
std::vector<pops::test::program_v5::CallbackProgramResource> dense_resources(
    const pops::AmrSystem<Dim>& system,
    const std::vector<pops::test::program_v5::CallbackProgramResource::Kind>& kinds) {
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto program_blocks = system.program_block_map();
  if (program_blocks.empty() || kinds.empty())
    throw std::logic_error("history fixture requires an exact Program resource table");
  const auto accepted_runtime = system.accepted_amr_runtime();
  if (!accepted_runtime)
    throw std::logic_error("history fixture resource has no accepted runtime");
  const int level_count = static_cast<int>(accepted_runtime->hierarchy().num_levels());
  std::vector<Resource> resources;
  resources.reserve(static_cast<std::size_t>(level_count) * program_blocks.size() * kinds.size());
  for (int level = 0; level < level_count; ++level) {
    for (std::size_t program_block = 0; program_block < program_blocks.size(); ++program_block) {
      auto state_view = system.prepared_amr_block_state(program_blocks[program_block], level);
      if (!state_view)
        throw std::logic_error("history fixture resource has no accepted block state");
      const auto& state = *state_view;
      for (const auto kind : kinds)
        resources.push_back({kind, resources.size(), 0, static_cast<int>(program_block), level,
                             static_cast<std::uint32_t>(state.ncomp()),
                             static_cast<std::uint32_t>(state.ghosts()[0])});
    }
  }
  return resources;
}

template <int Dim>
struct Fixture {
  pops::AmrSystem<Dim> system;

  explicit Fixture(int level_count = 2) : system(config(level_count)) {
    pops::test::install_amr_runtime_authority(system, "test.amr-history.fixture/runtime@1");
    if (level_count > 1) {
      const auto transitions = static_cast<std::size_t>(level_count - 1);
      system.set_temporal_relations(std::vector<std::int64_t>(transitions, 2),
                                    std::vector<std::int64_t>(transitions, 1),
                                    std::vector<std::string>(transitions, "integral_only"));
    }
    system.install_block_state_route("tracer", "state/tracer");
    install_advection(system);
    system.set_conservative_state("tracer", std::vector<double>(cell_count(config().shape), 1.0));
    if (level_count > 1)
      pops::test::install_prepared_threshold_union(system, {{"tracer", "u", 0.5}},
                                                   "test.amr-history.fixture-tagging@1");
    (void)system.accepted_amr_runtime();
    system.set_program_block_map({0});
  }

  static pops::AmrSystemConfig<Dim> config(int level_count = 2) {
    pops::AmrSystemConfig<Dim> result;
    result.level_count = level_count;
    if (level_count == 1) {
      result.transition_ratios.clear();
      result.transition_buffers.clear();
      result.transition_lookaheads.clear();
    }
    for (int axis = 0; axis < Dim; ++axis)
      result.shape[axis] = 8;
    return result;
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

template <int Dim>
pops::AmrSystemConfig<Dim> cumulative_regrid_config() {
  pops::AmrSystemConfig<Dim> result;
  result.level_count = 4;
  result.regrid_every = 1;
  result.explicit_bootstrap = true;
  result.transition_ratios.resize(3);
  result.transition_buffers.resize(3);
  result.transition_lookaheads.resize(3);
  for (int axis = 0; axis < Dim; ++axis) {
    result.shape[axis] = 8;
    for (std::size_t transition = 0; transition < 3; ++transition) {
      result.transition_ratios[transition][axis] = 2;
      result.transition_buffers[transition][axis] = 1;
      result.transition_lookaheads[transition][axis] = 1;
    }
  }
  return result;
}

std::vector<std::uint8_t> cumulative_double_bytes(const std::vector<double>& values) {
  std::vector<std::uint8_t> result(values.size() * sizeof(double));
  if (!result.empty())
    std::memcpy(result.data(), values.data(), result.size());
  return result;
}

struct CumulativeProgramBlockMapSnapshot final {
  std::vector<std::size_t> canonical_indices;
  std::string hierarchy_contract;
  std::string exact_contract;

  friend bool operator==(const CumulativeProgramBlockMapSnapshot&,
                         const CumulativeProgramBlockMapSnapshot&) = default;
};

struct CumulativeFluxBudgetSnapshot final {
  std::string program_hash;
  std::uint64_t generation = 0;
  std::size_t interface_coupling_application_bound = 0;
  std::size_t interface_coupling_identity_character_bound = 0;
  CumulativeProgramBlockMapSnapshot program_block_map;
  std::vector<std::array<std::size_t, 2>> blocks;
  std::string exact_contract;

  friend bool operator==(const CumulativeFluxBudgetSnapshot&,
                         const CumulativeFluxBudgetSnapshot&) = default;
};

struct CumulativeLedgerBudgetSnapshot final {
  std::size_t max_fragments_per_window = 0;
  std::size_t max_payload_terms_per_window = 0;
  std::size_t max_transaction_depth = 0;
  std::size_t max_identity_characters = 0;
  std::string exact_contract;

  friend bool operator==(const CumulativeLedgerBudgetSnapshot&,
                         const CumulativeLedgerBudgetSnapshot&) = default;
};

struct CumulativeClockRelationSnapshot final {
  int parent_level = 0;
  int child_level = 0;
  std::int64_t numerator = 0;
  std::int64_t denominator = 1;
  bool integral_only = false;

  friend bool operator==(const CumulativeClockRelationSnapshot&,
                         const CumulativeClockRelationSnapshot&) = default;
};

struct CumulativeHistorySnapshot final {
  struct Level final {
    int level = 0;
    bool initialized = false;
    int fill_count = 0;
    std::vector<double> slot_dt;
    std::vector<std::vector<std::uint8_t>> slots;

    friend bool operator==(const Level&, const Level&) = default;
  };

  std::string name;
  int depth = 0;
  int ncomp = 0;
  std::vector<Level> levels;

  friend bool operator==(const CumulativeHistorySnapshot&,
                         const CumulativeHistorySnapshot&) = default;
};

struct CumulativeAmrSnapshot final {
  std::size_t level_count = 0;
  std::string spatial_contract;
  std::uint64_t topology_epoch = 0;
  std::uint64_t materialization_generation = 0;
  std::vector<std::vector<std::uint8_t>> state_bytes;
  std::uint64_t program_state_revision = 0;
  std::vector<std::uint8_t> program_bytes;
  std::vector<std::vector<std::string>> program_state_manifest;
  std::vector<std::vector<std::string>> program_clock_manifest;
  std::vector<std::vector<std::string>> program_temporal_partition_manifest;
  std::vector<std::vector<std::string>> program_flux_manifest;
  std::vector<std::vector<std::string>> program_interface_flux_manifest;
  std::vector<std::vector<std::string>> program_sync_manifest;
  std::vector<CumulativeHistorySnapshot> histories;
  int checkpoint_regrid_count = 0;
  std::uint64_t checkpoint_topology_epoch = 0;
  std::vector<std::vector<std::string>> checkpoint_temporal_relations;
  std::vector<CumulativeClockRelationSnapshot> prepared_temporal_relations;
  CumulativeProgramBlockMapSnapshot program_block_map;
  CumulativeFluxBudgetSnapshot flux_budget;
  CumulativeLedgerBudgetSnapshot ledger_budget;

  friend bool operator==(const CumulativeAmrSnapshot&, const CumulativeAmrSnapshot&) = default;
};

template <int Dim>
CumulativeAmrSnapshot cumulative_snapshot(pops::AmrSystem<Dim>& system) {
  CumulativeAmrSnapshot snapshot;
  // Never retain the lease returned by accepted_amr_runtime() while reading any other accepted
  // image.  All subsequent values are copied through value-returning public accessors.
  {
    const auto accepted = system.accepted_amr_runtime();
    if (!accepted)
      throw std::logic_error("cumulative AMR witness has no accepted runtime");
    snapshot.level_count = accepted->hierarchy().num_levels();
    snapshot.spatial_contract = std::string(accepted->spatial_contract());
    snapshot.topology_epoch = accepted->topology_epoch();
    snapshot.materialization_generation = accepted->materialization_generation();
  }

  snapshot.state_bytes.reserve(snapshot.level_count);
  for (std::size_t level = 0; level < snapshot.level_count; ++level)
    snapshot.state_bytes.push_back(cumulative_double_bytes(
        system.block_level_state_global("tracer", static_cast<int>(level))));
  snapshot.program_state_revision = system.program_accepted_state_revision();
  snapshot.program_bytes = system.program_accepted_state();
  snapshot.program_state_manifest = system.program_accepted_state_manifest();
  snapshot.program_clock_manifest = system.program_clock_manifest();
  snapshot.program_temporal_partition_manifest = system.program_temporal_partition_manifest();
  snapshot.program_flux_manifest = system.program_flux_ledger_manifest();
  snapshot.program_interface_flux_manifest = system.program_interface_flux_ledger_manifest();
  snapshot.program_sync_manifest = system.program_sync_manifest();

  for (const std::string& name : system.history_names()) {
    CumulativeHistorySnapshot history;
    history.name = name;
    history.depth = system.history_depth(name);
    history.ncomp = system.history_ncomp(name);
    for (const int level : system.history_levels(name)) {
      CumulativeHistorySnapshot::Level image;
      image.level = level;
      image.initialized = system.history_initialized(name, level);
      image.fill_count = system.history_fill_count(name, level);
      image.slot_dt.reserve(static_cast<std::size_t>(history.depth));
      image.slots.reserve(static_cast<std::size_t>(history.depth));
      for (int slot = 0; slot < history.depth; ++slot) {
        image.slot_dt.push_back(system.history_slot_dt(name, level, slot));
        image.slots.push_back(cumulative_double_bytes(system.history_global(name, level, slot)));
      }
      history.levels.push_back(std::move(image));
    }
    snapshot.histories.push_back(std::move(history));
  }

  snapshot.checkpoint_regrid_count = system.checkpoint_regrid_count();
  snapshot.checkpoint_topology_epoch = system.checkpoint_topology_epoch();
  snapshot.checkpoint_temporal_relations = system.checkpoint_temporal_relations();
  for (const auto& relation : system.prepared_program_temporal_relations()) {
    const auto ratio = relation.temporal_ratio();
    snapshot.prepared_temporal_relations.push_back(
        {relation.parent_level(), relation.child_level(), ratio.numerator, ratio.denominator,
         relation.remainder_policy() == pops::amr::RemainderPolicy::IntegralOnly});
  }

  const auto program_block_map = system.prepared_amr_program_block_map();
  snapshot.program_block_map = {program_block_map.canonical_indices,
                                program_block_map.hierarchy_contract,
                                program_block_map.exact_contract};
  const auto flux_budget = system.prepared_amr_program_flux_expression_budget();
  snapshot.flux_budget.program_hash = flux_budget.program_hash;
  snapshot.flux_budget.generation = flux_budget.generation;
  snapshot.flux_budget.interface_coupling_application_bound =
      flux_budget.interface_coupling_application_bound;
  snapshot.flux_budget.interface_coupling_identity_character_bound =
      flux_budget.interface_coupling_identity_character_bound;
  snapshot.flux_budget.program_block_map = {flux_budget.program_block_map.canonical_indices,
                                            flux_budget.program_block_map.hierarchy_contract,
                                            flux_budget.program_block_map.exact_contract};
  snapshot.flux_budget.blocks.reserve(flux_budget.blocks.size());
  for (const auto& block : flux_budget.blocks)
    snapshot.flux_budget.blocks.push_back({block.rhs_basis_bound, block.coefficient_term_bound});
  snapshot.flux_budget.exact_contract = flux_budget.exact_contract;
  const auto ledger_budget = system.prepared_amr_interface_flux_ledger_budget();
  snapshot.ledger_budget = {ledger_budget.max_fragments_per_window,
                            ledger_budget.max_payload_terms_per_window,
                            ledger_budget.max_transaction_depth,
                            ledger_budget.max_identity_characters, ledger_budget.exact_contract};
  return snapshot;
}

void expect_cumulative_snapshot_equal(const CumulativeAmrSnapshot& expected,
                                      const CumulativeAmrSnapshot& actual) {
  EXPECT_EQ(actual.level_count, expected.level_count);
  EXPECT_EQ(actual.topology_epoch, expected.topology_epoch);
  EXPECT_EQ(actual.materialization_generation, expected.materialization_generation);
  EXPECT_EQ(actual.checkpoint_regrid_count, expected.checkpoint_regrid_count);
  EXPECT_EQ(actual.checkpoint_topology_epoch, expected.checkpoint_topology_epoch);
  EXPECT_EQ(actual.program_state_revision, expected.program_state_revision);
  EXPECT_EQ(actual.program_bytes, expected.program_bytes);
  EXPECT_EQ(actual.state_bytes, expected.state_bytes);
  EXPECT_TRUE(actual == expected);
}

}  // namespace

TEST(test_amr_history_ring, RetainsAndInterpolatesExactRankedState) {
  constexpr int Dim = pops::kNativeDimension;
  Fixture<Dim> fixture;
  using Resource = pops::test::program_v5::CallbackProgramResource;
  using History = pops::test::program_v5::CallbackProgramHistory;
  using ClockRelation = pops::test::program_v5::CallbackProgramClockRelation;
  const auto resources =
      dense_resources<Dim>(fixture.system, {Resource::Kind::rhs, Resource::Kind::state});
  struct Observation {
    int dispatches = 0;
    pops::Real interpolated_min = pops::Real(-1);
    pops::Real interpolated_max = pops::Real(-1);
  };
  auto observation = std::make_shared<Observation>();
  pops::test::install_explicit_amr_callback_program<Dim>(
      fixture.system, "test.amr-history/ring@1", "clock.macro", resources, {},
      [observation](auto& context, double dt) {
        context.begin_step(dt);
        auto& sample = context.rhs_scratch(0, 0, context.state(0));
        sample.set_val(observation->dispatches == 0 ? pops::Real(10) : pops::Real(20));
        context.store_history("tracer.rate", sample, 0);
        if (observation->dispatches == 0) {
          context.rotate_histories("clock.macro");
        } else {
          auto& interpolated = context.scratch_state(1, 0, sample);
          interpolated.set_val(pops::Real(-1));
          context.interpolate_history_linear(interpolated, "tracer.rate", 2, 0, "clock.macro",
                                             "clock.fast", -1, pops::Real(0));
          observation->interpolated_min = pops::reduce_min_local(interpolated);
          observation->interpolated_max = pops::reduce_max_local(interpolated);
        }
        ++observation->dispatches;
      },
      {History{"tracer.rate", 2, 1, 0, "tracer.U", "cell.conservative", "clock.macro",
               "dense.linear"}},
      {ClockRelation{"clock.macro", "clock.fast", 2}});
  ASSERT_NO_THROW(fixture.system.step(0.2));
  ASSERT_NO_THROW(fixture.system.step(0.4));
  EXPECT_EQ(observation->interpolated_min, pops::Real(15));
  EXPECT_EQ(observation->interpolated_max, pops::Real(15));
}

TEST(test_amr_history_ring, RejectedFacadeAttemptRestoresTopologyStateHistoryAndClock) {
  constexpr int Dim = pops::kNativeDimension;
  Fixture<Dim> fixture(1);
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(fixture.system, {Resource::Kind::rhs});
  struct Observation {
    int dispatches = 0;
    pops::Real provisional = pops::Real(-1);
  };
  auto observation = std::make_shared<Observation>();
  pops::test::install_explicit_amr_callback_program<Dim>(
      fixture.system, "test.amr-history/transaction@1", "clock.macro", resources, {},
      [observation](auto& context, double dt) {
        context.begin_step(dt);
        context.state(0).set_val(observation->dispatches == 0 ? pops::Real(3) : pops::Real(9));
        observation->provisional = pops::reduce_max_local(context.state(0));
        ++observation->dispatches;
      });
  ASSERT_NO_THROW(fixture.system.step(0.1));
  ASSERT_EQ(fixture.system.block_level_state_global("tracer", 0).front(), 3.0);
  std::size_t accepted_level_count = 0;
  {
    const auto accepted_before = fixture.system.accepted_amr_runtime();
    ASSERT_TRUE(accepted_before);
    accepted_level_count = accepted_before->hierarchy().num_levels();
  }
  EXPECT_EQ(fixture.system.macro_step(), 1);
  EXPECT_DOUBLE_EQ(fixture.system.time(), 0.1);

  fixture.system.begin_step_transaction();
  ASSERT_NO_THROW(fixture.system.step(0.1));
  EXPECT_EQ(observation->provisional, pops::Real(9));
  fixture.system.rollback_step_transaction();

  EXPECT_EQ(fixture.system.block_level_state_global("tracer", 0).front(), 3.0);
  const auto accepted_after = fixture.system.accepted_amr_runtime();
  ASSERT_TRUE(accepted_after);
  EXPECT_EQ(accepted_after->hierarchy().num_levels(), accepted_level_count);
  EXPECT_EQ(fixture.system.macro_step(), 1);
  EXPECT_DOUBLE_EQ(fixture.system.time(), 0.1);
}

TEST(test_amr_history_ring, AcceptedFacadeTransactionCommitsTopologyStateHistoryAndClock) {
  constexpr int Dim = pops::kNativeDimension;
  Fixture<Dim> fixture(1);
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(fixture.system, {Resource::Kind::rhs});
  struct Observation {
    int dispatches = 0;
  };
  auto observation = std::make_shared<Observation>();
  pops::test::install_explicit_amr_callback_program<Dim>(
      fixture.system, "test.amr-history/transaction-commit@1", "clock.macro", resources, {},
      [observation](auto& context, double dt) {
        context.begin_step(dt);
        const pops::Real accepted_value =
            observation->dispatches == 0 ? pops::Real(3) : pops::Real(9);
        context.state(0).set_val(accepted_value);
        ++observation->dispatches;
      });

  ASSERT_NO_THROW(fixture.system.step(0.1));
  std::size_t accepted_level_count = 0;
  {
    const auto accepted_before = fixture.system.accepted_amr_runtime();
    ASSERT_TRUE(accepted_before);
    accepted_level_count = accepted_before->hierarchy().num_levels();
  }
  ASSERT_EQ(fixture.system.block_level_state_global("tracer", 0).front(), 3.0);

  ASSERT_NO_THROW(fixture.system.step(0.1));

  EXPECT_EQ(fixture.system.block_level_state_global("tracer", 0).front(), 9.0);
  const auto accepted_after = fixture.system.accepted_amr_runtime();
  ASSERT_TRUE(accepted_after);
  EXPECT_EQ(accepted_after->hierarchy().num_levels(), accepted_level_count);
  EXPECT_EQ(fixture.system.macro_step(), 2);
  EXPECT_DOUBLE_EQ(fixture.system.time(), 0.2);
}

TEST(test_amr_history_ring, FineNonFiniteAfterCoarseSuccessRestoresCompleteAcceptedState) {
  constexpr int Dim = pops::kNativeDimension;
  Fixture<Dim> fixture;
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(fixture.system, {Resource::Kind::state});
  struct Observation {
    int accepted_attempts = 0;
    std::array<int, 2> rejected_attempt_levels{};
  };
  auto observation = std::make_shared<Observation>();
  pops::test::install_explicit_amr_callback_program<Dim>(
      fixture.system, "test.amr-history/fine-nonfinite-rollback@1", "clock.macro", resources, {},
      [observation](auto& context, double dt) {
        context.advance_hierarchy(dt, [&context, observation](double) {
          auto& state = context.state(0);
          if (observation->accepted_attempts == 0) {
            state.set_val(pops::Real(3));
            return;
          }
          if (context.level() == 0) {
            state.set_val(pops::Real(9));
            ++observation->rejected_attempt_levels[0];
            return;
          }
          state.set_val(std::numeric_limits<pops::Real>::quiet_NaN());
          ++observation->rejected_attempt_levels[1];
          throw pops::runtime::program::StepAttemptRejected(
              pops::SolveStatus::kInvalidEvaluation,
              pops::runtime::program::StepAttemptDisposition::kReject, 668, "fine-stage",
              "fine state is non-finite after the coarse candidate succeeded");
        });
        ++observation->accepted_attempts;
      });

  ASSERT_NO_THROW(fixture.system.step(0.1));
  ASSERT_EQ(fixture.system.macro_step(), 1);
  ASSERT_DOUBLE_EQ(fixture.system.time(), 0.1);
  std::size_t accepted_level_count = 0;
  {
    const auto accepted_runtime = fixture.system.accepted_amr_runtime();
    ASSERT_TRUE(accepted_runtime);
    accepted_level_count = accepted_runtime->hierarchy().num_levels();
  }
  ASSERT_EQ(accepted_level_count, 2U);

  try {
    fixture.system.step(0.1);
    FAIL() << "non-finite fine candidate was accepted";
  } catch (const pops::runtime::program::StepAttemptRejected& rejected) {
    EXPECT_EQ(rejected.status(), pops::SolveStatus::kInvalidEvaluation);
    EXPECT_EQ(rejected.disposition(), pops::runtime::program::StepAttemptDisposition::kReject);
    EXPECT_EQ(rejected.phase(), "fine-stage");
  }

  EXPECT_EQ(observation->rejected_attempt_levels, (std::array<int, 2>{1, 1}));
  EXPECT_EQ(fixture.system.macro_step(), 1);
  EXPECT_DOUBLE_EQ(fixture.system.time(), 0.1);
  for (std::size_t level = 0; level < accepted_level_count; ++level) {
    const auto state = fixture.system.block_level_state_global("tracer", static_cast<int>(level));
    ASSERT_FALSE(state.empty());
    EXPECT_EQ(*std::min_element(state.begin(), state.end()), 3.0);
    EXPECT_EQ(*std::max_element(state.begin(), state.end()), 3.0);
  }
  const auto restored_runtime = fixture.system.accepted_amr_runtime();
  ASSERT_TRUE(restored_runtime);
  EXPECT_EQ(restored_runtime->hierarchy().num_levels(), accepted_level_count);
}

TEST(test_amr_history_ring, ProviderCreatesAcceptedSnapshotWithoutInstallingRuntimeHooks) {
  constexpr int Dim = pops::kNativeDimension;
  Fixture<Dim> fixture;
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(fixture.system, {Resource::Kind::state});
  struct Observation {
    bool snapshot = false;
    bool restored = false;
  };
  auto observation = std::make_shared<Observation>();
  // Snapshot creation is a Program operation and therefore must be dispatched by the installed
  // ABI-v5 image.  Keep the named snapshot local to the callback so no service pointer escapes.
  pops::test::install_explicit_amr_callback_program<Dim>(
      fixture.system, "test.amr-history/snapshot-real@1", "clock.macro", resources, {},
      [observation](auto& context, double dt) {
        context.begin_step(dt);
        auto snapshot = context.create_accepted_context_snapshot();
        observation->snapshot = static_cast<bool>(snapshot);
        if (snapshot)
          observation->restored = static_cast<bool>(snapshot->prepare_restore());
      });
  ASSERT_NO_THROW(fixture.system.step(0.01));
  EXPECT_TRUE(observation->snapshot);
  EXPECT_TRUE(observation->restored);
}

TEST(test_amr_history_ring, RegisteredHistoryRejectsTopologyPublicationBeforeMutation) {
  constexpr int Dim = pops::kNativeDimension;
  Fixture<Dim> fixture;
  using Resource = pops::test::program_v5::CallbackProgramResource;
  using History = pops::test::program_v5::CallbackProgramHistory;
  const auto resources = dense_resources<Dim>(fixture.system, {Resource::Kind::state});
  pops::Index<Dim> lower{};
  pops::Index<Dim> upper{};
  pops::amr::hierarchy::LevelLayoutIdentity<Dim> accepted_layout_identity{};
  {
    const auto accepted_runtime = fixture.system.accepted_amr_runtime();
    ASSERT_TRUE(accepted_runtime);
    for (int axis = 0; axis < Dim; ++axis) {
      lower[axis] = accepted_runtime->hierarchy().layout(0).domain().lo[axis] + 2;
      upper[axis] = accepted_runtime->hierarchy().layout(0).domain().hi[axis] - 2;
    }
    accepted_layout_identity = accepted_runtime->hierarchy().layout(0).exact_identity();
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
      "test.amr-history.cluster", accepted_layout_identity, options, {}, boxes.boxes()};
  auto cluster =
      std::make_shared<pops::amr::tagging::ClusterResult<Dim>>(boxes, std::move(identity));
  std::array<int, Dim> ratio_components{};
  ratio_components.fill(2);
  const pops::amr::regridding::RegridPreparationBudget budget{
      .clustered_parent_layout = {16, 120},
      .fine_layout = {16, 120},
      .load_balance = {16, 16, std::numeric_limits<std::int64_t>::max()},
  };
  struct Observation {
    bool rejected = false;
  };
  auto observation = std::make_shared<Observation>();
  pops::test::install_explicit_amr_callback_program<Dim>(
      fixture.system, "test.amr-history/regrid-publication@1", "clock.macro", resources, {},
      [cluster, ratio_components, budget, observation](auto& context, double dt) {
        context.begin_step(dt);
        auto prepared = context.prepare_regrid(0, pops::amr::RefinementRatio<Dim>(ratio_components),
                                               std::move(*cluster), budget);
        if (!prepared.fine_layout())
          throw std::logic_error("history fixture did not prepare a fine layout");
        auto& parent = context.state(0);
        pops::MultiFab<Dim> child(prepared.fine_layout()->patches(),
                                  prepared.fine_layout()->distribution(), parent.local_rank(),
                                  parent.ncomp(), parent.ghosts());
        try {
          context.publish_regrid(std::move(prepared), std::move(child));
        } catch (const std::runtime_error&) {
          observation->rejected = true;
        }
      },
      {History{"tracer.rate", 2, 1, 0, "tracer.U", "cell.conservative", "clock.macro",
               "dense.linear"}});
  ASSERT_NO_THROW(fixture.system.step(0.01));
  EXPECT_TRUE(observation->rejected);
  auto after_runtime = fixture.system.accepted_amr_runtime();
  ASSERT_TRUE(after_runtime);
  EXPECT_EQ(after_runtime->hierarchy().num_levels(), 1U);
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

  {
    auto runtime = system.accepted_amr_runtime();
    ASSERT_TRUE(runtime);
    ASSERT_EQ(runtime->hierarchy().num_levels(), 3U);
    for (std::size_t level = 1; level < runtime->hierarchy().num_levels(); ++level)
      for (int axis = 0; axis < Dim; ++axis)
        EXPECT_EQ(runtime->hierarchy().layout(level).ratio_from_parent()[axis], 2);
  }
  EXPECT_EQ(system.checkpoint_temporal_relations(),
            (std::vector<std::vector<std::string>>{{"0", "1", "2", "1", "integral_only"},
                                                   {"1", "2", "2", "1", "integral_only"}}));
  EXPECT_TRUE(system.program_sync_manifest().empty());
  EXPECT_TRUE(system.program_interface_flux_ledger_manifest().empty());

  system.set_program_block_map({0});
  using Resource = pops::test::program_v5::CallbackProgramResource;
  using ClockRelation = pops::test::program_v5::CallbackProgramClockRelation;
  const auto resources = dense_resources<Dim>(system, {Resource::Kind::state});
  const std::array<int, 2> temporal_substeps{2, 2};
  struct Observation {
    std::array<int, 3> level_advances{};
    std::string refusal;
    std::size_t plan_size = 0;
    int first_substeps = 0;
    int second_substeps = 0;
  };
  auto observation = std::make_shared<Observation>();
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "test.amr-history/three-level@1", "clock.level.0", resources, {},
      [observation, temporal_substeps](auto& context, double dt) {
        // The synchronized advance owns the exact prepared subcycle plan.  The requested
        // two-transition shape is recorded here as a carrier; execution is expected to refuse
        // before entering any level body because this artifact intentionally has no flux budget.
        observation->plan_size = temporal_substeps.size();
        observation->first_substeps = temporal_substeps[0];
        observation->second_substeps = temporal_substeps[1];
        try {
          context.advance_synchronized_hierarchy(dt, [&context, observation](double) {
            ++observation->level_advances[static_cast<std::size_t>(context.level())];
            context.state(0).set_val(pops::Real(9));
          });
        } catch (const std::logic_error& error) {
          observation->refusal = error.what();
        }
      },
      {},
      {ClockRelation{"clock.level.0", "clock.level.1", 2},
       ClockRelation{"clock.level.1", "clock.level.2", 2}},
      std::vector<pops::runtime::program::ProgramFluxBudgetRecord>{});
  ASSERT_NO_THROW(system.step(0.125));

  EXPECT_EQ(observation->refusal, "installed AMR Program has no prepared flux-expression budget");
  EXPECT_EQ(observation->level_advances, (std::array<int, 3>{0, 0, 0}));
  EXPECT_EQ(observation->plan_size, 2U);
  EXPECT_EQ(observation->first_substeps, 2);
  EXPECT_EQ(observation->second_substeps, 2);
  auto after_runtime = system.accepted_amr_runtime();
  ASSERT_TRUE(after_runtime);
  for (std::size_t level = 0; level < after_runtime->hierarchy().num_levels(); ++level) {
    EXPECT_EQ(pops::reduce_min_local(after_runtime->hierarchy().state(level)), pops::Real(1));
    EXPECT_EQ(pops::reduce_max_local(after_runtime->hierarchy().state(level)), pops::Real(1));
  }
}

TEST(test_amr_history_ring, CumulativeRegridRetryPublishesFourLevelAcceptedAuthority) {
  constexpr int Dim = pops::kNativeDimension;
  auto config = cumulative_regrid_config<Dim>();
  config.regrid_every = 4;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "test.amr-history.cumulative/runtime@1");
  system.set_temporal_relations({2, 2, 2}, {1, 1, 1},
                                {"integral_only", "integral_only", "integral_only"});
  system.install_block_state_route("tracer", "state/tracer");
  install_advection(system);
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  pops::test::install_prepared_threshold_union(system, {{"tracer", "u", 0.5}},
                                               "test.amr-history.cumulative/tagging@1");

  {
    const auto accepted = system.accepted_amr_runtime();
    ASSERT_TRUE(accepted);
    ASSERT_EQ(accepted->hierarchy().num_levels(), 1U);
  }
  system.set_program_block_map({0});

  using Resource = pops::test::program_v5::CallbackProgramResource;
  using History = pops::test::program_v5::CallbackProgramHistory;
  using ClockRelation = pops::test::program_v5::CallbackProgramClockRelation;
  const auto resources = dense_resources<Dim>(system, {Resource::Kind::rhs, Resource::Kind::state});
  struct Observation {
    int dispatches = 0;
    pops::Real state_value = pops::Real(0);
  };
  auto observation = std::make_shared<Observation>();
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "test.amr-history.cumulative/program@1", "clock.level.0", resources, {},
      [observation](auto& context, double dt) {
        context.begin_step(dt);
        const pops::Real value = observation->state_value;
        context.state(0).set_val(value);
        auto& rhs = context.rhs_scratch(0, 0, context.state(0));
        rhs.set_val(value);
        context.store_history("tracer.rate", rhs, 0);
        context.rotate_histories("clock.level.0");
        ++observation->dispatches;
      },
      // `CallbackProgramHistory::depth` is the maximum lag accepted by the Program API.  The
      // runtime therefore materializes two slots here: current and lag one.
      {History{"tracer.rate", 1, 1, 0, "tracer.U", "cell.conservative", "clock.level.0",
               "dense.linear"}},
      {ClockRelation{"clock.level.0", "clock.level.1", 2},
       ClockRelation{"clock.level.1", "clock.level.2", 2},
       ClockRelation{"clock.level.2", "clock.level.3", 2}});

  // An AB2 deferred child lag is only valid after two *accepted* source samples.  Keep the
  // tagger in its required pre-materialization position, but write below its threshold while
  // warming the single-level history.  The following transaction is the sole cumulative
  // 0→1→2→3 regrid witness.
  constexpr double dt = 0.125;
  ASSERT_NO_THROW(system.step(dt));
  ASSERT_NO_THROW(system.step(dt));
  ASSERT_EQ(observation->dispatches, 2);
  ASSERT_EQ(system.history_fill_count("tracer.rate", 0), 2);
  // The tagger’s hierarchy preparation reads the accepted Program state rather than a facade-only
  // block write.  Publish one no-regrid Program step at the tagged value, then run the single
  // cumulative 0→1→2→3 transaction at the next regrid cadence.
  observation->state_value = pops::Real(3);
  ASSERT_NO_THROW(system.step(dt));
  ASSERT_EQ(observation->dispatches, 3);
  ASSERT_EQ(system.history_fill_count("tracer.rate", 0), 2);
  ASSERT_EQ(system.n_levels(), 1U);
  {
    const auto accepted_after_program_step = system.accepted_amr_runtime();
    ASSERT_TRUE(accepted_after_program_step);
    EXPECT_EQ(pops::reduce_max_local(accepted_after_program_step->hierarchy().state(0)),
              pops::Real(3));
  }
  observation->dispatches = 0;

  const auto baseline = cumulative_snapshot(system);
  ASSERT_EQ(baseline.level_count, 1U);
  ASSERT_EQ(baseline.histories.size(), 1U);
  ASSERT_EQ(system.history_depth("tracer.rate"), 2);
  ASSERT_EQ(baseline.histories.front().depth, 2);
  ASSERT_EQ(baseline.histories.front().levels.size(), 1U);
  ASSERT_EQ(baseline.histories.front().levels.front().slots.size(), 2U);
  ASSERT_EQ(baseline.histories.front().levels.front().slot_dt.size(), 2U);
  ASSERT_EQ(baseline.program_block_map.canonical_indices, (std::vector<std::size_t>{0}));
  ASSERT_FALSE(baseline.program_bytes.empty());
  const auto baseline_state = system.block_level_state_global("tracer", 0);
  ASSERT_FALSE(baseline_state.empty());
  EXPECT_TRUE(std::all_of(baseline_state.begin(), baseline_state.end(),
                          [](double value) { return value == 3.0; }));

  ASSERT_NO_THROW(system.begin_step_transaction());
  ASSERT_NO_THROW(system.step(dt));
  ASSERT_EQ(observation->dispatches, 1);
  ASSERT_NO_THROW(system.rollback_step_transaction());
  const auto rejected = cumulative_snapshot(system);
  expect_cumulative_snapshot_equal(baseline, rejected);

  ASSERT_NO_THROW(system.begin_step_transaction());
  ASSERT_NO_THROW(system.step(dt));
  ASSERT_EQ(observation->dispatches, 2);
  ASSERT_NO_THROW(system.commit_step_transaction());
  ASSERT_NO_THROW(system.finalize_step_transaction());

  const auto accepted = cumulative_snapshot(system);
  ASSERT_EQ(accepted.level_count, 4U);
  EXPECT_EQ(accepted.topology_epoch, baseline.topology_epoch + 3U);
  EXPECT_EQ(accepted.materialization_generation, baseline.materialization_generation + 3U);
  EXPECT_EQ(accepted.checkpoint_regrid_count, baseline.checkpoint_regrid_count + 3);
  EXPECT_EQ(accepted.checkpoint_topology_epoch, baseline.checkpoint_topology_epoch + 3U);
  EXPECT_FALSE(accepted.spatial_contract.empty());
  EXPECT_NE(accepted.program_bytes, baseline.program_bytes);
  EXPECT_EQ(accepted.checkpoint_temporal_relations,
            (std::vector<std::vector<std::string>>{{"0", "1", "2", "1", "integral_only"},
                                                   {"1", "2", "2", "1", "integral_only"},
                                                   {"2", "3", "2", "1", "integral_only"}}));
  EXPECT_EQ(accepted.prepared_temporal_relations,
            (std::vector<CumulativeClockRelationSnapshot>{
                {0, 1, 2, 1, true}, {1, 2, 2, 1, true}, {2, 3, 2, 1, true}}));
  EXPECT_EQ(accepted.program_block_map.canonical_indices, (std::vector<std::size_t>{0}));
  EXPECT_EQ(accepted.flux_budget.program_block_map, accepted.program_block_map);
  EXPECT_EQ(accepted.flux_budget.program_hash, system.installed_program_hash());
  EXPECT_EQ(accepted.flux_budget.generation, accepted.materialization_generation);
  EXPECT_FALSE(accepted.program_block_map.hierarchy_contract.empty());
  EXPECT_FALSE(accepted.program_block_map.exact_contract.empty());
  EXPECT_EQ(accepted.ledger_budget.max_transaction_depth, 1U);
  EXPECT_FALSE(accepted.ledger_budget.exact_contract.empty());
  ASSERT_EQ(accepted.flux_budget.blocks.size(), 1U);
  EXPECT_EQ(accepted.flux_budget.blocks.front(), (std::array<std::size_t, 2>{1, 1}));
  EXPECT_FALSE(accepted.program_state_manifest.empty());
  EXPECT_FALSE(accepted.program_clock_manifest.empty());
  const auto accepted_program =
      pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(accepted.program_bytes);
  ASSERT_EQ(accepted_program.pending_history_remaps.size(), 3U);
  for (std::size_t parent_level = 0; parent_level < accepted_program.pending_history_remaps.size();
       ++parent_level) {
    const auto& marker = accepted_program.pending_history_remaps[parent_level];
    EXPECT_EQ(marker.parent_level, static_cast<int>(parent_level));
    EXPECT_EQ(marker.child_level, static_cast<int>(parent_level + 1U));
    EXPECT_EQ(marker.prior_topology_epoch + 1U, marker.published_topology_epoch);
    EXPECT_EQ(marker.prior_materialization_generation + 1U,
              marker.published_materialization_generation);
    EXPECT_EQ(marker.prior_topology_epoch,
              baseline.topology_epoch + static_cast<std::uint64_t>(parent_level));
    EXPECT_EQ(marker.prior_materialization_generation,
              baseline.materialization_generation + static_cast<std::uint64_t>(parent_level));
    EXPECT_EQ(marker.published_topology_epoch,
              baseline.topology_epoch + static_cast<std::uint64_t>(parent_level + 1U));
    EXPECT_EQ(marker.published_materialization_generation,
              baseline.materialization_generation + static_cast<std::uint64_t>(parent_level + 1U));
  }
  ASSERT_EQ(accepted.histories.size(), 1U);
  EXPECT_EQ(accepted.histories.front().depth, baseline.histories.front().depth);
  EXPECT_EQ(accepted.histories.front().ncomp, baseline.histories.front().ncomp);
  EXPECT_EQ(system.history_depth("tracer.rate"), 2);
  ASSERT_EQ(accepted.histories.front().levels.size(), 4U);
  for (std::size_t level = 0; level < accepted.histories.front().levels.size(); ++level) {
    const auto& history_level = accepted.histories.front().levels[level];
    EXPECT_TRUE(history_level.initialized);
    EXPECT_EQ(history_level.level, static_cast<int>(level));
    EXPECT_EQ(history_level.slots.size(), std::size_t{2});
    EXPECT_EQ(history_level.slot_dt.size(), std::size_t{2});
  }

  // The serialized authority retains every individual forward edge.  These are intentionally
  // pure checkpoint corruptions: validation must reject a gap, a branch, and an authority that
  // does not reach the accepted generation without touching the live runtime.
  {
    auto gap = accepted_program;
    auto& edge = gap.pending_history_remaps.at(1);
    ++edge.prior_topology_epoch;
    ++edge.prior_materialization_generation;
    ++edge.published_topology_epoch;
    ++edge.published_materialization_generation;
    EXPECT_THROW((void)pops::runtime::program::serialize_amr_program_accepted_state(gap),
                 std::invalid_argument);
  }
  {
    auto branch = accepted_program;
    auto branch_history = branch.histories.front();
    branch_history.name = "ztracer.rate";
    branch.histories.push_back(branch_history);
    for (std::size_t level = 0; level < branch.level_clocks.size(); ++level) {
      for (int slot = 0; slot < branch_history.depth; ++slot) {
        const auto source = std::find_if(branch.history_slots.begin(), branch.history_slots.end(),
                                         [level, slot](const auto& provenance) {
                                           return provenance.name == "tracer.rate" &&
                                                  provenance.level == static_cast<int>(level) &&
                                                  provenance.slot == slot;
                                         });
        ASSERT_NE(source, branch.history_slots.end());
        auto duplicate = *source;
        duplicate.name = branch_history.name;
        branch.history_slots.push_back(std::move(duplicate));
      }
    }
    auto duplicate_edge = branch.pending_history_remaps.at(1);
    duplicate_edge.key = "pops.amr.level-history.v1/2/" +
                         std::to_string(branch_history.name.size()) + ":" + branch_history.name;
    ++duplicate_edge.prior_topology_epoch;
    ++duplicate_edge.prior_materialization_generation;
    ++duplicate_edge.published_topology_epoch;
    ++duplicate_edge.published_materialization_generation;
    branch.pending_history_remaps.push_back(std::move(duplicate_edge));
    std::sort(branch.pending_history_remaps.begin(), branch.pending_history_remaps.end(),
              [](const auto& left, const auto& right) { return left.key < right.key; });
    EXPECT_THROW((void)pops::runtime::program::serialize_amr_program_accepted_state(branch),
                 std::invalid_argument);
  }
  {
    auto final_mismatch = accepted_program;
    ++final_mismatch.topology_epoch;
    ++final_mismatch.materialization_generation;
    EXPECT_THROW((void)pops::runtime::program::serialize_amr_program_accepted_state(final_mismatch),
                 std::invalid_argument);
  }
}
