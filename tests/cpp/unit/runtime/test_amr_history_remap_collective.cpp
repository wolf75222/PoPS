#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "explicit_amr_program.hpp"

#include <pops/mesh/execution/for_each.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/amr_program_context.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pops::runtime::program::detail {

// This existing friend seam exposes only the actual native remap admission and test tampering.
template <int Dim>
struct AmrProgramHistoryRemapCollectiveTestAccess {
  using Context = AmrProgramContext<Dim>;
  using Field = MultiFab<Dim>;

  static auto& manager(Context& context) { return context.runtime_state().hist_; }
  static std::string key(Context& context, const std::string& name, int level) {
    return context.history_key_(name, level);
  }
  static void validate_collectively(Context& context) {
    (void)context.prepare_history_mutation_collectively_(
        [&] {
          context.validate_accepted_history_remap_metadata_();
          return std::string("tests.amr.declared-history-halo-metadata@1");
        }, [](const std::string& proof) -> const std::string& { return proof; },
        "tests.amr.declared-history-halo-metadata");
    // The first vote completes before a healthy peer may enter a ranked façade getter.
    (void)context.prepare_history_mutation_collectively_(
        [&] {
          context.validate_accepted_history_remap_();
          return std::string("tests.amr.declared-history-halo-remap@1");
        }, [](const std::string& proof) -> const std::string& { return proof; },
        "tests.amr.declared-history-halo-remap");
  }
  static void replace_slot_ghosts(Context& context, const std::string& name, int level,
                                  int slot_index, int depth) {
    auto& slot = manager(context).histories.at(key(context, name, level)).at(slot_index);
    Extent<Dim> ghosts{};
    for (int axis = 0; axis < Dim; ++axis) ghosts[axis] = depth;
    Field replacement(slot.layout(), slot.distribution(), slot.local_rank(), slot.ncomp(), ghosts);
    replacement.set_val(Real(17));
    for (std::size_t local = 0; local < slot.local_size(); ++local) {
      const auto source = std::as_const(slot).fab(local).view();
      const auto destination = replacement.fab(local).view();
      const int components = slot.ncomp();
      for_each_cell(slot.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        for (int component = 0; component < components; ++component)
          destination(cell, component) = source(cell, component);
      });
    }
    Kokkos::fence();
    slot = std::move(replacement);
  }
};
}  // namespace pops::runtime::program::detail

namespace {
constexpr int Dim = pops::kNativeDimension;
using Access = pops::runtime::program::detail::AmrProgramHistoryRemapCollectiveTestAccess<Dim>;
constexpr const char* kHistory = "tracer.potential";

struct ScalarModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 1;
  Law law{};
  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"tests.amr.history-declared-ghost.scalar", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& proof) const {
    for (int axis = 0; axis < Dim; ++axis) proof.scalar(law.velocity()[axis]);
  }
  static pops::VariableSet conservative_vars() {
    return {pops::VariableKind::Conservative, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  static pops::VariableSet primitive_vars() {
    return {pops::VariableKind::Primitive, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  POPS_HD auto recover(const State& state) const { return law.recover(state); }
  POPS_HD auto make_conservative(const Primitive& state) const { return law.make_conservative(state); }
  POPS_HD auto admissibility(const State& state) const { return law.admissibility(state); }
  template <int Axis> POPS_HD State flux(const State& state) const { return law.template flux<Axis>(state); }
  template <int Axis> POPS_HD pops::Real max_wave_speed(const State& state) const {
    return law.template max_wave_speed<Axis>(state);
  }
  template <int Axis>
  POPS_HD void wave_speeds(const State& state, pops::Real& lower, pops::Real& upper) const {
    law.template wave_speeds<Axis>(state, lower, upper);
  }
  POPS_HD State source(const State&, const pops::ProviderValues<0>&) const { return {}; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};

struct Fixture {
  pops::AmrSystem<Dim> system;
  std::shared_ptr<pops::runtime::program::AmrProgramContext<Dim>> context;

  static pops::AmrSystemConfig<Dim> config() {
    pops::AmrSystemConfig<Dim> value;
    value.level_count = 2;
    value.regrid_every = 0;
    for (int axis = 0; axis < Dim; ++axis) {
      value.shape[axis] = 16;
      value.transition_buffers.front()[axis] = 0;
      value.transition_lookaheads.front()[axis] = 0;
    }
    return value;
  }
  static std::size_t cells() {
    std::size_t total = 1;
    for (int axis = 0; axis < Dim; ++axis) total *= 16;
    return total;
  }
  Fixture() : system(config()) {
    pops::test::install_amr_runtime_authority(system, "tests.amr.history-declared-ghost/runtime");
    system.set_temporal_relations({1}, {1}, {"integral_only"});
    system.install_block_state_route("tracer", "state/tracer");
    pops::add_compiled_model<Dim>(system, "tracer", ScalarModel{}, "minmod", "rusanov",
        "conservative", "explicit", static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1,
        {}, {}, 0., static_cast<double>(pops::kWenoEpsilon), false,
        "tests.amr.history-declared-ghost/flux");
    system.set_conservative_state("tracer", std::vector<double>(cells(), 1.));
    pops::test::install_prepared_refine_coarsen_threshold(
        system, {"tracer", "u", .5, pops::test::PreparedThresholdRelation::Above},
        {"tracer", "u", .5, pops::test::PreparedThresholdRelation::Below},
        "tests.amr.history-declared-ghost/tagging");
    context = pops::runtime::program::make_program_execution_provider(&system);
    context->configure_primary_clock("clock.macro");
    context->declare_clock_relation("clock.macro", "clock.level.1", 1);
    context->install([owner = context](double dt) {
      owner->advance_hierarchy(dt, [](double) {});
    }, context);
    system.set_program_block_map({0});
    using Budget = typename pops::AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
    system.install_prepared_amr_program_flux_expression_budget(
        "tests.amr.history-declared-ghost/flux-budget", std::vector<Budget>(1, Budget{2, 1}), 0, 0);
    context->for_each_program_resource_level([&](int) {
      context->register_history(kHistory, 1, 1, 0, "scalar-history:" + std::string(kHistory),
          std::string(pops::runtime::program::kScalarOutputHistorySpace), "clock.macro", "none", 1);
    });
  }
  void fill_history(pops::Real scale = pops::Real(10)) {
    int sample = 0;
    for (double dt : {.1, .2}) {
      ++sample;
      context->advance_hierarchy(dt, [&](double) {
        auto& value = context->scalar_scratch(1, 0, context->state(0), 1, 1);
        value.set_val(scale * pops::Real(sample));
        context->store_history(kHistory, value, 0);
        context->rotate_histories("clock.macro");
      });
    }
  }
  void regrid(bool full) {
    std::vector<double> state(cells(), full ? 1. : .25);
    if (!full) {
      std::size_t centre = 0, stride = 1;
      for (int axis = 0; axis < Dim; ++axis) { centre += 8 * stride; stride *= 16; }
      state[centre] = 1.;
    }
    system.set_conservative_state("tracer", state);
    system.execute_prepared_tagging(0);
    ASSERT_TRUE(system.regrid_from_prepared_tagging(0));
  }
  void expect_declared_storage() {
    context->for_each_program_resource_level([&](int level) {
      const auto& live = context->state(0);
      for (int axis = 0; axis < Dim; ++axis) EXPECT_EQ(live.ghosts()[axis], 2);
      auto& manager = Access::manager(*context);
      const auto key = Access::key(*context, kHistory, level);
      EXPECT_EQ(manager.ghost_depth.at(key), 1);
      for (const auto& slot : manager.histories.at(key)) {
        EXPECT_EQ(slot.layout(), live.layout());
        EXPECT_EQ(slot.distribution(), live.distribution());
        EXPECT_EQ(slot.local_rank(), live.local_rank());
        EXPECT_EQ(slot.ncomp(), 1);
        for (int axis = 0; axis < Dim; ++axis) EXPECT_EQ(slot.ghosts()[axis], 1);
      }
    });
    EXPECT_NO_THROW(Access::validate_collectively(*context));
  }
};

TEST(AmrDeclaredHistoryGhosts, ScalarGhostOneSurvivesRealRegridOnStateGhostTwo) {
  Fixture fixture;
  fixture.fill_history();
  fixture.expect_declared_storage();
  fixture.regrid(false);
  fixture.expect_declared_storage();
  fixture.regrid(true);
  fixture.expect_declared_storage();
  const auto bytes = fixture.system.program_accepted_state();
  const auto wire = pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(bytes);
  ASSERT_EQ(wire.histories.size(), 1u);
  EXPECT_EQ(wire.histories.front().ghost_depth, 1);
  EXPECT_EQ(pops::runtime::program::serialize_amr_program_accepted_state(wire), bytes);

  // A fresh actual runtime follows the same topology/clock trajectory with different data.
  Fixture restored;
  restored.fill_history(pops::Real(80));
  restored.regrid(false);
  restored.regrid(true);
  for (int level = 0; level < 2; ++level)
    for (int slot = 0; slot < 2; ++slot)
      restored.system.restore_history(kHistory, level, slot,
          fixture.system.history_global(kHistory, level, slot));
  ASSERT_NO_THROW(restored.system.restore_checkpoint_accepted_state(bytes));
  restored.expect_declared_storage();
  EXPECT_EQ(restored.system.program_accepted_state(), bytes);
  for (int level = 0; level < 2; ++level)
    for (int slot = 0; slot < 2; ++slot)
      EXPECT_EQ(restored.system.history_global(kHistory, level, slot),
                fixture.system.history_global(kHistory, level, slot));
}

TEST(AmrDeclaredHistoryGhosts, SlotAndLevelGhostTamperingFailsCollectivelyAndRollsBack) {
  Fixture fixture;
  fixture.fill_history();
  fixture.expect_declared_storage();
  const auto before = fixture.system.program_accepted_state();
  const auto before_boxes = fixture.system.patch_boxes();
  const auto before_time = fixture.system.time();
  const auto before_step = fixture.system.macro_step();
  std::vector<std::vector<double>> before_values;
  for (int level = 0; level < 2; ++level)
    for (int slot = 0; slot < 2; ++slot)
      before_values.push_back(fixture.system.history_global(kHistory, level, slot));
  for (bool whole_level : {false, true}) {
    fixture.system.begin_step_transaction();
    // A single rank's corrupt candidate must converge before any later publication.
    if (pops::my_rank() == 0) {
      Access::replace_slot_ghosts(*fixture.context, kHistory, 0, 0, 2);
      if (whole_level) {
        Access::replace_slot_ghosts(*fixture.context, kHistory, 0, 1, 2);
        Access::manager(*fixture.context).ghost_depth.at(
            Access::key(*fixture.context, kHistory, 0)) = 2;
      }
    }
    EXPECT_THROW(Access::validate_collectively(*fixture.context), std::exception);
    ASSERT_NO_THROW(fixture.system.rollback_step_transaction());
    fixture.expect_declared_storage();
    EXPECT_EQ(fixture.system.program_accepted_state(), before);
    EXPECT_EQ(fixture.system.patch_boxes(), before_boxes);
    EXPECT_EQ(fixture.system.time(), before_time);
    EXPECT_EQ(fixture.system.macro_step(), before_step);
    std::size_t index = 0;
    for (int level = 0; level < 2; ++level)
      for (int slot = 0; slot < 2; ++slot)
        EXPECT_EQ(fixture.system.history_global(kHistory, level, slot), before_values[index++]);
  }
}

TEST(AmrDeclaredHistoryGhosts, ChangedWireGhostDescriptorCannotRestoreDifferentStorage) {
  Fixture fixture;
  fixture.fill_history();
  const auto before = fixture.system.program_accepted_state();
  auto wire = pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(before);
  ASSERT_EQ(wire.histories.size(), 1u);
  wire.histories.front().ghost_depth = 2;
  const auto changed = pops::runtime::program::serialize_amr_program_accepted_state(wire);
  EXPECT_THROW(fixture.system.restore_checkpoint_accepted_state(changed), std::exception);
  fixture.expect_declared_storage();
  EXPECT_EQ(fixture.system.program_accepted_state(), before);
}
}  // namespace
