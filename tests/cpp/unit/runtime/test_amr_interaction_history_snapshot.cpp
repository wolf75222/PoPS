#include "amr_interaction_history_snapshot_fixture.hpp"

#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>

namespace {
using namespace pops::test::interaction_history;
namespace program = pops::runtime::program;

TEST(AmrInteractionHistorySnapshot, ColdAndWarmCoarseRotationPreservesFineOracleAndEvolution) {
  Fixture fixture;
  pops::Real accepted = 2, selected = 2;
  std::array<int, 2> calls{};
  const std::array<double, 3> intervals{.125, .25, .0625};
  for (double dt : intervals) {
    calls = {};
    const auto integral = fixture.composite_integral(selected);
    ASSERT_EQ(integral, selected);  // The partial fine level replaces, never duplicates, volume.
    fixture.body = [&](double local_dt) {
      auto& context = *fixture.context;
      const auto image = Access::require(context);
      const auto samples = image->history.slot_sample;
      const auto dts = image->history.slot_dt;
      auto& state = context.state(0);
      context.store_history(kName, state, 0);
      // The coarse callback rotates LIVE storage before the fine callback starts.
      context.rotate_histories("clock.macro");
      for (const auto stage : {std::pair{0, 1}, std::pair{1, 2}, std::pair{1, 1}}) {
        context.set_stage_time(stage.first, stage.second);
        auto& output = fixture.interaction();
        fixture.expect_output(output, integral);
        EXPECT_EQ(image->history.slot_sample, samples);
        EXPECT_EQ(image->history.slot_dt, dts);
      }
      auto candidate = context.scratch_state_like(state);
      context.copy_grown_component_span(candidate, 0, state, 0, 1);
      context.axpy(candidate, pops::Real(local_dt), fixture.interaction());
      context.commit_many({{&state, &candidate}});
      ++calls.at(context.level());
    };
    EXPECT_NO_THROW(fixture.system.step(dt));
    EXPECT_EQ(calls, (std::array<int, 2>{1, 1}));
    EXPECT_EQ(Access::image_count(*fixture.context), 0u);
    const auto old = accepted;
    accepted += pops::Real(dt) * integral;
    fixture.context->for_each_program_resource_level([&](int) {
      expect_constant(fixture.context->state(0), accepted);
      expect_constant(fixture.context->history(kName, 1, 0), old);
    });
    selected = old;
  }
}

TEST(AmrInteractionHistorySnapshot, FullRingsAndDeclaredGhostsAreDeepCopiedAndDeduplicated) {
  Fixture fixture(true, 1, kBudget, 1);
  fixture.warm();
  auto& context = *fixture.context;
  auto& live = Access::manager(context);
  for (int level = 0; level < 2; ++level)
    for (std::size_t slot = 0; slot < live.histories.at(Access::key(context, level)).size(); ++slot)
      live.histories.at(Access::key(context, level))[slot].set_val(pops::Real(20 + slot));
  context.begin_step(.25);
  const auto image = Access::require(context, 100);
  EXPECT_EQ(Access::require(context, 101), image);
  EXPECT_EQ(Access::image_count(context), 1u);
  EXPECT_TRUE(image->cold_seed.empty());
  const auto samples = image->history.slot_sample;
  const auto dts = image->history.slot_dt;
  for (int level = 0; level < 2; ++level) {
    const auto key = Access::key(context, level);
    ASSERT_EQ(image->history.histories.at(key).size(), 3u);
    for (auto& slot : live.histories.at(key))
      slot.set_val(pops::Real(-99));
    std::swap(live.histories.at(key)[0], live.histories.at(key)[2]);
    live.slot_sample.at(key)[1].ordinal += 10;
    live.slot_dt.at(key)[1] = 123;
    for (std::size_t slot = 0; slot < 3; ++slot) {
      const auto& frozen = image->history.histories.at(key)[slot];
      for (int axis = 0; axis < Dim; ++axis)
        EXPECT_EQ(frozen.ghosts()[axis], 1);
      expect_constant(frozen, pops::Real(20 + slot), true);
    }
  }
  EXPECT_EQ(image->history.slot_sample, samples);
  EXPECT_EQ(image->history.slot_dt, dts);
  Access::revoke(context);
  EXPECT_THROW((void)Access::require(context), std::logic_error);
  // A retained local shared_ptr protects buffers while cache lookup fails closed.
  expect_constant(image->history.histories.at(Access::key(context, 0))[1], pops::Real(21), true);
}

TEST(AmrInteractionHistorySnapshot, ColdAcceptedSeedIsAlsoDeepCopied) {
  Fixture fixture;
  fixture.context->begin_step(.125);
  const auto image = Access::require(*fixture.context);
  ASSERT_EQ(image->cold_seed.size(), 2u);
  fixture.context->for_each_program_resource_level([&](int level) {
    fixture.context->state(0).set_val(pops::Real(91));
    expect_constant(image->cold_seed.at(level), pops::Real(2), true);
  });
}

TEST(AmrInteractionHistorySnapshot, ExactDeclarationsAreIdempotentAndRegistrySeals) {
  Fixture fixture;
  fixture.declaration(100, 1);
  const auto issue = [&](int owner, const char* name, int lag, const char* state, const char* space,
                         const char* clock, const char* interpolation, std::uint64_t budget,
                         const char* identity) {
    fixture.context->declare_spatial_interaction_history_source(
        100, owner, name, lag, state, space, clock, interpolation, budget, identity);
  };
  EXPECT_THROW(
      issue(1, kName, 1, "tracer.U", "cell.scalar", "clock.macro", "none", kBudget, kIdentity),
      std::exception);
  EXPECT_THROW(
      issue(0, "foreign", 1, "tracer.U", "cell.scalar", "clock.macro", "none", kBudget, kIdentity),
      std::exception);
  EXPECT_THROW(
      issue(0, kName, 2, "tracer.U", "cell.scalar", "clock.macro", "none", kBudget, kIdentity),
      std::exception);
  EXPECT_THROW(
      issue(0, kName, 1, "foreign", "cell.scalar", "clock.macro", "none", kBudget, kIdentity),
      std::exception);
  EXPECT_THROW(issue(0, kName, 1, "tracer.U", "foreign", "clock.macro", "none", kBudget, kIdentity),
               std::exception);
  EXPECT_THROW(issue(0, kName, 1, "tracer.U", "cell.scalar", "foreign", "none", kBudget, kIdentity),
               std::exception);
  EXPECT_THROW(
      issue(0, kName, 1, "tracer.U", "cell.scalar", "clock.macro", "foreign", kBudget, kIdentity),
      std::exception);
  EXPECT_THROW(
      issue(0, kName, 1, "tracer.U", "cell.scalar", "clock.macro", "none", kBudget - 1, kIdentity),
      std::exception);
  EXPECT_THROW(
      issue(0, kName, 1, "tracer.U", "cell.scalar", "clock.macro", "none", kBudget, "foreign"),
      std::exception);
  fixture.context->begin_step(.125);
  EXPECT_NO_THROW(fixture.declaration(100, 1));
  EXPECT_THROW(fixture.declaration(102, 1), std::logic_error);
}

TEST(AmrInteractionHistorySnapshot, RankLocalMetadataCorruptionsRefuseCaptureCollectively) {
  for (int corruption = 0; corruption < 13; ++corruption) {
    SCOPED_TRACE(corruption);
    Fixture fixture;
    fixture.warm();
    auto& context = *fixture.context;
    auto& history = Access::manager(context);
    const auto key = Access::key(context, 1);
    if (pops::my_rank() == 0) {
      switch (corruption) {
        case 0:
          history.fill_count.at(key) = 0;
          history.initialized.at(key) = false;
          history.slot_sample.at(key).assign(3, program::HistorySampleIdentity::zero_start());
          history.slot_dt.at(key).assign(3, pops::Real(0));
          break;
        case 1:
          ++history.slot_sample.at(key)[1].ordinal;
          break;
        case 2:
          history.slot_dt.at(key)[1] = std::nextafter(history.slot_dt.at(key)[1], pops::Real(1));
          break;
        case 3:
          Access::pending_remap(context, 1);
          break;
        case 4:
          history.owner.at(key) = 1;
          break;
        case 5:
          history.state_identity.at(key) = "foreign";
          break;
        case 6:
          history.space_identity.at(key) = "foreign";
          break;
        case 7:
          history.clock_identity.at(key) = "foreign";
          break;
        case 8:
          history.interpolation_identity.at(key) = "foreign";
          break;
        case 9:
          history.fill_count.at(key) = -1;
          break;
        case 10:
          history.initialized.at(key) = false;
          break;
        case 11:
          history.slot_sample.at(key)[1].kind = program::HistorySampleKind::UnknownLegacy;
          break;
        case 12:
          history.ghost_depth.at(key) = history.histories.at(key).front().ghosts()[0] + 1;
          break;
      }
    }
    EXPECT_THROW(context.begin_step(.25), std::exception);
    EXPECT_EQ(Access::image_count(context), 0u);
    context.for_each_program_resource_level(
        [&](int) { expect_constant(context.state(0), pops::Real(2)); });
  }
}

TEST(AmrInteractionHistorySnapshot, AttemptAndExactMacroTagsRefuseStaleImages) {
  Fixture fixture;
  auto& context = *fixture.context;
  EXPECT_THROW((void)Access::require(context), std::logic_error);
  context.begin_step(.125);
  const auto old = Access::require(context);
  context.begin_step(.25);
  ASSERT_NE(Access::require(context), old);
  Access::restore_old_image(context, old);
  EXPECT_THROW((void)Access::require(context), std::logic_error);
  for (int tag = 0; tag < 4; ++tag) {
    context.begin_step(.125);
    auto& image = Access::corrupt_image(context);
    if (tag == 0)
      ++image.epoch;
    if (tag == 1)
      ++image.generation;
    if (tag == 2)
      ++image.macro_step;
    if (tag == 3)
      image.time = std::nextafter(image.time, 1.);
    EXPECT_THROW((void)Access::require(context), std::logic_error);
  }
  context.begin_step(.125);
  Access::new_attempt(context);
  EXPECT_THROW((void)Access::require(context), std::logic_error);
  context.begin_step(.125);
  Access::cancel_mapping(context);
  EXPECT_THROW((void)Access::require(context), std::logic_error);
}

TEST(AmrInteractionHistorySnapshot, ColdAndWarmRatioTwoRefuseWithoutAcceptedPublication) {
  for (bool warm : {false, true}) {
    Fixture fixture(true, 2);
    if (warm) {
      // Publish identical authenticated macro samples without a half-window consumer.
      fixture.context->begin_step(.125);
      fixture.context->for_each_program_resource_level([&](int) { fixture.store(pops::Real(7)); });
    }
    const auto accepted = fixture.system.program_accepted_state();
    const auto history = Access::manager(*fixture.context);
    int fine_calls = 0;
    fixture.body = [&](double) {
      if (fixture.context->level() == 1)
        ++fine_calls;
      fixture.context->store_history(kName, fixture.context->state(0), 0);
      (void)fixture.interaction();
    };
    EXPECT_THROW(fixture.system.step(.125), std::exception);
    EXPECT_EQ(fine_calls, 1);
    EXPECT_EQ(fixture.system.program_accepted_state(), accepted);
    EXPECT_EQ(Access::manager(*fixture.context).slot_sample, history.slot_sample);
    EXPECT_EQ(Access::manager(*fixture.context).slot_dt, history.slot_dt);
    fixture.context->for_each_program_resource_level(
        [&](int) { expect_constant(fixture.context->state(0), pops::Real(2)); });
    EXPECT_EQ(Access::image_count(*fixture.context), 0u);
  }
}

TEST(AmrInteractionHistorySnapshot, SourceAndWorkspaceBudgetsFailBeforePublication) {
  {
    Fixture tiny(true, 1, 128);
    const auto accepted = tiny.system.program_accepted_state();
    EXPECT_THROW(tiny.system.step(.125), std::exception);
    EXPECT_EQ(tiny.system.program_accepted_state(), accepted);
    EXPECT_EQ(Access::image_count(*tiny.context), 0u);
  }
  Fixture baseline;
  baseline.context->begin_step(.125);
  const auto bytes = Access::require(*baseline.context)->retained_bytes;
  Fixture limited(true, 1, bytes + 256);
  const auto accepted = limited.system.program_accepted_state();
  bool captured = false;
  limited.body = [&](double) {
    captured = true;
    ASSERT_EQ(Access::image_count(*limited.context), 1u);
    EXPECT_EQ(Access::require(*limited.context)->retained_bytes, bytes);
    limited.context->store_history(kName, limited.context->state(0), 0);
    (void)limited.interaction();
  };
  std::string refusal;
  try {
    limited.system.step(.125);
  } catch (const std::exception& error) {
    refusal = error.what();
  }
  ASSERT_FALSE(refusal.empty());
  if (pops::n_ranks() == 1)
    EXPECT_NE(refusal.find("workspace budget"), std::string::npos);
  EXPECT_TRUE(captured);
  EXPECT_EQ(limited.system.program_accepted_state(), accepted);
  EXPECT_EQ(Access::image_count(*limited.context), 0u);
}

TEST(AmrInteractionHistorySnapshot, AllocationVoteAndOverflowConvergeAndEmptyRegistryStaysEmpty) {
  Fixture empty(false);
  empty.context->begin_step(.125);
  EXPECT_EQ(Access::image_count(*empty.context), 0u);
  EXPECT_THROW(empty.declaration(100, 1), std::logic_error);
  const auto& lane = empty.context->prepared_execution_lane();
  EXPECT_THROW(program::interaction_phase(lane,
                                          [] {
                                            if (pops::my_rank() == 0)
                                              throw std::bad_alloc();
                                          }),
               std::exception);
  EXPECT_THROW(program::interaction_phase(lane,
                                          [] {
                                            if (pops::my_rank() == 0)
                                              (void)program::interaction_product(
                                                  std::numeric_limits<std::size_t>::max(), 2);
                                          }),
               std::exception);
  EXPECT_THROW(program::interaction_add(std::numeric_limits<std::size_t>::max(), 1),
               std::overflow_error);
  EXPECT_NO_THROW(program::interaction_phase(lane, [] {}));
}

TEST(AmrInteractionHistorySnapshot, SuccessfulAndCancelledMappingRevokeImages) {
  Fixture fixture;
  bool completed = false;
  fixture.context->advance_mapping_hierarchy(
      .125, [&](double) { (void)Access::require(*fixture.context); }, fixture.context,
      [&] {
        completed = true;
        EXPECT_EQ(Access::image_count(*fixture.context), 0u);
      });
  EXPECT_TRUE(completed);
  EXPECT_EQ(Access::image_count(*fixture.context), 0u);
  EXPECT_THROW(fixture.context->advance_mapping_hierarchy(
                   .125,
                   [&](double) {
                     if (pops::my_rank() == 0)
                       throw std::runtime_error("negative mapping fixture");
                   },
                   fixture.context, [] {}),
               std::exception);
  EXPECT_EQ(Access::image_count(*fixture.context), 0u);
}

TEST(AmrInteractionHistorySnapshot, ContextRollbackRevokesAndNextAttemptRecaptures) {
  Fixture fixture;
  auto& context = *fixture.context;
  auto accepted = context.accepted_context_snapshot();
  context.begin_step(.125);
  const auto old = Access::require(context);
  auto restore = accepted->prepare_restore();
  restore->publish_restore();
  EXPECT_EQ(Access::image_count(context), 0u);
  EXPECT_THROW((void)Access::require(context), std::logic_error);
  context.begin_step(.25);
  EXPECT_NE(Access::require(context), old);
  expect_constant(Access::require(context)->cold_seed.at(0), pops::Real(2));
}

TEST(AmrInteractionHistorySnapshot, PublicCheckpointRestoreRejectsOldAttemptAtSameMacroTime) {
  Fixture fixture;
  fixture.warm();
  const auto bytes = fixture.system.program_accepted_state();
  const auto time = fixture.system.time();
  const auto step = fixture.system.macro_step();
  auto& context = *fixture.context;
  context.begin_step(.125);
  const auto old = Access::require(context);
  ASSERT_TRUE(old->attempt.visible());
  fixture.system.restore_checkpoint_accepted_state(bytes);
  EXPECT_FALSE(old->attempt.visible());
  EXPECT_THROW((void)Access::require(context), std::logic_error);
  EXPECT_EQ(fixture.system.time(), time);
  EXPECT_EQ(fixture.system.macro_step(), step);
  context.begin_step(.125);
  const auto restored = Access::require(context);
  EXPECT_NE(restored, old);
  EXPECT_EQ(restored->epoch, old->epoch);
  EXPECT_EQ(restored->generation, old->generation);
  EXPECT_EQ(restored->history.slot_sample, old->history.slot_sample);
  EXPECT_EQ(restored->history.slot_dt, old->history.slot_dt);
}

TEST(AmrInteractionHistorySnapshot, PublicRuntimeRebuildRevokesColdSourceBeforeRebind) {
  Fixture fixture;
  auto& context = *fixture.context;
  auto* old_runtime = fixture.system.engine();
  const auto boxes = fixture.system.patch_boxes();
  const auto owners = fixture.system.level_owner_ranks(1);
  context.begin_step(.125);
  const auto old = Access::require(context);
  fixture.system.rebuild_hierarchy(boxes, owners);
  EXPECT_NE(fixture.system.engine(), old_runtime);
  (void)context.program_resource_topology();
  EXPECT_EQ(Access::runtime(context), fixture.system.engine());
  EXPECT_EQ(Access::image_count(context), 0u);
  EXPECT_THROW((void)Access::require(context), std::logic_error);
  // Retention is independent of the replacement engine's ownership and buffers.
  expect_constant(old->cold_seed.at(0), pops::Real(2));
}

TEST(AmrInteractionHistorySnapshot, OuterTransactionAndExplicitContextRollbackReplayMacroImages) {
  Fixture fixture;
  fixture.warm();
  const auto before = fixture.system.program_accepted_state();
  const auto start_time = fixture.system.time();
  const auto start_step = fixture.system.macro_step();
  // A direct C++ install is not artifact-backed: its caller owns this context snapshot.
  auto context_snapshot = fixture.context->accepted_context_snapshot();
  const auto run = [&] {
    std::vector<std::pair<std::uint64_t, double>> sources;
    fixture.body = [&](double local_dt) {
      auto& context = *fixture.context;
      const auto image = Access::require(context);
      if (context.level() == 0)
        sources.emplace_back(image->macro_step, image->dt);
      auto& state = context.state(0);
      auto candidate = context.scratch_state_like(state);
      context.copy_grown_component_span(candidate, 0, state, 0, 1);
      context.axpy(candidate, pops::Real(local_dt), fixture.interaction());
      context.store_history(kName, state, 0);
      context.rotate_histories("clock.macro");
      context.commit_many({{&state, &candidate}});
    };
    for (double dt : {.125, .25, .0625})
      fixture.system.step(dt);
    return sources;
  };
  fixture.system.begin_step_transaction();
  const auto first = run();
  const auto after = fixture.system.program_accepted_state();
  const auto history = Access::manager(*fixture.context);
  fixture.context->begin_step(.125);
  const auto discarded = Access::require(*fixture.context);
  fixture.system.rollback_step_transaction();
  auto restore = context_snapshot->prepare_restore();
  restore->publish_restore();
  EXPECT_EQ(fixture.system.program_accepted_state(), before);
  EXPECT_EQ(fixture.system.time(), start_time);
  EXPECT_EQ(fixture.system.macro_step(), start_step);
  EXPECT_EQ(Access::image_count(*fixture.context), 0u);
  EXPECT_THROW((void)Access::require(*fixture.context), std::logic_error);
  EXPECT_FALSE(discarded->attempt.visible());
  EXPECT_EQ(run(), first);
  auto original = program::deserialize_amr_program_accepted_state<Dim>(after);
  const auto replayed =
      program::deserialize_amr_program_accepted_state<Dim>(fixture.system.program_accepted_state());
  ASSERT_TRUE(original.accepted_attempt.has_value());
  ASSERT_TRUE(replayed.accepted_attempt.has_value());
  EXPECT_GT(*replayed.accepted_attempt, *original.accepted_attempt);
  // Retired attempt IDs stay monotone; every other accepted/checkpoint field replays exactly.
  original.accepted_attempt = replayed.accepted_attempt;
  EXPECT_EQ(program::serialize_amr_program_accepted_state(original),
            program::serialize_amr_program_accepted_state(replayed));
  EXPECT_EQ(Access::manager(*fixture.context).slot_sample, history.slot_sample);
  EXPECT_EQ(Access::manager(*fixture.context).slot_dt, history.slot_dt);
  fixture.context->for_each_program_resource_level([&](int level) {
    const auto key = Access::key(*fixture.context, level);
    for (int lag : {1, 2}) {
      const auto& replayed = fixture.context->history(kName, lag, 0);
      const auto& original = history.histories.at(key).at(lag);
      for (std::size_t local = 0; local < replayed.local_size(); ++local) {
        const auto left = replayed.fab(local).view(), right = original.fab(local).view();
        EXPECT_EQ(pops::for_each_cell_reduce_max(
                      replayed.box(local),
                      [=] POPS_HD(const pops::Index<Dim>& cell) {
                        return std::bit_cast<program::InteractionRealWord>(left(cell, 0)) ==
                                       std::bit_cast<program::InteractionRealWord>(right(cell, 0))
                                   ? pops::Real(0)
                                   : pops::Real(1);
                      }),
                  pops::Real(0));
      }
    }
  });
}
}  // namespace
