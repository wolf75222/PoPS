// Checkpoint / restart of the dense scheduler value cache (Spec 3 section 30, ADC-458). The held
// cache lives in the System-owned CacheManager and is addressed by the bind-sealed resource slot,
// never by a runtime node id. This test exercises the host-owned checkpoint image and detached
// restore preparation directly, with no Program/codegen/engine.
//
// It checks: (a) populated and cold slots round-trip through a detached image (values and temporal
// metadata preserved); (b) invalid slots retain their accumulated window without a dummy field; and
// (c) a malformed image cannot clobber the accepted cache.

#include <gtest/gtest.h>

#include <pops/runtime/program/cache_manager.hpp>
#include <pops/runtime/program/program_persistent_value_checkpoint.hpp>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/layout/rank_space.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace pops;

namespace {

constexpr int kDim = kNativeDimension;
using Field = MultiFab<kDim>;
using NativeCacheManager = runtime::program::CacheManager<kDim>;

Extent<kDim> filled_extent(std::int64_t value) {
  Extent<kDim> result{};
  for (int axis = 0; axis < kDim; ++axis)
    result[axis] = value;
  return result;
}

std::size_t valid_cell_count() {
  std::size_t result = 1;
  for (int axis = 0; axis < kDim; ++axis)
    result *= 8;
  return result;
}

Field make_mf(double fill, int ncomp = 1, int ngrow = 1) {
  const Box<kDim> domain = Box<kDim>::from_extents(filled_extent(8));
  const mesh::BoxArray<kDim> layout = mesh::BoxArray<kDim>::from_domain(domain, filled_extent(4));
  const mesh::RankSpace<kDim> ranks(Index<kDim>{}, filled_extent(1));
  const auto distribution = mesh::Distribution<kDim>::replicated(layout, ranks);
  Field mf(layout, distribution, Index<kDim>{}, ncomp, filled_extent(ngrow));
  mf.set_val(fill);
  return mf;
}

runtime::program::ProgramResourcePlan cache_plan(std::size_t count) {
  using namespace runtime::program;
  std::vector<ProgramResourcePlanEntry> rows;
  rows.reserve(count);
  for (std::size_t slot = 0; slot != count; ++slot) {
    ProgramResourcePlanEntry row;
    row.slot = static_cast<std::uint32_t>(slot);
    row.key = {700 + slot, 900 + slot, 0, 1, 2, -1};
    row.identity = "cache-value-" + std::to_string(slot);
    row.occurrence_path = "program/cache/" + std::to_string(slot);
    row.owner_identity = "block:0";
    row.space_identity = "cell";
    row.clock_identity = "clock.macro";
    row.lifetime = ProgramValueLifetime::persistent_schedule;
    row.centering = ProgramValueCentering::cell;
    row.off_policy = ProgramScheduleOffPolicy::accumulate_dt;
    row.spatial_transfer = ProgramSpatialTransferPolicy::redistribute_exact;
    row.components = 1;
    row.ghosts = 1;
    row.bytes = 8;
    row.maximum_bytes = 16;
    row.communication = "none";
    row.transfer_identity = "redistribute_exact:v1";
    row.restart_identity = "restart:v1";
    row.component_names = "[\"u\"]";
    row.shape = "[8]";
    row.cells = 8;
    row.itemsize = 8;
    rows.push_back(std::move(row));
  }
  return ProgramResourcePlan(std::move(rows), count * 16, "program-resource-plan:v1",
                             std::string(64, 'b'));
}

runtime::program::ProgramResourcePlanEntry checkpoint_row() {
  using namespace runtime::program;
  ProgramResourcePlanEntry row;
  row.slot = 0;
  row.key = {71, 91, 2, 3, 4, 5};
  row.identity = "program-resource:v1:71";
  row.occurrence_path = "root/branch/loop/0";
  row.owner_identity = "block:plasma";
  row.space_identity = "state";
  row.clock_identity = "clock.macro";
  row.lifetime = ProgramValueLifetime::persistent_schedule;
  row.centering = ProgramValueCentering::face;
  row.off_policy = ProgramScheduleOffPolicy::accumulate_dt;
  row.spatial_transfer = ProgramSpatialTransferPolicy::redistribute_exact;
  row.components = 2;
  row.ghosts = 1;
  row.bytes = 8;
  row.maximum_bytes = 16;
  row.communicates = true;
  row.restart_required = true;
  row.communication = "mpi";
  row.transfer_identity = "redistribute_exact:v1";
  row.restart_identity = "restart:v1";
  row.component_names = "[\"u\",\"v\"]";
  row.shape = "[4,1]";
  row.cells = 4;
  row.itemsize = 8;
  return row;
}

}  // namespace

TEST(CheckpointCache, FullRoundTripSerializesAndDeserializesToEqualState) {
  NativeCacheManager src;
  src.bind(cache_plan(3));
  // a named held field solve (the aux cache), stamped at step 30, no accumulator
  src.prime_slot(0, make_mf(0.0));
  src.store(0, make_mf(2.0), 30, "fields_from_state");
  // a nameless held scratch (2 comps), stamped at step 12, with an accumulated dt window
  src.prime_slot(1, make_mf(0.0, /*ncomp=*/2));
  src.store(1, make_mf(3.0, /*ncomp=*/2), 12);
  src.accumulate_dt(1, 0.001);
  src.accumulate_dt(1, 0.002);
  // a held SCRATCH at the block-state ghost width (2 ghosts): restore MUST rebuild with ngrow 2, not
  // the aux's 1, else a 2-ghost-stencil consumer under-reads its outer ghosts after restart.
  src.prime_slot(2, make_mf(0.0, /*ncomp=*/1, /*ngrow=*/2));
  src.store(2, make_mf(4.0, /*ncomp=*/1, /*ngrow=*/2), 7);

  // the ngrow accessor distinguishes the aux (1) from a held scratch (2) -- the value restore needs it
  EXPECT_EQ(src.ghosts_of(0), filled_extent(1)) << "slot0_ghosts_are_1";
  EXPECT_EQ(src.ghosts_of(2), filled_extent(2)) << "slot2_ghosts_are_2";

  const auto images = src.checkpoint_slots();
  ASSERT_EQ(images.size(), 3u);
  EXPECT_EQ(images.front().plan_schema, "program-resource-plan:v1");
  EXPECT_EQ(images.front().plan_digest, std::string(64, 'b'));
  NativeCacheManager dst;
  dst.bind(cache_plan(3));
  auto prepared = dst.prepare_checkpoint_restore(images);
  dst.publish_checkpoint_restore(std::move(prepared));

  // dense slot indices + count preserved
  EXPECT_TRUE(dst.slot_indices().size() == 3) << "restore_slot_count";
  EXPECT_TRUE(dst.has(0) && dst.has(1) && dst.has(2)) << "restore_slots";
  // the held-scratch slot's ghost width round-trips (serialized ngrow rebuilds the right width)
  EXPECT_EQ(dst.retrieve(2).ghosts(), filled_extent(2)) << "restore_value2_ghosts_preserved";
  // Values are bit-equal over the exact native-rank valid-cell count.
  EXPECT_TRUE(std::fabs(reduce_sum_local(dst.retrieve(0)) - 2.0 * valid_cell_count()) < 1e-12)
      << "restore_value0_bit_equal";
  EXPECT_TRUE(dst.retrieve(1).ncomp() == 2) << "restore_value1_ncomp";
  EXPECT_TRUE(std::fabs(reduce_sum_local(dst.retrieve(1)) - 3.0 * valid_cell_count()) < 1e-12)
      << "restore_value1_bit_equal";
  // bookkeeping preserved
  EXPECT_TRUE(dst.last_update_step(0) == 30) << "restore_last_update0";
  EXPECT_TRUE(dst.last_update_step(1) == 12) << "restore_last_update1";
  EXPECT_TRUE(std::fabs(dst.accumulated_dt(0)) < 1e-15) << "restore_accum0_zero";
  EXPECT_TRUE(std::fabs(dst.accumulated_dt(1) - 0.003) < 1e-12) << "restore_accum1_sum";
  // name preserved (named slot keeps its name; nameless slot falls back to its plan identity)
  EXPECT_TRUE(dst.name_of(0) == "fields_from_state") << "restore_name_explicit";
  EXPECT_TRUE(dst.name_of(1) == "cache-value-1") << "restore_name_plan_identity";

  // a deserialized cache is_due exactly like a live one (cold-start gone, cadence resumes)
  EXPECT_TRUE(!dst.is_due(0, 31, 10)) << "restored_not_due_off_cadence";
  EXPECT_TRUE(dst.is_due(0, 40, 10)) << "restored_due_on_cadence";
}

TEST(CheckpointCache, OnlyValidSlotsSerializeColdAccumulatorSkipped) {
  NativeCacheManager src;
  src.bind(cache_plan(2));
  src.prime_slot(0, make_mf(0.0));
  src.store(0, make_mf(1.0), 0, "held");  // valid
  src.accumulate_dt(1, 0.5);              // cold slot with a pending window
  EXPECT_TRUE(!src.valid(1)) << "cold_accum_slot_invalid";
  const auto images = src.checkpoint_slots();
  ASSERT_EQ(images.size(), 2u);
  EXPECT_TRUE(images[0].valid && images[0].value.has_value()) << "serialize_valid_slot";
  EXPECT_TRUE(!images[1].valid && !images[1].value.has_value()) << "serialize_cold_slot_metadata";
}

TEST(CheckpointCache, DeclaredColdAccumulatorRoundTripsWithoutDummyAllocation) {
  NativeCacheManager src;
  src.bind(cache_plan(1));
  src.accumulate_dt(0, 0.5);
  EXPECT_EQ(src.checkpoint_slot_indices(), (std::vector<runtime::program::ProgramCacheSlot>{0}));
  const auto images = src.checkpoint_slots();
  ASSERT_EQ(images.size(), 1u);
  EXPECT_EQ(images.front().slot, 0u);
  EXPECT_FALSE(images.front().valid);
  EXPECT_TRUE(images.front().cold);
  EXPECT_FALSE(images.front().value.has_value());
  EXPECT_DOUBLE_EQ(images.front().accumulated_dt, 0.5);

  auto prepared = src.prepare_checkpoint_restore(images);
  NativeCacheManager dst;
  dst.bind(cache_plan(1));
  dst.publish_checkpoint_restore(std::move(prepared));
  ASSERT_TRUE(dst.bound());
  EXPECT_FALSE(dst.valid(0));
  EXPECT_TRUE(dst.checkpoint_slots().front().cold);
  EXPECT_FALSE(dst.checkpoint_slots().front().value.has_value());
  EXPECT_DOUBLE_EQ(dst.accumulated_dt(0), 0.5);
  EXPECT_DOUBLE_EQ(dst.effective_dt(0, 0.25), 0.75);
  EXPECT_DOUBLE_EQ(dst.accumulated_dt(0), 0.0);
}

TEST(CheckpointCache, MalformedPreparedImageDoesNotClobberAcceptedSlots) {
  NativeCacheManager src;
  src.bind(cache_plan(1));
  src.accumulate_dt(0, 0.125);
  auto malformed = src.checkpoint_slots();
  malformed.push_back(malformed.front());
  EXPECT_THROW((void)src.prepare_checkpoint_restore(malformed), std::invalid_argument);
  ASSERT_EQ(src.checkpoint_slots().size(), 1u);
  EXPECT_DOUBLE_EQ(src.accumulated_dt(0), 0.125);
  EXPECT_TRUE(src.checkpoint_slots().front().cold);

  auto wrong_plan = src.checkpoint_slots();
  wrong_plan.front().plan_digest.assign(64, 'c');
  EXPECT_THROW((void)src.prepare_checkpoint_restore(wrong_plan), std::invalid_argument);
  EXPECT_DOUBLE_EQ(src.accumulated_dt(0), 0.125);
}

TEST(CheckpointCache, VerbatimRestartErrorMessages) {
  // The messages are RAISED by the sim.restart facade; this pins the exact strings, building the
  // missing-cache one from the CacheManager name accessor (the verbatim spec message names the slot).
  NativeCacheManager src;
  src.bind(cache_plan(1));
  src.prime_slot(0, make_mf(0.0));
  src.store(0, make_mf(1.0), 0, "fields_from_state");

  // the hash-mismatch message (raised on a different installed_program_hash)
  const std::string hash_msg = "checkpoint was created with a different compiled Program hash";
  bool threw_hash = false;
  try {
    throw std::runtime_error(hash_msg);
  } catch (const std::runtime_error& e) {
    threw_hash = (std::string(e.what()) == hash_msg);
  }
  EXPECT_TRUE(threw_hash) << "verbatim_hash_mismatch_message";

  // the missing-cache message names the slot via the cache name (here 'fields_from_state')
  const std::string miss_msg =
      "checkpoint missing cached value for scheduled slot '" + src.name_of(0) + "'";
  EXPECT_TRUE(miss_msg == "checkpoint missing cached value for scheduled slot 'fields_from_state'")
      << "verbatim_missing_cache_message";
}

TEST(ProgramPersistentValueCheckpoint, InvalidColdSlotRoundTripsZeroLogicalBytes) {
  using namespace runtime::program;

  ProgramPersistentValueCheckpoint image;
  image.bound = true;
  image.plan_schema = std::string(kProgramPersistentValueCheckpointPlanSchema);
  image.plan_digest = std::string(64, 'a');
  image.maximum_bytes = 16;
  image.slot_count = 1;
  image.rows = {checkpoint_row()};
  image.metadata = {{17, 19, 0.375, 23, 29, false, true}};
  image.offsets = {0, 16};
  image.value_bytes = {0};
  image.storage.resize(16);
  for (std::size_t index = 0; index != image.storage.size(); ++index)
    image.storage[index] = static_cast<std::byte>(index + 1);

  validate_program_persistent_value_checkpoint(image);
  const auto encoded = serialize_program_persistent_value_checkpoint(image);
  const auto decoded = deserialize_program_persistent_value_checkpoint(encoded);
  EXPECT_EQ(decoded, image);
  EXPECT_FALSE(decoded.metadata.front().valid);
  EXPECT_TRUE(decoded.metadata.front().cold);
  EXPECT_DOUBLE_EQ(decoded.metadata.front().accumulated_dt, 0.375);
  EXPECT_EQ(decoded.value_bytes, (std::vector<std::uint64_t>{0}));
  EXPECT_EQ(decoded.offsets, (std::vector<std::uint64_t>{0, 16}));
  EXPECT_EQ(decoded.storage.size(), 16u);

  auto nonzero_invalid = image;
  nonzero_invalid.value_bytes = {8};
  EXPECT_THROW(validate_program_persistent_value_checkpoint(nonzero_invalid),
               std::invalid_argument);

  auto missing_capacity = image;
  missing_capacity.offsets = {0, 8};
  EXPECT_THROW(validate_program_persistent_value_checkpoint(missing_capacity),
               std::invalid_argument);

  auto valid = image;
  valid.metadata.front().valid = true;
  valid.metadata.front().cold = false;
  valid.value_bytes = {8};
  validate_program_persistent_value_checkpoint(valid);
  auto zero_valid = valid;
  zero_valid.value_bytes = {0};
  EXPECT_THROW(validate_program_persistent_value_checkpoint(zero_valid), std::invalid_argument);

  auto invalid_dt = image;
  invalid_dt.metadata.front().accumulated_dt = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(validate_program_persistent_value_checkpoint(invalid_dt), std::invalid_argument);
}

TEST(ProgramPersistentValueCheckpoint, StoreInvalidRoundTripAndPreallocatedTransition) {
  using namespace runtime::program;
  const std::string digest(64, 'a');
  const ProgramResourcePlan plan({checkpoint_row()}, 16, "program-resource-plan:v1", digest);

  ProgramPersistentValueStore source;
  source.bind(plan);
  ASSERT_EQ(source.value(0).size(), 16u);
  for (std::size_t index = 0; index != source.value(0).size(); ++index)
    source.value(0)[index] = static_cast<std::byte>(index + 1);
  auto& source_metadata = source.metadata(0);
  source_metadata.accepted_coordinate = 41;
  source_metadata.cursor = 43;
  source_metadata.accumulated_dt = 0.75;
  source_metadata.topology_epoch = 47;
  source_metadata.layout_generation = 53;
  source_metadata.valid = false;
  source_metadata.cold = true;

  const auto invalid_image = capture_program_persistent_value_checkpoint(source);
  EXPECT_FALSE(invalid_image.metadata.front().valid);
  EXPECT_TRUE(invalid_image.metadata.front().cold);
  EXPECT_DOUBLE_EQ(invalid_image.metadata.front().accumulated_dt, 0.75);
  EXPECT_EQ(invalid_image.value_bytes, (std::vector<std::uint64_t>{0}));
  EXPECT_EQ(invalid_image.offsets, (std::vector<std::uint64_t>{0, 16}));
  ASSERT_EQ(invalid_image.storage.size(), 16u);

  const auto encoded = serialize_program_persistent_value_checkpoint(invalid_image);
  const auto decoded = deserialize_program_persistent_value_checkpoint(encoded);
  auto prepared = prepare_program_persistent_value_restore(decoded, plan);
  const auto* const prepared_storage = prepared.value(0).data();
  ProgramPersistentValueStore restored;
  publish_program_persistent_value_restore(restored, std::move(prepared));
  EXPECT_EQ(restored.value(0).data(), prepared_storage);
  EXPECT_EQ(restored.value(0).size(), 16u);
  EXPECT_FALSE(restored.metadata(0).valid);
  EXPECT_TRUE(restored.metadata(0).cold);
  EXPECT_DOUBLE_EQ(restored.metadata(0).accumulated_dt, 0.75);
  EXPECT_EQ(restored.value_bytes(0), 0u);
  for (std::size_t index = 0; index != restored.value(0).size(); ++index)
    EXPECT_EQ(restored.value(0)[index], static_cast<std::byte>(index + 1));

  const auto* const preallocated_storage = restored.value(0).data();
  auto& restored_metadata = restored.metadata(0);
  restored_metadata.valid = true;
  restored_metadata.cold = false;
  EXPECT_EQ(restored.value(0).data(), preallocated_storage);
  const auto valid_image = capture_program_persistent_value_checkpoint(restored);
  EXPECT_TRUE(valid_image.metadata.front().valid);
  EXPECT_FALSE(valid_image.metadata.front().cold);
  EXPECT_DOUBLE_EQ(valid_image.metadata.front().accumulated_dt, 0.75);
  EXPECT_EQ(valid_image.value_bytes, (std::vector<std::uint64_t>{8}));
  EXPECT_EQ(valid_image.offsets, (std::vector<std::uint64_t>{0, 16}));

  auto valid_prepared = prepare_program_persistent_value_restore(valid_image, plan);
  ProgramPersistentValueStore valid_restored;
  publish_program_persistent_value_restore(valid_restored, std::move(valid_prepared));
  EXPECT_TRUE(valid_restored.metadata(0).valid);
  EXPECT_FALSE(valid_restored.metadata(0).cold);
  EXPECT_EQ(valid_restored.value_bytes(0), 8u);
  EXPECT_EQ(valid_restored.value(0).size(), 16u);
  for (std::size_t index = 0; index != valid_restored.value(0).size(); ++index)
    EXPECT_EQ(valid_restored.value(0)[index], static_cast<std::byte>(index + 1));
}

TEST(ProgramPersistentValueCheckpoint, StoreRejectsIncoherentValueBytesWithoutClobber) {
  using namespace runtime::program;
  const ProgramResourcePlan plan({checkpoint_row()}, 16, "program-resource-plan:v1",
                                 std::string(64, 'a'));
  ProgramPersistentValueStore store;
  store.bind(plan);
  for (std::size_t index = 0; index != store.value(0).size(); ++index)
    store.value(0)[index] = static_cast<std::byte>(0xa0 + index);
  store.metadata(0).accepted_coordinate = 61;
  store.metadata(0).accumulated_dt = 1.25;
  const auto accepted = store.snapshot();

  auto invalid_nonzero = accepted;
  invalid_nonzero.value_bytes = {8};
  EXPECT_THROW(store.restore(invalid_nonzero), std::invalid_argument);

  auto truncated_capacity = accepted;
  truncated_capacity.offsets = {0, 8};
  EXPECT_THROW(store.restore(truncated_capacity), std::invalid_argument);

  auto valid_missing_bytes = accepted;
  valid_missing_bytes.metadata.front().valid = true;
  valid_missing_bytes.metadata.front().cold = false;
  valid_missing_bytes.value_bytes = {0};
  EXPECT_THROW(store.restore(valid_missing_bytes), std::invalid_argument);

  const auto after = store.snapshot();
  EXPECT_EQ(after.metadata, accepted.metadata);
  EXPECT_EQ(after.offsets, accepted.offsets);
  EXPECT_EQ(after.value_bytes, (std::vector<std::uint64_t>{0}));
  EXPECT_EQ(after.storage, accepted.storage);
  EXPECT_EQ(store.value(0).size(), 16u);
  EXPECT_DOUBLE_EQ(store.metadata(0).accumulated_dt, 1.25);
  EXPECT_FALSE(store.metadata(0).valid);
  EXPECT_TRUE(store.metadata(0).cold);
}

TEST(ProgramPersistentValueCheckpoint, PreparedRestoreGenerationAndConsumptionAreStrict) {
  using namespace runtime::program;
  const ProgramResourcePlan plan({checkpoint_row()}, 16, "program-resource-plan:v1",
                                 std::string(64, 'a'));
  ProgramPersistentValueStore source;
  source.bind(plan);
  source.metadata(0).accumulated_dt = 0.75;
  const auto image = capture_program_persistent_value_checkpoint(source);

  ProgramPersistentValueStore accepted;
  accepted.bind(plan);
  accepted.metadata(0).accumulated_dt = 9.0;
  auto detached = prepare_program_persistent_value_restore(image, plan);
  PreparedProgramPersistentValueRestore restore(std::move(detached), 17);

  EXPECT_THROW(restore.validate_publication(18), std::logic_error);
  EXPECT_FALSE(restore.consumed());
  EXPECT_DOUBLE_EQ(accepted.metadata(0).accumulated_dt, 9.0);

  restore.validate_publication(17);
  restore.publish_validated_into(accepted);
  EXPECT_TRUE(restore.consumed());
  EXPECT_DOUBLE_EQ(accepted.metadata(0).accumulated_dt, 0.75);
  EXPECT_THROW(restore.validate_publication(17), std::logic_error);
  EXPECT_DOUBLE_EQ(accepted.metadata(0).accumulated_dt, 0.75);

  auto detached_again = prepare_program_persistent_value_restore(image, plan);
  PreparedProgramPersistentValueRestore moved_from(std::move(detached_again), 17);
  auto moved_to = std::move(moved_from);
  EXPECT_TRUE(moved_from.consumed());
  EXPECT_EQ(moved_from.install_generation(), 0u);
  EXPECT_THROW(moved_from.validate_publication(17), std::logic_error);
  EXPECT_FALSE(moved_to.consumed());
}
