// CacheManager: the per-slot value cache backing the unified Program scheduler (Spec 3, ADC-458).
// Exercises the due/store/retrieve/accumulate logic + cold-start + a MultiFab store-retrieve
// bit-identity, with no Program/codegen needed.

#include <gtest/gtest.h>

#include <pops/runtime/program/cache_manager.hpp>

#include <pops/core/foundation/allocator.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/layout/rank_space.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>

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

Field make_mf(double fill) {
  const Box<kDim> domain = Box<kDim>::from_extents(filled_extent(8));
  const mesh::BoxArray<kDim> layout = mesh::BoxArray<kDim>::from_domain(domain, filled_extent(4));
  const mesh::RankSpace<kDim> ranks(Index<kDim>{}, filled_extent(1));
  const auto distribution = mesh::Distribution<kDim>::replicated(layout, ranks);
  Field mf(layout, distribution, Index<kDim>{}, /*ncomp=*/1, filled_extent(1));
  mf.set_val(fill);
  return mf;
}

runtime::program::ProgramResourcePlan make_plan(std::size_t count) {
  using namespace runtime::program;
  std::vector<ProgramResourcePlanEntry> rows;
  rows.reserve(count);
  for (std::size_t slot = 0; slot != count; ++slot) {
    ProgramResourcePlanEntry row;
    row.slot = static_cast<std::uint32_t>(slot);
    row.key = {100 + slot, 200 + slot, 0, 1, 2, -1};
    row.identity = "cache-value-" + std::to_string(slot);
    row.occurrence_path = "program/root/" + std::to_string(slot);
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
                             std::string(64, 'a'));
}

NativeCacheManager make_cache(std::size_t count) {
  NativeCacheManager cache;
  cache.bind(make_plan(count));
  return cache;
}

}  // namespace

// Each TEST below builds its own fresh CacheManager: these are genuinely independent sections.

TEST(CacheManager, IsDueColdStartThenEveryN) {
  NativeCacheManager c = make_cache(1);
  EXPECT_TRUE(c.is_due(0, 0, 10)) << "cold_start_due";  // never stored -> due
  EXPECT_TRUE(!c.has(0)) << "cold_start_absent";
  const Field prototype = make_mf(0.0);
  c.prime_slot(0, prototype);
  c.store(0, make_mf(1.0), 0);
  EXPECT_TRUE(c.has(0)) << "present_after_store";
  EXPECT_TRUE(c.is_due(0, 0, 10)) << "due_at_step0";  // 0 % 10 == 0
  EXPECT_TRUE(!c.is_due(0, 1, 10)) << "not_due_at_step1";
  EXPECT_TRUE(!c.is_due(0, 9, 10)) << "not_due_at_step9";
  EXPECT_TRUE(c.is_due(0, 10, 10)) << "due_at_step10";
  EXPECT_TRUE(c.is_due(0, 20, 10)) << "due_at_step20";
  EXPECT_TRUE(c.is_due(0, 5, 1)) << "every1_always_due";  // every_n<=1 -> always
}

TEST(CacheManager, BindSealsPlanAndRejectsUnprimedHotStore) {
  NativeCacheManager c = make_cache(1);
  EXPECT_EQ(c.plan_schema(), "program-resource-plan:v1");
  EXPECT_EQ(c.plan_digest(), std::string(64, 'a'));
  EXPECT_EQ(c.plan_entry(0).identity, "cache-value-0");
  EXPECT_TRUE(c.cold(0));
  EXPECT_THROW(c.store(0, make_mf(1.0), 0), std::logic_error);

  // Explicit preparation is the only allocation boundary for a cache value.
  c.prime_slot(0, make_mf(0.0));
  c.store(0, make_mf(1.0), 0);
  EXPECT_FALSE(c.cold(0));
}

TEST(CacheManager, StoreRetrieveBitIdentity) {
  NativeCacheManager c = make_cache(1);
  Field v = make_mf(2.0);
  const double want = reduce_sum_local(v);
  c.prime_slot(0, v);
  c.store(0, v, 3);
  const Field& got = c.retrieve(0);
  EXPECT_TRUE(std::fabs(reduce_sum_local(got) - want) < 1e-12) << "retrieve_bit_identity";
  EXPECT_TRUE(c.last_update_step(0) == 3) << "last_update_step";
  // a second store overwrites + refreshes the step
  c.store(0, make_mf(5.0), 11);
  EXPECT_TRUE(std::fabs(reduce_sum_local(c.retrieve(0)) - 5.0 * valid_cell_count()) < 1e-12)
      << "store_overwrites";
  EXPECT_TRUE(c.last_update_step(0) == 11) << "last_update_step_refreshed";
}

TEST(CacheManager, AccumulateDtSumsSkippedDtAndStoreResetsIt) {
  NativeCacheManager c = make_cache(1);
  c.prime_slot(0, make_mf(0.0));
  c.store(0, make_mf(1.0), 0);
  EXPECT_TRUE(std::fabs(c.accumulated_dt(0)) < 1e-15) << "accum_zero_after_store";
  c.accumulate_dt(0, 0.001);
  c.accumulate_dt(0, 0.002);
  c.accumulate_dt(0, 0.0005);
  EXPECT_TRUE(std::fabs(c.accumulated_dt(0) - 0.0035) < 1e-12)
      << "accum_sum";            // real sum, not N*dt
  c.store(0, make_mf(1.0), 10);  // recompute resets the accumulator
  EXPECT_TRUE(std::fabs(c.accumulated_dt(0)) < 1e-15) << "accum_reset_on_store";
}

TEST(CacheManager, AccumulateDtOnColdSlotAndEffectiveDt) {
  // accumulate_dt on a COLD slot + effective_dt (the due-step read, ADC-458).
  NativeCacheManager c = make_cache(1);
  // a cold accumulate_dt slot accumulates from its first skipped step
  EXPECT_TRUE(std::fabs(c.accumulated_dt(0)) < 1e-15) << "accum_cold_zero";
  c.accumulate_dt(0, 0.01);  // skipped step 1 (dt varies)
  c.accumulate_dt(0, 0.02);  // skipped step 2
  EXPECT_TRUE(std::fabs(c.accumulated_dt(0) - 0.03) < 1e-12) << "accum_cold_sum";
  // due step: eff_dt = dt_now + sum(skipped) = 0.005 + 0.03; resets the accumulator
  const double eff = c.effective_dt(0, 0.005);
  EXPECT_TRUE(std::fabs(eff - 0.035) < 1e-12) << "effective_dt_sum";  // NOT N * dt_current
  EXPECT_TRUE(std::fabs(c.accumulated_dt(0)) < 1e-15) << "effective_dt_resets";
  // a fresh window then accumulates from zero again
  c.accumulate_dt(0, 0.004);
  EXPECT_TRUE(std::fabs(c.effective_dt(0, 0.006) - 0.010) < 1e-12) << "effective_dt_fresh_window";
}

TEST(CacheManager, NamedScratchCacheDeepCopiesOnStore) {
  // A held rhs / source / linear_combine caches its OWN scratch through the same store/retrieve API
  // the aux uses; a deep copy survives a mutation of the source buffer.
  NativeCacheManager c = make_cache(1);
  Field scratch = make_mf(3.0);
  c.prime_slot(0, scratch);
  c.store(0, scratch, 0, "scratch");  // cache the scratch (deep copy)
  scratch.set_val(99.0);              // mutate the live buffer after caching
  EXPECT_TRUE(std::fabs(reduce_sum_local(c.retrieve(0)) - 3.0 * valid_cell_count()) < 1e-12)
      << "scratch_cache_deep_copy";
  // restoring (scratch = retrieve) overwrites the live buffer with the cached content
  scratch = c.retrieve(0);
  EXPECT_TRUE(std::fabs(reduce_sum_local(scratch) - 3.0 * valid_cell_count()) < 1e-12)
      << "scratch_restore";
}

TEST(CacheManager, WarmStoreAndRestoreReuseExactLayoutStorage) {
  NativeCacheManager cache = make_cache(1);
  Field first = make_mf(1.0);
  cache.prime_slot(0, first);
  cache.store(0, first, 0);
  const Real* const cached_storage = cache.retrieve(0).fab(0).storage().data();

  Field second = make_mf(7.0);
  Field restored = make_mf(-3.0);
  const Real* const restored_storage = restored.fab(0).storage().data();
  const AllocationEventStats before = allocation_event_stats();
  cache.store(0, second, 1);
  cache.restore_into(0, restored);
  const AllocationEventStats after = allocation_event_stats();

  EXPECT_EQ(cache.retrieve(0).fab(0).storage().data(), cached_storage);
  EXPECT_EQ(restored.fab(0).storage().data(), restored_storage);
  EXPECT_EQ(after.fab_calls, before.fab_calls);
  EXPECT_EQ(after.fab_bytes, before.fab_bytes);
  EXPECT_EQ(after.communication_calls, before.communication_calls);
  EXPECT_EQ(after.communication_bytes, before.communication_bytes);
  EXPECT_DOUBLE_EQ(reduce_sum_local(restored), 7.0 * valid_cell_count());

  second.set_val(99.0);
  restored.set_val(-1.0);
  EXPECT_DOUBLE_EQ(reduce_sum_local(cache.retrieve(0)), 7.0 * valid_cell_count())
      << "cache storage remains an independent deep value";
}

TEST(CacheManager, MultipleIndependentSlotsAndClear) {
  NativeCacheManager c = make_cache(2);
  c.prime_slot(0, make_mf(0.0));
  c.prime_slot(1, make_mf(0.0));
  c.store(0, make_mf(1.0), 0);
  c.store(1, make_mf(2.0), 0);
  EXPECT_TRUE(c.size() == 2) << "two_slots";
  EXPECT_TRUE(std::fabs(reduce_sum_local(c.retrieve(0)) - valid_cell_count()) < 1e-12)
      << "slot0_independent";
  EXPECT_TRUE(std::fabs(reduce_sum_local(c.retrieve(1)) - 2.0 * valid_cell_count()) < 1e-12)
      << "slot1_independent";
  c.clear();
  EXPECT_TRUE(c.size() == 0 && !c.bound()) << "clear";
}
