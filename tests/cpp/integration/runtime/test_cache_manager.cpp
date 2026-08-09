// CacheManager: the per-node value cache backing the unified Program scheduler (Spec 3, ADC-458).
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

}  // namespace

// Each TEST below builds its own fresh CacheManager: these are genuinely independent sections.

TEST(CacheManager, IsDueColdStartThenEveryN) {
  NativeCacheManager c;
  EXPECT_TRUE(c.is_due(1, 0, 10)) << "cold_start_due";  // never stored -> due
  EXPECT_TRUE(!c.has(1)) << "cold_start_absent";
  c.store(1, make_mf(1.0), 0);
  EXPECT_TRUE(c.has(1)) << "present_after_store";
  EXPECT_TRUE(c.is_due(1, 0, 10)) << "due_at_step0";  // 0 % 10 == 0
  EXPECT_TRUE(!c.is_due(1, 1, 10)) << "not_due_at_step1";
  EXPECT_TRUE(!c.is_due(1, 9, 10)) << "not_due_at_step9";
  EXPECT_TRUE(c.is_due(1, 10, 10)) << "due_at_step10";
  EXPECT_TRUE(c.is_due(1, 20, 10)) << "due_at_step20";
  EXPECT_TRUE(c.is_due(1, 5, 1)) << "every1_always_due";  // every_n<=1 -> always
}

TEST(CacheManager, StoreRetrieveBitIdentity) {
  NativeCacheManager c;
  Field v = make_mf(2.0);
  const double want = reduce_sum_local(v);
  c.store(7, v, 3);
  const Field& got = c.retrieve(7);
  EXPECT_TRUE(std::fabs(reduce_sum_local(got) - want) < 1e-12) << "retrieve_bit_identity";
  EXPECT_TRUE(c.last_update_step(7) == 3) << "last_update_step";
  // a second store overwrites + refreshes the step
  c.store(7, make_mf(5.0), 11);
  EXPECT_TRUE(std::fabs(reduce_sum_local(c.retrieve(7)) - 5.0 * valid_cell_count()) < 1e-12)
      << "store_overwrites";
  EXPECT_TRUE(c.last_update_step(7) == 11) << "last_update_step_refreshed";
}

TEST(CacheManager, AccumulateDtSumsSkippedDtAndStoreResetsIt) {
  NativeCacheManager c;
  c.store(2, make_mf(1.0), 0);
  EXPECT_TRUE(std::fabs(c.accumulated_dt(2)) < 1e-15) << "accum_zero_after_store";
  c.accumulate_dt(2, 0.001);
  c.accumulate_dt(2, 0.002);
  c.accumulate_dt(2, 0.0005);
  EXPECT_TRUE(std::fabs(c.accumulated_dt(2) - 0.0035) < 1e-12)
      << "accum_sum";            // real sum, not N*dt
  c.store(2, make_mf(1.0), 10);  // recompute resets the accumulator
  EXPECT_TRUE(std::fabs(c.accumulated_dt(2)) < 1e-15) << "accum_reset_on_store";
}

TEST(CacheManager, AccumulateDtOnColdNodeAndEffectiveDt) {
  // accumulate_dt on a COLD node + effective_dt (the due-step read, ADC-458).
  NativeCacheManager c;
  // a cold accumulate_dt node accumulates from its first skipped step (no slot needed first)
  EXPECT_TRUE(std::fabs(c.accumulated_dt(3)) < 1e-15) << "accum_cold_zero";
  c.accumulate_dt(3, 0.01);  // skipped step 1 (dt varies)
  c.accumulate_dt(3, 0.02);  // skipped step 2
  EXPECT_TRUE(std::fabs(c.accumulated_dt(3) - 0.03) < 1e-12) << "accum_cold_sum";
  // due step: eff_dt = dt_now + sum(skipped) = 0.005 + 0.03; resets the accumulator
  const double eff = c.effective_dt(3, 0.005);
  EXPECT_TRUE(std::fabs(eff - 0.035) < 1e-12) << "effective_dt_sum";  // NOT N * dt_current
  EXPECT_TRUE(std::fabs(c.accumulated_dt(3)) < 1e-15) << "effective_dt_resets";
  // a fresh window then accumulates from zero again
  c.accumulate_dt(3, 0.004);
  EXPECT_TRUE(std::fabs(c.effective_dt(3, 0.006) - 0.010) < 1e-12) << "effective_dt_fresh_window";
}

TEST(CacheManager, NamedScratchCacheDeepCopiesOnStore) {
  // A held rhs / source / linear_combine caches its OWN scratch through the same store/retrieve API
  // the aux uses; a deep copy survives a mutation of the source buffer.
  NativeCacheManager c;
  Field scratch = make_mf(3.0);
  c.store(9, scratch, 0);  // cache the scratch (deep copy)
  scratch.set_val(99.0);   // mutate the live buffer after caching
  EXPECT_TRUE(std::fabs(reduce_sum_local(c.retrieve(9)) - 3.0 * valid_cell_count()) < 1e-12)
      << "scratch_cache_deep_copy";
  // restoring (scratch = retrieve) overwrites the live buffer with the cached content
  scratch = c.retrieve(9);
  EXPECT_TRUE(std::fabs(reduce_sum_local(scratch) - 3.0 * valid_cell_count()) < 1e-12)
      << "scratch_restore";
}

TEST(CacheManager, WarmStoreAndRestoreReuseExactLayoutStorage) {
  NativeCacheManager cache;
  Field first = make_mf(1.0);
  cache.store(4, first, 0);
  const Real* const cached_storage = cache.retrieve(4).fab(0).storage().data();

  Field second = make_mf(7.0);
  Field restored = make_mf(-3.0);
  const Real* const restored_storage = restored.fab(0).storage().data();
  const AllocationEventStats before = allocation_event_stats();
  cache.store(4, second, 1);
  cache.restore_into(4, restored);
  const AllocationEventStats after = allocation_event_stats();

  EXPECT_EQ(cache.retrieve(4).fab(0).storage().data(), cached_storage);
  EXPECT_EQ(restored.fab(0).storage().data(), restored_storage);
  EXPECT_EQ(after.fab_calls, before.fab_calls);
  EXPECT_EQ(after.fab_bytes, before.fab_bytes);
  EXPECT_EQ(after.communication_calls, before.communication_calls);
  EXPECT_EQ(after.communication_bytes, before.communication_bytes);
  EXPECT_DOUBLE_EQ(reduce_sum_local(restored), 7.0 * valid_cell_count());

  second.set_val(99.0);
  restored.set_val(-1.0);
  EXPECT_DOUBLE_EQ(reduce_sum_local(cache.retrieve(4)), 7.0 * valid_cell_count())
      << "cache storage remains an independent deep value";
}

TEST(CacheManager, MultipleIndependentNodesAndClear) {
  NativeCacheManager c;
  c.store(1, make_mf(1.0), 0);
  c.store(2, make_mf(2.0), 0);
  EXPECT_TRUE(c.size() == 2) << "two_slots";
  EXPECT_TRUE(std::fabs(reduce_sum_local(c.retrieve(1)) - valid_cell_count()) < 1e-12)
      << "node1_independent";
  EXPECT_TRUE(std::fabs(reduce_sum_local(c.retrieve(2)) - 2.0 * valid_cell_count()) < 1e-12)
      << "node2_independent";
  c.clear();
  EXPECT_TRUE(c.size() == 0 && !c.has(1)) << "clear";
}
