// Profiler: per-node / per-brick timing accumulation + counters + report (Spec 3 section 29-30,
// ADC-459). Exercises the data structure directly (no step needed): record/aggregate, RAII scope
// timing + nesting, the disabled no-op path, counters, and the report contents.

#include <gtest/gtest.h>

#include <pops/runtime/program/profiler.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

using pops::runtime::program::Profiler;
using pops::runtime::program::ProfileScope;

namespace {

void busy_us(int micros) {
  // A real elapsed interval (steady_clock-measurable) without depending on sleep precision.
  const auto t0 = std::chrono::steady_clock::now();
  const auto want = std::chrono::microseconds(micros);
  while (std::chrono::steady_clock::now() - t0 < want) {
    // spin
  }
}

}  // namespace

// Each TEST below builds its own fresh Profiler: these are genuinely independent sections, not
// phases of one ordered pipeline.

TEST(Profiler, RecordAggregatesCountTotalMinMax) {
  Profiler p;
  p.enable();
  p.record("riemann", 0.20);
  p.record("riemann", 0.10);
  p.record("riemann", 0.30);
  const auto* e = p.entry("riemann");
  ASSERT_TRUE(e != nullptr) << "entry_present";
  EXPECT_TRUE(e->count == 3) << "count";
  EXPECT_TRUE(std::abs(e->total_s - 0.60) < 1e-12) << "total";
  EXPECT_TRUE(std::abs(e->min_s - 0.10) < 1e-12) << "min";
  EXPECT_TRUE(std::abs(e->max_s - 0.30) < 1e-12) << "max";
  EXPECT_TRUE(std::abs(e->mean_s() - 0.20) < 1e-12) << "mean";
  EXPECT_TRUE(std::abs(p.total_s() - 0.60) < 1e-12) << "total_s";
  EXPECT_TRUE(p.scope_count() == 1) << "scope_count";
}

TEST(Profiler, DisabledIsNoop) {
  Profiler p;  // disabled by default
  EXPECT_TRUE(!p.enabled()) << "disabled_by_default";
  p.record("x", 1.0);
  p.count("k");
  EXPECT_TRUE(p.entry("x") == nullptr) << "disabled_record_noop";
  EXPECT_TRUE(p.counter("k") == 0) << "disabled_count_noop";
  p.enable();
  p.record("x", 1.0);
  p.disable();
  p.record("x", 1.0);  // ignored
  EXPECT_TRUE(p.entry("x") != nullptr && p.entry("x")->count == 1) << "reenable_then_disable";
}

TEST(Profiler, CountersAccumulateAndUnknownReadsZero) {
  Profiler p;
  p.enable();
  p.count("kernels", 3);
  p.count("kernels");
  p.count("cache_hits", 5);
  EXPECT_TRUE(p.counter("kernels") == 4) << "counter_sum";
  EXPECT_TRUE(p.counter("cache_hits") == 5) << "counter_value";
  EXPECT_TRUE(p.counter("never") == 0) << "counter_absent_zero";
}

TEST(Profiler, ScheduleDecisionCountsOnlyRealCacheTraffic) {
  Profiler p;
  // Disabled profiling preserves the decision and materializes no counters.
  EXPECT_TRUE(p.schedule_decision(true, true));
  EXPECT_TRUE(p.counter("nodes_due") == 0 && p.counter("cache_misses") == 0);

  p.enable();
  EXPECT_TRUE(p.schedule_decision(true, true));
  EXPECT_FALSE(p.schedule_decision(false, true));
  EXPECT_TRUE(p.schedule_decision(true, false));
  EXPECT_FALSE(p.schedule_decision(false, false));
  EXPECT_TRUE(p.counter("nodes_due") == 2);
  EXPECT_TRUE(p.counter("nodes_skipped") == 2);
  EXPECT_TRUE(p.counter("cache_misses") == 1);
  EXPECT_TRUE(p.counter("cache_hits") == 1);
}

TEST(Profiler, CountMaxTracksPeakNotSum) {
  // count_max tracks a PEAK, not a sum (scratch peak memory, ADC-459).
  Profiler p;
  p.enable();
  p.count_max("scratch_peak_bytes", 100);  // first-seen creates at the value
  EXPECT_TRUE(p.counter("scratch_peak_bytes") == 100) << "count_max_first";
  p.count_max("scratch_peak_bytes", 40);  // smaller: peak unchanged
  EXPECT_TRUE(p.counter("scratch_peak_bytes") == 100) << "count_max_keeps_peak";
  p.count_max("scratch_peak_bytes", 250);  // larger: peak rises
  EXPECT_TRUE(p.counter("scratch_peak_bytes") == 250) << "count_max_rises";
  // disabled count_max is a no-op
  p.disable();
  p.count_max("scratch_peak_bytes", 9999);
  p.enable();
  EXPECT_TRUE(p.counter("scratch_peak_bytes") == 250) << "count_max_disabled_noop";
}

TEST(Profiler, ProfileScopeTimesRealIntervalAndNests) {
  Profiler p;
  p.enable();
  {
    ProfileScope outer(p, "step");
    busy_us(300);
    {
      ProfileScope inner(p, "riemann");
      busy_us(150);
    }
  }
  const auto* step = p.entry("step");
  const auto* riemann = p.entry("riemann");
  EXPECT_TRUE(step != nullptr && step->count == 1) << "scope_outer_recorded";
  EXPECT_TRUE(riemann != nullptr && riemann->count == 1) << "scope_inner_recorded";
  // the outer scope encloses the inner, so it is at least as long
  EXPECT_TRUE(step != nullptr && riemann != nullptr && step->total_s >= riemann->total_s)
      << "scope_nesting_order";
  EXPECT_TRUE(step != nullptr && step->total_s > 0.0) << "scope_positive_time";
}

TEST(Profiler, DisabledProfileScopeRecordsNothing) {
  Profiler p;  // off
  {
    ProfileScope s(p, "ignored");
    busy_us(100);
  }
  EXPECT_TRUE(p.entry("ignored") == nullptr) << "disabled_scope_noop";
}

TEST(Profiler, ReportIsFirstSeenOrderAndContainsScopesAndCounters) {
  Profiler p;
  p.enable();
  p.record("fields", 0.5);
  p.record("transport", 0.25);
  p.count("nodes_skipped", 2);
  const std::string r = p.report();
  EXPECT_TRUE(r.find("fields") != std::string::npos) << "report_has_fields";
  EXPECT_TRUE(r.find("transport") != std::string::npos) << "report_has_transport";
  EXPECT_TRUE(r.find("nodes_skipped=2") != std::string::npos) << "report_has_counter";
  EXPECT_TRUE(r.find("fields") < r.find("transport")) << "report_first_seen_order";
}

TEST(Profiler, ConcurrentWritersAggregateExactlyAndPreserveEstablishedOrder) {
  constexpr int kThreads = 8;
  constexpr int kSamplesPerThread = 1500;
  constexpr std::uint64_t kConcurrentSamples = kThreads * kSamplesPerThread;

  Profiler p;
  p.enable();

  // Establish a deterministic first-seen order before the concurrent update phase. Concurrent
  // writers may arrive in any order, but they must not reorder existing names or lose updates.
  p.record("scope:first", 0.25);
  p.record("scope:second", 0.5);
  p.count("events", 0);
  p.count_max("peak", 0);

  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (int worker = 0; worker < kThreads; ++worker) {
    workers.emplace_back([&p, worker] {
      for (int sample = 0; sample < kSamplesPerThread; ++sample) {
        // Exercise both lock arrival orders rather than serializing all scope calls identically.
        if (((worker + sample) & 1) == 0) {
          p.record("scope:first", 0.25);
          p.record("scope:second", 0.5);
        } else {
          p.record("scope:second", 0.5);
          p.record("scope:first", 0.25);
        }
        p.count("events");
        p.count_max("peak", worker * kSamplesPerThread + sample);
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }

  const Profiler::Snapshot snapshot = p.snapshot();
  ASSERT_EQ(snapshot.scopes.size(), 2U);
  EXPECT_EQ(snapshot.scopes[0].name, "scope:first");
  EXPECT_EQ(snapshot.scopes[1].name, "scope:second");
  EXPECT_EQ(snapshot.scopes[0].count, kConcurrentSamples + 1);
  EXPECT_EQ(snapshot.scopes[1].count, kConcurrentSamples + 1);
  EXPECT_DOUBLE_EQ(snapshot.scopes[0].total_s, static_cast<double>(kConcurrentSamples + 1) * 0.25);
  EXPECT_DOUBLE_EQ(snapshot.scopes[1].total_s, static_cast<double>(kConcurrentSamples + 1) * 0.5);

  ASSERT_EQ(snapshot.counters.size(), 2U);
  EXPECT_EQ(snapshot.counters[0].name, "events");
  EXPECT_EQ(snapshot.counters[0].value, static_cast<std::int64_t>(kConcurrentSamples));
  EXPECT_EQ(snapshot.counters[1].name, "peak");
  EXPECT_EQ(snapshot.counters[1].value, static_cast<std::int64_t>(kConcurrentSamples - 1));

  const std::string report = p.report();
  EXPECT_LT(report.find("scope:first"), report.find("scope:second"));
  EXPECT_LT(report.find("events="), report.find("peak="));
}

TEST(Profiler, RuntimeSnapshotCopyAndRestorePreserveAccumulatedState) {
  Profiler source;
  source.enable();
  source.record("step", 0.25);
  source.count("kernels", 3);

  const Profiler snapshot(source);
  source.record("step", 0.75);
  source.count("kernels", 4);

  Profiler restored;
  restored = snapshot;
  const Profiler::Snapshot state = restored.snapshot();
  ASSERT_EQ(state.scopes.size(), 1U);
  EXPECT_EQ(state.scopes[0].name, "step");
  EXPECT_EQ(state.scopes[0].count, 1U);
  EXPECT_DOUBLE_EQ(state.scopes[0].total_s, 0.25);
  ASSERT_EQ(state.counters.size(), 1U);
  EXPECT_EQ(state.counters[0].name, "kernels");
  EXPECT_EQ(state.counters[0].value, 3);
  EXPECT_TRUE(state.enabled);
}

TEST(Profiler, ResetClearsEverything) {
  Profiler p;
  p.enable();
  p.record("a", 1.0);
  p.count("c");
  p.reset();
  EXPECT_TRUE(p.entry("a") == nullptr && p.counter("c") == 0 && p.scope_count() == 0) << "reset";
}
