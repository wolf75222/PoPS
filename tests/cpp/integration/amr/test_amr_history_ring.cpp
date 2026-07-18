// ADC-631 -- multistep HISTORY RINGS on the compiled-Program AMR route (design plan sections 2-3).
//
// The engine-level ring store + seams (detail::AmrHistoryOps) and the regrid remap hook, tested
// DIRECTLY on an AmrRuntime (no compiled .so): the per-level ring semantics (register / store / read /
// rotate), the flat checkpoint round-trip (history_global / restore_history), the per-level cold-start
// fill, and the regrid remap (a fine ring slot stays finite + correctly sized on the NEW layout, the
// coarse slot is untouched). The compiled-Program parity / checkpoint / replay live in the Python
// acceptance suite (Kokkos-gated); this locks the pure C++ ring mechanics.
//
// Fixture idiom (nvcc-safe, like test_amr_multiblock_regrid_union): build the AmrRuntime DIRECTLY via
// detail::make_shared_amr_layout + detail::dispatch_amr_block; the tag predicate is a NAMED functor
// (host loop of tag_cells, never on device).

#include <gtest/gtest.h>

#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // detail::make_shared_amr_layout / dispatch_amr_block
#include <pops/runtime/amr/amr_runtime.hpp>                  // AmrRuntime + detail::AmrHistoryOps
#include <pops/runtime/amr_system.hpp>                       // facade transaction boundary
#include <pops/runtime/program/step_transaction.hpp>        // StepAttemptRejected fault signal
#include <pops/runtime/program/amr_program_context.hpp>     // native AB2/reflux context
#include <pops/runtime/builders/factory/model_factory.hpp>  // detail::dispatch_model
#include <pops/runtime/config/model_spec.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

static ModelSpec exb_charge(double q, double B0) {
  ModelSpec s;
  s.transport = "exb";
  s.source = "none";
  s.elliptic = "charge";
  s.q = q;
  s.B0 = B0;
  return s;
}

// Tag if density (component 0) exceeds a threshold -- a per-block regrid criterion.
struct TagDensityAbove {
  Real thr;
  bool operator()(const ConstArray4& a, int i, int j) const { return a(i, j, 0) > thr; }
};

static std::vector<double> blob(int n, double cx, double cy, double amp, double base,
                                double width) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n, base);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n, y = (j + 0.5) / n;
      const double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      rho[static_cast<std::size_t>(j) * n + i] = base + amp * std::exp(-r2 / (width * width));
    }
  return rho;
}

static bool all_finite(const std::vector<double>& v) {
  for (double x : v)
    if (!std::isfinite(x))
      return false;
  return true;
}

static double dmax(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0;
  const std::size_t nn = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < nn; ++i)
    d = std::max(d, std::fabs(a[i] - b[i]));
  return d;
}

static bool same_patches(const std::vector<PatchBox>& a, const std::vector<PatchBox>& b) {
  if (a.size() != b.size())
    return false;
  for (std::size_t k = 0; k < a.size(); ++k)
    if (a[k].level != b[k].level || a[k].ilo != b[k].ilo || a[k].jlo != b[k].jlo ||
        a[k].ihi != b[k].ihi || a[k].jhi != b[k].jhi)
      return false;
  return true;
}

static double max_old_fine_child_group_spread(const std::vector<double>& history, int n,
                                              const std::vector<PatchBox>& old_patches) {
  const int nf = 2 * n;
  const std::size_t fine_offset = static_cast<std::size_t>(n) * n;
  double spread = 0.0;
  for (const PatchBox& patch : old_patches) {
    if (patch.level != 1)
      continue;
    const int ilo = patch.ilo + (patch.ilo & 1);
    const int jlo = patch.jlo + (patch.jlo & 1);
    for (int j = jlo; j + 1 <= patch.jhi; j += 2)
      for (int i = ilo; i + 1 <= patch.ihi; i += 2) {
        const auto at = [&](int ii, int jj) {
          return history[fine_offset + static_cast<std::size_t>(jj) * nf + ii];
        };
        const double lo = std::min({at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1)});
        const double hi = std::max({at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1)});
        spread = std::max(spread, hi - lo);
      }
  }
  return spread;
}

static AmrRuntime make_two_block(int N, double L, double B0, int manifest_ratio = kAmrRefRatio) {
  AmrBuildParams bp;
  bp.mesh.n = N;
  bp.mesh.L = L;
  bp.mesh.regrid_every = 0;
  bp.poisson.bc = BCRec{};  // periodic
  detail::SharedAmrLayout S = detail::make_shared_amr_layout(bp);
  std::vector<AmrRuntimeBlock> blocks;
  detail::dispatch_model(exb_charge(+1.0, B0), [&](auto m) {
    blocks.push_back(detail::dispatch_amr_block(m, "minmod", "rusanov", S, "a",
                                                blob(N, 0.35, 0.5, 0.8, 1.0, 0.10),
                                                /*has_density=*/true, 1.4, 1, false, false, 1));
  });
  detail::dispatch_model(exb_charge(-1.0, B0), [&](auto m) {
    blocks.push_back(detail::dispatch_amr_block(m, "minmod", "rusanov", S, "b",
                                                blob(N, 0.65, 0.5, 0.8, 1.0, 0.10),
                                                /*has_density=*/true, 1.4, 1, false, false, 1));
  });
  if (manifest_ratio != kAmrRefRatio) {
    S.refinement_ratios[0] = manifest_ratio;
    S.dx[1] = S.dx[0] / Real(manifest_ratio);
    S.dy[1] = S.dy[0] / Real(manifest_ratio);
    for (AmrRuntimeBlock& block : blocks) {
      (*block.levels)[1].dx = S.dx[1];
      (*block.levels)[1].dy = S.dy[1];
    }
  }
  AmrRuntime runtime(S.geom, S.runtime_hierarchy(), S.poisson_bc, std::move(blocks), S.base_per,
                     S.replicated_coarse, S.wall);
  runtime.set_parent_child_temporal_relations({::pops::amr::ParentChildClockRelation(
      0, 1, ::pops::amr::Rational(2, 1), ::pops::amr::RemainderPolicy::IntegralOnly)});
  return runtime;
}

static AmrRuntime* configure_native_ab2_regrid_system(AmrSystem& sim, int n,
                                                      int temporal_ratio = 1) {
  sim.set_temporal_relations({temporal_ratio}, {1}, {"integral_only"});
  sim.add_block("a", exb_charge(+1.0, 1.0), "minmod", "rusanov", "conservative", "explicit", 1);
  sim.add_block("b", exb_charge(-1.0, 1.0), "minmod", "rusanov", "conservative", "explicit", 1);
  sim.set_poisson("charge_density", "geometric_mg", "periodic");
  sim.set_refinement(1.2);
  sim.set_density("a", blob(n, 0.35, 0.5, 0.5, 1.0, 0.12));
  sim.set_density("b", blob(n, 0.65, 0.5, 0.5, 1.0, 0.12));
  sim.set_program_block_map({0, 1});
  if (!sim.uses_runtime_engine() || sim.engine() == nullptr)
    throw std::runtime_error("native AB2 fixture failed to build its AMR runtime engine");
  AmrRuntime* rt = sim.engine();
  rt->set_block_tag_predicate(0, TagDensityAbove{Real(1.2)});
  rt->set_block_tag_predicate(1, TagDensityAbove{Real(1.2)});
  return rt;
}

static void install_native_ab2_program(runtime::program::AmrProgramContext& context,
                                       std::function<void()> after_level = {}) {
  context.configure_primary_clock("clock.macro");
  context.register_history("a.rate", 1, -1, 0, "block.a.U", "cell.conservative", "clock.macro",
                           "dense.linear");
  context.install([&context, after_level = std::move(after_level)](double macro_dt) {
    context.advance_hierarchy(macro_dt, [&context, &after_level](double level_dt) {
      context.set_stage_time(0, 1);
      (void)context.solve_fields();
      MultiFab& state = context.state(0);
      MultiFab rate = context.rhs_scratch_like(state);
      context.rhs_into(0, state, rate, 17);
      context.store_history("a.rate", rate, 0);
      MultiFab& previous = context.history("a.rate", 1, 0);
      MultiFab next = context.scratch_state_like(state);
      context.lincomb(next, Real(1), state, Real(0), state);
      context.axpy(next, Real(1.5 * level_dt), rate, Real(level_dt), {{1, 3, 2}});
      context.axpy(next, Real(-0.5 * level_dt), previous, Real(level_dt), {{1, -1, 2}});
      context.lincomb(state, Real(0), state, Real(1), next);
      context.rotate_histories("clock.macro");
      if (after_level)
        after_level();
    });
  });
}

// Concatenated per-level flat of block 0's live state (the ground truth a stored ring slot mirrors).
static std::vector<double> block0_all_levels(AmrRuntime& rt) {
  std::vector<double> out;
  for (int k = 0; k < rt.nlev(); ++k) {
    const std::vector<double> lvl = rt.block_level_state(0, k);
    out.insert(out.end(), lvl.begin(), lvl.end());
  }
  return out;
}

#if defined(POPS_HAS_KOKKOS)
// Every TEST in this binary builds an AmrRuntime (Kokkos-dependent), so Kokkos is initialized once
// for the whole process via a GoogleTest global environment (ScopeGuard aborts if re-constructed
// after finalize, so it cannot live inside each TEST) -- the test_config_model_validation idiom.
class KokkosEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { guard_.emplace(); }
  void TearDown() override { guard_.reset(); }

 private:
  std::optional<Kokkos::ScopeGuard> guard_;
};

::testing::Environment* const kKokkosEnv =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
#endif

TEST(test_amr_history_ring, RegisterStoreReadRotate) {
  AmrRuntime rt = make_two_block(32, 1.0, 1.0);
  ASSERT_EQ(rt.nlev(), 2);

  // register lag=1 -> depth 2, all levels allocated.
  detail::AmrHistoryOps::register_history(rt, 0, "R", 1);
  EXPECT_EQ(detail::AmrHistoryOps::depth(rt, "R"), 2);
  EXPECT_FALSE(detail::AmrHistoryOps::initialized(rt, "R"));

  // Store block 0's per-level state into slot 0 of every level (what the AMR per-level loop does).
  const std::vector<double> s0 = block0_all_levels(rt);
  for (int k = 0; k < rt.nlev(); ++k)
    detail::AmrHistoryOps::store_history(rt, "R", k, rt.level_state(0, k), Real(0.01));
  EXPECT_TRUE(detail::AmrHistoryOps::initialized(rt, "R"));

  // slot 0 == the stored state; slot 1 == the SAME (per-level cold-start fill on the first store).
  EXPECT_EQ(dmax(detail::AmrHistoryOps::global(rt, "R", 0, false), s0), 0.0)
      << "slot0_equals_stored";
  EXPECT_EQ(dmax(detail::AmrHistoryOps::global(rt, "R", 1, false), s0), 0.0)
      << "cold_start_fills_deeper_slot";

  // Rotate: slot 1 <- slot 0 (the just-stored value); slot 0 recycled. Store a MUTATED value to slot 0
  // and check prev(lag=1) reads the pre-rotate value.
  detail::AmrHistoryOps::rotate_histories(rt);
  // advance the live state so the next store differs.
  rt.step(Real(0.01));
  const std::vector<double> s1 = block0_all_levels(rt);
  for (int k = 0; k < rt.nlev(); ++k)
    detail::AmrHistoryOps::store_history(rt, "R", k, rt.level_state(0, k), Real(0.01));
  // read lag 1 (level by level) == the FIRST stored state s0; lag 0 == the new state s1.
  EXPECT_EQ(dmax(detail::AmrHistoryOps::global(rt, "R", 1, false), s0), 0.0) << "prev_reads_older";
  EXPECT_EQ(dmax(detail::AmrHistoryOps::global(rt, "R", 0, false), s1), 0.0) << "cur_reads_newest";
  EXPECT_GT(dmax(s0, s1), 0.0) << "the_step_actually_changed_the_state";
}

TEST(test_amr_history_ring, CheckpointRoundTrip) {
  AmrRuntime rt = make_two_block(32, 1.0, 1.0);
  detail::AmrHistoryOps::register_history(rt, 0, "R", 2);  // depth 3
  for (int k = 0; k < rt.nlev(); ++k)
    detail::AmrHistoryOps::store_history(rt, "R", k, rt.level_state(0, k), Real(0.02));
  detail::AmrHistoryOps::restore_slot_dt(rt, "R", 1, 0.03);

  // Gather slot 1, wipe it into a fresh ring on another engine, and read it back identical.
  const std::vector<double> flat = detail::AmrHistoryOps::global(rt, "R", 1, false);
  AmrRuntime rt2 = make_two_block(32, 1.0, 1.0);
  // Restore cannot invent an owner for an unknown local name: the installed Program/layout owns
  // that qualified association, so recreate it explicitly before scattering checkpoint bytes.
  detail::AmrHistoryOps::register_history(rt2, 0, "R", 2);
  detail::AmrHistoryOps::restore(rt2, "R", 1, flat);
  detail::AmrHistoryOps::restore_slot_dt(rt2, "R", 1, 0.03);
  detail::AmrHistoryOps::set_initialized(rt2, "R", true);
  EXPECT_EQ(dmax(detail::AmrHistoryOps::global(rt2, "R", 1, false), flat), 0.0)
      << "flat_round_trip";
  EXPECT_EQ(detail::AmrHistoryOps::slot_dt(rt2, "R", 1), 0.03) << "slot_dt_round_trip";
}

TEST(test_amr_history_ring, NullRemapIsBitIdentical) {
  // The R1-risk lock: remapping the rings onto the SAME (fb, dmap) (what a layout-identical regrid
  // does in R6/R7b) is IDENTITY on the slots' valid cells -- the prolong writes first, then the
  // old-fine carry-over overwrites every covered cell with the original data.
  AmrRuntime rt = make_two_block(32, 1.0, 1.0);
  detail::AmrHistoryOps::register_history(rt, 0, "R", 1);
  for (int k = 0; k < rt.nlev(); ++k)
    detail::AmrHistoryOps::store_history(rt, "R", k, rt.level_state(0, k), Real(0.01));
  const std::vector<double> before0 = detail::AmrHistoryOps::global(rt, "R", 0, false);
  const std::vector<double> before1 = detail::AmrHistoryOps::global(rt, "R", 1, false);
  const MultiFab& fineU = rt.levels(0)[1].U;
  detail::AmrHistoryOps::remap_rings(rt, fineU.box_array(), fineU.dmap(), /*fk=*/1, /*pk=*/0,
                                     /*prolong=*/true);
  EXPECT_EQ(dmax(detail::AmrHistoryOps::global(rt, "R", 0, false), before0), 0.0)
      << "null_remap_slot0_bit_identical";
  EXPECT_EQ(dmax(detail::AmrHistoryOps::global(rt, "R", 1, false), before1), 0.0)
      << "null_remap_slot1_bit_identical";
}

TEST(test_amr_history_ring, RegridRemapKeepsSlotsConsistent) {
  AmrRuntime rt = make_two_block(32, 1.0, 1.0);
  detail::AmrHistoryOps::register_history(rt, 0, "R", 1);
  for (int k = 0; k < rt.nlev(); ++k)
    detail::AmrHistoryOps::store_history(rt, "R", k, rt.level_state(0, k), Real(0.01));
  // The coarse slot (level 0) is stable across a regrid -- snapshot it to prove it is untouched.
  const std::vector<double> coarse_before = detail::AmrHistoryOps::global(rt, "R", 0, false);
  const std::size_t nfine = static_cast<std::size_t>(rt.block_level_state(0, 1).size());

  // Activate a real regrid and fire it (a moving density front -> the fine layout changes).
  rt.set_regrid(/*every=*/1, /*grow=*/2, /*margin=*/2);
  rt.set_block_tag_predicate(0, TagDensityAbove{Real(1.2)});
  rt.set_block_tag_predicate(1, TagDensityAbove{Real(1.2)});
  rt.step(Real(0.01));  // macro_step 0: no regrid (fresh grid), but stores nothing to the ring
  rt.step(Real(0.01));  // macro_step 1 (every=1): regrid fires -> remap_rings runs
  ASSERT_GE(rt.regrid_count(), 1);

  // The ring's fine slot is defined on the NEW layout (finite, same global fine extent as U); the
  // coarse slot is untouched (the coarse layout is stable).
  const std::vector<double> global0 = detail::AmrHistoryOps::global(rt, "R", 0, false);
  EXPECT_TRUE(all_finite(global0)) << "ring_slots_finite_after_regrid";
  EXPECT_EQ(global0.size(), coarse_before.size()) << "flat_size_stable";
  // The coarse component of slot 0 is unchanged by the remap (only fine levels are rebuilt). Compare
  // the coarse prefix (block ncomp * n * n doubles) byte-for-byte.
  const std::size_t ncoarse = static_cast<std::size_t>(rt.block_level_state(0, 0).size());
  bool coarse_identical = true;
  for (std::size_t i = 0; i < ncoarse; ++i)
    if (global0[i] != coarse_before[i])
      coarse_identical = false;
  EXPECT_TRUE(coarse_identical) << "coarse_ring_slot_untouched_by_regrid";
  // The fine slice is the new fine extent (n<<1 squared * ncomp) and finite.
  EXPECT_EQ(global0.size() - ncoarse, nfine) << "fine_slice_matches_fine_extent";
}

TEST(test_amr_history_ring, ProgramContextRejectsNonRatioTwoProviderBeforeStep) {
  AmrRuntime rt = make_two_block(24, 1.0, 1.0, /*manifest_ratio=*/3);
  ASSERT_EQ(rt.nlev(), 2);
  ASSERT_EQ(rt.macro_step(), 0);
  try {
    runtime::program::AmrProgramContext context(&rt, nullptr);
    (void)context;
    FAIL() << "the ratio-2 Program reflux provider accepted a ratio-3 transition";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("supports only refinement ratio 2"), std::string::npos);
    EXPECT_NE(message.find("transition 0->1 resolved ratio 3"), std::string::npos);
  }
  EXPECT_EQ(rt.macro_step(), 0) << "provider validation must fail before the first native step";
}

TEST(test_amr_history_ring, LogicalSubcyclesPartitionEveryLevelWindowAndRestoreItExactly) {
  constexpr int n = 8;
  constexpr double macro_dt = 0.4;
  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 0;
  AmrSystem sim(cfg);
  AmrRuntime* rt = configure_native_ab2_regrid_system(sim, n, /*temporal_ratio=*/2);
  ASSERT_EQ(rt->nlev(), 2);
  runtime::program::AmrProgramContext context(rt, &sim);
  context.configure_primary_clock("clock.macro");
  context.declare_clock_relation("clock.macro", "clock.fast", 2);

  struct ObservedSnapshot {
    int level = -1;
    OperatorEvaluationSnapshot snapshot;
  };
  std::vector<ObservedSnapshot> children;
  std::vector<ObservedSnapshot> nested_children;
  std::vector<ObservedSnapshot> stale_parent_entry_probes;
  std::vector<ObservedSnapshot> stale_outer_probes;
  std::vector<ObservedSnapshot> reminted_outers;
  std::vector<ObservedSnapshot> parents_before;
  std::vector<ObservedSnapshot> stale_parent_probes;
  std::vector<ObservedSnapshot> parents_after;
  const OperatorFingerprint authority{UINT64_C(11), UINT64_C(12), UINT64_C(13), UINT64_C(14)};
  const OperatorFingerprint resources{UINT64_C(21), UINT64_C(22), UINT64_C(23), UINT64_C(24)};

  context.install([&](double dt) {
    context.advance_hierarchy(dt, [&](double) {
      const auto take_snapshot = [&]() {
        return context.operator_evaluation_snapshot(
            authority, context.state(0), resources);
      };
      context.set_stage_time(1, 3);
      parents_before.push_back({context.level(), take_snapshot()});
      auto ticks = context.subcycle_scope("clock.macro", "clock.fast", 2);
      for (int iteration = 0; iteration < 2; ++iteration) {
        ticks.iteration(iteration);
        auto child = context.logical_evaluation_scope(iteration, 2);
        if (iteration == 0) {
          const OperatorEvaluationSnapshot parent = parents_before.back().snapshot;
          stale_parent_entry_probes.push_back(
              {context.level(), context.probe_operator_evaluation(
                                    authority, parent.topology, resources, parent.revision)});
        }
        context.set_stage_time(1, 2);
        children.push_back({context.level(), take_snapshot()});
        if (iteration == 0) {
          const OperatorEvaluationSnapshot outer = children.back().snapshot;
          {
            auto nested_child = context.logical_evaluation_scope(0, 2);
            context.set_stage_time(1, 2);
            nested_children.push_back({context.level(), take_snapshot()});
          }
          stale_outer_probes.push_back(
              {context.level(), context.probe_operator_evaluation(
                                    authority, outer.topology, resources, outer.revision)});
          reminted_outers.push_back({context.level(), take_snapshot()});
          const OperatorEvaluationSnapshot& reminted = reminted_outers.back().snapshot;
          EXPECT_TRUE(context.probe_operator_evaluation(
                          authority, reminted.topology, resources, reminted.revision) == reminted);
        }
      }
      ticks.finish();
      const OperatorEvaluationSnapshot parent = parents_before.back().snapshot;
      stale_parent_probes.push_back(
          {context.level(), context.probe_operator_evaluation(
                                authority, parent.topology, resources, parent.revision)});
      parents_after.push_back({context.level(), take_snapshot()});
    });
  });
  const double initial_time = sim.time();
  sim.step(macro_dt);

  ASSERT_EQ(children.size(), 6u);
  ASSERT_EQ(nested_children.size(), 3u);
  ASSERT_EQ(stale_parent_entry_probes.size(), 3u);
  ASSERT_EQ(stale_outer_probes.size(), 3u);
  ASSERT_EQ(reminted_outers.size(), 3u);
  ASSERT_EQ(parents_before.size(), 3u);
  ASSERT_EQ(stale_parent_probes.size(), 3u);
  ASSERT_EQ(parents_after.size(), 3u);
  const std::array<int, 6> expected_levels{0, 0, 1, 1, 1, 1};
  const std::array<amr::Rational, 6> expected_phases{
      amr::Rational(1, 4), amr::Rational(3, 4), amr::Rational(1, 8),
      amr::Rational(3, 8), amr::Rational(5, 8), amr::Rational(7, 8)};
  const std::array<double, 6> expected_dt{
      macro_dt / 2.0, macro_dt / 2.0, macro_dt / 4.0,
      macro_dt / 4.0, macro_dt / 4.0, macro_dt / 4.0};
  const double coarse_child_dt = macro_dt / 2.0;
  const double fine_level_dt = macro_dt / 2.0;
  const double fine_child_dt = fine_level_dt / 2.0;
  const std::array<double, 6> expected_time{
      initial_time + 0.0 * coarse_child_dt + 0.5 * coarse_child_dt,
      initial_time + 1.0 * coarse_child_dt + 0.5 * coarse_child_dt,
      initial_time + 0.0 * fine_child_dt + 0.5 * fine_child_dt,
      initial_time + 1.0 * fine_child_dt + 0.5 * fine_child_dt,
      initial_time + fine_level_dt + 0.0 * fine_child_dt + 0.5 * fine_child_dt,
      initial_time + fine_level_dt + 1.0 * fine_child_dt + 0.5 * fine_child_dt};
  for (std::size_t index = 0; index < children.size(); ++index) {
    const auto& observed = children[index];
    EXPECT_EQ(observed.level, expected_levels[index]);
    EXPECT_EQ(observed.snapshot.stage_numerator, expected_phases[index].numerator);
    EXPECT_EQ(observed.snapshot.stage_denominator, expected_phases[index].denominator);
    EXPECT_EQ(std::bit_cast<double>(observed.snapshot.dt_bits), expected_dt[index]);
    EXPECT_EQ(std::bit_cast<double>(observed.snapshot.physical_time_bits), expected_time[index]);
    if (index > 0)
      EXPECT_NE(observed.snapshot.revision, children[index - 1].snapshot.revision);
  }
  for (std::size_t index = 0; index < parents_before.size(); ++index) {
    const std::size_t outer_index = index * 2;
    const OperatorEvaluationSnapshot& outer = children[outer_index].snapshot;
    const OperatorEvaluationSnapshot& nested = nested_children[index].snapshot;
    const OperatorEvaluationSnapshot& stale_outer = stale_outer_probes[index].snapshot;
    const OperatorEvaluationSnapshot& reminted_outer = reminted_outers[index].snapshot;
    EXPECT_EQ(stale_parent_entry_probes[index].level, parents_before[index].level);
    EXPECT_NE(stale_parent_entry_probes[index].snapshot.revision,
              parents_before[index].snapshot.revision);
    EXPECT_EQ(nested_children[index].level, children[outer_index].level);
    EXPECT_NE(nested.revision, outer.revision);
    EXPECT_EQ(stale_outer_probes[index].level, children[outer_index].level);
    EXPECT_NE(stale_outer.revision, outer.revision);
    EXPECT_EQ(stale_outer.stage_numerator, outer.stage_numerator);
    EXPECT_EQ(stale_outer.stage_denominator, outer.stage_denominator);
    EXPECT_EQ(stale_outer.dt_bits, outer.dt_bits);
    EXPECT_EQ(stale_outer.physical_time_bits, outer.physical_time_bits);
    EXPECT_NE(reminted_outer.revision, outer.revision);
    EXPECT_EQ(reminted_outer.stage_numerator, outer.stage_numerator);
    EXPECT_EQ(reminted_outer.stage_denominator, outer.stage_denominator);
    EXPECT_EQ(reminted_outer.dt_bits, outer.dt_bits);
    EXPECT_EQ(reminted_outer.physical_time_bits, outer.physical_time_bits);

    EXPECT_EQ(parents_after[index].level, parents_before[index].level);
    EXPECT_EQ(stale_parent_probes[index].level, parents_before[index].level);
    EXPECT_NE(stale_parent_probes[index].snapshot.revision,
              parents_before[index].snapshot.revision);
    EXPECT_EQ(stale_parent_probes[index].snapshot.stage_numerator,
              parents_before[index].snapshot.stage_numerator);
    EXPECT_EQ(stale_parent_probes[index].snapshot.stage_denominator,
              parents_before[index].snapshot.stage_denominator);
    EXPECT_EQ(stale_parent_probes[index].snapshot.dt_bits,
              parents_before[index].snapshot.dt_bits);
    EXPECT_EQ(stale_parent_probes[index].snapshot.physical_time_bits,
              parents_before[index].snapshot.physical_time_bits);
    EXPECT_EQ(parents_after[index].snapshot.stage_numerator,
              parents_before[index].snapshot.stage_numerator);
    EXPECT_EQ(parents_after[index].snapshot.stage_denominator,
              parents_before[index].snapshot.stage_denominator);
    EXPECT_EQ(parents_after[index].snapshot.dt_bits,
              parents_before[index].snapshot.dt_bits);
    EXPECT_EQ(parents_after[index].snapshot.physical_time_bits,
              parents_before[index].snapshot.physical_time_bits);
    EXPECT_NE(parents_after[index].snapshot.revision,
              parents_before[index].snapshot.revision);
  }
}

TEST(test_amr_history_ring, Ab2RegridRebindsLaggedResidualAndFluxOnTheNewTopology) {
  // Regression for the real ADC-631 x ADC-639 failure: R_(n-1) was remapped to the new fine boxes,
  // while its compact interface flux still described the old boxes.  The state update and reflux
  // therefore disagreed only on regrid steps and leaked coarse mass.  Drive the native Program
  // context directly (no Python/.so) through two history-populating steps and one real regrid.
  constexpr int n = 16;
  constexpr double dt = 2.0e-3;
  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 2;

  AmrSystem sim(cfg);
  sim.set_temporal_relations({1}, {1}, {"integral_only"});
  sim.add_block("a", exb_charge(+1.0, 1.0), "minmod", "rusanov", "conservative", "explicit", 1);
  sim.add_block("b", exb_charge(-1.0, 1.0), "minmod", "rusanov", "conservative", "explicit", 1);
  sim.set_poisson("charge_density", "geometric_mg", "periodic");
  sim.set_refinement(1.2);
  sim.set_density("a", blob(n, 0.35, 0.5, 0.5, 1.0, 0.12));
  sim.set_density("b", blob(n, 0.65, 0.5, 0.5, 1.0, 0.12));
  sim.set_program_block_map({0, 1});
  ASSERT_TRUE(sim.uses_runtime_engine());
  AmrRuntime* rt = sim.engine();
  ASSERT_NE(rt, nullptr);
  rt->set_block_tag_predicate(0, TagDensityAbove{Real(1.2)});
  rt->set_block_tag_predicate(1, TagDensityAbove{Real(1.2)});
  const std::vector<PatchBox> initial_patches = rt->patch_boxes();
  const double initial_mass = rt->composite_reduce("a", "sum", 0, {0});

  runtime::program::AmrProgramContext context(rt, &sim);
  context.configure_primary_clock("clock.macro");
  context.register_history("a.rate", 1, -1, 0, "block.a.U", "cell.conservative", "clock.macro",
                           "dense.linear");
  context.register_history("a.carry", 1, -1, 0, "block.a.U", "cell.conservative", "clock.macro",
                           "dense.linear");
  bool nonflux_carry_kept_old_fine_overlap = false;
  std::vector<double> lagged_rate_before_regrid;
  double lagged_rate_spread_after_regrid = -1.0;
  context.install([&context, &initial_patches, &lagged_rate_before_regrid,
                   &lagged_rate_spread_after_regrid, &nonflux_carry_kept_old_fine_overlap, rt,
                   n](double macro_dt) {
    context.advance_hierarchy(
        macro_dt,
        [&context, &initial_patches, &lagged_rate_before_regrid, &lagged_rate_spread_after_regrid,
         &nonflux_carry_kept_old_fine_overlap, rt, n](double level_dt) {
          context.set_stage_time(0, 1);
          (void)context.solve_fields();
          MultiFab& state = context.state(0);
          if (context.level() == 1 && context.history_flux_topology_rebind_count() == 1) {
            const std::vector<double> carry =
                pops::detail::AmrHistoryOps::global(*rt, "a.carry", 1, false);
            const std::vector<double> rate =
                pops::detail::AmrHistoryOps::global(*rt, "a.rate", 1, false);
            if (!lagged_rate_before_regrid.empty())
              lagged_rate_spread_after_regrid =
                  max_old_fine_child_group_spread(rate, n, initial_patches);
            const std::size_t coarse_size = rt->block_level_state(0, 0).size();
            for (std::size_t i = coarse_size; i < carry.size(); ++i)
              if (carry[i] == 11.0)
                nonflux_carry_kept_old_fine_overlap = true;
          }
          MultiFab carry = context.scratch_state_like(state);
          carry.set_val(Real(10 + context.level()));
          context.store_history("a.carry", carry, 0);
          MultiFab rate = context.rhs_scratch_like(state);
          context.rhs_into(0, state, rate, 17);
          context.store_history("a.rate", rate, 0);
          MultiFab& previous = context.history("a.rate", 1, 0);

          MultiFab next = context.scratch_state_like(state);
          context.lincomb(next, Real(1), state, Real(0), state);
          context.axpy(next, Real(1.5 * level_dt), rate, Real(level_dt), {{1, 3, 2}});
          context.axpy(next, Real(-0.5 * level_dt), previous, Real(level_dt), {{1, -1, 2}});
          context.lincomb(state, Real(0), state, Real(1), next);
          context.rotate_histories("clock.macro");
        });
  });

  sim.step(dt);
  sim.step(dt);
  EXPECT_EQ(context.history_flux_topology_rebind_count(), 0);
  lagged_rate_before_regrid = pops::detail::AmrHistoryOps::global(*rt, "a.rate", 1, false);
  const double lagged_rate_spread_before_regrid =
      max_old_fine_child_group_spread(lagged_rate_before_regrid, n, initial_patches);
  ASSERT_GT(lagged_rate_spread_before_regrid, 1.0e-12);
  sim.step(dt);  // macro_step 2: the tagged layout replaces the bootstrap fine boxes

  EXPECT_FALSE(same_patches(rt->patch_boxes(), initial_patches));
  EXPECT_EQ(rt->regrid_count(), 1);
  EXPECT_EQ(context.history_flux_topology_rebind_count(), 1);
  EXPECT_EQ(context.history_flux_topology_epoch(), rt->topology_epoch());
  EXPECT_TRUE(nonflux_carry_kept_old_fine_overlap)
      << "a history with no conservative flux authority must keep the normal old-fine overlap";
  EXPECT_NEAR(lagged_rate_spread_after_regrid, lagged_rate_spread_before_regrid, 1.0e-12)
      << "the conservative parent-average correction must retain old-fine subcell detail";
  EXPECT_LT(std::fabs(rt->composite_reduce("a", "sum", 0, {0}) - initial_mass), 1.0e-10)
      << "the remapped lagged residual and zero-mismatch flux authority must conserve AB2 mass";
}

TEST(test_amr_history_ring, RejectedAb2RegridRestoresTopologyHistoryFluxAndStateExactly) {
  constexpr int n = 16;
  constexpr double dt = 2.0e-3;
  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 2;
  AmrSystem sim(cfg);
  AmrRuntime* rt = configure_native_ab2_regrid_system(sim, n);
  runtime::program::AmrProgramContext context(rt, &sim);

  bool reject_after_rebind = false;
  bool saw_changed_topology = false;
  install_native_ab2_program(context, [&] {
    if (reject_after_rebind && context.level() == 1 &&
        context.history_flux_topology_rebind_count() == 1) {
      saw_changed_topology = true;
      throw runtime::program::StepAttemptRejected(SolveStatus::kIterationLimit, "solve",
                                                  "fault after AB2 history/flux topology rebind");
    }
  });

  const double initial_mass = rt->composite_reduce("a", "sum", 0, {0});
  sim.step(dt);
  sim.step(dt);
  ASSERT_TRUE(context.history_flux_topology_bound());
  ASSERT_EQ(context.history_flux_topology_rebind_count(), 0);

  const double time_before = sim.time();
  const int macro_before = sim.macro_step();
  const int regrids_before = rt->regrid_count();
  const std::uint64_t topology_before = rt->topology_epoch();
  const std::uint64_t history_topology_before = context.history_flux_topology_epoch();
  const int history_rebinds_before = context.history_flux_topology_rebind_count();
  const std::vector<PatchBox> patches_before = rt->patch_boxes();
  const std::vector<int> owners_before = rt->level_owner_ranks(1);
  const std::vector<double> a0_before = rt->block_level_state(0, 0);
  const std::vector<double> a1_before = rt->block_level_state(0, 1);
  const std::vector<double> b0_before = rt->block_level_state(1, 0);
  const std::vector<double> b1_before = rt->block_level_state(1, 1);
  const std::vector<double> ring0_before =
      pops::detail::AmrHistoryOps::global(*rt, "a.rate", 0, false);
  const std::vector<double> ring1_before =
      pops::detail::AmrHistoryOps::global(*rt, "a.rate", 1, false);
  // The accepted-state image contains the persistent exact contribution ledger and compact strips;
  // byte equality therefore checks the flux authority in addition to the engine-owned data ring.
  const std::vector<std::uint8_t> accepted_flux_before = sim.program_accepted_state();

  reject_after_rebind = true;
  EXPECT_THROW(sim.step(dt), runtime::program::StepAttemptRejected);
  EXPECT_TRUE(saw_changed_topology) << "the fault must occur after the real regrid/rebind";
  EXPECT_DOUBLE_EQ(sim.time(), time_before);
  EXPECT_EQ(sim.macro_step(), macro_before);
  EXPECT_EQ(rt->regrid_count(), regrids_before);
  EXPECT_EQ(rt->topology_epoch(), topology_before);
  EXPECT_TRUE(same_patches(rt->patch_boxes(), patches_before));
  EXPECT_EQ(rt->level_owner_ranks(1), owners_before);
  EXPECT_EQ(context.history_flux_topology_epoch(), history_topology_before);
  EXPECT_EQ(context.history_flux_topology_rebind_count(), history_rebinds_before);
  EXPECT_EQ(rt->block_level_state(0, 0), a0_before);
  EXPECT_EQ(rt->block_level_state(0, 1), a1_before);
  EXPECT_EQ(rt->block_level_state(1, 0), b0_before);
  EXPECT_EQ(rt->block_level_state(1, 1), b1_before);
  EXPECT_EQ(pops::detail::AmrHistoryOps::global(*rt, "a.rate", 0, false), ring0_before);
  EXPECT_EQ(pops::detail::AmrHistoryOps::global(*rt, "a.rate", 1, false), ring1_before);
  EXPECT_EQ(sim.program_accepted_state(), accepted_flux_before);

  reject_after_rebind = false;
  sim.step(dt);  // exact retry of the rejected regrid attempt
  EXPECT_FALSE(same_patches(rt->patch_boxes(), patches_before));
  EXPECT_EQ(context.history_flux_topology_rebind_count(), 1);
  EXPECT_LT(std::fabs(rt->composite_reduce("a", "sum", 0, {0}) - initial_mass), 1.0e-10);
}

TEST(test_amr_history_ring, AcceptedStateRestartReconstructsReboundFluxAuthorityForNextAb2Step) {
  constexpr int n = 16;
  constexpr double dt = 2.0e-3;
  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 2;
  AmrSystem sim(cfg);
  AmrRuntime* rt = configure_native_ab2_regrid_system(sim, n);
  runtime::program::AmrProgramContext original(rt, &sim);
  install_native_ab2_program(original);

  sim.step(dt);
  sim.step(dt);
  sim.step(dt);  // real regrid + parent-average history correction
  ASSERT_EQ(original.history_flux_topology_rebind_count(), 1);
  ASSERT_EQ(original.history_flux_topology_epoch(), rt->topology_epoch());
  const double checkpoint_mass = rt->composite_reduce("a", "sum", 0, {0});
  const int checkpoint_regrids = rt->regrid_count();
  const std::vector<std::uint8_t> checkpoint = sim.program_accepted_state();
  ASSERT_FALSE(checkpoint.empty());
  const auto accepted = runtime::program::deserialize_amr_program_accepted_state(checkpoint);
  const auto contributions = accepted.ring_flux_contributions.find("a.rate");
  ASSERT_NE(contributions, accepted.ring_flux_contributions.end());
  bool has_exact_lagged_flux_contribution = false;
  for (const auto& slot : contributions->second)
    for (const auto& level : slot)
      has_exact_lagged_flux_contribution = has_exact_lagged_flux_contribution || !level.empty();
  ASSERT_TRUE(has_exact_lagged_flux_contribution);

  // Reinstall a fresh context over the checkpointed engine/facade. Its first attempt imports the
  // accepted Program image, reconstructs the topology binding from the restored engine and consumes
  // the retained lagged contribution on the next AB2 step.
  sim.restore_program_accepted_state(checkpoint);
  runtime::program::AmrProgramContext restored(rt, &sim);
  ASSERT_FALSE(restored.history_flux_topology_bound());
  install_native_ab2_program(restored);
  ASSERT_FALSE(restored.history_flux_topology_bound());
  sim.step(dt);

  EXPECT_TRUE(restored.history_flux_topology_bound());
  EXPECT_EQ(restored.history_flux_topology_epoch(), rt->topology_epoch());
  EXPECT_EQ(restored.history_flux_topology_rebind_count(), 0)
      << "restart must bind the accepted topology without fabricating a new regrid";
  EXPECT_EQ(rt->regrid_count(), checkpoint_regrids);
  EXPECT_LT(std::fabs(rt->composite_reduce("a", "sum", 0, {0}) - checkpoint_mass), 1.0e-10)
      << "the first post-restart AB2 step must consume the restored lagged flux conservatively";
}

TEST(test_amr_history_ring, AcceptedFacadeTransactionCommitsTopologyStateHistoryAndClock) {
  constexpr int n = 32;
  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 1;

  AmrSystem sim(cfg);
  sim.set_temporal_relations({2}, {1}, {"integral_only"});
  sim.add_block("a", exb_charge(+1.0, 1.0), "minmod", "rusanov", "conservative", "explicit", 1);
  sim.add_block("b", exb_charge(-1.0, 1.0), "minmod", "rusanov", "conservative", "explicit", 1);
  sim.set_poisson("charge_density", "geometric_mg", "periodic");
  sim.set_refinement(1.2);
  sim.set_density("a", blob(n, 0.25, 0.5, 1.0, 1.0, 0.06));
  sim.set_density("b", blob(n, 0.75, 0.5, 1.0, 1.0, 0.06));
  ASSERT_TRUE(sim.uses_runtime_engine());
  AmrRuntime* rt = sim.engine();
  ASSERT_NE(rt, nullptr);
  rt->set_block_tag_predicate(0, TagDensityAbove{Real(1.2)});
  rt->set_block_tag_predicate(1, TagDensityAbove{Real(1.2)});
  detail::AmrHistoryOps::register_history(*rt, 0, "R", 1);
  sim.set_clock(0.25, 1);  // the accepted attempt performs a real due regrid

  const std::vector<PatchBox> patches_before = rt->patch_boxes();
  const std::vector<double> state_before = rt->block_level_state(0, 0);
  const int regrids_before = rt->regrid_count();

  sim.install_program_step([&](double dt) {
    rt->step(static_cast<Real>(dt));
    for (int k = 0; k < rt->nlev(); ++k)
      detail::AmrHistoryOps::store_history(*rt, "R", k, rt->level_state(0, k), Real(dt));
    detail::AmrHistoryOps::rotate_histories(*rt);
    sim.record_program_diagnostic("accepted", 7.0);
  });

  sim.begin_step_transaction();
  sim.step(0.01);
  ASSERT_FALSE(same_patches(rt->patch_boxes(), patches_before));
  ASSERT_GT(rt->regrid_count(), regrids_before);
  sim.commit_step_transaction();

  EXPECT_DOUBLE_EQ(sim.time(), 0.26);
  EXPECT_EQ(sim.macro_step(), 2);
  EXPECT_EQ(rt->macro_step(), 2);
  EXPECT_NE(rt->block_level_state(0, 0), state_before);
  EXPECT_TRUE(detail::AmrHistoryOps::initialized(*rt, "R"));
  ASSERT_EQ(sim.program_diagnostics().count("accepted"), 1u);
  EXPECT_DOUBLE_EQ(sim.program_diagnostics().at("accepted"), 7.0);
  // Commit makes the accepted state externally publishable but deliberately retains the rollback
  // snapshot. Finalize is the irreversible boundary after those publications succeed.
  sim.finalize_step_transaction();
  EXPECT_THROW(sim.rollback_step_transaction(), std::runtime_error);
}

TEST(test_amr_history_ring, RejectedFacadeAttemptRestoresTopologyStateHistoryAndClock) {
  constexpr int n = 32;
  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 1;

  AmrSystem sim(cfg);
  sim.set_temporal_relations({2}, {1}, {"integral_only"});
  sim.add_block("a", exb_charge(+1.0, 1.0), "minmod", "rusanov", "conservative", "explicit", 1);
  sim.add_block("b", exb_charge(-1.0, 1.0), "minmod", "rusanov", "conservative", "explicit", 1);
  sim.set_poisson("charge_density", "geometric_mg", "periodic");
  sim.set_refinement(1.2);
  sim.set_density("a", blob(n, 0.25, 0.5, 1.0, 1.0, 0.06));
  sim.set_density("b", blob(n, 0.75, 0.5, 1.0, 1.0, 0.06));
  ASSERT_TRUE(sim.uses_runtime_engine());
  AmrRuntime* rt = sim.engine();
  ASSERT_NE(rt, nullptr);
  rt->set_block_tag_predicate(0, TagDensityAbove{Real(1.2)});
  rt->set_block_tag_predicate(1, TagDensityAbove{Real(1.2)});
  detail::AmrHistoryOps::register_history(*rt, 0, "R", 1);
  sim.set_clock(0.25, 1);  // next native engine step is regrid-due

  const std::vector<PatchBox> patches_before = rt->patch_boxes();
  const std::vector<int> owners_before = rt->level_owner_ranks(1);
  const std::vector<double> state_a_before = rt->block_level_state(0, 0);
  const std::vector<double> state_b_before = rt->block_level_state(1, 1);
  const std::vector<double> aux_before = rt->level_aux_flat(1);
  const int solves_before = rt->solve_count();
  const int regrids_before = rt->regrid_count();
  bool topology_changed_during_attempt = false;

  sim.install_program_step([&](double dt) {
    rt->step(static_cast<Real>(dt));  // includes due regrid + multi-block advance
    topology_changed_during_attempt = !same_patches(rt->patch_boxes(), patches_before);
    for (int k = 0; k < rt->nlev(); ++k)
      detail::AmrHistoryOps::store_history(*rt, "R", k, rt->level_state(0, k), Real(dt));
    detail::AmrHistoryOps::rotate_histories(*rt);
    sim.record_program_diagnostic("provisional", 42.0);
    throw runtime::program::StepAttemptRejected(
        SolveStatus::kIterationLimit, "solve",
        "AMR fault after regrid and provisional publications");
  });

  EXPECT_THROW(sim.step(0.01), runtime::program::StepAttemptRejected);
  EXPECT_TRUE(topology_changed_during_attempt) << "fault must happen after a real topology change";
  EXPECT_DOUBLE_EQ(sim.time(), 0.25);
  EXPECT_EQ(sim.macro_step(), 1);
  EXPECT_EQ(rt->macro_step(), 1);
  EXPECT_TRUE(same_patches(rt->patch_boxes(), patches_before));
  EXPECT_EQ(rt->level_owner_ranks(1), owners_before);
  EXPECT_EQ(rt->block_level_state(0, 0), state_a_before);
  EXPECT_EQ(rt->block_level_state(1, 1), state_b_before);
  EXPECT_EQ(rt->level_aux_flat(1), aux_before);
  EXPECT_EQ(rt->solve_count(), solves_before);
  EXPECT_EQ(rt->regrid_count(), regrids_before);
  EXPECT_FALSE(detail::AmrHistoryOps::initialized(*rt, "R"));
  EXPECT_TRUE(sim.program_diagnostics().empty());

  // The CFL entry point brackets its preliminary field solve and active-bound publication too.
  topology_changed_during_attempt = false;
  EXPECT_THROW(sim.step_cfl(0.4), runtime::program::StepAttemptRejected);
  EXPECT_TRUE(topology_changed_during_attempt);
  EXPECT_DOUBLE_EQ(sim.time(), 0.25);
  EXPECT_EQ(sim.macro_step(), 1);
  EXPECT_EQ(rt->macro_step(), 1);
  EXPECT_TRUE(same_patches(rt->patch_boxes(), patches_before));
  EXPECT_EQ(rt->level_owner_ranks(1), owners_before);
  EXPECT_EQ(rt->block_level_state(0, 0), state_a_before);
  EXPECT_EQ(rt->block_level_state(1, 1), state_b_before);
  EXPECT_EQ(rt->level_aux_flat(1), aux_before);
  EXPECT_EQ(rt->solve_count(), solves_before);
  EXPECT_EQ(rt->regrid_count(), regrids_before);
  EXPECT_FALSE(detail::AmrHistoryOps::initialized(*rt, "R"));
  EXPECT_TRUE(sim.program_diagnostics().empty());
}
