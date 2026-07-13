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
#include <pops/runtime/program/step_transaction.hpp>         // StepAttemptRejected fault signal
#include <pops/runtime/builders/factory/model_factory.hpp>   // detail::dispatch_model
#include <pops/runtime/config/model_spec.hpp>

#include <cmath>
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

static std::vector<double> blob(int n, double cx, double cy, double amp, double base, double width) {
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

static AmrRuntime make_two_block(int N, double L, double B0) {
  AmrBuildParams bp;
  bp.mesh.n = N;
  bp.mesh.L = L;
  bp.mesh.regrid_every = 0;
  bp.poisson.bc = BCRec{};  // periodic
  const detail::SharedAmrLayout S = detail::make_shared_amr_layout(bp);
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
  return AmrRuntime(S.geom, S.runtime_hierarchy(), S.poisson_bc, std::move(blocks), S.base_per,
                    S.replicated_coarse, S.wall);
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
  EXPECT_EQ(dmax(detail::AmrHistoryOps::global(rt2, "R", 1, false), flat), 0.0) << "flat_round_trip";
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

TEST(test_amr_history_ring, RejectedFacadeAttemptRestoresTopologyStateHistoryAndClock) {
  constexpr int n = 32;
  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 1;

  AmrSystem sim(cfg);
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
        SolveStatus::kIterationLimit, "solve", "AMR fault after regrid and provisional publications");
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
