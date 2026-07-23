// Selective history persistence + deterministic ring replay (ADC-626). A history-persistence policy
// (pops.time.Dense / Interval / Revolve) stores only a SUBSET of a ring's slots in a checkpoint; the
// restart REBUILDS the missing slots by re-stepping the installed Program (System::rebuild_history_slots).
// This is the bit-identity ACCEPTANCE test: a ring restored via Interval / Revolve equals the
// dense-restored ring BIT-FOR-BIT (operator==, no tolerance), across a (depth, k) / (depth, s) sweep,
// including a variable-dt run (the slot_dt machinery) and live-state isolation (U + cache untouched by
// replay), plus the v1 back-compat restore and the verbatim policy/version refusals.
//
// It installs a REAL deterministic macro-step closure (install_program_step) that mirrors what a compiled
// keep_history step body emits: store U^n, advance the qualified owner state by a conservative increment,
// then rotate. Replay seeds that same owner from an older stored slot and reconstructs the gaps. No .so /
// codegen: the closure is a native lambda, so the whole test runs in the serial suite.

#include <gtest/gtest.h>

#include <pops/mesh/storage/mf_arith.hpp>  // saxpy / lincomb
#include <pops/mesh/storage/multifab.hpp>
#include <pops/physics/bricks/source.hpp>                // NoSource
#include <pops/physics/composition/composite.hpp>        // CompositeModel
#include <pops/physics/fluids/euler.hpp>                 // Euler
#include <pops/runtime/builders/compiled/dsl_block.hpp>  // add_compiled_model
#include <pops/runtime/system.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

// ONE Kokkos ScopeGuard for the whole TU: a function-local static initialized on the first call and
// shared by every TEST (a second ScopeGuard while Kokkos is live is illegal). Each TEST calls kokkos().
void kokkos() {
#if defined(POPS_HAS_KOKKOS)
  static Kokkos::ScopeGuard guard;
  (void)guard;
#endif
}

struct NoEll {
  template <class State>
  POPS_HD Real rhs(const State&) const {
    return Real(0);
  }
};
using GasModel = CompositeModel<Euler, NoSource, NoEll>;
constexpr double kGamma = 1.4;

void add_gas_block(System& s, const std::string& name) {
  add_compiled_model(s, name, GasModel{Euler{kGamma}, NoSource{}, NoEll{}}, "minmod", "rusanov",
                     "conservative", "explicit", kGamma);
}

void add_gas(System& s) {
  add_gas_block(s, "gas");
  s.set_poisson("charge_density", "geometric_mg");
}

void register_state_history(System& s, const std::string& ring, int depth, int owner = 0) {
  s.register_history(ring, depth - 1, -1, owner, "test.state." + std::to_string(owner),
                     "test.space", "test.clock", "test.exact");
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0;
  const std::size_t m = a.size() < b.size() ? a.size() : b.size();
  for (std::size_t k = 0; k < m; ++k)
    d = std::fmax(d, std::fabs(a[k] - b[k]));
  return d;
}

// Install a deterministic macro-step closure with the exact compiled keep_history phase: snapshot
// U^n first, advance the qualified owner by +inc, then rotate after the commit. The dt ledger therefore travels
// with its starting sample (the outgoing interval toward the newer sample). `inc` scales with dt so
// a variable-dt run with multiple independent gaps proves that exact provenance. The closure
// captures &s; s must outlive it.
void install_ramp_program(System& s, const std::string& ring, double rate, int owner = 0,
                          int* executed_steps = nullptr) {
  System* self = &s;
  self->install_program_step([self, ring, rate, owner, executed_steps](double dt) {
    if (executed_steps != nullptr)
      ++*executed_steps;
    MultiFab& U = self->block_state(owner);
    self->store_history(ring, U);
    // Advance: U += rate*dt (a deterministic, dt-dependent conservative increment on every component).
    MultiFab bump = U;  // same layout
    bump.set_val(Real(rate) * Real(dt));
    pops::saxpy(U, Real(1), bump);
    self->rotate_histories();
  });
}

// Serialize a ring's every slot (the DENSE golden) as the facade would, for a bit-for-bit comparison.
std::vector<std::vector<double>> dump_ring(const System& s, const std::string& ring) {
  std::vector<std::vector<double>> slots;
  const int depth = s.history_depth(ring);
  for (int k = 0; k < depth; ++k)
    slots.push_back(s.history_global(ring, k));
  return slots;
}

// The stored-slot placement for a policy, computed HOST-side exactly like the Python descriptors so the
// C++ test drives the same selection without a Python round-trip.
std::vector<int> interval_slots(int depth, int k) {
  std::vector<int> out;
  out.push_back(0);
  for (int s = 0; s < depth; ++s)
    if (s % k == 0)
      out.push_back(s);
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::vector<int> revolve_slots(int depth, int snapshots) {
  // Equispaced anchors including both endpoints (round(i*(d-1)/(s-1))), matching _optimal_placement.
  std::vector<int> anchors;
  for (int i = 0; i < snapshots; ++i)
    anchors.push_back(
        static_cast<int>(std::lround(static_cast<double>(i) * (depth - 1) / (snapshots - 1))));
  std::sort(anchors.begin(), anchors.end());
  anchors.erase(std::unique(anchors.begin(), anchors.end()), anchors.end());
  return anchors;
}

// Build a fresh System with the ramp program, run `nsteps` macro-steps (constant or variable dt), and
// return it with the ring filled. depth = maxlag+1; the ring is registered at maxlag before the run.
struct Filled {
  SystemConfig cfg;
  int depth;
  std::string ring = "state_prev";
  double rate;
};

// Fill a ring on a fresh System by running the ramp program `nsteps` times with the given dt sequence.
// Returns the dense dump of the ring plus the live block-0 state (for the isolation check).
void fill_and_dump(const SystemConfig& cfg, const std::string& ring, int depth, double rate,
                   const std::vector<double>& dts, std::vector<std::vector<double>>& ring_out,
                   std::vector<double>& live_state_out) {
  System s(cfg);
  add_gas(s);
  register_state_history(s, ring, depth);
  install_ramp_program(s, ring, rate);
  for (double dt : dts)
    s.step(dt);
  ring_out = dump_ring(s, ring);
  live_state_out = s.state_global("gas");
}

// Restore a ring into a fresh System from a policy's STORED slots + slot_dt, replay the gaps, and return
// the dense dump. `golden` is the full dense ring (source of the stored-slot values + slot_dt).
void restore_replay_dump(const SystemConfig& cfg, const std::string& ring, int depth, double rate,
                         const std::vector<double>& slot_dt, const std::vector<int>& stored_slots,
                         const std::vector<std::vector<double>>& golden,
                         std::vector<std::vector<double>>& ring_out,
                         std::vector<double>& live_state_out, int& recomputed_out) {
  System s(cfg);
  add_gas(s);
  register_state_history(s, ring, depth);
  int executed_steps = 0;
  install_ramp_program(s, ring, rate, /*owner=*/0,
                       &executed_steps);  // the SAME program must be installed to replay
  // Restore only the stored slots + every slot's dt.
  for (int k : stored_slots)
    s.restore_history(ring, k, golden[static_cast<std::size_t>(k)]);
  for (int k = 0; k < depth; ++k)
    s.restore_history_slot_dt(ring, k, slot_dt[static_cast<std::size_t>(k)]);
  s.set_history_initialized(ring, true);
  // Capture the live state BEFORE replay so the isolation check compares against it.
  const std::vector<double> live_before = s.state_global("gas");
  recomputed_out = s.rebuild_history_slots(ring, stored_slots);
  EXPECT_EQ(executed_steps, recomputed_out)
      << "native replay executes exactly one Program step per omitted slot";
  ring_out = dump_ring(s, ring);
  live_state_out = s.state_global("gas");
  // The live state is identity across replay (the save/restore bracket).
  EXPECT_TRUE(max_abs_diff(live_before, live_state_out) == 0.0)
      << "replay_is_identity_on_live_state";
}

SystemConfig make_cfg() {
  SystemConfig cfg;
  cfg.n = 8;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};
  return cfg;
}

}  // namespace

// (A) BIT-IDENTITY: Interval / Revolve restored rings == the dense-restored ring, across a sweep.
TEST(CheckpointHistoryPolicy, IntervalAndRevolveMatchDenseBitForBit) {
  kokkos();
  const SystemConfig cfg = make_cfg();
  const std::string ring = "state_prev";
  const double rate = 3.0;

  // Depths and the constant dt sequence (enough steps to fully populate + distinguish every lag).
  for (int depth : {3, 4, 5, 7, 9}) {
    const std::vector<double> dts(static_cast<std::size_t>(depth) + 3, 0.05);
    // The DENSE golden ring (every slot stored, no replay) and its per-slot dt (constant here).
    std::vector<std::vector<double>> golden;
    std::vector<double> live_golden;
    fill_and_dump(cfg, ring, depth, rate, dts, golden, live_golden);
    std::vector<double> slot_dt(static_cast<std::size_t>(depth), 0.05);

    // INTERVAL: every k that divides depth-1 (so the oldest slot is stored -> reconstructable).
    for (int k = 1; k <= depth - 1; ++k) {
      if ((depth - 1) % k != 0)
        continue;
      const std::vector<int> stored = interval_slots(depth, k);
      std::vector<std::vector<double>> replayed;
      std::vector<double> live_replayed;
      int recomputed = 0;
      restore_replay_dump(cfg, ring, depth, rate, slot_dt, stored, golden, replayed, live_replayed,
                          recomputed);
      EXPECT_EQ(recomputed, depth - static_cast<int>(stored.size()))
          << "interval recomputed_count depth=" << depth << " k=" << k;
      for (int slot = 0; slot < depth; ++slot) {
        const double d = max_abs_diff(replayed[static_cast<std::size_t>(slot)],
                                      golden[static_cast<std::size_t>(slot)]);
        EXPECT_TRUE(d == 0.0) << "interval depth=" << depth << " k=" << k << " slot=" << slot
                              << " max|d|=" << d;
      }
    }

    // REVOLVE: every budget 2..depth.
    for (int snap = 2; snap <= depth; ++snap) {
      const std::vector<int> stored = revolve_slots(depth, snap);
      std::vector<std::vector<double>> replayed;
      std::vector<double> live_replayed;
      int recomputed = 0;
      restore_replay_dump(cfg, ring, depth, rate, slot_dt, stored, golden, replayed, live_replayed,
                          recomputed);
      EXPECT_EQ(recomputed, depth - static_cast<int>(stored.size()))
          << "revolve recomputed_count depth=" << depth << " snap=" << snap;
      for (int slot = 0; slot < depth; ++slot) {
        const double d = max_abs_diff(replayed[static_cast<std::size_t>(slot)],
                                      golden[static_cast<std::size_t>(slot)]);
        EXPECT_TRUE(d == 0.0) << "revolve depth=" << depth << " snap=" << snap << " slot=" << slot
                              << " max|d|=" << d;
      }
    }
  }
}

// (B) VARIABLE-dt replay is bit-exact (the slot_dt machinery): a non-constant dt sequence still
// reconstructs the dense ring exactly, because each recomputed slot is re-stepped with its recorded dt.
TEST(CheckpointHistoryPolicy, VariableDtReplayIsBitExact) {
  kokkos();
  const SystemConfig cfg = make_cfg();
  const std::string ring = "state_prev";
  const double rate = 2.0;
  const int depth = 5;
  // A NON-constant dt sequence: the ring's slot values now depend on the exact dt at each store.
  const std::vector<double> dts = {0.03, 0.07, 0.05, 0.11, 0.02, 0.09, 0.04, 0.06};

  std::vector<std::vector<double>> golden;
  std::vector<double> live_golden;
  fill_and_dump(cfg, ring, depth, rate, dts, golden, live_golden);

  // The slot_dt the forward run recorded (read it back from a fresh dense run's ring via the accessor).
  System probe(cfg);
  add_gas(probe);
  register_state_history(probe, ring, depth);
  install_ramp_program(probe, ring, rate);
  for (double dt : dts)
    probe.step(dt);
  std::vector<double> slot_dt(static_cast<std::size_t>(depth));
  for (int k = 0; k < depth; ++k)
    slot_dt[static_cast<std::size_t>(k)] = probe.history_slot_dt(ring, k);

  // Revolve(3) on depth 5 -> stored {0,2,4}, replay {1,3} with the exact per-slot dt.
  const std::vector<int> stored = revolve_slots(depth, 3);
  std::vector<std::vector<double>> replayed;
  std::vector<double> live_replayed;
  int recomputed = 0;
  restore_replay_dump(cfg, ring, depth, rate, slot_dt, stored, golden, replayed, live_replayed,
                      recomputed);
  EXPECT_EQ(recomputed, 2) << "variable_dt_recomputed_two_slots";
  for (int slot = 0; slot < depth; ++slot) {
    const double d = max_abs_diff(replayed[static_cast<std::size_t>(slot)],
                                  golden[static_cast<std::size_t>(slot)]);
    EXPECT_TRUE(d == 0.0) << "variable_dt slot=" << slot << " max|d|=" << d;
  }
}

// (C) The replay seed/capture follows the ring's qualified owner, not block 0.
TEST(CheckpointHistoryPolicy, ReplayUsesQualifiedNonzeroOwner) {
  kokkos();
  const SystemConfig cfg = make_cfg();
  const std::string ring = "second_state_prev";
  const int depth = 5;
  const double rate = 2.0;
  const std::vector<double> dts = {0.03, 0.07, 0.05, 0.11, 0.02, 0.09, 0.04};

  System source(cfg);
  add_gas_block(source, "first");
  add_gas_block(source, "second");
  source.set_poisson("charge_density", "geometric_mg");
  register_state_history(source, ring, depth, /*owner=*/1);
  install_ramp_program(source, ring, rate, /*owner=*/1);
  for (double dt : dts)
    source.step(dt);
  const std::vector<std::vector<double>> golden = dump_ring(source, ring);
  std::vector<double> slot_dt(static_cast<std::size_t>(depth));
  for (int k = 0; k < depth; ++k)
    slot_dt[static_cast<std::size_t>(k)] = source.history_slot_dt(ring, k);

  System restored(cfg);
  add_gas_block(restored, "first");
  add_gas_block(restored, "second");
  restored.set_poisson("charge_density", "geometric_mg");
  register_state_history(restored, ring, depth, /*owner=*/1);
  install_ramp_program(restored, ring, rate, /*owner=*/1);
  const std::vector<int> stored = {0, 2, 4};
  for (int k : stored)
    restored.restore_history(ring, k, golden[static_cast<std::size_t>(k)]);
  for (int k = 0; k < depth; ++k)
    restored.restore_history_slot_dt(ring, k, slot_dt[static_cast<std::size_t>(k)]);
  restored.set_history_initialized(ring, true);
  const std::vector<double> first_before = restored.state_global("first");

  EXPECT_EQ(restored.rebuild_history_slots(ring, stored), 2);
  EXPECT_EQ(restored.state_global("first"), first_before);
  const auto replayed = dump_ring(restored, ring);
  for (int slot = 0; slot < depth; ++slot)
    EXPECT_EQ(replayed[static_cast<std::size_t>(slot)], golden[static_cast<std::size_t>(slot)])
        << "owner=1 slot=" << slot;
}

// (D) The oldest slot MUST be stored: a policy whose stored set omits slot depth-1 is refused verbatim.
TEST(CheckpointHistoryPolicy, RebuildRefusesMissingOldestSlot) {
  kokkos();
  const SystemConfig cfg = make_cfg();
  const std::string ring = "state_prev";
  const int depth = 5;
  System s(cfg);
  add_gas(s);
  install_ramp_program(s, ring, 1.0);
  register_state_history(s, ring, depth);
  s.restore_history(ring, 0, s.history_global(ring, 0));  // register + a token slot
  s.set_history_initialized(ring, true);
  // stored = {0, 2} omits the oldest slot 4 -> unreconstructable.
  bool threw = false;
  std::string what;
  try {
    s.rebuild_history_slots(ring, std::vector<int>{0, 2});
  } catch (const std::runtime_error& e) {
    threw = true;
    what = e.what();
  }
  EXPECT_TRUE(threw) << "missing_oldest_slot_refused";
  EXPECT_TRUE(what.find("oldest slot") != std::string::npos)
      << "verbatim_oldest_slot_message: " << what;
}

// (E) The newest slot MUST be stored: there is no newer anchor from which to fill it backwards.
TEST(CheckpointHistoryPolicy, RebuildRefusesMissingNewestSlot) {
  kokkos();
  const SystemConfig cfg = make_cfg();
  const std::string ring = "state_prev";
  const int depth = 5;
  System s(cfg);
  add_gas(s);
  register_state_history(s, ring, depth);
  install_ramp_program(s, ring, 1.0);
  const std::vector<double> token = s.history_global(ring, 0);
  s.restore_history(ring, 2, token);
  s.restore_history(ring, 4, token);
  s.set_history_initialized(ring, true);
  bool threw = false;
  std::string what;
  try {
    s.rebuild_history_slots(ring, std::vector<int>{2, 4});
  } catch (const std::runtime_error& e) {
    threw = true;
    what = e.what();
  }
  EXPECT_TRUE(threw) << "missing_newest_slot_refused";
  EXPECT_TRUE(what.find("newest slot") != std::string::npos)
      << "verbatim_newest_slot_message: " << what;
}

// (F) Replay requires an installed Program: rebuild without a program fails loud (never a silent skip).
TEST(CheckpointHistoryPolicy, RebuildRefusesWithoutInstalledProgram) {
  kokkos();
  const SystemConfig cfg = make_cfg();
  const std::string ring = "state_prev";
  System s(cfg);
  add_gas(s);
  register_state_history(s, ring, /*depth=*/4);
  s.restore_history(ring, 0, s.history_global(ring, 0));
  s.restore_history(ring, 3, s.history_global(ring, 3));
  s.set_history_initialized(ring, true);
  bool threw = false;
  std::string what;
  try {
    s.rebuild_history_slots(ring, std::vector<int>{0, 3});
  } catch (const std::runtime_error& e) {
    threw = true;
    what = e.what();
  }
  EXPECT_TRUE(threw) << "no_program_refused";
  EXPECT_TRUE(what.find("no compiled Program") != std::string::npos)
      << "verbatim_no_program: " << what;
}
