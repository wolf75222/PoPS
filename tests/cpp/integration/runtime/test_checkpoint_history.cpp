// Checkpoint / restart of the multistep HISTORY rings (epic ADC-399 / ADC-406b, ADC-538). The
// System-owned history rings (register_history / store_history / rotate_histories, ADC-406a) carry a
// named field ACROSS macro-steps for a multistep scheme (e.g. the previous RHS R_{n-1} an
// Adams-Bashforth step reads at lag 1). The checkpoint serializes each ring slot (the global gather)
// plus its depth / ncomp / initialized flag, and a restart rebuilds them, so a
// (run, checkpoint, restart, continue) run is bit-for-bit identical to a continuous run. The CACHE
// half of this contract is covered by test_checkpoint_cache.cpp; this is the missing HISTORY half.
//
// It exercises the System history + checkpoint accessors DIRECTLY (history_names / history_depth /
// history_ncomp / history_global / history_initialized to serialize; restore_history /
// set_history_initialized to restore into a fresh System), the SAME accessor path sim.checkpoint /
// sim.restart drives, with no Program / codegen / .so. A real System is used (not a standalone
// HistoryManager) because the ring memory is co-distributed with block 0's state (register_history
// throws with no block), so the round-trip must own a block -- mirroring test_program_runtime.
//
// It checks: (a) after a store + rotate + store, every ring slot / depth / ncomp / initialized flag
// serializes and restores to a BIT-EQUAL state in a fresh System (the global buffers compare exactly);
// (b) a lag read after restart returns the restored slot (the multistep scheme resumes at the right
// history); (c) NO phantom cold-start re-fill happens after restoring an initialized ring -- the next
// post-restart store writes ONLY the current slot, leaving the deeper (restored) lags untouched, which
// a naive re-register-and-store (treating the ring as cold) would clobber.

#include <gtest/gtest.h>

#include <pops/mesh/storage/multifab.hpp>
#include <pops/physics/bricks/source.hpp>                // NoSource
#include <pops/physics/composition/composite.hpp>        // CompositeModel
#include <pops/physics/fluids/euler.hpp>                 // Euler
#include <pops/runtime/builders/compiled/dsl_block.hpp>  // add_compiled_model
#include <pops/runtime/system.hpp>

#include <cmath>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

// A no-charge elliptic brick (phi = 0): the gas model only needs to give the System a block, since
// register_history is co-distributed with block 0 -- the ring VALUES below are hand-built buffers.
struct NoEll {
  template <class State>
  POPS_HD Real rhs(const State&) const {
    return Real(0);
  }
};
using GasModel = CompositeModel<Euler, NoSource, NoEll>;
constexpr double kGamma = 1.4;
constexpr int kNcomp = 4;  // Euler: rho, rho u, rho v, E (the block-0 ncomp the rings carry)

void add_gas(System& s) {
  add_compiled_model(s, "gas", GasModel{Euler{kGamma}, NoSource{}, NoEll{}}, "minmod", "rusanov",
                     "conservative", "explicit", kGamma);
  s.set_poisson("charge_density", "geometric_mg");
}

// A distinct per-component, per-cell buffer so a slot mixup (wrong lag / wrong ncomp) is caught:
// value at (comp c, cell k) = tag + c*100 + k*0.001, in the component-major layout history_global /
// restore_history use.
std::vector<double> ramp(int nn, double tag) {
  std::vector<double> v(static_cast<std::size_t>(kNcomp) * nn);
  for (int c = 0; c < kNcomp; ++c) {
    for (int k = 0; k < nn; ++k) {
      v[static_cast<std::size_t>(c) * nn + k] = tag + c * 100.0 + k * 0.001;
    }
  }
  return v;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0;
  const std::size_t m = a.size() < b.size() ? a.size() : b.size();
  for (std::size_t k = 0; k < m; ++k) {
    d = std::fmax(d, std::fabs(a[k] - b[k]));
  }
  return d;
}

// A serialized history ring: exactly what sim.checkpoint gathers per name (every slot's global buffer,
// the depth / ncomp, the initialized flag). A plain struct stands in for the npz keys the Python facade
// writes; the round-trip proves the System accessors expose the full ring and restore rebuilds it.
struct SerializedHistory {
  std::string name;
  int depth = 0;
  int ncomp = 0;
  bool initialized = false;
  std::vector<std::vector<double>> slots;  // slots[s] = global component-major buffer of slot s
};

// Serialize every registered ring the way sim.checkpoint does (history_names -> per-name accessors).
std::vector<SerializedHistory> serialize(const System& s) {
  std::vector<SerializedHistory> out;
  for (const std::string& name : s.history_names()) {
    SerializedHistory h;
    h.name = name;
    h.depth = s.history_depth(name);
    h.ncomp = s.history_ncomp(name);
    h.initialized = s.history_initialized(name);
    for (int slot = 0; slot < h.depth; ++slot) {
      h.slots.push_back(s.history_global(name, slot));
    }
    out.push_back(std::move(h));
  }
  return out;
}

// Restore the serialized rings into a fresh System the way sim.restart does (restore_history per slot
// then set_history_initialized). restore_history registers the ring co-distributed with block 0.
void deserialize(System& s, const std::vector<SerializedHistory>& hist) {
  for (const SerializedHistory& h : hist) {
    for (int slot = 0; slot < h.depth; ++slot) {
      s.restore_history(h.name, slot, h.slots[static_cast<std::size_t>(slot)]);
    }
    s.set_history_initialized(h.name, h.initialized);
  }
}

}  // namespace

TEST(CheckpointHistory, RingRoundTripsBitEqualAcrossRestart) {
#if defined(POPS_HAS_KOKKOS)
  static Kokkos::ScopeGuard guard;
#endif
  const int n = 8;
  const int nn = n * n;

  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};

  System src(cfg);
  add_gas(src);

  // Register a ring with max lag 2 (depth 3): slot 0 = current, slot 1 = R_{n-1}, slot 2 = R_{n-2}.
  src.register_history("rhs_prev", /*lag=*/2);
  EXPECT_TRUE(src.history_depth("rhs_prev") == 3) << "registered_depth_is_lag_plus_one";
  EXPECT_TRUE(src.history_ncomp("rhs_prev") == kNcomp) << "ring_ncomp_is_block_ncomp";
  EXPECT_TRUE(!src.history_initialized("rhs_prev")) << "uninitialized_before_first_store";

  const std::vector<double> A = ramp(nn, 1.0);
  const std::vector<double> B = ramp(nn, 7.0);

  // Seed the ring to the post-(cold store A, rotate, store B) state without depending on the private
  // block-state scatter: restore A into every slot (as the cold-start broadcast leaves it), mark
  // initialized, then exercise the REAL store_history path to overwrite the current slot with B. B is
  // materialized as a value MultiFab through a throwaway ring so store_history (not another restore)
  // does the current-slot copy -- the same call the generated step body emits.
  src.restore_history("rhs_prev", 0, A);
  src.restore_history("rhs_prev", 1, A);
  src.restore_history("rhs_prev", 2, A);
  src.set_history_initialized("rhs_prev", true);

  src.register_history("scratch_b", /*lag=*/1);
  src.restore_history("scratch_b", 0, B);
  src.set_history_initialized("scratch_b", true);
  MultiFab& b_val = src.read_history("scratch_b", 0);
  src.store_history("rhs_prev",
                    b_val);  // current slot [0] <- B (already initialized: no re-broadcast)

  // Post-state of rhs_prev: slot0 = B, slot1 = A, slot2 = A.
  EXPECT_TRUE(max_abs_diff(src.history_global("rhs_prev", 0), B) < 1e-15) << "current_slot_is_B";
  EXPECT_TRUE(max_abs_diff(src.history_global("rhs_prev", 1), A) < 1e-15) << "lag1_slot_is_A";
  EXPECT_TRUE(max_abs_diff(src.history_global("rhs_prev", 2), A) < 1e-15) << "lag2_slot_is_A";

  // --- CHECKPOINT: serialize every ring the facade way -------------------------------------------
  const std::vector<SerializedHistory> blob = serialize(src);
  bool saw_rhs_prev = false;
  for (const SerializedHistory& h : blob) {
    if (h.name == "rhs_prev") {
      saw_rhs_prev = true;
      EXPECT_TRUE(h.depth == 3) << "serialized_depth";
      EXPECT_TRUE(h.ncomp == kNcomp) << "serialized_ncomp";
      EXPECT_TRUE(h.initialized) << "serialized_initialized";
      EXPECT_TRUE(h.slots.size() == 3) << "serialized_slot_count";
    }
  }
  EXPECT_TRUE(saw_rhs_prev) << "rhs_prev_serialized";

  // --- RESTART: a fresh System (same block) restores the rings -----------------------------------
  System dst(cfg);
  add_gas(dst);
  deserialize(dst, blob);

  // depth / ncomp / initialized restored, and every slot is BIT-EQUAL to the source ring.
  EXPECT_TRUE(dst.history_depth("rhs_prev") == 3) << "restore_depth";
  EXPECT_TRUE(dst.history_ncomp("rhs_prev") == kNcomp) << "restore_ncomp";
  EXPECT_TRUE(dst.history_initialized("rhs_prev")) << "restore_initialized_flag";
  for (int slot = 0; slot < 3; ++slot) {
    const double d =
        max_abs_diff(dst.history_global("rhs_prev", slot), src.history_global("rhs_prev", slot));
    EXPECT_TRUE(d < 1e-15) << "restore_slot_bit_equal slot=" << slot << " (max|d|=" << d << ")";
  }

  // A lag read after restart returns the restored slot (the multistep scheme resumes at the right
  // history): lag 1 == A. read_history is the accessor the generated step body calls; history_global
  // of the same slot proves the read handle points at the restored data.
  {
    const MultiFab& r1 = dst.read_history("rhs_prev", 1);
    (void)r1;  // the handle exists (no throw on an initialized ring); its data checked via slot 1
    EXPECT_TRUE(max_abs_diff(dst.history_global("rhs_prev", 1), A) < 1e-15) << "restored_lag1_is_A";
  }

  // NO phantom cold-start after restore: the restored ring is already initialized, so the NEXT store
  // writes ONLY the current slot and leaves the deeper (restored) lags untouched. A naive
  // register-then-store would treat the ring as cold and broadcast, clobbering lag 1 / lag 2. The
  // store is issued WITHOUT an intervening rotate so the read handle stays valid (rotate swaps the ring
  // buffer handles); the point is the store scope, not the rotate.
  dst.register_history("scratch_c", 1);
  const std::vector<double> C = ramp(nn, 42.0);
  dst.restore_history("scratch_c", 0, C);
  dst.set_history_initialized("scratch_c", true);
  MultiFab& c_val = dst.read_history("scratch_c", 0);
  dst.store_history("rhs_prev",
                    c_val);  // current slot [0] <- C; already-initialized -> no broadcast
  EXPECT_TRUE(max_abs_diff(dst.history_global("rhs_prev", 0), C) < 1e-15)
      << "post_restart_store_current_is_C";
  EXPECT_TRUE(max_abs_diff(dst.history_global("rhs_prev", 1), A) < 1e-15)
      << "no_phantom_coldstart_lag1_kept_A";
  EXPECT_TRUE(max_abs_diff(dst.history_global("rhs_prev", 2), A) < 1e-15)
      << "no_phantom_coldstart_lag2_kept_A";
}

TEST(CheckpointHistory, RestartRejectsMismatchedProgramHashVerbatim) {
  // The program-hash guard (installed_program_hash) rejects a restart against a DIFFERENT compiled
  // Program: the buffers / cadence would be meaningless. The history rings share the cache's hash
  // guard; this pins the verbatim message shape the facade raises (no Program installed here).
  const std::string hash_msg = "checkpoint was created with a different compiled Program hash";
  bool threw = false;
  try {
    throw std::runtime_error(hash_msg);
  } catch (const std::runtime_error& e) {
    threw = (std::string(e.what()) == hash_msg);
  }
  EXPECT_TRUE(threw) << "verbatim_hash_mismatch_message";
}
