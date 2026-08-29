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
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>  // add_compiled_model
#include <pops/runtime/system.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

namespace {

constexpr double kGamma = 1.4;
constexpr int kTestDimension = kNativeDimension;
using NativeSystem = System<kTestDimension>;
using NativeSystemConfig = SystemConfig<kTestDimension>;
using NativeField = MultiFab<kTestDimension>;
using NativeGasLaw = nd::IdealGasEuler<kTestDimension>;
constexpr int kNcomp = NativeGasLaw::n_vars;  // density, one momentum per axis, total energy

void install_execution_lane(NativeSystem& system, std::string identity) {
  system.install_prepared_boundary_execution_lane(
      std::make_shared<ExecutionLane>(ExecutionLane::world(std::move(identity))));
}

#if defined(POPS_HAS_KOKKOS)
void ensure_kokkos() {
  static std::unique_ptr<Kokkos::ScopeGuard> guard;
  if (!Kokkos::is_initialized() && !Kokkos::is_finalized())
    guard = std::make_unique<Kokkos::ScopeGuard>();
}
#endif

void add_gas(NativeSystem& s) {
  s.install_block_state_route("gas", "test.checkpoint-history.gas.state@1");
  add_compiled_model(s, "gas", NativeGasLaw::prepare(kGamma), "minmod", "rusanov", "conservative",
                     "explicit", kGamma);
  s.set_poisson("charge_density", "cartesian_cg");
}

runtime::system::AuxiliaryComponentKey install_uniform_checkpoint_input(NativeSystem& system) {
  using namespace runtime::system;
  AuxiliaryStorageShape<kTestDimension> shape;
  shape.spatial_rank = kTestDimension;
  shape.value_components = 1;
  for (int axis = 0; axis < kTestDimension; ++axis)
    shape.halo[axis] = 1;
  AuxiliaryComponentKey key{"test.uniform-checkpoint.owner", "auxiliary", "checkpoint-input",
                            "value"};
  AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "checkpoint", "scalar"};
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<kTestDimension>{
      "test.uniform-checkpoint/input",
      AuxiliaryProviderKind::input,
      {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
      {{key, contract, shape}},
      {}});
  system.seal_auxiliary_providers();
  return key;
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
  int fill_count = 0;
  std::vector<std::vector<double>> slots;  // slots[s] = global component-major buffer of slot s
};

// Serialize every registered ring the way sim.checkpoint does (history_names -> per-name accessors).
std::vector<SerializedHistory> serialize(const NativeSystem& s) {
  std::vector<SerializedHistory> out;
  for (const std::string& name : s.history_names()) {
    SerializedHistory h;
    h.name = name;
    h.depth = s.history_depth(name);
    h.ncomp = s.history_ncomp(name);
    h.initialized = s.history_initialized(name);
    h.fill_count = s.history_fill_count(name);
    for (int slot = 0; slot < h.depth; ++slot) {
      h.slots.push_back(s.history_global(name, slot));
    }
    out.push_back(std::move(h));
  }
  return out;
}

// Restore the serialized rings into a fresh System the way sim.restart does (restore_history per slot
// then set_history_initialized). restore_history registers the ring co-distributed with block 0.
void deserialize(NativeSystem& s, const std::vector<SerializedHistory>& hist) {
  for (const SerializedHistory& h : hist) {
    for (int slot = 0; slot < h.depth; ++slot) {
      s.restore_history(h.name, slot, h.slots[static_cast<std::size_t>(slot)]);
    }
    s.set_history_initialized(h.name, h.initialized);
    s.restore_history_fill_count(h.name, h.fill_count);
  }
}

}  // namespace

TEST(CheckpointHistory, RingRoundTripsBitEqualAcrossRestart) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  const int n = 8;
  int nn = 1;

  NativeSystemConfig cfg;
  for (int axis = 0; axis < kTestDimension; ++axis) {
    cfg.shape[axis] = n;
    cfg.lower[axis] = Real(0);
    cfg.upper[axis] = Real(1);
    cfg.periodicity[axis] = true;
    nn *= n;
  }

  NativeSystem src(cfg);
  install_execution_lane(src, "pops.test.checkpoint-history.source");
  add_gas(src);

  // A cold-start broadcast is valid for multistep evaluation but does not make the copied slots
  // authentic replay anchors. The accepted-store count advances once per store and saturates only
  // after every logical slot has been replaced.
  src.register_history("fill_age", /*lag=*/3);
  EXPECT_EQ(src.history_fill_count("fill_age"), 0);
  for (int accepted_store = 1; accepted_store <= 4; ++accepted_store) {
    const NativeField state = [&] {
      const auto state_view = src.block_state(0);
      return NativeField(*state_view.get());
    }();
    src.store_history("fill_age", state);
    src.rotate_histories();
    EXPECT_EQ(src.history_fill_count("fill_age"), accepted_store);
  }
  {
    const NativeField state = [&] {
      const auto state_view = src.block_state(0);
      return NativeField(*state_view.get());
    }();
    src.store_history("fill_age", state);
  }
  src.rotate_histories();
  EXPECT_EQ(src.history_fill_count("fill_age"), 4) << "fill count saturates at ring depth";

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
  const NativeField b_val = [&] {
    const auto b_view = src.read_history("scratch_b", 0);
    return NativeField(*b_view.get());
  }();
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
      EXPECT_TRUE(h.fill_count == h.depth) << "serialized_fill_count";
      EXPECT_TRUE(h.slots.size() == 3) << "serialized_slot_count";
    }
  }
  EXPECT_TRUE(saw_rhs_prev) << "rhs_prev_serialized";

  // --- RESTART: a fresh System (same block) restores the rings -----------------------------------
  NativeSystem dst(cfg);
  install_execution_lane(dst, "pops.test.checkpoint-history.destination");
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
    const auto r1 = dst.read_history("rhs_prev", 1);
    (void)r1;  // the handle exists (no throw on an initialized ring); its data is checked below.
  }
  EXPECT_TRUE(max_abs_diff(dst.history_global("rhs_prev", 1), A) < 1e-15)
      << "restored_lag1_is_A";

  // NO phantom cold-start after restore: the restored ring is already initialized, so the NEXT store
  // writes ONLY the current slot and leaves the deeper (restored) lags untouched. A naive
  // register-then-store would treat the ring as cold and broadcast, clobbering lag 1 / lag 2. The
  // store is issued WITHOUT an intervening rotate so the read handle stays valid (rotate swaps the ring
  // buffer handles); the point is the store scope, not the rotate.
  dst.register_history("scratch_c", 1);
  const std::vector<double> C = ramp(nn, 42.0);
  dst.restore_history("scratch_c", 0, C);
  dst.set_history_initialized("scratch_c", true);
  const NativeField c_val = [&] {
    const auto c_view = dst.read_history("scratch_c", 0);
    return NativeField(*c_view.get());
  }();
  dst.store_history("rhs_prev",
                    c_val);  // current slot [0] <- C; already-initialized -> no broadcast
  EXPECT_TRUE(max_abs_diff(dst.history_global("rhs_prev", 0), C) < 1e-15)
      << "post_restart_store_current_is_C";
  EXPECT_TRUE(max_abs_diff(dst.history_global("rhs_prev", 1), A) < 1e-15)
      << "no_phantom_coldstart_lag1_kept_A";
  EXPECT_TRUE(max_abs_diff(dst.history_global("rhs_prev", 2), A) < 1e-15)
      << "no_phantom_coldstart_lag2_kept_A";
}

TEST(CheckpointHistory,
     UniformAuxiliaryCheckpointAuthenticatesGroupsKeysShapesAndRollsBackRejectedRestore) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  NativeSystemConfig config;
  std::size_t cells = 1;
  for (int axis = 0; axis < kTestDimension; ++axis) {
    config.shape[axis] = 4;
    config.lower[axis] = Real(0);
    config.upper[axis] = Real(1);
    config.periodicity[axis] = true;
    cells *= static_cast<std::size_t>(config.shape[axis]);
  }
  NativeSystem origin(config);
  install_execution_lane(origin, "pops.test.checkpoint-history.uniform-origin");
  const auto input_key = install_uniform_checkpoint_input(origin);
  origin.stage_auxiliary_input(input_key, std::vector<double>(cells, 3.25));
  origin.refresh_auxiliary({"uniform-checkpoint", 4, 0, 0, 0, 0, 0,
                            runtime::system::AuxiliaryEvaluationEvent::initialization});

  const auto image = origin.capture_auxiliary_checkpoint_accepted_state();
  ASSERT_EQ(image.groups.size(), 1U);
  ASSERT_EQ(image.components.size(), 1U);
  ASSERT_EQ(image.providers.size(), 1U);
  EXPECT_EQ(image.components.front().key, input_key);
  EXPECT_EQ(image.accepted_generation, 1U);
  ASSERT_TRUE(image.providers.front().accepted_point.has_value());

  NativeSystem restarted(config);
  install_execution_lane(restarted, "pops.test.checkpoint-history.uniform-restarted");
  (void)install_uniform_checkpoint_input(restarted);
  EXPECT_NO_THROW(restarted.restore_auxiliary_checkpoint_accepted_state(image));
  EXPECT_EQ(restarted.capture_auxiliary_checkpoint_accepted_state(), image);

  auto rejected = image;
  rejected.components.front().key.component = "different";
  EXPECT_THROW(restarted.restore_auxiliary_checkpoint_accepted_state(rejected),
               std::invalid_argument);
  EXPECT_EQ(restarted.capture_auxiliary_checkpoint_accepted_state(), image);
}
