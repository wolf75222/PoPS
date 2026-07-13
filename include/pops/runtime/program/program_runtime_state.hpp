#pragma once

// Compiled time-Program RUNTIME STATE, extracted out of System::Impl / AmrSystem::Impl (ADC-594). A
// compiled Program (epic ADC-399) installs a whole-system macro-step closure plus a small cluster of
// SYSTEM-OWNED state -- the cadence, the checkpoint guard hash, the name-based block map, the runtime
// params, the recorded diagnostics, the multistep history rings, the scheduler cache and the
// profiler. Historically these ~10 fields (and their ~25 methods) lived DIRECTLY on the System::Impl
// god-object, indistinguishable from the block / field / layout invariants. This header gathers them
// into ONE inspectable subsystem so the Program responsibilities are localized, unit-testable, and
// clearly separated from the mesh/block invariants.
//
// SHARED UNIFORM/AMR CONTRACT (the issue forbids two diverging Program subsystems). Both the uniform
// System and the AMR AmrSystem embed ONE `ProgramRuntimeState` and route their Program seams through
// it. The two runtimes use DIFFERENT SUBSETS of the fields, documented per member below:
//   - step_ / substeps_ / stride_ / installed_hash_ / block_map_ / block_params_ / diagnostics_ /
//     profiler_ : used by BOTH runtimes (identical semantics).
//   - dt_bound_ : UNIFORM ONLY. The uniform SystemStepper tightens the CFL dt with the Program's
//     exported dt bound; the AMR runtime has no dt-bound seam, so this closure stays EMPTY on AMR
//     (documented divergence, not a second subsystem).
//   - hist_ / cache_ : UNIFORM ONLY today. The uniform System serializes the multistep history rings
//     and the held-node scheduler cache through the checkpoint; the AMR runtime defers both (its
//     history / cache seams are not wired), so these stay EMPTY on AMR. Keeping the storage here (one
//     struct) means an AMR history/cache seam later plugs into the SAME fields, never a fork.
// WHO OWNS STEPPING: the cadence fields (step_ / substeps_ / stride_ / dt_bound_) are READ by the
// stepper, but the cadence LOOP lives at the call site, not here -- SystemStepper::run_program_cadence
// on the uniform side, AmrSystem::Impl::run_program_cadence_ on the AMR side. This struct only STORES
// the cadence; it never advances the clock (no Impl / grid dependency leaks in).
//
// GRID BOUNDARY. The self-contained logic (cadence guards, diagnostics, block params, history-ring
// introspection + rotate, cache passthrough) lives HERE as methods with Program-subsystem-worded
// errors. The bodies that allocate or gather a MultiFab (register / read / store / restore a history
// ring, gather a cache value) need the owning runtime's (ba, dm, block-0 ncomp, write_state) and so
// stay in the runtime, delegating their STORAGE to this struct's hist_ / cache_ members. This header
// therefore has NO Kokkos / MultiFab-allocation dependency beyond the MultiFab type the rings hold.

#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pops/core/foundation/types.hpp>          // Real
#include <pops/mesh/storage/multifab.hpp>          // MultiFab (history ring element)
#include <pops/runtime/config/runtime_params.hpp>  // RuntimeParams, kMaxRuntimeParams
#include <pops/runtime/program/cache_manager.hpp>  // CacheManager (held-node scheduler cache)
#include <pops/runtime/program/profiler.hpp>       // Profiler (per-node / per-brick timing)

namespace pops::runtime::program {

/// Multistep history ring buffers (ADC-406a), owned by the Program runtime state.
///
/// A name maps to a ring of (depth = max lag + 1) MultiFabs, newest at [0], each co-distributed with
/// block 0's state. The ring MEMORY is allocated by the owning runtime (it needs the block (ba, dm)),
/// so this struct holds only the storage + the cheap, grid-free bookkeeping (depth, initialized) and
/// the O(1) rotate. The grid-touching register / read / store / restore bodies live in the runtime
/// and reach these maps directly. Empty by default -> the single-step paths never touch it.
struct HistoryManager {
  std::map<std::string, std::vector<MultiFab>> histories;  // name -> ring (newest at [0])
  std::map<std::string, int> depth;                        // name -> ring length (max lag + 1)
  std::map<std::string, bool> initialized;                 // name -> stored at least once
  std::map<std::string, int> owner;                        // runtime block index (-1 legacy)
  std::map<std::string, std::string> state_identity;
  std::map<std::string, std::string> space_identity;
  std::map<std::string, std::string> clock_identity;
  std::map<std::string, std::string> interpolation_identity;
  /// PER-SLOT dt (ADC-626). slot_dt[name][s] = the macro-step dt whose commit produced the value now
  /// in slot s (slot 0 = newest). Filled by the runtime's store_history (which knows the current dt via
  /// ProgramRuntimeState::last_dt_) and rotated ALONGSIDE the ring, so a selective-persistence restart
  /// can re-step the recomputed slots with the EXACT recorded dt sequence (variable-dt replay is
  /// bit-exact). A plain data member (no method the stepper template instantiates) -> MockImpl-safe;
  /// empty by default so the dense / non-persistence paths never touch it.
  std::map<std::string, std::vector<Real>> slot_dt;

  /// Shift each ring one step (newest-to-oldest), called ONCE at the end of a macro-step. O(1)
  /// std::swap of the MultiFab handles (not a deep copy): the swap chain from the deepest slot down
  /// to 1 leaves every read slot k >= 1 holding slot k-1's old value and RECYCLES the now-oldest
  /// buffer into slot [0] (overwritten by the next store before any read). Grid-free -> lives here.
  /// The per-slot dt (ADC-626) rotates on the SAME chain (a scalar swap) so slot_dt stays aligned with
  /// the ring it annotates.
  void rotate() {
    for (auto& [name, ring] : histories) {
      for (std::size_t k = ring.size(); k-- > 1;)
        std::swap(ring[k], ring[k - 1]);
      auto dt_it = slot_dt.find(name);
      if (dt_it != slot_dt.end()) {
        std::vector<Real>& dts = dt_it->second;
        for (std::size_t k = dts.size(); k-- > 1;)
          std::swap(dts[k], dts[k - 1]);
      }
    }
  }

  /// Rotate only rings owned by one qualified logical clock. Generated multirate Programs use this
  /// overload; the unqualified rotate above remains the internal legacy seam.
  void rotate(const std::string& clock) {
    for (auto& [name, ring] : histories) {
      const auto qualified = clock_identity.find(name);
      if (qualified == clock_identity.end() || qualified->second != clock)
        continue;
      for (std::size_t k = ring.size(); k-- > 1;)
        std::swap(ring[k], ring[k - 1]);
      auto dt_it = slot_dt.find(name);
      if (dt_it != slot_dt.end()) {
        std::vector<Real>& dts = dt_it->second;
        for (std::size_t k = dts.size(); k-- > 1;)
          std::swap(dts[k], dts[k - 1]);
      }
    }
  }

  /// Names of the registered history rings (checkpoint enumeration).
  std::vector<std::string> names() const {
    std::vector<std::string> out;
    out.reserve(histories.size());
    for (const auto& [name, ring] : histories) {
      (void)ring;
      out.push_back(name);
    }
    return out;
  }
};

/// The compiled time-Program runtime state, extracted from the System / AmrSystem god-object (ADC-594).
///
/// A plain aggregate: the owning Impl embeds ONE instance and routes every Program seam through it. The
/// self-contained (grid-free) logic is exposed as methods with Program-subsystem-worded errors; the
/// grid-touching history / cache bodies delegate their STORAGE to hist_ / cache_ from the runtime. See
/// the file header for the shared Uniform/AMR contract (which fields each runtime uses).
struct ProgramRuntimeState {
  // --- fields read by the stepper (the ONLY Program state the stepper sees) -------------------------
  /// Installed macro-step body (ADC-399); empty -> the historical / native step path.
  std::function<void(double)> step_;
  /// OPTIONAL compiled-Program dt bound (ADC-417), UNIFORM ONLY. When a generated .so exports one, the
  /// uniform install stores a closure here and SystemStepper::step_cfl tightens dt to
  /// min(native CFL, program bound). EMPTY on the AMR runtime (no dt-bound seam) and when no Program
  /// exports a bound -> the native CFL is used UNCHANGED (documented Uniform/AMR divergence).
  std::function<Real(Real)> dt_bound_;
  /// GLOBAL macro-step cadence (ADC-411): substeps n runs step_ n times over eff_dt/n; stride M runs
  /// the program once per M macro-steps with eff_dt = M*dt (hold-then-catch-up). Default 1/1 ->
  /// byte-identical to a single step_(dt) call. Read by the stepper; guarded by set_cadence.
  int substeps_ = 1;
  int stride_ = 1;
  /// LAST macro-step dt handed to step_ (ADC-626). Set by the stepper right before each
  /// program_.step_(h) call (run_program_cadence, shared by step() and step_cfl()), so the runtime's
  /// store_history can tag the slot it produces with the dt that produced it (HistoryManager::slot_dt).
  /// A plain data field only assigned by the template (never a new method it instantiates) -> the mock
  /// System. Default 0 -> no program stepped yet.
  Real last_dt_ = Real(0);

  // --- checkpoint / binding identity ---------------------------------------------------------------
  /// IR hash of the installed compiled Program (the .so's pops_program_hash, ADC-406b). Empty until
  /// install records it; serialized in the checkpoint so a restart against a DIFFERENT Program is
  /// rejected fail-loud. Used by BOTH runtimes.
  std::string installed_hash_;
  /// NAME-based block binding (ADC-457): program-index -> runtime-block-index map. Entry p holds the
  /// runtime block index the Program's block p names. EMPTY = identity (positional convention). Used
  /// by BOTH runtimes; read by the (Amr)ProgramContext.
  std::vector<int> block_map_;

  // --- runtime data owned across the step closure --------------------------------------------------
  /// COMPILED-PROGRAM SCALAR DIAGNOSTICS (ADC-414): name -> last value recorded via P.record_scalar.
  /// Lives here (not the .so) so it outlives the step closure and Python can read it. Used by BOTH.
  std::map<std::string, Real> diagnostics_;
  /// COMPILED-PROGRAM RUNTIME PARAMETERS (ADC-510 / ADC-508): program-block index -> current
  /// RuntimeParams for a Program that reads dsl.Param(..., kind="runtime"). Seeded to the declaration
  /// defaults at install, overwritten at run time; the step closure reads the CURRENT value each step
  /// (no recompile). Lives here so the change reaches the captured context. Used by BOTH.
  std::map<int, RuntimeParams> block_params_;

  // --- owned subsystems ----------------------------------------------------------------------------
  /// PER-NODE / PER-BRICK PROFILER (ADC-459): disabled by default (no hot-path cost when off). On the
  /// uniform runtime System::step / solve_fields wrap themselves in a ProfileScope into it; on AMR the
  /// engine is wired to its address at build. Used by BOTH.
  Profiler profiler_;
  /// SCHEDULER VALUE CACHE (ADC-458), UNIFORM ONLY. The held-node cache (every(N).hold / accumulate_dt)
  /// keyed by IR node id; the uniform checkpoint serializes it. Empty on AMR (cache seam not wired).
  CacheManager cache_;
  /// MULTISTEP HISTORY (ADC-406a), UNIFORM ONLY. Ring buffers for multistep schemes; the uniform
  /// checkpoint serializes them. Empty on AMR (history seam not wired).
  HistoryManager hist_;

  // --- self-contained helpers (grid-free, Program-subsystem-worded errors) -------------------------

  /// Validate + set the GLOBAL macro-step cadence (ADC-411). @p runtime is the caller's runtime name
  /// ("System" / "AmrSystem") so the fail-loud message names the Program subsystem setter verbatim.
  /// @throws std::invalid_argument if @p substeps < 1 or @p stride < 1 (a non-positive cadence is
  /// meaningless). Preserves the historical message shape (`set_program_cadence: substeps >= 1 ...`).
  void set_cadence(int substeps, int stride, const std::string& runtime) {
    if (substeps < 1)
      throw std::invalid_argument(runtime + "::set_program_cadence: substeps >= 1 required (got " +
                                  std::to_string(substeps) + ")");
    if (stride < 1)
      throw std::invalid_argument(runtime + "::set_program_cadence: stride >= 1 required (got " +
                                  std::to_string(stride) + ")");
    substeps_ = substeps;
    stride_ = stride;
  }

  /// Record a compiled-Program scalar diagnostic (ADC-414): the installed Program writes named scalars
  /// via P.record_scalar; Python reads them after the step. Idempotent (last write wins).
  void record_diagnostic(const std::string& name, Real value) { diagnostics_[name] = value; }

  /// Read the named diagnostic, FAIL-LOUD if the Program never recorded it. @p runtime names the
  /// Program subsystem setter in the message (not a generic getter). @throws std::out_of_range.
  Real diagnostic(const std::string& name, const std::string& runtime) const {
    auto it = diagnostics_.find(name);
    if (it == diagnostics_.end())
      throw std::out_of_range(
          runtime + "::program_diagnostic: no diagnostic named '" + name +
          "' has been recorded (the installed Program must P.record_scalar it)");
    return it->second;
  }

  /// The whole name -> value diagnostics map (checkpoint / inspection). By value: inert copy.
  std::map<std::string, Real> diagnostics() const { return diagnostics_; }

  /// Seed a program block's RuntimeParams to its declaration defaults (ADC-510 / ADC-508). Idempotent
  /// (re-seeding resets to the baseline). Called by install. DEFENCE IN DEPTH (ADC-610): a block with
  /// more than kMaxRuntimeParams params is REJECTED here with a user-facing error instead of being
  /// SILENTLY TRUNCATED into the fixed-size device carrier -- the Python codegen enforces the same bound
  /// upstream, so this only fires for a hand-built .so with bogus pops_program_param_* metadata.
  void seed_params(int prog_block, const std::vector<double>& defaults) {
    const int count = static_cast<int>(defaults.size());
    if (count > kMaxRuntimeParams)
      throw std::runtime_error(
          "install_program: program block " + std::to_string(prog_block) + " declares " +
          std::to_string(count) + " runtime parameters > kMaxRuntimeParams=" +
          std::to_string(kMaxRuntimeParams) +
          " (include/pops/runtime/config/runtime_params.hpp); the fixed-size device carrier "
          "RuntimeParams cannot hold them. Regenerate the problem.so with the current headers.");
    RuntimeParams rp;
    rp.count = count;
    for (int k = 0; k < rp.count; ++k)
      rp.values[k] = static_cast<Real>(defaults[static_cast<std::size_t>(k)]);
    block_params_[prog_block] = rp;
  }

  /// Overwrite a program block's runtime parameter values (the no-recompile contract). @p runtime
  /// names the Program subsystem setter in both fail-loud messages. @throws std::out_of_range if the
  /// block was never seeded (the Program declares no runtime param for it), std::runtime_error on a
  /// value-count mismatch.
  void set_params(int prog_block, const std::vector<double>& values, const std::string& runtime) {
    auto it = block_params_.find(prog_block);
    if (it == block_params_.end())
      throw std::out_of_range(
          runtime + "::set_program_params: program block " + std::to_string(prog_block) +
          " has no runtime parameter (the installed compiled Program declares none for it; declare "
          "dsl.Param(..., kind='runtime') in the model the Program lowers, or omit params=)");
    RuntimeParams& rp = it->second;
    if (static_cast<int>(values.size()) != rp.count)
      throw std::runtime_error(runtime + "::set_program_params: program block " +
                               std::to_string(prog_block) + " expects " + std::to_string(rp.count) +
                               " runtime parameters, received " + std::to_string(values.size()));
    for (int k = 0; k < rp.count; ++k)
      rp.values[k] = static_cast<Real>(values[static_cast<std::size_t>(k)]);
  }

  /// Read a program block's current RuntimeParams. An unseeded block (no runtime param) returns a
  /// default RuntimeParams (count 0) -- a kernel that reads no param is unaffected. By value:
  /// device-clean, trivially copyable.
  RuntimeParams params(int prog_block) const {
    auto it = block_params_.find(prog_block);
    return it == block_params_.end() ? RuntimeParams{} : it->second;
  }
};

}  // namespace pops::runtime::program
