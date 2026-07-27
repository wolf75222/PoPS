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
//   - dt_bound_ : used by BOTH runtimes. Each target loader wraps the generated scalar IR in its
//     concrete context; both CFL routes tighten the native bound before running the Program.
//   - hist_ / cache_ : UNIFORM ONLY today. The uniform System serializes the multistep history rings
//     and the held-node scheduler cache through the checkpoint; the AMR runtime defers both (its
//     history / cache seams are not wired), so these stay EMPTY on AMR. Keeping the storage here (one
//     struct) means an AMR history/cache seam later plugs into the SAME fields, never a fork.
// WHO OWNS STEPPING: the cadence fields (step_ / substeps_ / stride_ / dt_bound_) are READ by the
// driver, but the cadence LOOP lives at the call site, not here -- SystemProgramDriver::run_program_cadence
// on the uniform side, AmrSystem::Impl::run_program_cadence_ on the AMR side. This struct only STORES
// the cadence; it never advances the clock (no Impl / grid dependency leaks in).
//
// GRID BOUNDARY. The self-contained logic (cadence guards, diagnostics, block params, history-ring
// introspection + rotate, cache passthrough) lives HERE as methods with Program-subsystem-worded
// errors. The bodies that allocate or gather a MultiFab (register / read / store / restore a history
// ring, gather a cache value) need the owning runtime's (ba, dm, block-0 ncomp, write_state) and so
// stay in the runtime, delegating their STORAGE to this struct's hist_ / cache_ members. This header
// therefore has NO Kokkos / MultiFab-allocation dependency beyond the MultiFab type the rings hold.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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
/// A name maps to a ring of (depth = max lag + 1) MultiFabs, newest at [0]. Qualified keep_history
/// rings carry their exact runtime block owner; legacy internal rings retain owner=-1. The ring MEMORY
/// is allocated by the owning runtime (it needs the shared block layout), so this struct holds only
/// storage plus cheap, grid-free bookkeeping and the O(1) rotate. Grid-touching register/read/store/
/// restore bodies live in the runtime and reach these maps directly. Empty by default.
struct HistoryManager {
  std::map<std::string, std::vector<MultiFab>> histories;  // name -> ring (newest at [0])
  std::map<std::string, int> depth;                        // name -> ring length (max lag + 1)
  std::map<std::string, bool> initialized;                 // name -> stored at least once
  /// Number of authentic accepted stores currently represented by logical ring slots. The first
  /// store cold-fills deeper slots with copies for multistep evaluation, but advances this count by
  /// one only. Saturates at depth; selective checkpoint replay is valid only at full depth.
  std::map<std::string, int> fill_count;
  /// A store has populated slot zero since this ring's last logical rotation. Multiple writes in one
  /// Program tick still create one accepted logical sample; rotate consumes and clears this flag.
  std::map<std::string, bool> store_pending;
  std::map<std::string, int> owner;  // runtime block index (-1 legacy)
  std::map<std::string, std::string> state_identity;
  std::map<std::string, std::string> space_identity;
  std::map<std::string, std::string> clock_identity;
  std::map<std::string, std::string> interpolation_identity;
  /// PER-SLOT outgoing dt (ADC-626). keep_history snapshots U^n before the tail commit; after the
  /// ledger rotates with the ring, slot_dt[name][s] is the macro-step interval from the sample in
  /// slot s toward the newer sample in slot s-1. Reconstructing omitted slot j from exact older
  /// anchor j+1 uses slot_dt[j+1]. This preserves exact variable-dt provenance even when a
  /// run-to-target controller clips only its terminal step. A plain data member (no method the
  /// stepper template instantiates) -> MockImpl-safe; empty by default so dense paths never touch it.
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
      if (store_pending[name]) {
        fill_count[name] = std::min(static_cast<int>(ring.size()), fill_count[name] + 1);
        store_pending[name] = false;
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
      if (store_pending[name]) {
        fill_count[name] = std::min(static_cast<int>(ring.size()), fill_count[name] + 1);
        store_pending[name] = false;
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
  /// Installed macro-step body (ADC-399); empty makes every public facade temporal operation fail
  /// before mutation.
  std::function<void(double)> step_;
  /// OPTIONAL compiled-Program dt bound (ADC-417). The target-specific loader stores a closure here
  /// over ProgramContext (uniform) or AmrProgramContext (AMR); step_cfl tightens dt to
  /// min(native CFL, program bound). EMPTY when no Program exports a bound, so the native CFL is used
  /// unchanged. The closure owns no Python callback and executes entirely in the compiled module.
  std::function<Real(Real)> dt_bound_;
  /// GLOBAL macro-step cadence (ADC-411): substeps n partitions one representable accepted window;
  /// stride M runs the program once per M macro-steps (hold-then-catch-up). The accepted dt arguments
  /// are retained as reproducible left-folded provenance, while one prepared facade endpoint owns the
  /// numerical interval. Default 1/1 -> byte-identical to a single step_(dt) call. Read by the stepper;
  /// guarded by set_cadence.
  int substeps_ = 1;
  int stride_ = 1;
  /// Reproducible left-folded accepted-dt provenance and public-macro-step count currently held by a
  /// GLOBAL stride window.
  ///
  /// A stride window may span unequal adaptive-CFL steps. Reconstructing it as `stride * current_dt`
  /// loses that history and can advance the Program over a different physical interval than the
  /// facade clock. The driver therefore accumulates every accepted held dt in double precision for
  /// provenance, but derives numerical substeps from the explicitly prepared facade endpoint. The
  /// left fold, count and exact start are strict checkpoint state on Uniform and AMR; none may be
  /// inferred from `(time, macro_step)` during restart.
  double cadence_window_dt_ = 0.0;
  int cadence_window_steps_ = 0;
  /// Exact accepted physical start of an active held window. Keeping this coordinate explicitly is
  /// essential at large physical times: reconstructing it as `accepted_time - accumulated_dt` loses
  /// low bits before the Program starts. Zero is the canonical inactive image.
  double cadence_window_start_time_ = 0.0;
  /// A strict checkpoint restore installs the window before restoring the facade clock. This token
  /// authenticates that one subsequent set_clock targets the exact macro-step used to validate the
  /// window; direct mid-window clock changes without the window image fail closed.
  bool cadence_clock_restore_pending_ = false;
  int cadence_clock_restore_macro_step_ = 0;
  /// LAST macro-step dt handed to step_ (ADC-626). Set by the stepper right before each
  /// program_.step_(h) call (run_program_cadence, shared by step() and step_cfl()), so the runtime's
  /// pre-commit store_history can tag its state sample with the outgoing interval that advances it
  /// toward the next accepted sample (HistoryManager::slot_dt). A plain data field only assigned by
  /// the template (never a new method it instantiates) -> the mock System. Default 0 -> no program
  /// stepped yet.
  Real last_dt_ = Real(0);

  struct PreparedCadenceStep {
    bool due = false;
    /// Reproducible left-fold of the accepted dt arguments. This is checkpoint provenance: it proves
    /// which accepted intervals were held, but it is not an endpoint reconstruction authority.
    double effective_dt = 0.0;
    /// Exact accepted facade coordinates delimiting the window after this public step. window_end is
    /// computed once as accepted_time + dt and is the sole endpoint authority for facade clocks,
    /// Program stages and AMR accepted clocks.
    double window_start = 0.0;
    double window_end = 0.0;
    /// Representable displacement whose addition to window_start reproduces window_end. A fresh
    /// one-step window keeps the authored dt exactly (preserving cadence 1/1); a multi-step catch-up
    /// uses window_end - window_start. Floating-point associativity permits this to differ by one or
    /// more low bits from effective_dt. Program substeps partition this displacement, never
    /// `window_start + effective_dt`, so their final stage is bitwise the facade endpoint.
    double numerical_dt = 0.0;
    int window_steps = 0;
  };

  struct PreparedCadenceSubstep {
    double start = 0.0;
    double end = 0.0;
    double dt = 0.0;
  };

  // --- checkpoint / binding identity ---------------------------------------------------------------
  /// IR hash of the installed compiled Program (the .so's pops_program_hash, ADC-406b). Empty until
  /// install records it; serialized in the checkpoint so a restart against a DIFFERENT Program is
  /// rejected fail-loud. Used by BOTH runtimes.
  std::string installed_hash_;
  /// Exact prepared-operator authorities exported by the installed artifact and authenticated by
  /// the native loader before its install entry runs. Contexts may issue an unverified hot-apply
  /// capability only for a member of this table.
  std::vector<std::array<std::uint64_t, 4>> operator_authorities_;
  /// Exact owner-qualified history rings whose selective replay passed the compiled Program's
  /// owner-affine/context-free validation. A missing artifact table leaves this empty, so direct
  /// policy mutation or handcrafted binding calls cannot bypass that proof.
  std::vector<std::pair<std::string, int>> history_replay_authorities_;
  /// True only after install_program has authenticated an artifact and its replay-authority table.
  /// Direct C++ install_program_step is an explicitly trusted native composition seam and remains
  /// usable by low-level runtime tests; it is not exposed through the Python bindings.
  bool artifact_backed_ = false;
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

  /// Require the whole-system Program before a public facade may start a temporal operation.
  ///
  /// This guard deliberately lives in the shared Program state so Uniform and AMR fail with the
  /// same contract. Facades call it before profiling, lazy hierarchy construction, transaction
  /// capture, field solves or clock changes; a missing Program therefore cannot mutate observable
  /// runtime state.
  void require_step_installed(const char* operation) const {
    if (!step_)
      throw std::logic_error(std::string(operation) +
                             " requires an installed whole-system Program");
  }

  bool authorizes_history_replay(const std::string& ring, int depth) const {
    return !artifact_backed_ ||
           std::find(history_replay_authorities_.begin(), history_replay_authorities_.end(),
                     std::make_pair(ring, depth)) != history_replay_authorities_.end();
  }

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
    if ((cadence_window_steps_ != 0 || cadence_window_dt_ != 0.0 ||
         cadence_window_start_time_ != 0.0 || cadence_clock_restore_pending_) &&
        (substeps != substeps_ || stride != stride_))
      throw std::logic_error(runtime +
                             "::set_program_cadence cannot change an active or restoring stride "
                             "window");
    substeps_ = substeps;
    stride_ = stride;
  }

  /// Validate the exact held-window image against one accepted facade cursor.
  void validate_cadence_window(int macro_step, const std::string& runtime) const {
    if (macro_step < 0)
      throw std::invalid_argument(runtime + " Program cadence requires macro_step >= 0");
    if (stride_ < 1 || substeps_ < 1)
      throw std::logic_error(runtime + " Program cadence configuration is invalid");
    if (cadence_window_steps_ < 0 || cadence_window_steps_ >= stride_)
      throw std::runtime_error(runtime + " Program cadence window has an invalid held-step count");
    if (cadence_window_steps_ != macro_step % stride_)
      throw std::runtime_error(
          runtime +
          " Program cadence window phase differs from the authoritative macro-step; restore the "
          "strict cadence-window checkpoint state before set_clock");
    if (!std::isfinite(cadence_window_dt_) || cadence_window_dt_ < 0.0)
      throw std::runtime_error(runtime +
                               " Program cadence window duration must be finite and non-negative");
    if (!std::isfinite(cadence_window_start_time_))
      throw std::runtime_error(runtime + " Program cadence window start time must be finite");
    if ((cadence_window_steps_ == 0) != (cadence_window_dt_ == 0.0))
      throw std::runtime_error(
          runtime +
          " Program cadence window duration/count are inconsistent (zero iff no step is held)");
    if (cadence_window_steps_ == 0 && cadence_window_start_time_ != 0.0)
      throw std::runtime_error(
          runtime + " Program cadence inactive window must use the canonical zero start time");
  }

  /// Bind an active held window to the exact accepted physical cursor that owns it.
  void validate_cadence_window_time(double accepted_time, const std::string& runtime) const {
    if (!std::isfinite(accepted_time))
      throw std::invalid_argument(runtime + " Program cadence requires a finite accepted time");
    if (cadence_window_steps_ != 0 && !(cadence_window_start_time_ < accepted_time))
      throw std::runtime_error(
          runtime + " Program cadence active window start must precede the accepted physical time");
  }

  /// Prepare one public facade step without mutating the accepted cadence image.
  PreparedCadenceStep prepare_cadence_step(double accepted_time, int macro_step, double dt,
                                           const std::string& runtime) const {
    if (cadence_clock_restore_pending_)
      throw std::runtime_error(runtime +
                               " Program cadence clock restore was not completed by set_clock");
    validate_cadence_window(macro_step, runtime);
    validate_cadence_window_time(accepted_time, runtime);
    if (!std::isfinite(dt) || !(dt > 0.0))
      throw std::invalid_argument(runtime + " Program cadence requires a finite positive dt");
    const double final_time = accepted_time + dt;
    if (!std::isfinite(final_time) || !(final_time > accepted_time))
      throw std::overflow_error(runtime +
                                " Program cadence dt does not advance the finite physical clock");
    const double effective_dt = cadence_window_dt_ + dt;
    if (!std::isfinite(effective_dt) || !(effective_dt > cadence_window_dt_))
      throw std::overflow_error(
          runtime + " Program cadence dt does not advance the accumulated window duration");
    const double window_start =
        cadence_window_steps_ == 0 ? accepted_time : cadence_window_start_time_;
    const double numerical_dt = cadence_window_steps_ == 0 ? dt : final_time - window_start;
    if (!std::isfinite(numerical_dt) || !(numerical_dt > 0.0) ||
        window_start + numerical_dt != final_time)
      throw std::overflow_error(
          runtime +
          " Program cadence cannot represent one coherent facade/Program window endpoint");
    const int window_steps = cadence_window_steps_ + 1;
    if (window_steps < 1 || window_steps > stride_)
      throw std::logic_error(runtime + " Program cadence window crossed its configured stride");
    return PreparedCadenceStep{window_steps == stride_,
                               effective_dt,
                               window_start,
                               final_time,
                               numerical_dt,
                               window_steps};
  }

  /// Return one immutable numerical partition of a due cadence window.
  ///
  /// The accepted-dt left fold (`effective_dt`) remains exact restart provenance. Numerical stages
  /// instead partition the representable coordinate interval [window_start, window_end], whose end is
  /// the prepared facade authority. Every `start + dt == end` check is performed before execution by
  /// validate_cadence_partition, so a coordinate that cannot be represented coherently fails before
  /// the first Program mutation.
  PreparedCadenceSubstep prepare_cadence_substep(const PreparedCadenceStep& step, int substep,
                                                 int substeps, const std::string& runtime) const {
    if (!step.due)
      throw std::logic_error(runtime + " Program cadence cannot partition a held window");
    if (substeps < 1 || substep < 0 || substep >= substeps)
      throw std::logic_error(runtime + " Program cadence received an invalid substep partition");
    const auto endpoint = [&](int boundary) {
      if (boundary == 0)
        return step.window_start;
      if (boundary == substeps)
        return step.window_end;
      return step.window_start +
             step.numerical_dt * (static_cast<double>(boundary) / static_cast<double>(substeps));
    };
    const double start = endpoint(substep);
    const double end = endpoint(substep + 1);
    const double dt = substeps == 1 ? step.numerical_dt : end - start;
    if (!std::isfinite(start) || !std::isfinite(end) || !std::isfinite(dt) || !(end > start) ||
        !(dt > 0.0) || start + dt != end)
      throw std::overflow_error(runtime +
                                " Program cadence partition cannot represent a positive coherent "
                                "substep endpoint");
    return PreparedCadenceSubstep{start, end, dt};
  }

  /// Fail before any Program call if an authored substep count collapses the representable window.
  void validate_cadence_partition(const PreparedCadenceStep& step, int substeps,
                                  const std::string& runtime) const {
    if (substeps < 1 || !step.due || !std::isfinite(step.window_start) ||
        !std::isfinite(step.window_end) || !std::isfinite(step.numerical_dt) ||
        !(step.window_end > step.window_start) || !(step.numerical_dt > 0.0) ||
        step.window_start + step.numerical_dt != step.window_end)
      throw std::logic_error(runtime + " Program cadence prepared an invalid numerical window");
    for (int substep = 0; substep < substeps; ++substep)
      (void)prepare_cadence_substep(step, substep, substeps, runtime);
  }

  /// Commit the already executed public step. A due window is consumed; a held window persists its
  /// exact accumulated duration for the next adaptive step/checkpoint.
  void commit_cadence_step(const PreparedCadenceStep& step, const std::string& runtime) {
    if (step.window_steps != cadence_window_steps_ + 1 || step.window_steps < 1 ||
        step.window_steps > stride_ || step.due != (step.window_steps == stride_) ||
        !std::isfinite(step.effective_dt) || !(step.effective_dt > 0.0) ||
        !std::isfinite(step.window_start) || !std::isfinite(step.window_end) ||
        !std::isfinite(step.numerical_dt) || !(step.window_end > step.window_start) ||
        !(step.numerical_dt > 0.0) || step.window_start + step.numerical_dt != step.window_end)
      throw std::logic_error(runtime +
                             " Program cadence attempted to commit an invalid step image");
    if (step.due) {
      cadence_window_dt_ = 0.0;
      cadence_window_steps_ = 0;
      cadence_window_start_time_ = 0.0;
    } else {
      cadence_window_dt_ = step.effective_dt;
      cadence_window_steps_ = step.window_steps;
      cadence_window_start_time_ = step.window_start;
    }
  }

  /// Install an authenticated checkpoint window before set_clock. No historical duration is guessed.
  void restore_cadence_window(double accumulated_dt, int held_steps, double window_start_time,
                              int macro_step, const std::string& runtime) {
    const double saved_dt = cadence_window_dt_;
    const int saved_steps = cadence_window_steps_;
    const double saved_start = cadence_window_start_time_;
    cadence_window_dt_ = accumulated_dt;
    cadence_window_steps_ = held_steps;
    cadence_window_start_time_ = window_start_time;
    try {
      validate_cadence_window(macro_step, runtime);
    } catch (...) {
      cadence_window_dt_ = saved_dt;
      cadence_window_steps_ = saved_steps;
      cadence_window_start_time_ = saved_start;
      throw;
    }
    cadence_clock_restore_pending_ = true;
    cadence_clock_restore_macro_step_ = macro_step;
  }

  /// Authenticate/consume the cadence image for a facade clock restore. At a clean stride boundary,
  /// direct set_clock remains valid; a mid-window cursor always requires the explicit checkpoint image.
  void consume_cadence_clock_restore(double accepted_time, int macro_step,
                                     const std::string& runtime) {
    if (cadence_clock_restore_pending_) {
      if (macro_step != cadence_clock_restore_macro_step_)
        throw std::runtime_error(runtime +
                                 " set_clock macro-step differs from the restored cadence window");
      validate_cadence_window(macro_step, runtime);
      validate_cadence_window_time(accepted_time, runtime);
      cadence_clock_restore_pending_ = false;
      return;
    }
    validate_cadence_window(macro_step, runtime);
    validate_cadence_window_time(accepted_time, runtime);
    if (cadence_window_steps_ != 0)
      throw std::runtime_error(
          runtime +
          " set_clock cannot reuse an active stride window; restore its strict checkpoint image");
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
          std::to_string(count) +
          " runtime parameters > kMaxRuntimeParams=" + std::to_string(kMaxRuntimeParams) +
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
