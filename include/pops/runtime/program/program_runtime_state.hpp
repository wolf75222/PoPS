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
// WHO OWNS STEPPING: this state owns the one topology-independent cadence LOOP as well as its fields
// (step_ / substeps_ / stride_ / dt_bound_). Uniform and AMR lend it only their accepted
// `(physical_time, macro_step)` cursor by reference; no Impl, grid or hierarchy dependency crosses
// this boundary. Exact-ranked ProgramContext<Dim> remains the sole implementation of operations
// invoked by the installed step closure, while runtime drivers merely enter this shared dispatcher.
//
// GRID BOUNDARY. The self-contained logic (cadence guards, diagnostics, block params, history-ring
// introspection + rotate, cache passthrough) lives HERE as methods with Program-subsystem-worded
// errors. The bodies that allocate or gather a MultiFab (register / read / store / restore a history
// ring, gather a cache value) need the owning runtime's (ba, dm, block-0 ncomp, write_state) and so
// stay in the runtime, delegating their STORAGE to this struct's hist_ / cache_ members. This header
// therefore has NO Kokkos / MultiFab-allocation dependency beyond the MultiFab type the rings hold.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <pops/core/foundation/types.hpp>  // Real
#include <pops/mesh/storage/multifab.hpp>  // MultiFab (history ring element)
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/runtime/config/runtime_params.hpp>    // RuntimeParams, kMaxRuntimeParams
#include <pops/runtime/program/cache_manager.hpp>    // CacheManager (held-node scheduler cache)
#include <pops/runtime/program/module_metadata.hpp>  // frozen checkpoint-shape metadata
#include <pops/runtime/program/profiler.hpp>         // Profiler (per-node / per-brick timing)

namespace pops::runtime::program {

/// Type-erased, already allocated image of one artifact context's accepted state.  Runtime
/// transactions capture it before entering scientific code and invoke only the noexcept publication
/// after the carrier and Program storage have been restored.
class AcceptedProgramContextSnapshot {
 public:
  AcceptedProgramContextSnapshot() = default;
  AcceptedProgramContextSnapshot(const AcceptedProgramContextSnapshot&) = delete;
  AcceptedProgramContextSnapshot& operator=(const AcceptedProgramContextSnapshot&) = delete;
  virtual ~AcceptedProgramContextSnapshot() = default;

  virtual std::unique_ptr<AcceptedProgramContextSnapshot> prepare_restore() const = 0;
  virtual void publish_restore() noexcept = 0;
};

using AcceptedProgramContextSnapshotFactory =
    std::function<std::unique_ptr<AcceptedProgramContextSnapshot>()>;

/// Multistep history ring buffers (ADC-406a), owned by the Program runtime state.
///
/// A name maps to a ring of (depth = max lag + 1) MultiFabs, newest at [0]. Qualified keep_history
/// rings carry their exact runtime block owner; legacy internal rings retain owner=-1. The ring MEMORY
/// is allocated by the owning runtime (it needs the shared block layout), so this struct holds only
/// storage plus cheap, grid-free bookkeeping and the O(1) rotate. Grid-touching register/read/store/
/// restore bodies live in the runtime and reach these maps directly. Empty by default.
template <int Dim>
struct HistoryManager {
  static_assert(Dim >= 1 && Dim <= 3, "HistoryManager only supports dimensions 1, 2, and 3");

  std::map<std::string, std::vector<MultiFab<Dim>>> histories;  // name -> ring (newest at [0])
  std::map<std::string, int> depth;                             // name -> ring length (max lag + 1)
  std::map<std::string, bool> initialized;                      // name -> stored at least once
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

/// Attempt-local native balance evidence emitted by one exact runtime operator.
///
/// The coordinate deliberately remains independent of a user-facing BalanceLedger route: native
/// operators know their qualified runtime block, hierarchy level and conservative component, while
/// the route-to-quantity selector is a separate planning authority. Keeping both identities
/// separate prevents a reflux correction from being silently relabelled as a complete balance.
struct AutomaticBalanceKey {
  int runtime_block = -1;
  int level = -1;
  int component = -1;
  std::string term;

  friend bool operator<(const AutomaticBalanceKey& left, const AutomaticBalanceKey& right) {
    if (left.runtime_block != right.runtime_block)
      return left.runtime_block < right.runtime_block;
    if (left.level != right.level)
      return left.level < right.level;
    if (left.component != right.component)
      return left.component < right.component;
    return left.term < right.term;
  }
};

/// Exact-ranked Program-owned field-boundary overlay. Generated installers write one attempt-local
/// stage; the Uniform loader commits it only after validating the complete registry. The durable
/// baseline remains distinct so replacing A(dynamic) by B(no export) cannot retain A's DSO-owned
/// function pointers.
template <int Dim>
struct ArtifactFieldBoundaryAuthority {
  std::optional<CompiledFieldBoundaryKernel<Dim>> kernel;
  std::optional<FieldLogicalTimePoint> point;
  std::vector<Real> parameters;
};

template <int Dim>
using ArtifactFieldBoundaryAuthorityRegistry =
    std::map<std::string, ArtifactFieldBoundaryAuthority<Dim>>;

template <int Dim>
struct ArtifactFieldBoundaryStage {
  ArtifactFieldBoundaryAuthorityRegistry<Dim> authorities;
  std::set<std::string> kernel_slots;
  std::set<std::string> point_slots;
  std::set<std::string> parameter_slots;
};

/// The compiled time-Program runtime state, extracted from the System / AmrSystem god-object (ADC-594).
///
/// A plain aggregate: the owning Impl embeds ONE instance and routes every Program seam through it. The
/// self-contained (grid-free) logic is exposed as methods with Program-subsystem-worded errors; the
/// grid-touching history / cache bodies delegate their STORAGE to hist_ / cache_ from the runtime. See
/// the file header for the shared Uniform/AMR contract (which fields each runtime uses).
template <int Dim>
struct ProgramRuntimeState {
  static_assert(Dim >= 1 && Dim <= 3, "ProgramRuntimeState only supports dimensions 1, 2, and 3");
  /// Static field-boundary authoring image captured before the first successful artifact overlay.
  std::optional<ArtifactFieldBoundaryAuthorityRegistry<Dim>> artifact_field_boundary_baseline_;
  /// Candidate sink active only while pops_install_field_boundaries executes.
  std::optional<ArtifactFieldBoundaryStage<Dim>> artifact_field_boundary_stage_;
  // --- fields read by the stepper (the ONLY Program state the stepper sees) -------------------------
  /// Installed macro-step body (ADC-399); empty makes every public facade temporal operation fail
  /// before mutation.
  std::function<void(double)> step_;
  /// AMR-only accepted-state requalification hook installed by the same generated artifact as
  /// `step_`. Explicit bootstrap grows the hierarchy after the Program context is constructed; this
  /// closure lets the facade ask that persistent context to republish its level-qualified clocks and
  /// histories before committing each hierarchy transition. Uniform leaves it empty.
  std::function<void()> hierarchy_refresh_;
  /// AMR-only, artifact-owned remap boundary. Unlike hierarchy_refresh_, this callback is reached
  /// only after AmrSystem published a topology and atomically exchanged a prepared history manager.
  /// Keeping it distinct prevents a generic hierarchy refresh from accepting stale history storage.
  std::function<void()> history_remap_accepted_;
  /// Artifact-owned accepted-boundary hooks used only by the strict AMR restart transaction.
  /// `restart_regrid_preflight_` validates every rank-local prerequisite before peers enter the
  /// scientific regrid; `restart_regrid_` then performs that tag/regrid pass;
  /// `restart_resync_` force-imports the facade bytes after rollback, even when their restored
  /// revision equals the context's last observed revision. `accepted_context_snapshot_` contributes
  /// context-owned ledgers and clocks to every outer accepted transaction. Uniform leaves all four
  /// empty.
  std::function<void()> restart_regrid_preflight_;
  std::function<void()> restart_regrid_;
  std::function<void()> restart_resync_;
  AcceptedProgramContextSnapshotFactory accepted_context_snapshot_;
  /// Monotone witness incremented only by install_unverified_step. Dynamic artifact loaders use it to
  /// prove that one installer invocation actually replaced the whole-system Program step.
  std::uint64_t step_install_generation_ = 0;
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
  /// Transient non-reentrancy lease for the one shared cadence dispatcher. It is neither checkpoint
  /// state nor accepted scientific state and is always released by RAII on success or failure.
  bool cadence_dispatch_active_ = false;
  /// A strict checkpoint restore stages, but does not yet install, one authenticated window. The
  /// subsequent set_clock must present the exact accepted (time, macro-step) pair that validated the
  /// staged image; only that call commits the window. A mismatch discards the staged transaction and
  /// leaves the accepted cadence image unchanged.
  bool cadence_clock_restore_pending_ = false;
  double cadence_clock_restore_dt_ = 0.0;
  int cadence_clock_restore_steps_ = 0;
  double cadence_clock_restore_start_time_ = 0.0;
  double cadence_clock_restore_last_dt_ = 0.0;
  double cadence_clock_restore_accepted_time_ = 0.0;
  int cadence_clock_restore_macro_step_ = 0;
  /// LAST accepted numerical interval handed to step_ (ADC-626). Set by the driver right before each
  /// program_.step_(h) call (dispatch_cadence_step, shared by both runtimes and their explicit/CFL
  /// entry points), so the runtime's
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
  /// Complete AMR checkpoint shape exported by the frozen DSO.  Unlike the live history manager it
  /// exists before Program prelude allocation and therefore bounds every configured future level.
  ProgramCheckpointMetadata checkpoint_metadata_;
  /// True only after install_program has authenticated an artifact and its replay-authority table.
  /// Direct C++ install_program_step remains a low-level composition seam for ordinary runtime tests,
  /// but it is never authority to recompute omitted checkpoint history.
  bool artifact_backed_ = false;
  /// NAME-based block binding (ADC-457): program-index -> runtime-block-index map. Entry p holds the
  /// runtime block index the Program's block p names. EMPTY means no authenticated mapping; positional
  /// identity is not inferred. Used by BOTH runtimes; read by the (Amr)ProgramContext.
  std::vector<int> block_map_;

  // --- runtime data owned across the step closure --------------------------------------------------
  /// COMPILED-PROGRAM SCALAR DIAGNOSTICS (ADC-414): name -> last value recorded via P.record_scalar.
  /// Lives here (not the .so) so it outlives the step closure and Python can read it. Used by BOTH.
  std::map<std::string, Real> diagnostics_;
  /// Reserved balance records for the current native attempt only. Unlike diagnostics_, this
  /// mailbox is cleared before every public step and is never checkpointed. Accepted balance
  /// consumers read it while the facade's outer transaction still retains U^n, so a missing term
  /// cannot silently reuse the preceding step.
  std::map<std::string, Real> step_balance_terms_;
  /// Native operator contributions captured only for a due Balance attempt. These values are keyed
  /// by their physical runtime coordinate instead of a user ledger route and are therefore not read
  /// by accepted_balance_terms(). The owning facade snapshots this map with the rest of the attempt,
  /// so rejection cannot leak automatic evidence into a retry.
  std::map<AutomaticBalanceKey, Real> automatic_balance_terms_;
  /// Monotone attempt-local decision emitted by generated code before any Program operator runs.
  /// It is the OR of the exact ConsumerGraph-derived route decisions for this public step. Keeping
  /// this separate from step_balance_terms_ lets projection operators execute before their later
  /// Program.record_balance sinks without losing due automatic evidence.
  bool automatic_balance_due_ = false;
  /// Attempt-local outer accepted-step target used by ConsumerGraph-fused balance guards. Program
  /// substeps temporarily publish their window-start macro step through the facade, so generated
  /// balance code must not infer the public target from `macro_step()+1`.
  bool balance_due_window_active_ = false;
  int balance_due_target_step_ = 0;
  /// Selective checkpoint reconstruction re-executes scientific Program code without accepting a
  /// public step. Balance evidence is therefore compiled off for that replay: it must neither query
  /// a nonexistent public-step due window nor populate the current accepted-attempt mailbox.
  bool balance_replay_active_ = false;
  /// A stride-held public step executes no Program work, so its exact discrete balance is the
  /// additive identity for every route. These transient flags distinguish that valid zero from a
  /// due Program that failed to publish all five terms; neither flag is checkpoint state.
  bool balance_step_completed_ = false;
  bool balance_program_was_due_ = false;
  /// Attempt-local identities of ProjectAndRecheck branches that actually executed. This report
  /// mailbox is cleared at attempt entry and consumed by the Python transaction coordinator before
  /// commit or rollback; it is deliberately not checkpoint or accepted scientific state.
  std::vector<std::string> step_projections_;
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
  CacheManager<Dim> cache_;
  /// MULTISTEP HISTORY (ADC-406a), UNIFORM ONLY. Ring buffers for multistep schemes; the uniform
  /// checkpoint serializes them. Empty on AMR (history seam not wired).
  HistoryManager<Dim> hist_;

  // --- self-contained helpers (grid-free, Program-subsystem-worded errors) -------------------------

  struct ArtifactStepInstallSnapshot {
    std::function<void(double)> step;
    std::function<void()> hierarchy_refresh;
    std::function<void()> history_remap_accepted;
    std::function<void()> restart_regrid_preflight;
    std::function<void()> restart_regrid;
    std::function<void()> restart_resync;
    AcceptedProgramContextSnapshotFactory accepted_context_snapshot;
    std::function<Real(Real)> dt_bound;
    std::uint64_t generation = 0;
    std::string installed_hash;
    std::vector<std::array<std::uint64_t, 4>> operator_authorities;
    std::vector<std::pair<std::string, int>> history_replay_authorities;
    ProgramCheckpointMetadata checkpoint_metadata;
    std::vector<int> block_map;
    std::map<int, RuntimeParams> block_params;
    std::map<std::string, Real> diagnostics;
    CacheManager<Dim> cache;
    HistoryManager<Dim> history;
    bool artifact_backed = false;
  };

  /// Complete accepted-state rollback candidate. Static artifact closures and installation
  /// authorities deliberately remain live; only state that an accepted attempt may mutate is
  /// copied. The profiler candidate retains the live profiler lock from preparation to publication.
  class PreparedProgramAcceptedRestore {
   public:
    PreparedProgramAcceptedRestore(const PreparedProgramAcceptedRestore&) = delete;
    PreparedProgramAcceptedRestore& operator=(const PreparedProgramAcceptedRestore&) = delete;
    PreparedProgramAcceptedRestore(PreparedProgramAcceptedRestore&&) noexcept = default;
    PreparedProgramAcceptedRestore& operator=(PreparedProgramAcceptedRestore&&) noexcept = default;

   private:
    friend struct ProgramRuntimeState;

    PreparedProgramAcceptedRestore(ProgramRuntimeState& owner, const ProgramRuntimeState& accepted)
        : owner_(&owner),
          cadence_window_dt_(accepted.cadence_window_dt_),
          cadence_window_steps_(accepted.cadence_window_steps_),
          cadence_window_start_time_(accepted.cadence_window_start_time_),
          cadence_dispatch_active_(accepted.cadence_dispatch_active_),
          cadence_clock_restore_pending_(accepted.cadence_clock_restore_pending_),
          cadence_clock_restore_dt_(accepted.cadence_clock_restore_dt_),
          cadence_clock_restore_steps_(accepted.cadence_clock_restore_steps_),
          cadence_clock_restore_start_time_(accepted.cadence_clock_restore_start_time_),
          cadence_clock_restore_last_dt_(accepted.cadence_clock_restore_last_dt_),
          cadence_clock_restore_accepted_time_(accepted.cadence_clock_restore_accepted_time_),
          cadence_clock_restore_macro_step_(accepted.cadence_clock_restore_macro_step_),
          last_dt_(accepted.last_dt_),
          diagnostics_(accepted.diagnostics_),
          step_balance_terms_(accepted.step_balance_terms_),
          automatic_balance_terms_(accepted.automatic_balance_terms_),
          automatic_balance_due_(accepted.automatic_balance_due_),
          balance_due_window_active_(accepted.balance_due_window_active_),
          balance_due_target_step_(accepted.balance_due_target_step_),
          balance_replay_active_(accepted.balance_replay_active_),
          balance_step_completed_(accepted.balance_step_completed_),
          balance_program_was_due_(accepted.balance_program_was_due_),
          block_params_(accepted.block_params_),
          cache_(accepted.cache_),
          hist_(accepted.hist_),
          profiler_restore_(owner.profiler_.prepare_restore(accepted.profiler_)) {}

    ProgramRuntimeState* owner_ = nullptr;
    double cadence_window_dt_ = 0.0;
    int cadence_window_steps_ = 0;
    double cadence_window_start_time_ = 0.0;
    bool cadence_dispatch_active_ = false;
    bool cadence_clock_restore_pending_ = false;
    double cadence_clock_restore_dt_ = 0.0;
    int cadence_clock_restore_steps_ = 0;
    double cadence_clock_restore_start_time_ = 0.0;
    double cadence_clock_restore_last_dt_ = 0.0;
    double cadence_clock_restore_accepted_time_ = 0.0;
    int cadence_clock_restore_macro_step_ = 0;
    Real last_dt_ = Real(0);
    std::map<std::string, Real> diagnostics_;
    std::map<std::string, Real> step_balance_terms_;
    std::map<AutomaticBalanceKey, Real> automatic_balance_terms_;
    bool automatic_balance_due_ = false;
    bool balance_due_window_active_ = false;
    int balance_due_target_step_ = 0;
    bool balance_replay_active_ = false;
    bool balance_step_completed_ = false;
    bool balance_program_was_due_ = false;
    std::map<int, RuntimeParams> block_params_;
    CacheManager<Dim> cache_;
    HistoryManager<Dim> hist_;
    Profiler::PreparedRestore profiler_restore_;
  };

  PreparedProgramAcceptedRestore prepare_accepted_restore(const ProgramRuntimeState& accepted) {
    if (this == &accepted)
      throw std::invalid_argument(
          "Program accepted restore requires an independent accepted image");
    return PreparedProgramAcceptedRestore(*this, accepted);
  }

  void publish_prepared_accepted_restore(PreparedProgramAcceptedRestore&& prepared) noexcept {
    if (prepared.owner_ != this)
      std::terminate();
    static_assert(noexcept(diagnostics_.swap(prepared.diagnostics_)));
    static_assert(noexcept(step_balance_terms_.swap(prepared.step_balance_terms_)));
    static_assert(noexcept(automatic_balance_terms_.swap(prepared.automatic_balance_terms_)));
    static_assert(noexcept(block_params_.swap(prepared.block_params_)));
    static_assert(std::is_nothrow_swappable_v<CacheManager<Dim>>);
    static_assert(std::is_nothrow_swappable_v<HistoryManager<Dim>>);
    cadence_window_dt_ = prepared.cadence_window_dt_;
    cadence_window_steps_ = prepared.cadence_window_steps_;
    cadence_window_start_time_ = prepared.cadence_window_start_time_;
    cadence_dispatch_active_ = prepared.cadence_dispatch_active_;
    cadence_clock_restore_pending_ = prepared.cadence_clock_restore_pending_;
    cadence_clock_restore_dt_ = prepared.cadence_clock_restore_dt_;
    cadence_clock_restore_steps_ = prepared.cadence_clock_restore_steps_;
    cadence_clock_restore_start_time_ = prepared.cadence_clock_restore_start_time_;
    cadence_clock_restore_last_dt_ = prepared.cadence_clock_restore_last_dt_;
    cadence_clock_restore_accepted_time_ = prepared.cadence_clock_restore_accepted_time_;
    cadence_clock_restore_macro_step_ = prepared.cadence_clock_restore_macro_step_;
    last_dt_ = prepared.last_dt_;
    diagnostics_.swap(prepared.diagnostics_);
    step_balance_terms_.swap(prepared.step_balance_terms_);
    automatic_balance_terms_.swap(prepared.automatic_balance_terms_);
    automatic_balance_due_ = prepared.automatic_balance_due_;
    balance_due_window_active_ = prepared.balance_due_window_active_;
    balance_due_target_step_ = prepared.balance_due_target_step_;
    balance_replay_active_ = prepared.balance_replay_active_;
    balance_step_completed_ = prepared.balance_step_completed_;
    balance_program_was_due_ = prepared.balance_program_was_due_;
    block_params_.swap(prepared.block_params_);
    std::swap(cache_, prepared.cache_);
    std::swap(hist_, prepared.hist_);
    profiler_.publish_prepared_restore(std::move(prepared.profiler_restore_));
  }

  const std::vector<int>& block_map() const noexcept { return block_map_; }

  Profiler& profiler() noexcept { return profiler_; }

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

  /// Install an ordinary native step without granting artifact-only replay authority.
  ///
  /// The dynamic Program loader calls the same facade seam while its artifact install entry runs,
  /// then publishes the validated `(ring, depth)` table and marks the completed install artifact
  /// backed. A direct caller never reaches that completion step. Replacing an artifact-backed step
  /// therefore revokes every artifact-derived authority before the new closure becomes usable.
  void install_unverified_step(std::function<void(double)> step) {
    if (!step)
      throw std::invalid_argument("Program install requires a non-empty whole-system step");
    if (step_install_generation_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("Program step-install generation overflow");
    step_ = std::move(step);
    hierarchy_refresh_ = nullptr;
    history_remap_accepted_ = nullptr;
    restart_regrid_preflight_ = nullptr;
    restart_regrid_ = nullptr;
    restart_resync_ = nullptr;
    accepted_context_snapshot_ = nullptr;
    dt_bound_ = nullptr;
    installed_hash_.clear();
    operator_authorities_.clear();
    history_replay_authorities_.clear();
    checkpoint_metadata_ = {};
    block_map_.clear();
    block_params_.clear();
    artifact_backed_ = false;
    ++step_install_generation_;
  }

  ArtifactStepInstallSnapshot capture_artifact_step_install() const {
    return ArtifactStepInstallSnapshot{step_,
                                       hierarchy_refresh_,
                                       history_remap_accepted_,
                                       restart_regrid_preflight_,
                                       restart_regrid_,
                                       restart_resync_,
                                       accepted_context_snapshot_,
                                       dt_bound_,
                                       step_install_generation_,
                                       installed_hash_,
                                       operator_authorities_,
                                       history_replay_authorities_,
                                       checkpoint_metadata_,
                                       block_map_,
                                       block_params_,
                                       diagnostics_,
                                       cache_,
                                       hist_,
                                       artifact_backed_};
  }

  /// Start one isolated artifact candidate after its complete rollback image has been captured.
  /// Histories, scheduled values and diagnostics are qualified by the installed Program: retaining
  /// any of them across A -> B could either expose removed names or let coincident node/ring ids read
  /// values produced by A. The dynamic loaders call this before the generated prelude; a failed
  /// candidate restores the snapshot above.
  void reset_artifact_candidate_state() {
    diagnostics_.clear();
    cache_.clear();
    hist_ = HistoryManager<Dim>{};
  }

  void rollback_artifact_step_install(ArtifactStepInstallSnapshot&& snapshot) noexcept {
    step_ = std::move(snapshot.step);
    hierarchy_refresh_ = std::move(snapshot.hierarchy_refresh);
    history_remap_accepted_ = std::move(snapshot.history_remap_accepted);
    restart_regrid_preflight_ = std::move(snapshot.restart_regrid_preflight);
    restart_regrid_ = std::move(snapshot.restart_regrid);
    restart_resync_ = std::move(snapshot.restart_resync);
    accepted_context_snapshot_ = std::move(snapshot.accepted_context_snapshot);
    dt_bound_ = std::move(snapshot.dt_bound);
    step_install_generation_ = snapshot.generation;
    installed_hash_ = std::move(snapshot.installed_hash);
    operator_authorities_ = std::move(snapshot.operator_authorities);
    history_replay_authorities_ = std::move(snapshot.history_replay_authorities);
    checkpoint_metadata_ = std::move(snapshot.checkpoint_metadata);
    block_map_ = std::move(snapshot.block_map);
    block_params_ = std::move(snapshot.block_params);
    diagnostics_ = std::move(snapshot.diagnostics);
    cache_ = std::move(snapshot.cache);
    hist_ = std::move(snapshot.history);
    artifact_backed_ = snapshot.artifact_backed;
  }

  void require_exact_artifact_step_install(const ArtifactStepInstallSnapshot& before,
                                           const std::string& runtime) const {
    if (!step_ || before.generation == std::numeric_limits<std::uint64_t>::max() ||
        step_install_generation_ != before.generation + 1 || artifact_backed_)
      throw std::runtime_error(runtime +
                               " artifact installer must install exactly one new unverified "
                               "whole-system Program step");
  }

  /// Attach the hierarchy callback emitted beside the installed AMR Program step. It remains part
  /// of the artifact-install rollback image so no DSO-backed closure survives a failed install.
  void install_hierarchy_refresh(std::function<void()> refresh, const std::string& runtime) {
    if (!step_)
      throw std::logic_error(runtime +
                             "::install_program_hierarchy_refresh requires an installed Program");
    if (!refresh)
      throw std::invalid_argument(runtime +
                                  "::install_program_hierarchy_refresh requires a non-empty hook");
    hierarchy_refresh_ = std::move(refresh);
  }

  /// Attach the exact post-publication history-remap callback emitted beside an AMR Program.
  void install_history_remap_accepted(std::function<void()> refresh, const std::string& runtime) {
    if (!step_)
      throw std::logic_error(
          runtime + "::install_program_history_remap_accepted requires an installed Program");
    if (!refresh)
      throw std::invalid_argument(
          runtime + "::install_program_history_remap_accepted requires a non-empty hook");
    history_remap_accepted_ = std::move(refresh);
  }

  /// Attach the restart-only callbacks emitted by the same authenticated AMR artifact.
  /// They participate in artifact-install rollback, so a failed DSO candidate cannot leave a
  /// callable stale context behind.
  void install_restart_hooks(std::function<void()> preflight, std::function<void()> regrid,
                             std::function<void()> resync,
                             AcceptedProgramContextSnapshotFactory accepted_context_snapshot,
                             const std::string& runtime) {
    if (!step_)
      throw std::logic_error(runtime +
                             "::install_program_restart_hooks requires an installed Program");
    if (!preflight || !regrid || !resync || !accepted_context_snapshot)
      throw std::invalid_argument(
          runtime +
          "::install_program_restart_hooks requires complete accepted-state "
          "hooks");
    restart_regrid_preflight_ = std::move(preflight);
    restart_regrid_ = std::move(regrid);
    restart_resync_ = std::move(resync);
    accepted_context_snapshot_ = std::move(accepted_context_snapshot);
  }

  std::unique_ptr<AcceptedProgramContextSnapshot> capture_accepted_context_snapshot(
      const std::string& runtime) const {
    if (!artifact_backed_)
      return {};
    if (!accepted_context_snapshot_)
      throw std::logic_error(runtime + " artifact lacks its accepted context snapshot hook");
    std::unique_ptr<AcceptedProgramContextSnapshot> snapshot = accepted_context_snapshot_();
    if (!snapshot)
      throw std::logic_error(runtime + " artifact returned an empty accepted context snapshot");
    return snapshot;
  }

  void preflight_regrid_on_restart(const std::string& runtime) const {
    if (!artifact_backed_)
      throw std::logic_error(runtime +
                             " RegridOnRestart requires an authenticated artifact-backed Program");
    if (!restart_regrid_preflight_ || !restart_regrid_ || !restart_resync_ ||
        !accepted_context_snapshot_)
      throw std::logic_error(runtime + " artifact lacks its restart preflight/regrid/resync hooks");
    restart_regrid_preflight_();
  }

  void regrid_on_restart(const std::string& runtime) const {
    if (!artifact_backed_ || !restart_regrid_preflight_ || !restart_regrid_ || !restart_resync_ ||
        !accepted_context_snapshot_)
      throw std::logic_error(runtime + " artifact lacks its prepared restart regrid authority");
    restart_regrid_();
  }

  void resync_after_restart_rollback(const std::string& runtime) const {
    if (!artifact_backed_)
      return;
    if (!restart_resync_)
      throw std::logic_error(runtime + " artifact lacks its restart rollback resync hook");
    restart_resync_();
  }

  /// Requalify Program-owned accepted state for the hierarchy currently exposed by the AMR engine.
  /// Direct low-level C++ steps have no artifact context and intentionally remain a no-op; an
  /// authenticated artifact must always provide the hook.
  void refresh_hierarchy_state(const std::string& runtime) const {
    if (!hierarchy_refresh_) {
      if (artifact_backed_)
        throw std::logic_error(runtime +
                               " artifact lacks its accepted-state hierarchy refresh hook");
      return;
    }
    hierarchy_refresh_();
  }

  /// Accept only the engine-owned prepared-history remap publication, never a generic refresh.
  void accept_history_remap(const std::string& runtime) const {
    if (!history_remap_accepted_)
      throw std::logic_error(runtime + " artifact lacks its accepted history-remap hook");
    history_remap_accepted_();
  }

  bool authorizes_history_replay(const std::string& ring, int depth) const {
    return artifact_backed_ &&
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

  /// Validate one exact held-window image against an accepted facade macro-step without mutating the
  /// live cadence state. This is also the preflight used by strict checkpoint restore.
  void validate_cadence_window_image(double accumulated_dt, int held_steps,
                                     double window_start_time, int macro_step,
                                     const std::string& runtime) const {
    if (macro_step < 0)
      throw std::invalid_argument(runtime + " Program cadence requires macro_step >= 0");
    if (stride_ < 1 || substeps_ < 1)
      throw std::logic_error(runtime + " Program cadence configuration is invalid");
    if (held_steps < 0 || held_steps >= stride_)
      throw std::runtime_error(runtime + " Program cadence window has an invalid held-step count");
    if (held_steps != macro_step % stride_)
      throw std::runtime_error(
          runtime +
          " Program cadence window phase differs from the authoritative macro-step; restore the "
          "strict cadence-window checkpoint state before set_clock");
    if (!std::isfinite(accumulated_dt) || accumulated_dt < 0.0)
      throw std::runtime_error(runtime +
                               " Program cadence window duration must be finite and non-negative");
    if (!std::isfinite(window_start_time))
      throw std::runtime_error(runtime + " Program cadence window start time must be finite");
    if ((held_steps == 0) != (accumulated_dt == 0.0))
      throw std::runtime_error(
          runtime +
          " Program cadence window duration/count are inconsistent (zero iff no step is held)");
    if (held_steps == 0 && window_start_time != 0.0)
      throw std::runtime_error(
          runtime + " Program cadence inactive window must use the canonical zero start time");
  }

  /// Validate an image's exact accepted physical cursor without mutating the live cadence state.
  void validate_cadence_window_time_image(int held_steps, double window_start_time,
                                          double accepted_time, const std::string& runtime) const {
    if (!std::isfinite(accepted_time))
      throw std::invalid_argument(runtime + " Program cadence requires a finite accepted time");
    if (held_steps != 0 && !(window_start_time < accepted_time))
      throw std::runtime_error(
          runtime + " Program cadence active window start must precede the accepted physical time");
  }

  /// Validate the currently accepted held-window image.
  void validate_cadence_window(int macro_step, const std::string& runtime) const {
    validate_cadence_window_image(cadence_window_dt_, cadence_window_steps_,
                                  cadence_window_start_time_, macro_step, runtime);
  }

  /// Bind the currently accepted held window to the physical cursor that owns it.
  void validate_cadence_window_time(double accepted_time, const std::string& runtime) const {
    validate_cadence_window_time_image(cadence_window_steps_, cadence_window_start_time_,
                                       accepted_time, runtime);
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

  /// Execute one accepted facade step through the single Uniform/AMR cadence dispatcher.
  ///
  /// The owning runtime lends its exact accepted cursor by reference. The dispatcher publishes each
  /// numerical substep's start coordinate while invoking the installed Program, restores the entry
  /// cursor after every failure, commits the held/due cadence image once, then advances the public
  /// cursor exactly once. Grid and hierarchy work remain inside the installed provider closure.
  void dispatch_cadence_step(double& physical_time_cursor, int& macro_step_cursor, double dt,
                             const std::string& runtime) {
    if (cadence_dispatch_active_)
      throw std::logic_error(runtime + " Program cadence dispatch is non-reentrant");
    if (!step_)
      throw std::logic_error(
          runtime + " Program cadence dispatch requires an installed whole-system Program");

    cadence_dispatch_active_ = true;
    struct CadenceDispatchLease {
      bool& active;
      ~CadenceDispatchLease() { active = false; }
    } dispatch_lease{cadence_dispatch_active_};

    const double accepted_time = physical_time_cursor;
    const int accepted_macro_step = macro_step_cursor;
    const PreparedCadenceStep cadence =
        prepare_cadence_step(accepted_time, accepted_macro_step, dt, runtime);
    if (accepted_macro_step == std::numeric_limits<int>::max())
      throw std::overflow_error(runtime + " Program cadence macro-step counter overflow");

    try {
      if (cadence.due) {
        validate_cadence_partition(cadence, substeps_, runtime);
        const int held_before_due = cadence.window_steps - 1;
        if (accepted_macro_step < held_before_due)
          throw std::logic_error(runtime + " Program cadence window starts before macro-step zero");
        const int window_start_macro_step = accepted_macro_step - held_before_due;
        run_balance_due_window(accepted_macro_step, runtime, [&] {
          for (int substep = 0; substep < substeps_; ++substep) {
            const PreparedCadenceSubstep partition =
                prepare_cadence_substep(cadence, substep, substeps_, runtime);
            physical_time_cursor = partition.start;
            macro_step_cursor = window_start_macro_step;
            last_dt_ = static_cast<Real>(partition.dt);
            step_(partition.dt);
            physical_time_cursor = partition.end;
          }
        });
        physical_time_cursor = accepted_time;
        macro_step_cursor = accepted_macro_step;
      }

      commit_cadence_step(cadence, runtime);
      physical_time_cursor = cadence.window_end;
      complete_balance_step(cadence.due);
      ++macro_step_cursor;
    } catch (...) {
      physical_time_cursor = accepted_time;
      macro_step_cursor = accepted_macro_step;
      throw;
    }
  }

  /// Stage an authenticated checkpoint window for one exact set_clock transaction. The accepted
  /// window is not mutated until the matching clock pair is consumed, and no historical duration is
  /// guessed.
  void restore_cadence_window(double accumulated_dt, int held_steps, double window_start_time,
                              double accepted_last_dt, double accepted_time, int macro_step,
                              const std::string& runtime) {
    if (cadence_clock_restore_pending_)
      throw std::logic_error(runtime +
                             " Program cadence already has a pending clock-restore transaction");
    validate_cadence_window_image(accumulated_dt, held_steps, window_start_time, macro_step,
                                  runtime);
    validate_cadence_window_time_image(held_steps, window_start_time, accepted_time, runtime);
    if (!std::isfinite(accepted_last_dt) || accepted_last_dt < 0.0)
      throw std::runtime_error(runtime +
                               " Program accepted last dt must be finite and non-negative");
    const Real native_last_dt = static_cast<Real>(accepted_last_dt);
    if (!std::isfinite(static_cast<double>(native_last_dt)) ||
        static_cast<double>(native_last_dt) != accepted_last_dt)
      throw std::runtime_error(runtime +
                               " Program accepted last dt is not exactly representable by the "
                               "runtime precision");
    cadence_clock_restore_dt_ = accumulated_dt;
    cadence_clock_restore_steps_ = held_steps;
    cadence_clock_restore_start_time_ = window_start_time;
    cadence_clock_restore_last_dt_ = accepted_last_dt;
    cadence_clock_restore_accepted_time_ = accepted_time;
    cadence_clock_restore_macro_step_ = macro_step;
    cadence_clock_restore_pending_ = true;
  }

  /// Discard a staged restore without touching the accepted cadence image. Facade set_clock calls
  /// this on every validation failure so a bad clock cannot strand a pending transaction.
  void cancel_cadence_clock_restore() noexcept {
    cadence_clock_restore_pending_ = false;
    cadence_clock_restore_dt_ = 0.0;
    cadence_clock_restore_steps_ = 0;
    cadence_clock_restore_start_time_ = 0.0;
    cadence_clock_restore_last_dt_ = 0.0;
    cadence_clock_restore_accepted_time_ = 0.0;
    cadence_clock_restore_macro_step_ = 0;
  }

  /// Authenticate/consume the cadence image for a facade clock restore. At a clean stride boundary,
  /// direct set_clock remains valid; a mid-window cursor always requires the explicit checkpoint image.
  void consume_cadence_clock_restore(double accepted_time, int macro_step,
                                     const std::string& runtime) {
    if (cadence_clock_restore_pending_) {
      if (macro_step != cadence_clock_restore_macro_step_ ||
          std::bit_cast<std::uint64_t>(accepted_time) !=
              std::bit_cast<std::uint64_t>(cadence_clock_restore_accepted_time_)) {
        cancel_cadence_clock_restore();
        throw std::runtime_error(runtime +
                                 " set_clock accepted time or macro-step differs from the restored "
                                 "cadence window");
      }
      try {
        validate_cadence_window_image(cadence_clock_restore_dt_, cadence_clock_restore_steps_,
                                      cadence_clock_restore_start_time_,
                                      cadence_clock_restore_macro_step_, runtime);
        validate_cadence_window_time_image(cadence_clock_restore_steps_,
                                           cadence_clock_restore_start_time_,
                                           cadence_clock_restore_accepted_time_, runtime);
      } catch (...) {
        cancel_cadence_clock_restore();
        throw;
      }
      cadence_window_dt_ = cadence_clock_restore_dt_;
      cadence_window_steps_ = cadence_clock_restore_steps_;
      cadence_window_start_time_ = cadence_clock_restore_start_time_;
      last_dt_ = static_cast<Real>(cadence_clock_restore_last_dt_);
      cancel_cadence_clock_restore();
      return;
    }
    validate_cadence_window(macro_step, runtime);
    validate_cadence_window_time(accepted_time, runtime);
    if (cadence_window_steps_ != 0)
      throw std::runtime_error(
          runtime +
          " set_clock cannot reuse an active stride window; restore its strict checkpoint image");
  }

  static bool has_reserved_balance_namespace(const std::string& name) noexcept {
    return name.rfind("pops.balance-term", 0) == 0;
  }

  static void require_balance_route(const std::string& route, const std::string& runtime) {
    static constexpr std::string_view kRoutePrefix = "pops.balance-ledger-route.v1:sha256:";
    if (route.size() != kRoutePrefix.size() + 64 ||
        route.compare(0, kRoutePrefix.size(), kRoutePrefix.data(), kRoutePrefix.size()) != 0 ||
        !std::all_of(route.begin() + static_cast<std::ptrdiff_t>(kRoutePrefix.size()), route.end(),
                     [](unsigned char value) {
                       return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
                     }))
      throw std::invalid_argument(runtime + " requires a canonical balance-ledger-route identity");
  }

  static void require_balance_due_contract(const std::string& contract,
                                           const std::string& runtime) {
    static constexpr std::string_view kContractPrefix = "pops.balance-due-contract.v1:sha256:";
    if (contract.size() != kContractPrefix.size() + 64 ||
        contract.compare(0, kContractPrefix.size(), kContractPrefix.data(),
                         kContractPrefix.size()) != 0 ||
        !std::all_of(contract.begin() + static_cast<std::ptrdiff_t>(kContractPrefix.size()),
                     contract.end(), [](unsigned char value) {
                       return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
                     }))
      throw std::invalid_argument(runtime + " requires a canonical balance-due-contract identity");
  }

  static void require_balance_term(const std::string& term, const std::string& runtime) {
    static constexpr std::array<std::string_view, 5> kTerms{
        "storage_change", "outward_boundary_flux", "sources", "reflux", "projection"};
    if (std::find(kTerms.begin(), kTerms.end(), std::string_view(term)) == kTerms.end())
      throw std::invalid_argument(runtime + " requires one canonical five-term balance name");
  }

  static void require_automatic_balance_term(const std::string& term, const std::string& runtime) {
    static constexpr std::array<std::string_view, 4> kTerms{"outward_boundary_flux", "sources",
                                                            "reflux", "projection"};
    if (std::find(kTerms.begin(), kTerms.end(), std::string_view(term)) == kTerms.end())
      throw std::invalid_argument(runtime +
                                  " requires one native operator balance contribution name");
  }

  /// Record a compiled-Program scalar. Ordinary P.record_scalar names remain inspectable after the
  /// step with last-write-wins semantics. The balance namespace has a separate typed sink.
  void record_diagnostic(const std::string& name, Real value) {
    if (has_reserved_balance_namespace(name))
      throw std::invalid_argument(
          "ProgramRuntimeState::record_diagnostic: pops.balance-term is a reserved namespace");
    diagnostics_[name] = value;
  }

  /// Record one validated Program.record_balance term. Not exposed through the Python runtime
  /// facade: only generated ProgramContext code reaches this sink.
  void record_balance_term(const std::string& route, const std::string& term, Real value,
                           const std::string& runtime) {
    require_balance_route(route, runtime + "::record_balance_term");
    require_balance_term(term, runtime + "::record_balance_term");
    if (!std::isfinite(static_cast<double>(value)))
      throw std::invalid_argument(runtime + "::record_balance_term requires a finite value");
    const std::string name = "pops.balance-term.v1:" + route + ":" + term;
    // A Program cadence may invoke the compiled body several times inside one public macro-step.
    // Terms are signed, time-integrated increments and therefore accumulate across invocations.
    auto [entry, inserted] = step_balance_terms_.try_emplace(name, value);
    if (!inserted)
      entry->second += value;
  }

  /// Whether generated code proved that at least one Balance route is due in this attempt.
  ///
  /// The exact ConsumerGraph-derived decision is emitted before any Program operator, so both an
  /// in-body projection and post-body reflux observe the same cadence without a second scheduler.
  [[nodiscard]] bool automatic_balance_capture_due() const noexcept {
    return !balance_replay_active_ && automatic_balance_due_;
  }

  /// Publish one generated ConsumerGraph due decision before Program operators execute.
  ///
  /// Several compiled Program invocations may share one outer accepted-step window. The marker is
  /// therefore monotone inside an attempt and is reset only at attempt entry. Static-false routes
  /// emit no call, so a run without Balance consumers retains no generated hot-path branch.
  void note_automatic_balance_capture_due(bool due, const std::string& runtime) {
    if (balance_replay_active_) {
      if (due)
        throw std::logic_error(runtime +
                               "::note_automatic_balance_capture_due cannot enable replay capture");
      return;
    }
    if (!balance_due_window_active_)
      throw std::logic_error(
          runtime + "::note_automatic_balance_capture_due requires an active public-step window");
    automatic_balance_due_ = automatic_balance_due_ || due;
  }

  /// Accumulate one signed, metric-integrated native operator contribution.
  ///
  /// This is intentionally not accepted_balance_terms(): automatic evidence remains qualified by
  /// block/level/component until a resolved quantity selector proves which BalanceLedger route owns
  /// it. The separation is fail-closed and lets boundary/source/projection producers join the same
  /// mailbox later without fabricating missing terms.
  void record_automatic_balance_term(int runtime_block, int level, int component,
                                     const std::string& term, Real value,
                                     const std::string& runtime) {
    if (!automatic_balance_capture_due())
      throw std::logic_error(runtime +
                             "::record_automatic_balance_term requires a due authored balance");
    if (runtime_block < 0 || level < 0 || component < 0)
      throw std::invalid_argument(
          runtime + "::record_automatic_balance_term requires non-negative coordinates");
    require_automatic_balance_term(term, runtime + "::record_automatic_balance_term");
    if (!std::isfinite(static_cast<double>(value)))
      throw std::invalid_argument(runtime +
                                  "::record_automatic_balance_term requires a finite value");
    auto [entry, inserted] = automatic_balance_terms_.try_emplace(
        AutomaticBalanceKey{runtime_block, level, component, term}, value);
    if (!inserted)
      entry->second += value;
  }

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

  void begin_step_projection_report() {
    step_projections_.clear();
    step_balance_terms_.clear();
    automatic_balance_terms_.clear();
    automatic_balance_due_ = false;
    balance_due_window_active_ = false;
    balance_due_target_step_ = 0;
    balance_step_completed_ = false;
    balance_program_was_due_ = false;
  }

  void complete_balance_step(bool program_was_due) noexcept {
    balance_step_completed_ = true;
    balance_program_was_due_ = program_was_due;
  }

  /// Return exactly the five native Program scalars recorded for one typed balance route during the
  /// current attempt. The facade separately proves that an external accepted-step transaction is
  /// active. No zero, stale value, or array-derived Python fallback is permitted.
  std::map<std::string, Real> accepted_balance_terms(const std::string& route,
                                                     const std::string& runtime) const {
    static constexpr std::array<const char*, 5> kTerms{"storage_change", "outward_boundary_flux",
                                                       "sources", "reflux", "projection"};
    require_balance_route(route, runtime + "::_accepted_balance_terms");
    std::map<std::string, Real> result;
    if (step_balance_terms_.empty() && balance_step_completed_ && !balance_program_was_due_) {
      for (const char* term : kTerms)
        result.emplace(term, Real(0));
      return result;
    }
    for (const char* term : kTerms) {
      const std::string record = "pops.balance-term.v1:" + route + ":" + term;
      const auto found = step_balance_terms_.find(record);
      if (found == step_balance_terms_.end())
        throw std::runtime_error(
            runtime + "::_accepted_balance_terms: current native attempt omitted term '" + term +
            "'; Program.record_balance must publish all five terms");
      if (!std::isfinite(static_cast<double>(found->second)))
        throw std::runtime_error(
            runtime +
            "::_accepted_balance_terms: current native attempt produced non-finite term '" + term +
            "'");
      result.emplace(term, found->second);
    }
    return result;
  }

  /// Resolve one public Balance route against exact native operator coordinates.
  ///
  /// Explicit Program records remain authoritative for every term not listed in @p automatic_terms.
  /// Reflux and projection may instead be selected from the attempt-local native mailbox. The
  /// selector is complete and owner-qualified: one runtime block, one conservative component and
  /// the full active contiguous hierarchy. A selected producer must have published every expected
  /// coordinate; missing evidence and duplicate Program/native authority fail instead of becoming
  /// zero or reusing a stale value.
  std::map<std::string, Real> selected_accepted_balance_terms(
      const std::string& route, int runtime_block, int component, const std::vector<int>& levels,
      const std::vector<std::string>& automatic_terms, const std::string& runtime) const {
    static constexpr std::array<const char*, 5> kTerms{"storage_change", "outward_boundary_flux",
                                                       "sources", "reflux", "projection"};
    require_balance_route(route, runtime + "::_selected_accepted_balance_terms");
    if (runtime_block < 0 || component < 0)
      throw std::invalid_argument(
          runtime + "::_selected_accepted_balance_terms requires non-negative coordinates");
    if (levels.empty() || levels.front() < 0 ||
        std::adjacent_find(levels.begin(), levels.end(),
                           [](int left, int right) { return right != left + 1; }) != levels.end())
      throw std::invalid_argument(
          runtime + "::_selected_accepted_balance_terms requires a non-empty contiguous hierarchy");
    if (!std::is_sorted(automatic_terms.begin(), automatic_terms.end()) ||
        std::adjacent_find(automatic_terms.begin(), automatic_terms.end()) != automatic_terms.end())
      throw std::invalid_argument(
          runtime + "::_selected_accepted_balance_terms requires sorted unique automatic terms");
    for (const std::string& term : automatic_terms)
      if (term != "reflux" && term != "projection")
        throw std::invalid_argument(
            runtime + "::_selected_accepted_balance_terms has no native producer for '" + term +
            "'");

    std::map<std::string, Real> result;
    if (step_balance_terms_.empty() && balance_step_completed_ && !balance_program_was_due_) {
      for (const char* term : kTerms)
        result.emplace(term, Real(0));
      return result;
    }
    for (const char* term_value : kTerms) {
      const std::string term = term_value;
      const bool automatic =
          std::binary_search(automatic_terms.begin(), automatic_terms.end(), term);
      const std::string record = "pops.balance-term.v1:" + route + ":" + term;
      const auto authored = step_balance_terms_.find(record);
      if (!automatic) {
        if (authored == step_balance_terms_.end())
          throw std::runtime_error(
              runtime +
              "::_selected_accepted_balance_terms: current native attempt omitted term '" + term +
              "'; Program.record_balance must publish every non-automatic term");
        if (!std::isfinite(static_cast<double>(authored->second)))
          throw std::runtime_error(
              runtime +
              "::_selected_accepted_balance_terms: current native attempt produced "
              "non-finite term '" +
              term + "'");
        result.emplace(term, authored->second);
        continue;
      }
      if (authored != step_balance_terms_.end())
        throw std::runtime_error(runtime + "::_selected_accepted_balance_terms: term '" + term +
                                 "' has both Program and native producer authority");

      Real value = Real(0);
      const std::size_t expected = term == "reflux" ? levels.size() - 1 : levels.size();
      for (std::size_t index = 0; index < expected; ++index) {
        const AutomaticBalanceKey key{runtime_block, levels[index], component, term};
        const auto found = automatic_balance_terms_.find(key);
        if (found == automatic_balance_terms_.end())
          throw std::runtime_error(
              runtime + "::_selected_accepted_balance_terms: native producer omitted term '" +
              term + "' at level " + std::to_string(levels[index]));
        if (!std::isfinite(static_cast<double>(found->second)))
          throw std::runtime_error(
              runtime +
              "::_selected_accepted_balance_terms: native producer returned non-finite "
              "term '" +
              term + "'");
        value += found->second;
      }
      if (!std::isfinite(static_cast<double>(value)))
        throw std::runtime_error(
            runtime + "::_selected_accepted_balance_terms: native term accumulation overflowed");
      result.emplace(term, value);
    }
    return result;
  }

  void begin_balance_due_window(int accepted_macro_step, const std::string& runtime) {
    if (balance_due_window_active_)
      throw std::logic_error(runtime + " balance due window is already active");
    if (balance_replay_active_)
      throw std::logic_error(runtime + " cannot enter a public-step window during balance replay");
    if (accepted_macro_step < 0 || accepted_macro_step == std::numeric_limits<int>::max())
      throw std::overflow_error(runtime + " balance due target step is not representable");
    balance_due_target_step_ = accepted_macro_step + 1;
    balance_due_window_active_ = true;
  }

  void end_balance_due_window() noexcept {
    balance_due_window_active_ = false;
    balance_due_target_step_ = 0;
  }

  template <class Body>
  void run_balance_due_window(int accepted_macro_step, const std::string& runtime, Body&& body) {
    begin_balance_due_window(accepted_macro_step, runtime);
    try {
      std::forward<Body>(body)();
    } catch (...) {
      end_balance_due_window();
      throw;
    }
    end_balance_due_window();
  }

  template <class Body>
  void run_balance_replay(const std::string& runtime, Body&& body) {
    if (balance_replay_active_)
      throw std::logic_error(runtime + " balance replay is already active");
    if (balance_due_window_active_)
      throw std::logic_error(runtime + " cannot enter balance replay inside a public-step window");
    balance_replay_active_ = true;
    try {
      std::forward<Body>(body)();
    } catch (...) {
      balance_replay_active_ = false;
      throw;
    }
    balance_replay_active_ = false;
  }

  bool balance_consumer_is_due(const std::string& contract, const std::string& route, int every_n,
                               const std::string& runtime) const {
    require_balance_due_contract(contract, runtime + "::balance_consumer_is_due");
    require_balance_route(route, runtime + "::balance_consumer_is_due");
    if (every_n <= 0)
      throw std::invalid_argument(runtime + "::balance_consumer_is_due requires a positive period");
    if (balance_replay_active_)
      return false;
    if (!balance_due_window_active_ || balance_due_target_step_ <= 0)
      throw std::logic_error(runtime +
                             "::balance_consumer_is_due requires an active public-step window");
    return balance_due_target_step_ % every_n == 0;
  }

  void note_step_projection(const std::string& name) {
    if (name.empty())
      throw std::invalid_argument("Program step projection identity cannot be empty");
    if (std::find(step_projections_.begin(), step_projections_.end(), name) ==
        step_projections_.end())
      step_projections_.push_back(name);
  }

  std::vector<std::string> consume_step_projections() {
    std::vector<std::string> result;
    result.swap(step_projections_);
    return result;
  }

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
