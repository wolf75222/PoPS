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
// this boundary. Exact-ranked ProgramExecutionServices<Dim> remains the sole implementation of operations
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
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <pops/core/foundation/types.hpp>  // Real
#include <pops/mesh/storage/multifab.hpp>  // MultiFab (history ring element)
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/numerics/time/amr/levels/amr_clock.hpp>
#include <pops/runtime/config/runtime_params.hpp>  // RuntimeParams, kMaxRuntimeParams
#include <pops/runtime/program/amr_program_checkpoint.hpp>
#include <pops/runtime/program/cache_manager.hpp>    // CacheManager (held-node scheduler cache)
#include <pops/runtime/program/module_metadata.hpp>  // frozen checkpoint-shape metadata
#include <pops/runtime/program/owned_program_installation.hpp>
#include <pops/runtime/program/program_persistent_value_store.hpp>
#include <pops/runtime/program/profiler.hpp>  // Profiler (per-node / per-brick timing)

namespace pops::runtime::program {

enum class AmrProgramHistoryRemapSource : std::uint8_t {
  RetainedChild = 1,
  ParentDeferred = 2,
  Removed = 3,
};

/// One canonical affected-ring decision prepared by the AMR lane before topology publication.
struct AmrProgramHistoryRemapEntry {
  std::string key;
  std::string parent_key;
  AmrProgramHistoryRemapSource source = AmrProgramHistoryRemapSource::Removed;

  bool operator==(const AmrProgramHistoryRemapEntry&) const = default;
};

/// Native-prepared authority for the one accepted history-ring transition induced by a real AMR
/// topology publication.  This is deliberately a value type: the artifact callback receives the
/// exact operation chosen by the owning AMR lane, rather than consulting mutable regrid state.
struct AmrProgramHistoryRemapDescriptor {
  int parent_level = -1;
  int child_level = -1;
  bool child_published = false;
  /// Geometry/ratio changed independently of a distribution-only ownership rebalance.  The
  /// accepted callback uses this exact engine-prepared fact to choose the same source for both
  /// numeric history slots and their FluxExpression provenance.
  bool child_physical_layout_changed = false;
  std::vector<AmrProgramHistoryRemapEntry> history_plan;
  /// Exact deferred-lag markers prepared from the rematerialized HistoryManager before the
  /// forward topology is published.  A detached accepted-context image deliberately cannot
  /// inspect the live history rings (depth, slot dt, or initialization); carrying the canonical
  /// result here lets it stage the same accepted marker map without a live callback or fallback.
  std::vector<AmrProgramPendingHistoryRemap> prepared_pending_history_remaps;
  std::uint64_t prior_topology_epoch = 0;
  std::uint64_t prior_materialization_generation = 0;
  std::uint64_t published_topology_epoch = 0;
  std::uint64_t published_materialization_generation = 0;
  std::int64_t accepted_macro_step = -1;
  std::int64_t temporal_numerator = 0;
  std::int64_t temporal_denominator = 0;
  bool integral_only = false;
  std::string operation_identity;

  bool operator==(const AmrProgramHistoryRemapDescriptor&) const = default;
};

/// Native-prepared authority for replacing every AMR history ring after a complete hierarchy
/// rebuild.  Unlike a parent/child regrid remap, this is a detached total replacement: each entry
/// carries the new exact key/level and the retained source provenance key for the same historical
/// identity.  It is intentionally a C++ runtime DTO only; no public facade or component ABI is
/// extended by this Candidate-time operation.
struct AmrProgramFullHistoryReseedEntry {
  std::string key;
  std::string source_key;
  std::string history_identity;
  int level = -1;

  bool operator==(const AmrProgramFullHistoryReseedEntry&) const = default;
};

struct AmrProgramFullHistoryReseedDescriptor {
  std::vector<AmrProgramFullHistoryReseedEntry> history_plan;
  std::uint64_t prior_topology_epoch = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t prior_materialization_generation = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t published_topology_epoch = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t published_materialization_generation = std::numeric_limits<std::uint64_t>::max();
  std::int64_t accepted_macro_step = -1;
  std::size_t level_count = 0;

  bool operator==(const AmrProgramFullHistoryReseedDescriptor&) const = default;
};

/// Fully value-owned topology authority needed to rebuild a cell-local temporal provider while
/// the next AMR hierarchy is still unpublished.  It deliberately carries only frozen forward
/// facts; a detached Program snapshot must never consult the last accepted adapter to recover a
/// layout, a lane, or an interface budget.
struct PreparedForwardAmrTemporalAuthority {
  std::uint64_t topology_epoch = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t materialization_generation = std::numeric_limits<std::uint64_t>::max();
  /// Candidate revision of the freshly serialized POPSAND5 accepted image.  The detached
  /// snapshot publishes this exact scalar with its pointer-only state exchange.
  std::uint64_t accepted_state_revision = std::numeric_limits<std::uint64_t>::max();
  std::string spatial_contract;
  std::string lane_identity;
  std::string collective_contract;
  std::vector<::pops::amr::ParentChildClockRelation> temporal_relations;
  std::size_t level_count = 0;
  std::size_t block_count = 0;
  /// Canonical ``block * level_count + level`` cell counts, derived from the staged hierarchy.
  std::vector<std::uint64_t> block_level_cell_counts;
  /// Canonical lower/upper entries for axes 0..Dim-1.
  std::vector<bool> periodic_faces;
  std::size_t coupling_count = 0;
  bool has_interface_flux_provider = false;
  std::string temporal_provider_identity;
  std::string flux_budget_contract;
  std::string coupling_contract;
  ::pops::amr::InterfaceFluxLedgerBudget interface_flux_ledger_budget;
};

/// Opaque, candidate-owned AMR field prototypes used to rebuild finite Program scratch arenas
/// against an unpublished forward topology.  The concrete carrier lives with the AMR adapter so
/// this type-erased boundary cannot expose an accepted facade or a live runtime fallback.
class PreparedForwardAmrScratchTopology {
 public:
  virtual ~PreparedForwardAmrScratchTopology() = default;
};

/// Type-erased, candidate-only portion of an AMR accepted checkpoint.  The regrid carrier owns
/// the forward hierarchy and rematerialized numeric HistoryManager; the detached Program snapshot
/// owns the complementary flux provenance, temporal authority and deferred markers.  Keeping this
/// split explicit prevents Candidate preparation from ever asking the last sealed facade to
/// serialize a checkpoint for an unpublished topology.
struct PreparedForwardAmrAcceptedContext {
  std::uint64_t topology_epoch = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t materialization_generation = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t accepted_state_revision = std::numeric_limits<std::uint64_t>::max();
  std::map<std::string, std::int64_t> logical_clock_ticks;
  std::vector<AmrProgramPendingHistoryRemap> pending_history_remaps;
  std::vector<std::uint8_t> history_flux_payload;
  CellTemporalPartitionAcceptedState temporal_partition;
  std::string flux_budget_contract;
  std::string coupling_contract;
  /// Face fragments and synchronization events carry patch addresses.  Forward preparation
  /// invalidates them and the carrier serializes empty collections until the next accepted step
  /// prepares new topology-qualified effects.
  bool topology_scoped_effects_invalidated = false;
};

/// Type-erased authority for a detached, topology-qualified AMR execution image.  This stays on
/// the C++ side of the v5 boundary: artifacts see it only through an accepted snapshot virtual
/// call, never through a new ABI callback or descriptor field.
class PreparedForwardAmrExecutionAuthority {
 public:
  PreparedForwardAmrExecutionAuthority() = default;
  PreparedForwardAmrExecutionAuthority(const PreparedForwardAmrExecutionAuthority&) = delete;
  PreparedForwardAmrExecutionAuthority& operator=(
      const PreparedForwardAmrExecutionAuthority&) = delete;
  virtual ~PreparedForwardAmrExecutionAuthority() = default;

  [[nodiscard]] virtual std::uint32_t native_dimension() const noexcept = 0;
  [[nodiscard]] virtual std::uint64_t topology_epoch() const noexcept = 0;
  [[nodiscard]] virtual std::uint64_t materialization_generation() const noexcept = 0;
  [[nodiscard]] virtual std::size_t configured_level_count() const noexcept { return 0; }
  [[nodiscard]] virtual std::size_t active_level_count() const noexcept { return 0; }
};

/// Type-erased, already allocated image of one artifact context's accepted state.  Runtime
/// transactions capture it before entering scientific code and invoke only the noexcept publication
/// after the carrier and Program storage have been restored.
class AcceptedProgramExecutionServicesSnapshot {
 public:
  AcceptedProgramExecutionServicesSnapshot() = default;
  AcceptedProgramExecutionServicesSnapshot(const AcceptedProgramExecutionServicesSnapshot&) =
      delete;
  AcceptedProgramExecutionServicesSnapshot& operator=(
      const AcceptedProgramExecutionServicesSnapshot&) = delete;
  virtual ~AcceptedProgramExecutionServicesSnapshot() = default;

  virtual std::unique_ptr<AcceptedProgramExecutionServicesSnapshot> prepare_restore() const = 0;
  /// Refresh this already allocated image from its bound owner. Implementations must reject a
  /// changed composition/capacity before mutating any candidate-facing state; the accepted-step
  /// carrier uses it to avoid rebuilding DSO-owned AMR history/clock images after warmup.
  virtual void refresh_from_owner_preallocated() {
    throw std::logic_error(
        "Program accepted execution snapshot does not provide an in-place refresh image");
  }
  /// Produce a detached forward-topology image.  The opaque token is meaningful only to the
  /// concrete snapshot and is returned to it at HiddenPublish for owner rebinding.
  virtual std::unique_ptr<AcceptedProgramExecutionServicesSnapshot> detach_for_forward(
      std::uint64_t, std::uint64_t, void*&) const {
    throw std::logic_error("Program accepted execution snapshot cannot stage a forward topology");
  }
  virtual void rebind_after_forward_publish(void*) noexcept { std::terminate(); }
  /// Installation-only publication into an adapter that is still exclusively candidate-owned.
  /// Unlike a forward topology publication this transfers only the prepared temporal/contract
  /// authority; checkpoint staging and topology-bound workspaces remain the host-primed image.
  virtual void publish_prepared_installation_temporal_authority(void*) noexcept {
    std::terminate();
  }
  virtual void prepare_forward_hierarchy_refresh(std::uint64_t, std::uint64_t) {
    throw std::logic_error(
        "Program accepted execution snapshot cannot prepare a forward hierarchy refresh");
  }
  virtual void prepare_forward_history_remap(const AmrProgramHistoryRemapDescriptor&) {
    throw std::logic_error(
        "Program accepted execution snapshot cannot prepare a forward history remap");
  }
  virtual void prepare_forward_full_history_reseed(const AmrProgramFullHistoryReseedDescriptor&) {
    throw std::logic_error(
        "Program accepted execution snapshot cannot prepare a forward full-history reseed");
  }
  /// Candidate-only cell-local provider rematerialization for a forward AMR hierarchy.  The
  /// authority is host-owned by the regrid carrier and contains no facade pointer.
  virtual void prepare_forward_temporal_partition(const PreparedForwardAmrTemporalAuthority&) {
    throw std::logic_error(
        "Program accepted execution snapshot cannot prepare a forward temporal partition");
  }
  /// Candidate-only rematerialization of bind-sealed transient Program resources. The concrete
  /// topology carrier supplies only forward field prototypes; publication remains a no-throw
  /// owner exchange after the AMR hierarchy itself is live.
  virtual void prepare_forward_scratch_rematerialization(
      const PreparedForwardAmrScratchTopology&) {
    throw std::logic_error(
        "Program accepted execution snapshot cannot prepare forward scratch resources");
  }
  /// Construct all DSO-private topology-bound execution resources while Candidate still owns the
  /// detached forward authority.  Publication is deliberately separate and noexcept.
  virtual void prepare_forward_execution_bundle(const PreparedForwardAmrExecutionAuthority&) {
    throw std::logic_error(
        "Program accepted execution snapshot cannot prepare a forward execution bundle");
  }
  /// Candidate-only checkpoint contribution for an unpublished forward AMR topology.  The result
  /// is intentionally value-owned and contains no facade/adapter pointer.
  virtual PreparedForwardAmrAcceptedContext prepare_forward_accepted_context(
      std::int64_t accepted_macro_step) const {
    (void)accepted_macro_step;
    throw std::logic_error(
        "Program accepted execution snapshot cannot prepare a forward accepted checkpoint");
  }
  /// Cold bind only.  Implementations use it to reserve finite accepted-image arenas before the
  /// first candidate; refresh/finalize paths never call it.
  virtual void prime_at_bind() {}
  /// Cold bind only, for a snapshot freshly copied after the live owner was already primed.
  /// Implementations restore copy-lost spare capacities without refreshing or republishing the
  /// owner a second time.
  virtual void prime_copied_image_at_bind() { prime_at_bind(); }
  /// Finite diagnostics that live beside an accepted execution context participate in the same
  /// Snapshot/Publish/Rollback authority.  Contexts without a separate provisional arena retain
  /// their existing semantics through these no-op defaults.
  virtual void snapshot_transaction_diagnostics_noexcept() noexcept {}
  virtual void publish_transaction_diagnostics_noexcept() noexcept {}
  virtual void restore_transaction_diagnostics_noexcept() noexcept {}
  virtual void publish_restore() noexcept = 0;
};

using AcceptedProgramExecutionServicesSnapshotFactory =
    std::function<std::unique_ptr<AcceptedProgramExecutionServicesSnapshot>()>;

/// A host wrapper around a snapshot allocated by an artifact.  The owner is declared before the
/// foreign snapshot so the latter is destroyed first; a retained snapshot can therefore outlive a
/// replacement without ever dispatching its virtual destructor through an unloaded image.
class RetainedAcceptedProgramExecutionServicesSnapshot final
    : public AcceptedProgramExecutionServicesSnapshot {
 public:
  RetainedAcceptedProgramExecutionServicesSnapshot(
      std::shared_ptr<OwnedProgramInstallation> artifact,
      std::unique_ptr<AcceptedProgramExecutionServicesSnapshot> snapshot)
      : artifact_(std::move(artifact)), snapshot_(std::move(snapshot)) {}

  std::unique_ptr<AcceptedProgramExecutionServicesSnapshot> prepare_restore() const override {
    auto restored = snapshot_->prepare_restore();
    if (!restored)
      return {};
    return std::make_unique<RetainedAcceptedProgramExecutionServicesSnapshot>(artifact_,
                                                                              std::move(restored));
  }

  void refresh_from_owner_preallocated() override {
    if (!snapshot_)
      throw std::logic_error("retained Program accepted snapshot is empty");
    snapshot_->refresh_from_owner_preallocated();
  }

  std::unique_ptr<AcceptedProgramExecutionServicesSnapshot> detach_for_forward(
      std::uint64_t topology_epoch, std::uint64_t materialization_generation,
      void*& rebind_token) const override {
    if (!snapshot_)
      throw std::logic_error("retained Program accepted snapshot is empty");
    auto detached =
        snapshot_->detach_for_forward(topology_epoch, materialization_generation, rebind_token);
    if (!detached)
      throw std::logic_error("retained Program forward snapshot is empty");
    return std::make_unique<RetainedAcceptedProgramExecutionServicesSnapshot>(artifact_,
                                                                              std::move(detached));
  }

  void rebind_after_forward_publish(void* rebind_token) noexcept override {
    if (!snapshot_)
      std::terminate();
    snapshot_->rebind_after_forward_publish(rebind_token);
  }

  void publish_prepared_installation_temporal_authority(void* rebind_token) noexcept override {
    if (!snapshot_)
      std::terminate();
    snapshot_->publish_prepared_installation_temporal_authority(rebind_token);
  }

  void prepare_forward_hierarchy_refresh(std::uint64_t topology_epoch,
                                         std::uint64_t materialization_generation) override {
    if (!snapshot_)
      throw std::logic_error("retained Program accepted snapshot is empty");
    snapshot_->prepare_forward_hierarchy_refresh(topology_epoch, materialization_generation);
  }

  void prepare_forward_history_remap(const AmrProgramHistoryRemapDescriptor& descriptor) override {
    if (!snapshot_)
      throw std::logic_error("retained Program accepted snapshot is empty");
    snapshot_->prepare_forward_history_remap(descriptor);
  }

  void prepare_forward_full_history_reseed(
      const AmrProgramFullHistoryReseedDescriptor& descriptor) override {
    if (!snapshot_)
      throw std::logic_error("retained Program accepted snapshot is empty");
    snapshot_->prepare_forward_full_history_reseed(descriptor);
  }

  void prepare_forward_temporal_partition(
      const PreparedForwardAmrTemporalAuthority& authority) override {
    if (!snapshot_)
      throw std::logic_error("retained Program accepted snapshot is empty");
    snapshot_->prepare_forward_temporal_partition(authority);
  }

  void prepare_forward_scratch_rematerialization(
      const PreparedForwardAmrScratchTopology& topology) override {
    if (!snapshot_)
      throw std::logic_error("retained Program accepted snapshot is empty");
    snapshot_->prepare_forward_scratch_rematerialization(topology);
  }

  void prepare_forward_execution_bundle(
      const PreparedForwardAmrExecutionAuthority& authority) override {
    if (!snapshot_)
      throw std::logic_error("retained Program accepted snapshot is empty");
    snapshot_->prepare_forward_execution_bundle(authority);
  }

  PreparedForwardAmrAcceptedContext prepare_forward_accepted_context(
      std::int64_t accepted_macro_step) const override {
    if (!snapshot_)
      throw std::logic_error("retained Program accepted snapshot is empty");
    return snapshot_->prepare_forward_accepted_context(accepted_macro_step);
  }

  void prime_at_bind() override {
    if (!snapshot_)
      throw std::logic_error("retained Program accepted snapshot is empty");
    snapshot_->prime_at_bind();
  }

  void prime_copied_image_at_bind() override {
    if (!snapshot_)
      throw std::logic_error("retained Program accepted snapshot is empty");
    snapshot_->prime_copied_image_at_bind();
  }

  void snapshot_transaction_diagnostics_noexcept() noexcept override {
    if (!snapshot_)
      std::terminate();
    snapshot_->snapshot_transaction_diagnostics_noexcept();
  }

  void publish_transaction_diagnostics_noexcept() noexcept override {
    if (!snapshot_)
      std::terminate();
    snapshot_->publish_transaction_diagnostics_noexcept();
  }

  void restore_transaction_diagnostics_noexcept() noexcept override {
    if (!snapshot_)
      std::terminate();
    snapshot_->restore_transaction_diagnostics_noexcept();
  }

  void publish_restore() noexcept override { snapshot_->publish_restore(); }

 private:
  std::shared_ptr<OwnedProgramInstallation> artifact_;
  std::unique_ptr<AcceptedProgramExecutionServicesSnapshot> snapshot_;
};

inline std::unique_ptr<AcceptedProgramExecutionServicesSnapshot>
OwnedProgramInstallation::invoke_accepted_snapshot() const {
  require_prepared_();
  if (candidate_.create_accepted_snapshot == nullptr)
    throw std::logic_error("Program installation has no accepted context snapshot candidate");
  AcceptedProgramExecutionServicesSnapshot* snapshot = pops::dynlib::invoke_with_host_exception(
      [&] { return candidate_.create_accepted_snapshot(candidate_.context); },
      "Program candidate accepted context snapshot");
  return std::unique_ptr<AcceptedProgramExecutionServicesSnapshot>(snapshot);
}

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
enum class AutomaticBalanceTerm : std::uint8_t {
  outward_boundary_flux = 0,
  sources = 1,
  reflux = 2,
  projection = 3,
};

struct AutomaticBalanceKey {
  int runtime_block = -1;
  int level = -1;
  int component = -1;
  AutomaticBalanceTerm term = AutomaticBalanceTerm::outward_boundary_flux;

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
  /// Immutable host-owned receipt retained with the accepted artifact. It ensures the copied
  /// authority tables and bind-sealed resource plan survive after the temporary publication image
  /// is gone, and participates in replacement rollback.
  struct ArtifactPublicationReceipt final {
    ProgramInstallationMetadata metadata;
    ProgramInstallationTables tables;
    ProgramResourcePlan resource_plan;
  };

  struct DiagnosticSlot final {
    Real value = Real(0);
    bool recorded = false;

    friend bool operator==(const DiagnosticSlot&, const DiagnosticSlot&) = default;
  };

  struct BalanceRouteSlot final {
    std::array<Real, 5> values{};
    std::array<bool, 5> recorded{};

    friend bool operator==(const BalanceRouteSlot&, const BalanceRouteSlot&) = default;
  };

  struct AutomaticBalanceSlot final {
    Real value = Real(0);
    bool recorded = false;

    friend bool operator==(const AutomaticBalanceSlot&, const AutomaticBalanceSlot&) = default;
  };

  struct StepProjectionSlot final {
    std::string identity;
    bool executed = false;

    friend bool operator==(const StepProjectionSlot&, const StepProjectionSlot&) = default;
  };

  using DiagnosticRegistry = std::map<std::string, DiagnosticSlot, std::less<>>;
  using BalanceRouteRegistry = std::map<std::string, BalanceRouteSlot, std::less<>>;
  using AutomaticBalanceRegistry = std::map<AutomaticBalanceKey, AutomaticBalanceSlot>;
  /// Static field-boundary authoring image captured before the first successful artifact overlay.
  std::optional<ArtifactFieldBoundaryAuthorityRegistry<Dim>> artifact_field_boundary_baseline_;
  /// Candidate sink active only while the v5 boundary-route table is prepared.
  std::optional<ArtifactFieldBoundaryStage<Dim>> artifact_field_boundary_stage_;
  // --- fields read by the stepper (the ONLY Program state the stepper sees) -------------------------
  /// Host-owned DSO image and candidate state for an installed artifact.  This deliberately
  /// precedes the closures below: C++ destroys members in reverse declaration order, so every
  /// DSO-backed closure dies before the candidate destroy callback and image close.
  std::shared_ptr<OwnedProgramInstallation> artifact_installation_;
  std::optional<ArtifactPublicationReceipt> artifact_publication_receipt_;
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
  std::function<void(const AmrProgramHistoryRemapDescriptor&)> history_remap_accepted_;
  /// Artifact-owned accepted-boundary hooks used only by the strict AMR restart transaction.
  /// `restart_regrid_preflight_` validates every rank-local prerequisite before peers enter the
  /// scientific regrid; `restart_regrid_` then performs that tag/regrid pass;
  /// `restart_resync_` force-imports the facade bytes after either accepted restart publication or
  /// rollback, even when their restored revision equals the context's last observed revision.
  /// `accepted_context_snapshot_` contributes context-owned ledgers and clocks to every outer
  /// accepted transaction. Uniform leaves all four empty.
  std::function<void()> restart_regrid_preflight_;
  std::function<void()> restart_regrid_;
  std::function<void()> restart_resync_;
  AcceptedProgramExecutionServicesSnapshotFactory accepted_context_snapshot_;
  /// Monotone witness incremented by each validated manual or prepared-artifact replacement. Dynamic
  /// artifact loaders use it to prove that one installer invocation replaced the whole-system step.
  std::uint64_t step_install_generation_ = 0;
  /// OPTIONAL compiled-Program dt bound (ADC-417). The target-specific loader stores a closure here
  /// over ProgramExecutionServices (uniform) or ProgramExecutionServices (AMR); step_cfl tightens dt to
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
  /// IR hash of the installed compiled Program candidate (ADC-406b). Empty until
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
  /// No direct step installer exists; omitted checkpoint history is replayable only from this
  /// artifact-owned authority.
  bool artifact_backed_ = false;
  /// NAME-based block binding (ADC-457): program-index -> runtime-block-index map. Entry p holds the
  /// runtime block index the Program's block p names. EMPTY means no authenticated mapping; positional
  /// identity is not inferred. Used by BOTH runtimes; read by the (Amr)ProgramExecutionServices.
  std::vector<int> block_map_;

  // --- runtime data owned across the step closure --------------------------------------------------
  /// COMPILED-PROGRAM SCALAR DIAGNOSTICS (ADC-414): name -> last value recorded via P.record_scalar.
  /// Lives here (not the .so) so it outlives the step closure and Python can read it. Used by BOTH.
  DiagnosticRegistry diagnostics_;
  /// Reserved balance records for the current native attempt only. Unlike diagnostics_, this
  /// mailbox is cleared before every public step and is never checkpointed. Accepted balance
  /// consumers read it while the facade's outer transaction still retains U^n, so a missing term
  /// cannot silently reuse the preceding step.
  BalanceRouteRegistry step_balance_terms_;
  /// Native operator contributions captured only for a due Balance attempt. These values are keyed
  /// by their physical runtime coordinate instead of a user ledger route and are therefore not read
  /// by accepted_balance_terms(). The owning facade snapshots this map with the rest of the attempt,
  /// so rejection cannot leak automatic evidence into a retry.
  AutomaticBalanceRegistry automatic_balance_terms_;
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
  std::vector<StepProjectionSlot> step_projections_;
  /// Bind-sealed composition witness for the three attempt-local registries above. Declarations
  /// are an installation/prelude operation; once sealed, every candidate write is a lookup into an
  /// already-owned node and an unknown identity fails without growing a container.
  bool transaction_authorities_bound_ = false;
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
  /// One bind-sealed persistent-value authority for the whole runtime. Execution providers are
  /// rematerializable implementation views, so they must never own this storage.
  ProgramPersistentValueStore persistent_values_;
  /// Uniform has no AMR hierarchy capability, but its ABI still publishes a stable one-level
  /// hierarchy identity. This runtime-owned token survives provider rematerialization.
  struct OneLevelHierarchyIdentity final {};
  OneLevelHierarchyIdentity one_level_hierarchy_identity_;
  /// SCHEDULER VALUE CACHE (ADC-458), UNIFORM ONLY. The held-node cache (every(N).hold / accumulate_dt)
  /// keyed by IR node id; the uniform checkpoint serializes it. Empty on AMR (cache seam not wired).
  CacheManager<Dim> cache_;
  /// MULTISTEP HISTORY (ADC-406a), UNIFORM ONLY. Ring buffers for multistep schemes; the uniform
  /// checkpoint serializes them. Empty on AMR (history seam not wired).
  HistoryManager<Dim> hist_;

  // --- self-contained helpers (grid-free, Program-subsystem-worded errors) -------------------------

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
          step_projections_(accepted.step_projections_),
          transaction_authorities_bound_(accepted.transaction_authorities_bound_),
          block_params_(accepted.block_params_),
          persistent_values_(accepted.persistent_values_.clone()),
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
    DiagnosticRegistry diagnostics_;
    BalanceRouteRegistry step_balance_terms_;
    AutomaticBalanceRegistry automatic_balance_terms_;
    bool automatic_balance_due_ = false;
    bool balance_due_window_active_ = false;
    int balance_due_target_step_ = 0;
    bool balance_replay_active_ = false;
    bool balance_step_completed_ = false;
    bool balance_program_was_due_ = false;
    std::vector<StepProjectionSlot> step_projections_;
    bool transaction_authorities_bound_ = false;
    std::map<int, RuntimeParams> block_params_;
    ProgramPersistentValueStore persistent_values_;
    CacheManager<Dim> cache_;
    HistoryManager<Dim> hist_;
    Profiler::PreparedRestore profiler_restore_;
  };

  PreparedProgramAcceptedRestore prepare_accepted_restore(const ProgramRuntimeState& accepted) {
    if (this == &accepted)
      throw std::invalid_argument(
          "Program accepted restore requires an independent accepted image");
    require_transaction_authority_shape_(accepted);
    return PreparedProgramAcceptedRestore(*this, accepted);
  }

  void publish_prepared_accepted_restore(PreparedProgramAcceptedRestore&& prepared) noexcept {
    if (prepared.owner_ != this)
      std::terminate();
    static_assert(noexcept(diagnostics_.swap(prepared.diagnostics_)));
    static_assert(noexcept(step_balance_terms_.swap(prepared.step_balance_terms_)));
    static_assert(noexcept(automatic_balance_terms_.swap(prepared.automatic_balance_terms_)));
    static_assert(noexcept(step_projections_.swap(prepared.step_projections_)));
    static_assert(noexcept(block_params_.swap(prepared.block_params_)));
    static_assert(noexcept(persistent_values_.swap(prepared.persistent_values_)));
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
    step_projections_.swap(prepared.step_projections_);
    transaction_authorities_bound_ = prepared.transaction_authorities_bound_;
    block_params_.swap(prepared.block_params_);
    persistent_values_.swap(prepared.persistent_values_);
    std::swap(cache_, prepared.cache_);
    std::swap(hist_, prepared.hist_);
    profiler_.publish_prepared_restore(std::move(prepared.profiler_restore_));
  }

  /// Refresh a bind-primed accepted Program image in place. Program installation and immutable
  /// closures are deliberately excluded: they are an installation authority, not candidate state.
  /// Every map/ring/Fab identity is checked before its first write, so a late registration or
  /// layout change fails at the snapshot boundary instead of allocating in a transaction.
  void copy_from_preallocated(const ProgramRuntimeState& source) {
    if (this == &source)
      return;
    if (substeps_ != source.substeps_ || stride_ != source.stride_ ||
        step_install_generation_ != source.step_install_generation_)
      throw std::logic_error("Program accepted image composition changed after preparation");
    require_transaction_authority_shape_(source);
    require_same_map_shape_(block_params_, source.block_params_);
    cadence_window_dt_ = source.cadence_window_dt_;
    cadence_window_steps_ = source.cadence_window_steps_;
    cadence_window_start_time_ = source.cadence_window_start_time_;
    cadence_dispatch_active_ = source.cadence_dispatch_active_;
    cadence_clock_restore_pending_ = source.cadence_clock_restore_pending_;
    cadence_clock_restore_dt_ = source.cadence_clock_restore_dt_;
    cadence_clock_restore_steps_ = source.cadence_clock_restore_steps_;
    cadence_clock_restore_start_time_ = source.cadence_clock_restore_start_time_;
    cadence_clock_restore_last_dt_ = source.cadence_clock_restore_last_dt_;
    cadence_clock_restore_accepted_time_ = source.cadence_clock_restore_accepted_time_;
    cadence_clock_restore_macro_step_ = source.cadence_clock_restore_macro_step_;
    last_dt_ = source.last_dt_;
    copy_scalar_map_(diagnostics_, source.diagnostics_);
    copy_scalar_map_(step_balance_terms_, source.step_balance_terms_);
    copy_scalar_map_(automatic_balance_terms_, source.automatic_balance_terms_);
    automatic_balance_due_ = source.automatic_balance_due_;
    balance_due_window_active_ = source.balance_due_window_active_;
    balance_due_target_step_ = source.balance_due_target_step_;
    balance_replay_active_ = source.balance_replay_active_;
    balance_step_completed_ = source.balance_step_completed_;
    balance_program_was_due_ = source.balance_program_was_due_;
    for (std::size_t index = 0; index < step_projections_.size(); ++index)
      step_projections_[index].executed = source.step_projections_[index].executed;
    copy_scalar_map_(block_params_, source.block_params_);
    persistent_values_.copy_from_preallocated(source.persistent_values_);
    if (cache_.bound() != source.cache_.bound())
      throw std::logic_error("Program accepted image cache binding changed after preparation");
    if (cache_.bound())
      cache_.copy_from_preallocated(source.cache_);
    copy_history_from_preallocated_(hist_, source.hist_);
    profiler_.copy_from_preallocated(source.profiler_);
  }

 private:
  template <class Map>
  static void require_same_map_shape_(const Map& destination, const Map& source) {
    if (destination.size() != source.size())
      throw std::logic_error("Program accepted image map composition changed after preparation");
    for (const auto& [key, value] : source) {
      (void)value;
      if (destination.find(key) == destination.end())
        throw std::logic_error("Program accepted image map key changed after preparation");
    }
  }

  template <class Map>
  static void copy_scalar_map_(Map& destination, const Map& source) {
    require_same_map_shape_(destination, source);
    for (const auto& [key, value] : source) {
      const auto found = destination.find(key);
      found->second = value;
    }
  }

  void require_transaction_authority_shape_(const ProgramRuntimeState& source) const {
    if (!transaction_authorities_bound_ || !source.transaction_authorities_bound_)
      throw std::logic_error("Program transaction authorities were not sealed at bind");
    require_same_map_shape_(diagnostics_, source.diagnostics_);
    require_same_map_shape_(step_balance_terms_, source.step_balance_terms_);
    require_same_map_shape_(automatic_balance_terms_, source.automatic_balance_terms_);
    if (step_projections_.size() != source.step_projections_.size())
      throw std::logic_error(
          "Program accepted image projection composition changed after preparation");
    for (std::size_t index = 0; index < step_projections_.size(); ++index)
      if (step_projections_[index].identity != source.step_projections_[index].identity)
        throw std::logic_error(
            "Program accepted image projection identity changed after preparation");
  }

  static void copy_string_(std::string& destination, const std::string& source) {
    if (source.size() > destination.capacity())
      throw std::logic_error("Program accepted image string capacity was not primed");
    destination.assign(source.data(), source.size());
  }

  static void copy_field_(MultiFab<Dim>& destination, const MultiFab<Dim>& source) {
    if (destination.layout() != source.layout() ||
        destination.distribution() != source.distribution() ||
        destination.local_rank() != source.local_rank() || destination.ncomp() != source.ncomp() ||
        destination.ghosts() != source.ghosts() || destination.local_size() != source.local_size())
      throw std::logic_error("Program accepted image field layout changed after preparation");
    for (std::size_t local = 0; local < source.local_size(); ++local) {
      if (destination.global_index(local) != source.global_index(local) ||
          destination.fab(local).size() != source.fab(local).size())
        throw std::logic_error("Program accepted image field ownership changed after preparation");
    }
    if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                             typename MultiFab<Dim>::memory_space>::accessible) {
      // Kokkos::deep_copy constructs profiling labels on this host-to-host path.  Those labels
      // allocate even though both resident images and their exact shape were already prepared.
      // Synchronize prior kernels once, then copy directly between the sealed host-accessible
      // views.  Device-only backends retain the asynchronous Kokkos copy below.
      ::pops::device_fence();
      for (std::size_t local = 0; local < source.local_size(); ++local) {
        const auto& source_storage = source.fab(local).storage();
        auto& destination_storage = destination.fab(local).storage();
        std::copy_n(source_storage.data(), source_storage.extent(0), destination_storage.data());
      }
    } else {
      for (std::size_t local = 0; local < source.local_size(); ++local)
        Kokkos::deep_copy(destination.fab(local).storage(), source.fab(local).storage());
    }
  }

  static void copy_string_map_(std::map<std::string, std::string>& destination,
                               const std::map<std::string, std::string>& source) {
    if (destination.size() != source.size())
      throw std::logic_error("Program accepted image string-map composition changed");
    for (const auto& [key, value] : source) {
      const auto found = destination.find(key);
      if (found == destination.end())
        throw std::logic_error("Program accepted image string-map key changed");
      copy_string_(found->second, value);
    }
  }

  static void copy_history_from_preallocated_(HistoryManager<Dim>& destination,
                                              const HistoryManager<Dim>& source) {
    if (destination.histories.size() != source.histories.size())
      throw std::logic_error("Program accepted image history composition changed");
    for (const auto& [name, source_ring] : source.histories) {
      const auto found = destination.histories.find(name);
      if (found == destination.histories.end() || found->second.size() != source_ring.size())
        throw std::logic_error("Program accepted image history ring changed");
      for (std::size_t slot = 0; slot < source_ring.size(); ++slot)
        copy_field_(found->second[slot], source_ring[slot]);
    }
    copy_scalar_map_(destination.depth, source.depth);
    copy_scalar_map_(destination.initialized, source.initialized);
    copy_scalar_map_(destination.fill_count, source.fill_count);
    copy_scalar_map_(destination.store_pending, source.store_pending);
    copy_scalar_map_(destination.owner, source.owner);
    copy_string_map_(destination.state_identity, source.state_identity);
    copy_string_map_(destination.space_identity, source.space_identity);
    copy_string_map_(destination.clock_identity, source.clock_identity);
    copy_string_map_(destination.interpolation_identity, source.interpolation_identity);
    if (destination.slot_dt.size() != source.slot_dt.size())
      throw std::logic_error("Program accepted image history dt composition changed");
    for (const auto& [name, source_dts] : source.slot_dt) {
      const auto found = destination.slot_dt.find(name);
      if (found == destination.slot_dt.end() || source_dts.size() > found->second.capacity())
        throw std::logic_error("Program accepted image history dt capacity was not primed");
      found->second.assign(source_dts.begin(), source_dts.end());
    }
  }

 public:
  const std::vector<int>& block_map() const noexcept { return block_map_; }

  Profiler& profiler() noexcept { return profiler_; }

  ProgramPersistentValueStore& persistent_values() noexcept { return persistent_values_; }
  const ProgramPersistentValueStore& persistent_values() const noexcept {
    return persistent_values_;
  }
  [[nodiscard]] const std::optional<ArtifactPublicationReceipt>& artifact_publication_receipt()
      const noexcept {
    return artifact_publication_receipt_;
  }
  void* one_level_hierarchy_identity() noexcept { return &one_level_hierarchy_identity_; }
  const void* one_level_hierarchy_identity() const noexcept {
    return &one_level_hierarchy_identity_;
  }

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

  /// Fully prepared replacement image. Every operation that can allocate, validate a resource
  /// digest, or capture a DSO callback happens here, before it can mutate a live Program state.
  /// The subsequent publish is a closed sequence of noexcept swaps/exchanges.
  class PreparedArtifactPublication final {
   public:
    static PreparedArtifactPublication prepare(PreparedProgramInstallation installation,
                                               std::uint64_t next_generation) {
      if (!installation.prepared())
        throw std::invalid_argument("Program artifact publication requires a prepared handle");
      if (next_generation == 0 || installation.generation() != next_generation)
        throw std::invalid_argument(
            "Program artifact publication generation differs from its preparation image");
      // Copy the immutable receipt while the complete handle is still inspectable.  The payload
      // transfer below is then allocation-free apart from the shared owner control block, and all
      // of it still happens before any live runtime state is exchanged.
      ProgramInstallationMetadata metadata = installation.metadata();
      ProgramInstallationTables tables = installation.tables();
      auto payload = std::move(installation).release_publication_payload();
      auto artifact = std::make_shared<OwnedProgramInstallation>(std::move(payload.owner));
      return prepare_complete_(std::move(artifact), next_generation, std::move(metadata),
                               std::move(tables), std::move(payload.resource_plan),
                               std::move(payload.persistent_values));
    }

   private:
    static PreparedArtifactPublication prepare_complete_(
        std::shared_ptr<OwnedProgramInstallation> artifact, std::uint64_t next_generation,
        ProgramInstallationMetadata metadata, ProgramInstallationTables tables,
        ProgramResourcePlan resource_plan, ProgramPersistentValueStore persistent_values) {
      PreparedArtifactPublication result(std::move(artifact));
      result.next_generation_ = next_generation;
      result.metadata_ = std::move(metadata);
      result.tables_ = std::move(tables);
      result.resource_plan_ = std::move(resource_plan);
      result.persistent_values_ = std::move(persistent_values);
      // Dense caches are an explicit resource-plan consumer.  Bind before the DSO owner can be
      // published so no generated cache path can grow a node-id table after installation.
      result.cache_.bind(result.resource_plan_);
      result.receipt_.emplace(
          ArtifactPublicationReceipt{result.metadata_, result.tables_, result.resource_plan_});
      result.step_ = [artifact = result.artifact_](double dt) { artifact->invoke_step(dt); };
      if (result.artifact_->candidate().dt_bound != nullptr) {
        result.dt_bound_ = [artifact = result.artifact_](Real cfl) {
          const std::optional<double> value = artifact->invoke_dt_bound(static_cast<double>(cfl));
          if (!value)
            throw std::logic_error("Program artifact lost its declared dt-bound callback");
          return static_cast<Real>(*value);
        };
      }
      if (result.artifact_->candidate().hierarchy_refresh != nullptr) {
        result.hierarchy_refresh_ = [artifact = result.artifact_] {
          artifact->invoke_hierarchy_refresh();
        };
        result.history_remap_accepted_ =
            [artifact = result.artifact_](const AmrProgramHistoryRemapDescriptor& descriptor) {
              artifact->invoke_history_remap_accepted(&descriptor);
            };
        result.restart_regrid_preflight_ = [artifact = result.artifact_] {
          artifact->invoke_restart_regrid_preflight();
        };
        result.restart_regrid_ = [artifact = result.artifact_] {
          artifact->invoke_restart_regrid();
        };
        result.restart_resync_ = [artifact = result.artifact_] {
          artifact->invoke_restart_resync();
        };
        result.accepted_context_snapshot_ = [artifact = result.artifact_] {
          auto snapshot = artifact->invoke_accepted_snapshot();
          if (!snapshot)
            return std::unique_ptr<AcceptedProgramExecutionServicesSnapshot>{};
          return std::unique_ptr<AcceptedProgramExecutionServicesSnapshot>(
              std::make_unique<RetainedAcceptedProgramExecutionServicesSnapshot>(
                  artifact, std::move(snapshot)));
        };
      }
      return result;
    }

   public:
    /// Attach all host-resolved Program authority before publication.  The loader has already
    /// validated these values against the frozen DSO tables; keeping them in this image prevents
    /// an install prelude from borrowing the accepted ProgramRuntimeState as scratch storage.
    void set_resolved_authority(std::string installed_hash,
                                std::vector<std::array<std::uint64_t, 4>> operators,
                                std::vector<std::pair<std::string, int>> history_replay,
                                ProgramCheckpointMetadata checkpoint, std::vector<int> block_map,
                                std::map<int, RuntimeParams> block_params, bool state_free) {
      if (installed_hash.empty())
        throw std::invalid_argument("Program artifact authority is incomplete");
      if (state_free != tables_.blocks.empty())
        throw std::invalid_argument(
            "Program artifact state-free authority disagrees with its authenticated block table");
      const bool has_block_owned_authority =
          !operators.empty() || !history_replay.empty() || !checkpoint.histories.empty();
      if (state_free) {
        if (!block_map.empty() || !block_params.empty() || has_block_owned_authority)
          throw std::invalid_argument(
              "state-free Program artifact authority carries block-owned state");
      } else if (block_map.empty()) {
        throw std::invalid_argument("Program artifact authority is incomplete");
      }
      installed_hash_ = std::move(installed_hash);
      operator_authorities_ = std::move(operators);
      history_replay_authorities_ = std::move(history_replay);
      checkpoint_metadata_ = std::move(checkpoint);
      block_map_ = std::move(block_map);
      block_params_ = std::move(block_params);
      artifact_backed_ = true;
    }

    void adopt_prepared_transaction_authorities(ProgramRuntimeState& staged) {
      staged.bind_transaction_authorities();
      diagnostics_.swap(staged.diagnostics_);
      step_balance_terms_.swap(staged.step_balance_terms_);
      automatic_balance_terms_.swap(staged.automatic_balance_terms_);
      step_projections_.swap(staged.step_projections_);
      // The publication image receives the staged declaration nodes only here.  Its prior
      // bound bit describes the empty construction image, not these swapped registries; reseal
      // the final owner before any detached accepted transaction image can capture it.
      transaction_authorities_bound_ = false;
      bind_transaction_authorities_();
    }

    void adopt_prepared_execution_state(ProgramRuntimeState& staged) {
      adopt_prepared_transaction_authorities(staged);
      persistent_values_.swap(staged.persistent_values_);
      // The staged Program can only replace the fresh bound cache when it inherited exactly the
      // same finite table.  A legacy/unbound staging state must not erase the plan-bound image.
      if (staged.cache_.has_same_bound_plan(cache_))
        cache_.swap(staged.cache_);
      using std::swap;
      swap(hist_, staged.hist_);
    }

    /// Build the mutable Program portion of an accepted transaction image from this final
    /// publication, without retaining the previously installed DSO copied into `staged`.
    /// Installation callbacks and immutable receipts are deliberately absent: transaction
    /// capture/restore only refreshes the finite scientific registries below, while the live
    /// ProgramRuntimeState remains the sole installation authority.
    [[nodiscard]] ProgramRuntimeState prepare_accepted_transaction_image(
        const ProgramRuntimeState& staged) const {
      ProgramRuntimeState image(staged);
      image.artifact_installation_.reset();
      image.artifact_publication_receipt_.reset();
      image.artifact_field_boundary_baseline_.reset();
      image.artifact_field_boundary_stage_.reset();
      image.step_ = {};
      image.hierarchy_refresh_ = {};
      image.history_remap_accepted_ = {};
      image.restart_regrid_preflight_ = {};
      image.restart_regrid_ = {};
      image.restart_resync_ = {};
      image.accepted_context_snapshot_ = {};
      image.dt_bound_ = {};
      image.step_install_generation_ = next_generation_;
      image.diagnostics_ = diagnostics_;
      image.step_balance_terms_ = step_balance_terms_;
      image.automatic_balance_terms_ = automatic_balance_terms_;
      image.step_projections_ = step_projections_;
      // This image is the resident Program portion of an AMR rollback carrier.  Seal the exact
      // final registry shape it owns rather than inheriting a construction-state bit from either
      // the detached staging state or the publication image.
      image.transaction_authorities_bound_ = false;
      image.bind_transaction_authorities();
      image.block_map_ = block_map_;
      image.block_params_ = block_params_;
      image.persistent_values_ = persistent_values_.clone();
      image.cache_ = cache_;
      image.hist_ = hist_;
      image.installed_hash_.clear();
      image.operator_authorities_.clear();
      image.history_replay_authorities_.clear();
      image.checkpoint_metadata_ = {};
      image.artifact_backed_ = false;
      return image;
    }

    /// Installation-only cache materialization.  The finite slot table was bound in
    /// prepare_complete_; this is the sole allocation boundary for scheduled values.
    void prime_cache_slot(ProgramCacheSlot slot, const MultiFab<Dim>& prototype) {
      cache_.prime_slot(slot, prototype);
    }

    /// Histories are built into a disconnected ProgramRuntimeState by System/AmrSystem before
    /// publication.  The final exchange below is noexcept and cannot expose a half-registered ring.
    void adopt_prepared_histories(HistoryManager<Dim> histories) noexcept {
      using std::swap;
      swap(hist_, histories);
    }

    void set_field_boundary_baseline(
        std::optional<ArtifactFieldBoundaryAuthorityRegistry<Dim>> baseline) noexcept {
      boundary_baseline_ = std::move(baseline);
    }

    /// Immutable evidence retained until the owner-last publication.  Installers use this rather
    /// than a symbolic manifest when they authenticate the final all-rank publication decision.
    [[nodiscard]] const ProgramResourcePlan& resource_plan() const noexcept {
      return resource_plan_;
    }
    [[nodiscard]] const ProgramInstallationMetadata& metadata() const noexcept { return metadata_; }
    [[nodiscard]] const ProgramInstallationTables& tables() const noexcept { return tables_; }
    [[nodiscard]] const ProgramCheckpointMetadata& checkpoint_metadata() const noexcept {
      return checkpoint_metadata_;
    }

    /// Invoke the artifact-owned accepted-context factory while this publication image still owns
    /// the DSO but before the host swaps any live authority.  AMR uses the resulting detached
    /// snapshot to serialize its initial POPSAND5 image during installation; doing that after
    /// owner-last publication would turn a cold bootstrap into an observable live mutation.
    [[nodiscard]] std::unique_ptr<AcceptedProgramExecutionServicesSnapshot>
    prepare_accepted_context_snapshot() const {
      if (!accepted_context_snapshot_)
        throw std::logic_error(
            "prepared Program artifact lacks its accepted context snapshot factory");
      auto snapshot = accepted_context_snapshot_();
      if (!snapshot)
        throw std::logic_error(
            "prepared Program artifact returned an empty accepted context snapshot");
      return snapshot;
    }

   private:
    explicit PreparedArtifactPublication(std::shared_ptr<OwnedProgramInstallation> artifact)
        : artifact_(std::move(artifact)) {}
    friend struct ProgramRuntimeState;

    void bind_transaction_authorities_() {
      for (const auto& [name, slot] : diagnostics_) {
        (void)slot;
        if (name.empty() || ProgramRuntimeState::has_reserved_balance_namespace(name))
          throw std::logic_error("Program diagnostic registry is invalid at bind");
      }
      for (const auto& [route, slot] : step_balance_terms_) {
        (void)slot;
        ProgramRuntimeState::require_balance_route(
            route,
            "ProgramRuntimeState::PreparedArtifactPublication::bind_transaction_authorities");
      }
      for (const auto& slot : step_projections_)
        if (slot.identity.empty())
          throw std::logic_error("Program projection registry is invalid at bind");
      transaction_authorities_bound_ = true;
    }

    std::shared_ptr<OwnedProgramInstallation> artifact_;
    ProgramInstallationMetadata metadata_;
    ProgramInstallationTables tables_;
    ProgramResourcePlan resource_plan_;
    std::optional<ArtifactPublicationReceipt> receipt_;
    ProgramPersistentValueStore persistent_values_;
    std::optional<ArtifactFieldBoundaryAuthorityRegistry<Dim>> boundary_baseline_;
    DiagnosticRegistry diagnostics_;
    BalanceRouteRegistry step_balance_terms_;
    AutomaticBalanceRegistry automatic_balance_terms_;
    std::vector<StepProjectionSlot> step_projections_;
    bool transaction_authorities_bound_ = false;
    CacheManager<Dim> cache_;
    HistoryManager<Dim> hist_;
    std::function<void(double)> step_;
    std::function<void()> hierarchy_refresh_;
    std::function<void(const AmrProgramHistoryRemapDescriptor&)> history_remap_accepted_;
    std::function<void()> restart_regrid_preflight_, restart_regrid_, restart_resync_;
    AcceptedProgramExecutionServicesSnapshotFactory accepted_context_snapshot_;
    std::function<Real(Real)> dt_bound_;
    std::uint64_t next_generation_ = 0;
    std::string installed_hash_;
    std::vector<std::array<std::uint64_t, 4>> operator_authorities_;
    std::vector<std::pair<std::string, int>> history_replay_authorities_;
    ProgramCheckpointMetadata checkpoint_metadata_;
    std::vector<int> block_map_;
    std::map<int, RuntimeParams> block_params_;
    bool artifact_backed_ = false;
  };

  /// Install one already-prepared artifact owner through a prebuilt publication image. Candidate
  /// state, callbacks, tables and persistent slots are all prepared before the first live mutation.
  void install_prepared_artifact(PreparedProgramInstallation installation) {
    if (step_install_generation_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("Program step-install generation overflow");
    auto prepared =
        PreparedArtifactPublication::prepare(std::move(installation), step_install_generation_ + 1);
    publish_prepared_artifact_(std::move(prepared));
  }

  /// No-throw final half of a host aggregate publication.  The caller must have constructed the
  /// image and closed its collective checks first; this method performs no validation/allocation.
  void publish_prepared_artifact(PreparedArtifactPublication&& prepared) noexcept {
    publish_prepared_artifact_(std::move(prepared));
  }

 private:
  void publish_prepared_artifact_(PreparedArtifactPublication&& prepared) noexcept {
    static_assert(noexcept(step_.swap(prepared.step_)));
    static_assert(noexcept(hierarchy_refresh_.swap(prepared.hierarchy_refresh_)));
    static_assert(noexcept(history_remap_accepted_.swap(prepared.history_remap_accepted_)));
    static_assert(noexcept(restart_regrid_preflight_.swap(prepared.restart_regrid_preflight_)));
    static_assert(noexcept(restart_regrid_.swap(prepared.restart_regrid_)));
    static_assert(noexcept(restart_resync_.swap(prepared.restart_resync_)));
    static_assert(noexcept(accepted_context_snapshot_.swap(prepared.accepted_context_snapshot_)));
    static_assert(noexcept(dt_bound_.swap(prepared.dt_bound_)));
    static_assert(noexcept(persistent_values_.swap(prepared.persistent_values_)));
    static_assert(noexcept(diagnostics_.swap(prepared.diagnostics_)));
    static_assert(noexcept(step_balance_terms_.swap(prepared.step_balance_terms_)));
    static_assert(noexcept(automatic_balance_terms_.swap(prepared.automatic_balance_terms_)));
    static_assert(noexcept(step_projections_.swap(prepared.step_projections_)));
    static_assert(std::is_nothrow_swappable_v<CacheManager<Dim>>);
    static_assert(std::is_nothrow_swappable_v<HistoryManager<Dim>>);
    static_assert(noexcept(artifact_publication_receipt_.swap(prepared.receipt_)));
    static_assert(noexcept(artifact_field_boundary_baseline_.swap(prepared.boundary_baseline_)));
    static_assert(noexcept(installed_hash_.swap(prepared.installed_hash_)));
    static_assert(noexcept(operator_authorities_.swap(prepared.operator_authorities_)));
    static_assert(noexcept(history_replay_authorities_.swap(prepared.history_replay_authorities_)));
    static_assert(noexcept(std::swap(checkpoint_metadata_, prepared.checkpoint_metadata_)));
    static_assert(noexcept(block_map_.swap(prepared.block_map_)));
    static_assert(noexcept(block_params_.swap(prepared.block_params_)));
    static_assert(noexcept(artifact_installation_.swap(prepared.artifact_)));
    step_.swap(prepared.step_);
    hierarchy_refresh_.swap(prepared.hierarchy_refresh_);
    history_remap_accepted_.swap(prepared.history_remap_accepted_);
    restart_regrid_preflight_.swap(prepared.restart_regrid_preflight_);
    restart_regrid_.swap(prepared.restart_regrid_);
    restart_resync_.swap(prepared.restart_resync_);
    accepted_context_snapshot_.swap(prepared.accepted_context_snapshot_);
    dt_bound_.swap(prepared.dt_bound_);
    persistent_values_.swap(prepared.persistent_values_);
    diagnostics_.swap(prepared.diagnostics_);
    step_balance_terms_.swap(prepared.step_balance_terms_);
    automatic_balance_terms_.swap(prepared.automatic_balance_terms_);
    step_projections_.swap(prepared.step_projections_);
    transaction_authorities_bound_ = prepared.transaction_authorities_bound_;
    std::swap(cache_, prepared.cache_);
    std::swap(hist_, prepared.hist_);
    artifact_publication_receipt_.swap(prepared.receipt_);
    artifact_field_boundary_baseline_.swap(prepared.boundary_baseline_);
    installed_hash_.swap(prepared.installed_hash_);
    operator_authorities_.swap(prepared.operator_authorities_);
    history_replay_authorities_.swap(prepared.history_replay_authorities_);
    std::swap(checkpoint_metadata_, prepared.checkpoint_metadata_);
    block_map_.swap(prepared.block_map_);
    block_params_.swap(prepared.block_params_);
    artifact_backed_ = prepared.artifact_backed_;
    step_install_generation_ = prepared.next_generation_;
    // The old owner remains in ``prepared`` until its old closures have been destroyed at scope
    // exit.  This preserves the candidate-destroy-before-dlclose ordering across replacements.
    artifact_installation_.swap(prepared.artifact_);
  }

 public:
  /// Start one isolated artifact candidate before the detached preparation image is populated.
  /// Histories, scheduled values and diagnostics are qualified by the installed Program: retaining
  /// any of them across A -> B could either expose removed names or let coincident node/ring ids read
  /// values produced by A. The dynamic loaders call this before the generated prelude; a failed
  /// candidate is discarded without publishing this state.
  void reset_artifact_candidate_state() {
    diagnostics_.clear();
    step_balance_terms_.clear();
    automatic_balance_terms_.clear();
    step_projections_.clear();
    transaction_authorities_bound_ = false;
    cache_.clear();
    hist_ = HistoryManager<Dim>{};
  }

  std::unique_ptr<AcceptedProgramExecutionServicesSnapshot> capture_accepted_context_snapshot(
      const std::string& runtime) const {
    if (!artifact_backed_)
      return {};
    if (!accepted_context_snapshot_)
      throw std::logic_error(runtime + " artifact lacks its accepted context snapshot hook");
    std::unique_ptr<AcceptedProgramExecutionServicesSnapshot> snapshot =
        accepted_context_snapshot_();
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

  void resync_after_restart(const std::string& runtime) const {
    if (!artifact_backed_)
      return;
    if (!restart_resync_)
      throw std::logic_error(runtime + " artifact lacks its restart resync hook");
    restart_resync_();
  }

  /// Requalify Program-owned accepted state for the hierarchy currently exposed by the AMR engine.
  /// An authenticated artifact always provides this hook; an uninstalled runtime remains a no-op.
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
  void accept_history_remap(const AmrProgramHistoryRemapDescriptor& descriptor,
                            const std::string& runtime) const {
    if (!history_remap_accepted_)
      throw std::logic_error(runtime + " artifact lacks its accepted history-remap hook");
    if (descriptor.parent_level < 0 || descriptor.child_level != descriptor.parent_level + 1 ||
        descriptor.prior_topology_epoch == std::numeric_limits<std::uint64_t>::max() ||
        descriptor.prior_materialization_generation == std::numeric_limits<std::uint64_t>::max() ||
        descriptor.published_topology_epoch != descriptor.prior_topology_epoch + 1 ||
        descriptor.published_materialization_generation !=
            descriptor.prior_materialization_generation + 1 ||
        descriptor.accepted_macro_step < 0 || descriptor.operation_identity.empty())
      throw std::invalid_argument(runtime +
                                  " received an unauthenticated history-remap descriptor");
    history_remap_accepted_(descriptor);
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

  /// Replay one already-authenticated historical interval without constructing a second scheduler.
  /// Selective checkpoint reconstruction has its own restored cursor; it therefore bypasses cadence
  /// accounting but still uses this state-owned direct invocation so no runtime reaches `step_`
  /// outside the sole Program dispatcher authority.
  void replay_step(double dt, const std::string& runtime) {
    if (!step_)
      throw std::logic_error(runtime +
                             " Program replay requires an installed whole-system Program");
    if (!std::isfinite(dt) || !(dt > 0.0))
      throw std::invalid_argument(runtime + " Program replay requires a finite positive interval");
    last_dt_ = static_cast<Real>(dt);
    run_balance_replay(runtime, [&] { step_(dt); });
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

  static bool has_reserved_balance_namespace(std::string_view name) noexcept {
    return name.rfind("pops.balance-term", 0) == 0;
  }

  static void require_balance_route(std::string_view route, std::string_view runtime) {
    static constexpr std::string_view kRoutePrefix = "pops.balance-ledger-route.v1:sha256:";
    if (route.size() != kRoutePrefix.size() + 64 ||
        route.compare(0, kRoutePrefix.size(), kRoutePrefix) != 0 ||
        !std::all_of(route.begin() + static_cast<std::ptrdiff_t>(kRoutePrefix.size()), route.end(),
                     [](unsigned char value) {
                       return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
                     }))
      throw std::invalid_argument(std::string(runtime) +
                                  " requires a canonical balance-ledger-route identity");
  }

  static void require_balance_due_contract(std::string_view contract, std::string_view runtime) {
    static constexpr std::string_view kContractPrefix = "pops.balance-due-contract.v1:sha256:";
    if (contract.size() != kContractPrefix.size() + 64 ||
        contract.compare(0, kContractPrefix.size(), kContractPrefix) != 0 ||
        !std::all_of(contract.begin() + static_cast<std::ptrdiff_t>(kContractPrefix.size()),
                     contract.end(), [](unsigned char value) {
                       return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
                     }))
      throw std::invalid_argument(std::string(runtime) +
                                  " requires a canonical balance-due-contract identity");
  }

  static constexpr std::array<std::string_view, 5> kBalanceTerms{
      "storage_change", "outward_boundary_flux", "sources", "reflux", "projection"};

  static std::size_t balance_term_index_(std::string_view term, std::string_view runtime) {
    const auto found = std::find(kBalanceTerms.begin(), kBalanceTerms.end(), term);
    if (found == kBalanceTerms.end())
      throw std::invalid_argument(std::string(runtime) +
                                  " requires one canonical five-term balance name");
    return static_cast<std::size_t>(found - kBalanceTerms.begin());
  }

  static AutomaticBalanceTerm automatic_balance_term_(std::string_view term,
                                                      std::string_view runtime) {
    if (term == "outward_boundary_flux")
      return AutomaticBalanceTerm::outward_boundary_flux;
    if (term == "sources")
      return AutomaticBalanceTerm::sources;
    if (term == "reflux")
      return AutomaticBalanceTerm::reflux;
    if (term == "projection")
      return AutomaticBalanceTerm::projection;
    throw std::invalid_argument(std::string(runtime) +
                                " requires one native operator balance contribution name");
  }

  void require_transaction_authorities_bound_(std::string_view operation) const {
    if (!transaction_authorities_bound_)
      throw std::logic_error(std::string(operation) +
                             " requires bind-sealed Program transaction authorities");
  }

  void require_transaction_authorities_unbound_(std::string_view operation) const {
    if (transaction_authorities_bound_)
      throw std::logic_error(std::string(operation) +
                             " cannot change Program transaction authorities after bind");
  }

  /// Installation/prelude declarations. Each call materializes at most one cold node; bind only
  /// validates and seals the resulting finite shape. Duplicate declarations are idempotent.
  void declare_diagnostic(std::string name) {
    require_transaction_authorities_unbound_("ProgramRuntimeState::declare_diagnostic");
    if (name.empty() || has_reserved_balance_namespace(name))
      throw std::invalid_argument(
          "Program diagnostic declaration requires a non-empty non-reserved identity");
    diagnostics_.try_emplace(std::move(name), DiagnosticSlot{});
  }

  void declare_balance_route(std::string route) {
    require_transaction_authorities_unbound_("ProgramRuntimeState::declare_balance_route");
    require_balance_route(route, "ProgramRuntimeState::declare_balance_route");
    step_balance_terms_.try_emplace(std::move(route), BalanceRouteSlot{});
  }

  void declare_automatic_balance_term(int runtime_block, int level, int component,
                                      std::string_view term) {
    require_transaction_authorities_unbound_("ProgramRuntimeState::declare_automatic_balance_term");
    if (runtime_block < 0 || level < 0 || component < 0)
      throw std::invalid_argument(
          "Program automatic balance declaration requires non-negative coordinates");
    const auto term_id =
        automatic_balance_term_(term, "ProgramRuntimeState::declare_automatic_balance_term");
    automatic_balance_terms_.try_emplace(
        AutomaticBalanceKey{runtime_block, level, component, term_id}, AutomaticBalanceSlot{});
  }

  void declare_step_projection(std::string name) {
    require_transaction_authorities_unbound_("ProgramRuntimeState::declare_step_projection");
    if (name.empty())
      throw std::invalid_argument("Program step projection identity cannot be empty");
    const auto found =
        std::find_if(step_projections_.begin(), step_projections_.end(),
                     [&name](const StepProjectionSlot& slot) { return slot.identity == name; });
    if (found == step_projections_.end())
      step_projections_.push_back(StepProjectionSlot{std::move(name), false});
  }

  void bind_transaction_authorities() {
    if (transaction_authorities_bound_)
      return;
    for (const auto& [name, slot] : diagnostics_) {
      (void)slot;
      if (name.empty() || has_reserved_balance_namespace(name))
        throw std::logic_error("Program diagnostic registry is invalid at bind");
    }
    for (const auto& [route, slot] : step_balance_terms_) {
      (void)slot;
      require_balance_route(route, "ProgramRuntimeState::bind_transaction_authorities");
    }
    for (const auto& slot : step_projections_)
      if (slot.identity.empty())
        throw std::logic_error("Program projection registry is invalid at bind");
    transaction_authorities_bound_ = true;
  }

  [[nodiscard]] bool transaction_authorities_bound() const noexcept {
    return transaction_authorities_bound_;
  }

  /// Record a compiled-Program scalar. Ordinary P.record_scalar names remain inspectable after the
  /// step with last-write-wins semantics. The balance namespace has a separate typed sink.
  void record_diagnostic(std::string_view name, Real value) {
    require_transaction_authorities_bound_("ProgramRuntimeState::record_diagnostic");
    if (has_reserved_balance_namespace(name))
      throw std::invalid_argument(
          "ProgramRuntimeState::record_diagnostic: pops.balance-term is a reserved namespace");
    const auto found = diagnostics_.find(name);
    if (found == diagnostics_.end())
      throw std::logic_error(
          "ProgramRuntimeState::record_diagnostic: diagnostic was not declared before bind");
    found->second.value = value;
    found->second.recorded = true;
  }

  /// Record one validated Program.record_balance term. Not exposed through the Python runtime
  /// facade: only generated ProgramExecutionServices code reaches this sink.
  void record_balance_term(std::string_view route, std::string_view term, Real value,
                           std::string_view runtime) {
    require_transaction_authorities_bound_("ProgramRuntimeState::record_balance_term");
    require_balance_route(route, runtime);
    const std::size_t term_index = balance_term_index_(term, runtime);
    if (!std::isfinite(static_cast<double>(value)))
      throw std::invalid_argument(std::string(runtime) +
                                  "::record_balance_term requires a finite value");
    const auto route_slot = step_balance_terms_.find(route);
    if (route_slot == step_balance_terms_.end())
      throw std::logic_error(std::string(runtime) +
                             "::record_balance_term route was not declared before bind");
    // A Program cadence may invoke the compiled body several times inside one public macro-step.
    // Terms are signed, time-integrated increments and therefore accumulate across invocations.
    route_slot->second.values[term_index] += value;
    route_slot->second.recorded[term_index] = true;
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
                                     std::string_view term, Real value, std::string_view runtime) {
    require_transaction_authorities_bound_("ProgramRuntimeState::record_automatic_balance_term");
    if (!automatic_balance_capture_due())
      throw std::logic_error(std::string(runtime) +
                             "::record_automatic_balance_term requires a due authored balance");
    if (runtime_block < 0 || level < 0 || component < 0)
      throw std::invalid_argument(
          std::string(runtime) +
          "::record_automatic_balance_term requires non-negative coordinates");
    const auto term_id = automatic_balance_term_(term, runtime);
    if (!std::isfinite(static_cast<double>(value)))
      throw std::invalid_argument(std::string(runtime) +
                                  "::record_automatic_balance_term requires a finite value");
    const auto entry = automatic_balance_terms_.find(
        AutomaticBalanceKey{runtime_block, level, component, term_id});
    if (entry == automatic_balance_terms_.end())
      throw std::logic_error(
          std::string(runtime) +
          "::record_automatic_balance_term coordinate was not declared before bind");
    entry->second.value += value;
    entry->second.recorded = true;
  }

  /// Read the named diagnostic, FAIL-LOUD if the Program never recorded it. @p runtime names the
  /// Program subsystem setter in the message (not a generic getter). @throws std::out_of_range.
  Real diagnostic(std::string_view name, std::string_view runtime) const {
    auto it = diagnostics_.find(name);
    if (it == diagnostics_.end() || !it->second.recorded)
      throw std::out_of_range(
          std::string(runtime) + "::program_diagnostic: no diagnostic named '" + std::string(name) +
          "' has been recorded (the installed Program must P.record_scalar it)");
    return it->second.value;
  }

  /// The whole name -> value diagnostics map (checkpoint / inspection). By value: inert copy.
  std::map<std::string, Real> diagnostics() const {
    std::map<std::string, Real> result;
    for (const auto& [name, slot] : diagnostics_)
      if (slot.recorded)
        result.emplace(name, slot.value);
    return result;
  }

  void begin_step_projection_report() {
    require_transaction_authorities_bound_("ProgramRuntimeState::begin_step_projection_report");
    for (auto& projection : step_projections_)
      projection.executed = false;
    for (auto& [route, slot] : step_balance_terms_) {
      (void)route;
      slot.values.fill(Real(0));
      slot.recorded.fill(false);
    }
    for (auto& [key, slot] : automatic_balance_terms_) {
      (void)key;
      slot.value = Real(0);
      slot.recorded = false;
    }
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
  std::map<std::string, Real> accepted_balance_terms(std::string_view route,
                                                     std::string_view runtime) const {
    require_transaction_authorities_bound_("ProgramRuntimeState::accepted_balance_terms");
    require_balance_route(route, runtime);
    const auto route_slot = step_balance_terms_.find(route);
    if (route_slot == step_balance_terms_.end())
      throw std::logic_error(std::string(runtime) +
                             "::_accepted_balance_terms route was not declared before bind");
    std::map<std::string, Real> result;
    const bool any_recorded =
        std::any_of(route_slot->second.recorded.begin(), route_slot->second.recorded.end(),
                    [](bool recorded) { return recorded; });
    if (!any_recorded && balance_step_completed_ && !balance_program_was_due_) {
      for (const std::string_view term : kBalanceTerms)
        result.emplace(std::string(term), Real(0));
      return result;
    }
    for (std::size_t index = 0; index < kBalanceTerms.size(); ++index) {
      const std::string_view term = kBalanceTerms[index];
      if (!route_slot->second.recorded[index])
        throw std::runtime_error(
            std::string(runtime) +
            "::_accepted_balance_terms: current native attempt omitted term '" + std::string(term) +
            "'; Program.record_balance must publish all five terms");
      if (!std::isfinite(static_cast<double>(route_slot->second.values[index])))
        throw std::runtime_error(
            std::string(runtime) +
            "::_accepted_balance_terms: current native attempt produced non-finite term '" +
            std::string(term) + "'");
      result.emplace(std::string(term), route_slot->second.values[index]);
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
      std::string_view route, int runtime_block, int component, const std::vector<int>& levels,
      const std::vector<std::string>& automatic_terms, std::string_view runtime) const {
    require_transaction_authorities_bound_("ProgramRuntimeState::selected_accepted_balance_terms");
    require_balance_route(route, runtime);
    if (runtime_block < 0 || component < 0)
      throw std::invalid_argument(
          std::string(runtime) +
          "::_selected_accepted_balance_terms requires non-negative coordinates");
    if (levels.empty() || levels.front() < 0 ||
        std::adjacent_find(levels.begin(), levels.end(),
                           [](int left, int right) { return right != left + 1; }) != levels.end())
      throw std::invalid_argument(
          std::string(runtime) +
          "::_selected_accepted_balance_terms requires a non-empty contiguous hierarchy");
    if (!std::is_sorted(automatic_terms.begin(), automatic_terms.end()) ||
        std::adjacent_find(automatic_terms.begin(), automatic_terms.end()) != automatic_terms.end())
      throw std::invalid_argument(
          std::string(runtime) +
          "::_selected_accepted_balance_terms requires sorted unique automatic terms");
    for (const std::string& term : automatic_terms)
      if (term != "reflux" && term != "projection")
        throw std::invalid_argument(
            std::string(runtime) +
            "::_selected_accepted_balance_terms has no native producer for '" + std::string(term) +
            "'");

    const auto route_slot = step_balance_terms_.find(route);
    if (route_slot == step_balance_terms_.end())
      throw std::logic_error(
          std::string(runtime) +
          "::_selected_accepted_balance_terms route was not declared before bind");
    std::map<std::string, Real> result;
    const bool any_recorded =
        std::any_of(route_slot->second.recorded.begin(), route_slot->second.recorded.end(),
                    [](bool recorded) { return recorded; });
    if (!any_recorded && balance_step_completed_ && !balance_program_was_due_) {
      for (const std::string_view term : kBalanceTerms)
        result.emplace(std::string(term), Real(0));
      return result;
    }
    for (std::size_t term_index = 0; term_index < kBalanceTerms.size(); ++term_index) {
      const std::string_view term = kBalanceTerms[term_index];
      const bool automatic =
          std::binary_search(automatic_terms.begin(), automatic_terms.end(), term);
      const bool authored = route_slot->second.recorded[term_index];
      if (!automatic) {
        if (!authored)
          throw std::runtime_error(
              std::string(runtime) +
              "::_selected_accepted_balance_terms: current native attempt omitted term '" +
              std::string(term) +
              "'; Program.record_balance must publish every non-automatic term");
        if (!std::isfinite(static_cast<double>(route_slot->second.values[term_index])))
          throw std::runtime_error(
              std::string(runtime) +
              "::_selected_accepted_balance_terms: current native attempt produced "
              "non-finite term '" +
              std::string(term) + "'");
        result.emplace(std::string(term), route_slot->second.values[term_index]);
        continue;
      }
      if (authored)
        throw std::runtime_error(std::string(runtime) +
                                 "::_selected_accepted_balance_terms: term '" + std::string(term) +
                                 "' has both Program and native producer authority");

      Real value = Real(0);
      const std::size_t expected = term == "reflux" ? levels.size() - 1 : levels.size();
      const auto automatic_term = automatic_balance_term_(term, runtime);
      for (std::size_t index = 0; index < expected; ++index) {
        const AutomaticBalanceKey key{runtime_block, levels[index], component, automatic_term};
        const auto found = automatic_balance_terms_.find(key);
        if (found == automatic_balance_terms_.end() || !found->second.recorded)
          throw std::runtime_error(
              std::string(runtime) +
              "::_selected_accepted_balance_terms: native producer omitted term '" +
              std::string(term) + "' at level " + std::to_string(levels[index]));
        if (!std::isfinite(static_cast<double>(found->second.value)))
          throw std::runtime_error(
              std::string(runtime) +
              "::_selected_accepted_balance_terms: native producer returned non-finite "
              "term '" +
              std::string(term) + "'");
        value += found->second.value;
      }
      if (!std::isfinite(static_cast<double>(value)))
        throw std::runtime_error(
            std::string(runtime) +
            "::_selected_accepted_balance_terms: native term accumulation overflowed");
      result.emplace(std::string(term), value);
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

  void note_step_projection(std::string_view name) {
    require_transaction_authorities_bound_("ProgramRuntimeState::note_step_projection");
    if (name.empty())
      throw std::invalid_argument("Program step projection identity cannot be empty");
    const auto found =
        std::find_if(step_projections_.begin(), step_projections_.end(),
                     [name](const StepProjectionSlot& slot) { return slot.identity == name; });
    if (found == step_projections_.end())
      throw std::logic_error(
          "ProgramRuntimeState::note_step_projection identity was not declared before bind");
    found->executed = true;
  }

  /// Allocation-free view used while an accepted-step transaction is hot. The view retains the
  /// bind-frozen declaration order and exposes activity bits only; callers must not materialize a
  /// string container until the native generation has sealed or rolled back.
  [[nodiscard]] std::span<const StepProjectionSlot> step_projection_report_view() const noexcept {
    return step_projections_;
  }

  std::vector<std::string> consume_step_projections() {
    std::vector<std::string> result;
    const auto report = step_projection_report_view();
    result.reserve(report.size());
    for (auto& slot : step_projections_) {
      if (slot.executed)
        result.push_back(slot.identity);
      slot.executed = false;
    }
    return result;
  }

  /// Seed a program block's RuntimeParams to its declaration defaults (ADC-510 / ADC-508). Idempotent
  /// (re-seeding resets to the baseline). Called by install. DEFENCE IN DEPTH (ADC-610): a block with
  /// more than kMaxRuntimeParams params is REJECTED here with a user-facing error instead of being
  /// SILENTLY TRUNCATED into the fixed-size device carrier -- the Python codegen enforces the same bound
  /// upstream, so this only fires for a hand-built .so with bogus parameter-table records.
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
