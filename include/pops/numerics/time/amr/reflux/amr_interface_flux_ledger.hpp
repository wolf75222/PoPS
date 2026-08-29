#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/numerics/time/amr/levels/amr_clock.hpp>

namespace pops::amr {

namespace detail {

using InterfaceFluxClockCoordinate = std::tuple<int, std::int64_t, std::int64_t, std::int64_t>;
using InterfaceFluxWindowCoordinate =
    std::tuple<InterfaceFluxClockCoordinate, InterfaceFluxClockCoordinate>;

inline InterfaceFluxClockCoordinate interface_flux_clock_coordinate(const ClockStamp& value) {
  return {value.level, value.macro_step, value.phase.numerator, value.phase.denominator};
}

inline InterfaceFluxWindowCoordinate interface_flux_window_coordinate(const ClockWindow& value) {
  return {interface_flux_clock_coordinate(value.begin), interface_flux_clock_coordinate(value.end)};
}

template <class Payload, class = void>
struct InterfaceFluxSnapshotPayloadElement {
  using type = Payload;
  static constexpr bool sequence = false;
};

template <class Payload>
struct InterfaceFluxSnapshotPayloadElement<Payload, std::void_t<typename Payload::value_type>> {
  using type = typename Payload::value_type;
  static constexpr bool sequence = true;
};

}  // namespace detail

/// The two consumers of one canonical coarse/fine interface flux.  Their
/// outward normals are opposite, so the same physical flux is accumulated
/// with opposite signs.
enum class InterfaceFluxOrientation { CoarseOutward, FineOutward };

/// Complete identity of one graph-authored interface-flux fragment.  Exact
/// clock coordinates, rather than rounded physical time, define temporal
/// identity.  Physical times remain validated duration metadata.
struct InterfaceFluxFragmentKey {
  std::string interface_identity;
  std::uint64_t topology_epoch = 0;
  int coarse_level = 0;
  int fine_level = 1;
  ClockStamp clock;
  std::string stage_identity;
  std::string graph_identity;
  std::string rate_identity;
  std::string application_identity;
  ClockWindow interval;
  InterfaceFluxOrientation orientation = InterfaceFluxOrientation::CoarseOutward;
  std::size_t left_block = 0;
  std::size_t right_block = 1;

  friend bool operator<(const InterfaceFluxFragmentKey& a, const InterfaceFluxFragmentKey& b) {
    return std::make_tuple(a.interface_identity, a.topology_epoch, a.coarse_level, a.fine_level,
                           detail::interface_flux_clock_coordinate(a.clock), a.stage_identity,
                           a.graph_identity, a.rate_identity, a.application_identity,
                           detail::interface_flux_window_coordinate(a.interval), a.orientation,
                           a.left_block, a.right_block) <
           std::make_tuple(b.interface_identity, b.topology_epoch, b.coarse_level, b.fine_level,
                           detail::interface_flux_clock_coordinate(b.clock), b.stage_identity,
                           b.graph_identity, b.rate_identity, b.application_identity,
                           detail::interface_flux_window_coordinate(b.interval), b.orientation,
                           b.left_block, b.right_block);
  }
};

/// Fragment metadata that is numerical rather than identifying. Face measure
/// is retained for auditing. The paired shared-interface RHS already applies
/// geometry exactly once; this ledger is provenance and must not inject the
/// same flux into reflux a second time. Substep duration is the authored local
/// dt, never reconstructed from rounded physical timestamps.
struct InterfaceFluxFragmentMeasure {
  Rational stage_weight{1, 1};
  double face_measure = 0.0;
  double substep_duration = 0.0;
  bool stage_weight_resolved = true;
};

template <class Payload>
struct InterfaceFluxFragment {
  InterfaceFluxFragmentKey key;
  InterfaceFluxFragmentMeasure measure;
  Payload payload;
};

/// Identity after graph stages have been integrated.  Orientation remains
/// explicit because the coarse and fine consumers receive opposite updates.
struct InterfaceFluxAccumulationKey {
  std::string interface_identity;
  std::uint64_t topology_epoch = 0;
  int coarse_level = 0;
  int fine_level = 1;
  std::string graph_identity;
  std::string rate_identity;
  std::string application_identity;
  ClockWindow interval;
  InterfaceFluxOrientation orientation = InterfaceFluxOrientation::CoarseOutward;
  std::size_t left_block = 0;
  std::size_t right_block = 1;

  friend bool operator<(const InterfaceFluxAccumulationKey& a,
                        const InterfaceFluxAccumulationKey& b) {
    return std::make_tuple(a.interface_identity, a.topology_epoch, a.coarse_level, a.fine_level,
                           a.graph_identity, a.rate_identity, a.application_identity,
                           detail::interface_flux_window_coordinate(a.interval), a.orientation,
                           a.left_block, a.right_block) <
           std::make_tuple(b.interface_identity, b.topology_epoch, b.coarse_level, b.fine_level,
                           b.graph_identity, b.rate_identity, b.application_identity,
                           detail::interface_flux_window_coordinate(b.interval), b.orientation,
                           b.left_block, b.right_block);
  }
};

inline double interface_flux_orientation_sign(InterfaceFluxOrientation orientation) {
  switch (orientation) {
    case InterfaceFluxOrientation::CoarseOutward:
      return -1.0;
    case InterfaceFluxOrientation::FineOutward:
      return 1.0;
  }
  throw std::invalid_argument("invalid AMR interface-flux orientation");
}

inline InterfaceFluxAccumulationKey interface_flux_accumulation_key(
    const InterfaceFluxFragmentKey& key) {
  return {key.interface_identity, key.topology_epoch, key.coarse_level,         key.fine_level,
          key.graph_identity,     key.rate_identity,  key.application_identity, key.interval,
          key.orientation,        key.left_block,     key.right_block};
}

/// Convert one physical-flux sample to an oriented, time-integrated audit
/// contribution. Geometry stays explicit in the measure but is not multiplied
/// here because the shared-interface RHS already applied its unique face/cell
/// conversion.
inline double interface_flux_fragment_scale(const InterfaceFluxFragmentKey& key,
                                            const InterfaceFluxFragmentMeasure& measure) {
  return interface_flux_orientation_sign(key.orientation) * measure.stage_weight.value() *
         measure.substep_duration;
}

/// Artifact- and hierarchy-authenticated capacity for one accepted root window.  The ledger retains
/// exactly the latest accepted window; a new outer commit replaces the preceding image atomically.
struct InterfaceFluxLedgerBudget {
  std::size_t max_fragments_per_window = 0;
  std::size_t max_payload_terms_per_window = 0;
  std::size_t max_transaction_depth = 1;
  /// Exact total character ceiling for all dynamic fragment identities in one window.  It is
  /// supplied by ProgramFluxBudgetRecord at bind and owns the snapshot string arena budget.
  std::size_t max_identity_characters = 0;
  std::string exact_contract;

  friend bool operator==(const InterfaceFluxLedgerBudget&,
                         const InterfaceFluxLedgerBudget&) = default;
};

/// Validate the complete topology, clock, geometry and duration identity shared by live
/// accumulation and accepted-state restart.  Accepted-state readers additionally require a resolved
/// Program stage weight; live ledgers may carry the unresolved placeholder until contribution
/// algebra closes the outer attempt.
inline void validate_interface_flux_fragment(const InterfaceFluxFragmentKey& key,
                                             const InterfaceFluxFragmentMeasure& measure,
                                             std::uint64_t topology_epoch) {
  if (key.topology_epoch != topology_epoch)
    throw std::invalid_argument("AMR interface-flux fragment uses a stale topology epoch");
  if (key.interface_identity.empty() || key.stage_identity.empty() || key.graph_identity.empty() ||
      key.rate_identity.empty() || key.application_identity.empty() || key.coarse_level < 0 ||
      key.fine_level != key.coarse_level + 1 || key.left_block == key.right_block)
    throw std::invalid_argument("AMR interface-flux fragment is not fully qualified");
  if (key.clock.level != key.coarse_level && key.clock.level != key.fine_level)
    throw std::invalid_argument("AMR interface-flux clock is outside its coarse/fine level pair");
  switch (key.orientation) {
    case InterfaceFluxOrientation::CoarseOutward:
    case InterfaceFluxOrientation::FineOutward:
      break;
    default:
      throw std::invalid_argument("invalid AMR interface-flux orientation");
  }
  if (key.interval.begin.level != key.clock.level || key.interval.end.level != key.clock.level ||
      key.interval.begin.macro_step != key.clock.macro_step ||
      key.interval.end.macro_step != key.clock.macro_step ||
      !(key.interval.begin.phase < key.interval.end.phase) ||
      key.clock.phase < key.interval.begin.phase || key.interval.end.phase < key.clock.phase)
    throw std::invalid_argument("AMR interface-flux clock is outside its exact temporal interval");
  const double begin_time = key.interval.begin.physical_time;
  const double end_time = key.interval.end.physical_time;
  const double reconstructed_duration = end_time - begin_time;
  const double timestamp_scale =
      std::max({1.0, std::abs(begin_time), std::abs(end_time), std::abs(measure.substep_duration)});
  const double timestamp_tolerance = 8.0 * std::numeric_limits<double>::epsilon() * timestamp_scale;
  if (!std::isfinite(key.clock.physical_time) || !std::isfinite(begin_time) ||
      !std::isfinite(end_time) || !(end_time > begin_time) ||
      key.clock.physical_time < begin_time || key.clock.physical_time > end_time ||
      !(measure.face_measure > 0.0) || !std::isfinite(measure.face_measure) ||
      !(measure.substep_duration > 0.0) || !std::isfinite(measure.substep_duration) ||
      std::abs(reconstructed_duration - measure.substep_duration) > timestamp_tolerance)
    throw std::invalid_argument(
        "AMR interface-flux fragment requires consistent finite positive geometry/time");
}

/// Transactional store for refined multi-block interface fluxes.  Pending
/// fragments are invisible to aggregate() until the outer transaction commits;
/// rollback therefore cannot publish a rejected stage.  The ledger is bound to
/// one topology epoch and rejects stale fragments before storing them.
template <class Payload>
class TransactionalInterfaceFluxLedger {
 public:
  using Entry = InterfaceFluxFragment<Payload>;
  using payload_element = typename detail::InterfaceFluxSnapshotPayloadElement<Payload>::type;
  static_assert(
      std::is_trivially_copyable_v<payload_element>,
      "AMR interface-flux dense payload elements must be copyable without hot allocation");

  /// Bind-sized, allocation-free snapshot representation.  It deliberately stores no dynamic
  /// string or payload owner per slot: all identities occupy one authenticated character arena
  /// and all payload terms one flat arena.  The dense images are the live ledger authority;
  /// ordinary Entry values are cold checkpoint/diagnostic materializations only.
  struct SnapshotTextRef {
    std::size_t offset = 0;
    std::size_t size = 0;
  };

  struct SnapshotSlot {
    std::array<SnapshotTextRef, 5> identities{};
    std::uint64_t topology_epoch = 0;
    int coarse_level = 0;
    int fine_level = 1;
    ClockStamp clock;
    ClockWindow interval;
    InterfaceFluxOrientation orientation = InterfaceFluxOrientation::CoarseOutward;
    std::size_t left_block = 0;
    std::size_t right_block = 1;
    InterfaceFluxFragmentMeasure measure;
    std::size_t payload_offset = 0;
    std::size_t payload_size = 0;
  };

  /// Non-owning ingress/egress representation.  Candidate execution only ever
  /// handles this view: it is copied directly into a bind-sized dense image.
  /// `Entry` remains a cold checkpoint/diagnostic DTO and is never retained by
  /// the ledger on the execution path.
  struct FragmentKeyView {
    std::string_view interface_identity;
    std::uint64_t topology_epoch = 0;
    int coarse_level = 0;
    int fine_level = 1;
    ClockStamp clock;
    std::string_view stage_identity;
    std::string_view graph_identity;
    std::string_view rate_identity;
    std::string_view application_identity;
    ClockWindow interval;
    InterfaceFluxOrientation orientation = InterfaceFluxOrientation::CoarseOutward;
    std::size_t left_block = 0;
    std::size_t right_block = 1;
  };

  struct FragmentInput {
    FragmentKeyView key;
    InterfaceFluxFragmentMeasure measure;
    std::span<const payload_element> payload;
  };

  struct FragmentView {
    FragmentKeyView key;
    const InterfaceFluxFragmentMeasure& measure;
    std::span<const payload_element> payload;
  };

  /// Test-visible witness of bind-sized carrier capacities.  It intentionally reports capacities,
  /// not mutable addresses: accepted 0 -> N -> M windows may change logical size but never grow a
  /// transaction carrier after bind.
  struct HotCarrierCapacities {
    std::size_t pending_entries = 0;
    std::size_t published_entries = 0;
    std::size_t begin_savepoints = 0;
    std::size_t commit_savepoints = 0;
    std::size_t accumulation_entries = 0;
    std::size_t begin_contract = 0;
    std::size_t commit_contract = 0;
    std::size_t accumulation_contract = 0;

    friend bool operator==(const HotCarrierCapacities&, const HotCarrierCapacities&) = default;
  };

  class PreparedBudget {
   public:
    PreparedBudget(PreparedBudget&&) noexcept = default;
    PreparedBudget& operator=(PreparedBudget&&) noexcept = default;
    PreparedBudget(const PreparedBudget&) = delete;
    PreparedBudget& operator=(const PreparedBudget&) = delete;
    std::string_view exact_contract() const noexcept { return budget_.exact_contract; }

   private:
    friend class TransactionalInterfaceFluxLedger;
    PreparedBudget() = default;
    InterfaceFluxLedgerBudget budget_;
    /// A complete bind-sized replacement.  Constructing it is the only path
    /// that may allocate for an inactive -> active budget transition; the
    /// eventual publication exchanges this image without reserve/resize.
    std::unique_ptr<TransactionalInterfaceFluxLedger> replacement_;
  };

  class PreparedBegin {
   public:
    PreparedBegin(PreparedBegin&& other) noexcept { *this = std::move(other); }
    PreparedBegin& operator=(PreparedBegin&& other) noexcept {
      if (this != &other) {
        cancel_();
        owner_ = std::exchange(other.owner_, nullptr);
        published_ = std::exchange(other.published_, false);
      }
      return *this;
    }
    PreparedBegin(const PreparedBegin&) = delete;
    PreparedBegin& operator=(const PreparedBegin&) = delete;
    ~PreparedBegin() { cancel_(); }

    std::string_view exact_contract() const noexcept {
      return owner_ == nullptr ? std::string_view{} : owner_->prepared_begin_contract_;
    }

   private:
    friend class TransactionalInterfaceFluxLedger;
    PreparedBegin() = default;
    explicit PreparedBegin(TransactionalInterfaceFluxLedger* owner) : owner_(owner) {}
    void cancel_() noexcept {
      if (owner_ != nullptr && !published_)
        owner_->prepared_begin_active_ = false;
      owner_ = nullptr;
    }
    TransactionalInterfaceFluxLedger* owner_ = nullptr;
    bool published_ = false;
  };

  class PreparedAccumulation {
   public:
    PreparedAccumulation(PreparedAccumulation&& other) noexcept { *this = std::move(other); }
    PreparedAccumulation& operator=(PreparedAccumulation&& other) noexcept {
      if (this != &other) {
        cancel_();
        owner_ = std::exchange(other.owner_, nullptr);
        published_ = std::exchange(other.published_, false);
      }
      return *this;
    }
    PreparedAccumulation(const PreparedAccumulation&) = delete;
    PreparedAccumulation& operator=(const PreparedAccumulation&) = delete;
    ~PreparedAccumulation() { cancel_(); }

    std::string_view exact_contract() const noexcept {
      return owner_ == nullptr ? std::string_view{} : owner_->prepared_accumulation_contract_;
    }

   private:
    friend class TransactionalInterfaceFluxLedger;
    PreparedAccumulation() = default;
    explicit PreparedAccumulation(TransactionalInterfaceFluxLedger* owner) : owner_(owner) {}
    void cancel_() noexcept {
      if (owner_ != nullptr && !published_)
        owner_->prepared_accumulation_active_ = false;
      owner_ = nullptr;
    }
    TransactionalInterfaceFluxLedger* owner_ = nullptr;
    bool published_ = false;
  };

  /// Complete candidate publication for one transaction close.  Preparing it copies every
  /// retained payload and reserves the final accepted vector while the live transaction and its
  /// rollback savepoint remain untouched.  Publication only swaps vectors, and the candidate then
  /// owns the prior live image until its caller crosses the enclosing accepted-state boundary.
  class PreparedCommit {
   public:
    PreparedCommit(PreparedCommit&& other) noexcept { *this = std::move(other); }
    PreparedCommit& operator=(PreparedCommit&& other) noexcept {
      if (this != &other) {
        finalize_();
        owner_ = std::exchange(other.owner_, nullptr);
        outer_ = std::exchange(other.outer_, false);
        published_ = std::exchange(other.published_, false);
        rollback_size_ = std::exchange(other.rollback_size_, 0);
        rollback_identity_ = std::exchange(other.rollback_identity_, 0);
        rollback_payload_ = std::exchange(other.rollback_payload_, 0);
      }
      return *this;
    }
    PreparedCommit(const PreparedCommit&) = delete;
    PreparedCommit& operator=(const PreparedCommit&) = delete;
    ~PreparedCommit() { finalize_(); }

    std::string_view exact_contract() const noexcept {
      return owner_ == nullptr ? std::string_view{} : owner_->prepared_commit_contract_;
    }
    bool published() const noexcept { return published_; }

   private:
    friend class TransactionalInterfaceFluxLedger;
    PreparedCommit() = default;
    PreparedCommit(TransactionalInterfaceFluxLedger* owner, bool outer, std::size_t rollback_size,
                   std::size_t rollback_identity, std::size_t rollback_payload)
        : owner_(owner),
          outer_(outer),
          rollback_size_(rollback_size),
          rollback_identity_(rollback_identity),
          rollback_payload_(rollback_payload) {}
    void finalize_() noexcept {
      if (owner_ != nullptr && published_)
        owner_->finalize_prepared_commit_noexcept_(*this);
      owner_ = nullptr;
    }
    TransactionalInterfaceFluxLedger* owner_ = nullptr;
    bool outer_ = false;
    bool published_ = false;
    std::size_t rollback_size_ = 0;
    std::size_t rollback_identity_ = 0;
    std::size_t rollback_payload_ = 0;
  };

  explicit TransactionalInterfaceFluxLedger(std::uint64_t topology_epoch,
                                            InterfaceFluxLedgerBudget budget)
      : topology_epoch_(topology_epoch), budget_(std::move(budget)) {
    validate_budget_();
    prime_transaction_carriers_();
    prime_dense_images_at_bind_();
  }

  std::uint64_t topology_epoch() const { return topology_epoch_; }
  bool in_transaction() const { return savepoint_size_ != 0; }
  std::size_t transaction_depth() const { return savepoint_size_; }
  std::size_t pending_size() const { return pending_.size; }
  std::size_t published_size() const { return published_.size; }
  bool empty() const { return pending_.size == 0 && published_.size == 0; }
  const InterfaceFluxLedgerBudget& budget() const noexcept { return budget_; }

  /// Cold bind/regrid-preparation seam for AcceptedContextSnapshot.  This primes the complete
  /// finite rollback image directly from ProgramFluxBudgetRecord bounds; it must never run after
  /// an accepted step or during a finalizer.
  void prime_snapshot_arenas_at_bind() {
    if (in_transaction())
      throw std::logic_error("AMR interface-flux snapshot cannot prime an active transaction");
    snapshot_arenas_primed_ = true;
  }

  /// Re-prime the non-payload carriers of a copied accepted image while installation is still
  /// cold.  A `std::string` copy deliberately carries its contents, not a source `reserve()`: a
  /// snapshot cloned from a bound ledger would otherwise lose the capacity needed to encode the
  /// next begin/commit contract after rollback swaps it back into the live owner.  This operation
  /// is therefore bind-only; it never changes dense images or a logical savepoint and refuses
  /// every active/prepared attempt.
  void prime_hot_carriers_at_bind() {
    if (in_transaction() || prepared_begin_active_ || prepared_commit_active_ ||
        prepared_accumulation_active_)
      throw std::logic_error("AMR interface-flux hot carrier cannot prime an active attempt");
    prime_transaction_carriers_();
  }

  std::size_t snapshot_identity_arena_capacity() const noexcept { return pending_.identity.size(); }
  std::size_t snapshot_payload_arena_capacity() const noexcept { return pending_.payload.size(); }
  HotCarrierCapacities hot_carrier_capacities() const noexcept {
    return {pending_.slots.capacity(),
            published_.slots.capacity(),
            prepared_begin_savepoints_.capacity(),
            prepared_commit_savepoints_.capacity(),
            prepared_accumulation_.slots.capacity(),
            prepared_begin_contract_.capacity(),
            prepared_commit_contract_.capacity(),
            prepared_accumulation_contract_.capacity()};
  }

  /// Refresh a resident accepted image without creating any entry, payload or identity storage.
  /// The complete interface graph is bind-sealed; a changed shape/key is therefore a new cold
  /// authority, never a reason to grow the rollback carrier during a candidate.
  void copy_from_preallocated(const TransactionalInterfaceFluxLedger& source) {
    require_preallocated_copy_from(source);
    copy_dense_image_(pending_, source.pending_);
    copy_dense_image_(published_, source.published_);
    std::copy_n(source.savepoints_.begin(), source.savepoint_size_, savepoints_.begin());
    savepoint_size_ = source.savepoint_size_;
  }

  /// Check a resident-copy operation without changing either ledger.  AMR accepted-state
  /// snapshots compose this with their other carriers so they can reject the full image before
  /// the first rollback-visible write.
  void require_preallocated_copy_from(const TransactionalInterfaceFluxLedger& source) const {
    require_preallocated_copy_contract_(source);
    require_dense_image_(pending_, source.pending_);
    require_dense_image_(published_, source.published_);
  }

  /// Cold bind-only witness.  The dense slot and arena images are created from
  /// the authenticated budget in the constructor; no per-Entry strings or
  /// payload containers are accepted or retained here.
  void prime_snapshot_slots_at_bind() {
    if (in_transaction())
      throw std::logic_error("AMR interface-flux snapshot cannot prime an active transaction");
    snapshot_slots_primed_ = true;
  }

  PreparedBudget prepare_budget(InterfaceFluxLedgerBudget budget) const {
    if (in_transaction())
      throw std::runtime_error("cannot replace an active AMR interface-flux ledger budget");
    validate_budget_(budget);
    PreparedBudget prepared;
    prepared.budget_ = std::move(budget);
    if (prepared.budget_ == budget_)
      return prepared;
    if (!empty())
      throw std::logic_error("AMR interface-flux ledger cannot replace a nonempty bound image");
    auto replacement =
        std::make_unique<TransactionalInterfaceFluxLedger>(topology_epoch_, prepared.budget_);
    replacement->snapshot_slots_primed_ = snapshot_slots_primed_;
    replacement->snapshot_arenas_primed_ = snapshot_arenas_primed_;
    prepared.replacement_ = std::move(replacement);
    return prepared;
  }

  void publish_prepared_budget(PreparedBudget& prepared) noexcept {
    if (prepared.budget_ == budget_) {
      if (prepared.replacement_ != nullptr)
        std::terminate();
      return;
    }
    if (prepared.replacement_ == nullptr || in_transaction() || !empty())
      std::terminate();
    swap_bound_image_noexcept_(*prepared.replacement_);
    prepared.replacement_.reset();
  }

  PreparedBegin prepare_begin() const {
    if (savepoint_size_ >= budget_.max_transaction_depth)
      throw std::runtime_error(
          "AMR interface-flux ledger transaction depth exceeds its authenticated budget");
    if (prepared_begin_active_)
      throw std::logic_error("AMR interface-flux ledger begin carrier was not consumed");
    if (savepoint_size_ + 1 > prepared_begin_savepoints_.capacity())
      throw std::logic_error("AMR interface-flux ledger begin carrier was not primed");
    std::copy_n(savepoints_.begin(), savepoint_size_, prepared_begin_savepoints_.begin());
    prepared_begin_size_ = savepoint_size_;
    prepared_begin_savepoints_[prepared_begin_size_++] = pending_.cursor();
    transaction_contract_into_(prepared_begin_contract_, "pops.amr-interface-flux-ledger.begin",
                               pending_, published_, prepared_begin_savepoints_,
                               prepared_begin_size_);
    prepared_begin_active_ = true;
    return PreparedBegin(const_cast<TransactionalInterfaceFluxLedger*>(this));
  }

  void publish_prepared_begin(PreparedBegin& prepared) noexcept {
    static_assert(std::is_nothrow_swappable_v<decltype(savepoints_)>);
    if (prepared.owner_ != this || !prepared_begin_active_ || prepared.published_)
      std::terminate();
    savepoints_.swap(prepared_begin_savepoints_);
    savepoint_size_ = prepared_begin_size_;
    prepared_begin_active_ = false;
    prepared.published_ = true;
    prepared.owner_ = nullptr;
  }

  void begin() {
    PreparedBegin prepared = prepare_begin();
    publish_prepared_begin(prepared);
  }

  PreparedCommit prepare_commit() const {
    if (!in_transaction())
      throw std::runtime_error("AMR interface-flux ledger commit without active transaction");
    if (savepoint_size_ == 1)
      for_each_(pending_, [&](FragmentView entry) {
        if (!entry.measure.stage_weight_resolved)
          throw std::runtime_error(
              "AMR interface-flux ledger cannot publish an unresolved Program stage weight");
      });
    if (prepared_commit_active_)
      throw std::logic_error("AMR interface-flux ledger commit carrier was not consumed");
    const bool outer = savepoint_size_ == 1;
    std::copy_n(savepoints_.begin(), savepoint_size_, prepared_commit_savepoints_.begin());
    prepared_commit_size_ = savepoint_size_ - 1;
    const DenseImage& candidate_pending = outer ? empty_image_() : pending_;
    const DenseImage& candidate_accepted = outer ? pending_ : published_;
    transaction_contract_into_(prepared_commit_contract_, "pops.amr-interface-flux-ledger.commit",
                               candidate_pending, candidate_accepted, prepared_commit_savepoints_,
                               prepared_commit_size_);
    prepared_commit_active_ = true;
    const Cursor rollback = savepoints_[savepoint_size_ - 1];
    return PreparedCommit(const_cast<TransactionalInterfaceFluxLedger*>(this), outer, rollback.size,
                          rollback.identity, rollback.payload);
  }

  void publish_prepared_commit(PreparedCommit& prepared) noexcept {
    static_assert(std::is_nothrow_swappable_v<decltype(pending_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(published_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(savepoints_)>);
    if (prepared.owner_ != this || !prepared_commit_active_ || prepared.published_)
      std::terminate();
    if (prepared.outer_)
      pending_.swap(published_);
    savepoints_.swap(prepared_commit_savepoints_);
    savepoint_size_ = prepared_commit_size_;
    prepared_commit_active_ = false;
    prepared.published_ = true;
  }

  void restore_prepared_commit(PreparedCommit& prepared) noexcept {
    if (!prepared.published_)
      std::terminate();
    if (prepared.owner_ != this || !prepared.published_)
      std::terminate();
    if (prepared.outer_)
      pending_.swap(published_);
    prepared_commit_savepoints_[prepared_commit_size_++] = {
        prepared.rollback_size_, prepared.rollback_identity_, prepared.rollback_payload_};
    savepoints_.swap(prepared_commit_savepoints_);
    savepoint_size_ = prepared_commit_size_;
    prepared.published_ = false;
  }

  void commit() {
    PreparedCommit prepared = prepare_commit();
    publish_prepared_commit(prepared);
  }

  void rollback() {
    if (!in_transaction())
      throw std::runtime_error("AMR interface-flux ledger rollback without active transaction");
    pending_.restore(savepoints_[savepoint_size_ - 1]);
    --savepoint_size_;
  }

  void clear() {
    if (in_transaction())
      throw std::runtime_error("cannot clear an active AMR interface-flux ledger transaction");
    pending_.clear();
    published_.clear();
  }

  /// Bind the ledger to a newer hierarchy.  Accepted fragments from the old
  /// topology cannot be reused after regrid.
  void advance_topology_epoch(std::uint64_t topology_epoch) {
    if (in_transaction())
      throw std::runtime_error("cannot advance AMR interface-flux topology during a transaction");
    if (topology_epoch < topology_epoch_)
      throw std::invalid_argument("AMR interface-flux topology epoch cannot move backward");
    if (topology_epoch == topology_epoch_)
      return;
    clear();
    topology_epoch_ = topology_epoch;
  }

  void accumulate(InterfaceFluxFragmentKey key, InterfaceFluxFragmentMeasure measure,
                  Payload payload) {
    const FragmentInput input{key_view_(key), measure, payload_span_(payload)};
    PreparedAccumulation prepared = prepare_accumulation(std::span<const FragmentInput>(&input, 1));
    publish_prepared_accumulation(prepared);
  }

  PreparedAccumulation prepare_accumulation(std::span<const FragmentInput> entries) const {
    if (!in_transaction())
      throw std::runtime_error("AMR interface-flux accumulation requires an active transaction");
    if (prepared_accumulation_active_)
      throw std::logic_error("AMR interface-flux accumulation carrier was not consumed");
    if (entries.size() > budget_.max_fragments_per_window - pending_.size)
      throw std::length_error("AMR interface-flux fragment budget exceeded before allocation");
    std::size_t terms = pending_.payload_used;
    std::size_t identities = pending_.identity_used;
    for (std::size_t candidate_index = 0; candidate_index < entries.size(); ++candidate_index) {
      const FragmentInput& candidate = entries[candidate_index];
      validate_(candidate.key, candidate.measure);
      if (contains_identity_(pending_, candidate.key) ||
          std::any_of(
              entries.begin(), entries.begin() + static_cast<std::ptrdiff_t>(candidate_index),
              [&](const FragmentInput& entry) { return same_identity_(entry.key, candidate.key); }))
        throw std::runtime_error(
            "AMR interface-flux attempt contains a duplicate stage/clock fragment identity");
      const std::size_t candidate_terms = candidate.payload.size();
      if (candidate_terms > budget_.max_payload_terms_per_window -
                                std::min(terms, budget_.max_payload_terms_per_window))
        throw std::length_error(
            "AMR interface-flux payload-term budget exceeded before allocation");
      terms += candidate_terms;
      const std::size_t chars = identity_size_(candidate.key);
      if (chars > budget_.max_identity_characters - identities)
        throw std::length_error(
            "AMR interface-flux identity arena budget exceeded before mutation");
      identities += chars;
    }
    if (entries.size() > prepared_accumulation_.slots.size())
      throw std::logic_error("AMR interface-flux accumulation carrier was not primed");
    prepared_accumulation_.clear();
    for (const FragmentInput& candidate : entries)
      encode_(prepared_accumulation_, candidate);
    transaction_contract_with_append_into_(
        prepared_accumulation_contract_, "pops.amr-interface-flux-ledger.accumulate", pending_,
        prepared_accumulation_, published_, savepoints_, savepoint_size_);
    prepared_accumulation_active_ = true;
    return PreparedAccumulation(const_cast<TransactionalInterfaceFluxLedger*>(this));
  }

  void publish_prepared_accumulation(PreparedAccumulation& prepared) noexcept {
    static_assert(std::is_nothrow_swappable_v<decltype(pending_)>);
    if (prepared.owner_ != this || !prepared_accumulation_active_ || prepared.published_)
      std::terminate();
    if (prepared_accumulation_.size > pending_.slots.size() - pending_.size ||
        prepared_accumulation_.identity_used > pending_.identity.size() - pending_.identity_used ||
        prepared_accumulation_.payload_used > pending_.payload.size() - pending_.payload_used)
      std::terminate();
    append_(pending_, prepared_accumulation_);
    prepared_accumulation_.clear();
    prepared_accumulation_active_ = false;
    prepared.published_ = true;
    prepared.owner_ = nullptr;
  }

  void resolve_pending_stage_weight(std::size_t index, Rational stage_weight) {
    if (!in_transaction())
      throw std::runtime_error(
          "AMR interface-flux stage-weight resolution requires an active transaction");
    if (transaction_depth() != 1)
      throw std::runtime_error(
          "AMR interface-flux stage weights resolve only in the outer attempt transaction");
    if (index >= pending_.size)
      throw std::out_of_range("AMR interface-flux pending fragment index is out of range");
    SnapshotSlot& entry = pending_.slots[index];
    if (entry.measure.stage_weight_resolved)
      throw std::logic_error("AMR interface-flux stage weight was already resolved");
    entry.measure.stage_weight = stage_weight;
    entry.measure.stage_weight_resolved = true;
  }

  template <class Axpy>
  std::map<InterfaceFluxAccumulationKey, Payload> aggregate(Axpy&& axpy) const {
    std::map<InterfaceFluxAccumulationKey, Payload> result;
    for_each_(published_, [&](FragmentView entry) {
      Entry cold = materialize_(entry);
      axpy(result[interface_flux_accumulation_key(cold.key)],
           interface_flux_fragment_scale(cold.key, cold.measure), cold.payload);
    });
    return result;
  }

  template <class Visitor>
  void for_each_pending(Visitor&& visitor) const {
    for_each_(pending_, std::forward<Visitor>(visitor));
  }
  template <class Visitor>
  void for_each_published(Visitor&& visitor) const {
    for_each_(published_, std::forward<Visitor>(visitor));
  }

  /// Cold checkpoint/diagnostic materialization only.
  std::vector<Entry> cold_pending_fragments() const { return materialize_all_(pending_); }
  std::vector<Entry> cold_published_fragments() const { return materialize_all_(published_); }

 private:
  struct Cursor {
    std::size_t size = 0, identity = 0, payload = 0;
  };
  struct DenseImage {
    std::vector<SnapshotSlot> slots;
    std::vector<char> identity;
    std::vector<payload_element> payload;
    std::size_t size = 0, identity_used = 0, payload_used = 0;
    Cursor cursor() const noexcept { return {size, identity_used, payload_used}; }
    void restore(Cursor cursor) noexcept {
      size = cursor.size;
      identity_used = cursor.identity;
      payload_used = cursor.payload;
    }
    void clear() noexcept { size = identity_used = payload_used = 0; }
    void swap(DenseImage& other) noexcept {
      slots.swap(other.slots);
      identity.swap(other.identity);
      payload.swap(other.payload);
      using std::swap;
      swap(size, other.size);
      swap(identity_used, other.identity_used);
      swap(payload_used, other.payload_used);
    }
  };

  static FragmentKeyView key_view_(const InterfaceFluxFragmentKey& key) {
    return {key.interface_identity, key.topology_epoch, key.coarse_level,
            key.fine_level,         key.clock,          key.stage_identity,
            key.graph_identity,     key.rate_identity,  key.application_identity,
            key.interval,           key.orientation,    key.left_block,
            key.right_block};
  }
  static std::span<const payload_element> payload_span_(const Payload& value) {
    if constexpr (detail::InterfaceFluxSnapshotPayloadElement<Payload>::sequence)
      return {value.data(), value.size()};
    else
      return {std::addressof(value), 1};
  }
  static std::size_t identity_size_(const FragmentKeyView& key) {
    return key.interface_identity.size() + key.stage_identity.size() + key.graph_identity.size() +
           key.rate_identity.size() + key.application_identity.size();
  }
  static InterfaceFluxFragmentKey materialize_key_(const FragmentKeyView& key) {
    return {std::string(key.interface_identity),
            key.topology_epoch,
            key.coarse_level,
            key.fine_level,
            key.clock,
            std::string(key.stage_identity),
            std::string(key.graph_identity),
            std::string(key.rate_identity),
            std::string(key.application_identity),
            key.interval,
            key.orientation,
            key.left_block,
            key.right_block};
  }
  static Payload materialize_payload_(std::span<const payload_element> input) {
    if constexpr (detail::InterfaceFluxSnapshotPayloadElement<Payload>::sequence)
      return Payload(input.begin(), input.end());
    else {
      if (input.size() != 1)
        throw std::logic_error("AMR interface-flux scalar payload shape changed");
      return input.front();
    }
  }
  static bool same_identity_(const FragmentKeyView& a, const FragmentKeyView& b) {
    return a.interface_identity == b.interface_identity && a.topology_epoch == b.topology_epoch &&
           a.coarse_level == b.coarse_level && a.fine_level == b.fine_level && a.clock == b.clock &&
           a.stage_identity == b.stage_identity && a.graph_identity == b.graph_identity &&
           a.rate_identity == b.rate_identity && a.application_identity == b.application_identity &&
           a.interval.begin.level == b.interval.begin.level &&
           a.interval.begin.macro_step == b.interval.begin.macro_step &&
           a.interval.begin.phase == b.interval.begin.phase &&
           a.interval.begin.physical_time == b.interval.begin.physical_time &&
           a.interval.end.level == b.interval.end.level &&
           a.interval.end.macro_step == b.interval.end.macro_step &&
           a.interval.end.phase == b.interval.end.phase &&
           a.interval.end.physical_time == b.interval.end.physical_time &&
           a.orientation == b.orientation && a.left_block == b.left_block &&
           a.right_block == b.right_block;
  }
  FragmentKeyView key_view_(const DenseImage& image, const SnapshotSlot& slot) const noexcept {
    const auto text = [&](SnapshotTextRef ref) {
      return std::string_view(image.identity.data() + ref.offset, ref.size);
    };
    return {text(slot.identities[0]),
            slot.topology_epoch,
            slot.coarse_level,
            slot.fine_level,
            slot.clock,
            text(slot.identities[1]),
            text(slot.identities[2]),
            text(slot.identities[3]),
            text(slot.identities[4]),
            slot.interval,
            slot.orientation,
            slot.left_block,
            slot.right_block};
  }
  FragmentView view_(const DenseImage& image, std::size_t index) const noexcept {
    const SnapshotSlot& slot = image.slots[index];
    return {key_view_(image, slot), slot.measure,
            std::span<const payload_element>(image.payload.data() + slot.payload_offset,
                                             slot.payload_size)};
  }
  template <class Visitor>
  void for_each_(const DenseImage& image, Visitor&& visitor) const {
    for (std::size_t index = 0; index < image.size; ++index)
      visitor(view_(image, index));
  }
  bool contains_identity_(const DenseImage& image, const FragmentKeyView& key) const {
    bool found = false;
    for_each_(image, [&](FragmentView entry) { found = found || same_identity_(entry.key, key); });
    return found;
  }
  static void copy_text_(SnapshotTextRef& ref, DenseImage& image, std::string_view text) noexcept {
    ref = {image.identity_used, text.size()};
    std::copy(text.begin(), text.end(),
              image.identity.begin() + static_cast<std::ptrdiff_t>(image.identity_used));
    image.identity_used += text.size();
  }
  void encode_(DenseImage& image, const FragmentInput& input) const noexcept {
    SnapshotSlot& slot = image.slots[image.size++];
    copy_text_(slot.identities[0], image, input.key.interface_identity);
    copy_text_(slot.identities[1], image, input.key.stage_identity);
    copy_text_(slot.identities[2], image, input.key.graph_identity);
    copy_text_(slot.identities[3], image, input.key.rate_identity);
    copy_text_(slot.identities[4], image, input.key.application_identity);
    slot.topology_epoch = input.key.topology_epoch;
    slot.coarse_level = input.key.coarse_level;
    slot.fine_level = input.key.fine_level;
    slot.clock = input.key.clock;
    slot.interval = input.key.interval;
    slot.orientation = input.key.orientation;
    slot.left_block = input.key.left_block;
    slot.right_block = input.key.right_block;
    slot.measure = input.measure;
    slot.payload_offset = image.payload_used;
    slot.payload_size = input.payload.size();
    std::copy(input.payload.begin(), input.payload.end(),
              image.payload.begin() + static_cast<std::ptrdiff_t>(image.payload_used));
    image.payload_used += input.payload.size();
  }
  void append_(DenseImage& destination, const DenseImage& source) noexcept {
    for_each_(source, [&](FragmentView view) {
      encode_(destination, {view.key, view.measure, view.payload});
    });
  }
  Entry materialize_(FragmentView view) const {
    return {materialize_key_(view.key), view.measure, materialize_payload_(view.payload)};
  }
  std::vector<Entry> materialize_all_(const DenseImage& image) const {
    std::vector<Entry> result;
    result.reserve(image.size);
    for_each_(image, [&](FragmentView view) { result.push_back(materialize_(view)); });
    return result;
  }
  void validate_(const FragmentKeyView& key, const InterfaceFluxFragmentMeasure& measure) const {
    if (key.topology_epoch != topology_epoch_)
      throw std::invalid_argument("AMR interface-flux fragment uses a stale topology epoch");
    if (key.interface_identity.empty() || key.stage_identity.empty() ||
        key.graph_identity.empty() || key.rate_identity.empty() ||
        key.application_identity.empty() || key.coarse_level < 0 ||
        key.fine_level != key.coarse_level + 1 || key.left_block == key.right_block)
      throw std::invalid_argument("AMR interface-flux fragment is not fully qualified");
    if (key.clock.level != key.coarse_level && key.clock.level != key.fine_level)
      throw std::invalid_argument("AMR interface-flux clock is outside its coarse/fine level pair");
    if (key.orientation != InterfaceFluxOrientation::CoarseOutward &&
        key.orientation != InterfaceFluxOrientation::FineOutward)
      throw std::invalid_argument("invalid AMR interface-flux orientation");
    if (key.interval.begin.level != key.clock.level || key.interval.end.level != key.clock.level ||
        key.interval.begin.macro_step != key.clock.macro_step ||
        key.interval.end.macro_step != key.clock.macro_step ||
        !(key.interval.begin.phase < key.interval.end.phase) ||
        key.clock.phase < key.interval.begin.phase || key.interval.end.phase < key.clock.phase)
      throw std::invalid_argument(
          "AMR interface-flux clock is outside its exact temporal interval");
    const double begin_time = key.interval.begin.physical_time;
    const double end_time = key.interval.end.physical_time;
    const double duration = end_time - begin_time;
    const double scale = std::max(
        {1.0, std::abs(begin_time), std::abs(end_time), std::abs(measure.substep_duration)});
    const double tolerance = 8.0 * std::numeric_limits<double>::epsilon() * scale;
    if (!std::isfinite(key.clock.physical_time) || !std::isfinite(begin_time) ||
        !std::isfinite(end_time) || !(end_time > begin_time) ||
        key.clock.physical_time < begin_time || key.clock.physical_time > end_time ||
        !(measure.face_measure > 0.0) || !std::isfinite(measure.face_measure) ||
        !(measure.substep_duration > 0.0) || !std::isfinite(measure.substep_duration) ||
        std::abs(duration - measure.substep_duration) > tolerance)
      throw std::invalid_argument(
          "AMR interface-flux fragment requires consistent finite positive geometry/time");
  }
  void require_dense_image_(const DenseImage& destination, const DenseImage& source) const {
    if (source.size > destination.slots.size() ||
        source.identity_used > destination.identity.size() ||
        source.payload_used > destination.payload.size())
      throw std::length_error("AMR interface-flux snapshot arena budget exceeded before mutation");
  }
  void require_preallocated_copy_contract_(const TransactionalInterfaceFluxLedger& source) const {
    if (in_transaction() || source.in_transaction() || topology_epoch_ != source.topology_epoch_ ||
        budget_ != source.budget_ || savepoint_size_ != source.savepoint_size_)
      throw std::logic_error("AMR interface-flux ledger changed its frozen authority");
    const HotCarrierCapacities destination = hot_carrier_capacities();
    const HotCarrierCapacities input = source.hot_carrier_capacities();
    if (savepoints_.capacity() < source.savepoints_.capacity() ||
        destination.begin_savepoints < input.begin_savepoints ||
        destination.commit_savepoints < input.commit_savepoints ||
        destination.begin_contract < input.begin_contract ||
        destination.commit_contract < input.commit_contract ||
        destination.accumulation_contract < input.accumulation_contract)
      throw std::logic_error("AMR interface-flux ledger hot carriers were not cold-primed");
  }
  void copy_dense_image_(DenseImage& destination, const DenseImage& source) {
    destination.clear();
    for_each_(source, [&](FragmentView entry) {
      encode_(destination, {entry.key, entry.measure, entry.payload});
    });
  }
  static const DenseImage& empty_image_() {
    static const DenseImage empty;
    return empty;
  }

  static void contract_require_(const std::string& destination, std::size_t count) {
    if (count > destination.capacity() - destination.size())
      throw std::logic_error("AMR interface-flux contract carrier was not primed (capacity=" +
                             std::to_string(destination.capacity()) +
                             ", size=" + std::to_string(destination.size()) +
                             ", append=" + std::to_string(count) + ")");
  }
  static void contract_bytes_(std::string& destination, const void* data, std::size_t size) {
    contract_require_(destination, size);
    destination.append(static_cast<const char*>(data), size);
  }
  template <class Value>
  static void contract_scalar_(std::string& destination, Value value) {
    static_assert(std::is_trivially_copyable_v<Value>);
    contract_bytes_(destination, std::addressof(value), sizeof(value));
  }
  static void contract_text_(std::string& destination, std::string_view text) {
    contract_scalar_(destination, static_cast<std::uint64_t>(text.size()));
    contract_bytes_(destination, text.data(), text.size());
  }
  static void contract_clock_(std::string& destination, const ClockStamp& clock) {
    contract_scalar_(destination, clock.level);
    contract_scalar_(destination, clock.macro_step);
    contract_scalar_(destination, clock.phase.numerator);
    contract_scalar_(destination, clock.phase.denominator);
    contract_scalar_(destination, clock.physical_time);
  }
  static void contract_view_(std::string& destination, FragmentView view) {
    contract_text_(destination, view.key.interface_identity);
    contract_scalar_(destination, view.key.topology_epoch);
    contract_scalar_(destination, view.key.coarse_level);
    contract_scalar_(destination, view.key.fine_level);
    contract_clock_(destination, view.key.clock);
    contract_text_(destination, view.key.stage_identity);
    contract_text_(destination, view.key.graph_identity);
    contract_text_(destination, view.key.rate_identity);
    contract_text_(destination, view.key.application_identity);
    contract_clock_(destination, view.key.interval.begin);
    contract_clock_(destination, view.key.interval.end);
    contract_scalar_(destination, view.key.orientation);
    contract_scalar_(destination, view.key.left_block);
    contract_scalar_(destination, view.key.right_block);
    contract_scalar_(destination, view.measure.stage_weight.numerator);
    contract_scalar_(destination, view.measure.stage_weight.denominator);
    contract_scalar_(destination, view.measure.face_measure);
    contract_scalar_(destination, view.measure.substep_duration);
    contract_scalar_(destination, view.measure.stage_weight_resolved);
    contract_scalar_(destination, static_cast<std::uint64_t>(view.payload.size()));
    for (const payload_element& payload : view.payload)
      contract_scalar_(destination, payload);
  }
  void transaction_contract_into_(std::string& destination, std::string_view phase,
                                  const DenseImage& pending, const DenseImage& accepted,
                                  std::span<const Cursor> savepoints,
                                  std::size_t savepoint_count) const {
    destination.clear();
    contract_text_(destination, phase);
    contract_scalar_(destination, std::uint32_t{2});
    contract_scalar_(destination, topology_epoch_);
    contract_text_(destination, budget_.exact_contract);
    contract_scalar_(destination, budget_.max_fragments_per_window);
    contract_scalar_(destination, budget_.max_payload_terms_per_window);
    contract_scalar_(destination, budget_.max_transaction_depth);
    contract_scalar_(destination, budget_.max_identity_characters);
    contract_scalar_(destination, savepoint_count);
    contract_scalar_(destination, pending.size);
    contract_scalar_(destination, accepted.size);
    for (std::size_t index = 0; index < savepoint_count; ++index) {
      contract_scalar_(destination, savepoints[index].size);
      contract_scalar_(destination, savepoints[index].identity);
      contract_scalar_(destination, savepoints[index].payload);
    }
    for_each_(pending, [&](FragmentView view) { contract_view_(destination, view); });
    for_each_(accepted, [&](FragmentView view) { contract_view_(destination, view); });
  }
  void transaction_contract_with_append_into_(std::string& destination, std::string_view phase,
                                              const DenseImage& pending, const DenseImage& appended,
                                              const DenseImage& accepted,
                                              std::span<const Cursor> savepoints,
                                              std::size_t savepoint_count) const {
    destination.clear();
    contract_text_(destination, phase);
    contract_scalar_(destination, std::uint32_t{2});
    contract_scalar_(destination, topology_epoch_);
    contract_text_(destination, budget_.exact_contract);
    contract_scalar_(destination, budget_.max_fragments_per_window);
    contract_scalar_(destination, budget_.max_payload_terms_per_window);
    contract_scalar_(destination, budget_.max_transaction_depth);
    contract_scalar_(destination, budget_.max_identity_characters);
    contract_scalar_(destination, savepoint_count);
    contract_scalar_(destination, pending.size + appended.size);
    contract_scalar_(destination, accepted.size);
    for (std::size_t index = 0; index < savepoint_count; ++index) {
      contract_scalar_(destination, savepoints[index].size);
      contract_scalar_(destination, savepoints[index].identity);
      contract_scalar_(destination, savepoints[index].payload);
    }
    for_each_(pending, [&](FragmentView view) { contract_view_(destination, view); });
    for_each_(appended, [&](FragmentView view) { contract_view_(destination, view); });
    for_each_(accepted, [&](FragmentView view) { contract_view_(destination, view); });
  }

  static void validate_budget_(const InterfaceFluxLedgerBudget& budget) {
    if (budget.exact_contract.empty() || budget.max_transaction_depth == 0)
      throw std::invalid_argument("AMR interface-flux ledger requires an exact budget contract");
    if ((budget.max_fragments_per_window == 0) != (budget.max_payload_terms_per_window == 0))
      throw std::invalid_argument(
          "AMR interface-flux ledger budget must be either inactive or fully bounded");
    if ((budget.max_fragments_per_window == 0) != (budget.max_identity_characters == 0))
      throw std::invalid_argument(
          "AMR interface-flux ledger identity arena must be either inactive or fully bounded");
  }

  void validate_budget_() const { validate_budget_(budget_); }

  void prime_transaction_carriers_() {
    // Constructor/bind time only.  The vector and contract carriers are subsequently reused by
    // PreparedBegin/PreparedCommit with size-only resets.
    savepoints_.resize(budget_.max_transaction_depth);
    prepared_begin_savepoints_.resize(budget_.max_transaction_depth);
    prepared_commit_savepoints_.resize(budget_.max_transaction_depth);
    const auto checked_add = [](std::size_t left, std::size_t right) {
      if (left > std::numeric_limits<std::size_t>::max() - right)
        throw std::length_error("AMR interface-flux transaction contract capacity exceeds size_t");
      return left + right;
    };
    std::size_t contract_capacity = budget_.exact_contract.size();
    contract_capacity = checked_add(contract_capacity, budget_.max_identity_characters);
    contract_capacity =
        checked_add(contract_capacity,
                    budget_.max_fragments_per_window >
                            (std::numeric_limits<std::size_t>::max() - contract_capacity) / 320
                        ? std::numeric_limits<std::size_t>::max()
                        : budget_.max_fragments_per_window * 320);
    contract_capacity =
        checked_add(contract_capacity,
                    budget_.max_payload_terms_per_window >
                            (std::numeric_limits<std::size_t>::max() - contract_capacity) / 32
                        ? std::numeric_limits<std::size_t>::max()
                        : budget_.max_payload_terms_per_window * 32);
    contract_capacity = checked_add(contract_capacity, 256);
    prepared_begin_contract_.reserve(contract_capacity);
    prepared_commit_contract_.reserve(contract_capacity);
    prepared_accumulation_contract_.reserve(contract_capacity);
  }

  void finalize_prepared_commit_noexcept_(PreparedCommit& prepared) noexcept {
    if (!prepared.outer_ || !prepared.published_)
      return;
    // The old published image is retained for compensation until finalization,
    // then discarded by a logical-size reset only.
    pending_.clear();
    prepared.published_ = false;
  }

  void swap_bound_image_noexcept_(TransactionalInterfaceFluxLedger& other) noexcept {
    static_assert(std::is_nothrow_swappable_v<InterfaceFluxLedgerBudget>);
    static_assert(std::is_nothrow_swappable_v<DenseImage>);
    static_assert(std::is_nothrow_swappable_v<decltype(savepoints_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(prepared_begin_savepoints_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(prepared_commit_savepoints_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(prepared_begin_contract_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(prepared_commit_contract_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(prepared_accumulation_contract_)>);
    using std::swap;
    swap(topology_epoch_, other.topology_epoch_);
    swap(budget_, other.budget_);
    pending_.swap(other.pending_);
    published_.swap(other.published_);
    savepoints_.swap(other.savepoints_);
    swap(savepoint_size_, other.savepoint_size_);
    prepared_begin_savepoints_.swap(other.prepared_begin_savepoints_);
    swap(prepared_begin_size_, other.prepared_begin_size_);
    prepared_commit_savepoints_.swap(other.prepared_commit_savepoints_);
    swap(prepared_commit_size_, other.prepared_commit_size_);
    prepared_accumulation_.swap(other.prepared_accumulation_);
    prepared_begin_contract_.swap(other.prepared_begin_contract_);
    prepared_commit_contract_.swap(other.prepared_commit_contract_);
    prepared_accumulation_contract_.swap(other.prepared_accumulation_contract_);
    swap(prepared_begin_active_, other.prepared_begin_active_);
    swap(prepared_commit_active_, other.prepared_commit_active_);
    swap(prepared_accumulation_active_, other.prepared_accumulation_active_);
    swap(snapshot_slots_primed_, other.snapshot_slots_primed_);
    swap(snapshot_arenas_primed_, other.snapshot_arenas_primed_);
  }

  std::uint64_t topology_epoch_;
  InterfaceFluxLedgerBudget budget_;
  DenseImage pending_;
  DenseImage published_;
  std::vector<Cursor> savepoints_;
  std::size_t savepoint_size_ = 0;
  mutable std::vector<Cursor> prepared_begin_savepoints_;
  mutable std::size_t prepared_begin_size_ = 0;
  mutable std::vector<Cursor> prepared_commit_savepoints_;
  mutable std::size_t prepared_commit_size_ = 0;
  mutable DenseImage prepared_accumulation_;
  mutable std::string prepared_begin_contract_;
  mutable std::string prepared_commit_contract_;
  mutable std::string prepared_accumulation_contract_;
  mutable bool prepared_begin_active_ = false;
  mutable bool prepared_commit_active_ = false;
  mutable bool prepared_accumulation_active_ = false;
  bool snapshot_slots_primed_ = false;
  bool snapshot_arenas_primed_ = false;

  void prime_dense_images_at_bind_() {
    const auto prime = [&](DenseImage& image) {
      image.slots.resize(budget_.max_fragments_per_window);
      image.identity.resize(budget_.max_identity_characters);
      image.payload.resize(budget_.max_payload_terms_per_window);
    };
    prime(pending_);
    prime(published_);
    prime(prepared_accumulation_);
  }
};

}  // namespace pops::amr
