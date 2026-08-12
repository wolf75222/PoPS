#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <stdexcept>
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
  };

  class PreparedBegin {
   public:
    PreparedBegin(PreparedBegin&&) noexcept = default;
    PreparedBegin& operator=(PreparedBegin&&) noexcept = default;
    PreparedBegin(const PreparedBegin&) = delete;
    PreparedBegin& operator=(const PreparedBegin&) = delete;

    std::string_view exact_contract() const noexcept { return exact_contract_; }

   private:
    friend class TransactionalInterfaceFluxLedger;
    PreparedBegin() = default;
    std::vector<std::size_t> savepoints_;
    std::string exact_contract_;
  };

  class PreparedAccumulation {
   public:
    PreparedAccumulation(PreparedAccumulation&&) noexcept = default;
    PreparedAccumulation& operator=(PreparedAccumulation&&) noexcept = default;
    PreparedAccumulation(const PreparedAccumulation&) = delete;
    PreparedAccumulation& operator=(const PreparedAccumulation&) = delete;

    std::string_view exact_contract() const noexcept { return exact_contract_; }

   private:
    friend class TransactionalInterfaceFluxLedger;
    PreparedAccumulation() = default;
    std::vector<Entry> pending_;
    std::string exact_contract_;
  };

  /// Complete candidate publication for one transaction close.  Preparing it copies every
  /// retained payload and reserves the final accepted vector while the live transaction and its
  /// rollback savepoint remain untouched.  Publication only swaps vectors, and the candidate then
  /// owns the prior live image until its caller crosses the enclosing accepted-state boundary.
  class PreparedCommit {
   public:
    PreparedCommit(PreparedCommit&&) noexcept = default;
    PreparedCommit& operator=(PreparedCommit&&) noexcept = default;
    PreparedCommit(const PreparedCommit&) = delete;
    PreparedCommit& operator=(const PreparedCommit&) = delete;

    std::string_view exact_contract() const noexcept { return exact_contract_; }
    bool published() const noexcept { return published_; }

   private:
    friend class TransactionalInterfaceFluxLedger;
    PreparedCommit() = default;
    std::vector<Entry> pending_;
    std::vector<Entry> accepted_;
    std::vector<std::size_t> savepoints_;
    std::string exact_contract_;
    bool published_ = false;
  };

  explicit TransactionalInterfaceFluxLedger(std::uint64_t topology_epoch,
                                            InterfaceFluxLedgerBudget budget)
      : topology_epoch_(topology_epoch), budget_(std::move(budget)) {
    validate_budget_();
  }

  std::uint64_t topology_epoch() const { return topology_epoch_; }
  bool in_transaction() const { return !savepoints_.empty(); }
  std::size_t transaction_depth() const { return savepoints_.size(); }
  std::size_t pending_size() const { return pending_.size(); }
  std::size_t published_size() const { return published_.size(); }
  bool empty() const { return pending_.empty() && published_.empty(); }
  const std::vector<Entry>& pending_entries() const { return pending_; }
  const std::vector<Entry>& published_entries() const { return published_; }
  const InterfaceFluxLedgerBudget& budget() const noexcept { return budget_; }

  PreparedBudget prepare_budget(InterfaceFluxLedgerBudget budget) const {
    if (in_transaction())
      throw std::runtime_error("cannot replace an active AMR interface-flux ledger budget");
    validate_budget_(budget);
    require_within_budget_(published_, budget, "accepted");
    PreparedBudget prepared;
    prepared.budget_ = std::move(budget);
    return prepared;
  }

  void publish_prepared_budget(PreparedBudget& prepared) noexcept {
    static_assert(std::is_nothrow_swappable_v<InterfaceFluxLedgerBudget>);
    std::swap(budget_, prepared.budget_);
  }

  PreparedBegin prepare_begin() const {
    if (savepoints_.size() >= budget_.max_transaction_depth)
      throw std::runtime_error(
          "AMR interface-flux ledger transaction depth exceeds its authenticated budget");
    PreparedBegin prepared;
    prepared.savepoints_ = savepoints_;
    prepared.savepoints_.push_back(pending_.size());
    prepared.exact_contract_ = transaction_contract_("pops.amr-interface-flux-ledger.begin",
                                                     pending_, published_, prepared.savepoints_);
    return prepared;
  }

  void publish_prepared_begin(PreparedBegin& prepared) noexcept {
    static_assert(std::is_nothrow_swappable_v<decltype(savepoints_)>);
    savepoints_.swap(prepared.savepoints_);
  }

  void begin() {
    PreparedBegin prepared = prepare_begin();
    publish_prepared_begin(prepared);
  }

  PreparedCommit prepare_commit() const {
    if (!in_transaction())
      throw std::runtime_error("AMR interface-flux ledger commit without active transaction");
    PreparedCommit prepared;
    require_within_budget_(pending_, budget_, "pending");
    prepared.pending_ = pending_;
    prepared.accepted_ = published_;
    prepared.savepoints_ = savepoints_;
    if (savepoints_.size() == 1)
      for (const Entry& entry : pending_)
        if (!entry.measure.stage_weight_resolved)
          throw std::runtime_error(
              "AMR interface-flux ledger cannot publish an unresolved Program stage weight");
    prepared.savepoints_.pop_back();
    if (prepared.savepoints_.empty()) {
      // Explicit retention policy: the accepted ledger is the latest complete root window, not a
      // topology-lifetime concatenation.  The prepared guard receives the previous accepted image
      // on publication and can restore it until the enclosing accepted transaction is irrevocable.
      prepared.accepted_ = pending_;
      prepared.pending_.clear();
    }
    prepared.exact_contract_ =
        transaction_contract_("pops.amr-interface-flux-ledger.commit", prepared.pending_,
                              prepared.accepted_, prepared.savepoints_);
    return prepared;
  }

  void publish_prepared_commit(PreparedCommit& prepared) noexcept {
    static_assert(std::is_nothrow_swappable_v<decltype(pending_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(published_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(savepoints_)>);
    pending_.swap(prepared.pending_);
    published_.swap(prepared.accepted_);
    savepoints_.swap(prepared.savepoints_);
    prepared.published_ = true;
  }

  void restore_prepared_commit(PreparedCommit& prepared) noexcept {
    if (!prepared.published_)
      std::terminate();
    pending_.swap(prepared.pending_);
    published_.swap(prepared.accepted_);
    savepoints_.swap(prepared.savepoints_);
    prepared.published_ = false;
  }

  void commit() {
    PreparedCommit prepared = prepare_commit();
    publish_prepared_commit(prepared);
  }

  void rollback() {
    if (!in_transaction())
      throw std::runtime_error("AMR interface-flux ledger rollback without active transaction");
    pending_.resize(savepoints_.back());
    savepoints_.pop_back();
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
    std::vector<Entry> entries;
    entries.reserve(1);
    entries.push_back({std::move(key), measure, std::move(payload)});
    PreparedAccumulation prepared = prepare_accumulation(std::move(entries));
    publish_prepared_accumulation(prepared);
  }

  PreparedAccumulation prepare_accumulation(std::vector<Entry> entries) const {
    if (!in_transaction())
      throw std::runtime_error("AMR interface-flux accumulation requires an active transaction");
    PreparedAccumulation prepared;
    if (entries.size() > budget_.max_fragments_per_window -
                             std::min(pending_.size(), budget_.max_fragments_per_window))
      throw std::length_error("AMR interface-flux fragment budget exceeded before allocation");
    std::size_t terms = payload_terms_(pending_);
    for (std::size_t candidate_index = 0; candidate_index < entries.size(); ++candidate_index) {
      const Entry& candidate = entries[candidate_index];
      validate_(candidate.key, candidate.measure);
      if (std::any_of(
              pending_.begin(), pending_.end(),
              [&](const Entry& entry) { return same_identity_(entry.key, candidate.key); }) ||
          std::any_of(entries.begin(),
                      entries.begin() + static_cast<std::ptrdiff_t>(candidate_index),
                      [&](const Entry& entry) { return same_identity_(entry.key, candidate.key); }))
        throw std::runtime_error(
            "AMR interface-flux attempt contains a duplicate stage/clock fragment identity");
      const std::size_t candidate_terms = payload_terms_(candidate.payload);
      if (candidate_terms > budget_.max_payload_terms_per_window -
                                std::min(terms, budget_.max_payload_terms_per_window))
        throw std::length_error(
            "AMR interface-flux payload-term budget exceeded before allocation");
      terms += candidate_terms;
    }
    prepared.pending_ = pending_;
    prepared.pending_.reserve(pending_.size() + entries.size());
    for (Entry& candidate : entries)
      prepared.pending_.push_back(std::move(candidate));
    prepared.exact_contract_ = transaction_contract_("pops.amr-interface-flux-ledger.accumulate",
                                                     prepared.pending_, published_, savepoints_);
    return prepared;
  }

  void publish_prepared_accumulation(PreparedAccumulation& prepared) noexcept {
    static_assert(std::is_nothrow_swappable_v<decltype(pending_)>);
    pending_.swap(prepared.pending_);
  }

  void resolve_pending_stage_weight(std::size_t index, Rational stage_weight) {
    if (!in_transaction())
      throw std::runtime_error(
          "AMR interface-flux stage-weight resolution requires an active transaction");
    if (transaction_depth() != 1)
      throw std::runtime_error(
          "AMR interface-flux stage weights resolve only in the outer attempt transaction");
    if (index >= pending_.size())
      throw std::out_of_range("AMR interface-flux pending fragment index is out of range");
    Entry& entry = pending_[index];
    if (entry.measure.stage_weight_resolved)
      throw std::logic_error("AMR interface-flux stage weight was already resolved");
    entry.measure.stage_weight = stage_weight;
    entry.measure.stage_weight_resolved = true;
  }

  template <class Axpy>
  std::map<InterfaceFluxAccumulationKey, Payload> aggregate(Axpy&& axpy) const {
    std::map<InterfaceFluxAccumulationKey, Payload> result;
    for (const Entry& entry : published_) {
      axpy(result[interface_flux_accumulation_key(entry.key)],
           interface_flux_fragment_scale(entry.key, entry.measure), entry.payload);
    }
    return result;
  }

 private:
  static void append_clock_contract_(ExactContractBuilder& exact, const ClockStamp& clock) {
    exact.scalar(clock.level)
        .scalar(clock.macro_step)
        .scalar(clock.phase.numerator)
        .scalar(clock.phase.denominator)
        .scalar(clock.physical_time);
  }

  static void append_entry_contract_(ExactContractBuilder& exact, const Entry& entry) {
    exact.text(entry.key.interface_identity)
        .scalar(entry.key.topology_epoch)
        .scalar(entry.key.coarse_level)
        .scalar(entry.key.fine_level);
    append_clock_contract_(exact, entry.key.clock);
    exact.text(entry.key.stage_identity)
        .text(entry.key.graph_identity)
        .text(entry.key.rate_identity)
        .text(entry.key.application_identity);
    append_clock_contract_(exact, entry.key.interval.begin);
    append_clock_contract_(exact, entry.key.interval.end);
    exact.scalar(entry.key.orientation)
        .scalar(static_cast<std::uint64_t>(entry.key.left_block))
        .scalar(static_cast<std::uint64_t>(entry.key.right_block))
        .scalar(entry.measure.stage_weight.numerator)
        .scalar(entry.measure.stage_weight.denominator)
        .scalar(entry.measure.face_measure)
        .scalar(entry.measure.substep_duration)
        .scalar(entry.measure.stage_weight_resolved);
    if constexpr (requires(ExactContractBuilder& candidate) { candidate.scalar(entry.payload); }) {
      exact.scalar(entry.payload);
    } else if constexpr (requires { entry.payload.size(); }) {
      exact.scalar(static_cast<std::uint64_t>(entry.payload.size()));
      for (const auto& component : entry.payload) {
        if constexpr (requires(ExactContractBuilder& candidate) { candidate.scalar(component); })
          exact.scalar(component);
      }
    }
  }

  std::string transaction_contract_(std::string_view phase, const std::vector<Entry>& pending,
                                    const std::vector<Entry>& accepted,
                                    const std::vector<std::size_t>& savepoints) const {
    ExactContractBuilder exact;
    exact.text(phase)
        .scalar(std::uint32_t{1})
        .scalar(topology_epoch_)
        .text("latest-accepted-root-window")
        .bytes(budget_.exact_contract)
        .scalar(static_cast<std::uint64_t>(budget_.max_fragments_per_window))
        .scalar(static_cast<std::uint64_t>(budget_.max_payload_terms_per_window))
        .scalar(static_cast<std::uint64_t>(budget_.max_transaction_depth))
        .scalar(static_cast<std::uint64_t>(savepoints.size()))
        .scalar(static_cast<std::uint64_t>(pending.size()))
        .scalar(static_cast<std::uint64_t>(accepted.size()));
    for (const std::size_t savepoint : savepoints)
      exact.scalar(static_cast<std::uint64_t>(savepoint));
    for (const Entry& entry : pending)
      append_entry_contract_(exact, entry);
    for (const Entry& entry : accepted)
      append_entry_contract_(exact, entry);
    return std::move(exact).release();
  }

  static bool same_identity_(const InterfaceFluxFragmentKey& left,
                             const InterfaceFluxFragmentKey& right) {
    return !(left < right) && !(right < left);
  }

  void validate_(const InterfaceFluxFragmentKey& key,
                 const InterfaceFluxFragmentMeasure& measure) const {
    validate_interface_flux_fragment(key, measure, topology_epoch_);
  }

  static std::size_t payload_terms_(const Payload& payload) {
    if constexpr (requires { payload.size(); })
      return static_cast<std::size_t>(payload.size());
    return 1;
  }

  static std::size_t payload_terms_(const std::vector<Entry>& entries) {
    std::size_t result = 0;
    for (const Entry& entry : entries) {
      const std::size_t count = payload_terms_(entry.payload);
      if (count > std::numeric_limits<std::size_t>::max() - result)
        throw std::length_error("AMR interface-flux payload-term count exceeds size_t");
      result += count;
    }
    return result;
  }

  static void validate_budget_(const InterfaceFluxLedgerBudget& budget) {
    if (budget.exact_contract.empty() || budget.max_transaction_depth == 0)
      throw std::invalid_argument("AMR interface-flux ledger requires an exact budget contract");
    if ((budget.max_fragments_per_window == 0) != (budget.max_payload_terms_per_window == 0))
      throw std::invalid_argument(
          "AMR interface-flux ledger budget must be either inactive or fully bounded");
  }

  void validate_budget_() const { validate_budget_(budget_); }

  static void require_within_budget_(const std::vector<Entry>& entries,
                                     const InterfaceFluxLedgerBudget& budget,
                                     std::string_view phase) {
    if (entries.size() > budget.max_fragments_per_window ||
        payload_terms_(entries) > budget.max_payload_terms_per_window)
      throw std::length_error("AMR interface-flux " + std::string(phase) +
                              " image exceeds its authenticated window budget");
  }

  std::uint64_t topology_epoch_;
  InterfaceFluxLedgerBudget budget_;
  std::vector<Entry> pending_;
  std::vector<Entry> published_;
  std::vector<std::size_t> savepoints_;
};

}  // namespace pops::amr
