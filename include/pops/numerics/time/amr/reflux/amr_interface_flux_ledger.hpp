#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

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
  ClockWindow interval;
  InterfaceFluxOrientation orientation = InterfaceFluxOrientation::CoarseOutward;

  friend bool operator<(const InterfaceFluxFragmentKey& a, const InterfaceFluxFragmentKey& b) {
    return std::make_tuple(a.interface_identity, a.topology_epoch, a.coarse_level, a.fine_level,
                           detail::interface_flux_clock_coordinate(a.clock), a.stage_identity,
                           detail::interface_flux_window_coordinate(a.interval), a.orientation) <
           std::make_tuple(b.interface_identity, b.topology_epoch, b.coarse_level, b.fine_level,
                           detail::interface_flux_clock_coordinate(b.clock), b.stage_identity,
                           detail::interface_flux_window_coordinate(b.interval), b.orientation);
  }
};

/// Fragment metadata that is numerical rather than identifying.  Face measure
/// is retained for auditing; the downstream reflux route applies geometry
/// exactly once, as for FluxMeasure.
struct InterfaceFluxFragmentMeasure {
  Rational stage_weight{1, 1};
  double face_measure = 0.0;
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
  ClockWindow interval;
  InterfaceFluxOrientation orientation = InterfaceFluxOrientation::CoarseOutward;

  friend bool operator<(const InterfaceFluxAccumulationKey& a,
                        const InterfaceFluxAccumulationKey& b) {
    return std::make_tuple(a.interface_identity, a.topology_epoch, a.coarse_level, a.fine_level,
                           detail::interface_flux_window_coordinate(a.interval), a.orientation) <
           std::make_tuple(b.interface_identity, b.topology_epoch, b.coarse_level, b.fine_level,
                           detail::interface_flux_window_coordinate(b.interval), b.orientation);
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
  return {key.interface_identity, key.topology_epoch, key.coarse_level,
          key.fine_level,         key.interval,       key.orientation};
}

/// Convert one physical-flux sample to an oriented, time-integrated
/// contribution.  Geometry stays auditable but is not multiplied here because
/// the reflux route owns the unique face/cell measure conversion.
inline double interface_flux_fragment_scale(const InterfaceFluxFragmentKey& key,
                                            const InterfaceFluxFragmentMeasure& measure) {
  return interface_flux_orientation_sign(key.orientation) * measure.stage_weight.value() *
         (key.interval.end.physical_time - key.interval.begin.physical_time);
}

/// Transactional store for refined multi-block interface fluxes.  Pending
/// fragments are invisible to aggregate() until the outer transaction commits;
/// rollback therefore cannot publish a rejected stage.  The ledger is bound to
/// one topology epoch and rejects stale fragments before storing them.
template <class Payload>
class TransactionalInterfaceFluxLedger {
 public:
  using Entry = InterfaceFluxFragment<Payload>;

  explicit TransactionalInterfaceFluxLedger(std::uint64_t topology_epoch)
      : topology_epoch_(topology_epoch) {}

  std::uint64_t topology_epoch() const { return topology_epoch_; }
  bool in_transaction() const { return !savepoints_.empty(); }
  std::size_t transaction_depth() const { return savepoints_.size(); }
  std::size_t pending_size() const { return pending_.size(); }
  std::size_t published_size() const { return published_.size(); }
  bool empty() const { return pending_.empty() && published_.empty(); }
  const std::vector<Entry>& published_entries() const { return published_; }

  void begin() { savepoints_.push_back(pending_.size()); }

  void commit() {
    if (!in_transaction())
      throw std::runtime_error("AMR interface-flux ledger commit without active transaction");
    savepoints_.pop_back();
    if (in_transaction())
      return;
    for (Entry& entry : pending_)
      published_.push_back(std::move(entry));
    pending_.clear();
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
    if (!in_transaction())
      throw std::runtime_error("AMR interface-flux accumulation requires an active transaction");
    validate_(key, measure);
    pending_.push_back({std::move(key), measure, std::move(payload)});
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
  void validate_(const InterfaceFluxFragmentKey& key,
                 const InterfaceFluxFragmentMeasure& measure) const {
    if (key.topology_epoch != topology_epoch_)
      throw std::invalid_argument("AMR interface-flux fragment uses a stale topology epoch");
    if (key.interface_identity.empty() || key.stage_identity.empty() || key.coarse_level < 0 ||
        key.fine_level != key.coarse_level + 1)
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
      throw std::invalid_argument(
          "AMR interface-flux clock is outside its exact temporal interval");
    const double begin_time = key.interval.begin.physical_time;
    const double end_time = key.interval.end.physical_time;
    if (!std::isfinite(key.clock.physical_time) || !std::isfinite(begin_time) ||
        !std::isfinite(end_time) || !(end_time > begin_time) ||
        key.clock.physical_time < begin_time || key.clock.physical_time > end_time ||
        !(measure.face_measure > 0.0) || !std::isfinite(measure.face_measure))
      throw std::invalid_argument(
          "AMR interface-flux fragment requires finite positive geometry/time");
  }

  std::uint64_t topology_epoch_;
  std::vector<Entry> pending_;
  std::vector<Entry> published_;
  std::vector<std::size_t> savepoints_;
};

}  // namespace pops::amr
