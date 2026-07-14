#pragma once

#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pops/numerics/time/amr/levels/amr_clock.hpp>

namespace pops::amr {

enum class FluxOrientation { XMinus, XPlus, YMinus, YPlus };

/// The conservative identity is semantic, never a storage address.  Distinct
/// rates or fluxes may share a state buffer without sharing a ledger entry.
struct FluxLedgerKey {
  std::string owner;
  std::string state;
  std::string rate;
  std::string flux;
  int level = 0;
  ClockStamp clock;

  friend bool operator<(const FluxLedgerKey& a, const FluxLedgerKey& b) {
    return std::tie(a.owner, a.state, a.rate, a.flux, a.level, a.clock.macro_step,
                    a.clock.phase.numerator, a.clock.phase.denominator) <
           std::tie(b.owner, b.state, b.rate, b.flux, b.level, b.clock.macro_step,
                    b.clock.phase.numerator, b.clock.phase.denominator);
  }
};

/// Auditable measure attached to every accumulation.  The payload is kept in
/// physical flux units; weight, face measure and duration are explicit inputs.
struct FluxMeasure {
  Rational stage_weight{1, 1};
  FluxOrientation orientation = FluxOrientation::XMinus;
  double face_measure = 0.0;
  double substep_duration = 0.0;
};

/// Numerical scale used when the accepted ledger feeds the finite-volume reflux correction.  The
/// payload is a physical face flux, while route_reflux_integrated applies the face/cell measure ratio
/// through 1/dx or 1/dy.  Multiplying face_measure here would therefore count geometry twice.  The
/// measure remains in the ledger as auditable geometry, but weight*duration is the unique numerical
/// conversion from an accepted entry to its dt-integrated interface flux.
inline double numerical_reflux_scale(const FluxMeasure& measure) {
  return measure.stage_weight.value() * measure.substep_duration;
}

template <class Payload>
struct FluxLedgerEntry {
  FluxLedgerKey key;
  FluxMeasure measure;
  Payload payload;
};

/// Nested transaction-local conservative ledger.  A rejected inner or outer
/// attempt restores the exact preceding entries; no contribution can leak to
/// a retry.  Aggregation is deliberately delegated to the numerical payload's
/// exact axpy so ordering and RK/ARK weights remain scheme-defined.
template <class Payload>
class TransactionalFluxLedger {
 public:
  using Entry = FluxLedgerEntry<Payload>;

  void begin() { savepoints_.push_back(entries_.size()); }

  void commit() {
    if (savepoints_.empty())
      throw std::runtime_error("AMR flux ledger commit without active transaction");
    savepoints_.pop_back();
  }

  void rollback() {
    if (savepoints_.empty())
      throw std::runtime_error("AMR flux ledger rollback without active transaction");
    entries_.resize(savepoints_.back());
    savepoints_.pop_back();
  }

  bool in_transaction() const { return !savepoints_.empty(); }
  std::size_t transaction_depth() const { return savepoints_.size(); }
  std::size_t size() const { return entries_.size(); }
  bool empty() const { return entries_.empty(); }
  const std::vector<Entry>& entries() const { return entries_; }

  void clear() {
    if (!savepoints_.empty())
      throw std::runtime_error("cannot clear an active AMR flux ledger transaction");
    entries_.clear();
  }

  void accumulate(FluxLedgerKey key, FluxMeasure measure, Payload payload) {
    if (!in_transaction())
      throw std::runtime_error("AMR flux accumulation requires an active transaction");
    if (key.owner.empty() || key.state.empty() || key.rate.empty() || key.flux.empty() ||
        key.level < 0 || key.clock.level != key.level)
      throw std::invalid_argument("AMR flux ledger key is not fully qualified");
    if (!(measure.face_measure > 0.0) || !std::isfinite(measure.face_measure) ||
        !(measure.substep_duration > 0.0) || !std::isfinite(measure.substep_duration))
      throw std::invalid_argument("AMR flux ledger measure must have finite positive geometry/time");
    entries_.push_back({std::move(key), measure, std::move(payload)});
  }

  template <class Axpy>
  std::map<FluxLedgerKey, Payload> aggregate(Axpy&& axpy) const {
    std::map<FluxLedgerKey, Payload> result;
    for (const Entry& entry : entries_) {
      axpy(result[entry.key], numerical_reflux_scale(entry.measure), entry.payload);
    }
    return result;
  }

 private:
  std::vector<Entry> entries_;
  std::vector<std::size_t> savepoints_;
};

}  // namespace pops::amr
