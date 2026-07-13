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

  void begin() { savepoints_.push_back(entries_); }

  void commit() {
    if (savepoints_.empty())
      throw std::runtime_error("AMR flux ledger commit without active transaction");
    savepoints_.pop_back();
  }

  void rollback() {
    if (savepoints_.empty())
      throw std::runtime_error("AMR flux ledger rollback without active transaction");
    entries_ = std::move(savepoints_.back());
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
      const double scale = entry.measure.stage_weight.value() * entry.measure.face_measure *
                           entry.measure.substep_duration;
      axpy(result[entry.key], scale, entry.payload);
    }
    return result;
  }

 private:
  std::vector<Entry> entries_;
  std::vector<std::vector<Entry>> savepoints_;
};

}  // namespace pops::amr
