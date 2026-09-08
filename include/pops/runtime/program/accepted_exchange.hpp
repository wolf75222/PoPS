#pragma once

#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <exception>
#include <optional>

#include <bit>
#include <algorithm>
#include <string_view>
#include <cmath>
#include <cstdint>
#include <set>
#include <span>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace pops::runtime::program {

/// One resolved mathematical occurrence selected by the accepted temporal quadrature.
/// Numerical evaluations and nonlinear iterates do not stage records by themselves. The generated
/// affine update supplies its signed weight once, retaining repeated mathematical use explicitly.
struct ExchangeRecord {
  std::string operation_identity;
  std::string occurrence_identity;
  std::string evaluation_context;
  std::string quadrature_identity;
  int orientation = 1;
  double face_measure = 1.0;
  double numerical_flux = 0.0;
  double temporal_weight = 0.0;
  int multiplicity = 1;

  double integrated_amount() const {
    return static_cast<double>(orientation) * face_measure * numerical_flux * temporal_weight *
           static_cast<double>(multiplicity);
  }

  void validate() const {
    if (operation_identity.empty() || occurrence_identity.empty() || evaluation_context.empty() ||
        quadrature_identity.empty())
      throw std::invalid_argument(
          "exchange record requires exact operation/occurrence/context/quadrature identities");
    if ((orientation != -1 && orientation != 1) || multiplicity <= 0 ||
        !std::isfinite(face_measure) || face_measure <= 0.0 || !std::isfinite(numerical_flux) ||
        !std::isfinite(temporal_weight) || !std::isfinite(integrated_amount()))
      throw std::invalid_argument(
          "exchange record has invalid orientation, measure, weight or multiplicity");
  }

  /// Static IR identities repeat on each invocation. Qualify them with the exact runtime
  /// interval so several accepted substeps can share an enclosing provisional ledger.
  void qualify_runtime_point(const runtime::multiblock::BoundaryEvaluationPoint& point) {
    if (point.clock.empty() || point.tick < 0 || point.level < 0 || point.substep < 0 ||
        point.stage < 0 || point.stage_fraction < ::pops::amr::Rational(0, 1) ||
        ::pops::amr::Rational(1, 1) < point.stage_fraction || !std::isfinite(point.dt) ||
        point.dt <= 0 || !std::isfinite(point.physical_time))
      throw std::invalid_argument("accepted exchange requires a complete runtime evaluation point");
    evaluation_context = "pops.exchange.frame.v1/" + std::to_string(point.clock.size()) + ":" +
                         point.clock + "/" + std::to_string(point.tick) + "/" +
                         std::to_string(point.level) + "/" + std::to_string(point.substep) + "/" +
                         std::to_string(point.stage) + "/" +
                         std::to_string(point.stage_fraction.numerator) + "/" +
                         std::to_string(point.stage_fraction.denominator) + "/" +
                         std::to_string(std::bit_cast<std::uint64_t>(point.dt)) + "/" +
                         std::to_string(std::bit_cast<std::uint64_t>(point.physical_time)) + "/" +
                         std::to_string(evaluation_context.size()) + ":" + evaluation_context;
  }

  auto key() const {
    return std::tie(operation_identity, occurrence_identity, evaluation_context,
                    quadrature_identity);
  }
};

/// Attempt-owned mailbox inside ProgramRuntimeState, hence inside every accepted snapshot.
/// It is not a second commit authority. A nested acceptance leaves these records provisional in
/// the enclosing native transaction; rollback restores the same mailbox as states and histories.
class AcceptedExchangeLedger {
 public:
  void stage(ExchangeRecord record) {
    record.validate();
    const auto [entry, inserted] =
        keys_.emplace(record.operation_identity, record.occurrence_identity,
                      record.evaluation_context, record.quadrature_identity);
    if (!inserted)
      throw std::invalid_argument("duplicate accepted exchange occurrence/quadrature contribution");
    try {
      records_.push_back(std::move(record));
    } catch (...) {
      keys_.erase(entry);
      throw;
    }
  }

  const std::vector<ExchangeRecord>& records() const noexcept { return records_; }

  /// Restore a staging checkpoint when a peer cannot append its own contribution. No allocation
  /// occurs here, so collective failure recovery does not copy the accumulated face records.
  void restore_size(std::size_t size) noexcept {
    while (records_.size() > size) {
      const auto entry = keys_.find(records_.back().key());
      if (entry != keys_.end())
        keys_.erase(entry);
      records_.pop_back();
    }
  }
  void clear() noexcept {
    records_.clear();
    keys_.clear();
  }
  void swap(AcceptedExchangeLedger& other) noexcept {
    records_.swap(other.records_);
    keys_.swap(other.keys_);
  }

  /// Canonical accepted mailbox image. It is independent of diagnostic projections and retains
  /// binary64 weights/fluxes exactly; the enclosing restart transaction remains the sole publisher.
  std::vector<std::uint8_t> checkpoint() const {
    std::vector<std::uint8_t> bytes{'P', 'O', 'P', 'S', 'E', 'X', '0', '1'};
    const auto word = [&](std::uint64_t value) {
      for (unsigned byte = 0; byte < 8; ++byte)
        bytes.push_back(static_cast<std::uint8_t>(value >> (8 * byte)));
    };
    const auto text = [&](const std::string& value) {
      word(value.size());
      bytes.insert(bytes.end(), value.begin(), value.end());
    };
    word(records_.size());
    for (const auto& record : records_) {
      record.validate();
      text(record.operation_identity);
      text(record.occurrence_identity);
      text(record.evaluation_context);
      text(record.quadrature_identity);
      word(std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(record.orientation)));
      word(std::bit_cast<std::uint64_t>(record.face_measure));
      word(std::bit_cast<std::uint64_t>(record.numerical_flux));
      word(std::bit_cast<std::uint64_t>(record.temporal_weight));
      word(static_cast<std::uint64_t>(record.multiplicity));
    }
    return bytes;
  }

  static AcceptedExchangeLedger from_checkpoint(std::span<const std::uint8_t> bytes) {
    constexpr std::string_view magic = "POPSEX01";
    if (bytes.size() < 16 || !std::equal(magic.begin(), magic.end(), bytes.begin()))
      throw std::invalid_argument("accepted exchange checkpoint has an invalid header");
    std::size_t cursor = magic.size();
    const auto word = [&]() {
      if (bytes.size() - cursor < 8)
        throw std::invalid_argument("accepted exchange checkpoint is truncated");
      std::uint64_t value = 0;
      for (unsigned byte = 0; byte < 8; ++byte)
        value |= static_cast<std::uint64_t>(bytes[cursor++]) << (8 * byte);
      return value;
    };
    const auto text = [&]() {
      const auto size = word();
      if (size > bytes.size() - cursor)
        throw std::invalid_argument("accepted exchange checkpoint text exceeds its byte image");
      std::string value(reinterpret_cast<const char*>(bytes.data() + cursor), size);
      cursor += static_cast<std::size_t>(size);
      return value;
    };
    const auto count = word();
    if (count > (bytes.size() - cursor) / 72)
      throw std::invalid_argument("accepted exchange checkpoint count exceeds its byte image");
    AcceptedExchangeLedger candidate;
    for (std::uint64_t index = 0; index < count; ++index) {
      ExchangeRecord record;
      record.operation_identity = text();
      record.occurrence_identity = text();
      record.evaluation_context = text();
      record.quadrature_identity = text();
      const auto orientation = std::bit_cast<std::int64_t>(word());
      if (orientation != -1 && orientation != 1)
        throw std::invalid_argument("accepted exchange checkpoint has an invalid orientation");
      record.orientation = static_cast<int>(orientation);
      record.face_measure = std::bit_cast<double>(word());
      record.numerical_flux = std::bit_cast<double>(word());
      record.temporal_weight = std::bit_cast<double>(word());
      const auto multiplicity = word();
      if (multiplicity == 0 ||
          multiplicity > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("accepted exchange checkpoint has an invalid multiplicity");
      record.multiplicity = static_cast<int>(multiplicity);
      candidate.stage(std::move(record));
    }
    if (cursor != bytes.size())
      throw std::invalid_argument("accepted exchange checkpoint has trailing bytes");
    return candidate;
  }

 private:
  using Key = std::tuple<std::string, std::string, std::string, std::string>;
  std::vector<ExchangeRecord> records_;
  std::set<Key, std::less<>> keys_;
};

/// The producer is rank-local and must not enter collectives. Its record count may be zero
/// or differ from peers. Qualify every local record before one provider-boundary fence.
template <class Producer, class Qualify>
std::vector<ExchangeRecord> prepare_exchange_batch(Producer&& producer, Qualify&& qualify,
                                                   const ExecutionLane& lane) {
  std::vector<ExchangeRecord> records;
  std::exception_ptr error;
  try {
    std::forward<Producer>(producer)([&](ExchangeRecord record) {
      qualify(record);
      records.push_back(std::move(record));
    });
  } catch (...) {
    error = std::current_exception();
  }
  if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && error)
      std::rethrow_exception(error);
    throw std::runtime_error("Program exchange batch preparation failed collectively");
  }
  return records;
}

/// The existing ledger remains the sole mailbox and the enclosing native transaction remains
/// the sole acceptance authority. A detached batch candidate changes no transaction depth.
inline void stage_exchange_batch_collectively(AcceptedExchangeLedger& ledger,
                                              std::span<ExchangeRecord> records,
                                              const ExecutionLane& lane) {
  std::optional<AcceptedExchangeLedger> candidate;
  std::exception_ptr error;
  try {
    candidate.emplace(ledger);
    for (auto& record : records)
      candidate->stage(std::move(record));
  } catch (...) {
    error = std::current_exception();
  }
  if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && error)
      std::rethrow_exception(error);
    throw std::runtime_error("Program exchange batch staging failed collectively");
  }
  ledger.swap(*candidate);
}

}  // namespace pops::runtime::program
