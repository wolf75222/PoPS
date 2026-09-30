#pragma once

#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <pops/runtime/program/collective_step_rejection.hpp>

#include <exception>

#include <bit>
#include <algorithm>
#include <string_view>
#include <cmath>
#include <cstdint>
#include <set>
#include <span>
#include <limits>
#include <map>
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
  // A resolved external face, distinct from an interior incidence in the same ledger.
  // Older producers leave these fields unset and cannot supply an integral-state transfer.
  int trace_axis = -1;
  int trace_side = -1;
  int trace_component = -1;
  bool exterior_trace = false;
  std::string source_evaluation_identity;

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
    if (exterior_trace && (trace_axis < 0 || (trace_side != 0 && trace_side != 1) ||
                           trace_component < 0 || source_evaluation_identity.empty()))
      throw std::invalid_argument("external exchange trace has incomplete axis/side/component");
    if (exterior_trace && evaluation_context.starts_with("pops.exchange.frame.v1/")) {
      const auto suffix = "/" + std::to_string(source_evaluation_identity.size()) + ":" +
                          source_evaluation_identity;
      if (!evaluation_context.ends_with(suffix))
        throw std::invalid_argument("external exchange trace conflicts with its runtime evaluation frame");
    }
  }

  /// Static IR identities repeat on each invocation. Qualify them with the exact runtime
  /// interval so several accepted substeps can share an enclosing provisional ledger.
  void qualify_runtime_point(const runtime::multiblock::BoundaryEvaluationPoint& point) {
    if (point.clock.empty() || point.tick < 0 || point.level < 0 || point.substep < 0 ||
        point.stage < 0 || point.stage_fraction < ::pops::amr::Rational(0, 1) ||
        ::pops::amr::Rational(1, 1) < point.stage_fraction || !std::isfinite(point.dt) ||
        point.dt <= 0 || !std::isfinite(point.physical_time))
      throw std::invalid_argument("accepted exchange requires a complete runtime evaluation point");
    source_evaluation_identity = evaluation_context;
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
  struct IntegralState {
    double initial = 0.0;
    double value = 0.0;
  };

  void declare_integral(std::string identity, double initial) {
    if (identity.empty() || !std::isfinite(initial))
      throw std::invalid_argument("integral state requires an exact identity and finite initial value");
    const auto [it, inserted] = integrals_.try_emplace(std::move(identity), IntegralState{initial, initial});
    if (!inserted && std::bit_cast<std::uint64_t>(it->second.initial) !=
                         std::bit_cast<std::uint64_t>(initial))
      throw std::invalid_argument("integral state was redeclared with a different initial value");
  }

  double integral(std::string_view identity) const {
    const auto it = integrals_.find(std::string(identity));
    if (it == integrals_.end())
      throw std::invalid_argument("integral state is absent from the installed Program");
    return it->second.value;
  }

  bool same_integral_declarations(const AcceptedExchangeLedger& other) const noexcept {
    if (integrals_.size() != other.integrals_.size()) return false;
    auto left = integrals_.begin();
    auto right = other.integrals_.begin();
    for (; left != integrals_.end(); ++left, ++right)
      if (left->first != right->first ||
          std::bit_cast<std::uint64_t>(left->second.initial) !=
              std::bit_cast<std::uint64_t>(right->second.initial)) return false;
    return true;
  }

  std::string integral_value_contract() const {
    ExactContractBuilder contract;
    contract.text("pops.accepted-integral-state.v1")
        .sequence(integrals_, [](ExactContractBuilder& row, const auto& entry) {
          row.text(entry.first).scalar(entry.second.initial).scalar(entry.second.value);
        });
    return std::move(contract).release();
  }

  struct TraceSelection {
    std::string operation;
    std::string occurrence;
    int axis = -1;
    int side = -1;
    int component = -1;
    std::string evaluation;
  };

  struct PreparedTrace {
    std::vector<std::size_t> indices;
    double local_amount = 0.0;
  };

  PreparedTrace prepare_trace(const TraceSelection& selection) const {
    if (selection.operation.empty() || selection.occurrence.empty() ||
        selection.evaluation.empty() || selection.axis < 0 ||
        (selection.side != 0 && selection.side != 1) || selection.component < 0)
      throw std::invalid_argument("integral transfer requires an exact external trace selector");
    PreparedTrace result;
    for (std::size_t i = 0; i < records_.size(); ++i) {
      const auto& record = records_[i];
      if (record.operation_identity != selection.operation ||
          record.occurrence_identity != selection.occurrence ||
          record.source_evaluation_identity != selection.evaluation || !record.exterior_trace ||
          record.trace_axis != selection.axis || record.trace_side != selection.side ||
          record.trace_component != selection.component || consumed_.contains(record.key()))
        continue;
      result.local_amount += record.integrated_amount();
      if (!std::isfinite(result.local_amount))
        throw std::overflow_error("integral trace amount is non-finite");
      result.indices.push_back(i);
    }
    return result;
  }

  double apply_trace(std::string_view identity, const PreparedTrace& trace, double global_amount,
                     double scale) {
    const auto it = integrals_.find(std::string(identity));
    if (it == integrals_.end() || !std::isfinite(global_amount) || !std::isfinite(scale) ||
        scale == 0.0)
      throw std::invalid_argument("integral transfer has absent state or non-finite amount/scale");
    const double next = it->second.value + scale * global_amount;
    if (!std::isfinite(next))
      throw std::overflow_error("integral transfer candidate is non-finite");
    for (const auto index : trace.indices)
      if (!consumed_.insert(records_.at(index).key()).second)
        throw std::invalid_argument("accepted external trace was consumed twice");
    it->second.value = next;
    return next;
  }

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
    consumed_.clear();
  }
  void reset_artifact() noexcept {
    clear();
    integrals_.clear();
  }
  void swap(AcceptedExchangeLedger& other) noexcept {
    records_.swap(other.records_);
    keys_.swap(other.keys_);
    consumed_.swap(other.consumed_);
    integrals_.swap(other.integrals_);
  }

  /// Canonical accepted mailbox image. It is independent of diagnostic projections and retains
  /// binary64 weights/fluxes exactly; the enclosing restart transaction remains the sole publisher.
  std::vector<std::uint8_t> checkpoint(bool force_extended=false) const {
    const bool extended = force_extended || !integrals_.empty();
    std::vector<std::uint8_t> bytes{'P', 'O', 'P', 'S', 'E', 'X', '0',
                                    static_cast<std::uint8_t>(extended ? '2' : '1')};
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
      if (extended) {
        word(static_cast<std::uint64_t>(static_cast<std::int64_t>(record.trace_axis) + 1));
        word(static_cast<std::uint64_t>(static_cast<std::int64_t>(record.trace_side) + 1));
        word(static_cast<std::uint64_t>(static_cast<std::int64_t>(record.trace_component) + 1));
        word(record.exterior_trace ? 1 : 0);
        text(record.source_evaluation_identity);
      }
    }
    if (extended) {
      word(integrals_.size());
      for (const auto& [identity, state] : integrals_) {
        text(identity);
        word(std::bit_cast<std::uint64_t>(state.initial));
        word(std::bit_cast<std::uint64_t>(state.value));
      }
      word(consumed_.size());
      for (const auto& [operation, occurrence, context, quadrature] : consumed_) {
        text(operation);
        text(occurrence);
        text(context);
        text(quadrature);
      }
    }
    return bytes;
  }

  static AcceptedExchangeLedger from_checkpoint(std::span<const std::uint8_t> bytes) {
    constexpr std::string_view prefix = "POPSEX0";
    if (bytes.size() < 16 || !std::equal(prefix.begin(), prefix.end(), bytes.begin()) ||
        (bytes[7] != '1' && bytes[7] != '2'))
      throw std::invalid_argument("accepted exchange checkpoint has an invalid header");
    const bool extended = bytes[7] == '2';
    std::size_t cursor = 8;
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
      if (extended) {
        const auto axis = word();
        const auto side = word();
        const auto component = word();
        const auto exterior = word();
        if (axis > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
            side > 2 || component > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
            exterior > 1)
          throw std::invalid_argument("accepted exchange checkpoint has invalid trace support");
        record.trace_axis = static_cast<int>(axis) - 1;
        record.trace_side = static_cast<int>(side) - 1;
        record.trace_component = static_cast<int>(component) - 1;
        record.exterior_trace = exterior != 0;
        record.source_evaluation_identity = text();
      }
      candidate.stage(std::move(record));
    }
    if (extended) {
      const auto integral_count = word();
      if (integral_count == 0 || integral_count > (bytes.size() - cursor) / 24)
        throw std::invalid_argument("accepted exchange checkpoint has invalid integral count");
      for (std::uint64_t index = 0; index < integral_count; ++index) {
        auto identity = text();
        const double initial = std::bit_cast<double>(word());
        const double value = std::bit_cast<double>(word());
        if (identity.empty() || !std::isfinite(initial) || !std::isfinite(value) ||
            !candidate.integrals_.emplace(std::move(identity), IntegralState{initial, value}).second)
          throw std::invalid_argument("accepted exchange checkpoint has invalid integral state");
      }
      const auto consumed_count = word();
      if (consumed_count > count || consumed_count > (bytes.size() - cursor) / 36)
        throw std::invalid_argument("accepted exchange checkpoint has invalid consumed trace count");
      for (std::uint64_t index = 0; index < consumed_count; ++index) {
        Key key{text(), text(), text(), text()};
        const auto record = std::find_if(candidate.records_.begin(), candidate.records_.end(),
                                         [&](const ExchangeRecord& item) { return item.key() == key; });
        if (record == candidate.records_.end() || !record->exterior_trace ||
            !candidate.consumed_.insert(std::move(key)).second)
          throw std::invalid_argument("accepted exchange checkpoint has an unknown consumed trace");
      }
    }
    if (cursor != bytes.size())
      throw std::invalid_argument("accepted exchange checkpoint has trailing bytes");
    return candidate;
  }

 private:
  using Key = std::tuple<std::string, std::string, std::string, std::string>;
  std::vector<ExchangeRecord> records_;
  std::set<Key, std::less<>> keys_;
  std::set<Key, std::less<>> consumed_;
  std::map<std::string, IntegralState, std::less<>> integrals_;
};

inline void require_integral_values_agree_collectively(const AcceptedExchangeLedger& ledger,
                                                        const ExecutionLane& lane) {
  std::string payload;
  std::exception_ptr error;
  try {
    payload = ledger.integral_value_contract();
  } catch (...) {
    error = std::current_exception();
  }
  if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && error) std::rethrow_exception(error);
    throw std::runtime_error("integral state contract preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"accepted-integral-state", payload}}, lane))
    throw std::runtime_error("integral state values differ across MPI ranks");
}

inline void declare_integral_collectively(AcceptedExchangeLedger& ledger,
                                          std::string_view identity, double initial,
                                          const ExecutionLane& lane) {
  AcceptedExchangeLedger candidate;
  std::exception_ptr error;
  std::string contract_payload;
  try {
    candidate = ledger;
    candidate.declare_integral(std::string(identity), initial);
    ExactContractBuilder contract;
    contract.text("pops.integral-declaration.v1").text(identity).scalar(initial);
    contract_payload = std::move(contract).release();
  } catch (...) {
    error = std::current_exception();
  }
  if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && error) std::rethrow_exception(error);
    throw std::runtime_error("integral state declaration failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"integral-declaration", contract_payload}}, lane))
    throw std::runtime_error("integral state declaration differs across MPI ranks");
  require_integral_values_agree_collectively(candidate, lane);
  ledger.swap(candidate);
}

/// Prepare every local contribution, vote before summing, then publish the same finite scalar on
/// every rank. A copy absorbs allocations; the final swap is noexcept and cannot half-publish.
inline double consume_external_trace_collectively(
    AcceptedExchangeLedger& ledger, std::string_view integral_identity,
    const AcceptedExchangeLedger::TraceSelection& selection, double scale,
    const ExecutionLane& lane) {
  AcceptedExchangeLedger candidate;
  AcceptedExchangeLedger::PreparedTrace trace;
  std::exception_ptr error;
  std::string contract_payload;
  try {
    candidate = ledger;
    trace = candidate.prepare_trace(selection);
    ExactContractBuilder contract;
    contract.text("pops.integral-trace-consumption.v1")
        .text(integral_identity).text(selection.operation).text(selection.occurrence)
        .text(selection.evaluation)
        .scalar(static_cast<std::int32_t>(selection.axis))
        .scalar(static_cast<std::int32_t>(selection.side))
        .scalar(static_cast<std::int32_t>(selection.component)).scalar(scale);
    contract_payload = std::move(contract).release();
  } catch (...) {
    error = std::current_exception();
  }
  if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && error) std::rethrow_exception(error);
    throw std::runtime_error("accepted external trace preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"integral-trace-consumption", contract_payload}}, lane))
    throw std::runtime_error("accepted external trace selector differs across MPI ranks");
  require_integral_values_agree_collectively(candidate, lane);
  const auto count = all_reduce_sum(static_cast<long>(trace.indices.size()), lane);
  if (count == 0)
    throw std::invalid_argument("accepted external trace has no unconsumed face contribution");
  const double amount = all_reduce_sum(trace.local_amount, lane);
  error = nullptr;
  double value = 0.0;
  try {
    value = candidate.apply_trace(integral_identity, trace, amount, scale);
  } catch (...) {
    error = std::current_exception();
  }
  if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && error) std::rethrow_exception(error);
    throw std::runtime_error("accepted external trace publication failed collectively");
  }
  ledger.swap(candidate);
  return value;
}

/// The producer is rank-local and must not enter collectives. Its record count may be zero
/// or differ from peers. Qualify every local record before one provider-boundary fence.
template <class Producer, class Qualify>
std::vector<ExchangeRecord> prepare_exchange_batch(Producer&& producer, Qualify&& qualify,
                                                   const ExecutionLane& lane) {
  std::vector<ExchangeRecord> records;
  collective_step_rejection_phase(
      lane.communicator(),
      {"pops.exchange-batch.prepare.v1", "pops.exchange-batch.prepare", false, false},
      "Program exchange batch preparation failed collectively", [&] {
        std::forward<Producer>(producer)([&](ExchangeRecord record) {
          qualify(record);
          records.push_back(std::move(record));
        });
      });
  return records;
}

/// The existing ledger remains the sole mailbox and the enclosing native transaction remains
/// the sole acceptance authority. The producer is complete before append starts; failure restores
/// the exact pre-batch prefix without copying prior records or consuming transaction depth.
inline void stage_exchange_batch_collectively(AcceptedExchangeLedger& ledger,
                                              std::span<ExchangeRecord> records,
                                              const ExecutionLane& lane) {
  const auto prior_size = ledger.records().size();
  std::exception_ptr error;
  try {
    for (auto& record : records)
      ledger.stage(std::move(record));
  } catch (...) {
    error = std::current_exception();
  }
  try {
    collectively_rethrow_exception(error, lane,
                                   "Program exchange batch staging failed collectively");
  } catch (...) {
    ledger.restore_size(prior_size);
    throw;
  }
}

}  // namespace pops::runtime::program
