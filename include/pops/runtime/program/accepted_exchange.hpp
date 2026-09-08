#pragma once

#include <pops/runtime/multiblock/evaluation_point.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <set>
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
        point.stage < 0 || point.stage_fraction < amr::Rational(0, 1) ||
        amr::Rational(1, 1) < point.stage_fraction || !std::isfinite(point.dt) || point.dt <= 0 || !std::isfinite(point.physical_time))
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

 private:
  using Key = std::tuple<std::string, std::string, std::string, std::string>;
  std::vector<ExchangeRecord> records_;
  std::set<Key, std::less<>> keys_;
};

}  // namespace pops::runtime::program
