#pragma once

#include <cmath>
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
    for (const auto& existing : records_)
      if (existing.key() == record.key())
        throw std::invalid_argument(
            "duplicate accepted exchange occurrence/quadrature contribution");
    records_.push_back(std::move(record));
  }

  const std::vector<ExchangeRecord>& records() const noexcept { return records_; }
  void clear() noexcept { records_.clear(); }
  void swap(AcceptedExchangeLedger& other) noexcept { records_.swap(other.records_); }

 private:
  std::vector<ExchangeRecord> records_;
};

}  // namespace pops::runtime::program
