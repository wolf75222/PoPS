#pragma once

/// @file
/// @brief Canonical schedule for explicit admissibility checks and projection.

#include <pops/core/identity/prepared_provider.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace pops {

enum class EnforcementPhase : std::uint8_t {
  kInitialization = 0,
  kReconstruction = 1,
  kSourceSolve = 2,
  kBoundary = 3,
  kAcceptance = 4,
};

struct EnforcementRule {
  bool check = false;
  bool project_if_invalid = false;
};

/// Fixed-order schedule shared by Uniform, AMR, native and compiled consumers.
class EnforcementSchedule final {
 public:
  static constexpr std::size_t phase_count = 5;

  explicit EnforcementSchedule(std::array<EnforcementRule, phase_count> rules) : rules_(rules) {
    for (const EnforcementRule rule : rules_)
      if (rule.project_if_invalid && !rule.check)
        throw std::invalid_argument("projection schedule requires an admissibility check");
  }

  [[nodiscard]] constexpr EnforcementRule at(EnforcementPhase phase) const noexcept {
    return rules_[static_cast<std::size_t>(phase)];
  }

  void serialize_exact(ExactContractBuilder& contract) const {
    contract.text("pops.enforcement-schedule").scalar(std::uint32_t{1});
    for (std::size_t phase = 0; phase < phase_count; ++phase)
      contract.scalar(static_cast<std::uint8_t>(phase))
          .scalar(rules_[phase].check)
          .scalar(rules_[phase].project_if_invalid);
  }

  [[nodiscard]] std::string exact_contract() const {
    ExactContractBuilder contract;
    serialize_exact(contract);
    return std::move(contract).release();
  }

 private:
  std::array<EnforcementRule, phase_count> rules_{};
};

}  // namespace pops
