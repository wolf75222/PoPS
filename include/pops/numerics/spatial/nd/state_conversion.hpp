/// @file
/// @brief Model-independent state recovery and conversion results.

#pragma once

#include <pops/core/foundation/types.hpp>

namespace pops::nd {

enum class StateConversionStatus : unsigned char {
  Success = 0,
  NonFiniteState = 1,
  NonPositiveDensity = 2,
  NonPositivePressure = 3,
  InvalidEquationOfState = 4,
};

template <class State>
struct StateConversion {
  State value{};
  StateConversionStatus status = StateConversionStatus::NonFiniteState;

  POPS_HD constexpr bool succeeded() const { return status == StateConversionStatus::Success; }
};

}  // namespace pops::nd
