/// @file
/// @brief Exact time/stage identity shared by dimension-generic runtime operators.

#pragma once

#include <pops/numerics/time/amr/levels/amr_clock.hpp>

#include <cstdint>
#include <limits>
#include <string>

namespace pops::runtime::multiblock {

/// One immutable residual-evaluation point. Operators receive this value from the Program
/// scheduler; they never reconstruct a stage clock from rounded physical time.
struct BoundaryEvaluationPoint {
  std::string clock;
  std::int64_t tick = 0;
  int level = 0;
  int substep = 0;
  int stage = 0;
  ::pops::amr::Rational stage_fraction{0, 1};
  double dt = std::numeric_limits<double>::quiet_NaN();
  double physical_time = std::numeric_limits<double>::quiet_NaN();

  friend bool operator==(const BoundaryEvaluationPoint&, const BoundaryEvaluationPoint&) = default;
};

}  // namespace pops::runtime::multiblock
