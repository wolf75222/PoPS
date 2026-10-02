#pragma once

#include <pops/numerics/elliptic/interface/field_nonlinear.hpp>
#include <pops/parallel/collective_exception.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <array>
#include <cstdint>
#include <cstring>

namespace pops {

// All kinds enter the same fixed-size exact byte consensus before any admission.
// No struct padding, hashes, heap allocation, residual evaluation or norm reduction.
inline Real collective_field_newton_stop_tolerance(const FieldNewtonOptions& options,
                                                   Real reference, const ExecutionLane& lane) {
  std::array<char, sizeof(std::int32_t) + 3 * sizeof(Real)> minimum{};
  const auto kind = static_cast<std::int32_t>(options.convergence.kind);
  std::memcpy(minimum.data(), &kind, sizeof(kind));
  std::array<Real, 3> coefficients{};
  switch (options.convergence.kind) {
    case FieldNewtonConvergenceKind::kLegacy:
      coefficients[0] = options.tolerance;
      break;
    case FieldNewtonConvergenceKind::kRelative:
      coefficients[1] = options.convergence.relative;
      coefficients[2] = options.convergence.absolute;
      break;
    case FieldNewtonConvergenceKind::kAbsolute:
      coefficients[2] = options.convergence.absolute;
      break;
    default:
      break;  // Validation below is voted on every rank, including invalid kinds.
  }
  for (std::size_t i = 0; i < coefficients.size(); ++i)
    std::memcpy(minimum.data() + sizeof(kind) + i * sizeof(Real), &coefficients[i], sizeof(Real));
  auto maximum = minimum;
  all_reduce_min_inplace(minimum.data(), minimum.size(), lane);
  all_reduce_max_inplace(maximum.data(), maximum.size(), lane);
  if (minimum != maximum)
    throw std::invalid_argument("Original Newton convergence differs between execution lane ranks");
  Real cutoff = Real(0);
  std::exception_ptr error;
  try {
    cutoff = field_newton_stop_tolerance(options, reference);
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, "Original Newton computed cutoff");
  return cutoff;
}

}  // namespace pops
