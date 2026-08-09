/// @file
/// @brief Solver-independent preparation and diagnostics options for local nonlinear solves.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/diagnostics/runtime_diagnostics.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/runtime/numerical_defaults.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

namespace pops {

/// Public preparation policy. It contains no solver implementation or spatial-storage authority.
struct NewtonOptions {
  int max_iters = kNewtonDefaultMaxIters;
  Real rel_tol = kNewtonDefaultRelTol;
  Real abs_tol = kNewtonDefaultAbsTol;
  Real fd_eps = kNewtonDefaultFdEps;
  Real damping = kNewtonDefaultDamping;
};

inline void validate_newton_options(const NewtonOptions& options, const char* where) {
  const std::string prefix = std::string(where) + " : ";
  if (options.max_iters < 1)
    throw std::runtime_error(prefix + "newton_max_iters >= 1");
  if (!std::isfinite(options.rel_tol) || !std::isfinite(options.abs_tol) ||
      !std::isfinite(options.fd_eps) || options.rel_tol < Real(0) || options.abs_tol < Real(0) ||
      (options.rel_tol == Real(0) && options.abs_tol == Real(0)) || options.fd_eps <= Real(0))
    throw std::runtime_error(prefix +
                             "newton_rel_tol/abs_tol >= 0 with at least one positive tolerance, "
                             "and newton_fd_eps > 0");
  if (!std::isfinite(options.damping) || !(options.damping > Real(0) && options.damping <= Real(1)))
    throw std::runtime_error(prefix + "newton_damping in (0, 1]");
}

/// Compatibility inspection aggregate. The common SolveReport remains authoritative.
struct NewtonReport {
  bool enabled = false;
  bool converged = true;
  Real max_residual = Real(0);
  Real max_iters_used = Real(0);
  double n_failed = 0;
  SolveFailureLocation failure{};
  SolveReport solve{};
  RuntimeDiagnosticsReport diagnostics =
      make_runtime_diagnostics_report("pops.numerics.time.prepared_local_nonlinear");

  void reset() { *this = NewtonReport{}; }
};

}  // namespace pops
