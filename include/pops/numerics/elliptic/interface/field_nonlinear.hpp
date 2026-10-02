#pragma once

#include <pops/core/foundation/types.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pops {

enum class FieldNewtonConvergenceKind { kLegacy, kRelative, kAbsolute };
struct FieldNewtonConvergence {
  FieldNewtonConvergenceKind kind = FieldNewtonConvergenceKind::kLegacy;
  Real relative = Real(0);
  Real absolute = Real(0);
};

inline void validate_field_newton_convergence(const FieldNewtonConvergence& policy) {
  if (!std::isfinite(policy.relative) || !std::isfinite(policy.absolute) ||
      policy.relative < Real(0) || policy.absolute < Real(0))
    throw std::invalid_argument("invalid Newton convergence coefficients");
  switch (policy.kind) {
    case FieldNewtonConvergenceKind::kLegacy:
      if (policy.relative != Real(0) || policy.absolute != Real(0))
        throw std::invalid_argument("legacy Newton convergence has no coefficients");
      break;
    case FieldNewtonConvergenceKind::kRelative:
      if (!(policy.relative > Real(0)))
        throw std::invalid_argument("relative Newton convergence must be positive");
      break;
    case FieldNewtonConvergenceKind::kAbsolute:
      if (policy.relative != Real(0) || !(policy.absolute > Real(0)))
        throw std::invalid_argument("absolute Newton convergence must be positive");
      break;
    default:
      throw std::invalid_argument("unknown Newton convergence policy");
  }
}

struct FieldNewtonOptions {
  Real tolerance = Real(1.0e-8);
  int max_iterations = 20;
  Real linear_tolerance = Real(1.0e-3);
  int linear_max_iterations = 80;
  int restart = 30;
  Real armijo = Real(1.0e-4);
  Real minimum_step = Real(1.0 / 1024.0);
  FieldNewtonConvergence convergence{};
};

inline void validate_field_newton_options(const FieldNewtonOptions& options) {
  validate_field_newton_convergence(options.convergence);
  const auto finite = [](Real value) { return std::isfinite(static_cast<double>(value)); };
  if (!finite(options.tolerance) || !(options.tolerance > Real(0)) || options.max_iterations < 1 ||
      !finite(options.linear_tolerance) || !(options.linear_tolerance > Real(0)) ||
      options.linear_max_iterations < 1 || options.restart < 1 ||
      options.restart > options.linear_max_iterations || !finite(options.armijo) ||
      !(options.armijo > Real(0)) || !(options.armijo < Real(1)) || !finite(options.minimum_step) ||
      !(options.minimum_step > Real(0)) || !(options.minimum_step < Real(1)))
    throw std::invalid_argument("invalid FieldNewtonOptions");
}

// Reference is the complete Original residual norm, never a preconditioned residual.
inline Real field_newton_stop_tolerance(const FieldNewtonOptions& options, Real reference) {
  validate_field_newton_options(options);
  if (!std::isfinite(reference) || reference < Real(0))
    throw std::invalid_argument("invalid Original residual reference norm");
  switch (options.convergence.kind) {
    case FieldNewtonConvergenceKind::kLegacy:
      return options.tolerance * std::max(Real(1), reference);
    case FieldNewtonConvergenceKind::kRelative: {
      const Real cutoff = std::max(options.convergence.absolute, options.convergence.relative * reference);
      if (!std::isfinite(cutoff))
        throw std::invalid_argument("typed Newton computed cutoff is not finite representable Real");
      return cutoff;
    }
    case FieldNewtonConvergenceKind::kAbsolute:
      return options.convergence.absolute;
  }
  throw std::invalid_argument("unknown Newton convergence policy");
}

}  // namespace pops
