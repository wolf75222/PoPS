#pragma once

#include <pops/core/foundation/types.hpp>

#include <cmath>
#include <stdexcept>

namespace pops {

struct FieldNewtonOptions {
  Real tolerance = Real(1.0e-8);
  int max_iterations = 20;
  Real linear_tolerance = Real(1.0e-3);
  int linear_max_iterations = 80;
  int restart = 30;
  Real armijo = Real(1.0e-4);
  Real minimum_step = Real(1.0 / 1024.0);
};

inline void validate_field_newton_options(const FieldNewtonOptions& options) {
  const auto finite = [](Real value) { return std::isfinite(static_cast<double>(value)); };
  if (!finite(options.tolerance) || !(options.tolerance > Real(0)) ||
      options.max_iterations < 1 || !finite(options.linear_tolerance) ||
      !(options.linear_tolerance > Real(0)) || options.linear_max_iterations < 1 ||
      options.restart < 1 || !finite(options.armijo) || !(options.armijo > Real(0)) ||
      !(options.armijo < Real(1)) || !finite(options.minimum_step) ||
      !(options.minimum_step > Real(0)) || !(options.minimum_step < Real(1)))
    throw std::invalid_argument("invalid FieldNewtonOptions");
}

}  // namespace pops
