#pragma once

#include <pops/core/foundation/types.hpp>

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
  if (!(options.tolerance > Real(0)) || options.max_iterations < 1 ||
      !(options.linear_tolerance > Real(0)) || options.linear_max_iterations < 1 ||
      options.restart < 1 || options.restart > 50 || !(options.armijo > Real(0)) ||
      !(options.armijo < Real(1)) || !(options.minimum_step > Real(0)) ||
      !(options.minimum_step < Real(1)))
    throw std::invalid_argument("invalid FieldNewtonOptions");
}

}  // namespace pops
