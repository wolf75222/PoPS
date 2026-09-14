#pragma once
#include <pops/core/model/native_call.hpp>

// External arithmetic only. The equation, speed and discretization live in Python.
namespace migration_arithmetic {
POPS_HD inline pops::NativeCallResult<1> multiply(pops::Real x, pops::Real y) {
  pops::NativeCallResult<1> result;
  result.status = pops::EvaluationStatus::kOk;
  result.values[0] = x * y;
  return result;
}
POPS_HD inline pops::NativeCallResult<2> jacobian(pops::Real x, pops::Real y) {
  pops::NativeCallResult<2> result;
  result.status = pops::EvaluationStatus::kOk;
  result.values[0] = y;
  result.values[1] = x;
  return result;
}
POPS_HD inline pops::NativeCallResult<2> approximate(pops::Real x, pops::Real y) {
  auto result = jacobian(x, y);
  result.values[0] *= pops::Real(0.99);
  result.values[1] *= pops::Real(0.99);
  return result;
}
}  // namespace migration_arithmetic
