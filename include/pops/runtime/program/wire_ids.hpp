#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace pops::runtime::program {

/// Stable Krylov method ids emitted by compiled Programs. Values are append-only; reserved values
/// are tombstones and must never dispatch.
enum LinearSolveMethod : int {
  kLinearSolveCg = 0,
  kLinearSolveBicgstab = 1,
  kLinearSolveGmres = 2,
  kLinearSolveRichardson = 3,
  kLinearSolveReserved4 = 4,
};

inline void validate_linear_solve_method(int method, const char* where) {
  switch (method) {
    case kLinearSolveCg:
    case kLinearSolveBicgstab:
    case kLinearSolveGmres:
    case kLinearSolveRichardson:
      return;
    default:
      throw std::runtime_error(std::string(where) + ": unknown LinearSolveMethod wire id " +
                               std::to_string(method));
  }
}

inline void validate_prepared_field_slot(std::string_view field_slot_identity,
                                         const char* where) {
  if (!field_slot_identity.empty())
    return;
  throw std::runtime_error(std::string(where) +
                           ": prepared field-slot identity must be non-empty");
}

}  // namespace pops::runtime::program
