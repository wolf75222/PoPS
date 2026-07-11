#pragma once

#include <stdexcept>
#include <string>

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

/// Stable assembly field-role ids emitted by compiled Programs.
enum AssemblyFieldRole : int {
  kEpsX = 0,
  kEpsY = 1,
  kAxy = 2,
  kAyx = 3,
  kRhs = 4,
  kFlux = 5,
  kPhi = 6,
  kAssemblyFieldReserved7 = 7,
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

inline void validate_assembly_write_role(int role, const char* where) {
  if (role >= kEpsX && role <= kFlux)
    return;
  throw std::runtime_error(std::string(where) +
                           ": unknown or non-write AssemblyFieldRole wire id " +
                           std::to_string(role));
}

inline void validate_assembly_read_role(int role, const char* where) {
  if (role == kPhi)
    return;
  throw std::runtime_error(std::string(where) +
                           ": unknown or non-read AssemblyFieldRole wire id " +
                           std::to_string(role));
}

}  // namespace pops::runtime::program
