#pragma once

#include <pops/numerics/elliptic/mg/geometric_mg.hpp>

namespace pops::validation {

/// Test-only access to the retired embedded-boundary hardening experiment.
///
/// Production solves must use a prepared field-solver route whose SolveOutcome is consumed exactly
/// once. The historical hardening algorithm remains reachable only by focused numerical
/// validation while its behavior is compared with the supported prepared route.
struct GeometricMGValidationAccess {
  static int solve_robust(GeometricMG& solver, Real rel_tol, int max_cycles,
                          Real abs_tol = Real(0)) {
    return solver.solve_robust(rel_tol, max_cycles, abs_tol);
  }
};

}  // namespace pops::validation
