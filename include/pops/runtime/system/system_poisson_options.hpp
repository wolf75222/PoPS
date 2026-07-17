#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/runtime/numerical_defaults.hpp>

/// @file
/// @brief GeometricMgOptions: the resolved V-cycle knobs the System Poisson forwards to the native
///        GeometricMG solver (ADC-613).
///
/// Before ADC-613 the typed pops.solvers.elliptic.GeometricMG descriptor recorded rel_tol /
/// max_cycles / smoother / coarse but nothing downstream read them: the field solver always built
/// GeometricMG with default ctor args and called the no-argument solve(), i.e. rel_tol=1e-8,
/// max_cycles=50, min_coarse=2, nu1=nu2=2, nbottom=50. This POD carries the resolved scalars from
/// set_poisson to the solver construction + the solve() call so the descriptor is honoured.
///
/// Every member DEFAULTS to the matching numerical_defaults.hpp kMG* constant, so a System that
/// never touches the tolerance/cycle knobs reproduces the historical trajectory bit-for-bit.

namespace pops {

/// The V-cycle knobs of the shared system Poisson (ADC-613). Defaults are the ADC-603 kMG*
/// constants (numerical_defaults.hpp): a GeometricMgOptions built with no overrides drives the
/// exact historical V-cycle (rel_tol 1e-8, max_cycles 50, min_coarse 2, nu 2/2, nbottom 50).
struct GeometricMgOptions {
  Real rel_tol = kMGDefaultRelTol;       ///< relative residual stop (max(rel_tol*r0, abs_tol)).
  Real abs_tol = kMGDefaultAbsTol;       ///< absolute residual floor (0 = purely relative).
  int max_cycles = kMGDefaultMaxCycles;  ///< V-cycle cap.
  int min_coarse = kMGDefaultMinCoarse;  ///< stop coarsening below this per-axis cell count.
  int nu1 = kMGDefaultPreSmooth;         ///< pre-smoothing Gauss-Seidel sweeps.
  int nu2 = kMGDefaultPostSmooth;        ///< post-smoothing Gauss-Seidel sweeps.
  int nbottom = kMGDefaultBottomSweeps;  ///< coarsest-grid (bottom) Gauss-Seidel sweeps.
  int coarse_threshold =
      kMGDefaultCoarseThreshold;  ///< ADC-644: total-cell coarsening ceiling (0 = off).
};

}  // namespace pops
