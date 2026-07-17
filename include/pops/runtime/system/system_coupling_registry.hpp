#pragma once

#include <pops/core/foundation/types.hpp>                   // Real
#include <pops/coupling/source/coupled_source_program.hpp>  // CsProgram (per-cell frequency bytecode)
#include <pops/coupling/source/coupling_operator.hpp>  // CouplingOperatorView (inspect metadata)

#include <functional>
#include <string>
#include <vector>

/// @file
/// @brief The inter-species COUPLING registry of a System (ADC-578).
///
/// Extracted from the inline coupling members of `System::Impl`: the splitting-source operators, the
/// GLOBAL host dt bounds, the constant / per-cell coupled-source frequency bounds, and the typed
/// coupling-operator inspect views. Grouping them names one subsystem: "the couplings and the step
/// bounds they impose".
///
/// STEPPER VISIBILITY: `operators`, `dt_bounds`, `coupled_freqs` and `coupled_freq_exprs` ARE read by
/// SystemStepper (apply_couplings / step_cfl / step bounds). Impl re-exposes them under their exact
/// historical names via REFERENCE ALIASES (couplings / dt_bounds_ / coupled_freqs_ /
/// coupled_freq_exprs_) so system_stepper.hpp and the MockImpl stay byte-unchanged. `coupled_operators`
/// is METADATA ONLY (never read by the stepper) -> accessed registry-direct.
///
/// OWNERSHIP CONTRACT: every field is FROZEN AT BIND (populated only by the structural setters
/// add_coupled_source / add_coupling_operator / add_dt_bound, refused once bound) and READ during run
/// by the stepper. Nothing here is checkpointed (re-declared by replaying the composition).

namespace pops {
namespace runtime {
namespace system {

/// GLOBAL time-step bound (System::add_dt_bound): evaluated ONCE per step (host) by step_cfl /
/// step_adaptive. Hook for non-cell-local constraints (multi-block coupling, Schur/Poisson,
/// scheduler). Empty (default) -> historical step policy, bit-identical.
struct GlobalDtBound {
  std::string label;
  std::function<double()> fn;
};

/// DECLARED constant frequency of a coupled source (CoupledSource.frequency). The couplings apply
/// ONCE per MACRO-step, so the bound is on the macro-dt: dt <= cfl / mu, WITHOUT a substeps/stride
/// factor. Empty (default) -> no bound.
struct CoupledFreq {
  std::string label;
  double mu;
};

/// PER-CELL frequency of a coupled source (CoupledSource.frequency with an Expr): a bytecode program
/// mu(U) evaluated per cell at EVERY step (MAX reduction, global all_reduce_max), bound
/// dt <= cfl / max(mu). The inputs REUSE the resolve() resolution of the input registers (sidx,
/// comp); the constants match the source. Stored only AFTER full validation.
struct CoupledFreqExpr {
  std::string label;
  CsProgram prog;
  struct In {
    int sidx, comp;
  };
  std::vector<In> ins;  // (species, component) of the inputs (same as the source; resolved once)
  int n_in = 0;
  std::vector<Real> kconsts;  // constants loaded into r[n_in ..] (same as the source)
};

/// Data-only registry of the couplings and the step bounds they impose.
struct SystemCouplingRegistry {
  /// inter-species coupled sources applied by splitting (AFTER transport). Read by the stepper.
  std::vector<std::function<void(Real)>> operators;
  /// GLOBAL host dt bounds (add_dt_bound). Read by the stepper.
  std::vector<GlobalDtBound> dt_bounds;
  /// constant coupled-source frequency bounds. Read by the stepper.
  std::vector<CoupledFreq> coupled_freqs;
  /// per-cell coupled-source frequency bounds. Read by the stepper.
  std::vector<CoupledFreqExpr> coupled_freq_exprs;
  /// TYPED coupling-operator inspect views (label + declared conservation / frequency contracts), in
  /// registration order. METADATA ONLY: never read by the stepper.
  std::vector<CouplingOperatorView> coupled_operators;
};

}  // namespace system
}  // namespace runtime
}  // namespace pops
