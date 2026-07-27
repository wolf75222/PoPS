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
/// STEPPER VISIBILITY: `dt_bounds`, `coupled_freqs` and `coupled_freq_exprs` are read by
/// SystemProgramDriver for `step_cfl`; `operators` are consumed only by explicit Program lowering. Impl
/// re-exposes the bound collections under their exact
/// historical names via REFERENCE ALIASES (couplings / dt_bounds_ / coupled_freqs_ /
/// coupled_freq_exprs_). `coupled_operators` is METADATA ONLY -> accessed registry-direct.
///
/// OWNERSHIP CONTRACT: every field is FROZEN AT BIND (populated only by the structural setters
/// add_coupled_source / add_coupling_operator / add_dt_bound, refused once bound) and READ during run
/// by the stepper. Nothing here is checkpointed (re-declared by replaying the composition).

namespace pops {
namespace runtime {
namespace system {

/// GLOBAL time-step bound (System::add_dt_bound): evaluated ONCE per `step_cfl` (host). Hook for
/// non-cell-local constraints (multi-block coupling, Schur/Poisson, scheduler). Empty means no
/// additional Program macro-step constraint.
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
