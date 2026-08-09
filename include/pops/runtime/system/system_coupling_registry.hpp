#pragma once

#include <pops/core/foundation/types.hpp>              // Real
#include <pops/coupling/source/coupling_operator.hpp>  // CouplingOperatorView (inspect metadata)
#include <pops/mesh/storage/multifab.hpp>

#include <cstddef>
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
/// `System<Dim>::step_cfl`; `operators` are consumed only by explicit Program lowering.
/// `coupled_operators` is metadata only and is inspected directly from the registry.
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

/// Prepared exact-ranked reduction of a state-dependent coupling frequency. The provider captures
/// its authenticated fields and returns the collective maximum; the generic registry never stores
/// one provider's bytecode or array view.
struct PreparedCoupledFrequency {
  std::string label;
  std::function<Real()> maximum_frequency;
};

/// One executable coupling receives the complete, System-indexed state pack selected by the
/// Program.  Keeping the state pack explicit lets a Program apply an operator-split source to its
/// uncommitted endpoint candidates and publish the whole group only after coupling and projection
/// succeed; no operator has to borrow or mutate the accepted live states.
template <int Dim>
using PreparedCouplingOperator = std::function<void(Real, const std::vector<MultiFab<Dim>*>&)>;

/// Prepared registry of the couplings and the step bounds they impose.
template <int Dim>
struct SystemCouplingRegistry {
  static_assert(Dim >= 1 && Dim <= 3,
                "SystemCouplingRegistry only supports dimensions 1, 2, and 3");

  /// Inter-species coupled sources applied by an explicit Program node after transport.  Each
  /// operator consumes the exact simultaneous candidate-state pack supplied by that Program.
  std::vector<PreparedCouplingOperator<Dim>> operators;
  /// GLOBAL host dt bounds (add_dt_bound). Read by the stepper.
  std::vector<GlobalDtBound> dt_bounds;
  /// constant coupled-source frequency bounds. Read by the stepper.
  std::vector<CoupledFreq> coupled_freqs;
  /// per-cell coupled-source frequency bounds. Read by the stepper.
  std::vector<PreparedCoupledFrequency> coupled_frequencies;
  /// TYPED coupling-operator inspect views (label + declared conservation / frequency contracts), in
  /// registration order. METADATA ONLY: never read by the stepper.
  std::vector<CouplingOperatorView> coupled_operators;

  std::size_t apply(Real dt, const std::vector<MultiFab<Dim>*>& states) const {
    for (const auto& op : operators)
      op(dt, states);
    return operators.size();
  }
};

}  // namespace system
}  // namespace runtime
}  // namespace pops
