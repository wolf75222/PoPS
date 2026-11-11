#pragma once

/// @file
/// @brief The original-residual gate of a mesh solve (spec 0.3 SV8), decided outside its
///        realization.
///
/// SV8: the gates of the problem decide whether a solve succeeded, in the order of
/// `relations.solve` (python/pops/relations.py): admissibility predicates, the selection condition,
/// the boundary relations, the re-check of the original residual ||R(u)|| <= atol + rtol * scale in
/// the request's norm, then the request test. A realization's own convergence flag is evidence and
/// never the authority. A Program mesh solve passes this gate after its realization returns a
/// solved candidate and before that candidate is accepted: the generated code evaluates R(u) through
/// the problem's own residual, measures it and the request scale in the request's norm, and refuses
/// the candidate here when the bound fails. The refusal is `SolveFailed` (T39), reported with the
/// status `safeguard_failure` and the action that the outcome's consumer selects for that status.

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <cmath>
#include <exception>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace pops::runtime::program {

/// One evaluation of the original-residual gate. `norm` is ||R(u)|| and `scale` the request scale,
/// both in the request's norm; `atol` and `rtol` are the request's tolerances.
struct OriginalResidualGate {
  Real norm = Real(0);
  Real atol = Real(0);
  Real rtol = Real(0);
  Real scale = Real(0);

  /// atol + rtol * scale, the bound of SV8.
  [[nodiscard]] Real bound() const noexcept { return atol + rtol * scale; }

  /// ||R(u)|| <= atol + rtol * scale, for a finite norm and bound. The quotient form
  /// (||R(u)|| - atol) / scale <= rtol states the same relation and is also accepted: the rounded
  /// product can refuse by one ulp a residual that the realization's own stop accepts, since that
  /// stop reads rtol as a quotient (`satisfies_stopping_controls`).
  [[nodiscard]] bool satisfied() const noexcept {
    if (norm <= bound())
      return true;
    return std::isfinite(static_cast<double>(scale)) && scale > Real(0) &&
           (norm - atol) / scale <= rtol;
  }

  /// The reason the gate refuses, in the order of `relations.solve`, or empty when it passes: a
  /// norm that is not finite or negative, an accuracy whose bound is not finite, then a residual
  /// above its bound.
  [[nodiscard]] std::string refusal() const {
    const Real limit = bound();
    const char* cause = nullptr;
    if (!std::isfinite(static_cast<double>(norm)) || norm < Real(0))
      cause = "invalid_norm";
    else if (!std::isfinite(static_cast<double>(limit)) || limit < Real(0))
      cause = "invalid_accuracy";
    else if (!satisfied())
      cause = "original_residual";
    if (cause == nullptr)
      return {};
    std::ostringstream text;
    text.precision(17);
    text << "SV8 gate " << cause << ": ||R(u)||=" << norm << " bound=" << limit << " (atol=" << atol
         << " rtol=" << rtol << " scale=" << scale << ")";
    return text.str();
  }
};

/// Refuse a solved report whose gate fails. A report that is already a failure stays as it is: the
/// gates run on a solved candidate only. Returns true when the report is (still) solved.
inline bool apply_original_residual_gate(SolveReport& report, const OriginalResidualGate& gate,
                                         SolveAction refusal_action, std::string_view label) {
  if (!report.solved_value_available())
    return false;
  std::string refusal = gate.refusal();
  if (refusal.empty())
    return true;
  report.mark_failed(
      SolveStatus::kSafeguardFailure, refusal_action,
      std::string(label) + " " + std::move(refusal) + "; realization reported " + report.reason);
  return false;
}

/// The same gate on an unconsumed outcome (its publication hooks roll the candidate back when the
/// refused outcome is consumed).
inline bool apply_original_residual_gate(SolveOutcome& outcome, const OriginalResidualGate& gate,
                                         SolveAction refusal_action, std::string_view label) {
  if (!outcome.report().solved_value_available())
    return false;
  std::string refusal = gate.refusal();
  if (refusal.empty())
    return true;
  outcome.refuse_candidate(SolveStatus::kSafeguardFailure, refusal_action,
                           std::string(label) + " " + std::move(refusal) +
                               "; realization reported " + outcome.report().reason);
  return false;
}

/// The gate on an unconsumed outcome whose measurement may itself fail. A gate that cannot be
/// evaluated cannot certify the candidate (SV8), so a measurement that throws refuses it as an
/// invalid evaluation; the outcome is then consumed like any failed solve and never escapes
/// unconsumed.
template <class Measure>
bool apply_measured_original_residual_gate(SolveOutcome& outcome, Measure&& measure,
                                           SolveAction refusal_action, std::string_view label) {
  if (!outcome.report().solved_value_available())
    return false;
  OriginalResidualGate gate;
  try {
    gate = std::forward<Measure>(measure)();
  } catch (const std::exception& error) {
    outcome.refuse_candidate(SolveStatus::kInvalidEvaluation, refusal_action,
                             std::string(label) + " SV8 gate measurement failed: " + error.what() +
                                 "; realization reported " + outcome.report().reason);
    return false;
  }
  return apply_original_residual_gate(outcome, gate, refusal_action, label);
}

/// ||R(u)|| and the request scale ||R(0)|| of a prepared affine problem, in the problem's own
/// metric (the request norm of its solves). R is the problem's scientific residual b - A(u), which
/// includes its boundary relations; `residual` receives R and must not alias `rhs`, `iterate` or
/// `zero`, and `zero` holds the zero iterate with the operator's footprint.
template <class Problem, class Field>
[[nodiscard]] OriginalResidualGate prepared_affine_original_residual(
    const Problem& problem, Field& residual, const Field& rhs, const Field& iterate,
    const Field& zero, Real atol, Real rtol) {
  OriginalResidualGate gate;
  gate.atol = atol;
  gate.rtol = rtol;
  problem.true_residual(residual, rhs, zero);
  gate.scale = problem.residual_norm(residual);
  problem.true_residual(residual, rhs, iterate);
  gate.norm = problem.residual_norm(residual);
  return gate;
}

/// The problem's own residual and request norm, reached through the operator session of the
/// workspace the solve bound (a bound problem lends its public session to no one else). The norm is
/// the request's physical norm; a provider application that fails yields a NaN norm, which the gate
/// refuses as invalid_norm.
template <int Dim>
struct PreparedKrylovResidual {
  const PreparedAffineLinearProblem<Dim>& problem;
  KrylovWorkspace<Dim>& workspace;
  KrylovPhysicalNorm physical_norm;

  void true_residual(MultiFab<Dim>& out, const MultiFab<Dim>& rhs,
                     const MultiFab<Dim>& iterate) const {
    ::pops::detail::workspace_true_residual(problem, workspace, out, rhs, iterate);
  }
  [[nodiscard]] Real residual_norm(const MultiFab<Dim>& value) const {
    if (!::pops::detail::KrylovWorkspaceAccess::collective_provider_apply_result(workspace)
             .succeeded())
      return std::numeric_limits<Real>::quiet_NaN();
    return ::pops::detail::workspace_physical_residual_norm(problem, workspace, value,
                                                            physical_norm);
  }
};

/// The gate of a prepared Krylov solve: R is measured on the candidate the outcome would publish
/// (the workspace-private iterate, not the destination it is copied to on Accept), with R(0) as the
/// request scale, in the request's physical norm.
template <int Dim>
[[nodiscard]] OriginalResidualGate prepared_krylov_original_residual(
    const PreparedAffineLinearProblem<Dim>& problem, KrylovWorkspace<Dim>& workspace,
    KrylovPhysicalNorm physical_norm, MultiFab<Dim>& residual, const MultiFab<Dim>& rhs,
    const MultiFab<Dim>& zero, Real atol, Real rtol) {
  return prepared_affine_original_residual(
      PreparedKrylovResidual<Dim>{problem, workspace, physical_norm}, residual, rhs,
      ::pops::detail::KrylovWorkspaceAccess::publication_candidate(workspace), zero, atol, rtol);
}

/// The l2 norm over every component and rank, sqrt(sum of squares), as the spatial residual
/// request measures it ("spatial_residual_l2"). Every rank receives the same bits.
template <class Field>
[[nodiscard]] Real summed_l2_norm(const Field& field, const ExecutionLane& lane) {
  const Real squared = static_cast<Real>(all_reduce_sum(dot_all_local(field, field), lane));
  if (!std::isfinite(static_cast<double>(squared)))
    return squared;
  if (squared < Real(0))
    return std::numeric_limits<Real>::quiet_NaN();
  return squared > Real(0) ? std::sqrt(squared) : Real(0);
}

}  // namespace pops::runtime::program
