#pragma once

/// @file
/// @brief Allocation-free Krylov algorithms over a prepared affine problem.
///
/// This is deliberately the only public generic Krylov entry point.  Callers must prepare and
/// authenticate the operator evaluation, bind persistent workspace, and pass typed controls.  The
/// algorithms therefore never guess that an affine A is linear, lazily build a preconditioner,
/// allocate scratch in an iteration, or publish an Arnoldi/preconditioned residual as scientific
/// convergence.

#include <pops/numerics/elliptic/linear/krylov_workspace.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace pops {
namespace detail {

inline bool finite(Real value) { return std::isfinite(static_cast<double>(value)); }

inline void validate_controls(const KrylovControls& controls) {
  if (controls.max_iterations <= 0)
    throw std::invalid_argument("prepared Krylov solve requires max_iterations > 0");
  if (!finite(controls.rel_tol) || controls.rel_tol < Real(0))
    throw std::invalid_argument("prepared Krylov solve requires finite rel_tol >= 0");
  if (!finite(controls.abs_tol) || controls.abs_tol < Real(0))
    throw std::invalid_argument("prepared Krylov solve requires finite abs_tol >= 0");
  if (controls.rel_tol == Real(0) && controls.abs_tol == Real(0))
    throw std::invalid_argument("prepared Krylov solve requires a non-zero stopping tolerance");
  if (controls.method == KrylovMethod::kGmres) {
    if (controls.restart < 1)
      throw std::invalid_argument("prepared GMRES restart must be positive");
  } else if (controls.restart != 0) {
    throw std::invalid_argument("restart belongs only to prepared GMRES");
  }
  if (controls.method == KrylovMethod::kRichardson &&
      (!finite(controls.relaxation) || controls.relaxation <= Real(0)))
    throw std::invalid_argument("prepared Richardson requires finite relaxation > 0");
}

inline Real reference_denominator(Real reference) {
  return reference > Real(0) ? reference : Real(1);
}

inline Real stopping_threshold(Real reference, const KrylovControls& controls) {
  return std::max(controls.rel_tol * reference, controls.abs_tol);
}

template <class Report>
inline void set_report_residuals(Report& report, Real reference, Real residual) {
  report.rel_residual = residual / reference_denominator(reference);
  if constexpr (requires { report.reference_residual_norm; })
    report.reference_residual_norm = reference;
  if constexpr (requires { report.residual_norm; })
    report.residual_norm = residual;
}

inline SolveReport report(Real reference, Real residual, int iterations, SolveStatus status) {
  SolveReport result;
  result.iters = iterations;
  set_report_residuals(result, reference, residual);
  if (status == SolveStatus::kSolved)
    result.mark_solved();
  else
    result.mark_failed(status);
  return result;
}

inline Real true_residual_norm(const PreparedAffineLinearProblem& problem, MultiFab& scratch,
                               const MultiFab& rhs, const MultiFab& iterate) {
  problem.true_residual(scratch, rhs, iterate);
  return PreparedFieldAlgebra::norm(scratch);
}

inline SolveReport checked_report(const PreparedAffineLinearProblem& problem, MultiFab& scratch,
                                  const MultiFab& rhs, const MultiFab& iterate, Real reference,
                                  int iterations, SolveStatus status) {
  const Real residual = true_residual_norm(problem, scratch, rhs, iterate);
  return report(reference, residual, iterations,
                finite(residual) ? status : SolveStatus::kInvalidEvaluation);
}

inline SolveReport solve_richardson(const PreparedAffineLinearProblem& problem,
                                    KrylovWorkspace& workspace, MultiFab& iterate,
                                    const MultiFab& rhs, const KrylovControls& controls) {
  MultiFab& effective_rhs = workspace.field(0);
  MultiFab& residual = workspace.field(1);
  problem.effective_rhs(effective_rhs, rhs);
  const Real reference = PreparedFieldAlgebra::norm(effective_rhs);
  Real residual_norm = true_residual_norm(problem, residual, rhs, iterate);
  if (!finite(reference) || !finite(residual_norm))
    return report(reference, residual_norm, 0, SolveStatus::kInvalidEvaluation);
  const Real threshold = stopping_threshold(reference, controls);
  if (residual_norm <= threshold)
    return report(reference, residual_norm, 0, SolveStatus::kSolved);

  for (int iteration = 1; iteration <= controls.max_iterations; ++iteration) {
    PreparedFieldAlgebra::axpy(iterate, controls.relaxation, residual);
    residual_norm = true_residual_norm(problem, residual, rhs, iterate);
    if (!finite(residual_norm))
      return report(reference, residual_norm, iteration, SolveStatus::kInvalidEvaluation);
    if (residual_norm <= threshold)
      return report(reference, residual_norm, iteration, SolveStatus::kSolved);
  }
  return report(reference, residual_norm, controls.max_iterations,
                SolveStatus::kIterationLimit);
}

inline SolveReport solve_cg(const PreparedAffineLinearProblem& problem,
                            KrylovWorkspace& workspace, MultiFab& iterate, const MultiFab& rhs,
                            const KrylovControls& controls) {
  MultiFab& effective_rhs = workspace.field(0);
  MultiFab& residual = workspace.field(1);
  MultiFab& direction = workspace.field(2);
  MultiFab& applied = workspace.field(3);
  problem.effective_rhs(effective_rhs, rhs);
  const Real reference = PreparedFieldAlgebra::norm(effective_rhs);
  Real residual_norm = true_residual_norm(problem, residual, rhs, iterate);
  if (!finite(reference) || !finite(residual_norm))
    return report(reference, residual_norm, 0, SolveStatus::kInvalidEvaluation);
  const Real threshold = stopping_threshold(reference, controls);
  if (residual_norm <= threshold)
    return report(reference, residual_norm, 0, SolveStatus::kSolved);

  PreparedFieldAlgebra::copy(direction, residual);
  Real squared = PreparedFieldAlgebra::dot(residual, residual);
  for (int iteration = 1; iteration <= controls.max_iterations; ++iteration) {
    problem.apply_linear(applied, direction);
    const Real curvature = PreparedFieldAlgebra::dot(direction, applied);
    if (!finite(curvature) || !finite(squared))
      return checked_report(problem, applied, rhs, iterate, reference, iteration - 1,
                            SolveStatus::kInvalidEvaluation);
    // A certified SPD operator has strictly positive curvature.  Refuse the mathematical loss of
    // definiteness, not a dimensioned absolute epsilon that would reject a valid rescaled system.
    if (curvature <= Real(0))
      return checked_report(problem, applied, rhs, iterate, reference, iteration - 1,
                            SolveStatus::kBreakdown);

    const Real alpha = squared / curvature;
    PreparedFieldAlgebra::axpy(iterate, alpha, direction);
    PreparedFieldAlgebra::axpy(residual, -alpha, applied);
    Real next_squared = PreparedFieldAlgebra::dot(residual, residual);
    residual_norm = next_squared >= Real(0) ? std::sqrt(next_squared)
                                            : std::numeric_limits<Real>::quiet_NaN();
    if (!finite(residual_norm))
      return checked_report(problem, applied, rhs, iterate, reference, iteration,
                            SolveStatus::kInvalidEvaluation);
    if (residual_norm <= threshold) {
      const Real confirmed = true_residual_norm(problem, applied, rhs, iterate);
      if (!finite(confirmed))
        return report(reference, confirmed, iteration, SolveStatus::kInvalidEvaluation);
      if (confirmed <= threshold)
        return report(reference, confirmed, iteration, SolveStatus::kSolved);
      PreparedFieldAlgebra::copy(residual, applied);
      next_squared = confirmed * confirmed;
    }
    if (iteration == controls.max_iterations)
      return checked_report(problem, applied, rhs, iterate, reference, iteration,
                            SolveStatus::kIterationLimit);
    if (squared <= Real(0))
      return checked_report(problem, applied, rhs, iterate, reference, iteration,
                            SolveStatus::kBreakdown);
    const Real beta = next_squared / squared;
    PreparedFieldAlgebra::lincomb(direction, Real(1), residual, beta, direction);
    squared = next_squared;
  }
  return checked_report(problem, applied, rhs, iterate, reference, controls.max_iterations,
                        SolveStatus::kIterationLimit);
}

inline SolveReport solve_bicgstab(const PreparedAffineLinearProblem& problem,
                                  KrylovWorkspace& workspace, MultiFab& iterate,
                                  const MultiFab& rhs, const KrylovControls& controls) {
  MultiFab& effective_rhs = workspace.field(0);
  MultiFab& residual = workspace.field(1);
  MultiFab& shadow = workspace.field(2);
  MultiFab& direction = workspace.field(3);
  MultiFab& applied = workspace.field(4);
  MultiFab& intermediate = workspace.field(5);
  MultiFab& second_applied = workspace.field(6);
  MultiFab& prepared_direction = problem.has_preconditioner() ? workspace.field(7) : direction;
  MultiFab& prepared_intermediate = problem.has_preconditioner() ? workspace.field(8) : intermediate;

  problem.effective_rhs(effective_rhs, rhs);
  const Real reference = PreparedFieldAlgebra::norm(effective_rhs);
  Real residual_norm = true_residual_norm(problem, residual, rhs, iterate);
  if (!finite(reference) || !finite(residual_norm))
    return report(reference, residual_norm, 0, SolveStatus::kInvalidEvaluation);
  const Real threshold = stopping_threshold(reference, controls);
  if (residual_norm <= threshold)
    return report(reference, residual_norm, 0, SolveStatus::kSolved);

  PreparedFieldAlgebra::copy(shadow, residual);
  PreparedFieldAlgebra::zero(direction);
  PreparedFieldAlgebra::zero(applied);
  Real rho_previous = Real(1);
  Real alpha = Real(1);
  Real omega = Real(1);

  for (int iteration = 1; iteration <= controls.max_iterations; ++iteration) {
    const Real rho = PreparedFieldAlgebra::dot(shadow, residual);
    if (!finite(rho))
      return checked_report(problem, second_applied, rhs, iterate, reference, iteration - 1,
                            SolveStatus::kInvalidEvaluation);
    if (rho == Real(0))
      return checked_report(problem, second_applied, rhs, iterate, reference, iteration - 1,
                            SolveStatus::kBreakdown);

    if (iteration == 1) {
      PreparedFieldAlgebra::copy(direction, residual);
    } else {
      if (omega == Real(0))
        return checked_report(problem, second_applied, rhs, iterate, reference, iteration - 1,
                              SolveStatus::kBreakdown);
      const Real beta = (rho / rho_previous) * (alpha / omega);
      PreparedFieldAlgebra::axpy(direction, -omega, applied);
      PreparedFieldAlgebra::lincomb(direction, Real(1), residual, beta, direction);
    }

    if (problem.has_preconditioner())
      problem.apply_preconditioner(prepared_direction, direction);
    problem.apply_linear(applied, prepared_direction);
    const Real denominator = PreparedFieldAlgebra::dot(shadow, applied);
    if (!finite(denominator) || denominator == Real(0))
      return checked_report(problem, second_applied, rhs, iterate, reference, iteration - 1,
                            finite(denominator) ? SolveStatus::kBreakdown
                                                : SolveStatus::kInvalidEvaluation);
    alpha = rho / denominator;
    PreparedFieldAlgebra::lincomb(intermediate, Real(1), residual, -alpha, applied);

    const Real intermediate_norm = PreparedFieldAlgebra::norm(intermediate);
    if (!finite(intermediate_norm))
      return checked_report(problem, second_applied, rhs, iterate, reference, iteration - 1,
                            SolveStatus::kInvalidEvaluation);
    bool alpha_committed = false;
    if (intermediate_norm <= threshold) {
      PreparedFieldAlgebra::axpy(iterate, alpha, prepared_direction);
      alpha_committed = true;
      const Real confirmed = true_residual_norm(problem, second_applied, rhs, iterate);
      if (!finite(confirmed))
        return report(reference, confirmed, iteration, SolveStatus::kInvalidEvaluation);
      if (confirmed <= threshold)
        return report(reference, confirmed, iteration, SolveStatus::kSolved);
      PreparedFieldAlgebra::copy(intermediate, second_applied);
    }

    if (problem.has_preconditioner())
      problem.apply_preconditioner(prepared_intermediate, intermediate);
    problem.apply_linear(second_applied, prepared_intermediate);
    const Real second_norm_squared = PreparedFieldAlgebra::dot(second_applied, second_applied);
    const Real projection = PreparedFieldAlgebra::dot(second_applied, intermediate);
    if (!finite(second_norm_squared) || !finite(projection))
      return checked_report(problem, second_applied, rhs, iterate, reference, iteration - 1,
                            SolveStatus::kInvalidEvaluation);
    if (second_norm_squared <= Real(0))
      return checked_report(problem, second_applied, rhs, iterate, reference,
                            alpha_committed ? iteration : iteration - 1,
                            SolveStatus::kBreakdown);
    omega = projection / second_norm_squared;
    if (!finite(omega) || omega == Real(0))
      return checked_report(problem, second_applied, rhs, iterate, reference,
                            alpha_committed ? iteration : iteration - 1,
                            finite(omega) ? SolveStatus::kBreakdown
                                          : SolveStatus::kInvalidEvaluation);

    if (!alpha_committed)
      PreparedFieldAlgebra::axpy(iterate, alpha, prepared_direction);
    PreparedFieldAlgebra::axpy(iterate, omega, prepared_intermediate);
    residual_norm = true_residual_norm(problem, second_applied, rhs, iterate);
    if (!finite(residual_norm))
      return report(reference, residual_norm, iteration, SolveStatus::kInvalidEvaluation);
    if (residual_norm <= threshold)
      return report(reference, residual_norm, iteration, SolveStatus::kSolved);
    PreparedFieldAlgebra::copy(residual, second_applied);
    rho_previous = rho;
  }
  return report(reference, residual_norm, controls.max_iterations,
                SolveStatus::kIterationLimit);
}

inline void reset_gmres_scalars(KrylovWorkspace& workspace, int restart) {
  for (int row = 0; row <= restart; ++row) {
    workspace.rotated_rhs(row) = Real(0);
    if (row < restart) {
      workspace.cosine(row) = Real(0);
      workspace.sine(row) = Real(0);
      workspace.solution_coefficient(row) = Real(0);
    }
    for (int column = 0; column < restart; ++column)
      workspace.h(row, column) = Real(0);
  }
}

inline bool solve_gmres_upper(KrylovWorkspace& workspace, int dimension) {
  for (int row = dimension - 1; row >= 0; --row) {
    Real value = workspace.rotated_rhs(row);
    for (int column = row + 1; column < dimension; ++column)
      value -= workspace.h(row, column) * workspace.solution_coefficient(column);
    const Real diagonal = workspace.h(row, row);
    if (!finite(diagonal) || diagonal == Real(0))
      return false;
    workspace.solution_coefficient(row) = value / diagonal;
  }
  return true;
}

inline SolveReport solve_gmres(const PreparedAffineLinearProblem& problem,
                               KrylovWorkspace& workspace, MultiFab& iterate,
                               const MultiFab& rhs, const KrylovControls& controls) {
  const int restart = controls.restart;
  MultiFab& effective_rhs = workspace.field(0);
  const auto basis = [&workspace](int index) -> MultiFab& {
    return workspace.field(static_cast<std::size_t>(index + 1));
  };
  MultiFab& applied_or_residual = workspace.field(static_cast<std::size_t>(restart + 2));
  MultiFab& prepared_vector = workspace.field(static_cast<std::size_t>(restart + 3));

  problem.effective_rhs(effective_rhs, rhs);
  const Real reference = PreparedFieldAlgebra::norm(effective_rhs);
  Real residual_norm = true_residual_norm(problem, applied_or_residual, rhs, iterate);
  if (!finite(reference) || !finite(residual_norm))
    return report(reference, residual_norm, 0, SolveStatus::kInvalidEvaluation);
  const Real threshold = stopping_threshold(reference, controls);
  if (residual_norm <= threshold)
    return report(reference, residual_norm, 0, SolveStatus::kSolved);

  int iterations = 0;
  while (iterations < controls.max_iterations) {
    problem.apply_preconditioner(prepared_vector, applied_or_residual);
    const Real beta = PreparedFieldAlgebra::norm(prepared_vector);
    if (!finite(beta))
      return report(reference, residual_norm, iterations, SolveStatus::kInvalidEvaluation);
    if (beta == Real(0))
      return report(reference, residual_norm, iterations, SolveStatus::kBreakdown);
    // The Arnoldi estimate lives in the left-preconditioned norm whereas `threshold` and
    // `residual_norm` live in the physical norm.  Map the remaining physical reduction into the
    // current preconditioned cycle instead of comparing quantities with unrelated scales.  This
    // keeps a scalar rescaling of an otherwise identical preconditioner from changing restart
    // behaviour, costs no extra preconditioner application or collective, and remains only a
    // request for the authoritative true-residual confirmation below.
    const Real estimate_threshold = beta * (threshold / residual_norm);
    PreparedFieldAlgebra::lincomb(basis(0), Real(1) / beta, prepared_vector, Real(0), prepared_vector);
    reset_gmres_scalars(workspace, restart);
    workspace.rotated_rhs(0) = beta;

    int dimension = 0;
    bool estimate_reached = false;
    bool invalid = false;
    for (int column = 0; column < restart && iterations < controls.max_iterations; ++column) {
      problem.apply_linear(applied_or_residual, basis(column));
      problem.apply_preconditioner(prepared_vector, applied_or_residual);
      for (int row = 0; row <= column; ++row) {
        workspace.h(row, column) = PreparedFieldAlgebra::dot(prepared_vector, basis(row));
        PreparedFieldAlgebra::axpy(prepared_vector, -workspace.h(row, column), basis(row));
      }
      workspace.h(column + 1, column) = PreparedFieldAlgebra::norm(prepared_vector);
      if (!finite(workspace.h(column + 1, column))) {
        invalid = true;
        break;
      }
      const bool lucky_breakdown = workspace.h(column + 1, column) == Real(0);
      if (!lucky_breakdown) {
        PreparedFieldAlgebra::lincomb(basis(column + 1),
                                  Real(1) / workspace.h(column + 1, column), prepared_vector,
                                  Real(0), prepared_vector);
      }

      for (int row = 0; row < column; ++row) {
        const Real first = workspace.h(row, column);
        const Real second = workspace.h(row + 1, column);
        workspace.h(row, column) =
            workspace.cosine(row) * first + workspace.sine(row) * second;
        workspace.h(row + 1, column) =
            -workspace.sine(row) * first + workspace.cosine(row) * second;
      }
      const Real diagonal = workspace.h(column, column);
      const Real subdiagonal = workspace.h(column + 1, column);
      const Real magnitude = std::hypot(diagonal, subdiagonal);
      if (!finite(magnitude) || magnitude == Real(0)) {
        dimension = column + 1;
        ++iterations;
        invalid = !finite(magnitude);
        break;
      }
      workspace.cosine(column) = diagonal / magnitude;
      workspace.sine(column) = subdiagonal / magnitude;
      workspace.h(column, column) = magnitude;
      workspace.h(column + 1, column) = Real(0);
      workspace.rotated_rhs(column + 1) =
          -workspace.sine(column) * workspace.rotated_rhs(column);
      workspace.rotated_rhs(column) *= workspace.cosine(column);
      dimension = column + 1;
      ++iterations;
      estimate_reached =
          std::fabs(workspace.rotated_rhs(column + 1)) <= estimate_threshold;
      if (estimate_reached || lucky_breakdown)
        break;
    }

    if (invalid)
      return checked_report(problem, applied_or_residual, rhs, iterate, reference, iterations,
                            SolveStatus::kInvalidEvaluation);
    if (dimension == 0 || !solve_gmres_upper(workspace, dimension))
      return checked_report(problem, applied_or_residual, rhs, iterate, reference, iterations,
                            SolveStatus::kBreakdown);
    for (int column = 0; column < dimension; ++column)
      PreparedFieldAlgebra::axpy(iterate, workspace.solution_coefficient(column), basis(column));

    residual_norm = true_residual_norm(problem, applied_or_residual, rhs, iterate);
    if (!finite(residual_norm))
      return report(reference, residual_norm, iterations, SolveStatus::kInvalidEvaluation);
    // An Arnoldi or preconditioned estimate may only request this confirmation.  It never publishes
    // success by itself; the raw scientific residual b-A(u) above is authoritative.
    if (residual_norm <= threshold)
      return report(reference, residual_norm, iterations, SolveStatus::kSolved);
    if (iterations == controls.max_iterations)
      break;
    (void)estimate_reached;
  }
  return report(reference, residual_norm, iterations, SolveStatus::kIterationLimit);
}

}  // namespace detail

/// Solve one explicitly prepared affine problem with persistent workspace.  There are no legacy raw
/// callback overloads: preparation, property checks, exact snapshot binding, and memory footprint are
/// mandatory parts of the API rather than optional caller conventions.
inline SolveReport solve_prepared_affine(const PreparedAffineLinearProblem& problem,
                                         KrylovWorkspace& workspace, MultiFab& iterate,
                                         const MultiFab& rhs, const KrylovControls& controls) {
  detail::validate_controls(controls);
  problem.require_current();
  problem.require_operator_field(iterate, "solve_prepared_affine(iterate)");
  problem.require_vector_field(rhs, "solve_prepared_affine(rhs)");
  workspace.require_bound(problem, controls);
  if (controls.method == KrylovMethod::kCg && !problem.properties().certifies_spd())
    throw std::invalid_argument("prepared CG requires an authenticated SPD operator property");
  if ((controls.method == KrylovMethod::kCg ||
       controls.method == KrylovMethod::kRichardson) &&
      problem.has_preconditioner())
    throw std::invalid_argument("prepared CG/Richardson do not expose a preconditioner slot");

  switch (controls.method) {
    case KrylovMethod::kCg:
      return detail::solve_cg(problem, workspace, iterate, rhs, controls);
    case KrylovMethod::kBicgstab:
      return detail::solve_bicgstab(problem, workspace, iterate, rhs, controls);
    case KrylovMethod::kGmres:
      return detail::solve_gmres(problem, workspace, iterate, rhs, controls);
    case KrylovMethod::kRichardson:
      return detail::solve_richardson(problem, workspace, iterate, rhs, controls);
  }
  throw std::invalid_argument("unknown prepared Krylov method");
}

}  // namespace pops
