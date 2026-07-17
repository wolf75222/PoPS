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
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace pops {
namespace detail {

/// The algorithms are the sole consumers of persistent workspace storage. Keeping this access
/// object private to detail prevents callers from replacing warmed fields or scalar buffers while
/// still letting every Krylov method share the same allocation-free storage.
struct KrylovWorkspaceAccess {
  static MultiFab& field(KrylovWorkspace& workspace, std::size_t index) {
    return workspace.field(index);
  }
  static Real& h(KrylovWorkspace& workspace, int row, int column) {
    return workspace.h(row, column);
  }
  static Real& cosine(KrylovWorkspace& workspace, int index) {
    return workspace.cosine(index);
  }
  static Real& sine(KrylovWorkspace& workspace, int index) { return workspace.sine(index); }
  static Real& rotated_rhs(KrylovWorkspace& workspace, int index) {
    return workspace.rotated_rhs(index);
  }
  static Real& solution_coefficient(KrylovWorkspace& workspace, int index) {
    return workspace.solution_coefficient(index);
  }
  static double* gmres_reduction_data(KrylovWorkspace& workspace) {
    return workspace.gmres_reduction_data();
  }
  static std::size_t gmres_reduction_size(const KrylovWorkspace& workspace) {
    return workspace.gmres_reduction_size();
  }
};

inline bool finite(Real value) {
  return std::isfinite(static_cast<double>(value));
}

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
  if (controls.method != KrylovMethod::kRichardson && controls.relaxation != Real(1))
    throw std::invalid_argument("relaxation belongs only to prepared Richardson");
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
  PreparedProblemAccess::true_residual(problem, scratch, rhs, iterate);
  return PreparedProblemAccess::residual_norm(problem, scratch);
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
  MultiFab& effective_rhs = KrylovWorkspaceAccess::field(workspace, 0);
  MultiFab& residual = KrylovWorkspaceAccess::field(workspace, 1);
  PreparedProblemAccess::effective_rhs(problem, effective_rhs, rhs);
  const Real reference = PreparedProblemAccess::residual_norm(problem, effective_rhs);
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
  return report(reference, residual_norm, controls.max_iterations, SolveStatus::kIterationLimit);
}

inline SolveReport solve_cg(const PreparedAffineLinearProblem& problem, KrylovWorkspace& workspace,
                            MultiFab& iterate, const MultiFab& rhs,
                            const KrylovControls& controls) {
  MultiFab& effective_rhs = KrylovWorkspaceAccess::field(workspace, 0);
  MultiFab& residual = KrylovWorkspaceAccess::field(workspace, 1);
  MultiFab& direction = KrylovWorkspaceAccess::field(workspace, 2);
  MultiFab& applied = KrylovWorkspaceAccess::field(workspace, 3);
  PreparedProblemAccess::effective_rhs(problem, effective_rhs, rhs);
  const Real reference = PreparedProblemAccess::residual_norm(problem, effective_rhs);
  Real residual_norm = true_residual_norm(problem, residual, rhs, iterate);
  if (!finite(reference) || !finite(residual_norm))
    return report(reference, residual_norm, 0, SolveStatus::kInvalidEvaluation);
  const Real threshold = stopping_threshold(reference, controls);
  if (residual_norm <= threshold)
    return report(reference, residual_norm, 0, SolveStatus::kSolved);

  PreparedFieldAlgebra::copy(direction, residual);
  Real squared = PreparedProblemAccess::inner_product(problem, residual, residual);
  for (int iteration = 1; iteration <= controls.max_iterations; ++iteration) {
    PreparedProblemAccess::apply_linear(problem, applied, direction);
    const Real curvature = PreparedProblemAccess::inner_product(problem, direction, applied);
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
    Real next_squared = PreparedProblemAccess::inner_product(problem, residual, residual);
    residual_norm =
        next_squared >= Real(0) ? std::sqrt(next_squared) : std::numeric_limits<Real>::quiet_NaN();
    if (!finite(residual_norm))
      return checked_report(problem, applied, rhs, iterate, reference, iteration,
                            SolveStatus::kInvalidEvaluation);
    bool restart_recurrence = false;
    if (residual_norm <= threshold) {
      const Real confirmed = true_residual_norm(problem, applied, rhs, iterate);
      if (!finite(confirmed))
        return report(reference, confirmed, iteration, SolveStatus::kInvalidEvaluation);
      if (confirmed <= threshold)
        return report(reference, confirmed, iteration, SolveStatus::kSolved);
      PreparedFieldAlgebra::copy(residual, applied);
      next_squared = confirmed * confirmed;
      if (!finite(next_squared))
        return report(reference, confirmed, iteration, SolveStatus::kInvalidEvaluation);
      restart_recurrence = true;
    }
    if (iteration == controls.max_iterations)
      return checked_report(problem, applied, rhs, iterate, reference, iteration,
                            SolveStatus::kIterationLimit);
    if (restart_recurrence) {
      // The recursive residual reached the tolerance but the scientific residual did not. Replace
      // the complete CG recurrence; mixing refreshed r with the old p/rTr state is not CG.
      PreparedFieldAlgebra::copy(direction, residual);
    } else {
      if (squared <= Real(0))
        return checked_report(problem, applied, rhs, iterate, reference, iteration,
                              SolveStatus::kBreakdown);
      const Real beta = next_squared / squared;
      PreparedFieldAlgebra::lincomb(direction, Real(1), residual, beta, direction);
    }
    squared = next_squared;
  }
  return checked_report(problem, applied, rhs, iterate, reference, controls.max_iterations,
                        SolveStatus::kIterationLimit);
}

inline SolveReport solve_bicgstab(const PreparedAffineLinearProblem& problem,
                                  KrylovWorkspace& workspace, MultiFab& iterate,
                                  const MultiFab& rhs, const KrylovControls& controls) {
  MultiFab& effective_rhs = KrylovWorkspaceAccess::field(workspace, 0);
  MultiFab& residual = KrylovWorkspaceAccess::field(workspace, 1);
  MultiFab& shadow = KrylovWorkspaceAccess::field(workspace, 2);
  MultiFab& direction = KrylovWorkspaceAccess::field(workspace, 3);
  MultiFab& applied = KrylovWorkspaceAccess::field(workspace, 4);
  MultiFab& intermediate = KrylovWorkspaceAccess::field(workspace, 5);
  MultiFab& second_applied = KrylovWorkspaceAccess::field(workspace, 6);
  MultiFab& prepared_direction =
      problem.has_preconditioner() ? KrylovWorkspaceAccess::field(workspace, 7) : direction;
  MultiFab& prepared_intermediate =
      problem.has_preconditioner() ? KrylovWorkspaceAccess::field(workspace, 8) : intermediate;

  PreparedProblemAccess::effective_rhs(problem, effective_rhs, rhs);
  const Real reference = PreparedProblemAccess::residual_norm(problem, effective_rhs);
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
  bool restart_recurrence = false;

  for (int iteration = 1; iteration <= controls.max_iterations; ++iteration) {
    const Real rho = PreparedProblemAccess::inner_product(problem, shadow, residual);
    if (!finite(rho))
      return checked_report(problem, second_applied, rhs, iterate, reference, iteration - 1,
                            SolveStatus::kInvalidEvaluation);
    if (rho == Real(0))
      return checked_report(problem, second_applied, rhs, iterate, reference, iteration - 1,
                            SolveStatus::kBreakdown);

    if (iteration == 1 || restart_recurrence) {
      PreparedFieldAlgebra::copy(direction, residual);
      restart_recurrence = false;
    } else {
      if (omega == Real(0))
        return checked_report(problem, second_applied, rhs, iterate, reference, iteration - 1,
                              SolveStatus::kBreakdown);
      const Real beta = (rho / rho_previous) * (alpha / omega);
      PreparedFieldAlgebra::axpy(direction, -omega, applied);
      PreparedFieldAlgebra::lincomb(direction, Real(1), residual, beta, direction);
    }

    if (problem.has_preconditioner())
      PreparedProblemAccess::apply_preconditioner(problem, prepared_direction, direction);
    PreparedProblemAccess::apply_linear(problem, applied, prepared_direction);
    const Real denominator = PreparedProblemAccess::inner_product(problem, shadow, applied);
    if (!finite(denominator) || denominator == Real(0))
      return checked_report(
          problem, second_applied, rhs, iterate, reference, iteration - 1,
          finite(denominator) ? SolveStatus::kBreakdown : SolveStatus::kInvalidEvaluation);
    alpha = rho / denominator;
    PreparedFieldAlgebra::lincomb(intermediate, Real(1), residual, -alpha, applied);

    const Real intermediate_norm = PreparedProblemAccess::residual_norm(problem, intermediate);
    if (!finite(intermediate_norm))
      return checked_report(problem, second_applied, rhs, iterate, reference, iteration - 1,
                            SolveStatus::kInvalidEvaluation);
    bool alpha_committed = false;
    bool refreshed = false;
    if (intermediate_norm <= threshold) {
      PreparedFieldAlgebra::axpy(iterate, alpha, prepared_direction);
      alpha_committed = true;
      const Real confirmed = true_residual_norm(problem, second_applied, rhs, iterate);
      if (!finite(confirmed))
        return report(reference, confirmed, iteration, SolveStatus::kInvalidEvaluation);
      if (confirmed <= threshold)
        return report(reference, confirmed, iteration, SolveStatus::kSolved);
      PreparedFieldAlgebra::copy(intermediate, second_applied);
      refreshed = true;
    }

    if (problem.has_preconditioner())
      PreparedProblemAccess::apply_preconditioner(problem, prepared_intermediate, intermediate);
    PreparedProblemAccess::apply_linear(problem, second_applied, prepared_intermediate);
    const Real second_norm_squared =
        PreparedProblemAccess::inner_product(problem, second_applied, second_applied);
    const Real projection =
        PreparedProblemAccess::inner_product(problem, second_applied, intermediate);
    if (!finite(second_norm_squared) || !finite(projection))
      return checked_report(problem, second_applied, rhs, iterate, reference, iteration - 1,
                            SolveStatus::kInvalidEvaluation);
    if (second_norm_squared <= Real(0))
      return checked_report(problem, second_applied, rhs, iterate, reference,
                            alpha_committed ? iteration : iteration - 1, SolveStatus::kBreakdown);
    omega = projection / second_norm_squared;
    if (!finite(omega) || omega == Real(0))
      return checked_report(
          problem, second_applied, rhs, iterate, reference,
          alpha_committed ? iteration : iteration - 1,
          finite(omega) ? SolveStatus::kBreakdown : SolveStatus::kInvalidEvaluation);

    if (!alpha_committed)
      PreparedFieldAlgebra::axpy(iterate, alpha, prepared_direction);
    PreparedFieldAlgebra::axpy(iterate, omega, prepared_intermediate);
    // The BiCGStab recurrence already supplies the next residual. Recomputing b-A(x) here would
    // add a third operator application to every full iteration. Its norm may request an
    // authoritative true-residual confirmation, but can never publish success by itself.
    PreparedFieldAlgebra::lincomb(residual, Real(1), intermediate, -omega, second_applied);
    residual_norm = PreparedProblemAccess::residual_norm(problem, residual);
    if (!finite(residual_norm))
      return checked_report(problem, second_applied, rhs, iterate, reference, iteration,
                            SolveStatus::kInvalidEvaluation);
    if (residual_norm <= threshold) {
      const Real confirmed = true_residual_norm(problem, second_applied, rhs, iterate);
      if (!finite(confirmed))
        return report(reference, confirmed, iteration, SolveStatus::kInvalidEvaluation);
      if (confirmed <= threshold)
        return report(reference, confirmed, iteration, SolveStatus::kSolved);
      // Recursive drift produced a false candidate. Refresh from the scientific residual before
      // continuing.
      PreparedFieldAlgebra::copy(residual, second_applied);
      residual_norm = confirmed;
      refreshed = true;
    }
    // A failed true-residual confirmation is an explicit recurrence-replacement event. Restart
    // BiCGStab from that corrected residual instead of mixing refreshed r with stale p/rho state.
    if (refreshed) {
      PreparedFieldAlgebra::copy(shadow, residual);
      restart_recurrence = true;
    }
    rho_previous = rho;
  }
  return checked_report(problem, second_applied, rhs, iterate, reference, controls.max_iterations,
                        SolveStatus::kIterationLimit);
}

inline void reset_gmres_scalars(KrylovWorkspace& workspace, int restart) {
  for (int row = 0; row <= restart; ++row) {
    KrylovWorkspaceAccess::rotated_rhs(workspace, row) = Real(0);
    if (row < restart) {
      KrylovWorkspaceAccess::cosine(workspace, row) = Real(0);
      KrylovWorkspaceAccess::sine(workspace, row) = Real(0);
      KrylovWorkspaceAccess::solution_coefficient(workspace, row) = Real(0);
    }
    for (int column = 0; column < restart; ++column)
      KrylovWorkspaceAccess::h(workspace, row, column) = Real(0);
  }
}

inline bool solve_gmres_upper(KrylovWorkspace& workspace, int dimension) {
  for (int row = dimension - 1; row >= 0; --row) {
    Real value = KrylovWorkspaceAccess::rotated_rhs(workspace, row);
    for (int column = row + 1; column < dimension; ++column)
      value -= KrylovWorkspaceAccess::h(workspace, row, column) *
               KrylovWorkspaceAccess::solution_coefficient(workspace, column);
    const Real diagonal = KrylovWorkspaceAccess::h(workspace, row, row);
    if (!finite(diagonal) || diagonal == Real(0))
      return false;
    KrylovWorkspaceAccess::solution_coefficient(workspace, row) = value / diagonal;
  }
  return true;
}

inline SolveReport solve_gmres(const PreparedAffineLinearProblem& problem,
                               KrylovWorkspace& workspace, MultiFab& iterate, const MultiFab& rhs,
                               const KrylovControls& controls) {
  const int restart = controls.restart;
  MultiFab& effective_rhs = KrylovWorkspaceAccess::field(workspace, 0);
  const auto basis = [&workspace](int index) -> MultiFab& {
    return KrylovWorkspaceAccess::field(workspace, static_cast<std::size_t>(index + 1));
  };
  MultiFab& applied_or_residual =
      KrylovWorkspaceAccess::field(workspace, static_cast<std::size_t>(restart + 2));
  MultiFab* prepared_vector =
      problem.has_preconditioner()
          ? &KrylovWorkspaceAccess::field(workspace, static_cast<std::size_t>(restart + 3))
          : nullptr;
  if (KrylovWorkspaceAccess::gmres_reduction_size(workspace) <
      static_cast<std::size_t>(restart + 1))
    throw std::logic_error("prepared GMRES reduction workspace is undersized");

  PreparedProblemAccess::effective_rhs(problem, effective_rhs, rhs);
  const Real reference = PreparedProblemAccess::residual_norm(problem, effective_rhs);
  Real residual_norm = true_residual_norm(problem, applied_or_residual, rhs, iterate);
  if (!finite(reference) || !finite(residual_norm))
    return report(reference, residual_norm, 0, SolveStatus::kInvalidEvaluation);
  const Real threshold = stopping_threshold(reference, controls);
  if (residual_norm <= threshold)
    return report(reference, residual_norm, 0, SolveStatus::kSolved);

  int iterations = 0;
  while (iterations < controls.max_iterations) {
    MultiFab* initial_vector = &applied_or_residual;
    if (prepared_vector != nullptr) {
      PreparedProblemAccess::apply_preconditioner(problem, *prepared_vector, applied_or_residual);
      initial_vector = prepared_vector;
    }
    const Real beta = PreparedProblemAccess::residual_norm(problem, *initial_vector);
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
    PreparedFieldAlgebra::lincomb(basis(0), Real(1) / beta, *initial_vector, Real(0),
                                  *initial_vector);
    reset_gmres_scalars(workspace, restart);
    KrylovWorkspaceAccess::rotated_rhs(workspace, 0) = beta;

    int dimension = 0;
    bool estimate_reached = false;
    bool invalid = false;
    for (int column = 0; column < restart && iterations < controls.max_iterations; ++column) {
      PreparedProblemAccess::apply_linear(problem, applied_or_residual, basis(column));
      MultiFab* arnoldi_vector = &applied_or_residual;
      if (prepared_vector != nullptr) {
        PreparedProblemAccess::apply_preconditioner(problem, *prepared_vector, applied_or_residual);
        arnoldi_vector = prepared_vector;
      }
      // Classical Arnoldi computes all local projections before one vector reduction. The final
      // slot carries the unprojected ||w||^2 used only by the DGKS norm-loss criterion. The
      // projected norm itself is evaluated directly: deriving it as ||w||^2-sum(h^2) assumes an
      // exactly orthonormal basis and can silently mis-normalize a finite-precision CGS basis.
      double* reductions = KrylovWorkspaceAccess::gmres_reduction_data(workspace);
      for (int row = 0; row <= column; ++row)
        reductions[row] = static_cast<double>(PreparedProblemAccess::local_inner_product(
            problem, *arnoldi_vector, basis(row)));
      reductions[column + 1] = static_cast<double>(PreparedProblemAccess::local_inner_product(
          problem, *arnoldi_vector, *arnoldi_vector));
      all_reduce_sum_inplace(reductions, column + 2);

      bool finite_column = finite(static_cast<Real>(reductions[column + 1]));
      for (int row = 0; row <= column; ++row) {
        const Real projection = static_cast<Real>(reductions[row]);
        finite_column = finite_column && finite(projection);
        KrylovWorkspaceAccess::h(workspace, row, column) = projection;
        PreparedFieldAlgebra::axpy(*arnoldi_vector, -projection, basis(row));
      }
      const Real raw_square = static_cast<Real>(reductions[column + 1]);
      Real arnoldi_norm =
          finite_column && raw_square >= Real(0)
              ? PreparedProblemAccess::residual_norm(problem, *arnoldi_vector)
              : std::numeric_limits<Real>::quiet_NaN();

      // Selective CGS2 restores MGS-class robustness on the hard columns without returning to one
      // MPI collective per basis vector. A second batched pass is paid only when the first
      // projection loses at least half the vector norm (the standard DGKS trigger).
      constexpr Real kReorthogonalizeRatio = Real(0.5);
      if (finite(arnoldi_norm) && raw_square > Real(0) &&
          arnoldi_norm <= kReorthogonalizeRatio * std::sqrt(raw_square)) {
        for (int row = 0; row <= column; ++row)
          reductions[row] = static_cast<double>(PreparedProblemAccess::local_inner_product(
              problem, *arnoldi_vector, basis(row)));
        all_reduce_sum_inplace(reductions, column + 1);
        for (int row = 0; row <= column; ++row) {
          const Real correction = static_cast<Real>(reductions[row]);
          finite_column = finite_column && finite(correction);
          KrylovWorkspaceAccess::h(workspace, row, column) += correction;
          PreparedFieldAlgebra::axpy(*arnoldi_vector, -correction, basis(row));
        }
        arnoldi_norm = finite_column
                           ? PreparedProblemAccess::residual_norm(problem, *arnoldi_vector)
                           : std::numeric_limits<Real>::quiet_NaN();
      }
      KrylovWorkspaceAccess::h(workspace, column + 1, column) = arnoldi_norm;
      if (!finite(arnoldi_norm)) {
        invalid = true;
        break;
      }
      const bool lucky_breakdown = arnoldi_norm == Real(0);
      if (!lucky_breakdown) {
        PreparedFieldAlgebra::lincomb(basis(column + 1), Real(1) / arnoldi_norm, *arnoldi_vector,
                                      Real(0), *arnoldi_vector);
      }

      for (int row = 0; row < column; ++row) {
        const Real first = KrylovWorkspaceAccess::h(workspace, row, column);
        const Real second = KrylovWorkspaceAccess::h(workspace, row + 1, column);
        KrylovWorkspaceAccess::h(workspace, row, column) =
            KrylovWorkspaceAccess::cosine(workspace, row) * first +
            KrylovWorkspaceAccess::sine(workspace, row) * second;
        KrylovWorkspaceAccess::h(workspace, row + 1, column) =
            -KrylovWorkspaceAccess::sine(workspace, row) * first +
            KrylovWorkspaceAccess::cosine(workspace, row) * second;
      }
      const Real diagonal = KrylovWorkspaceAccess::h(workspace, column, column);
      const Real subdiagonal = KrylovWorkspaceAccess::h(workspace, column + 1, column);
      const Real magnitude = std::hypot(diagonal, subdiagonal);
      if (!finite(magnitude) || magnitude == Real(0)) {
        dimension = column + 1;
        ++iterations;
        invalid = !finite(magnitude);
        break;
      }
      KrylovWorkspaceAccess::cosine(workspace, column) = diagonal / magnitude;
      KrylovWorkspaceAccess::sine(workspace, column) = subdiagonal / magnitude;
      KrylovWorkspaceAccess::h(workspace, column, column) = magnitude;
      KrylovWorkspaceAccess::h(workspace, column + 1, column) = Real(0);
      KrylovWorkspaceAccess::rotated_rhs(workspace, column + 1) =
          -KrylovWorkspaceAccess::sine(workspace, column) *
          KrylovWorkspaceAccess::rotated_rhs(workspace, column);
      KrylovWorkspaceAccess::rotated_rhs(workspace, column) *=
          KrylovWorkspaceAccess::cosine(workspace, column);
      dimension = column + 1;
      ++iterations;
      estimate_reached = std::fabs(
                             KrylovWorkspaceAccess::rotated_rhs(workspace, column + 1)) <=
                         estimate_threshold;
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
      PreparedFieldAlgebra::axpy(
          iterate, KrylovWorkspaceAccess::solution_coefficient(workspace, column), basis(column));

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
  if (iterate.shares_storage_with(rhs))
    throw std::invalid_argument(
        "solve_prepared_affine requires iterate and rhs to use distinct storage");
  workspace.require_bound(problem, controls);
  if (controls.method == KrylovMethod::kCg && !problem.properties().certifies_spd())
    throw std::invalid_argument("prepared CG requires an authenticated SPD operator property");
  if ((controls.method == KrylovMethod::kCg || controls.method == KrylovMethod::kRichardson) &&
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
