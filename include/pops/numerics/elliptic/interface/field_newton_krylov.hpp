/// @file
/// @brief Allocation-free exact-ranked damped Newton--GMRES field solve.

#pragma once

#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/interface/field_nonlinear.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pops {

/// Persistent workspace and algorithm for one scalar nonlinear field on a uniform exact-ranked
/// layout. Residual, JVP and gauge providers are direct callable objects selected before the solve;
/// every Krylov allocation is completed by construction; no Python callback, registry lookup or
/// dimension switch occurs in Newton/GMRES.
template <int Dim>
class FieldNewtonKrylovWorkspace final {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "FieldNewtonKrylovWorkspace supports dimensions 1, 2, and 3");
  using field_type = MultiFab<Dim>;

  FieldNewtonKrylovWorkspace(const mesh::BoxArray<Dim>& layout,
                             const mesh::Distribution<Dim>& distribution, Index<Dim> local_rank,
                             FieldNewtonOptions options)
      : options_(options),
        residual_(layout, distribution, local_rank, 1, Extent<Dim>{}),
        trial_(layout, distribution, local_rank, 1, Extent<Dim>{}),
        trial_residual_(layout, distribution, local_rank, 1, Extent<Dim>{}),
        correction_(layout, distribution, local_rank, 1, Extent<Dim>{}),
        linear_residual_(layout, distribution, local_rank, 1, Extent<Dim>{}),
        image_(layout, distribution, local_rank, 1, Extent<Dim>{}),
        work_(layout, distribution, local_rank, 1, Extent<Dim>{}) {
    validate_field_newton_options(options_);
    if (options_.restart > options_.linear_max_iterations)
      throw std::invalid_argument(
          "field Newton GMRES restart cannot exceed its linear iteration budget");
    const std::size_t restart = static_cast<std::size_t>(options_.restart);
    if (restart > std::numeric_limits<std::size_t>::max() / (restart + 1u))
      throw std::length_error("field Newton GMRES Hessenberg extent overflows size_t");
    basis_.reserve(restart + 1u);
    for (std::size_t index = 0; index <= restart; ++index)
      basis_.emplace_back(layout, distribution, local_rank, 1, Extent<Dim>{});
    hessenberg_.resize((restart + 1u) * restart);
    cosine_.resize(restart);
    sine_.resize(restart);
    rotated_rhs_.resize(restart + 1u);
    coefficients_.resize(restart);
  }

  FieldNewtonKrylovWorkspace(const FieldNewtonKrylovWorkspace&) = delete;
  FieldNewtonKrylovWorkspace& operator=(const FieldNewtonKrylovWorkspace&) = delete;
  FieldNewtonKrylovWorkspace(FieldNewtonKrylovWorkspace&&) = default;
  FieldNewtonKrylovWorkspace& operator=(FieldNewtonKrylovWorkspace&&) = default;

  const FieldNewtonOptions& options() const noexcept { return options_; }

  template <class ResidualProvider, class JvpProvider, class GaugeProvider>
  SolveReport solve(field_type& iterate, ResidualProvider&& evaluate_residual,
                    JvpProvider&& apply_jvp, GaugeProvider&& apply_gauge) {
    auto&& residual_provider = evaluate_residual;
    auto&& jvp_provider = apply_jvp;
    auto&& gauge_provider = apply_gauge;
    authenticate_(iterate, "iterate");
    gauge_provider(iterate);
    residual_provider(iterate, residual_, 0);
    Kokkos::fence();

    SolveReport report;
    report.evaluations = 1;
    const Real initial_norm = norm_(residual_);
    report.reference_residual_norm = initial_norm;
    report.residual_norm = initial_norm;
    const Real relative_denominator = initial_norm > Real(0) ? initial_norm : Real(1);
    report.rel_residual = initial_norm / relative_denominator;
    if (!finite_(initial_norm)) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                         "field_newton_non_finite_initial_residual");
      return report;
    }
    const Real nonlinear_stop =
        options_.tolerance * std::max(Real(1), report.reference_residual_norm);
    if (initial_norm <= nonlinear_stop) {
      report.rel_residual = Real(0);
      report.mark_solved("field_newton_initial_residual");
      return report;
    }

    for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
      correction_.set_val(Real(0));
      const Real linear_stop = options_.linear_tolerance * report.residual_norm;
      const LinearResult linear =
          solve_linear_(iterate, residual_, linear_stop, jvp_provider, iteration);
      report.evaluations += linear.evaluations;
      if (!linear.converged) {
        report.iters = iteration;
        report.mark_failed(SolveStatus::kBreakdown, SolveAction::kRejectAttempt,
                           "field_newton_gmres_breakdown");
        return report;
      }

      gauge_provider(correction_);
      const Real full_step_norm = norm_(correction_);
      if (!finite_(full_step_norm)) {
        report.iters = iteration;
        report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kRejectAttempt,
                           "field_newton_non_finite_correction");
        return report;
      }

      bool accepted = false;
      Real step = Real(1);
      Real accepted_step = Real(0);
      Real accepted_norm = std::numeric_limits<Real>::infinity();
      while (step >= options_.minimum_step) {
        lincomb(trial_, Real(1), iterate, step, correction_);
        gauge_provider(trial_);
        residual_provider(trial_, trial_residual_, iteration + 1);
        Kokkos::fence();
        ++report.evaluations;
        const Real trial_norm = norm_(trial_residual_);
        if (finite_(trial_norm) &&
            trial_norm <= (Real(1) - options_.armijo * step) * report.residual_norm) {
          copy_(trial_, iterate);
          copy_(trial_residual_, residual_);
          accepted_norm = trial_norm;
          accepted_step = step;
          accepted = true;
          break;
        }
        step *= Real(0.5);
      }
      report.iters = iteration + 1;
      report.step_norm = accepted_step * full_step_norm;
      if (!accepted) {
        report.mark_failed(SolveStatus::kBreakdown, SolveAction::kRejectAttempt,
                           "field_newton_line_search_failed");
        return report;
      }

      report.residual_norm = accepted_norm;
      report.rel_residual = accepted_norm / relative_denominator;
      if (accepted_norm <= nonlinear_stop) {
        report.mark_solved("field_newton_converged");
        return report;
      }
    }

    report.mark_failed(SolveStatus::kIterationLimit, SolveAction::kRejectAttempt,
                       "field_newton_iteration_limit");
    return report;
  }

 private:
  struct LinearResult {
    bool converged = false;
    int evaluations = 0;
  };

  template <class JvpProvider>
  LinearResult solve_linear_(const field_type& iterate, const field_type& rhs, Real stop,
                             JvpProvider& apply_jvp, int nonlinear_iteration) {
    copy_(rhs, linear_residual_);
    Real beta = norm_(linear_residual_);
    LinearResult result;
    if (!finite_(beta))
      return result;
    if (beta <= stop) {
      result.converged = true;
      return result;
    }

    int completed = 0;
    while (completed < options_.linear_max_iterations) {
      const int cycle = std::min(options_.restart, options_.linear_max_iterations - completed);
      copy_(linear_residual_, basis_[0]);
      scale(basis_[0], Real(1) / beta);
      std::fill(hessenberg_.begin(), hessenberg_.end(), Real(0));
      std::fill(cosine_.begin(), cosine_.end(), Real(0));
      std::fill(sine_.begin(), sine_.end(), Real(0));
      std::fill(rotated_rhs_.begin(), rotated_rhs_.end(), Real(0));
      rotated_rhs_[0] = beta;

      int used = 0;
      bool cycle_converged = false;
      for (int column = 0; column < cycle; ++column) {
        apply_jvp(iterate, basis_[static_cast<std::size_t>(column)], work_, nonlinear_iteration);
        Kokkos::fence();
        ++result.evaluations;
        for (int row = 0; row <= column; ++row) {
          h_(row, column) = dot(work_, basis_[static_cast<std::size_t>(row)]);
          saxpy(work_, -h_(row, column), basis_[static_cast<std::size_t>(row)]);
        }
        h_(column + 1, column) = norm_(work_);
        if (h_(column + 1, column) > Real(0)) {
          copy_(work_, basis_[static_cast<std::size_t>(column + 1)]);
          scale(basis_[static_cast<std::size_t>(column + 1)], Real(1) / h_(column + 1, column));
        }

        for (int rotation = 0; rotation < column; ++rotation) {
          const Real upper = h_(rotation, column);
          const Real lower = h_(rotation + 1, column);
          h_(rotation, column) = cosine_[static_cast<std::size_t>(rotation)] * upper +
                                 sine_[static_cast<std::size_t>(rotation)] * lower;
          h_(rotation + 1, column) = -sine_[static_cast<std::size_t>(rotation)] * upper +
                                     cosine_[static_cast<std::size_t>(rotation)] * lower;
        }
        const Real diagonal = h_(column, column);
        const Real subdiagonal = h_(column + 1, column);
        const Real magnitude = std::hypot(diagonal, subdiagonal);
        if (!finite_(magnitude) || magnitude == Real(0)) {
          used = column;
          break;
        }
        cosine_[static_cast<std::size_t>(column)] = diagonal / magnitude;
        sine_[static_cast<std::size_t>(column)] = subdiagonal / magnitude;
        h_(column, column) = magnitude;
        h_(column + 1, column) = Real(0);
        const Real value = rotated_rhs_[static_cast<std::size_t>(column)];
        rotated_rhs_[static_cast<std::size_t>(column)] =
            cosine_[static_cast<std::size_t>(column)] * value;
        rotated_rhs_[static_cast<std::size_t>(column + 1)] =
            -sine_[static_cast<std::size_t>(column)] * value;
        used = column + 1;
        ++completed;
        if (std::abs(rotated_rhs_[static_cast<std::size_t>(column + 1)]) <= stop) {
          cycle_converged = true;
          break;
        }
      }
      if (used == 0)
        return result;
      if (!update_correction_(used))
        return result;
      if (cycle_converged) {
        result.converged = true;
        return result;
      }
      apply_jvp(iterate, correction_, image_, nonlinear_iteration);
      Kokkos::fence();
      ++result.evaluations;
      lincomb(linear_residual_, Real(1), rhs, Real(-1), image_);
      beta = norm_(linear_residual_);
      if (!finite_(beta))
        return result;
      if (beta <= stop) {
        result.converged = true;
        return result;
      }
    }
    return result;
  }

  bool update_correction_(int used) {
    for (int reverse = used; reverse != 0; --reverse) {
      const int row = reverse - 1;
      Real value = rotated_rhs_[static_cast<std::size_t>(row)];
      for (int column = row + 1; column < used; ++column)
        value -= h_(row, column) * coefficients_[static_cast<std::size_t>(column)];
      const Real diagonal = h_(row, row);
      if (!finite_(diagonal) || diagonal == Real(0))
        return false;
      coefficients_[static_cast<std::size_t>(row)] = value / diagonal;
      if (!finite_(coefficients_[static_cast<std::size_t>(row)]))
        return false;
    }
    for (int index = 0; index < used; ++index)
      saxpy(correction_, coefficients_[static_cast<std::size_t>(index)],
            basis_[static_cast<std::size_t>(index)]);
    return true;
  }

  Real& h_(int row, int column) {
    return hessenberg_[static_cast<std::size_t>(column) *
                           static_cast<std::size_t>(options_.restart + 1) +
                       static_cast<std::size_t>(row)];
  }

  static void copy_(const field_type& source, field_type& destination) {
    if (source.layout() != destination.layout() ||
        source.distribution() != destination.distribution() ||
        source.local_rank() != destination.local_rank() || source.ncomp() != 1 ||
        destination.ncomp() != 1)
      throw std::invalid_argument("field Newton workspace layout differs from its vector");
    lincomb(destination, Real(1), source, Real(0), source);
  }

  void authenticate_(const field_type& field, const char* role) const {
    if (field.layout() != residual_.layout() || field.distribution() != residual_.distribution() ||
        field.local_rank() != residual_.local_rank() || field.ncomp() != 1)
      throw std::invalid_argument(std::string("field Newton ") + role +
                                  " differs from its prepared exact-ranked layout");
  }

  static Real norm_(const field_type& field) {
    const Real squared = dot(field, field);
    if (!finite_(squared))
      return squared;
    if (squared < Real(0))
      return std::numeric_limits<Real>::quiet_NaN();
    return squared > Real(0) ? std::sqrt(squared) : Real(0);
  }

  static bool finite_(Real value) noexcept { return std::isfinite(static_cast<double>(value)); }

  FieldNewtonOptions options_;
  field_type residual_;
  field_type trial_;
  field_type trial_residual_;
  field_type correction_;
  field_type linear_residual_;
  field_type image_;
  field_type work_;
  std::vector<field_type> basis_;
  std::vector<Real> hessenberg_;
  std::vector<Real> cosine_;
  std::vector<Real> sine_;
  std::vector<Real> rotated_rhs_;
  std::vector<Real> coefficients_;
};

}  // namespace pops
