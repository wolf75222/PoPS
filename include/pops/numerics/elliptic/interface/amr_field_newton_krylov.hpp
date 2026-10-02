/// @file
/// @brief Allocation-free exact-ranked Newton--GMRES over one AMR field hierarchy.

#pragma once

#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/parallel/collective_exception.hpp>
#include <pops/numerics/elliptic/interface/field_nonlinear.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {

/// Persistent nonlinear/Krylov storage for one ordered field vector carried by an exact AMR hierarchy.
///
/// Covered parent cells are excluded from every scalar product through immutable active-cell masks;
/// level cell measures keep the Krylov norm physically consistent across refinement.  Residual, JVP
/// and gauge callbacks are direct C++ callables selected before the solve.  The destination hierarchy
/// is published only after Newton produces a solved value, so a rejected attempt cannot leak a trial
/// iterate into the runtime-owned candidate.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class AmrFieldNewtonKrylovWorkspace final {
 public:
  static_assert(Dim >= 1 && Dim <= 3,
                "AmrFieldNewtonKrylovWorkspace supports dimensions 1, 2, and 3");
  using field_type = MultiFab<Dim, MemorySpace>;
  using hierarchy_type = std::vector<field_type>;

  AmrFieldNewtonKrylovWorkspace(std::span<const field_type* const> layouts,
                                std::span<const field_type* const> active_cells,
                                std::span<const Real> cell_measures, FieldNewtonOptions options)
      : options_(options),
        active_cells_(active_cells.begin(), active_cells.end()),
        cell_measures_(cell_measures.begin(), cell_measures.end()) {
    validate_field_newton_options(options_);
    if (layouts.empty() || active_cells_.size() != layouts.size() ||
        cell_measures_.size() != layouts.size())
      throw std::invalid_argument(
          "AMR field Newton requires one active mask and cell measure per level");
    const int components = layouts.front() ? layouts.front()->ncomp() : 0;
    if (components <= 0)
      throw std::invalid_argument("AMR field Newton requires nonempty vector components");
    for (std::size_t level = 0; level < layouts.size(); ++level) {
      if (layouts[level] == nullptr || active_cells_[level] == nullptr ||
          layouts[level]->ncomp() != components || active_cells_[level]->ncomp() != 1 ||
          !same_layout_(*layouts[level], *active_cells_[level]) ||
          !finite_(cell_measures_[level]) || !(cell_measures_[level] > Real(0)))
        throw std::invalid_argument(
            "AMR field Newton received an invalid exact-ranked hierarchy layout");
    }
    if (options_.restart > options_.linear_max_iterations)
      throw std::invalid_argument(
          "AMR field Newton GMRES restart cannot exceed its linear iteration budget");
    const std::size_t restart = static_cast<std::size_t>(options_.restart);
    if (restart > std::numeric_limits<std::size_t>::max() / (restart + 1u))
      throw std::length_error("AMR field Newton GMRES Hessenberg extent overflows size_t");

    iterate_ = make_hierarchy_(layouts);
    residual_ = make_hierarchy_(layouts);
    trial_ = make_hierarchy_(layouts);
    trial_residual_ = make_hierarchy_(layouts);
    correction_ = make_hierarchy_(layouts);
    linear_residual_ = make_hierarchy_(layouts);
    image_ = make_hierarchy_(layouts);
    work_ = make_hierarchy_(layouts);
    basis_.reserve(restart + 1u);
    for (std::size_t index = 0; index <= restart; ++index)
      basis_.push_back(make_hierarchy_(layouts));
    hessenberg_.resize((restart + 1u) * restart);
    cosine_.resize(restart);
    sine_.resize(restart);
    rotated_rhs_.resize(restart + 1u);
    coefficients_.resize(restart);
  }

  AmrFieldNewtonKrylovWorkspace(const AmrFieldNewtonKrylovWorkspace&) = delete;
  AmrFieldNewtonKrylovWorkspace& operator=(const AmrFieldNewtonKrylovWorkspace&) = delete;
  AmrFieldNewtonKrylovWorkspace(AmrFieldNewtonKrylovWorkspace&&) = default;
  AmrFieldNewtonKrylovWorkspace& operator=(AmrFieldNewtonKrylovWorkspace&&) = default;

  const FieldNewtonOptions& options() const noexcept { return options_; }

  template <class ResidualProvider, class JvpProvider, class GaugeProvider>
  SolveReport solve(std::span<field_type* const> destination, ResidualProvider&& evaluate_residual,
                    JvpProvider&& apply_jvp, GaugeProvider&& apply_gauge, const ExecutionLane& lane,
                    bool collective_local_phases = false) {
    auto identity = [&](const hierarchy_type& input, hierarchy_type& output) { copy_(input, output); };
    return solve_preconditioned(destination, std::forward<ResidualProvider>(evaluate_residual),
                                std::forward<JvpProvider>(apply_jvp),
                                std::forward<GaugeProvider>(apply_gauge), identity, lane,
                                collective_local_phases);
  }

  /// Stationary linear right-preconditioner realization. The callback owns any
  /// collective preparation/authority protocol; it must produce one full tower.
  template <class ResidualProvider, class JvpProvider, class GaugeProvider,
            class RightPreconditioner>
  SolveReport solve_preconditioned(std::span<field_type* const> destination,
                                  ResidualProvider&& evaluate_residual, JvpProvider&& apply_jvp,
                                  GaugeProvider&& apply_gauge, RightPreconditioner&& apply_right,
                                  const ExecutionLane& lane, bool collective_local_phases = false) {
    return solve_rebuilt_preconditioned(destination, std::forward<ResidualProvider>(evaluate_residual),
        std::forward<JvpProvider>(apply_jvp), std::forward<GaugeProvider>(apply_gauge),
        std::forward<RightPreconditioner>(apply_right), [](const auto&, auto&, int) {},
        lane, collective_local_phases);
  }

  /// Rebuild once at each nonlinear iterate, then keep the right factor stationary
  /// throughout that GMRES solve. Preparation owns its collective protocol.
  template <class ResidualProvider, class JvpProvider, class GaugeProvider,
            class RightPreconditioner, class PrepareRight>
  SolveReport solve_rebuilt_preconditioned(std::span<field_type* const> destination,
      ResidualProvider&& evaluate_residual, JvpProvider&& apply_jvp,
      GaugeProvider&& apply_gauge, RightPreconditioner&& apply_right,
      PrepareRight&& prepare_right, const ExecutionLane& lane,
      bool collective_local_phases = false) {
    const auto* previous_lane = local_lane_;
    local_lane_ = collective_local_phases ? &lane : nullptr;
    struct ResetLane {
      const ExecutionLane*& target;
      const ExecutionLane* previous;
      ~ResetLane() { target = previous; }
    } reset{local_lane_, previous_lane};
    local_phase_([&] { authenticate_(destination, "destination"); });
    copy_from_external_(destination, iterate_);
    auto&& residual_provider = evaluate_residual;
    auto&& jvp_provider = apply_jvp;
    auto&& gauge_provider = apply_gauge;
    gauge_provider(iterate_);
    residual_provider(iterate_, residual_, 0);
    project_unknowns_(residual_);
    local_phase_([] { Kokkos::fence(); });

    SolveReport report;
    report.evaluations = 1;
    const Real initial_norm = norm_(residual_, lane);
    report.reference_residual_norm = initial_norm;
    report.residual_norm = initial_norm;
    const Real relative_denominator = initial_norm > Real(0) ? initial_norm : Real(1);
    report.rel_residual = initial_norm / relative_denominator;
    if (!finite_(initial_norm)) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                         "amr_field_newton_non_finite_initial_residual");
      return report;
    }
    Real nonlinear_stop = Real(0);
    if (options_.convergence.kind == FieldNewtonConvergenceKind::kLegacy) {
      nonlinear_stop = field_newton_stop_tolerance(options_, report.reference_residual_norm);
    } else {
      std::exception_ptr cutoff_error;
      try { nonlinear_stop = field_newton_stop_tolerance(options_, report.reference_residual_norm); }
      catch (...) { cutoff_error = std::current_exception(); }
      // This vote is mandatory even when optional local-algebra guards are disabled.
      // All ranks reject before initial admission or the next Krylov collective.
      collectively_rethrow_exception(cutoff_error, lane, "typed Original Newton computed cutoff");
    }
    if (initial_norm <= nonlinear_stop) {
      report.rel_residual = Real(0);
      copy_to_external_(iterate_, destination);
      report.mark_solved("amr_field_newton_initial_residual");
      return report;
    }

    for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
      prepare_right(iterate_, jvp_provider, iteration);
      set_zero_(correction_);
      const Real linear_stop = options_.linear_tolerance * report.residual_norm;
      const LinearResult linear =
          solve_linear_(iterate_, residual_, linear_stop, jvp_provider, iteration, lane, apply_right);
      report.evaluations += linear.evaluations;
      if (!linear.converged) {
        report.iters = iteration;
        std::ostringstream diagnostic;
        diagnostic << "amr_field_newton_gmres_breakdown:" << linear_failure_name_(linear.failure)
                   << ":newton=" << iteration << ":columns=" << linear.columns
                   << ":evaluations=" << linear.evaluations << std::hexfloat
                   << ":stop=" << linear_stop << ":beta=" << linear.residual_norm
                   << ":projected=" << linear.projected_norm << ":pivot=" << linear.pivot;
        report.mark_failed(SolveStatus::kBreakdown, SolveAction::kRejectAttempt,
                           diagnostic.str());
        return report;
      }

      gauge_provider(correction_);
      const Real full_step_norm = norm_(correction_, lane);
      if (!finite_(full_step_norm)) {
        report.iters = iteration;
        report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kRejectAttempt,
                           "amr_field_newton_non_finite_correction");
        return report;
      }

      bool accepted = false;
      Real step = Real(1);
      Real accepted_step = Real(0);
      Real accepted_norm = std::numeric_limits<Real>::infinity();
      while (step >= options_.minimum_step) {
        lincomb_(trial_, Real(1), iterate_, step, correction_);
        gauge_provider(trial_);
        residual_provider(trial_, trial_residual_, iteration + 1);
        project_unknowns_(trial_residual_);
        local_phase_([] { Kokkos::fence(); });
        ++report.evaluations;
        const Real trial_norm = norm_(trial_residual_, lane);
        if (finite_(trial_norm) &&
            trial_norm <= (Real(1) - options_.armijo * step) * report.residual_norm) {
          copy_(trial_, iterate_);
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
                           "amr_field_newton_line_search_failed");
        return report;
      }

      report.residual_norm = accepted_norm;
      report.rel_residual = accepted_norm / relative_denominator;
      if (accepted_norm <= nonlinear_stop) {
        copy_to_external_(iterate_, destination);
        report.mark_solved("amr_field_newton_converged");
        return report;
      }
    }

    report.mark_failed(SolveStatus::kIterationLimit, SolveAction::kRejectAttempt,
                       "amr_field_newton_iteration_limit");
    return report;
  }

 private:
  enum class LinearFailure {
    kIterationLimit, kNonfiniteInitialNorm, kZeroArnoldiColumn, kNonfiniteArnoldiColumn,
    kZeroTriangularPivot, kNonfiniteTriangularPivot, kNonfiniteTriangularCoefficient,
    kNonfiniteRecomputedNorm
  };
  static const char* linear_failure_name_(LinearFailure failure) noexcept {
    switch (failure) {
      case LinearFailure::kIterationLimit: return "iteration_limit";
      case LinearFailure::kNonfiniteInitialNorm: return "nonfinite_initial_norm";
      case LinearFailure::kZeroArnoldiColumn: return "zero_arnoldi_column";
      case LinearFailure::kNonfiniteArnoldiColumn: return "nonfinite_arnoldi_column";
      case LinearFailure::kZeroTriangularPivot: return "zero_triangular_pivot";
      case LinearFailure::kNonfiniteTriangularPivot: return "nonfinite_triangular_pivot";
      case LinearFailure::kNonfiniteTriangularCoefficient: return "nonfinite_triangular_coefficient";
      case LinearFailure::kNonfiniteRecomputedNorm: return "nonfinite_recomputed_norm";
    }
    return "invalid_failure";
  }
  struct LinearResult {
    bool converged = false;
    int evaluations = 0;
    int columns = 0;
    Real residual_norm = 0;
    Real projected_norm = 0;
    Real pivot = 0;
    LinearFailure failure = LinearFailure::kIterationLimit;
  };

  template <class JvpProvider, class RightPreconditioner>
  LinearResult solve_linear_(const hierarchy_type& iterate, const hierarchy_type& rhs, Real stop,
                             JvpProvider& apply_jvp, int nonlinear_iteration,
                             const ExecutionLane& lane, RightPreconditioner& apply_right) {
    copy_(rhs, linear_residual_);
    Real beta = norm_(linear_residual_, lane);
    LinearResult result;
    result.residual_norm = beta;
    if (!finite_(beta)) {
      result.failure = LinearFailure::kNonfiniteInitialNorm;
      return result;
    }
    if (beta <= stop) {
      result.converged = true;
      return result;
    }

    int completed = 0;
    while (completed < options_.linear_max_iterations) {
      const int cycle = std::min(options_.restart, options_.linear_max_iterations - completed);
      copy_(linear_residual_, basis_[0]);
      scale_(basis_[0], Real(1) / beta);
      std::fill(hessenberg_.begin(), hessenberg_.end(), Real(0));
      std::fill(cosine_.begin(), cosine_.end(), Real(0));
      std::fill(sine_.begin(), sine_.end(), Real(0));
      std::fill(rotated_rhs_.begin(), rotated_rhs_.end(), Real(0));
      rotated_rhs_[0] = beta;

      int used = 0;
      for (int column = 0; column < cycle; ++column) {
        apply_right(basis_[static_cast<std::size_t>(column)], image_);
        project_unknowns_(image_);
        apply_jvp(iterate, image_, work_, nonlinear_iteration);
        project_unknowns_(work_);
        local_phase_([] { Kokkos::fence(); });
        ++result.evaluations;
        for (int row = 0; row <= column; ++row) {
          h_(row, column) = dot_(work_, basis_[static_cast<std::size_t>(row)], lane);
          saxpy_(work_, -h_(row, column), basis_[static_cast<std::size_t>(row)]);
        }
        // A second MGS pass restores orthogonality after cancellation without
        // changing the Krylov column, restart or iteration budgets.
        for (int row = 0; row <= column; ++row) {
          const Real correction = dot_(work_, basis_[static_cast<std::size_t>(row)], lane);
          h_(row, column) += correction;
          saxpy_(work_, -correction, basis_[static_cast<std::size_t>(row)]);
        }
        h_(column + 1, column) = norm_(work_, lane);
        if (h_(column + 1, column) > Real(0)) {
          copy_(work_, basis_[static_cast<std::size_t>(column + 1)]);
          scale_(basis_[static_cast<std::size_t>(column + 1)], Real(1) / h_(column + 1, column));
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
        result.pivot = magnitude;
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
        result.columns = completed;
        result.projected_norm = std::abs(rotated_rhs_[static_cast<std::size_t>(column + 1)]);
        if (std::abs(rotated_rhs_[static_cast<std::size_t>(column + 1)]) <= stop) {
          break;
        }
      }
      if (used == 0) {
        result.failure = finite_(result.pivot) ? LinearFailure::kZeroArnoldiColumn
                                             : LinearFailure::kNonfiniteArnoldiColumn;
        return result;
      }
      if (!update_correction_(used, result, apply_right))
        return result;
      // A small rotated RHS only ends Arnoldi. The requested linear tolerance
      // is authenticated by the actual JVP on the complete correction, also
      // for a happy breakdown or a direction-dependent approximate JVP.
      apply_jvp(iterate, correction_, image_, nonlinear_iteration);
      project_unknowns_(image_);
      local_phase_([] { Kokkos::fence(); });
      ++result.evaluations;
      lincomb_(linear_residual_, Real(1), rhs, Real(-1), image_);
      beta = norm_(linear_residual_, lane);
      result.residual_norm = beta;
      if (!finite_(beta)) {
        result.failure = LinearFailure::kNonfiniteRecomputedNorm;
        return result;
      }
      if (beta <= stop) {
        result.converged = true;
        return result;
      }
    }
    return result;
  }

  template <class RightPreconditioner>
  bool update_correction_(int used, LinearResult& result, RightPreconditioner& apply_right) {
    for (int reverse = used; reverse != 0; --reverse) {
      const int row = reverse - 1;
      Real value = rotated_rhs_[static_cast<std::size_t>(row)];
      for (int column = row + 1; column < used; ++column)
        value -= h_(row, column) * coefficients_[static_cast<std::size_t>(column)];
      const Real diagonal = h_(row, row);
      if (!finite_(diagonal) || diagonal == Real(0)) {
        result.pivot = diagonal;
        result.failure = finite_(diagonal) ? LinearFailure::kZeroTriangularPivot
                                         : LinearFailure::kNonfiniteTriangularPivot;
        return false;
      }
      coefficients_[static_cast<std::size_t>(row)] = value / diagonal;
      if (!finite_(coefficients_[static_cast<std::size_t>(row)])) {
        result.pivot = diagonal;
        result.failure = LinearFailure::kNonfiniteTriangularCoefficient;
        return false;
      }
    }
    for (int index = 0; index < used; ++index) {
      apply_right(basis_[static_cast<std::size_t>(index)], image_);
      project_unknowns_(image_);
      saxpy_(correction_, coefficients_[static_cast<std::size_t>(index)], image_);
    }
    return true;
  }

  // Covered/EB-inactive cells remain available to the operator but are not Krylov DOFs.
  // Project both defects and JVP images so corrections cannot evolve those stored values.
  void project_unknowns_(hierarchy_type& fields) const {
    local_phase_([&] {
      authenticate_owned_(fields, "projection");
      for (std::size_t level = 0; level < fields.size(); ++level)
        for (std::size_t local = 0; local < fields[level].local_size(); ++local) {
          const auto values = fields[level].fab(local).view();
          const auto active = std::as_const(*active_cells_[level]).fab(local).view();
          const int components = fields[level].ncomp();
          for_each_cell(fields[level].box(local), [=] POPS_HD(const Index<Dim>& cell) {
            if (!(active(cell, 0) >= Real(0.5)))
              for (int component = 0; component < components; ++component)
                values(cell, component) = Real(0);
          });
        }
    });
  }

  static hierarchy_type make_hierarchy_(std::span<const field_type* const> layouts) {
    hierarchy_type result;
    result.reserve(layouts.size());
    for (const field_type* layout : layouts)
      result.emplace_back(layout->layout(), layout->distribution(), layout->local_rank(),
                          layout->ncomp(), Extent<Dim>{});
    return result;
  }

  static bool same_layout_(const field_type& left, const field_type& right) noexcept {
    return left.layout() == right.layout() && left.distribution() == right.distribution() &&
           left.local_rank() == right.local_rank();
  }

  void authenticate_(std::span<field_type* const> fields, const char* role) const {
    if (fields.size() != iterate_.size())
      throw std::invalid_argument(std::string("AMR field Newton ") + role +
                                  " has the wrong level count");
    for (std::size_t level = 0; level < fields.size(); ++level)
      if (fields[level] == nullptr || fields[level]->ncomp() != iterate_[level].ncomp() ||
          !same_layout_(*fields[level], iterate_[level]))
        throw std::invalid_argument(std::string("AMR field Newton ") + role +
                                    " differs from its prepared exact-ranked hierarchy");
  }

  void authenticate_owned_(const hierarchy_type& fields, const char* role) const {
    if (fields.size() != iterate_.size())
      throw std::invalid_argument(std::string("AMR field Newton ") + role +
                                  " has the wrong level count");
    for (std::size_t level = 0; level < fields.size(); ++level)
      if (fields[level].ncomp() != iterate_[level].ncomp() ||
          !same_layout_(fields[level], iterate_[level]))
        throw std::invalid_argument(std::string("AMR field Newton ") + role +
                                    " differs from its prepared exact-ranked hierarchy");
  }

  void copy_field_(const field_type& source, field_type& destination) {
    local_phase_([&] {
      if (!same_layout_(source, destination) || source.ncomp() != destination.ncomp())
        throw std::invalid_argument("AMR field Newton vector layouts differ");
      lincomb(destination, Real(1), source, Real(0), source);
    });
  }

  void copy_(const hierarchy_type& source, hierarchy_type& destination) {
    local_phase_([&] {
      authenticate_owned_(source, "copy source");
      authenticate_owned_(destination, "copy destination");
    });
    for (std::size_t level = 0; level < source.size(); ++level)
      copy_field_(source[level], destination[level]);
  }

  void copy_from_external_(std::span<field_type* const> source, hierarchy_type& destination) {
    local_phase_([&] {
      authenticate_(source, "copy source");
      authenticate_owned_(destination, "copy destination");
    });
    for (std::size_t level = 0; level < source.size(); ++level)
      copy_field_(*source[level], destination[level]);
  }

  void copy_to_external_(const hierarchy_type& source, std::span<field_type* const> destination) {
    local_phase_([&] {
      authenticate_owned_(source, "copy source");
      authenticate_(destination, "copy destination");
    });
    for (std::size_t level = 0; level < source.size(); ++level)
      copy_field_(source[level], *destination[level]);
    local_phase_([] { Kokkos::fence(); });
  }

  void set_zero_(hierarchy_type& fields) {
    local_phase_([&] {
      for (field_type& field : fields)
        field.set_val(Real(0));
    });
  }

  void scale_(hierarchy_type& fields, Real factor) {
    local_phase_([&] {
      for (field_type& field : fields)
        scale(field, factor);
    });
  }

  void saxpy_(hierarchy_type& destination, Real factor, const hierarchy_type& source) {
    local_phase_([&] {
      if (destination.size() != source.size())
        throw std::invalid_argument("AMR field Newton hierarchy sizes differ");
      for (std::size_t level = 0; level < destination.size(); ++level)
        saxpy(destination[level], factor, source[level]);
    });
  }

  void lincomb_(hierarchy_type& destination, Real left_factor, const hierarchy_type& left,
                Real right_factor, const hierarchy_type& right) {
    local_phase_([&] {
      if (destination.size() != left.size() || destination.size() != right.size())
        throw std::invalid_argument("AMR field Newton hierarchy sizes differ");
      for (std::size_t level = 0; level < destination.size(); ++level)
        lincomb(destination[level], left_factor, left[level], right_factor, right[level]);
    });
  }

  Real dot_(const hierarchy_type& left, const hierarchy_type& right,
            const ExecutionLane& lane) const {
    Real local_result = Real(0);
    local_phase_([&] {
      if (left.size() != active_cells_.size() || right.size() != active_cells_.size())
        throw std::invalid_argument("AMR field Newton dot hierarchy size differs");
      authenticate_owned_(left, "dot left");
      authenticate_owned_(right, "dot right");
      for (std::size_t level = 0; level < left.size(); ++level) {
        // A replicated level has one physical owner. Counting every replica weights
        // mixed replicated/distributed hierarchies differently across MPI sizes.
        if (left[level].distribution().replicated() &&
            left[level].local_rank() != left[level].rank_space().coordinate(0))
          continue;
        for (int component = 0; component < left[level].ncomp(); ++component)
          local_result += cell_measures_[level] * dot_active_local(left[level], right[level],
                                                                   component, active_cells_[level]);
      }
    });
    return static_cast<Real>(all_reduce_sum(local_result, lane));
  }

  Real norm_(const hierarchy_type& fields, const ExecutionLane& lane) const {
    const Real squared = dot_(fields, fields, lane);
    if (!finite_(squared))
      return squared;
    if (squared < Real(0))
      return std::numeric_limits<Real>::quiet_NaN();
    return squared > Real(0) ? std::sqrt(squared) : Real(0);
  }

  Real& h_(int row, int column) {
    return hessenberg_[static_cast<std::size_t>(column) *
                           static_cast<std::size_t>(options_.restart + 1) +
                       static_cast<std::size_t>(row)];
  }

  static bool finite_(Real value) noexcept { return std::isfinite(static_cast<double>(value)); }

  template <class Operation>
  void local_phase_(Operation&& operation) const {
    if (!local_lane_) {
      operation();
      return;
    }
    std::exception_ptr error;
    try {
      operation();
      Kokkos::fence();
    } catch (...) {
      error = std::current_exception();
    }
    collectively_rethrow_exception(error, *local_lane_, "AMR field Newton local algebra");
  }

  FieldNewtonOptions options_;
  const ExecutionLane* local_lane_ = nullptr;
  std::vector<const field_type*> active_cells_;
  std::vector<Real> cell_measures_;
  hierarchy_type iterate_;
  hierarchy_type residual_;
  hierarchy_type trial_;
  hierarchy_type trial_residual_;
  hierarchy_type correction_;
  hierarchy_type linear_residual_;
  hierarchy_type image_;
  hierarchy_type work_;
  std::vector<hierarchy_type> basis_;
  std::vector<Real> hessenberg_;
  std::vector<Real> cosine_;
  std::vector<Real> sine_;
  std::vector<Real> rotated_rhs_;
  std::vector<Real> coefficients_;
};

}  // namespace pops
