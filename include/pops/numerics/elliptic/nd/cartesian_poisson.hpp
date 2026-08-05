/// @file
/// @brief Prepared compile-time-ranked Cartesian Poisson solver.

#pragma once

#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace pops::elliptic::nd {

enum class CartesianBoundaryKind : unsigned char { periodic, dirichlet, neumann };

template <int Dim>
struct CartesianPoissonOptions {
  static_assert(Dim >= 1 && Dim <= 3, "CartesianPoissonOptions supports dimensions 1, 2, and 3");

  std::array<CartesianBoundaryKind, 2 * Dim> boundaries{};
  std::array<Real, 2 * Dim> boundary_values{};
  Real relative_tolerance = Real{1e-10};
  Real absolute_tolerance = Real{0};
  int maximum_iterations = 2000;

  static CartesianPoissonOptions from_topology(
      const BoundaryTopology<Dim>& topology,
      CartesianBoundaryKind physical_kind = CartesianBoundaryKind::dirichlet) {
    if (physical_kind == CartesianBoundaryKind::periodic)
      throw std::invalid_argument(
          "Cartesian Poisson physical faces cannot be declared periodic implicitly");
    CartesianPoissonOptions result;
    for (int axis = 0; axis < Dim; ++axis) {
      for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
        const Face<Dim> face{axis, side};
        result.boundaries[static_cast<std::size_t>(face.ordinal())] =
            topology.is_periodic(face) ? CartesianBoundaryKind::periodic : physical_kind;
      }
    }
    return result;
  }
};

namespace detail {

inline std::size_t checked_multiply(std::size_t left, std::size_t right, const char* operation) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(operation);
  return left * right;
}

template <int Dim>
HaloScheduleBudget exact_halo_budget(const mesh::BoxArray<Dim>& layout, const Box<Dim>& domain,
                                     int components) {
  if (components < 1)
    throw std::invalid_argument("Cartesian Poisson halo component count must be positive");
  const std::size_t boxes = layout.size();
  const std::size_t pairs =
      checked_multiply(boxes, boxes, "Cartesian Poisson halo pair budget overflow");
  std::size_t images = 1;
  for (int axis = 0; axis < Dim; ++axis)
    images = checked_multiply(images, 3, "Cartesian Poisson image budget overflow");
  const std::size_t work =
      checked_multiply(pairs, images, "Cartesian Poisson halo work budget overflow");
  const std::size_t jobs = checked_multiply(work, static_cast<std::size_t>(2 * Dim),
                                            "Cartesian Poisson halo job budget overflow");
  const std::int64_t signed_cells = domain.numPts();
  if (signed_cells <= 0)
    throw std::invalid_argument("Cartesian Poisson halo domain must be non-empty");
  const std::size_t cells = static_cast<std::size_t>(signed_cells);
  const std::size_t elements = checked_multiply(
      checked_multiply(jobs, cells, "Cartesian Poisson halo element budget overflow"),
      static_cast<std::size_t>(components), "Cartesian Poisson halo component budget overflow");
  return HaloScheduleBudget{
      mesh::BoxArrayValidationBudget{boxes, pairs},
      work,
      jobs,
      images,
      checked_multiply(boxes, std::size_t{2}, "Cartesian Poisson peer budget overflow"),
      elements,
      elements,
      elements};
}

template <int Dim>
struct CopyComponentKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> source{};
  int destination_component = 0;
  int source_component = 0;

  POPS_HD void operator()(const Index<Dim>& index) const {
    destination(index, destination_component) = source(index, source_component);
  }
};

template <int Dim>
struct AddConstantKernel {
  FieldView<Real, Dim> values{};
  Real increment = 0;

  POPS_HD void operator()(const Index<Dim>& index) const { values(index, 0) += increment; }
};

template <int Dim>
struct PhysicalGhostKernel {
  FieldView<Real, Dim> values{};
  Box<Dim> domain{};
  std::array<CartesianBoundaryKind, 2 * Dim> boundaries{};
  std::array<Real, 2 * Dim> boundary_values{};
  Real spacing[Dim]{};

  POPS_HD void operator()(const Index<Dim>& destination) const {
    Index<Dim> source = destination;
    bool physical = false;
    for (int axis = 0; axis < Dim; ++axis) {
      if (destination[axis] < domain.lo[axis]) {
        const int face = 2 * axis;
        if (boundaries[static_cast<std::size_t>(face)] == CartesianBoundaryKind::periodic)
          continue;
        source[axis] = domain.lo[axis];
        physical = true;
      } else if (destination[axis] > domain.hi[axis]) {
        const int face = 2 * axis + 1;
        if (boundaries[static_cast<std::size_t>(face)] == CartesianBoundaryKind::periodic)
          continue;
        source[axis] = domain.hi[axis];
        physical = true;
      }
    }
    if (!physical)
      return;

    Real result = values(source, 0);
    for (int axis = 0; axis < Dim; ++axis) {
      int face = -1;
      if (destination[axis] < domain.lo[axis])
        face = 2 * axis;
      else if (destination[axis] > domain.hi[axis])
        face = 2 * axis + 1;
      if (face < 0)
        continue;
      const CartesianBoundaryKind kind = boundaries[static_cast<std::size_t>(face)];
      if (kind == CartesianBoundaryKind::dirichlet)
        result = Real(2) * boundary_values[static_cast<std::size_t>(face)] - result;
      else if (kind == CartesianBoundaryKind::neumann)
        result += boundary_values[static_cast<std::size_t>(face)] * spacing[axis];
    }
    values(destination, 0) = result;
  }
};

template <int Dim>
struct NegativeLaplacianKernel {
  FieldView<Real, Dim> output{};
  FieldView<const Real, Dim> input{};
  Real inverse_spacing_squared[Dim]{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    Real value = 0;
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> lower = index;
      Index<Dim> upper = index;
      --lower[axis];
      ++upper[axis];
      value += (Real(2) * input(index, 0) - input(lower, 0) - input(upper, 0)) *
               inverse_spacing_squared[axis];
    }
    output(index, 0) = value;
  }
};

template <int Dim>
void copy_component(const MultiFab<Dim>& source, int source_component, MultiFab<Dim>& destination,
                    int destination_component) {
  if (source.layout() != destination.layout() ||
      source.distribution() != destination.distribution() ||
      source.local_rank() != destination.local_rank())
    throw std::invalid_argument("Cartesian Poisson component copy requires one exact layout");
  if (source_component < 0 || source_component >= source.ncomp() || destination_component < 0 ||
      destination_component >= destination.ncomp())
    throw std::out_of_range("Cartesian Poisson component copy index is outside its field");
  for (std::size_t local = 0; local < source.local_size(); ++local)
    for_each_cell(source.box(local),
                  CopyComponentKernel<Dim>{destination.fab(local).view(), source.fab(local).view(),
                                           destination_component, source_component});
}

}  // namespace detail

/// Prepared constant-coefficient cell-centred solve of ``-laplacian(phi) = rhs``.
///
/// The implementation owns every Krylov vector and one authenticated halo schedule. It is one
/// compile-time-ranked algorithm: axes are visited by a bounded loop and no dimensional fallback
/// exists. Remote schedules are rejected before the candidate iterate is touched until the exact ND
/// transport component is present.
template <int Dim>
class CartesianPoissonSolver {
 public:
  using field_type = MultiFab<Dim>;

  CartesianPoissonSolver(const Geometry<Dim>& geometry, const mesh::BoxArray<Dim>& layout,
                         const mesh::Distribution<Dim>& distribution, Index<Dim> local_rank,
                         BoundaryTopology<Dim> topology, CartesianPoissonOptions<Dim> options)
      : geometry_(geometry),
        topology_(topology),
        options_(options),
        candidate_(layout, distribution, local_rank, 1, unit_ghosts_()),
        halo_(layout, distribution, local_rank, 1, unit_ghosts_()),
        rhs_(layout, distribution, local_rank, 1, Extent<Dim>{}),
        residual_(layout, distribution, local_rank, 1, Extent<Dim>{}),
        direction_(layout, distribution, local_rank, 1, Extent<Dim>{}),
        image_(layout, distribution, local_rank, 1, Extent<Dim>{}),
        schedule_(prepare_halo_schedule(halo_, geometry.domain(), topology,
                                        detail::exact_halo_budget(layout, geometry.domain(), 1))) {
    std::exception_ptr validation_error;
    try {
      validate_options_();
    } catch (...) {
      validation_error = std::current_exception();
    }
    if (all_reduce_max(validation_error ? 1L : 0L) != 0) {
      if (n_ranks() == 1 && validation_error)
        std::rethrow_exception(validation_error);
      throw std::runtime_error("Cartesian Poisson preparation failed collectively");
    }
    const bool distributed_halo = all_reduce_max(schedule_.has_remote_jobs() ? 1L : 0L) != 0;
    if (distributed_halo) {
      lane_ = std::make_unique<ExecutionLane>(ExecutionLane::duplicate_world_collectively(
          "pops.cartesian-poisson.nd" + std::to_string(Dim) + "/halo"));
      HaloExchangeContext context{};
      context.context_generation = 1;
      context.schedule_generation = 1;
      exchange_ = std::make_unique<HaloExchange<Dim>>(schedule_, *lane_, context);
    }
  }

  CartesianPoissonSolver(const CartesianPoissonSolver&) = delete;
  CartesianPoissonSolver& operator=(const CartesianPoissonSolver&) = delete;
  CartesianPoissonSolver(CartesianPoissonSolver&&) = default;
  CartesianPoissonSolver& operator=(CartesianPoissonSolver&&) = default;

  field_type& rhs() noexcept { return rhs_; }
  const field_type& rhs() const noexcept { return rhs_; }
  field_type& candidate() noexcept { return candidate_; }
  const field_type& candidate() const noexcept { return candidate_; }
  int maximum_iterations() const noexcept { return options_.maximum_iterations; }

  SolveReport solve(const field_type& warm_start) {
    std::exception_ptr layout_error;
    try {
      authenticate_(warm_start, "warm start");
    } catch (...) {
      layout_error = std::current_exception();
    }
    if (all_reduce_max(layout_error ? 1L : 0L) != 0) {
      if (n_ranks() == 1 && layout_error)
        std::rethrow_exception(layout_error);
      throw std::runtime_error("Cartesian Poisson solve layout validation failed collectively");
    }

    detail::copy_component(warm_start, 0, candidate_, 0);
    if (singular_()) {
      project_mean_(rhs_);
      project_mean_(candidate_);
    }

    apply_(candidate_, image_);
    lincomb(residual_, Real(1), rhs_, Real(-1), image_);
    detail::copy_component(residual_, 0, direction_, 0);

    const Real reference_squared = dot(rhs_, rhs_);
    Real residual_squared = dot(residual_, residual_);
    SolveReport report;
    report.evaluations = 1;
    report.reference_residual_norm =
        reference_squared > Real(0) ? std::sqrt(reference_squared) : Real(0);
    report.residual_norm = residual_squared > Real(0) ? std::sqrt(residual_squared) : Real(0);
    const Real denominator =
        report.reference_residual_norm > Real(0) ? report.reference_residual_norm : Real(1);
    report.rel_residual = report.residual_norm / denominator;
    const Real stop = std::max(options_.absolute_tolerance,
                               options_.relative_tolerance * report.reference_residual_norm);
    if (!finite_(report.reference_residual_norm) || !finite_(report.residual_norm)) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                         "cartesian_poisson_non_finite_initial_residual");
      return report;
    }
    if (report.residual_norm <= stop) {
      fill_candidate_ghosts_();
      report.mark_solved("cartesian_poisson_initial_residual");
      return report;
    }

    Real previous = residual_squared;
    for (int iteration = 0; iteration < options_.maximum_iterations; ++iteration) {
      apply_(direction_, image_);
      ++report.evaluations;
      const Real curvature = dot(direction_, image_);
      if (!finite_(curvature) || !(curvature > Real(0))) {
        report.iters = iteration;
        report.mark_failed(SolveStatus::kBreakdown, SolveAction::kFailRun,
                           "cartesian_poisson_non_positive_curvature");
        return report;
      }
      const Real alpha = previous / curvature;
      report.step_norm =
          std::abs(alpha) * std::sqrt(std::max(dot(direction_, direction_), Real(0)));
      saxpy(candidate_, alpha, direction_);
      saxpy(residual_, -alpha, image_);
      if (singular_())
        project_mean_(candidate_);

      residual_squared = dot(residual_, residual_);
      report.iters = iteration + 1;
      report.residual_norm = residual_squared > Real(0) ? std::sqrt(residual_squared) : Real(0);
      report.rel_residual = report.residual_norm / denominator;
      if (!finite_(report.residual_norm) || !finite_(report.step_norm)) {
        report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                           "cartesian_poisson_non_finite_iteration");
        return report;
      }
      if (report.residual_norm <= stop) {
        fill_candidate_ghosts_();
        report.mark_solved("cartesian_poisson_converged");
        return report;
      }
      if (!(previous > Real(0))) {
        report.mark_failed(SolveStatus::kBreakdown, SolveAction::kFailRun,
                           "cartesian_poisson_zero_recurrence_norm");
        return report;
      }
      const Real beta = residual_squared / previous;
      lincomb(direction_, Real(1), residual_, beta, direction_);
      previous = residual_squared;
    }

    report.mark_failed(SolveStatus::kIterationLimit, SolveAction::kFailRun,
                       "cartesian_poisson_iteration_limit");
    return report;
  }

 private:
  static Extent<Dim> unit_ghosts_() {
    Extent<Dim> result{};
    for (int axis = 0; axis < Dim; ++axis)
      result[axis] = 1;
    return result;
  }

  void validate_options_() const {
    if (!finite_(options_.relative_tolerance) || options_.relative_tolerance < Real(0) ||
        !finite_(options_.absolute_tolerance) || options_.absolute_tolerance < Real(0) ||
        options_.maximum_iterations < 1)
      throw std::invalid_argument("Cartesian Poisson iteration controls are invalid");
    for (int axis = 0; axis < Dim; ++axis) {
      const Face<Dim> lower{axis, BoundarySide::lower};
      const Face<Dim> upper{axis, BoundarySide::upper};
      const auto low = options_.boundaries[static_cast<std::size_t>(lower.ordinal())];
      const auto high = options_.boundaries[static_cast<std::size_t>(upper.ordinal())];
      if ((low == CartesianBoundaryKind::periodic) != topology_.is_periodic(lower) ||
          (high == CartesianBoundaryKind::periodic) != topology_.is_periodic(upper))
        throw std::invalid_argument(
            "Cartesian Poisson boundary kinds differ from the exact topology");
      if (!finite_(options_.boundary_values[static_cast<std::size_t>(lower.ordinal())]) ||
          !finite_(options_.boundary_values[static_cast<std::size_t>(upper.ordinal())]))
        throw std::invalid_argument("Cartesian Poisson boundary values must be finite");
    }
  }

  bool singular_() const noexcept {
    for (const CartesianBoundaryKind boundary : options_.boundaries)
      if (boundary == CartesianBoundaryKind::dirichlet)
        return false;
    return true;
  }

  void authenticate_(const field_type& field, const char* role) const {
    if (field.layout() != rhs_.layout() || field.distribution() != rhs_.distribution() ||
        field.local_rank() != rhs_.local_rank() || field.ncomp() != 1)
      throw std::invalid_argument(std::string("Cartesian Poisson ") + role +
                                  " does not match its prepared scalar layout");
  }

  void project_mean_(field_type& field) const {
    const std::int64_t cell_count = geometry_.domain().numPts();
    if (cell_count <= 0)
      throw std::logic_error("Cartesian Poisson domain unexpectedly has no cells");
    const Real mean = reduce_sum(field) / static_cast<Real>(cell_count);
    for (std::size_t local = 0; local < field.local_size(); ++local)
      for_each_cell(field.box(local),
                    detail::AddConstantKernel<Dim>{field.fab(local).view(), -mean});
  }

  void fill_candidate_ghosts_() {
    detail::copy_component(candidate_, 0, halo_, 0);
    fill_halo_();
    for (std::size_t local = 0; local < candidate_.local_size(); ++local) {
      const auto& source = static_cast<const field_type&>(halo_).fab(local);
      for_each_cell(
          candidate_.fab(local).grown_box(),
          detail::CopyComponentKernel<Dim>{candidate_.fab(local).view(), source.view(), 0, 0});
    }
    Kokkos::fence();
  }

  void fill_halo_() {
    if (exchange_)
      exchange_->execute(halo_, *lane_);
    else
      fill_boundary(halo_, schedule_);
    Real spacing[Dim]{};
    for (int axis = 0; axis < Dim; ++axis)
      spacing[axis] = geometry_.spacing(axis);
    for (std::size_t local = 0; local < halo_.local_size(); ++local) {
      detail::PhysicalGhostKernel<Dim> kernel{halo_.fab(local).view(),
                                              geometry_.domain(),
                                              options_.boundaries,
                                              options_.boundary_values,
                                              {}};
      for (int axis = 0; axis < Dim; ++axis)
        kernel.spacing[axis] = spacing[axis];
      for_each_cell(halo_.fab(local).grown_box(), kernel);
    }
    Kokkos::fence();
  }

  void apply_(const field_type& input, field_type& output) {
    authenticate_(input, "operator input");
    authenticate_(output, "operator output");
    detail::copy_component(input, 0, halo_, 0);
    fill_halo_();
    Real inverse_spacing_squared[Dim]{};
    for (int axis = 0; axis < Dim; ++axis) {
      const Real inverse = Real(1) / geometry_.spacing(axis);
      inverse_spacing_squared[axis] = inverse * inverse;
    }
    for (std::size_t local = 0; local < output.local_size(); ++local) {
      const auto& source = static_cast<const field_type&>(halo_).fab(local);
      detail::NegativeLaplacianKernel<Dim> kernel{output.fab(local).view(), source.view(), {}};
      for (int axis = 0; axis < Dim; ++axis)
        kernel.inverse_spacing_squared[axis] = inverse_spacing_squared[axis];
      for_each_cell(output.box(local), kernel);
    }
    Kokkos::fence();
  }

  static bool finite_(Real value) noexcept { return std::isfinite(static_cast<double>(value)); }

  Geometry<Dim> geometry_;
  BoundaryTopology<Dim> topology_;
  CartesianPoissonOptions<Dim> options_;
  field_type candidate_;
  field_type halo_;
  field_type rhs_;
  field_type residual_;
  field_type direction_;
  field_type image_;
  HaloSchedule<Dim> schedule_;
  std::unique_ptr<ExecutionLane> lane_;
  std::unique_ptr<HaloExchange<Dim>> exchange_;
};

}  // namespace pops::elliptic::nd
