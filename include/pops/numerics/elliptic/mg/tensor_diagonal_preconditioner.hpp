/// @file
/// @brief Prepared one-V-cycle preconditioner for the diagonal of an exact tensor field.
#pragma once

#include <pops/numerics/elliptic/linear/prepared_affine_problem.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

namespace pops::elliptic::mg {

namespace tensor_diagonal_detail {

template <int Dim>
struct ExtractDiagonal {
  FieldView<const Real, Dim> tensor;
  FieldView<Real, Dim> diagonal;
  POPS_HD void operator()(const Index<Dim>& cell) const {
    for (int axis = 0; axis < Dim; ++axis)
      diagonal(cell, axis) = tensor(cell, axis * Dim + axis);
  }
};

/// Every factory receives private mutable storage. Coefficients are refreshed only at the
/// prepared-operator snapshot boundary; a Krylov application never allocates or rebuilds a solver.
template <int Dim>
class Session {
 public:
  using field_type = MultiFab<Dim>;
  Session(std::shared_ptr<const field_type> prototype,
          std::shared_ptr<const field_type> coefficients, Geometry<Dim> geometry,
          PhysicalBoundaryConditions<Dim> boundary, GeometricMultigridOptions options,
          PreparedVectorDistribution<Dim> distribution, const ExecutionLane& lane)
      : prototype_(std::move(prototype)), coefficients_(std::move(coefficients)),
        geometry_(std::move(geometry)), boundary_(std::move(boundary)), options_(options),
        distribution_(std::move(distribution)), lane_(&lane) {}

  void prepare() {
    const bool invalid = !prototype_ || !coefficients_ ||
        (prototype_ && prototype_->ncomp() != 1) ||
        (coefficients_ && coefficients_->ncomp() != Dim * Dim) ||
        (prototype_ && coefficients_ &&
         (prototype_->layout() != coefficients_->layout() ||
          prototype_->distribution() != coefficients_->distribution() ||
          prototype_->local_rank() != coefficients_->local_rank()));
    if (all_reduce_max(invalid ? 1L : 0L, *lane_) != 0)
      throw std::invalid_argument("tensor diagonal MG requires its exact scalar/tensor layout");

    if (!diagonal_) {
      long allocation_failure = 0;
      try {
        diagonal_ = std::make_unique<field_type>(prototype_->layout(), prototype_->distribution(),
            prototype_->local_rank(), Dim, detail::unit_ghosts<Dim>());
      } catch (...) {
        allocation_failure = 1;
      }
      if (all_reduce_max(allocation_failure, *lane_) != 0) {
        diagonal_.reset();
        throw std::runtime_error("tensor diagonal MG storage preparation failed collectively");
      }
    }
    for (int local = 0; local < diagonal_->local_size(); ++local)
      for_each_cell(diagonal_->box(local), ExtractDiagonal<Dim>{
          coefficients_->fab(local).view(), diagonal_->fab(local).view()});

    if (!solver_) {
      solver_ = std::make_unique<GeometricMG<Dim>>(geometry_, prototype_->layout(),
          prototype_->distribution(), prototype_->local_rank(), boundary_, *lane_, options_);
      long nullspace_failure = 0;
      try {
        solver_->install_nullspace(FieldNullspacePlan<Dim>{}, distribution_);
      } catch (...) {
        nullspace_failure = 1;
      }
      if (all_reduce_max(nullspace_failure, *lane_) != 0) {
        solver_.reset();
        throw std::runtime_error("tensor diagonal MG nullspace preparation failed collectively");
      }
      solver_->install_diagonal_coefficient(*diagonal_);
    } else {
      solver_->refresh_diagonal_coefficient(*diagonal_);
    }
  }

  void apply(field_type& out, const field_type& in) {
    if (!solver_)
      throw std::logic_error("tensor diagonal MG must be prepared before application");
    solver_->apply_one_cycle(in, out);
  }

  std::size_t allocation_count() const noexcept {
    // Field OBJECTS, including zero-owner objects, must agree across the prepared MPI lane.
    // Communication metadata and rank-dependent FAB bytes are deliberately not called fields.
    return (diagonal_ ? 1u : 0u) + (solver_ ? 7u * static_cast<std::size_t>(solver_->num_levels()) : 0u);
  }

 private:
  std::shared_ptr<const field_type> prototype_, coefficients_;
  Geometry<Dim> geometry_;
  PhysicalBoundaryConditions<Dim> boundary_;
  GeometricMultigridOptions options_;
  PreparedVectorDistribution<Dim> distribution_;
  const ExecutionLane* lane_;
  std::unique_ptr<field_type> diagonal_;
  std::unique_ptr<GeometricMG<Dim>> solver_;
};

}  // namespace tensor_diagonal_detail

template <int Dim>
PreparedLinearPreconditionerSessionFactory<Dim> make_tensor_diagonal_mg_sessions(
    std::shared_ptr<const MultiFab<Dim>> prototype,
    std::shared_ptr<const MultiFab<Dim>> coefficients, Geometry<Dim> geometry,
    PhysicalBoundaryConditions<Dim> boundary, PreparedVectorDistribution<Dim> distribution,
    int minimum_coarse_extent = 2, int pre_sweeps = 2, int post_sweeps = 2,
    int bottom_sweeps = 50) {
  GeometricMultigridOptions options;
  options.minimum_coarse_extent = minimum_coarse_extent;
  options.pre_sweeps = pre_sweeps;
  options.post_sweeps = post_sweeps;
  options.bottom_sweeps = bottom_sweeps;
  options.coarse_cell_threshold = 0;
  options.smoother = GeometricSmoother::red_black_gauss_seidel;
  options.prolongation = GeometricProlongation::constant_injection;
  options.pre_smooth_bottom = true;
  options.field_domain.reset();
  return [prototype = std::move(prototype), coefficients = std::move(coefficients),
          geometry = std::move(geometry), boundary = std::move(boundary),
          distribution = std::move(distribution), options](const ExecutionLane& lane) {
    auto session = std::make_shared<tensor_diagonal_detail::Session<Dim>>(
        prototype, coefficients, geometry, boundary, options, distribution, lane);
    return PreparedLinearPreconditionerSessionCallbacks<Dim>{
        [session]() { session->prepare(); },
        [session](MultiFab<Dim>& out, const MultiFab<Dim>& in) { session->apply(out, in); },
        [session]() { return session->allocation_count(); }};
  };
}

}  // namespace pops::elliptic::mg
