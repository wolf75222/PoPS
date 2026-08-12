/// @file
/// @brief Prepared compile-time-ranked Cartesian Poisson solver.

#pragma once

#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/numerics/elliptic/interface/field_newton_krylov.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/runtime/numerical_defaults.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace pops::elliptic::nd {

enum class CartesianBoundaryKind : unsigned char { periodic, dirichlet, neumann, mixed };

template <int Dim>
struct CartesianPoissonOptions {
  static_assert(Dim >= 1 && Dim <= 3, "CartesianPoissonOptions supports dimensions 1, 2, and 3");

  std::array<CartesianBoundaryKind, 2 * Dim> boundaries{};
  std::array<Real, 2 * Dim> boundary_alpha{};
  std::array<Real, 2 * Dim> boundary_beta{};
  std::array<Real, 2 * Dim> boundary_values{};
  Real relative_tolerance = kCartesianCGDefaultRelTol;
  Real absolute_tolerance = kCartesianCGDefaultAbsTol;
  int maximum_iterations = kCartesianCGDefaultMaxIterations;

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
        const std::size_t ordinal = static_cast<std::size_t>(face.ordinal());
        result.boundaries[ordinal] =
            topology.is_periodic(face) ? CartesianBoundaryKind::periodic : physical_kind;
        if (!topology.is_periodic(face)) {
          result.boundary_alpha[ordinal] =
              physical_kind == CartesianBoundaryKind::dirichlet ? Real(1) : Real(0);
          result.boundary_beta[ordinal] =
              physical_kind == CartesianBoundaryKind::neumann ? Real(1) : Real(0);
        }
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
struct PhysicalGhostKernel {
  FieldView<Real, Dim> values{};
  Box<Dim> domain{};
  std::array<CartesianBoundaryKind, 2 * Dim> boundaries{};
  std::array<Real, 2 * Dim> boundary_alpha{};
  std::array<Real, 2 * Dim> boundary_beta{};
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
      const std::size_t ordinal = static_cast<std::size_t>(face);
      if (boundaries[ordinal] == CartesianBoundaryKind::periodic)
        continue;
      const int layer = destination[axis] < domain.lo[axis] ? domain.lo[axis] - destination[axis]
                                                            : destination[axis] - domain.hi[axis];
      const Real distance = Real(2 * layer - 1) * spacing[axis];
      const Real alpha = boundary_alpha[ordinal];
      const Real beta = boundary_beta[ordinal];
      result = (boundary_values[ordinal] - result * (alpha / Real(2) - beta / distance)) /
               (alpha / Real(2) + beta / distance);
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
/// exists. Distributed schedules use the same exact-ranked halo transport and authenticated
/// communicator lane as the rest of the native mesh runtime.
template <int Dim>
class CartesianPoissonSolver {
 public:
  using field_type = MultiFab<Dim>;

  CartesianPoissonSolver(const Geometry<Dim>& geometry, const mesh::BoxArray<Dim>& layout,
                         const mesh::Distribution<Dim>& distribution, Index<Dim> local_rank,
                         BoundaryTopology<Dim> topology, CartesianPoissonOptions<Dim> options,
                         const ExecutionLane& lane)
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
                                        detail::exact_halo_budget(layout, geometry.domain(), 1))),
        lane_(&lane),
        lane_borrow_(lane.borrow_immutably()) {
    std::exception_ptr validation_error;
    try {
      validate_options_();
    } catch (...) {
      validation_error = std::current_exception();
    }
    if (all_reduce_max(validation_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && validation_error)
        std::rethrow_exception(validation_error);
      throw std::runtime_error("Cartesian Poisson preparation failed collectively");
    }
    const bool distributed_halo = all_reduce_max(schedule_.has_remote_jobs() ? 1L : 0L, lane) != 0;
    if (distributed_halo) {
      HaloExchangeContext context{};
      context.context_generation = 1;
      context.schedule_generation = 1;
      exchange_ = std::make_unique<HaloExchange<Dim>>(schedule_, lane, context);
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
  int maximum_iterations() const noexcept {
    return newton_workspace_ ? newton_workspace_->options().max_iterations
                             : options_.maximum_iterations;
  }

  void install_boundary_kernel(CompiledFieldBoundaryKernel<Dim> kernel) {
    kernel.validate();
    boundary_kernel_ = std::move(kernel);
    boundary_contexts_.reset();
  }

  /// Publish one already-validated generated boundary overlay.  The System loader performs every
  /// fallible check before its registry commit; this final move cannot allocate and therefore cannot
  /// leave the materialized solver out of sync with the committed plan.
  void replace_boundary_kernel(std::optional<CompiledFieldBoundaryKernel<Dim>> kernel) noexcept {
    static_assert(
        std::is_nothrow_move_assignable_v<std::optional<CompiledFieldBoundaryKernel<Dim>>>);
    boundary_kernel_ = std::move(kernel);
    boundary_contexts_.reset();
  }

  void install_newton(FieldNewtonOptions options) {
    validate_field_newton_options(options);
    if (newton_workspace_)
      throw std::logic_error("Cartesian Poisson Newton authority is already installed");
    newton_workspace_.emplace(rhs_.layout(), rhs_.distribution(), rhs_.local_rank(), options);
  }

  void set_boundary_contexts(std::shared_ptr<const PreparedFieldBoundaryContextSet<Dim>> contexts) {
    if (!boundary_kernel_)
      throw std::logic_error("Cartesian Poisson has no compiled dynamic boundary kernel");
    if (!contexts || contexts->size() != 1 || contexts->contexts().front().failure == nullptr)
      throw std::invalid_argument(
          "Cartesian Poisson dynamic boundary requires a fallible execution channel");
    boundary_contexts_ = std::move(contexts);
  }

  void install_nullspace(FieldNullspacePlan<Dim> plan,
                         PreparedVectorDistribution<Dim> distribution) {
    if (nullspace_installed_)
      throw std::logic_error("Cartesian Poisson nullspace authority is already installed");
    const std::array<PreparedVectorDistribution<Dim>, 1> distributions{distribution};
    distribution.require_collective_layout(rhs_, "Cartesian Poisson nullspace preparation", *lane_);
    validate_field_nullspace_basis<Dim>({&rhs_}, plan,
                                        std::span<const PreparedVectorDistribution<Dim>>(
                                            distributions.data(), distributions.size()),
                                        *lane_);
    if (singular_() != !plan.empty())
      throw std::invalid_argument(
          "Cartesian Poisson nullspace plan disagrees with the prepared operator kernel");
    nullspace_plan_ = std::move(plan);
    nullspace_distribution_ = std::move(distribution);
    nullspace_installed_ = true;
  }

  SolveReport solve(const field_type& warm_start, const ExecutionLane& lane) {
    if (all_reduce_max(&lane == lane_ ? 0L : 1L, *lane_) != 0)
      throw std::invalid_argument("Cartesian Poisson solve requires its prepared execution lane");
    std::exception_ptr layout_error;
    try {
      authenticate_(warm_start, "warm start");
    } catch (...) {
      layout_error = std::current_exception();
    }
    if (all_reduce_max(layout_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && layout_error)
        std::rethrow_exception(layout_error);
      throw std::runtime_error("Cartesian Poisson solve layout validation failed collectively");
    }

    if (!nullspace_installed_)
      throw std::logic_error("Cartesian Poisson solve has no prepared nullspace authority");

    detail::copy_component(warm_start, 0, candidate_, 0);
    apply_field_gauge(candidate_, nullspace_plan_, nullspace_distribution_, lane);

    if (newton_workspace_) {
      SolveReport report;
      try {
        report = newton_workspace_->solve(
            candidate_,
            [this](const field_type& iterate, field_type& residual, int iteration) {
              evaluate_residual_(iterate, residual, iteration);
            },
            [this](const field_type& iterate, const field_type& direction, field_type& output,
                   int iteration) { apply_linearized_(iterate, direction, output, iteration); },
            [this](field_type& value) {
              apply_field_gauge(value, nullspace_plan_, nullspace_distribution_, *lane_);
            },
            lane);
      } catch (const FieldNullspaceIncompatibleRhs& error) {
        report.mark_failed(SolveStatus::kIncompatibleRhs, SolveAction::kFailRun, error.what());
      } catch (const FieldNullspaceInvalidEvaluation& error) {
        report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun, error.what());
      }
      if (report.solved_value_available()) {
        apply_field_gauge(candidate_, nullspace_plan_, nullspace_distribution_, lane);
        fill_candidate_ghosts_(report.iters);
      }
      return report;
    }
    if (boundary_kernel_ && boundary_kernel_->observes_iteration)
      throw std::logic_error(
          "iterate-dependent Cartesian boundary requires a prepared Newton authority");

    try {
      evaluate_residual_(candidate_, residual_, 0);
    } catch (const FieldNullspaceIncompatibleRhs& error) {
      SolveReport report;
      report.mark_failed(SolveStatus::kIncompatibleRhs, SolveAction::kFailRun, error.what());
      return report;
    } catch (const FieldNullspaceInvalidEvaluation& error) {
      SolveReport report;
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun, error.what());
      return report;
    }
    detail::copy_component(residual_, 0, direction_, 0);

    const Real reference_squared = static_cast<Real>(all_reduce_sum(dot_local(rhs_, rhs_), lane));
    Real residual_squared =
        static_cast<Real>(all_reduce_sum(dot_local(residual_, residual_), lane));
    SolveReport report;
    report.evaluations = 1;
    report.reference_residual_norm = norm_from_squared_(reference_squared);
    report.residual_norm = norm_from_squared_(residual_squared);
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
      apply_field_gauge(candidate_, nullspace_plan_, nullspace_distribution_, lane);
      fill_candidate_ghosts_(0);
      report.mark_solved("cartesian_poisson_initial_residual");
      return report;
    }

    Real previous = residual_squared;
    for (int iteration = 0; iteration < options_.maximum_iterations; ++iteration) {
      apply_linearized_(candidate_, direction_, image_, 0);
      ++report.evaluations;
      const Real curvature = static_cast<Real>(all_reduce_sum(dot_local(direction_, image_), lane));
      if (!finite_(curvature) || !(curvature > Real(0))) {
        report.iters = iteration;
        report.mark_failed(SolveStatus::kBreakdown, SolveAction::kFailRun,
                           "cartesian_poisson_non_positive_curvature");
        return report;
      }
      const Real alpha = previous / curvature;
      report.step_norm =
          std::abs(alpha) *
          std::sqrt(std::max(
              static_cast<Real>(all_reduce_sum(dot_local(direction_, direction_), lane)), Real(0)));
      saxpy(candidate_, alpha, direction_);
      saxpy(residual_, -alpha, image_);

      residual_squared = static_cast<Real>(all_reduce_sum(dot_local(residual_, residual_), lane));
      report.iters = iteration + 1;
      report.residual_norm = norm_from_squared_(residual_squared);
      report.rel_residual = report.residual_norm / denominator;
      if (!finite_(report.residual_norm) || !finite_(report.step_norm)) {
        report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                           "cartesian_poisson_non_finite_iteration");
        return report;
      }
      if (report.residual_norm <= stop) {
        apply_field_gauge(candidate_, nullspace_plan_, nullspace_distribution_, lane);
        fill_candidate_ghosts_(0);
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
    if (!finite_(options_.relative_tolerance) || options_.relative_tolerance <= Real(0) ||
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
      for (const Face<Dim> face : {lower, upper}) {
        const std::size_t ordinal = static_cast<std::size_t>(face.ordinal());
        const CartesianBoundaryKind kind = options_.boundaries[ordinal];
        const Real alpha = options_.boundary_alpha[ordinal];
        const Real beta = options_.boundary_beta[ordinal];
        if (!finite_(alpha) || !finite_(beta))
          throw std::invalid_argument("Cartesian Poisson Robin coefficients must be finite");
        if (kind == CartesianBoundaryKind::periodic) {
          if (alpha != Real(0) || beta != Real(0))
            throw std::invalid_argument(
                "Cartesian Poisson periodic faces cannot carry Robin coefficients");
          continue;
        }
        if (kind == CartesianBoundaryKind::dirichlet && (alpha != Real(1) || beta != Real(0)))
          throw std::invalid_argument(
              "Cartesian Poisson Dirichlet faces require alpha=1 and beta=0");
        if (kind == CartesianBoundaryKind::neumann && (alpha != Real(0) || beta != Real(1)))
          throw std::invalid_argument("Cartesian Poisson Neumann faces require alpha=0 and beta=1");
        if (kind == CartesianBoundaryKind::mixed && alpha == Real(0) && beta == Real(0))
          throw std::invalid_argument(
              "Cartesian Poisson mixed faces require a non-degenerate Robin pair");
        const Real denominator = alpha / Real(2) + beta / geometry_.spacing(axis);
        const Real scale = std::max(
            Real(1), std::max(std::abs(alpha / Real(2)), std::abs(beta / geometry_.spacing(axis))));
        if (!finite_(denominator) ||
            std::abs(denominator) <= Real(64) * std::numeric_limits<Real>::epsilon() * scale)
          throw std::invalid_argument(
              "Cartesian Poisson boundary elimination denominator is singular");
      }
    }
  }

  bool singular_() const noexcept {
    for (std::size_t face = 0; face < options_.boundaries.size(); ++face)
      if (options_.boundaries[face] != CartesianBoundaryKind::periodic &&
          options_.boundary_alpha[face] != Real(0))
        return false;
    return true;
  }

  void authenticate_(const field_type& field, const char* role) const {
    if (field.layout() != rhs_.layout() || field.distribution() != rhs_.distribution() ||
        field.local_rank() != rhs_.local_rank() || field.ncomp() != 1)
      throw std::invalid_argument(std::string("Cartesian Poisson ") + role +
                                  " does not match its prepared scalar layout");
  }

  void fill_candidate_ghosts_(int iteration) {
    fill_residual_halo_(candidate_, iteration);
    for (std::size_t local = 0; local < candidate_.local_size(); ++local) {
      const auto& source = static_cast<const field_type&>(halo_).fab(local);
      for_each_cell(
          candidate_.fab(local).grown_box(),
          detail::CopyComponentKernel<Dim>{candidate_.fab(local).view(), source.view(), 0, 0});
    }
    Kokkos::fence();
  }

  void fill_static_halo_(bool homogeneous) {
    if (exchange_)
      exchange_->execute(halo_, *lane_);
    else
      fill_boundary(halo_, schedule_);
    Real spacing[Dim]{};
    for (int axis = 0; axis < Dim; ++axis)
      spacing[axis] = geometry_.spacing(axis);
    for (std::size_t local = 0; local < halo_.local_size(); ++local) {
      std::array<Real, 2 * Dim> values = options_.boundary_values;
      if (homogeneous)
        values.fill(Real(0));
      detail::PhysicalGhostKernel<Dim> kernel{halo_.fab(local).view(),
                                              geometry_.domain(),
                                              options_.boundaries,
                                              options_.boundary_alpha,
                                              options_.boundary_beta,
                                              values,
                                              {}};
      for (int axis = 0; axis < Dim; ++axis)
        kernel.spacing[axis] = spacing[axis];
      for_each_cell(halo_.fab(local).grown_box(), kernel);
    }
    Kokkos::fence();
  }

  FieldBoundaryExecutionContext<Dim> boundary_context_(int iteration) const {
    if (!boundary_contexts_)
      throw std::logic_error(
          "Cartesian Poisson dynamic boundary has no prepared execution context");
    return boundary_contexts_->view(0, iteration);
  }

  void synchronize_boundary_failure_(FieldBoundaryExecutionContext<Dim>& context,
                                     const char* message) {
    Kokkos::fence();
    if (context.failure->synchronize_across_ranks(*lane_))
      throw std::runtime_error(message);
  }

  void fill_residual_halo_(const field_type& iterate, int iteration) {
    detail::copy_component(iterate, 0, halo_, 0);
    fill_static_halo_(false);
    if (boundary_kernel_) {
      auto context = boundary_context_(iteration);
      context.failure->reset();
      for (int face = 0; face < 2 * Dim; ++face)
        boundary_kernel_->prepare_residual_view(face, iterate, halo_, geometry_, context);
      synchronize_boundary_failure_(
          context, "Cartesian Poisson dynamic boundary evaluation failed collectively");
    }
  }

  void fill_jvp_halo_(const field_type& iterate, const field_type& direction, int iteration) {
    detail::copy_component(direction, 0, halo_, 0);
    fill_static_halo_(true);
    if (boundary_kernel_) {
      auto context = boundary_context_(iteration);
      context.failure->reset();
      for (int face = 0; face < 2 * Dim; ++face)
        boundary_kernel_->prepare_jvp_view(face, iterate, direction, halo_, geometry_, context);
      synchronize_boundary_failure_(context,
                                    "Cartesian Poisson dynamic boundary JVP failed collectively");
    }
  }

  void apply_negative_laplacian_(field_type& output) {
    authenticate_(output, "operator output");
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

  void evaluate_residual_(const field_type& iterate, field_type& output, int iteration) {
    authenticate_(iterate, "residual iterate");
    authenticate_(output, "residual output");
    fill_residual_halo_(iterate, iteration);
    apply_negative_laplacian_(image_);
    lincomb(output, Real(1), rhs_, Real(-1), image_);
    if (boundary_kernel_) {
      auto context = boundary_context_(iteration);
      context.failure->reset();
      for (int face = 0; face < 2 * Dim; ++face)
        boundary_kernel_->add_residual(face, iterate, output, geometry_, context);
      synchronize_boundary_failure_(
          context, "Cartesian Poisson dynamic residual closure failed collectively");
    }
    require_field_nullspace_compatible(output, nullspace_plan_, nullspace_distribution_, *lane_);
  }

  void apply_linearized_(const field_type& iterate, const field_type& direction, field_type& output,
                         int iteration) {
    authenticate_(iterate, "linearization iterate");
    authenticate_(direction, "linearization direction");
    authenticate_(output, "operator output");
    fill_jvp_halo_(iterate, direction, iteration);
    apply_negative_laplacian_(output);
    if (boundary_kernel_) {
      auto context = boundary_context_(iteration);
      context.failure->reset();
      for (int face = 0; face < 2 * Dim; ++face)
        boundary_kernel_->apply_jvp(face, iterate, direction, output, geometry_, context);
      synchronize_boundary_failure_(context,
                                    "Cartesian Poisson dynamic JVP closure failed collectively");
    }
  }

  static bool finite_(Real value) noexcept { return std::isfinite(static_cast<double>(value)); }

  static Real norm_from_squared_(Real squared) noexcept {
    if (!finite_(squared))
      return squared;
    if (squared < Real(0))
      return std::numeric_limits<Real>::quiet_NaN();
    return squared > Real(0) ? std::sqrt(squared) : Real(0);
  }

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
  const ExecutionLane* lane_ = nullptr;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  std::unique_ptr<HaloExchange<Dim>> exchange_;
  std::optional<CompiledFieldBoundaryKernel<Dim>> boundary_kernel_;
  std::shared_ptr<const PreparedFieldBoundaryContextSet<Dim>> boundary_contexts_;
  std::optional<FieldNewtonKrylovWorkspace<Dim>> newton_workspace_;
  FieldNullspacePlan<Dim> nullspace_plan_;
  PreparedVectorDistribution<Dim> nullspace_distribution_ =
      PreparedVectorDistribution<Dim>::distributed();
  bool nullspace_installed_ = false;
};

}  // namespace pops::elliptic::nd
