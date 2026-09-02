/// @file
/// @brief Exact compile-time-ranked geometric multigrid for Cartesian scalar elliptic fields.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp>
#include <pops/numerics/elliptic/interface/elliptic_solver.hpp>
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_workspace.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft_multifab.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>
#include <pops/runtime/numerical_defaults.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::elliptic::mg {

/// Immutable controls of one prepared V-cycle hierarchy.
struct GeometricMultigridOptions {
  Real relative_tolerance = kMGDefaultRelTol;
  Real absolute_tolerance = kMGDefaultAbsTol;
  int maximum_cycles = kMGDefaultMaxCycles;
  int minimum_coarse_extent = kMGDefaultMinCoarse;
  int pre_sweeps = kMGDefaultPreSmooth;
  int post_sweeps = kMGDefaultPostSmooth;
  int bottom_sweeps = kMGDefaultBottomSweeps;
  int coarse_cell_threshold = kMGDefaultCoarseThreshold;
  Real jacobi_relaxation = Real(2) / Real(3);
  Real reaction = Real(0);
  bool allow_coarsening = true;

  bool operator==(const GeometricMultigridOptions&) const = default;
};

struct GeometricMultigridCapabilities {
  bool scalar_constant_coefficient = true;
  bool scalar_reaction = true;
  bool variable_diagonal = true;
  bool cross_tensor = false;
  bool embedded_boundary = true;

  constexpr bool operator==(const GeometricMultigridCapabilities&) const = default;
};

namespace detail {

inline std::size_t checked_size_product(std::size_t left, std::size_t right,
                                        const char* operation) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(operation);
  return left * right;
}

inline std::size_t checked_size_sum(std::size_t left, std::size_t right, const char* operation) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::length_error(operation);
  return left + right;
}

template <int Dim>
Extent<Dim> unit_ghosts() {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = 1;
  return result;
}

template <int Dim>
Extent<Dim> ratio_two() {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = 2;
  return result;
}

template <int Dim>
mesh::BoxArrayValidationBudget exact_layout_budget(const mesh::BoxArray<Dim>& layout) {
  const std::size_t pairs = layout.size() < 2
                                ? 0
                                : checked_size_product(layout.size(), layout.size() - 1,
                                                       "geometric MG layout-pair budget overflow") /
                                      2;
  return {layout.size(), pairs};
}

template <int Dim>
CopyScheduleBudget exact_copy_budget(const mesh::BoxArray<Dim>& left,
                                     const mesh::BoxArray<Dim>& right) {
  const auto overlap_pairs = [](std::size_t boxes) {
    return boxes < 2
               ? std::size_t{0}
               : checked_size_product(boxes, boxes - 1, "geometric MG overlap budget overflow") / 2;
  };
  const std::size_t pairs =
      checked_size_product(left.size(), right.size(), "geometric MG copy-pair budget overflow");
  return {pairs, pairs, overlap_pairs(left.size()), overlap_pairs(right.size())};
}

template <int Dim>
HaloScheduleBudget exact_halo_budget(const mesh::BoxArray<Dim>& layout, const Box<Dim>& domain) {
  const std::size_t boxes = layout.size();
  const std::size_t pairs =
      checked_size_product(boxes, boxes, "geometric MG halo-pair budget overflow");
  std::size_t images = 1;
  for (int axis = 0; axis < Dim; ++axis)
    images = checked_size_product(images, 3, "geometric MG image budget overflow");
  const std::size_t work =
      checked_size_product(pairs, images, "geometric MG halo-work budget overflow");
  const std::size_t jobs = checked_size_product(work, static_cast<std::size_t>(2 * Dim),
                                                "geometric MG halo-job budget overflow");
  const std::int64_t signed_cells = domain.numPts();
  if (signed_cells <= 0)
    throw std::invalid_argument("geometric MG halo domain must be non-empty");
  const std::size_t cells = static_cast<std::size_t>(signed_cells);
  const std::size_t elements =
      checked_size_product(jobs, cells, "geometric MG halo-element budget overflow");
  return {exact_layout_budget(layout),
          work,
          jobs,
          images,
          checked_size_product(boxes, std::size_t{2}, "geometric MG peer budget overflow"),
          elements,
          elements,
          elements};
}

template <int Dim>
BoundaryScheduleBudget exact_boundary_budget() {
  std::size_t entries = 1;
  for (int axis = 0; axis < Dim; ++axis)
    entries = checked_size_product(entries, 3, "geometric MG boundary budget overflow");
  return {entries - 1};
}

template <int Dim>
mesh::Distribution<Dim> rebind_distribution(const mesh::BoxArray<Dim>& layout,
                                            const mesh::Distribution<Dim>& model) {
  if (layout.size() != model.box_count())
    throw std::invalid_argument("geometric MG coarsening must retain the global patch cardinality");
  if (model.replicated())
    return mesh::Distribution<Dim>::replicated(layout, model.rank_space());
  return mesh::Distribution<Dim>::partitioned(layout, model.rank_space(), model.owners());
}

template <int Dim>
PhysicalBoundaryConditions<Dim> boundary_for_geometry(const PhysicalBoundaryConditions<Dim>& source,
                                                      const Geometry<Dim>& geometry,
                                                      bool homogeneous_values) {
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    spacing[axis] = geometry.spacing(axis);
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> face{axis, side};
      PhysicalBoundaryFace law = source.at(face);
      if (homogeneous_values)
        law.value = Real(0);
      faces[static_cast<std::size_t>(face.ordinal())] = law;
    }
  }
  return PhysicalBoundaryConditions<Dim>{source.topology(), faces, spacing};
}

template <int Dim>
PhysicalBoundaryConditions<Dim> coefficient_boundary_for_geometry(
    const PhysicalBoundaryConditions<Dim>& source, const Geometry<Dim>& geometry) {
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    spacing[axis] = geometry.spacing(axis);
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> face{axis, side};
      PhysicalBoundaryFace law = source.at(face);
      if (!source.topology().is_periodic(face))
        law = PhysicalBoundaryFace{PhysicalBoundaryKind::constant_extrapolation};
      faces[static_cast<std::size_t>(face.ordinal())] = law;
    }
  }
  return PhysicalBoundaryConditions<Dim>{source.topology(), faces, spacing};
}

template <int Dim>
std::string options_contract(const GeometricMultigridOptions& options) {
  ExactContractBuilder contract;
  contract.text("pops.elliptic.geometric-mg-options")
      .scalar(std::uint32_t{2})
      .scalar(std::int32_t{Dim})
      .scalar(options.relative_tolerance)
      .scalar(options.absolute_tolerance)
      .scalar(options.maximum_cycles)
      .scalar(options.minimum_coarse_extent)
      .scalar(options.pre_sweeps)
      .scalar(options.post_sweeps)
      .scalar(options.bottom_sweeps)
      .scalar(options.coarse_cell_threshold)
      .scalar(options.jacobi_relaxation)
      .scalar(options.reaction);
  return std::move(contract).release();
}

template <int Dim>
void validate_options(const GeometricMultigridOptions& options) {
  if (!std::isfinite(static_cast<double>(options.relative_tolerance)) ||
      options.relative_tolerance <= Real(0) ||
      !std::isfinite(static_cast<double>(options.absolute_tolerance)) ||
      options.absolute_tolerance < Real(0) || options.maximum_cycles < 1 ||
      options.minimum_coarse_extent < 1 || options.pre_sweeps < 0 || options.post_sweeps < 0 ||
      options.bottom_sweeps < 1 || options.coarse_cell_threshold < 0 ||
      !std::isfinite(static_cast<double>(options.jacobi_relaxation)) ||
      options.jacobi_relaxation <= Real(0) || options.jacobi_relaxation >= Real(2) ||
      !std::isfinite(static_cast<double>(options.reaction)) || options.reaction < Real(0))
    throw std::invalid_argument("geometric MG controls are invalid");
}

template <int Dim>
void validate_boundary(const Geometry<Dim>& geometry,
                       const PhysicalBoundaryConditions<Dim>& boundary) {
  for (int axis = 0; axis < Dim; ++axis) {
    if (boundary.spacing()[axis] != geometry.spacing(axis))
      throw std::invalid_argument("geometric MG boundary spacing differs from its exact geometry");
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> face{axis, side};
      const PhysicalBoundaryFace law = boundary.at(face);
      if (boundary.topology().is_periodic(face)) {
        if (law.kind != PhysicalBoundaryKind::external)
          throw std::invalid_argument("geometric MG periodic faces cannot carry a physical law");
      } else if (law.kind == PhysicalBoundaryKind::external) {
        throw std::invalid_argument(
            "geometric MG requires every physical face to own a native affine law");
      }
    }
  }
}

template <int Dim>
bool is_singular(const PhysicalBoundaryConditions<Dim>& boundary, Real reaction) {
  if (reaction > Real(0))
    return false;
  for (int axis = 0; axis < Dim; ++axis)
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> face{axis, side};
      if (boundary.topology().is_periodic(face))
        continue;
      const PhysicalBoundaryFace law = boundary.at(face);
      if (law.kind == PhysicalBoundaryKind::dirichlet ||
          (law.kind == PhysicalBoundaryKind::robin && law.alpha != Real(0)))
        return false;
    }
  return true;
}

}  // namespace detail

/// Geometric V-cycle over one immutable exact-ranked layout hierarchy.
///
/// Every level is produced by an exact factor-two coarsening of every axis and every patch. The
/// smoother, operator, transfers, halo exchange and boundary fill are one algorithm over `Dim`;
/// there is no dimension-erased mesh or 2D compatibility path.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class GeometricMG {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "GeometricMG only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  using field_type = MultiFab<Dim, MemorySpace>;
  using request_type = EllipticBuildRequest<Dim>;
  using nonlinear_workspace_type = AmrFieldNewtonKrylovWorkspace<Dim, MemorySpace>;
  using nonlinear_hierarchy_type = typename nonlinear_workspace_type::hierarchy_type;

  GeometricMG(request_type request, const ExecutionLane& lane,
              GeometricMultigridOptions options = {})
      : lane_(&lane),
        lane_borrow_(lane.borrow_immutably()),
        geometry_(request.geometry),
        boundary_(request.boundary),
        options_(options),
        singular_(detail::is_singular(boundary_, options_.reaction)) {
    std::exception_ptr validation_error;
    try {
      detail::validate_options<Dim>(options_);
      detail::validate_boundary(geometry_, boundary_);
      if (request.geometry.domain().empty() || request.boxes.empty() ||
          !request.boxes.tiles_exactly(request.geometry.domain(), request.layout_budget) ||
          !request.distribution.matches_layout(request.boxes) ||
          !request.distribution.rank_space().contains(request.local_rank) ||
          request.distribution.rank_space().size() != static_cast<std::size_t>(lane.size()) ||
          request.distribution.rank_space().linear_rank(request.local_rank) !=
              static_cast<std::size_t>(lane.rank()))
        throw std::invalid_argument("geometric MG build request is not an exact collective layout");
      for (int axis = 0; axis < Dim; ++axis)
        if (request.rhs_ghosts[axis] != 0 || request.phi_ghosts[axis] < 1)
          throw std::invalid_argument(
              "geometric MG requires a ghost-free RHS and at least one solution ghost");
    } catch (...) {
      validation_error = std::current_exception();
    }
    if (all_reduce_max(validation_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && validation_error)
        std::rethrow_exception(validation_error);
      throw std::runtime_error("geometric MG preparation failed collectively");
    }

    prepared_operator_contract_ = expected_operator_contract(request, options_);
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"pops.geometric-mg.execution-lane", lane.identity()},
             {"pops.geometric-mg.operator", prepared_operator_contract_.exact_fingerprint()}},
            lane))
      throw std::invalid_argument(
          "geometric MG prepared lane or operator differs across communicator ranks");
    build_hierarchy_(request, lane);
    try_prepare_fft_bottom_();
  }

  GeometricMG(const Geometry<Dim>& geometry, const mesh::BoxArray<Dim>& layout,
              const mesh::Distribution<Dim>& distribution, Index<Dim> local_rank,
              PhysicalBoundaryConditions<Dim> boundary, const ExecutionLane& lane,
              GeometricMultigridOptions options = {})
      : GeometricMG(request_type{geometry, layout, distribution, local_rank, std::move(boundary),
                                 Extent<Dim>{}, detail::unit_ghosts<Dim>(),
                                 detail::exact_layout_budget(layout)},
                    lane, options) {}

  GeometricMG(const GeometricMG&) = delete;
  GeometricMG& operator=(const GeometricMG&) = delete;
  GeometricMG(GeometricMG&&) noexcept = default;
  GeometricMG& operator=(GeometricMG&&) noexcept = default;

  static constexpr GeometricMultigridCapabilities capabilities() noexcept { return {}; }
  static constexpr EllipticOperatorIdentity operator_identity() noexcept {
    return {"pops.elliptic.geometric-mg.nd", 2};
  }
  static EllipticOperatorContract expected_operator_contract(
      const request_type& request, GeometricMultigridOptions options = {}) {
    detail::validate_options<Dim>(options);
    return make_expected_elliptic_operator_contract(operator_identity(), request,
                                                    detail::options_contract<Dim>(options));
  }

  field_type& phi() noexcept { return levels_.front()->phi; }
  const field_type& phi() const noexcept { return levels_.front()->phi; }
  field_type& rhs() noexcept { return levels_.front()->rhs; }
  const field_type& rhs() const noexcept { return levels_.front()->rhs; }
  const Geometry<Dim>& geom() const noexcept { return geometry_; }
  const PhysicalBoundaryConditions<Dim>& boundary() const noexcept { return boundary_; }
  const GeometricMultigridOptions& options() const noexcept { return options_; }
  const EllipticOperatorContract& prepared_operator_contract() const noexcept {
    return prepared_operator_contract_;
  }
  int num_levels() const noexcept { return static_cast<int>(levels_.size()); }
  bool fft_coarse_prepared() const noexcept { return static_cast<bool>(fft_bottom_); }
  bool used_fft_coarse() const noexcept { return used_fft_coarse_; }
  ::pops::elliptic::PoissonFftBottomKind fft_coarse_kind() const noexcept {
    return fft_bottom_ ? fft_bottom_->kind() : ::pops::elliptic::PoissonFftBottomKind::none;
  }
  int maximum_iterations() const noexcept {
    if (newton_workspace_)
      return newton_workspace_->options().max_iterations;
    if (linear_boundary_workspace_)
      return linear_boundary_workspace_->options().max_iterations;
    return options_.maximum_cycles;
  }
  Real residual() const noexcept { return last_report_.residual_norm; }
  const field_type& residual_field() const noexcept { return levels_.front()->residual; }
  const SolveReport& last_solve_report() const noexcept { return last_report_; }
  bool borrows_execution_lane(const ExecutionLane& lane) const noexcept { return lane_ == &lane; }

  void install_nullspace(FieldNullspacePlan<Dim> plan,
                         PreparedVectorDistribution<Dim> distribution) {
    if (nullspace_workspace_)
      throw std::logic_error("geometric MG nullspace authority is already installed");
    if (singular_ != !plan.empty())
      throw std::invalid_argument(
          "geometric MG nullspace plan disagrees with the prepared operator kernel");
    std::vector<const MultiFab<Dim>*> layouts{&rhs()};
    std::vector<PreparedVectorDistribution<Dim>> distributions{std::move(distribution)};
    nullspace_workspace_ = std::make_unique<FieldNullspaceWorkspace<Dim>>(
        std::move(plan), std::move(layouts), std::move(distributions), *lane_);
  }

  void install_newton(FieldNewtonOptions options) {
    if (newton_workspace_)
      throw std::logic_error("geometric MG Newton authority is already installed");
    validate_field_newton_options(options);
    Level& fine = *levels_.front();
    const std::array<const field_type*, 1> layouts{&fine.phi};
    const std::array<const field_type*, 1> masks{&fine.active};
    const std::array<Real, 1> measures{fine_cell_measure_()};
    newton_workspace_.emplace(layouts, masks, measures, options);
    linear_boundary_workspace_.reset();
    ensure_boundary_view_();
    prepare_dynamic_view_();
  }

  void install_coefficient(const field_type& conductivity) {
    if (levels_.empty())
      throw std::logic_error("geometric MG coefficient install requires a prepared hierarchy");
    WeightedPoissonFields<Dim, MemorySpace> probe;
    probe.coefficient = &conductivity;
    validate_weighted_poisson_fields(levels_.front()->phi, probe, "geometric MG coefficient");
    Level& fine = *levels_.front();
    if (!fine.coefficient)
      fine.coefficient.emplace(fine.phi.layout(), fine.phi.distribution(), fine.phi.local_rank(), 1,
                               detail::unit_ghosts<Dim>());
    copy_scalar_valid(conductivity, *fine.coefficient);
    for (std::size_t level = 1; level < levels_.size(); ++level) {
      Level& coarse = *levels_[level];
      if (!coarse.coefficient)
        coarse.coefficient.emplace(coarse.phi.layout(), coarse.phi.distribution(),
                                   coarse.phi.local_rank(), 1, detail::unit_ghosts<Dim>());
      const CopyScheduleBudget budget =
          detail::exact_copy_budget(coarse.coefficient->layout(),
                                    coarsen(levels_[level - 1]->coefficient->layout(), 2));
      average_down(*levels_[level - 1]->coefficient, *coarse.coefficient, 2, budget);
    }
    for (auto& level : levels_)
      fill_coefficient_ghosts_(*level);
    fft_bottom_.reset();
    used_fft_coarse_ = false;
  }

  void install_embedded_boundary(const field_type& active, const field_type& inverse_volume,
                                 const field_type& aperture_lower,
                                 const field_type& aperture_upper) {
    if (levels_.size() != 1)
      throw std::invalid_argument(
          "geometric MG embedded boundary refuses metric restriction through V-cycle coarsening; "
          "install on a single-level hierarchy or a FAC level with its own metric");
    Level& fine = *levels_.front();
    WeightedPoissonFields<Dim, MemorySpace> probe;
    probe.inverse_volume = &inverse_volume;
    probe.aperture_lower = &aperture_lower;
    probe.aperture_upper = &aperture_upper;
    probe.active = &active;
    validate_weighted_poisson_fields(fine.phi, probe, "geometric MG embedded boundary");
    if (fine.inverse_volume)
      throw std::logic_error("geometric MG embedded-boundary authority is already installed");
    copy_scalar_valid(active, fine.active);
    fine.inverse_volume.emplace(inverse_volume.layout(), inverse_volume.distribution(),
                                inverse_volume.local_rank(), 1, Extent<Dim>{});
    copy_scalar_valid(inverse_volume, *fine.inverse_volume);
    fine.aperture_lower.emplace(aperture_lower.layout(), aperture_lower.distribution(),
                                aperture_lower.local_rank(), Dim, Extent<Dim>{});
    fine.aperture_upper.emplace(aperture_upper.layout(), aperture_upper.distribution(),
                                aperture_upper.local_rank(), Dim, Extent<Dim>{});
    copy_vector_valid_(aperture_lower, *fine.aperture_lower);
    copy_vector_valid_(aperture_upper, *fine.aperture_upper);
    fft_bottom_.reset();
    used_fft_coarse_ = false;
  }

  void install_boundary_kernel(CompiledFieldBoundaryKernel<Dim> kernel) {
    if (boundary_kernel_)
      throw std::logic_error("geometric MG boundary kernel is already installed");
    kernel.validate();
    Level& fine = *levels_.front();
    ensure_boundary_view_();
    boundary_kernel_ = std::move(kernel);
    boundary_contexts_.reset();
    if (!newton_workspace_ && !boundary_kernel_->observes_iteration) {
      const FieldNewtonOptions options = linear_boundary_newton_options_();
      const std::array<const field_type*, 1> layouts{&fine.phi};
      const std::array<const field_type*, 1> masks{&fine.active};
      const std::array<Real, 1> measures{fine_cell_measure_()};
      linear_boundary_workspace_.emplace(layouts, masks, measures, options);
    }
    prepare_dynamic_view_();
  }

  void set_boundary_contexts(std::shared_ptr<const PreparedFieldBoundaryContextSet<Dim>> contexts,
                             std::size_t level = 0) {
    if (!boundary_kernel_)
      throw std::logic_error("geometric MG has no compiled dynamic boundary kernel");
    if (!contexts || level >= contexts->size() || contexts->contexts()[level].failure == nullptr)
      throw std::invalid_argument(
          "geometric MG dynamic boundary requires a fallible execution channel");
    boundary_contexts_ = std::move(contexts);
    boundary_context_level_ = level;
  }

  SolveReport solve() {
    Level& fine = *levels_.front();
    if (!nullspace_workspace_)
      throw std::logic_error("geometric MG solve has no prepared nullspace authority");
    try {
      nullspace_workspace_->require_compatible(fine.rhs);
    } catch (const FieldNullspaceIncompatibleRhs& error) {
      SolveReport report;
      report.mark_failed(SolveStatus::kIncompatibleRhs, SolveAction::kFailRun, error.what());
      last_report_ = report;
      return last_report_;
    } catch (const FieldNullspaceInvalidEvaluation& error) {
      SolveReport report;
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun, error.what());
      last_report_ = report;
      return last_report_;
    }
    nullspace_workspace_->apply_gauge(fine.phi);
    used_fft_coarse_ = false;
    fft_bottom_attempted_ = false;
    fft_bottom_report_ = {};

    if (newton_workspace_ || boundary_kernel_)
      return solve_dynamic_(fine);

    compute_residual_(fine);
    const Real reference = global_norm_inf_(fine.residual);
    SolveReport report;
    report.evaluations = 1;
    if (!std::isfinite(static_cast<double>(reference))) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                         "geometric_mg_non_finite_initial_residual");
      last_report_ = report;
      return last_report_;
    }
    report.reference_residual_norm = reference;
    report.residual_norm = reference;
    report.rel_residual = reference > Real(0) ? Real(1) : Real(0);
    const Real stop =
        std::max(options_.absolute_tolerance,
                 options_.relative_tolerance * std::max(reference, Real(1)));
    if (reference <= stop) {
      fill_ghosts_(fine);
      report.mark_solved("geometric_mg_initial_residual");
      last_report_ = report;
      return last_report_;
    }

    for (int cycle = 0; cycle < options_.maximum_cycles; ++cycle) {
      v_cycle_(0);
      if (fft_bottom_attempted_ && !fft_bottom_report_.solved()) {
        report.mark_failed(fft_bottom_report_.status, SolveAction::kFailRun,
                           std::string("geometric_mg_fft_bottom:") + fft_bottom_report_.reason);
        last_report_ = report;
        return last_report_;
      }
      nullspace_workspace_->apply_gauge(fine.phi);
      compute_residual_(fine);
      ++report.evaluations;
      report.iters = cycle + 1;
      report.residual_norm = global_norm_inf_(fine.residual);
      report.rel_residual = report.residual_norm / reference;
      if (!std::isfinite(static_cast<double>(report.residual_norm))) {
        report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                           "geometric_mg_non_finite_cycle");
        last_report_ = report;
        return last_report_;
      }
      if (report.residual_norm <= stop) {
        fill_ghosts_(fine);
        report.mark_solved("geometric_mg_converged");
        last_report_ = report;
        return last_report_;
      }
    }

    report.mark_failed(SolveStatus::kIterationLimit, SolveAction::kFailRun,
                       "geometric_mg_iteration_limit");
    last_report_ = report;
    return last_report_;
  }

 private:
  struct Level {
    Geometry<Dim> geometry;
    PhysicalBoundaryConditions<Dim> boundary;
    field_type phi;
    field_type rhs;
    field_type residual;
    field_type scratch;
    field_type correction;
    field_type active;
    std::optional<field_type> coefficient{};
    std::optional<field_type> inverse_volume{};
    std::optional<field_type> aperture_lower{};
    std::optional<field_type> aperture_upper{};
    HaloSchedule<Dim> halo_schedule;
    PreparedPhysicalBoundary<Dim> physical_boundary;
    PreparedPhysicalBoundary<Dim> homogeneous_physical_boundary;
    PreparedPhysicalBoundary<Dim> coefficient_boundary;
    std::unique_ptr<HaloExchange<Dim, MemorySpace>> exchange;

    Level(Geometry<Dim> level_geometry, mesh::BoxArray<Dim> layout,
          mesh::Distribution<Dim> distribution, Index<Dim> local_rank,
          PhysicalBoundaryConditions<Dim> level_boundary, std::uint64_t generation,
          const ExecutionLane& lane)
        : geometry(level_geometry),
          boundary(std::move(level_boundary)),
          phi(layout, distribution, local_rank, 1, detail::unit_ghosts<Dim>()),
          rhs(layout, distribution, local_rank, 1, Extent<Dim>{}),
          residual(layout, distribution, local_rank, 1, Extent<Dim>{}),
          scratch(layout, distribution, local_rank, 1, Extent<Dim>{}),
          correction(layout, distribution, local_rank, 1, Extent<Dim>{}),
          active(layout, distribution, local_rank, 1, Extent<Dim>{}),
          halo_schedule(
              prepare_halo_schedule(phi, geometry.domain(), boundary.topology(),
                                    detail::exact_halo_budget(layout, geometry.domain()))),
          physical_boundary(prepare_physical_boundary(geometry.domain(), detail::unit_ghosts<Dim>(),
                                                      boundary,
                                                      detail::exact_boundary_budget<Dim>())),
          homogeneous_physical_boundary(
              prepare_physical_boundary(geometry.domain(), detail::unit_ghosts<Dim>(),
                                        detail::boundary_for_geometry(boundary, geometry, true),
                                        detail::exact_boundary_budget<Dim>())),
          coefficient_boundary(prepare_physical_boundary(
              geometry.domain(), detail::unit_ghosts<Dim>(),
              detail::coefficient_boundary_for_geometry(boundary, geometry),
              detail::exact_boundary_budget<Dim>())) {
      active.set_val(Real(1));
      const bool remote = all_reduce_max(halo_schedule.has_remote_jobs() ? 1L : 0L, lane) != 0;
      if (remote) {
        HaloExchangeContext context{};
        context.context_generation = generation + 1;
        context.schedule_generation = generation + 1;
        exchange = std::make_unique<HaloExchange<Dim, MemorySpace>>(halo_schedule, lane, context);
      }
    }
  };

  static bool coarsenable_(const Geometry<Dim>& geometry, const mesh::BoxArray<Dim>& layout,
                           const GeometricMultigridOptions& options) {
    if (!options.allow_coarsening)
      return false;
    const std::int64_t cells = geometry.domain().numPts();
    if (options.coarse_cell_threshold > 0 &&
        cells <= static_cast<std::int64_t>(options.coarse_cell_threshold))
      return false;
    const Extent<Dim> ratio = detail::ratio_two<Dim>();
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t length = geometry.domain().length(axis);
      if (length % 2 != 0 || length / 2 < options.minimum_coarse_extent)
        return false;
    }
    for (const Box<Dim>& patch : layout.boxes())
      if (refine(coarsen(patch, ratio), ratio) != patch)
        return false;
    return true;
  }

  void build_hierarchy_(const request_type& request, const ExecutionLane& lane) {
    Geometry<Dim> level_geometry = request.geometry;
    mesh::BoxArray<Dim> level_layout = request.boxes;
    mesh::Distribution<Dim> level_distribution = request.distribution;
    std::uint64_t generation = 0;
    while (true) {
      const bool correction_level = !levels_.empty();
      auto level_boundary =
          detail::boundary_for_geometry(request.boundary, level_geometry, correction_level);
      levels_.push_back(std::make_unique<Level>(level_geometry, level_layout, level_distribution,
                                                request.local_rank, std::move(level_boundary),
                                                generation, lane));
      if (!coarsenable_(level_geometry, level_layout, options_))
        break;
      const Extent<Dim> ratio = detail::ratio_two<Dim>();
      const mesh::BoxArray<Dim> coarse_layout = coarsen(level_layout, ratio);
      const Box<Dim> coarse_domain = coarsen(level_geometry.domain(), ratio);
      const auto budget = detail::exact_layout_budget(coarse_layout);
      if (!coarse_layout.tiles_exactly(coarse_domain, budget))
        throw std::invalid_argument(
            "geometric MG exact coarsening no longer tiles its coarse domain");
      level_distribution = detail::rebind_distribution(coarse_layout, level_distribution);
      level_layout = coarse_layout;
      level_geometry =
          Geometry<Dim>::from_bounds(coarse_domain, level_geometry.lower(), level_geometry.upper());
      ++generation;
    }
  }

  EllipticBuildRequest<Dim> fft_request_from_level_(const Level& level) const {
    return EllipticBuildRequest<Dim>{
        level.geometry,
        level.phi.layout(),
        level.phi.distribution(),
        level.phi.local_rank(),
        level.boundary,
        Extent<Dim>{},
        detail::unit_ghosts<Dim>(),
        detail::exact_layout_budget(level.phi.layout()),
    };
  }

  void try_prepare_fft_bottom_() {
    fft_bottom_.reset();
    used_fft_coarse_ = false;
    fft_bottom_attempted_ = false;
    fft_bottom_report_ = {};
    if (levels_.empty() || uses_weighted_operator_(*levels_.back()) || options_.reaction != Real(0))
      return;
    fft_bottom_ = ::pops::elliptic::PoissonFftMultiFabAdapter<Dim>::try_make(
        fft_request_from_level_(*levels_.back()), *lane_, options_.reaction, false, false);
  }

  void fill_ghosts_(Level& level) {
    if (level.exchange)
      level.exchange->execute(level.phi, *lane_);
    else
      fill_boundary(level.phi, level.halo_schedule);
    fill_physical_boundary(level.phi, level.physical_boundary);
  }

  void fill_residual_boundary_view_(const field_type& source, int iteration) {
    if (!boundary_view_)
      throw std::logic_error("geometric MG dynamic boundary view is absent");
    Level& fine = *levels_.front();
    copy_scalar_valid(source, *boundary_view_);
    if (fine.exchange)
      fine.exchange->execute(*boundary_view_, *lane_);
    else
      fill_boundary(*boundary_view_, fine.halo_schedule);
    fill_physical_boundary(*boundary_view_, fine.physical_boundary);
    if (!boundary_kernel_)
      return;
    FieldBoundaryExecutionContext<Dim> context = boundary_context_at_(iteration);
    context.failure->reset();
    for (int face = 0; face < 2 * Dim; ++face)
      boundary_kernel_->prepare_residual_view(face, source, *boundary_view_, fine.geometry,
                                              context);
    synchronize_boundary_failure_(context,
                                  "geometric MG dynamic boundary residual failed collectively");
  }

  void fill_jvp_boundary_view_(const field_type& iterate, const field_type& direction,
                               int iteration) {
    if (!boundary_view_)
      throw std::logic_error("geometric MG dynamic boundary view is absent");
    Level& fine = *levels_.front();
    copy_scalar_valid(direction, *boundary_view_);
    if (fine.exchange)
      fine.exchange->execute(*boundary_view_, *lane_);
    else
      fill_boundary(*boundary_view_, fine.halo_schedule);
    fill_physical_boundary(*boundary_view_, fine.homogeneous_physical_boundary);
    if (!boundary_kernel_)
      return;
    FieldBoundaryExecutionContext<Dim> context = boundary_context_at_(iteration);
    context.failure->reset();
    for (int face = 0; face < 2 * Dim; ++face)
      boundary_kernel_->prepare_jvp_view(face, iterate, direction, *boundary_view_, fine.geometry,
                                         context);
    synchronize_boundary_failure_(context, "geometric MG dynamic boundary JVP failed collectively");
  }

  void evaluate_dynamic_residual_(const field_type& iterate, field_type& output, int iteration) {
    Level& fine = *levels_.front();
    fill_residual_boundary_view_(iterate, iteration);
    apply_level_operator_(fine, *boundary_view_, fine.scratch);
    lincomb(output, Real(1), fine.rhs, Real(-1), fine.scratch);
    if (boundary_kernel_) {
      FieldBoundaryExecutionContext<Dim> context = boundary_context_at_(iteration);
      context.failure->reset();
      for (int face = 0; face < 2 * Dim; ++face)
        boundary_kernel_->add_residual(face, iterate, output, fine.geometry, context);
      synchronize_boundary_failure_(context,
                                    "geometric MG dynamic residual closure failed collectively");
    }
    nullspace_workspace_->require_compatible(output);
  }

  void apply_dynamic_linearized_(const field_type& iterate, const field_type& direction,
                                 field_type& output, int iteration) {
    Level& fine = *levels_.front();
    fill_jvp_boundary_view_(iterate, direction, iteration);
    apply_level_operator_(fine, *boundary_view_, output);
    if (boundary_kernel_) {
      FieldBoundaryExecutionContext<Dim> context = boundary_context_at_(iteration);
      context.failure->reset();
      for (int face = 0; face < 2 * Dim; ++face)
        boundary_kernel_->apply_jvp(face, iterate, direction, output, fine.geometry, context);
      synchronize_boundary_failure_(context,
                                    "geometric MG dynamic JVP closure failed collectively");
    }
  }

  SolveReport solve_dynamic_(Level& fine) {
    if (boundary_kernel_ && !boundary_contexts_)
      throw std::logic_error("geometric MG dynamic boundary has no execution context");
    if (boundary_kernel_ && boundary_kernel_->observes_iteration && !newton_workspace_)
      throw std::logic_error(
          "iterate-dependent geometric MG boundary requires a prepared Newton authority");
    auto* workspace = newton_workspace_ ? &*newton_workspace_ : &*linear_boundary_workspace_;
    SolveReport report;
    try {
      report = workspace->solve(
          dynamic_candidate_view_,
          [this](const nonlinear_hierarchy_type& iterate, nonlinear_hierarchy_type& residual,
                 int iteration) {
            evaluate_dynamic_residual_(iterate.front(), residual.front(), iteration);
          },
          [this](const nonlinear_hierarchy_type& iterate, const nonlinear_hierarchy_type& direction,
                 nonlinear_hierarchy_type& output, int iteration) {
            apply_dynamic_linearized_(iterate.front(), direction.front(), output.front(),
                                      iteration);
          },
          [this](nonlinear_hierarchy_type& values) {
            nullspace_workspace_->apply_gauge(values.front());
          },
          *lane_);
    } catch (const FieldNullspaceIncompatibleRhs& error) {
      report.mark_failed(SolveStatus::kIncompatibleRhs, SolveAction::kFailRun, error.what());
    } catch (const FieldNullspaceInvalidEvaluation& error) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun, error.what());
    }
    if (report.solved_value_available()) {
      nullspace_workspace_->apply_gauge(fine.phi);
      fill_residual_boundary_view_(fine.phi, report.iters);
      for (std::size_t local = 0; local < fine.phi.local_size(); ++local) {
        const auto source = static_cast<const field_type&>(*boundary_view_).fab(local).view();
        const auto destination = fine.phi.fab(local).view();
        for_each_cell(fine.phi.fab(local).grown_box(), [=] POPS_HD(const Index<Dim>& cell) {
          destination(cell, 0) = source(cell, 0);
        });
      }
      ::pops::device_fence();
    }
    last_report_ = report;
    return last_report_;
  }

  FieldBoundaryExecutionContext<Dim> boundary_context_at_(int iteration) const {
    if (!boundary_contexts_)
      throw std::logic_error("geometric MG dynamic boundary context is absent");
    return boundary_contexts_->view(boundary_context_level_, iteration);
  }

  void synchronize_boundary_failure_(FieldBoundaryExecutionContext<Dim>& context,
                                     const char* message) {
    ::pops::device_fence();
    if (context.failure->synchronize_across_ranks(*lane_))
      throw std::runtime_error(message);
  }

  FieldNewtonOptions linear_boundary_newton_options_() const {
    FieldNewtonOptions options;
    options.tolerance = std::max(options_.relative_tolerance, options_.absolute_tolerance > Real(0)
                                                                  ? options_.absolute_tolerance
                                                                  : options_.relative_tolerance);
    options.max_iterations = 1;
    options.linear_tolerance = options_.relative_tolerance;
    options.linear_max_iterations = std::max(1, options_.maximum_cycles);
    options.restart = std::min(30, options.linear_max_iterations);
    validate_field_newton_options(options);
    return options;
  }

  Real fine_cell_measure_() const noexcept {
    Real measure = Real(1);
    for (int axis = 0; axis < Dim; ++axis)
      measure *= levels_.front()->geometry.spacing(axis);
    return measure;
  }

  void prepare_dynamic_view_() {
    dynamic_candidate_view_.clear();
    dynamic_candidate_view_.push_back(&levels_.front()->phi);
  }

  void ensure_boundary_view_() {
    if (boundary_view_)
      return;
    Level& fine = *levels_.front();
    boundary_view_.emplace(fine.phi.layout(), fine.phi.distribution(), fine.phi.local_rank(), 1,
                           detail::unit_ghosts<Dim>());
  }

  WeightedPoissonFields<Dim, MemorySpace> weighted_fields_(Level& level) const {
    WeightedPoissonFields<Dim, MemorySpace> fields;
    if (level.coefficient)
      fields.coefficient = &*level.coefficient;
    if (level.inverse_volume)
      fields.inverse_volume = &*level.inverse_volume;
    if (level.aperture_lower)
      fields.aperture_lower = &*level.aperture_lower;
    if (level.aperture_upper)
      fields.aperture_upper = &*level.aperture_upper;
    if (level.inverse_volume)
      fields.active = &level.active;
    return fields;
  }

  bool uses_weighted_operator_(const Level& level) const noexcept {
    return level.coefficient.has_value() || level.inverse_volume.has_value();
  }

  void apply_level_operator_(Level& level, const field_type& input, field_type& output) {
    if (!uses_weighted_operator_(level)) {
      apply_poisson_operator_valid(input, level.geometry, output, options_.reaction);
      return;
    }
    apply_weighted_poisson_operator_valid(input, level.geometry, output, options_.reaction,
                                          weighted_fields_(level));
  }

  void fill_coefficient_ghosts_(Level& level) {
    if (!level.coefficient)
      return;
    if (level.exchange)
      level.exchange->execute(*level.coefficient, *lane_);
    else
      fill_boundary(*level.coefficient, level.halo_schedule);
    fill_physical_boundary(*level.coefficient, level.coefficient_boundary);
  }

  static void copy_vector_valid_(const field_type& source, field_type& destination) {
    if (source.layout() != destination.layout() || source.ncomp() != destination.ncomp())
      throw std::invalid_argument("geometric MG vector copy requires one exact layout");
    for (std::size_t local = 0; local < source.local_size(); ++local) {
      const auto in = source.fab(local).view();
      const auto out = destination.fab(local).view();
      const int components = source.ncomp();
      for_each_cell(source.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        for (int component = 0; component < components; ++component)
          out(cell, component) = in(cell, component);
      });
    }
    ::pops::device_fence();
  }

  void smooth_(Level& level, int sweeps) {
    for (int sweep = 0; sweep < sweeps; ++sweep) {
      fill_ghosts_(level);
      fill_coefficient_ghosts_(level);
      if (uses_weighted_operator_(level))
        damped_jacobi_weighted_update_valid(level.phi, level.rhs, level.geometry, level.scratch,
                                            options_.jacobi_relaxation, options_.reaction,
                                            weighted_fields_(level));
      else
        damped_jacobi_update_valid(level.phi, level.rhs, level.geometry, level.scratch,
                                   options_.jacobi_relaxation, options_.reaction);
      copy_scalar_valid(level.scratch, level.phi);
    }
  }

  void compute_residual_(Level& level) {
    fill_ghosts_(level);
    fill_coefficient_ghosts_(level);
    if (uses_weighted_operator_(level))
      weighted_poisson_residual_valid(level.phi, level.rhs, level.geometry, level.residual,
                                      options_.reaction, weighted_fields_(level));
    else
      poisson_residual_valid(level.phi, level.rhs, level.geometry, level.residual,
                             options_.reaction);
  }

  void v_cycle_(std::size_t level_index) {
    Level& level = *levels_.at(level_index);
    if (level_index + 1 == levels_.size()) {
      if (fft_bottom_) {
        fft_bottom_report_ = fft_bottom_->apply(level.rhs, level.phi);
        fft_bottom_attempted_ = true;
        used_fft_coarse_ = true;
        if (fft_bottom_report_.solved())
          fill_ghosts_(level);
        return;
      }
      smooth_(level, options_.bottom_sweeps);
      return;
    }

    smooth_(level, options_.pre_sweeps);
    compute_residual_(level);
    Level& coarse = *levels_.at(level_index + 1);
    coarse.phi.set_val(Real(0));
    coarse.rhs.set_val(Real(0));
    const CopyScheduleBudget restriction_budget =
        detail::exact_copy_budget(coarse.rhs.layout(), coarsen(level.residual.layout(), 2));
    average_down(level.residual, coarse.rhs, 2, restriction_budget);
    v_cycle_(level_index + 1);

    level.correction.set_val(Real(0));
    const CopyScheduleBudget prolongation_budget =
        detail::exact_copy_budget(coarsen(level.correction.layout(), 2), coarse.phi.layout());
    interpolate(coarse.phi, level.correction, 2, prolongation_budget);
    saxpy(level.phi, Real(1), level.correction);
    smooth_(level, options_.post_sweeps);
  }

  Real global_norm_inf_(const field_type& field) const {
    return static_cast<Real>(all_reduce_max(static_cast<double>(norm_inf(field)), *lane_));
  }

  const ExecutionLane* lane_ = nullptr;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  Geometry<Dim> geometry_;
  PhysicalBoundaryConditions<Dim> boundary_;
  GeometricMultigridOptions options_{};
  bool singular_ = false;
  std::vector<std::unique_ptr<Level>> levels_{};
  std::unique_ptr<FieldNullspaceWorkspace<Dim>> nullspace_workspace_{};
  std::optional<CompiledFieldBoundaryKernel<Dim>> boundary_kernel_{};
  std::shared_ptr<const PreparedFieldBoundaryContextSet<Dim>> boundary_contexts_{};
  std::size_t boundary_context_level_ = 0;
  std::optional<field_type> boundary_view_{};
  std::optional<nonlinear_workspace_type> newton_workspace_{};
  std::optional<nonlinear_workspace_type> linear_boundary_workspace_{};
  std::vector<field_type*> dynamic_candidate_view_{};
  EllipticOperatorContract prepared_operator_contract_{};
  SolveReport last_report_{};
  std::unique_ptr<::pops::elliptic::PoissonFftMultiFabAdapter<Dim>> fft_bottom_{};
  SolveReport fft_bottom_report_{};
  bool used_fft_coarse_ = false;
  bool fft_bottom_attempted_ = false;
};

static_assert(EllipticSolver<GeometricMG<1>>);
static_assert(EllipticSolver<GeometricMG<2>>);
static_assert(EllipticSolver<GeometricMG<3>>);

}  // namespace pops::elliptic::mg
