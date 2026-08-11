/// @file
/// @brief Exact compile-time-ranked composite FAC Poisson solver for nested Cartesian AMR.

#pragma once

#include <pops/amr/refinement_ratio.hpp>
#include <pops/amr/transfer/transfer_provider.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/interface/elliptic_solver.hpp>
#include <pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp>
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/numerics/elliptic/mg/composite_fac_nlevel.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>
#include <pops/runtime/numerical_defaults.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
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

template <int Dim>
struct CompositeFacBuildRequest {
  static_assert(Dim >= 1 && Dim <= 3,
                "CompositeFacBuildRequest only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;

  std::vector<EllipticBuildRequest<Dim>> levels;
  std::vector<::pops::amr::RefinementRatio<Dim>> ratios;
};

struct CompositeFacCapabilities {
  bool scalar_constant_coefficient = true;
  bool partial_refinement = true;
  bool arbitrary_level_count = true;
  bool replicated_mpi = true;
  bool distributed_mpi = false;
  bool variable_diagonal = false;
  bool cross_tensor = false;
  bool embedded_boundary = false;

  constexpr bool operator==(const CompositeFacCapabilities&) const = default;
};

namespace detail {

inline void validate_fac_options(const CompositeFacOptions& options) {
  if (options.max_iters < 1 || options.fine_sweeps < 1 || options.coarse_cycles < 1 ||
      !std::isfinite(static_cast<double>(options.rel_tol)) || options.rel_tol <= Real(0) ||
      !std::isfinite(static_cast<double>(options.abs_tol)) || options.abs_tol < Real(0) ||
      !std::isfinite(static_cast<double>(options.coarse_rel_tol)) ||
      options.coarse_rel_tol <= Real(0) ||
      !std::isfinite(static_cast<double>(options.coarse_abs_tol)) ||
      options.coarse_abs_tol < Real(0))
    throw std::invalid_argument("composite FAC controls are invalid");
}

template <int Dim>
std::string fac_options_contract(const CompositeFacOptions& options, Real reaction) {
  ExactContractBuilder contract;
  contract.text("pops.elliptic.composite-fac-options")
      .scalar(std::uint32_t{2})
      .scalar(std::int32_t{Dim})
      .scalar(options.max_iters)
      .scalar(options.fine_sweeps)
      .scalar(options.rel_tol)
      .scalar(options.abs_tol)
      .scalar(options.coarse_rel_tol)
      .scalar(options.coarse_abs_tol)
      .scalar(options.coarse_cycles)
      .scalar(options.verbose)
      .scalar(reaction);
  return std::move(contract).release();
}

template <int Dim>
std::string fac_hierarchy_contract(const CompositeFacBuildRequest<Dim>& request) {
  ExactContractBuilder contract;
  contract.text("pops.elliptic.composite-fac-hierarchy")
      .scalar(std::uint32_t{2})
      .scalar(std::int32_t{Dim})
      .scalar(static_cast<std::uint64_t>(request.levels.size()));
  for (const auto& level : request.levels)
    contract.bytes(elliptic_contract_detail::build_request_contract(level));
  contract.scalar(static_cast<std::uint64_t>(request.ratios.size()));
  for (const auto& ratio : request.ratios)
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(ratio[axis]);
  return std::move(contract).release();
}

template <int Dim>
std::string fac_build_contract(const CompositeFacBuildRequest<Dim>& request,
                               const CompositeFacOptions& options, Real reaction,
                               const ExecutionLane& lane) {
  ExactContractBuilder contract;
  contract.text("pops.elliptic.composite-fac-build")
      .scalar(std::uint32_t{2})
      .scalar(std::int32_t{Dim})
      .text(lane.identity())
      .bytes(fac_hierarchy_contract(request))
      .bytes(fac_options_contract<Dim>(options, reaction));
  return std::move(contract).release();
}

template <int Dim>
Extent<Dim> ratio_extent(const ::pops::amr::RefinementRatio<Dim>& ratio) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = ratio[axis];
  return result;
}

}  // namespace detail

/// Composite multilevel correction over a replicated nested AMR hierarchy.
///
/// The cycle restricts the active residual from every refined level into the covered parent cells,
/// solves that composite correction with a true geometric V-cycle on the complete coarse level,
/// prolongs the correction through the hierarchy, relaxes each uncovered level and averages the
/// accepted fine solution back down. Sparse fine layouts are first-class. Distributed refined
/// ownership is rejected explicitly until its inter-level transfer transport is prepared; MPI may
/// still execute this provider redundantly with replicated levels.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class CompositeFacPoisson {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "CompositeFacPoisson only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  using field_type = MultiFab<Dim, MemorySpace>;
  using request_type = CompositeFacBuildRequest<Dim>;
  using nonlinear_workspace_type = AmrFieldNewtonKrylovWorkspace<Dim, MemorySpace>;
  using nonlinear_hierarchy_type = typename nonlinear_workspace_type::hierarchy_type;

  CompositeFacPoisson(request_type request, const ExecutionLane& lane,
                      CompositeFacOptions options = {}, Real reaction = Real(0))
      : lane_(&lane),
        lane_borrow_(lane.borrow_immutably()),
        options_(options),
        reaction_(reaction) {
    std::exception_ptr validation_error;
    try {
      detail::validate_fac_options(options_);
      if (!std::isfinite(static_cast<double>(reaction_)) || reaction_ < Real(0))
        throw std::invalid_argument("composite FAC reaction must be finite and non-negative");
      validate_request_(request, lane);
    } catch (...) {
      validation_error = std::current_exception();
    }
    if (all_reduce_max(validation_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && validation_error)
        std::rethrow_exception(validation_error);
      throw std::runtime_error("composite FAC preparation failed collectively");
    }

    build_levels_(request);
    build_connections_(request.ratios);
    build_coarse_solver_(request.levels.front(), lane);
    exact_prepared_contract_ = detail::fac_build_contract(request, options_, reaction_, lane);
  }

  CompositeFacPoisson(const CompositeFacPoisson&) = delete;
  CompositeFacPoisson& operator=(const CompositeFacPoisson&) = delete;
  CompositeFacPoisson(CompositeFacPoisson&&) noexcept = default;
  CompositeFacPoisson& operator=(CompositeFacPoisson&&) noexcept = default;

  static constexpr CompositeFacCapabilities capabilities() noexcept { return {}; }
  static constexpr EllipticOperatorIdentity operator_identity() noexcept {
    return {"pops.elliptic.composite-fac.nd", 2};
  }
  static std::string expected_prepared_contract(const request_type& request,
                                                const ExecutionLane& lane,
                                                CompositeFacOptions options = {},
                                                Real reaction = Real(0)) {
    detail::validate_fac_options(options);
    if (!std::isfinite(static_cast<double>(reaction)) || reaction < Real(0))
      throw std::invalid_argument("composite FAC reaction must be finite and non-negative");
    return detail::fac_build_contract(request, options, reaction, lane);
  }

  std::string_view exact_prepared_contract() const noexcept { return exact_prepared_contract_; }
  int n_levels() const noexcept { return static_cast<int>(levels_.size()); }
  int maximum_iterations() const noexcept {
    if (newton_workspace_)
      return newton_workspace_->options().max_iterations;
    if (linear_boundary_workspace_)
      return linear_boundary_workspace_->options().max_iterations;
    return options_.max_iters;
  }
  field_type& rhs_level(int level) { return levels_.at(static_cast<std::size_t>(level))->rhs; }
  const field_type& rhs_level(int level) const {
    return levels_.at(static_cast<std::size_t>(level))->rhs;
  }
  field_type& phi_level(int level) { return levels_.at(static_cast<std::size_t>(level))->phi; }
  const field_type& phi_level(int level) const {
    return levels_.at(static_cast<std::size_t>(level))->phi;
  }
  const SolveReport& last_solve_report() const noexcept { return last_report_; }
  bool borrows_execution_lane(const ExecutionLane& lane) const noexcept { return lane_ == &lane; }

  void install_newton(FieldNewtonOptions options) {
    if (newton_workspace_)
      throw std::logic_error("composite FAC Newton authority is already installed");
    validate_field_newton_options(options);
    const auto layouts = newton_layouts_();
    const auto masks = active_masks_();
    const auto measures = level_cell_measures_();
    newton_workspace_.emplace(layouts, masks, measures, options);
    linear_boundary_workspace_.reset();
    prepare_dynamic_views_();
  }

  void install_boundary_kernel(CompiledFieldBoundaryKernel<Dim> kernel) {
    if (boundary_kernel_)
      throw std::logic_error("composite FAC boundary kernel is already installed");
    kernel.validate();
    boundary_kernel_ = std::move(kernel);
    boundary_contexts_.clear();
    if (!newton_workspace_ && !boundary_kernel_->observes_iteration) {
      const FieldNewtonOptions options = linear_boundary_newton_options_();
      const auto layouts = newton_layouts_();
      const auto masks = active_masks_();
      const auto measures = level_cell_measures_();
      linear_boundary_workspace_.emplace(layouts, masks, measures, options);
    }
    prepare_dynamic_views_();
  }

  void set_boundary_contexts(std::vector<FieldBoundaryExecutionContext<Dim>> contexts) {
    if (!boundary_kernel_)
      throw std::logic_error("composite FAC has no compiled dynamic boundary kernel");
    if (contexts.size() != levels_.size())
      throw std::invalid_argument(
          "composite FAC requires one dynamic boundary context per live AMR level");
    for (const FieldBoundaryExecutionContext<Dim>& context : contexts)
      if (context.failure == nullptr)
        throw std::invalid_argument(
            "composite FAC dynamic boundary requires fallible execution channels");
    boundary_contexts_ = std::move(contexts);
  }

  void install_nullspace(FieldNullspacePlan<Dim> plan,
                         std::vector<PreparedVectorDistribution<Dim>> distributions) {
    if (nullspace_workspace_)
      throw std::logic_error("composite FAC nullspace authority is already installed");
    if (distributions.size() != levels_.size())
      throw std::invalid_argument(
          "composite FAC nullspace authority requires one distribution per hierarchy level");
    if (singular_() != !plan.empty())
      throw std::invalid_argument(
          "composite FAC nullspace plan disagrees with the prepared operator kernel");

    std::vector<const MultiFab<Dim>*> rhs_layouts;
    std::vector<MultiFab<Dim>*> candidates;
    rhs_layouts.reserve(levels_.size());
    candidates.reserve(levels_.size());
    for (const auto& level : levels_) {
      rhs_layouts.push_back(&level->rhs);
      candidates.push_back(&level->phi);
    }
    auto workspace =
        std::make_unique<FieldNullspaceWorkspace<Dim>>(plan, rhs_layouts, distributions, *lane_);
    FieldNullspacePlan<Dim> coarse_plan = coarse_correction_plan_(plan);
    coarse_solver_->install_nullspace(std::move(coarse_plan), distributions.front());

    nullspace_rhs_ = std::move(rhs_layouts);
    nullspace_candidates_ = std::move(candidates);
    nullspace_workspace_ = std::move(workspace);
  }

  SolveReport solve() {
    if (!nullspace_workspace_)
      throw std::logic_error("composite FAC solve has no prepared nullspace authority");
    try {
      nullspace_workspace_->require_compatible(nullspace_rhs_);
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
    nullspace_workspace_->apply_gauge(nullspace_candidates_);

    if (newton_workspace_ || boundary_kernel_)
      return solve_dynamic_();

    compute_composite_residual_();
    const Real reference = composite_residual_norm_();
    SolveReport report;
    report.evaluations = 1;
    if (!std::isfinite(static_cast<double>(reference))) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                         "composite_fac_non_finite_initial_residual");
      last_report_ = report;
      return last_report_;
    }
    report.reference_residual_norm = reference;
    report.residual_norm = reference;
    report.rel_residual = reference > Real(0) ? Real(1) : Real(0);
    const Real stop = std::max(options_.abs_tol, options_.rel_tol * reference);
    if (reference <= stop) {
      fill_all_ghosts_();
      report.mark_solved("composite_fac_initial_residual");
      last_report_ = report;
      return last_report_;
    }

    const int pre = (options_.fine_sweeps + 1) / 2;
    const int post = options_.fine_sweeps / 2;
    for (int iteration = 0; iteration < options_.max_iters; ++iteration) {
      for (std::size_t level = 1; level < levels_.size(); ++level)
        smooth_level_(level, pre);

      compute_composite_residual_();
      restrict_residual_tower_();
      coarse_solver_->phi().set_val(Real(0));
      copy_scalar_valid(levels_.front()->residual, coarse_solver_->rhs());
      const SolveReport coarse_report = coarse_solver_->solve();
      report.evaluations += coarse_report.evaluations;
      if (!coarse_report.solved()) {
        report.iters = iteration;
        report.residual_norm = composite_residual_norm_();
        report.rel_residual = report.residual_norm / reference;
        report.mark_failed(coarse_report.status, SolveAction::kFailRun,
                           "composite_fac_coarse_correction_failed");
        last_report_ = report;
        return last_report_;
      }

      copy_scalar_valid(coarse_solver_->phi(), levels_.front()->correction);
      report.step_norm = global_norm_inf_(levels_.front()->correction);
      add_uncovered_(*levels_.front(), levels_.front()->correction);
      prolong_correction_tower_();
      for (std::size_t level = 1; level < levels_.size(); ++level)
        smooth_level_(level, post);
      average_solution_down_();
      nullspace_workspace_->apply_gauge(nullspace_candidates_);

      compute_composite_residual_();
      ++report.evaluations;
      report.iters = iteration + 1;
      report.residual_norm = composite_residual_norm_();
      report.rel_residual = report.residual_norm / reference;
      if (!std::isfinite(static_cast<double>(report.residual_norm))) {
        report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                           "composite_fac_non_finite_iteration");
        last_report_ = report;
        return last_report_;
      }
      if (report.residual_norm <= stop) {
        fill_all_ghosts_();
        report.mark_solved("composite_fac_converged");
        last_report_ = report;
        return last_report_;
      }
    }

    report.mark_failed(SolveStatus::kIterationLimit, SolveAction::kFailRun,
                       "composite_fac_iteration_limit");
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
    field_type residual_operator_view;
    field_type direction_operator_view;
    field_type covered;
    field_type active;
    HaloSchedule<Dim> halo_schedule;
    PreparedPhysicalBoundary<Dim> physical_boundary;
    PreparedPhysicalBoundary<Dim> homogeneous_physical_boundary;

    Level(const EllipticBuildRequest<Dim>& request, bool full_domain)
        : geometry(request.geometry),
          boundary(request.boundary),
          phi(request.boxes, request.distribution, request.local_rank, 1,
              detail::unit_ghosts<Dim>()),
          rhs(request.boxes, request.distribution, request.local_rank, 1, Extent<Dim>{}),
          residual(request.boxes, request.distribution, request.local_rank, 1, Extent<Dim>{}),
          scratch(request.boxes, request.distribution, request.local_rank, 1, Extent<Dim>{}),
          correction(request.boxes, request.distribution, request.local_rank, 1,
                     detail::unit_ghosts<Dim>()),
          residual_operator_view(request.boxes, request.distribution, request.local_rank, 1,
                                 detail::unit_ghosts<Dim>()),
          direction_operator_view(request.boxes, request.distribution, request.local_rank, 1,
                                  detail::unit_ghosts<Dim>()),
          covered(request.boxes, request.distribution, request.local_rank, 1, Extent<Dim>{}),
          active(request.boxes, request.distribution, request.local_rank, 1, Extent<Dim>{}),
          halo_schedule(prepare_halo_schedule(
              phi, geometry.domain(), boundary.topology(),
              full_domain ? HaloLayoutCoverage::full_domain : HaloLayoutCoverage::sparse_level,
              detail::exact_halo_budget(request.boxes, geometry.domain()))),
          physical_boundary(prepare_physical_boundary(geometry.domain(), detail::unit_ghosts<Dim>(),
                                                      boundary,
                                                      detail::exact_boundary_budget<Dim>())),
          homogeneous_physical_boundary(
              prepare_physical_boundary(geometry.domain(), detail::unit_ghosts<Dim>(),
                                        detail::boundary_for_geometry(boundary, geometry, true),
                                        detail::exact_boundary_budget<Dim>())) {
      if (halo_schedule.has_remote_jobs())
        throw std::invalid_argument(
            "composite FAC distributed halo transport is not a registered capability");
      phi.set_val(Real(0));
      rhs.set_val(Real(0));
      residual.set_val(Real(0));
      scratch.set_val(Real(0));
      correction.set_val(Real(0));
      residual_operator_view.set_val(Real(0));
      direction_operator_view.set_val(Real(0));
      covered.set_val(Real(0));
      active.set_val(Real(1));
    }
  };

  struct Connection {
    ::pops::amr::RefinementRatio<Dim> ratio;
    std::vector<fac_detail::InjectionTransfer<Dim>> coarse_fine_phi;
    std::vector<fac_detail::InjectionTransfer<Dim>> coarse_fine_residual_view;
    std::vector<fac_detail::InjectionTransfer<Dim>> coarse_fine_direction_view;
    std::vector<fac_detail::CellTransfer<Dim>> residual_restriction;
    std::vector<fac_detail::CellTransfer<Dim>> solution_restriction;
    std::vector<fac_detail::CellTransfer<Dim>> direction_restriction;
    std::vector<fac_detail::CellTransfer<Dim>> correction_prolongation;
  };

  static FieldNullspacePlan<Dim> coarse_correction_plan_(
      const FieldNullspacePlan<Dim>& hierarchy_plan) {
    if (hierarchy_plan.empty())
      return {};
    FieldNullspacePlan<Dim> result;
    result.identity = hierarchy_plan.identity + ":fac-coarse-correction";
    result.layout_identity = hierarchy_plan.layout_identity + ":fac-coarse-correction";
    result.bases.reserve(hierarchy_plan.bases.size());
    result.gauges.reserve(hierarchy_plan.bases.size());
    for (const FieldNullspaceBasis<Dim>& source : hierarchy_plan.bases) {
      FieldNullspaceBasis<Dim> basis;
      basis.identity = source.identity;
      basis.provenance = source.provenance;
      basis.recipe_identity = source.recipe_identity + ":fac-coarse-correction";
      basis.field_component = source.field_component;
      if (!source.masks.empty()) {
        if (!source.masks.front())
          throw std::invalid_argument(
              "composite FAC coarse nullspace basis has no level-zero mask");
        basis.masks.push_back(source.masks.front());
      }
      basis.cell_measure.push_back(source.measure(0));
      result.gauges.push_back(FieldGaugeConstraint{basis.identity, Real(0)});
      result.bases.push_back(std::move(basis));
    }
    return result;
  }

  static void validate_request_(const request_type& request, const ExecutionLane& lane) {
    if (request.levels.empty() || request.ratios.size() + 1 != request.levels.size())
      throw std::invalid_argument(
          "composite FAC requires one ratio between each adjacent pair of levels");
    for (std::size_t level = 0; level < request.levels.size(); ++level) {
      const auto& current = request.levels[level];
      detail::validate_boundary(current.geometry, current.boundary);
      if (current.geometry.domain().empty() || current.boxes.empty() ||
          !current.distribution.matches_layout(current.boxes) ||
          !current.distribution.rank_space().contains(current.local_rank) ||
          current.distribution.rank_space().size() != static_cast<std::size_t>(lane.size()) ||
          current.distribution.rank_space().linear_rank(current.local_rank) !=
              static_cast<std::size_t>(lane.rank()) ||
          !current.boxes.is_disjoint_within(current.geometry.domain(), current.layout_budget))
        throw std::invalid_argument("composite FAC level has an invalid exact-ranked layout");
      if (level == 0 &&
          !current.boxes.tiles_exactly(current.geometry.domain(), current.layout_budget))
        throw std::invalid_argument("composite FAC coarse level must tile its complete domain");
      if (lane.size() > 1 && !current.distribution.replicated())
        throw std::invalid_argument("composite FAC currently requires replicated levels under MPI");
      for (int axis = 0; axis < Dim; ++axis)
        if (current.rhs_ghosts[axis] != 0 || current.phi_ghosts[axis] < 1)
          throw std::invalid_argument(
              "composite FAC requires a ghost-free RHS and one solution ghost");
      if (level == 0)
        continue;
      const auto& parent = request.levels[level - 1];
      const auto& ratio = request.ratios[level - 1];
      fac_detail::require_ratio(ratio);
      const Extent<Dim> ratio_value = detail::ratio_extent(ratio);
      if (current.geometry != parent.geometry.refine(ratio_value))
        throw std::invalid_argument(
            "composite FAC adjacent geometries do not match their exact refinement ratio");
      for (const Box<Dim>& patch : current.boxes.boxes())
        if (refine(coarsen(patch, ratio_value), ratio_value) != patch)
          throw std::invalid_argument(
              "composite FAC fine patches are not exactly aligned to their parent index space");
    }
  }

  void build_levels_(const request_type& request) {
    levels_.reserve(request.levels.size());
    for (std::size_t level = 0; level < request.levels.size(); ++level)
      levels_.push_back(std::make_unique<Level>(request.levels[level], level == 0));
  }

  void build_connections_(const std::vector<::pops::amr::RefinementRatio<Dim>>& ratios) {
    connections_.reserve(ratios.size());
    for (std::size_t parent_index = 0; parent_index < ratios.size(); ++parent_index) {
      Level& parent = *levels_[parent_index];
      Level& child = *levels_[parent_index + 1];
      Connection connection{ratios[parent_index], {}, {}, {}, {}, {}, {}, {}};
      mark_coverage_(parent, child, connection.ratio);
      prepare_connection_(parent, child, connection);
      connections_.push_back(std::move(connection));
    }
  }

  void mark_coverage_(Level& parent, const Level& child,
                      const ::pops::amr::RefinementRatio<Dim>& ratio) {
    for (const Box<Dim>& fine_patch : child.phi.layout().boxes()) {
      const Box<Dim> footprint = coarsen(fine_patch, detail::ratio_extent(ratio));
      for (std::size_t local = 0; local < parent.covered.local_size(); ++local) {
        const Box<Dim> region = parent.covered.box(local).intersect(footprint);
        if (!region.empty()) {
          for_each_cell(
              region, fac_detail::SetScalarKernel<Dim>{parent.covered.fab(local).view(), Real(1)});
          for_each_cell(region,
                        fac_detail::SetScalarKernel<Dim>{parent.active.fab(local).view(), Real(0)});
        }
      }
    }
    Kokkos::fence();
  }

  void prepare_connection_(Level& parent, Level& child, Connection& connection) {
    using Provider =
        ::pops::amr::transfer::TransferProvider<Dim, ::pops::amr::transfer::Centering::Cell>;
    const Provider restriction = Provider::conservative_restriction();
    const Provider prolongation = Provider::linear_prolongation();

    for (std::size_t child_local = 0; child_local < child.phi.local_size(); ++child_local) {
      const std::size_t child_global = child.phi.global_index(child_local);
      const Box<Dim>& fine_valid = child.phi.box(child_local);
      const Extent<Dim> ratio_value = detail::ratio_extent(connection.ratio);
      const Box<Dim> footprint = coarsen(fine_valid, ratio_value);
      std::int64_t restricted_cells = 0;
      for (std::size_t parent_local = 0; parent_local < parent.phi.local_size(); ++parent_local) {
        const Box<Dim> region = parent.phi.box(parent_local).intersect(footprint);
        if (region.empty())
          continue;
        restricted_cells += region.numPts();
        const auto child_residual_view =
            static_cast<const field_type&>(child.residual).fab(child_local).view();
        const auto child_phi_view =
            static_cast<const field_type&>(child.phi).fab(child_local).view();
        connection.residual_restriction.push_back(
            restriction.prepare(child_residual_view, parent.residual.fab(parent_local).view(),
                                region, connection.ratio));
        connection.solution_restriction.push_back(restriction.prepare(
            child_phi_view, parent.phi.fab(parent_local).view(), region, connection.ratio));
        connection.direction_restriction.push_back(restriction.prepare(
            static_cast<const field_type&>(child.correction).fab(child_local).view(),
            parent.correction.fab(parent_local).view(), region, connection.ratio));
      }
      if (restricted_cells != footprint.numPts())
        throw std::invalid_argument(
            "composite FAC fine footprint is not completely nested in its parent layout");

      std::int64_t prolonged_cells = 0;
      for (std::size_t parent_local = 0; parent_local < parent.correction.local_size();
           ++parent_local) {
        const Box<Dim> region =
            fine_valid.intersect(refine(parent.correction.box(parent_local), ratio_value));
        if (region.empty())
          continue;
        prolonged_cells += region.numPts();
        const auto parent_correction_view =
            static_cast<const field_type&>(parent.correction).fab(parent_local).view();
        connection.correction_prolongation.push_back(
            prolongation.prepare(parent_correction_view, child.correction.fab(child_local).view(),
                                 region, connection.ratio));
      }
      if (prolonged_cells != fine_valid.numPts())
        throw std::invalid_argument(
            "composite FAC correction prolongation does not cover a fine patch exactly");

      std::vector<Box<Dim>> pending{child.phi.fab(child_local).grown_box()};
      for (const Box<Dim>& valid : child.phi.layout().boxes())
        fac_detail::subtract_from_regions(pending, valid);
      for (const HaloJob<Dim>& halo : child.halo_schedule.canonical_jobs())
        if (halo.destination_box == child_global)
          fac_detail::subtract_from_regions(pending, halo.destination_region);

      for (std::size_t parent_local = 0; parent_local < parent.phi.local_size(); ++parent_local) {
        const Box<Dim> parent_reach = refine(parent.phi.fab(parent_local).grown_box(), ratio_value);
        std::vector<Box<Dim>> next;
        for (const Box<Dim>& region : pending) {
          const Box<Dim> destination = region.intersect(parent_reach);
          if (!destination.empty()) {
            const auto source = static_cast<const field_type&>(parent.phi).fab(parent_local).view();
            connection.coarse_fine_phi.push_back(fac_detail::InjectionTransfer<Dim>{
                source, child.phi.fab(child_local).view(), destination, connection.ratio, {}});
            const auto residual_source =
                static_cast<const field_type&>(parent.residual_operator_view)
                    .fab(parent_local)
                    .view();
            connection.coarse_fine_residual_view.push_back(fac_detail::InjectionTransfer<Dim>{
                residual_source,
                child.residual_operator_view.fab(child_local).view(),
                destination,
                connection.ratio,
                {}});
            const auto direction_source =
                static_cast<const field_type&>(parent.direction_operator_view)
                    .fab(parent_local)
                    .view();
            connection.coarse_fine_direction_view.push_back(fac_detail::InjectionTransfer<Dim>{
                direction_source,
                child.direction_operator_view.fab(child_local).view(),
                destination,
                connection.ratio,
                {}});
          }
          std::vector<Box<Dim>> remainder = fac_detail::subtract_box(region, destination);
          next.insert(next.end(), remainder.begin(), remainder.end());
        }
        pending = std::move(next);
      }
      if (!pending.empty())
        throw std::invalid_argument(
            "composite FAC could not prepare every coarse/fine ghost destination");
    }
  }

  void build_coarse_solver_(const EllipticBuildRequest<Dim>& coarse_request,
                            const ExecutionLane& lane) {
    GeometricMultigridOptions controls;
    controls.relative_tolerance = options_.coarse_rel_tol;
    controls.absolute_tolerance = options_.coarse_abs_tol;
    controls.maximum_cycles = options_.coarse_cycles;
    controls.reaction = reaction_;
    EllipticBuildRequest<Dim> correction_request = coarse_request;
    correction_request.boundary =
        detail::boundary_for_geometry(coarse_request.boundary, coarse_request.geometry, true);
    coarse_solver_ = std::make_unique<GeometricMG<Dim, MemorySpace>>(std::move(correction_request),
                                                                     lane, controls);
  }

  void fill_level_ghosts_(std::size_t level_index, field_type& field, bool interpolate_parent) {
    Level& level = *levels_.at(level_index);
    fill_boundary(field, level.halo_schedule);
    if (interpolate_parent && level_index > 0)
      fac_detail::execute_injections(connections_.at(level_index - 1).coarse_fine_phi);
    fill_physical_boundary(field, level.physical_boundary);
  }

  void fill_all_ghosts_() {
    for (std::size_t level = 0; level < levels_.size(); ++level)
      fill_level_ghosts_(level, levels_[level]->phi, true);
  }

  void smooth_level_(std::size_t level_index, int sweeps) {
    Level& level = *levels_.at(level_index);
    Real inverse_spacing_squared[Dim]{};
    Real diagonal = reaction_;
    for (int axis = 0; axis < Dim; ++axis) {
      const Real inverse = Real(1) / level.geometry.spacing(axis);
      inverse_spacing_squared[axis] = inverse * inverse;
      diagonal += Real(2) * inverse_spacing_squared[axis];
    }
    for (int sweep = 0; sweep < sweeps; ++sweep) {
      fill_level_ghosts_(level_index, level.phi, true);
      for (std::size_t local = 0; local < level.phi.local_size(); ++local) {
        const auto phi_view = static_cast<const field_type&>(level.phi).fab(local).view();
        const auto rhs_view = static_cast<const field_type&>(level.rhs).fab(local).view();
        const auto covered_view = static_cast<const field_type&>(level.covered).fab(local).view();
        fac_detail::MaskedJacobiKernel<Dim> kernel{level.scratch.fab(local).view(),
                                                   phi_view,
                                                   rhs_view,
                                                   covered_view,
                                                   {},
                                                   Real(1) / diagonal,
                                                   Real(2) / Real(3),
                                                   reaction_};
        for (int axis = 0; axis < Dim; ++axis)
          kernel.inverse_spacing_squared[axis] = inverse_spacing_squared[axis];
        for_each_cell(level.phi.box(local), kernel);
      }
      Kokkos::fence();
      copy_scalar_valid(level.scratch, level.phi);
    }
  }

  void compute_level_residual_(std::size_t level_index) {
    Level& level = *levels_.at(level_index);
    fill_level_ghosts_(level_index, level.phi, true);
    poisson_residual_valid(level.phi, level.rhs, level.geometry, level.residual, reaction_);
    for (std::size_t local = 0; local < level.residual.local_size(); ++local) {
      const auto covered_view = static_cast<const field_type&>(level.covered).fab(local).view();
      for_each_cell(level.residual.box(local), fac_detail::MaskResidualKernel<Dim>{
                                                   level.residual.fab(local).view(), covered_view});
    }
    Kokkos::fence();
  }

  void compute_composite_residual_() {
    for (std::size_t level = 0; level < levels_.size(); ++level)
      compute_level_residual_(level);
  }

  void restrict_residual_tower_() {
    for (std::size_t child = levels_.size(); child-- > 1;)
      fac_detail::execute_transfers(connections_.at(child - 1).residual_restriction);
  }

  void prolong_correction_tower_() {
    for (std::size_t parent = 0; parent < connections_.size(); ++parent) {
      Level& parent_level = *levels_[parent];
      Level& child_level = *levels_[parent + 1];
      fill_level_ghosts_(parent, parent_level.correction, false);
      child_level.correction.set_val(Real(0));
      fac_detail::execute_transfers(connections_[parent].correction_prolongation);
      saxpy(child_level.phi, Real(1), child_level.correction);
    }
  }

  void average_solution_down_() {
    for (std::size_t child = levels_.size(); child-- > 1;)
      fac_detail::execute_transfers(connections_.at(child - 1).solution_restriction);
  }

  void add_uncovered_(Level& level, const field_type& correction) {
    for (std::size_t local = 0; local < level.phi.local_size(); ++local) {
      const auto covered_view = static_cast<const field_type&>(level.covered).fab(local).view();
      for_each_cell(level.phi.box(local),
                    fac_detail::MaskedAddKernel<Dim>{level.phi.fab(local).view(),
                                                     correction.fab(local).view(), covered_view});
    }
    Kokkos::fence();
  }

  std::vector<const field_type*> newton_layouts_() const {
    std::vector<const field_type*> result;
    result.reserve(levels_.size());
    for (const auto& level : levels_)
      result.push_back(&level->phi);
    return result;
  }

  std::vector<const field_type*> active_masks_() const {
    std::vector<const field_type*> result;
    result.reserve(levels_.size());
    for (const auto& level : levels_)
      result.push_back(&level->active);
    return result;
  }

  std::vector<Real> level_cell_measures_() const {
    std::vector<Real> result;
    result.reserve(levels_.size());
    for (const auto& level : levels_) {
      Real measure = Real(1);
      for (int axis = 0; axis < Dim; ++axis)
        measure *= level->geometry.spacing(axis);
      result.push_back(measure);
    }
    return result;
  }

  void prepare_dynamic_views_() {
    candidate_view_.resize(levels_.size());
    dynamic_const_view_.resize(levels_.size());
    dynamic_mutable_view_.resize(levels_.size());
    for (std::size_t level = 0; level < levels_.size(); ++level)
      candidate_view_[level] = &levels_[level]->phi;
  }

  static void copy_valid_(const field_type& source, field_type& destination) {
    copy_scalar_valid(source, destination);
  }

  static void copy_grown_(const field_type& source, field_type& destination) {
    if (source.layout() != destination.layout() ||
        source.distribution() != destination.distribution() ||
        source.local_rank() != destination.local_rank() || source.ncomp() != 1 ||
        destination.ncomp() != 1 || source.ghosts() != destination.ghosts())
      throw std::invalid_argument("composite FAC dynamic operator views differ from candidates");
    for (std::size_t local = 0; local < source.local_size(); ++local) {
      const auto values = source.fab(local).view();
      const auto published = destination.fab(local).view();
      for_each_cell(source.fab(local).grown_box(),
                    [=] POPS_HD(const Index<Dim>& cell) { published(cell, 0) = values(cell, 0); });
    }
    Kokkos::fence();
  }

  void stage_iterate_(const nonlinear_hierarchy_type& iterate) {
    if (iterate.size() != levels_.size())
      throw std::invalid_argument("composite FAC nonlinear iterate has the wrong level count");
    for (std::size_t level = 0; level < levels_.size(); ++level)
      copy_valid_(iterate[level], levels_[level]->phi);
    average_solution_down_();
  }

  void stage_direction_(const nonlinear_hierarchy_type& direction) {
    if (direction.size() != levels_.size())
      throw std::invalid_argument("composite FAC nonlinear direction has the wrong level count");
    for (std::size_t level = 0; level < levels_.size(); ++level)
      copy_valid_(direction[level], levels_[level]->correction);
    for (std::size_t child = levels_.size(); child-- > 1;)
      fac_detail::execute_transfers(connections_.at(child - 1).direction_restriction);
  }

  FieldBoundaryExecutionContext<Dim>& boundary_context_at_(std::size_t level, int iteration) {
    if (boundary_contexts_.size() != levels_.size())
      throw std::logic_error("composite FAC dynamic boundary contexts are absent");
    boundary_contexts_[level].point.iteration = iteration;
    return boundary_contexts_[level];
  }

  void synchronize_boundary_failure_(FieldBoundaryExecutionContext<Dim>& context,
                                     const char* message) {
    Kokkos::fence();
    if (context.failure->synchronize_across_ranks(*lane_))
      throw std::runtime_error(message);
  }

  void fill_dynamic_residual_ghosts_(std::size_t level_index, int iteration) {
    Level& level = *levels_.at(level_index);
    copy_valid_(level.phi, level.residual_operator_view);
    fill_boundary(level.residual_operator_view, level.halo_schedule);
    if (level_index > 0)
      fac_detail::execute_injections(connections_.at(level_index - 1).coarse_fine_residual_view);
    fill_physical_boundary(level.residual_operator_view, level.physical_boundary);
    if (boundary_kernel_) {
      auto& context = boundary_context_at_(level_index, iteration);
      context.failure->reset();
      for (int face = 0; face < 2 * Dim; ++face)
        boundary_kernel_->prepare_residual_view(face, level.phi, level.residual_operator_view,
                                                level.geometry, context);
      synchronize_boundary_failure_(context,
                                    "composite FAC dynamic boundary residual failed collectively");
    }
  }

  void fill_dynamic_jvp_ghosts_(std::size_t level_index, int iteration) {
    Level& level = *levels_.at(level_index);
    copy_valid_(level.correction, level.direction_operator_view);
    fill_boundary(level.direction_operator_view, level.halo_schedule);
    if (level_index > 0)
      fac_detail::execute_injections(connections_.at(level_index - 1).coarse_fine_direction_view);
    fill_physical_boundary(level.direction_operator_view, level.homogeneous_physical_boundary);
    if (boundary_kernel_) {
      auto& context = boundary_context_at_(level_index, iteration);
      context.failure->reset();
      for (int face = 0; face < 2 * Dim; ++face)
        boundary_kernel_->prepare_jvp_view(face, level.phi, level.correction,
                                           level.direction_operator_view, level.geometry, context);
      synchronize_boundary_failure_(context,
                                    "composite FAC dynamic boundary JVP failed collectively");
    }
  }

  static void mask_covered_(Level& level, field_type& values) {
    for (std::size_t local = 0; local < values.local_size(); ++local) {
      const auto covered = static_cast<const field_type&>(level.covered).fab(local).view();
      for_each_cell(values.box(local),
                    fac_detail::MaskResidualKernel<Dim>{values.fab(local).view(), covered});
    }
    Kokkos::fence();
  }

  void evaluate_dynamic_residual_(const nonlinear_hierarchy_type& iterate,
                                  nonlinear_hierarchy_type& output, int iteration) {
    if (output.size() != levels_.size())
      throw std::invalid_argument("composite FAC nonlinear residual has the wrong level count");
    stage_iterate_(iterate);
    for (std::size_t level_index = 0; level_index < levels_.size(); ++level_index) {
      Level& level = *levels_[level_index];
      fill_dynamic_residual_ghosts_(level_index, iteration);
      poisson_residual_valid(level.residual_operator_view, level.rhs, level.geometry,
                             level.residual, reaction_);
      if (boundary_kernel_) {
        auto& context = boundary_context_at_(level_index, iteration);
        context.failure->reset();
        for (int face = 0; face < 2 * Dim; ++face)
          boundary_kernel_->add_residual(face, level.phi, level.residual, level.geometry, context);
        synchronize_boundary_failure_(context,
                                      "composite FAC dynamic residual closure failed collectively");
      }
      mask_covered_(level, level.residual);
      copy_valid_(level.residual, output[level_index]);
      dynamic_const_view_[level_index] = &output[level_index];
    }
    nullspace_workspace_->require_compatible(dynamic_const_view_);
  }

  void apply_dynamic_linearized_(const nonlinear_hierarchy_type& iterate,
                                 const nonlinear_hierarchy_type& direction,
                                 nonlinear_hierarchy_type& output, int iteration) {
    if (output.size() != levels_.size())
      throw std::invalid_argument("composite FAC nonlinear JVP has the wrong level count");
    stage_iterate_(iterate);
    stage_direction_(direction);
    for (std::size_t level_index = 0; level_index < levels_.size(); ++level_index) {
      Level& level = *levels_[level_index];
      fill_dynamic_jvp_ghosts_(level_index, iteration);
      apply_poisson_operator_valid(level.direction_operator_view, level.geometry, level.scratch,
                                   reaction_);
      if (boundary_kernel_) {
        auto& context = boundary_context_at_(level_index, iteration);
        context.failure->reset();
        for (int face = 0; face < 2 * Dim; ++face)
          boundary_kernel_->apply_jvp(face, level.phi, level.correction, level.scratch,
                                      level.geometry, context);
        synchronize_boundary_failure_(context,
                                      "composite FAC dynamic JVP closure failed collectively");
      }
      mask_covered_(level, level.scratch);
      copy_valid_(level.scratch, output[level_index]);
    }
  }

  void apply_dynamic_gauge_(nonlinear_hierarchy_type& values) {
    if (values.size() != levels_.size())
      throw std::invalid_argument("composite FAC nonlinear gauge has the wrong level count");
    for (std::size_t level = 0; level < levels_.size(); ++level)
      dynamic_mutable_view_[level] = &values[level];
    nullspace_workspace_->apply_gauge(dynamic_mutable_view_);
  }

  SolveReport solve_dynamic_() {
    if (boundary_kernel_ && boundary_contexts_.size() != levels_.size())
      throw std::logic_error("composite FAC dynamic boundary has no level-qualified contexts");
    if (boundary_kernel_ && boundary_kernel_->observes_iteration && !newton_workspace_)
      throw std::logic_error(
          "iterate-dependent composite FAC boundary requires a prepared Newton authority");
    auto* workspace = newton_workspace_ ? &*newton_workspace_ : &*linear_boundary_workspace_;
    SolveReport report;
    try {
      report = workspace->solve(
          candidate_view_,
          [this](const nonlinear_hierarchy_type& iterate, nonlinear_hierarchy_type& residual,
                 int iteration) { evaluate_dynamic_residual_(iterate, residual, iteration); },
          [this](const nonlinear_hierarchy_type& iterate, const nonlinear_hierarchy_type& direction,
                 nonlinear_hierarchy_type& output, int iteration) {
            apply_dynamic_linearized_(iterate, direction, output, iteration);
          },
          [this](nonlinear_hierarchy_type& values) { apply_dynamic_gauge_(values); }, *lane_);
    } catch (const FieldNullspaceIncompatibleRhs& error) {
      report.mark_failed(SolveStatus::kIncompatibleRhs, SolveAction::kFailRun, error.what());
    } catch (const FieldNullspaceInvalidEvaluation& error) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun, error.what());
    }
    if (report.solved_value_available()) {
      average_solution_down_();
      nullspace_workspace_->apply_gauge(nullspace_candidates_);
      for (std::size_t level = 0; level < levels_.size(); ++level) {
        fill_dynamic_residual_ghosts_(level, report.iters);
        copy_grown_(levels_[level]->residual_operator_view, levels_[level]->phi);
      }
    }
    last_report_ = report;
    return last_report_;
  }

  FieldNewtonOptions linear_boundary_newton_options_() const {
    FieldNewtonOptions options;
    options.tolerance = std::max(options_.rel_tol,
                                 options_.abs_tol > Real(0) ? options_.abs_tol : options_.rel_tol);
    options.max_iterations = 1;
    options.linear_tolerance = options_.rel_tol;
    options.linear_max_iterations = std::max(1, options_.max_iters);
    options.restart = std::min(30, options.linear_max_iterations);
    validate_field_newton_options(options);
    return options;
  }

  Real global_norm_inf_(const field_type& field) const {
    return static_cast<Real>(all_reduce_max(static_cast<double>(norm_inf(field)), *lane_));
  }

  Real composite_residual_norm_() const {
    Real result = Real(0);
    for (const auto& level : levels_)
      result = std::max(result, norm_inf(level->residual));
    return static_cast<Real>(all_reduce_max(static_cast<double>(result), *lane_));
  }

  bool singular_() const noexcept {
    return detail::is_singular(levels_.front()->boundary, reaction_);
  }

  const ExecutionLane* lane_ = nullptr;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  CompositeFacOptions options_{};
  Real reaction_ = Real(0);
  std::vector<std::unique_ptr<Level>> levels_{};
  std::vector<Connection> connections_{};
  std::unique_ptr<GeometricMG<Dim, MemorySpace>> coarse_solver_{};
  std::vector<const MultiFab<Dim>*> nullspace_rhs_{};
  std::vector<MultiFab<Dim>*> nullspace_candidates_{};
  std::unique_ptr<FieldNullspaceWorkspace<Dim>> nullspace_workspace_{};
  std::optional<CompiledFieldBoundaryKernel<Dim>> boundary_kernel_{};
  std::vector<FieldBoundaryExecutionContext<Dim>> boundary_contexts_{};
  std::optional<nonlinear_workspace_type> newton_workspace_{};
  std::optional<nonlinear_workspace_type> linear_boundary_workspace_{};
  std::vector<field_type*> candidate_view_{};
  std::vector<const field_type*> dynamic_const_view_{};
  std::vector<field_type*> dynamic_mutable_view_{};
  std::string exact_prepared_contract_{};
  SolveReport last_report_{};
};

}  // namespace pops::elliptic::mg
