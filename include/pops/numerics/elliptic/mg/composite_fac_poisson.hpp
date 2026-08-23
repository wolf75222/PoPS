/// @file
/// @brief Exact compile-time-ranked composite FAC Poisson solver for nested Cartesian AMR.

#pragma once

#include <pops/amr/refinement_ratio.hpp>
#include <pops/amr/transfer/transfer_provider.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/halo_exchange.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/amr/partitioned_region_transfer.hpp>
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
  bool distributed_mpi = true;
  bool variable_diagonal = true;
  bool cross_tensor = false;
  bool embedded_boundary = true;

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

/// Composite multilevel correction over a nested Cartesian AMR hierarchy.
///
/// The cycle restricts the active residual from every refined level into the covered parent cells,
/// solves that composite correction with a true geometric V-cycle on the complete coarse level,
/// prolongs the correction through the hierarchy, relaxes each uncovered level and averages the
/// accepted fine solution back down. Sparse fine layouts are first-class. Replicated levels keep
/// the local TransferProvider path. Partitioned levels use HaloExchange for same-level ghosts and
/// RegionTransfer for remote coarse/fine gather, restriction, flux mismatch, and prolongation.
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
    bool needs_owned = false;
    for (const auto& level : levels_)
      needs_owned = needs_owned || level->halo_schedule.has_remote_jobs();
    for (const Connection& connection : connections_)
      needs_owned = needs_owned || connection.has_remote_transport();
    if (all_reduce_max(needs_owned ? 1L : 0L, lane) != 0) {
      owned_lane_.emplace(ExecutionLane::duplicate_world_collectively(
          "pops.elliptic.composite-fac.distributed"));
      lane_ = &*owned_lane_;
    }
    for (std::size_t level = 0; level < levels_.size(); ++level) {
      const bool remote =
          all_reduce_max(levels_[level]->halo_schedule.has_remote_jobs() ? 1L : 0L, *lane_) != 0;
      if (remote) {
        HaloExchangeContext context{};
        context.context_generation = level + 1;
        context.schedule_generation = level + 1;
        levels_[level]->halo_exchange.emplace(levels_[level]->halo_schedule, *lane_, context);
      }
    }
    for (Connection& connection : connections_)
      connection.attach_lane(*lane_);
    build_coarse_solver_(request.levels.front(), *lane_);
    try_prepare_fft_coarse_();
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
  bool fft_coarse_prepared() const noexcept { return static_cast<bool>(fft_coarse_); }
  bool used_fft_coarse() const noexcept { return used_fft_coarse_; }
  ::pops::elliptic::PoissonFftBottomKind fft_coarse_kind() const noexcept {
    return fft_coarse_ ? fft_coarse_->kind() : ::pops::elliptic::PoissonFftBottomKind::none;
  }
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
  bool has_remote_same_level_halo() const noexcept {
    return std::any_of(levels_.begin(), levels_.end(),
                       [](const auto& level) { return level->halo_schedule.has_remote_jobs(); });
  }
  bool has_remote_parent_gather() const noexcept {
    return std::any_of(connections_.begin(), connections_.end(), [](const Connection& connection) {
      return connection.gather && connection.gather->plan().has_remote_jobs();
    });
  }
  bool has_remote_fine_restriction() const noexcept {
    return std::any_of(connections_.begin(), connections_.end(), [](const Connection& connection) {
      return connection.restriction && connection.restriction->plan().has_remote_jobs();
    });
  }

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
    boundary_contexts_.reset();
    if (!newton_workspace_ && !boundary_kernel_->observes_iteration) {
      const FieldNewtonOptions options = linear_boundary_newton_options_();
      const auto layouts = newton_layouts_();
      const auto masks = active_masks_();
      const auto measures = level_cell_measures_();
      linear_boundary_workspace_.emplace(layouts, masks, measures, options);
    }
    prepare_dynamic_views_();
  }

  void set_boundary_contexts(std::shared_ptr<const PreparedFieldBoundaryContextSet<Dim>> contexts) {
    if (!boundary_kernel_)
      throw std::logic_error("composite FAC has no compiled dynamic boundary kernel");
    if (!contexts || contexts->size() != levels_.size())
      throw std::invalid_argument(
          "composite FAC requires one dynamic boundary context per live AMR level");
    for (const FieldBoundaryExecutionContext<Dim>& context : contexts->contexts())
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
    attach_composite_nullspace_support_(plan);

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

  void install_coefficient(int level, const field_type& conductivity) {
    Level& target = *levels_.at(static_cast<std::size_t>(level));
    WeightedPoissonFields<Dim, MemorySpace> probe;
    probe.coefficient = &conductivity;
    validate_weighted_poisson_fields(target.phi, probe, "composite FAC coefficient");
    if (!target.coefficient)
      target.coefficient.emplace(target.phi.layout(), target.phi.distribution(),
                                 target.phi.local_rank(), 1, detail::unit_ghosts<Dim>());
    copy_scalar_valid(conductivity, *target.coefficient);
    fill_coefficient_ghosts_(static_cast<std::size_t>(level));
    if (level == 0 && coarse_solver_)
      coarse_solver_->install_coefficient(*target.coefficient);
    if (level == 0) {
      fft_coarse_.reset();
      used_fft_coarse_ = false;
    }
    rebuild_weighted_flux_mismatches_();
  }

  void install_embedded_boundary(int level, const field_type& active,
                                 const field_type& inverse_volume,
                                 const field_type& aperture_lower,
                                 const field_type& aperture_upper) {
    Level& target = *levels_.at(static_cast<std::size_t>(level));
    WeightedPoissonFields<Dim, MemorySpace> probe;
    probe.inverse_volume = &inverse_volume;
    probe.aperture_lower = &aperture_lower;
    probe.aperture_upper = &aperture_upper;
    probe.active = &active;
    validate_weighted_poisson_fields(target.phi, probe, "composite FAC embedded boundary");
    if (nullspace_workspace_)
      throw std::logic_error(
          "composite FAC embedded-boundary install requires the metric before nullspace authority");
    if (target.inverse_volume)
      throw std::logic_error("composite FAC embedded-boundary authority is already installed");
    copy_scalar_valid(active, target.active);
    target.inverse_volume.emplace(inverse_volume.layout(), inverse_volume.distribution(),
                                  inverse_volume.local_rank(), 1, Extent<Dim>{});
    copy_scalar_valid(inverse_volume, *target.inverse_volume);
    target.aperture_lower.emplace(aperture_lower.layout(), aperture_lower.distribution(),
                                  aperture_lower.local_rank(), Dim, Extent<Dim>{});
    target.aperture_upper.emplace(aperture_upper.layout(), aperture_upper.distribution(),
                                  aperture_upper.local_rank(), Dim, Extent<Dim>{});
    copy_vector_valid_(aperture_lower, *target.aperture_lower);
    copy_vector_valid_(aperture_upper, *target.aperture_upper);
    if (level == 0 && coarse_solver_) {
      if (coarse_solver_->num_levels() > 1)
        rebuild_coarse_solver_single_level_();
      coarse_solver_->install_embedded_boundary(target.active, *target.inverse_volume,
                                                *target.aperture_lower, *target.aperture_upper);
    }
    if (level == 0) {
      fft_coarse_.reset();
      used_fft_coarse_ = false;
    }
    rebuild_weighted_flux_mismatches_();
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
    used_fft_coarse_ = false;

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
    // Re-solves start with ||R(0)|| already near the last stop. Scale rel_tol by
    // the zero-iterate forcing (masked RHS) so the floor cannot fall below roundoff.
    const Real forcing = composite_forcing_norm_();
    if (!std::isfinite(static_cast<double>(forcing))) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                         "composite_fac_non_finite_forcing");
      last_report_ = report;
      return last_report_;
    }
    const Real stop = std::max(options_.abs_tol, options_.rel_tol * std::max(reference, forcing));
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
      SolveReport coarse_report;
      if (fft_coarse_) {
        coarse_report = fft_coarse_->apply(levels_.front()->residual, levels_.front()->correction);
        used_fft_coarse_ = true;
        if (coarse_report.solved())
          fill_correction_ghosts_(0);
      } else {
        coarse_solver_->phi().set_val(Real(0));
        copy_scalar_valid(levels_.front()->residual, coarse_solver_->rhs());
        coarse_report = coarse_solver_->solve();
        if (coarse_report.solved())
          copy_scalar_valid(coarse_solver_->phi(), levels_.front()->correction);
      }
      report.evaluations += coarse_report.evaluations;
      if (!coarse_report.solved()) {
        report.iters = iteration;
        report.residual_norm = composite_residual_norm_();
        report.rel_residual = report.residual_norm / reference;
        report.mark_failed(
            coarse_report.status, SolveAction::kFailRun,
            std::string("composite_fac_coarse_correction_failed:") + coarse_report.reason +
                " rel=" + std::to_string(static_cast<double>(coarse_report.rel_residual)) +
                " iters=" + std::to_string(coarse_report.iters));
        last_report_ = report;
        return last_report_;
      }

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
    std::optional<field_type> coefficient{};
    std::optional<field_type> inverse_volume{};
    std::optional<field_type> aperture_lower{};
    std::optional<field_type> aperture_upper{};
    HaloSchedule<Dim> halo_schedule;
    PreparedPhysicalBoundary<Dim> physical_boundary;
    PreparedPhysicalBoundary<Dim> homogeneous_physical_boundary;
    PreparedPhysicalBoundary<Dim> coefficient_boundary;
    std::optional<HaloExchange<Dim, MemorySpace>> halo_exchange{};

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
                                        detail::exact_boundary_budget<Dim>())),
          coefficient_boundary(prepare_physical_boundary(
              geometry.domain(), detail::unit_ghosts<Dim>(),
              detail::coefficient_boundary_for_geometry(boundary, geometry),
              detail::exact_boundary_budget<Dim>())) {
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
    using transfer_job = ::pops::elliptic::amr::partitioned_transfer::RegionTransferJob<Dim>;
    using transfer_plan = ::pops::elliptic::amr::partitioned_transfer::RegionTransferPlan<Dim>;
    using transport_type = ::pops::elliptic::amr::partitioned_transfer::RegionTransport<Dim, MemorySpace>;

    struct ScratchPatch {
      std::size_t fine_patch = 0;
      Fab<Dim, MemorySpace> parent_staging{};
      Fab<Dim, MemorySpace> restricted{};
      Fab<Dim, MemorySpace> flux_increment{};
      Fab<Dim, MemorySpace> covered_staging{};
      std::vector<Box<Dim>> ghost_regions{};
    };

    Level* parent = nullptr;
    Level* child = nullptr;
    ::pops::amr::RefinementRatio<Dim> ratio;
    std::vector<fac_detail::QuadraticInterpolationTransfer<Dim>> coarse_fine_phi;
    std::vector<fac_detail::QuadraticInterpolationTransfer<Dim>> coarse_fine_residual_view;
    std::vector<fac_detail::QuadraticInterpolationTransfer<Dim>> coarse_fine_direction_view;
    std::vector<fac_detail::QuadraticInterpolationTransfer<Dim>> coarse_fine_correction;
    std::vector<fac_detail::CellTransfer<Dim>> residual_restriction;
    std::vector<fac_detail::CellTransfer<Dim>> solution_restriction;
    std::vector<fac_detail::CellTransfer<Dim>> direction_restriction;
    std::vector<fac_detail::CellTransfer<Dim>> correction_prolongation;
    std::vector<fac_detail::FluxMismatchTransfer<Dim>> flux_mismatch;
    std::vector<fac_detail::FluxMismatchTransfer<Dim>> dynamic_flux_mismatch_residual;
    std::vector<fac_detail::FluxMismatchTransfer<Dim>> dynamic_flux_mismatch_direction;
    std::vector<ScratchPatch> scratch{};
    std::vector<std::size_t> scratch_by_fine_patch{};
    std::unique_ptr<transport_type> gather{};
    std::unique_ptr<transport_type> restriction{};
    std::unique_ptr<transport_type> flux{};
    std::optional<field_type> increment{};
    static constexpr std::size_t no_scratch = std::numeric_limits<std::size_t>::max();

    bool has_remote_transport() const noexcept {
      return (gather && gather->plan().has_remote_jobs()) ||
             (restriction && restriction->plan().has_remote_jobs());
    }

    void attach_lane(const ExecutionLane& lane) {
      if (gather)
        gather->attach_lane(lane);
      if (restriction)
        restriction->attach_lane(lane);
      if (flux)
        flux->attach_lane(lane);
    }

    ScratchPatch& scratch_for(std::size_t fine_patch) {
      const std::size_t local = scratch_by_fine_patch.at(fine_patch);
      if (local == no_scratch)
        throw std::out_of_range("composite FAC scratch patch is not local");
      return scratch.at(local);
    }

    void gather_parent(const field_type& source) {
      if (!gather)
        return;
      auto source_view = [&source](const transfer_job& job) -> FieldView<const Real, Dim> {
        return source.fab_global(job.source_patch).view();
      };
      auto destination_view = [this](const transfer_job& job) -> FieldView<Real, Dim> {
        return scratch_for(job.destination_patch).parent_staging.view();
      };
      gather->execute(source_view, destination_view);
      bool periodic[Dim]{};
      for (int axis = 0; axis < Dim; ++axis)
        periodic[axis] =
            parent->boundary.topology().is_periodic(Face<Dim>{axis, BoundarySide::lower});
      for (ScratchPatch& patch : scratch) {
        fac_detail::WrapStagingKernel<Dim> kernel{patch.parent_staging.view(),
                                                  parent->geometry.domain(),
                                                  patch.parent_staging.box(),
                                                  {}};
        for (int axis = 0; axis < Dim; ++axis)
          kernel.periodic[axis] = periodic[axis];
        for_each_cell(patch.parent_staging.box(), kernel);
      }
      Kokkos::fence();
    }

    void interpolate_remote(field_type& destination) {
      if (scratch.empty())
        return;
      const ::pops::amr::transfer::IndexMapping<Dim> mapping{parent->geometry.domain().lo,
                                                            child->geometry.domain().lo};
      for (ScratchPatch& patch : scratch) {
        const auto coarse = std::as_const(patch.parent_staging).view();
        auto fine = destination.fab_global(patch.fine_patch).view();
        for (const Box<Dim>& region : patch.ghost_regions)
          for_each_cell(region, fac_detail::QuadraticInterpolationTransfer<Dim>{
                                    coarse, fine, region, ratio, mapping, child->geometry.domain()});
      }
      Kokkos::fence();
    }

    void restrict_remote(const field_type& source, field_type& destination) {
      if (!restriction)
        return;
      const Real inverse_children = Real(1) / static_cast<Real>(ratio.child_count());
      for (ScratchPatch& patch : scratch) {
        const auto fine = source.fab_global(patch.fine_patch).view();
        auto coarse = patch.restricted.view();
        for_each_cell(patch.restricted.box(),
                      fac_detail::RestrictionKernel<Dim>{fine, coarse, parent->geometry.domain(),
                                                         child->geometry.domain(), ratio,
                                                         inverse_children});
      }
      Kokkos::fence();
      auto source_view = [this](const transfer_job& job) -> FieldView<const Real, Dim> {
        return std::as_const(scratch_for(job.source_patch).restricted).view();
      };
      auto destination_view = [&destination](const transfer_job& job) -> FieldView<Real, Dim> {
        return destination.fab_global(job.destination_patch).view();
      };
      restriction->execute(source_view, destination_view);
    }

    void prolong_remote(field_type& destination) {
      if (!gather)
        return;
      gather_parent(parent->correction);
      const Extent<Dim> ratio_value = detail::ratio_extent(ratio);
      for (ScratchPatch& patch : scratch) {
        const auto coarse = std::as_const(patch.parent_staging).view();
        auto fine = destination.fab_global(patch.fine_patch).view();
        const Box<Dim>& fine_valid = child->phi.layout()[patch.fine_patch];
        for (std::size_t parent_patch = 0; parent_patch < parent->phi.layout().size(); ++parent_patch) {
          if (parent->phi.contains_local(parent_patch) && child->phi.contains_local(patch.fine_patch) &&
              patch_owner(*parent, parent_patch) == patch_owner(*child, patch.fine_patch))
            continue;
          const Box<Dim> region =
              fine_valid.intersect(refine(parent->phi.layout()[parent_patch], ratio_value));
          if (region.empty())
            continue;
          fac_detail::LinearInterpolationKernel<Dim> kernel{
              coarse, fine, parent->geometry.domain(), child->geometry.domain(), ratio, {}};
          for (int axis = 0; axis < Dim; ++axis)
            kernel.periodic[axis] =
                parent->boundary.topology().is_periodic(Face<Dim>{axis, BoundarySide::lower});
          for_each_cell(region, kernel);
        }
      }
      Kokkos::fence();
    }

    void apply_remote_flux(const field_type& parent_field, const field_type& child_field,
                           field_type& parent_residual, Real sign) {
      if (!flux || !increment)
        return;
      gather_parent(parent_field);
      for (ScratchPatch& patch : scratch) {
        patch.flux_increment.set_val(Real(0));
        const Box<Dim> footprint = patch.restricted.box();
        const auto parent_view = std::as_const(patch.parent_staging).view();
        const auto fine_view = std::as_const(child_field).fab_global(patch.fine_patch).view();
        const auto covered = std::as_const(patch.covered_staging).view();
        auto increment_view = patch.flux_increment.view();
        for (int axis = 0; axis < Dim; ++axis) {
          Real transverse = Real(1);
          for (int transverse_axis = 0; transverse_axis < Dim; ++transverse_axis)
            if (transverse_axis != axis)
              transverse *= static_cast<Real>(ratio[transverse_axis]);
          const Real fine_face_weight = static_cast<Real>(ratio[axis]) / transverse;
          const Real inverse_spacing = Real(1) / parent->geometry.spacing(axis);
          const Real inverse_spacing_squared = inverse_spacing * inverse_spacing;
          for (const int child_side : {-1, 1}) {
            Box<Dim> interface = footprint;
            Index<Dim> geometry_shift{};
            if (child_side < 0) {
              --interface.lo[axis];
              interface.hi[axis] = interface.lo[axis];
            } else {
              ++interface.hi[axis];
              interface.lo[axis] = interface.hi[axis];
            }
            const Box<Dim>& parent_domain = parent->geometry.domain();
            if (parent->boundary.topology().is_periodic(Face<Dim>{axis, BoundarySide::lower})) {
              const int length = static_cast<int>(parent_domain.length(axis));
              if (length > 0 && interface.hi[axis] < parent_domain.lo[axis]) {
                geometry_shift[axis] = -length;
                interface.lo[axis] += length;
                interface.hi[axis] += length;
              } else if (length > 0 && interface.lo[axis] > parent_domain.hi[axis]) {
                geometry_shift[axis] = length;
                interface.lo[axis] -= length;
                interface.hi[axis] -= length;
              }
            }
            const Box<Dim> destination = interface.intersect(patch.flux_increment.box());
            if (destination.empty())
              continue;
            for_each_cell(destination, fac_detail::FluxMismatchTransfer<Dim>{
                                           parent_view, fine_view, increment_view, covered, destination,
                                           ratio, axis, child_side, inverse_spacing_squared,
                                           fine_face_weight, sign, geometry_shift});
          }
        }
      }
      Kokkos::fence();
      increment->set_val(Real(0));
      auto source_view = [this](const transfer_job& job) -> FieldView<const Real, Dim> {
        return std::as_const(scratch_for(job.source_patch).flux_increment).view();
      };
      auto destination_view = [this](const transfer_job& job) -> FieldView<Real, Dim> {
        return increment->fab_global(job.destination_patch).view();
      };
      flux->execute(source_view, destination_view);
      for (std::size_t local = 0; local < parent_residual.local_size(); ++local) {
        for_each_cell(parent_residual.box(local),
                      fac_detail::AddKernel<Dim>{parent_residual.fab(local).view(),
                                                 std::as_const(*increment).fab(local).view()});
        for_each_cell(parent_residual.box(local),
                      fac_detail::MaskResidualKernel<Dim>{
                          parent_residual.fab(local).view(),
                          std::as_const(parent->covered).fab(local).view()});
      }
      Kokkos::fence();
    }

    static Index<Dim> patch_owner(const Level& level, std::size_t patch) {
      return level.phi.distribution().replicated() ? level.phi.local_rank()
                                                   : level.phi.distribution().owner(patch);
    }
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
      Connection connection{};
      connection.ratio = ratios[parent_index];
      connection.parent = &parent;
      connection.child = &child;
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

  static Index<Dim> patch_owner_(const Level& level, std::size_t patch) {
    return level.phi.distribution().replicated() ? level.phi.local_rank()
                                                 : level.phi.distribution().owner(patch);
  }

  void prepare_connection_(Level& parent, Level& child, Connection& connection) {
    using Provider =
        ::pops::amr::transfer::TransferProvider<Dim, ::pops::amr::transfer::Centering::Cell>;
    using transfer_job = typename Connection::transfer_job;
    const Provider restriction = Provider::conservative_restriction();
    const Provider prolongation = Provider::linear_prolongation();
    const Extent<Dim> ratio_value = detail::ratio_extent(connection.ratio);
    const ::pops::amr::transfer::IndexMapping<Dim> mapping{parent.geometry.domain().lo,
                                                           child.geometry.domain().lo};
    connection.scratch_by_fine_patch.assign(child.phi.layout().size(), Connection::no_scratch);
    std::vector<std::vector<Box<Dim>>> remote_ghosts(child.phi.layout().size());
    std::vector<transfer_job> gather_jobs;
    std::vector<transfer_job> restriction_jobs;

    for (std::size_t fine_patch = 0; fine_patch < child.phi.layout().size(); ++fine_patch) {
      const Box<Dim>& fine_valid = child.phi.layout()[fine_patch];
      const Box<Dim> footprint = coarsen(fine_valid, ratio_value);
      const Box<Dim> staging = footprint.grow(2).intersect(parent.geometry.domain());
      const bool child_local = child.phi.contains_local(fine_patch);
      const std::size_t child_local_index =
          child_local ? child.phi.local_index_of(fine_patch) : 0;
      std::int64_t restricted_cells = 0;
      std::int64_t prolonged_cells = 0;
      for (std::size_t parent_patch = 0; parent_patch < parent.phi.layout().size(); ++parent_patch) {
        const Box<Dim>& parent_valid = parent.phi.layout()[parent_patch];
        const Box<Dim> restricted_region = parent_valid.intersect(footprint);
        if (!restricted_region.empty()) {
          restricted_cells += restricted_region.numPts();
          const bool parent_local = parent.phi.contains_local(parent_patch);
          if (child_local && parent_local) {
            const auto child_residual_view =
                static_cast<const field_type&>(child.residual).fab(child_local_index).view();
            const auto child_phi_view =
                static_cast<const field_type&>(child.phi).fab(child_local_index).view();
            connection.residual_restriction.push_back(
                restriction.prepare(child_residual_view, parent.residual.fab_global(parent_patch).view(),
                                    restricted_region, connection.ratio));
            connection.solution_restriction.push_back(restriction.prepare(
                child_phi_view, parent.phi.fab_global(parent_patch).view(), restricted_region,
                connection.ratio));
            connection.direction_restriction.push_back(restriction.prepare(
                static_cast<const field_type&>(child.correction).fab(child_local_index).view(),
                parent.correction.fab_global(parent_patch).view(), restricted_region,
                connection.ratio));
          } else if (patch_owner_(parent, parent_patch) != patch_owner_(child, fine_patch)) {
            restriction_jobs.push_back(transfer_job{fine_patch, parent_patch,
                                                    patch_owner_(child, fine_patch),
                                                    patch_owner_(parent, parent_patch),
                                                    restricted_region, restricted_region});
          }
        }
        const Box<Dim> prolonged_region = fine_valid.intersect(refine(parent_valid, ratio_value));
        if (!prolonged_region.empty()) {
          prolonged_cells += prolonged_region.numPts();
          if (child_local && parent.phi.contains_local(parent_patch)) {
            const auto parent_correction_view =
                static_cast<const field_type&>(parent.correction).fab_global(parent_patch).view();
            connection.correction_prolongation.push_back(
                prolongation.prepare(parent_correction_view,
                                     child.correction.fab(child_local_index).view(), prolonged_region,
                                     connection.ratio));
          }
        }
        const Box<Dim> gathered = staging.intersect(parent_valid);
        if (!gathered.empty() &&
            patch_owner_(parent, parent_patch) != patch_owner_(child, fine_patch)) {
          gather_jobs.push_back(transfer_job{parent_patch, fine_patch, patch_owner_(parent, parent_patch),
                                             patch_owner_(child, fine_patch), gathered, gathered});
        }
      }
      if (restricted_cells != footprint.numPts())
        throw std::invalid_argument(
            "composite FAC fine footprint is not completely nested in its parent layout");
      if (prolonged_cells != fine_valid.numPts())
        throw std::invalid_argument(
            "composite FAC correction prolongation does not cover a fine patch exactly");
      if (child_local)
        append_flux_mismatches_(parent, child, connection, child_local_index, footprint);

      const Box<Dim> grown = fine_valid.grow(1);
      std::vector<Box<Dim>> pending{grown.intersect(child.geometry.domain())};
      for (const Box<Dim>& halo : fac_detail::subtract_box(grown, child.geometry.domain())) {
        bool periodic_image = true;
        for (int axis = 0; axis < Dim; ++axis) {
          if (halo.hi[axis] < child.geometry.domain().lo[axis] ||
              halo.lo[axis] > child.geometry.domain().hi[axis])
            periodic_image = periodic_image && child.boundary.topology().is_periodic(
                                                   Face<Dim>{axis, BoundarySide::lower});
        }
        if (periodic_image && !halo.empty())
          pending.push_back(halo);
      }
      for (const Box<Dim>& valid : child.phi.layout().boxes())
        fac_detail::subtract_from_regions(pending, valid);
      for (const HaloJob<Dim>& halo : child.halo_schedule.canonical_jobs())
        if (halo.destination_box == fine_patch)
          fac_detail::subtract_from_regions(pending, halo.destination_region);

      const auto periodic_shift = [&](const Box<Dim>& region) {
        Index<Dim> shift{};
        for (int axis = 0; axis < Dim; ++axis) {
          if (!child.boundary.topology().is_periodic(Face<Dim>{axis, BoundarySide::lower}))
            continue;
          const int length = static_cast<int>(child.geometry.domain().length(axis));
          if (length <= 0)
            continue;
          if (region.hi[axis] < child.geometry.domain().lo[axis])
            shift[axis] = length;
          else if (region.lo[axis] > child.geometry.domain().hi[axis])
            shift[axis] = -length;
        }
        return shift;
      };
      const auto negate = [](const Index<Dim>& value) {
        Index<Dim> result{};
        for (int axis = 0; axis < Dim; ++axis)
          result[axis] = -value[axis];
        return result;
      };
      for (std::size_t parent_patch = 0; parent_patch < parent.phi.layout().size(); ++parent_patch) {
        const Box<Dim> parent_reach = refine(parent.phi.layout()[parent_patch], ratio_value);
        std::vector<Box<Dim>> next;
        for (const Box<Dim>& region : pending) {
          const Index<Dim> shift = periodic_shift(region);
          const Box<Dim> wrapped = region.shift(shift);
          const Box<Dim> matched = wrapped.intersect(parent_reach);
          const Box<Dim> destination = matched.empty() ? Box<Dim>{} : matched.shift(negate(shift));
          if (!destination.empty()) {
            if (child_local && parent.phi.contains_local(parent_patch)) {
              const auto source =
                  static_cast<const field_type&>(parent.phi).fab_global(parent_patch).view();
              connection.coarse_fine_phi.push_back({source, child.phi.fab(child_local_index).view(),
                                                    destination, connection.ratio, mapping,
                                                    child.geometry.domain()});
              const auto residual_source =
                  static_cast<const field_type&>(parent.residual_operator_view)
                      .fab_global(parent_patch)
                      .view();
              connection.coarse_fine_residual_view.push_back(
                  {residual_source, child.residual_operator_view.fab(child_local_index).view(),
                   destination, connection.ratio, mapping, child.geometry.domain()});
              const auto direction_source =
                  static_cast<const field_type&>(parent.direction_operator_view)
                      .fab_global(parent_patch)
                      .view();
              connection.coarse_fine_direction_view.push_back(
                  {direction_source, child.direction_operator_view.fab(child_local_index).view(),
                   destination, connection.ratio, mapping, child.geometry.domain()});
              const auto correction_source =
                  static_cast<const field_type&>(parent.correction).fab_global(parent_patch).view();
              connection.coarse_fine_correction.push_back(
                  {correction_source, child.correction.fab(child_local_index).view(), destination,
                   connection.ratio, mapping, child.geometry.domain()});
            } else if (patch_owner_(parent, parent_patch) != patch_owner_(child, fine_patch)) {
              remote_ghosts[fine_patch].push_back(destination);
              const Box<Dim> parent_stencil =
                  coarsen(wrapped, ratio_value).grow(1).intersect(parent.geometry.domain());
              const Box<Dim> extra = parent.phi.layout()[parent_patch].intersect(parent_stencil);
              if (!extra.empty()) {
                transfer_job extra_job{parent_patch, fine_patch, patch_owner_(parent, parent_patch),
                                       patch_owner_(child, fine_patch), extra, extra};
                bool seen = false;
                for (const transfer_job& job : gather_jobs)
                  if (job == extra_job) {
                    seen = true;
                    break;
                  }
                if (!seen)
                  gather_jobs.push_back(extra_job);
              }
            }
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

    if (!gather_jobs.empty() || !restriction_jobs.empty()) {
      std::vector<transfer_job> flux_jobs;
      flux_jobs.reserve(gather_jobs.size());
      for (const transfer_job& job : gather_jobs)
        flux_jobs.push_back(transfer_job{job.destination_patch, job.source_patch, job.destination_rank,
                                         job.source_rank, job.destination_region, job.source_region});
      const auto gather_budget = Connection::transfer_plan::budget_from_jobs(gather_jobs);
      const auto restriction_budget = Connection::transfer_plan::budget_from_jobs(restriction_jobs);
      const auto flux_budget = Connection::transfer_plan::budget_from_jobs(flux_jobs);
      connection.gather = std::make_unique<typename Connection::transport_type>(
          typename Connection::transfer_plan{
              parent.phi.rank_space(), parent.phi.local_rank(), 1, std::move(gather_jobs),
              gather_budget});
      connection.restriction = std::make_unique<typename Connection::transport_type>(
          typename Connection::transfer_plan{
              parent.phi.rank_space(), parent.phi.local_rank(), 1, std::move(restriction_jobs),
              restriction_budget});
      connection.flux = std::make_unique<typename Connection::transport_type>(
          typename Connection::transfer_plan{parent.phi.rank_space(), parent.phi.local_rank(), 1,
                                             std::move(flux_jobs), flux_budget});
      connection.increment.emplace(parent.residual.layout(), parent.residual.distribution(),
                                   parent.residual.local_rank(), 1, Extent<Dim>{});
      for (std::size_t fine_patch = 0; fine_patch < child.phi.layout().size(); ++fine_patch) {
        if (!child.phi.contains_local(fine_patch))
          continue;
        const Box<Dim>& valid = child.phi.layout()[fine_patch];
        const Box<Dim> restricted_box = coarsen(valid, ratio_value);
        const Box<Dim> staging = restricted_box.grow(2);
        typename Connection::ScratchPatch patch;
        patch.fine_patch = fine_patch;
        patch.parent_staging = Fab<Dim, MemorySpace>(staging, 1, Extent<Dim>{});
        patch.restricted = Fab<Dim, MemorySpace>(restricted_box, 1, Extent<Dim>{});
        patch.flux_increment = Fab<Dim, MemorySpace>(staging, 1, Extent<Dim>{});
        patch.covered_staging = Fab<Dim, MemorySpace>(staging, 1, Extent<Dim>{});
        patch.flux_increment.set_val(Real(0));
        patch.covered_staging.set_val(Real(0));
        for_each_cell(restricted_box,
                      fac_detail::SetScalarKernel<Dim>{patch.covered_staging.view(), Real(1)});
        patch.ghost_regions = std::move(remote_ghosts[fine_patch]);
        connection.scratch_by_fine_patch[fine_patch] = connection.scratch.size();
        connection.scratch.push_back(std::move(patch));
      }
    }
  }

  void append_flux_mismatches_(Level& parent, Level& child, Connection& connection,
                               std::size_t child_local, const Box<Dim>& footprint) {
    for (int axis = 0; axis < Dim; ++axis) {
      Real transverse_refinement = Real(1);
      for (int transverse = 0; transverse < Dim; ++transverse)
        if (transverse != axis)
          transverse_refinement *= static_cast<Real>(connection.ratio[transverse]);
      const Real fine_face_weight =
          static_cast<Real>(connection.ratio[axis]) / transverse_refinement;
      const Real inverse_spacing = Real(1) / parent.geometry.spacing(axis);
      const Real inverse_spacing_squared = inverse_spacing * inverse_spacing;

      for (const int child_side : {-1, 1}) {
        Box<Dim> interface = footprint;
        if (child_side < 0) {
          --interface.lo[axis];
          interface.hi[axis] = interface.lo[axis];
        } else {
          ++interface.hi[axis];
          interface.lo[axis] = interface.hi[axis];
        }
        Index<Dim> geometry_shift{};
        const Box<Dim>& parent_domain = parent.geometry.domain();
        if (parent.boundary.topology().is_periodic(Face<Dim>{axis, BoundarySide::lower})) {
          const int length = static_cast<int>(parent_domain.length(axis));
          if (length > 0 && interface.hi[axis] < parent_domain.lo[axis]) {
            geometry_shift[axis] = -length;
            interface.lo[axis] += length;
            interface.hi[axis] += length;
          } else if (length > 0 && interface.lo[axis] > parent_domain.hi[axis]) {
            geometry_shift[axis] = length;
            interface.lo[axis] -= length;
            interface.hi[axis] -= length;
          }
        }
        for (std::size_t parent_local = 0; parent_local < parent.phi.local_size(); ++parent_local) {
          const Box<Dim> destination = parent.phi.box(parent_local).intersect(interface);
          if (destination.empty())
            continue;
          const auto covered =
              static_cast<const field_type&>(parent.covered).fab(parent_local).view();
          const auto append = [&](const field_type& parent_field, const field_type& child_field,
                                  field_type& residual,
                                  std::vector<fac_detail::FluxMismatchTransfer<Dim>>& transfers,
                                  Real sign) {
            fac_detail::FluxMismatchTransfer<Dim> transfer{
                static_cast<const field_type&>(parent_field).fab(parent_local).view(),
                static_cast<const field_type&>(child_field).fab(child_local).view(),
                residual.fab(parent_local).view(), covered, destination, connection.ratio, axis,
                child_side, inverse_spacing_squared, fine_face_weight, sign, geometry_shift};
            if (parent.coefficient)
              transfer.parent_coefficient =
                  std::as_const(*parent.coefficient).fab(parent_local).view();
            if (child.coefficient)
              transfer.fine_coefficient =
                  std::as_const(*child.coefficient).fab(child_local).view();
            if (parent.aperture_lower)
              transfer.parent_aperture_lower =
                  std::as_const(*parent.aperture_lower).fab(parent_local).view();
            if (parent.aperture_upper)
              transfer.parent_aperture_upper =
                  std::as_const(*parent.aperture_upper).fab(parent_local).view();
            if (child.aperture_lower)
              transfer.fine_aperture_lower =
                  std::as_const(*child.aperture_lower).fab(child_local).view();
            if (child.aperture_upper)
              transfer.fine_aperture_upper =
                  std::as_const(*child.aperture_upper).fab(child_local).view();
            if (parent.inverse_volume)
              transfer.parent_inverse_volume =
                  std::as_const(*parent.inverse_volume).fab(parent_local).view();
            transfers.push_back(transfer);
          };
          append(parent.phi, child.phi, parent.residual, connection.flux_mismatch, Real(1));
          append(parent.residual_operator_view, child.residual_operator_view, parent.residual,
                 connection.dynamic_flux_mismatch_residual, Real(1));
          append(parent.direction_operator_view, child.direction_operator_view, parent.scratch,
                 connection.dynamic_flux_mismatch_direction, Real(-1));
        }
      }
    }
  }

  void rebuild_weighted_flux_mismatches_() {
    for (Connection& connection : connections_) {
      connection.flux_mismatch.clear();
      connection.dynamic_flux_mismatch_residual.clear();
      connection.dynamic_flux_mismatch_direction.clear();
    }
    for (std::size_t parent = 0; parent < connections_.size(); ++parent) {
      Level& parent_level = *levels_[parent];
      Level& child_level = *levels_[parent + 1];
      Connection& connection = connections_[parent];
      const Extent<Dim> ratio_value = detail::ratio_extent(connection.ratio);
      for (std::size_t local = 0; local < child_level.phi.local_size(); ++local) {
        const Box<Dim> footprint = coarsen(child_level.phi.box(local), ratio_value);
        append_flux_mismatches_(parent_level, child_level, connection, local, footprint);
      }
    }
  }

  void build_coarse_solver_(const EllipticBuildRequest<Dim>& coarse_request,
                            const ExecutionLane& lane, bool allow_coarsening = true) {
    coarse_request_ = coarse_request;
    GeometricMultigridOptions controls;
    controls.relative_tolerance = options_.coarse_rel_tol;
    controls.absolute_tolerance = options_.coarse_abs_tol;
    controls.maximum_cycles = options_.coarse_cycles;
    controls.reaction = reaction_;
    controls.allow_coarsening = allow_coarsening;
    EllipticBuildRequest<Dim> correction_request = coarse_request;
    correction_request.boundary =
        detail::boundary_for_geometry(coarse_request.boundary, coarse_request.geometry, true);
    coarse_solver_ = std::make_unique<GeometricMG<Dim, MemorySpace>>(std::move(correction_request),
                                                                     lane, controls);
  }

  void rebuild_coarse_solver_single_level_() {
    if (!coarse_request_)
      throw std::logic_error("composite FAC coarse request is not prepared");
    build_coarse_solver_(*coarse_request_, *lane_, false);
    if (levels_.front()->coefficient)
      coarse_solver_->install_coefficient(*levels_.front()->coefficient);
    if (nullspace_workspace_) {
      const auto& distribution = levels_.front()->phi.distribution();
      coarse_solver_->install_nullspace(
          coarse_correction_plan_(nullspace_workspace_->plan()),
          distribution.replicated() ? PreparedVectorDistribution<Dim>::replicated()
                                    : PreparedVectorDistribution<Dim>::distributed());
    }
    try_prepare_fft_coarse_();
  }

  void try_prepare_fft_coarse_() {
    fft_coarse_.reset();
    used_fft_coarse_ = false;
    if (!coarse_request_ || lane_ == nullptr)
      return;
    const Level& coarse = *levels_.front();
    EllipticBuildRequest<Dim> request = *coarse_request_;
    request.boundary =
        detail::boundary_for_geometry(request.boundary, request.geometry, true);
    request.rhs_ghosts = {};
    request.phi_ghosts = detail::unit_ghosts<Dim>();
    request.layout_budget = detail::exact_layout_budget(request.boxes);
    fft_coarse_ = ::pops::elliptic::PoissonFftMultiFabAdapter<Dim>::try_make(
        request, *lane_, reaction_, coarse.coefficient.has_value(),
        coarse.inverse_volume.has_value());
  }

  void same_level_fill_(Level& level, field_type& field) {
    if (level.halo_exchange)
      level.halo_exchange->execute(field, *lane_);
    else
      fill_boundary(field, level.halo_schedule);
  }

  void fill_level_ghosts_(std::size_t level_index, field_type& field, bool interpolate_parent) {
    Level& level = *levels_.at(level_index);
    // Quadratic C/F interpolation reads the parent one-cell stencil, including its patch ghosts.
    // Refresh that hierarchy first because a preceding coarse correction changes parent valid data.
    if (interpolate_parent && level_index > 0)
      fill_level_ghosts_(level_index - 1, levels_[level_index - 1]->phi, true);
    same_level_fill_(level, field);
    if (interpolate_parent && level_index > 0) {
      fac_detail::execute_quadratic_interpolations(
          connections_.at(level_index - 1).coarse_fine_phi);
      connections_.at(level_index - 1).gather_parent(levels_[level_index - 1]->phi);
      connections_.at(level_index - 1).interpolate_remote(field);
    }
    fill_physical_boundary(field, level.physical_boundary);
  }

  void fill_all_ghosts_() {
    for (std::size_t level = 0; level < levels_.size(); ++level)
      fill_level_ghosts_(level, levels_[level]->phi, true);
  }

  void fill_correction_ghosts_(std::size_t level_index) {
    Level& level = *levels_.at(level_index);
    same_level_fill_(level, level.correction);
    if (level_index > 0) {
      fac_detail::execute_quadratic_interpolations(
          connections_.at(level_index - 1).coarse_fine_correction);
      connections_.at(level_index - 1).gather_parent(levels_[level_index - 1]->correction);
      connections_.at(level_index - 1).interpolate_remote(level.correction);
    }
    fill_physical_boundary(level.correction, level.homogeneous_physical_boundary);
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
    fields.covered = &level.covered;
    return fields;
  }

  bool uses_weighted_operator_(const Level& level) const noexcept {
    return level.coefficient.has_value() || level.inverse_volume.has_value();
  }

  void fill_coefficient_ghosts_(std::size_t level_index) {
    Level& level = *levels_.at(level_index);
    if (!level.coefficient)
      return;
    same_level_fill_(level, *level.coefficient);
    for (std::size_t local = 0; local < level.coefficient->local_size(); ++local)
      for_each_cell(level.coefficient->fab(local).grown_box(),
                    fac_detail::ExtrudeScalarValidToGhosts<Dim>{
                        level.coefficient->fab(local).view(), level.coefficient->box(local)});
    Kokkos::fence();
    fill_physical_boundary(*level.coefficient, level.coefficient_boundary);
  }

  static void copy_vector_valid_(const field_type& source, field_type& destination) {
    if (source.layout() != destination.layout() || source.ncomp() != destination.ncomp())
      throw std::invalid_argument("composite FAC vector copy requires one exact layout");
    for (std::size_t local = 0; local < source.local_size(); ++local) {
      const auto in = source.fab(local).view();
      const auto out = destination.fab(local).view();
      const int components = source.ncomp();
      for_each_cell(source.box(local),
                    detail::CopyComponentsKernel<Dim, decltype(in), decltype(out)>{in, out,
                                                                                     components});
    }
    Kokkos::fence();
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
      fill_coefficient_ghosts_(level_index);
      const field_type* effective_rhs = &level.rhs;
      if (level_index + 1 < levels_.size()) {
        // An intermediate level is the coarse side of its child interface.  Its smoother must
        // use the composite RHS with the same coarse/fine flux replacement as the residual.
        copy_scalar_valid(level.rhs, level.residual);
        fill_level_ghosts_(level_index + 1, levels_[level_index + 1]->phi, true);
        fac_detail::execute_flux_mismatches(connections_.at(level_index).flux_mismatch);
        connections_.at(level_index).apply_remote_flux(
            level.phi, levels_[level_index + 1]->phi, level.residual, Real(1));
        effective_rhs = &level.residual;
      }
      if (uses_weighted_operator_(level)) {
        damped_jacobi_weighted_update_valid(level.phi, *effective_rhs, level.geometry, level.scratch,
                                            Real(2) / Real(3), reaction_,
                                            weighted_fields_(level));
      } else {
        for (std::size_t local = 0; local < level.phi.local_size(); ++local) {
          const auto phi_view = static_cast<const field_type&>(level.phi).fab(local).view();
          const auto rhs_view = static_cast<const field_type&>(*effective_rhs).fab(local).view();
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
      }
      copy_scalar_valid(level.scratch, level.phi);
    }
  }

  void compute_level_residual_(std::size_t level_index) {
    Level& level = *levels_.at(level_index);
    fill_level_ghosts_(level_index, level.phi, true);
    fill_coefficient_ghosts_(level_index);
    if (uses_weighted_operator_(level)) {
      weighted_poisson_residual_valid(level.phi, level.rhs, level.geometry, level.residual,
                                      reaction_, weighted_fields_(level));
    } else {
      poisson_residual_valid(level.phi, level.rhs, level.geometry, level.residual, reaction_);
      for (std::size_t local = 0; local < level.residual.local_size(); ++local) {
        const auto covered_view = static_cast<const field_type&>(level.covered).fab(local).view();
        for_each_cell(level.residual.box(local),
                      fac_detail::MaskResidualKernel<Dim>{level.residual.fab(local).view(),
                                                          covered_view});
      }
      Kokkos::fence();
    }
  }

  void compute_composite_residual_() {
    for (std::size_t level = 0; level < levels_.size(); ++level)
      compute_level_residual_(level);
    for (Connection& connection : connections_) {
      fac_detail::execute_flux_mismatches(connection.flux_mismatch);
      connection.apply_remote_flux(connection.parent->phi, connection.child->phi,
                                   connection.parent->residual, Real(1));
    }
  }

  void restrict_residual_tower_() {
    for (std::size_t child = levels_.size(); child-- > 1;) {
      fac_detail::execute_transfers(connections_.at(child - 1).residual_restriction);
      connections_.at(child - 1).restrict_remote(levels_[child]->residual,
                                                 levels_[child - 1]->residual);
    }
  }

  void prolong_correction_tower_() {
    for (std::size_t parent = 0; parent < connections_.size(); ++parent) {
      Level& parent_level = *levels_[parent];
      Level& child_level = *levels_[parent + 1];
      fill_correction_ghosts_(parent);
      child_level.correction.set_val(Real(0));
      fac_detail::execute_transfers(connections_[parent].correction_prolongation);
      connections_[parent].prolong_remote(child_level.correction);
      saxpy(child_level.phi, Real(1), child_level.correction);
    }
  }

  void average_solution_down_() {
    for (std::size_t child = levels_.size(); child-- > 1;) {
      fac_detail::execute_transfers(connections_.at(child - 1).solution_restriction);
      connections_.at(child - 1).restrict_remote(levels_[child]->phi, levels_[child - 1]->phi);
    }
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

  void attach_composite_nullspace_support_(FieldNullspacePlan<Dim>& plan) const {
    if (plan.empty())
      return;
    const std::vector<Real> measures = level_cell_measures_();
    std::vector<std::shared_ptr<const MultiFab<Dim>>> coverage;
    coverage.reserve(levels_.size());
    for (const auto& level : levels_) {
      auto mask = std::make_shared<MultiFab<Dim>>(
          level->active.layout(), level->active.distribution(), level->active.local_rank(), 1,
          Extent<Dim>{});
      copy_scalar_valid(level->active, *mask);
      coverage.push_back(std::move(mask));
    }
    for (FieldNullspaceBasis<Dim>& basis : plan.bases) {
      basis.coverage = coverage;
      basis.cell_measure = measures;
    }
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
                    detail::CopyComponentsKernel<Dim, decltype(values), decltype(published)>{
                        values, published, 1});
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
    for (std::size_t child = levels_.size(); child-- > 1;) {
      fac_detail::execute_transfers(connections_.at(child - 1).direction_restriction);
      connections_.at(child - 1).restrict_remote(levels_[child]->correction,
                                                 levels_[child - 1]->correction);
    }
  }

  FieldBoundaryExecutionContext<Dim> boundary_context_at_(std::size_t level, int iteration) const {
    if (!boundary_contexts_ || boundary_contexts_->size() != levels_.size())
      throw std::logic_error("composite FAC dynamic boundary contexts are absent");
    return boundary_contexts_->view(level, iteration);
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
    same_level_fill_(level, level.residual_operator_view);
    if (level_index > 0) {
      fac_detail::execute_quadratic_interpolations(
          connections_.at(level_index - 1).coarse_fine_residual_view);
      connections_.at(level_index - 1).gather_parent(levels_[level_index - 1]->residual_operator_view);
      connections_.at(level_index - 1).interpolate_remote(level.residual_operator_view);
    }
    fill_physical_boundary(level.residual_operator_view, level.physical_boundary);
    if (boundary_kernel_) {
      auto context = boundary_context_at_(level_index, iteration);
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
    same_level_fill_(level, level.direction_operator_view);
    if (level_index > 0) {
      fac_detail::execute_quadratic_interpolations(
          connections_.at(level_index - 1).coarse_fine_direction_view);
      connections_.at(level_index - 1).gather_parent(levels_[level_index - 1]->direction_operator_view);
      connections_.at(level_index - 1).interpolate_remote(level.direction_operator_view);
    }
    fill_physical_boundary(level.direction_operator_view, level.homogeneous_physical_boundary);
    if (boundary_kernel_) {
      auto context = boundary_context_at_(level_index, iteration);
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
      fill_coefficient_ghosts_(level_index);
      if (uses_weighted_operator_(level)) {
        weighted_poisson_residual_valid(level.residual_operator_view, level.rhs, level.geometry,
                                        level.residual, reaction_, weighted_fields_(level));
      } else {
        poisson_residual_valid(level.residual_operator_view, level.rhs, level.geometry,
                               level.residual, reaction_);
      }
      if (boundary_kernel_) {
        auto context = boundary_context_at_(level_index, iteration);
        context.failure->reset();
        for (int face = 0; face < 2 * Dim; ++face)
          boundary_kernel_->add_residual(face, level.phi, level.residual, level.geometry, context);
        synchronize_boundary_failure_(context,
                                      "composite FAC dynamic residual closure failed collectively");
      }
      mask_covered_(level, level.residual);
    }
    for (Connection& connection : connections_) {
      fac_detail::execute_flux_mismatches(connection.dynamic_flux_mismatch_residual);
      connection.apply_remote_flux(connection.parent->residual_operator_view,
                                   connection.child->residual_operator_view,
                                   connection.parent->residual, Real(1));
    }
    for (std::size_t level_index = 0; level_index < levels_.size(); ++level_index) {
      Level& level = *levels_[level_index];
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
      fill_coefficient_ghosts_(level_index);
      if (uses_weighted_operator_(level)) {
        apply_weighted_poisson_operator_valid(level.direction_operator_view, level.geometry,
                                              level.scratch, reaction_, weighted_fields_(level));
      } else {
        apply_poisson_operator_valid(level.direction_operator_view, level.geometry, level.scratch,
                                     reaction_);
      }
      if (boundary_kernel_) {
        auto context = boundary_context_at_(level_index, iteration);
        context.failure->reset();
        for (int face = 0; face < 2 * Dim; ++face)
          boundary_kernel_->apply_jvp(face, level.phi, level.correction, level.scratch,
                                      level.geometry, context);
        synchronize_boundary_failure_(context,
                                      "composite FAC dynamic JVP closure failed collectively");
      }
      mask_covered_(level, level.scratch);
    }
    for (Connection& connection : connections_) {
      fac_detail::execute_flux_mismatches(connection.dynamic_flux_mismatch_direction);
      connection.apply_remote_flux(connection.parent->direction_operator_view,
                                   connection.child->direction_operator_view,
                                   connection.parent->scratch, Real(-1));
    }
    for (std::size_t level_index = 0; level_index < levels_.size(); ++level_index) {
      Level& level = *levels_[level_index];
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
    if (boundary_kernel_ && (!boundary_contexts_ || boundary_contexts_->size() != levels_.size()))
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

  Real composite_forcing_norm_() const {
    Real result = Real(0);
    for (const auto& level : levels_)
      result = std::max(result, reduce_active_norm_inf_local(level->rhs, 0, &level->active));
    return static_cast<Real>(all_reduce_max(static_cast<double>(result), *lane_));
  }

  bool singular_() const noexcept {
    return detail::is_singular(levels_.front()->boundary, reaction_);
  }

  const ExecutionLane* lane_ = nullptr;
  std::optional<ExecutionLane> owned_lane_{};
  ExecutionLane::ImmutableBorrow lane_borrow_;
  CompositeFacOptions options_{};
  Real reaction_ = Real(0);
  std::vector<std::unique_ptr<Level>> levels_{};
  std::vector<Connection> connections_{};
  std::optional<EllipticBuildRequest<Dim>> coarse_request_{};
  std::unique_ptr<GeometricMG<Dim, MemorySpace>> coarse_solver_{};
  std::unique_ptr<::pops::elliptic::PoissonFftMultiFabAdapter<Dim>> fft_coarse_{};
  bool used_fft_coarse_ = false;
  std::vector<const MultiFab<Dim>*> nullspace_rhs_{};
  std::vector<MultiFab<Dim>*> nullspace_candidates_{};
  std::unique_ptr<FieldNullspaceWorkspace<Dim>> nullspace_workspace_{};
  std::optional<CompiledFieldBoundaryKernel<Dim>> boundary_kernel_{};
  std::shared_ptr<const PreparedFieldBoundaryContextSet<Dim>> boundary_contexts_{};
  std::optional<nonlinear_workspace_type> newton_workspace_{};
  std::optional<nonlinear_workspace_type> linear_boundary_workspace_{};
  std::vector<field_type*> candidate_view_{};
  std::vector<const field_type*> dynamic_const_view_{};
  std::vector<field_type*> dynamic_mutable_view_{};
  std::string exact_prepared_contract_{};
  SolveReport last_report_{};
};

}  // namespace pops::elliptic::mg
