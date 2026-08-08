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
                               const CompositeFacOptions& options, Real reaction) {
  ExactContractBuilder contract;
  contract.text("pops.elliptic.composite-fac-build")
      .scalar(std::uint32_t{2})
      .scalar(std::int32_t{Dim})
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
  static_assert(Dim >= 1 && Dim <= 3,
                "CompositeFacPoisson only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  using field_type = MultiFab<Dim, MemorySpace>;
  using request_type = CompositeFacBuildRequest<Dim>;

  CompositeFacPoisson(request_type request, CompositeFacOptions options = {},
                      Real reaction = Real(0))
      : options_(options), reaction_(reaction) {
    std::exception_ptr validation_error;
    try {
      detail::validate_fac_options(options_);
      if (!std::isfinite(static_cast<double>(reaction_)) || reaction_ < Real(0))
        throw std::invalid_argument("composite FAC reaction must be finite and non-negative");
      validate_request_(request);
    } catch (...) {
      validation_error = std::current_exception();
    }
    if (all_reduce_max(validation_error ? 1L : 0L) != 0) {
      if (n_ranks() == 1 && validation_error)
        std::rethrow_exception(validation_error);
      throw std::runtime_error("composite FAC preparation failed collectively");
    }

    build_levels_(request);
    build_connections_(request.ratios);
    build_coarse_solver_(request.levels.front());
    exact_prepared_contract_ = detail::fac_build_contract(request, options_, reaction_);
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
                                                CompositeFacOptions options = {},
                                                Real reaction = Real(0)) {
    detail::validate_fac_options(options);
    if (!std::isfinite(static_cast<double>(reaction)) || reaction < Real(0))
      throw std::invalid_argument("composite FAC reaction must be finite and non-negative");
    return detail::fac_build_contract(request, options, reaction);
  }

  std::string_view exact_prepared_contract() const noexcept { return exact_prepared_contract_; }
  int n_levels() const noexcept { return static_cast<int>(levels_.size()); }
  int maximum_iterations() const noexcept { return options_.max_iters; }
  field_type& rhs_level(int level) { return levels_.at(static_cast<std::size_t>(level))->rhs; }
  const field_type& rhs_level(int level) const {
    return levels_.at(static_cast<std::size_t>(level))->rhs;
  }
  field_type& phi_level(int level) { return levels_.at(static_cast<std::size_t>(level))->phi; }
  const field_type& phi_level(int level) const {
    return levels_.at(static_cast<std::size_t>(level))->phi;
  }
  const SolveReport& last_solve_report() const noexcept { return last_report_; }

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
    auto workspace = std::make_unique<FieldNullspaceWorkspace<Dim>>(
        plan, rhs_layouts, distributions);
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

    compute_composite_residual_();
    const Real reference = composite_residual_norm_();
    SolveReport report;
    report.reference_residual_norm = reference;
    report.residual_norm = reference;
    report.rel_residual = reference > Real(0) ? Real(1) : Real(0);
    report.evaluations = 1;
    const Real stop = std::max(options_.abs_tol, options_.rel_tol * reference);
    if (!std::isfinite(static_cast<double>(reference))) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                         "composite_fac_non_finite_initial_residual");
      last_report_ = report;
      return last_report_;
    }
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
    field_type covered;
    HaloSchedule<Dim> halo_schedule;
    PreparedPhysicalBoundary<Dim> physical_boundary;

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
          covered(request.boxes, request.distribution, request.local_rank, 1, Extent<Dim>{}),
          halo_schedule(prepare_halo_schedule(
              phi, geometry.domain(), boundary.topology(),
              full_domain ? HaloLayoutCoverage::full_domain : HaloLayoutCoverage::sparse_level,
              detail::exact_halo_budget(request.boxes, geometry.domain()))),
          physical_boundary(prepare_physical_boundary(
              geometry.domain(), detail::unit_ghosts<Dim>(), boundary,
              detail::exact_boundary_budget<Dim>())) {
      if (halo_schedule.has_remote_jobs())
        throw std::invalid_argument(
            "composite FAC distributed halo transport is not a registered capability");
      phi.set_val(Real(0));
      rhs.set_val(Real(0));
      residual.set_val(Real(0));
      scratch.set_val(Real(0));
      correction.set_val(Real(0));
      covered.set_val(Real(0));
    }
  };

  struct Connection {
    ::pops::amr::RefinementRatio<Dim> ratio;
    std::vector<fac_detail::InjectionTransfer<Dim>> coarse_fine_phi;
    std::vector<fac_detail::CellTransfer<Dim>> residual_restriction;
    std::vector<fac_detail::CellTransfer<Dim>> solution_restriction;
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

  static void validate_request_(const request_type& request) {
    if (request.levels.empty() || request.ratios.size() + 1 != request.levels.size())
      throw std::invalid_argument(
          "composite FAC requires one ratio between each adjacent pair of levels");
    for (std::size_t level = 0; level < request.levels.size(); ++level) {
      const auto& current = request.levels[level];
      detail::validate_boundary(current.geometry, current.boundary);
      if (current.geometry.domain().empty() || current.boxes.empty() ||
          !current.distribution.matches_layout(current.boxes) ||
          !current.distribution.rank_space().contains(current.local_rank) ||
          current.distribution.rank_space().size() != static_cast<std::size_t>(n_ranks()) ||
          current.distribution.rank_space().linear_rank(current.local_rank) !=
              static_cast<std::size_t>(my_rank()) ||
          !current.boxes.is_disjoint_within(current.geometry.domain(), current.layout_budget))
        throw std::invalid_argument("composite FAC level has an invalid exact-ranked layout");
      if (level == 0 &&
          !current.boxes.tiles_exactly(current.geometry.domain(), current.layout_budget))
        throw std::invalid_argument("composite FAC coarse level must tile its complete domain");
      if (n_ranks() > 1 && !current.distribution.replicated())
        throw std::invalid_argument(
            "composite FAC currently requires replicated levels under MPI");
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
      Connection connection{ratios[parent_index], {}, {}, {}, {}};
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
        if (!region.empty())
          for_each_cell(region, fac_detail::SetScalarKernel<Dim>{
                                    parent.covered.fab(local).view(), Real(1)});
      }
    }
    Kokkos::fence();
  }

  void prepare_connection_(Level& parent, Level& child, Connection& connection) {
    using Provider = ::pops::amr::transfer::TransferProvider<
        Dim, ::pops::amr::transfer::Centering::Cell>;
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
        connection.residual_restriction.push_back(restriction.prepare(
            child_residual_view, parent.residual.fab(parent_local).view(), region,
            connection.ratio));
        connection.solution_restriction.push_back(restriction.prepare(
            child_phi_view, parent.phi.fab(parent_local).view(), region, connection.ratio));
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
        connection.correction_prolongation.push_back(prolongation.prepare(
            parent_correction_view, child.correction.fab(child_local).view(), region,
            connection.ratio));
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
        const Box<Dim> parent_reach =
            refine(parent.phi.fab(parent_local).grown_box(), ratio_value);
        std::vector<Box<Dim>> next;
        for (const Box<Dim>& region : pending) {
          const Box<Dim> destination = region.intersect(parent_reach);
          if (!destination.empty()) {
            const auto source =
                static_cast<const field_type&>(parent.phi).fab(parent_local).view();
            connection.coarse_fine_phi.push_back(fac_detail::InjectionTransfer<Dim>{
                source, child.phi.fab(child_local).view(), destination, connection.ratio, {}});
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

  void build_coarse_solver_(const EllipticBuildRequest<Dim>& coarse_request) {
    GeometricMultigridOptions controls;
    controls.relative_tolerance = options_.coarse_rel_tol;
    controls.absolute_tolerance = options_.coarse_abs_tol;
    controls.maximum_cycles = options_.coarse_cycles;
    controls.reaction = reaction_;
    EllipticBuildRequest<Dim> correction_request = coarse_request;
    correction_request.boundary =
        detail::boundary_for_geometry(coarse_request.boundary, coarse_request.geometry, true);
    coarse_solver_ =
        std::make_unique<GeometricMG<Dim, MemorySpace>>(std::move(correction_request), controls);
  }

  void fill_level_ghosts_(std::size_t level_index, field_type& field,
                          bool interpolate_parent) {
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
        const auto covered_view =
            static_cast<const field_type&>(level.covered).fab(local).view();
        fac_detail::MaskedJacobiKernel<Dim> kernel{
            level.scratch.fab(local).view(), phi_view, rhs_view, covered_view, {},
            Real(1) / diagonal, Real(2) / Real(3), reaction_};
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
      const auto covered_view =
          static_cast<const field_type&>(level.covered).fab(local).view();
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
      const auto covered_view =
          static_cast<const field_type&>(level.covered).fab(local).view();
      for_each_cell(level.phi.box(local), fac_detail::MaskedAddKernel<Dim>{
                                                level.phi.fab(local).view(),
                                                correction.fab(local).view(),
                                                covered_view});
    }
    Kokkos::fence();
  }

  Real global_norm_inf_(const field_type& field) const {
    return static_cast<Real>(all_reduce_max(static_cast<double>(norm_inf(field))));
  }

  Real composite_residual_norm_() const {
    Real result = Real(0);
    for (const auto& level : levels_)
      result = std::max(result, norm_inf(level->residual));
    return static_cast<Real>(all_reduce_max(static_cast<double>(result)));
  }

  bool singular_() const noexcept {
    return detail::is_singular(levels_.front()->boundary, reaction_);
  }

  CompositeFacOptions options_{};
  Real reaction_ = Real(0);
  std::vector<std::unique_ptr<Level>> levels_{};
  std::vector<Connection> connections_{};
  std::unique_ptr<GeometricMG<Dim, MemorySpace>> coarse_solver_{};
  std::vector<const MultiFab<Dim>*> nullspace_rhs_{};
  std::vector<MultiFab<Dim>*> nullspace_candidates_{};
  std::unique_ptr<FieldNullspaceWorkspace<Dim>> nullspace_workspace_{};
  std::string exact_prepared_contract_{};
  SolveReport last_report_{};
};

}  // namespace pops::elliptic::mg
