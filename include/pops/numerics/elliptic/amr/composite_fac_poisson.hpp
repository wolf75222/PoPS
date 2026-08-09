/// @file
/// @brief Exact-rank partitioned-MPI composite FAC Poisson solver.

#pragma once

#include <pops/amr/refinement_ratio.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/halo_exchange.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/amr/partitioned_region_transfer.hpp>
#include <pops/numerics/elliptic/interface/elliptic_solver.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
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

namespace pops::elliptic::amr {

struct CompositeFacPreparationBudget {
  std::size_t levels = 0;
  std::size_t connections = 0;
  std::size_t parent_child_patch_pairs = 0;
  std::size_t interpolation_regions = 0;
  std::size_t local_scratch_cells = 0;
  HaloScheduleBudget same_level_halo{};
  partitioned_transfer::RegionTransferBudget parent_gather{};
  partitioned_transfer::RegionTransferBudget fine_restriction{};
};

template <int Dim>
struct CompositeFacBuildRequest {
  static_assert(Dim >= 1 && Dim <= 3,
                "CompositeFacBuildRequest only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  std::vector<EllipticBuildRequest<Dim>> levels{};
  std::vector<::pops::amr::RefinementRatio<Dim>> ratios{};
  CompositeFacPreparationBudget budget{};
};

struct CompositeFacCapabilities {
  bool scalar_constant_coefficient = true;
  bool partial_refinement = true;
  bool arbitrary_level_count = true;
  bool partitioned_mpi = true;
  bool replicated_mpi = false;
  bool conservative_restriction = true;
  bool injection_prolongation = true;
  bool periodic_sparse_levels = false;
  bool singular_nullspace = false;
  bool variable_coefficient = false;
  bool embedded_boundary = false;

  constexpr bool operator==(const CompositeFacCapabilities&) const = default;
};

namespace fac_detail {

inline std::size_t checked_product(std::size_t left, std::size_t right, const char* message) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(message);
  return left * right;
}

inline void checked_add(std::size_t& total, std::size_t value, std::size_t limit,
                        const char* message) {
  if (total > limit || value > limit - total)
    throw std::length_error(message);
  total += value;
}

template <int Dim>
Extent<Dim> unit_ghosts() {
  Extent<Dim> ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    ghosts[axis] = 1;
  return ghosts;
}

template <int Dim>
Extent<Dim> ratio_extent(const ::pops::amr::RefinementRatio<Dim>& ratio) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = ratio[axis];
  return result;
}

template <int Dim>
BoundaryScheduleBudget exact_boundary_budget() {
  std::size_t entries = 1;
  for (int axis = 0; axis < Dim; ++axis)
    entries =
        checked_product(entries, std::size_t{3}, "partitioned FAC boundary budget exceeds size_t");
  return {entries - 1};
}

template <int Dim>
PhysicalBoundaryConditions<Dim> homogeneous_boundary(const PhysicalBoundaryConditions<Dim>& source,
                                                     const Geometry<Dim>& geometry) {
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    spacing[axis] = geometry.spacing(axis);
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> face{axis, side};
      PhysicalBoundaryFace law = source.at(face);
      law.value = Real(0);
      faces[static_cast<std::size_t>(face.ordinal())] = law;
    }
  }
  return PhysicalBoundaryConditions<Dim>{source.topology(), faces, spacing};
}

template <int Dim>
void validate_boundary(const Geometry<Dim>& geometry,
                       const PhysicalBoundaryConditions<Dim>& boundary) {
  for (int axis = 0; axis < Dim; ++axis) {
    if (boundary.spacing()[axis] != geometry.spacing(axis))
      throw std::invalid_argument(
          "partitioned FAC boundary spacing differs from its exact geometry");
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> face{axis, side};
      if (!boundary.topology().is_periodic(face) &&
          boundary.at(face).kind == PhysicalBoundaryKind::external)
        throw std::invalid_argument("partitioned FAC requires a native law on every physical face");
    }
  }
}

template <int Dim>
bool singular(const PhysicalBoundaryConditions<Dim>& boundary, Real reaction) {
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

template <int Dim>
std::vector<Box<Dim>> subtract_box(const Box<Dim>& subject, const Box<Dim>& cut) {
  const Box<Dim> overlap = subject.intersect(cut);
  if (overlap.empty())
    return subject.empty() ? std::vector<Box<Dim>>{} : std::vector<Box<Dim>>{subject};
  if (overlap == subject)
    return {};
  std::vector<Box<Dim>> result;
  result.reserve(static_cast<std::size_t>(2 * Dim));
  Box<Dim> remainder = subject;
  for (int axis = 0; axis < Dim; ++axis) {
    if (remainder.lo[axis] < overlap.lo[axis]) {
      Box<Dim> lower = remainder;
      lower.hi[axis] = overlap.lo[axis] - 1;
      result.push_back(lower);
      remainder.lo[axis] = overlap.lo[axis];
    }
    if (overlap.hi[axis] < remainder.hi[axis]) {
      Box<Dim> upper = remainder;
      upper.lo[axis] = overlap.hi[axis] + 1;
      result.push_back(upper);
      remainder.hi[axis] = overlap.hi[axis];
    }
  }
  return result;
}

template <int Dim>
void subtract_from(std::vector<Box<Dim>>& regions, const Box<Dim>& cut) {
  std::vector<Box<Dim>> next;
  for (const Box<Dim>& region : regions) {
    auto pieces = subtract_box(region, cut);
    next.insert(next.end(), pieces.begin(), pieces.end());
  }
  regions = std::move(next);
}

template <int Dim>
Box<Dim> clipped_growth(const Box<Dim>& valid, const Box<Dim>& domain) {
  return valid.grow(1).intersect(domain);
}

template <int Dim>
Index<Dim> parent_index(const Index<Dim>& fine, const Box<Dim>& coarse_domain,
                        const Box<Dim>& fine_domain,
                        const ::pops::amr::RefinementRatio<Dim>& ratio) {
  Index<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::int64_t relative = static_cast<std::int64_t>(fine[axis]) - fine_domain.lo[axis];
    const std::int64_t quotient = relative / ratio[axis];
    const std::int64_t remainder = relative % ratio[axis];
    const std::int64_t parent = static_cast<std::int64_t>(coarse_domain.lo[axis]) +
                                (remainder < 0 ? quotient - 1 : quotient);
    if (parent < std::numeric_limits<int>::min() || parent > std::numeric_limits<int>::max())
      throw std::overflow_error("partitioned FAC parent index exceeds native coordinates");
    result[axis] = static_cast<int>(parent);
  }
  return result;
}

template <int Dim>
struct SetScalarKernel {
  FieldView<Real, Dim> values{};
  Real value = Real(0);
  POPS_HD void operator()(const Index<Dim>& index) const { values(index, 0) = value; }
};

template <int Dim>
struct MaskResidualKernel {
  FieldView<Real, Dim> residual{};
  FieldView<const Real, Dim> covered{};
  POPS_HD void operator()(const Index<Dim>& index) const {
    if (covered(index, 0) >= Real(0.5))
      residual(index, 0) = Real(0);
  }
};

template <int Dim>
struct JacobiKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> iterate{};
  FieldView<const Real, Dim> rhs{};
  FieldView<const Real, Dim> covered{};
  Real inverse_spacing_squared[Dim]{};
  Real inverse_diagonal = Real(0);
  Real reaction = Real(0);
  Real relaxation = Real(2) / Real(3);
  bool mask_covered = true;

  POPS_HD void operator()(const Index<Dim>& index) const {
    if (mask_covered && covered(index, 0) >= Real(0.5)) {
      destination(index, 0) = iterate(index, 0);
      return;
    }
    Real image = reaction * iterate(index, 0);
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> lower = index;
      Index<Dim> upper = index;
      --lower[axis];
      ++upper[axis];
      image += (Real(2) * iterate(index, 0) - iterate(lower, 0) - iterate(upper, 0)) *
               inverse_spacing_squared[axis];
    }
    destination(index, 0) =
        iterate(index, 0) + relaxation * inverse_diagonal * (rhs(index, 0) - image);
  }
};

template <int Dim>
struct ActiveAddKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> correction{};
  FieldView<const Real, Dim> active{};
  POPS_HD void operator()(const Index<Dim>& index) const {
    if (active(index, 0) >= Real(0.5))
      destination(index, 0) += correction(index, 0);
  }
};

template <int Dim>
struct InjectionKernel {
  FieldView<const Real, Dim> coarse{};
  FieldView<Real, Dim> fine{};
  Box<Dim> coarse_domain{};
  Box<Dim> fine_domain{};
  ::pops::amr::RefinementRatio<Dim> ratio{};
  POPS_HD void operator()(const Index<Dim>& index) const {
    Index<Dim> parent{};
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t relative = static_cast<std::int64_t>(index[axis]) - fine_domain.lo[axis];
      const std::int64_t quotient = relative / ratio[axis];
      const std::int64_t remainder = relative % ratio[axis];
      parent[axis] = static_cast<int>(static_cast<std::int64_t>(coarse_domain.lo[axis]) +
                                      (remainder < 0 ? quotient - 1 : quotient));
    }
    fine(index, 0) = coarse(parent, 0);
  }
};

template <int Dim>
struct RestrictionKernel {
  FieldView<const Real, Dim> fine{};
  FieldView<Real, Dim> coarse{};
  Box<Dim> coarse_domain{};
  Box<Dim> fine_domain{};
  ::pops::amr::RefinementRatio<Dim> ratio{};
  Real inverse_children = Real(1);

  POPS_HD void operator()(const Index<Dim>& parent) const {
    Index<Dim> base{};
    for (int axis = 0; axis < Dim; ++axis)
      base[axis] = static_cast<int>(
          static_cast<std::int64_t>(fine_domain.lo[axis]) +
          (static_cast<std::int64_t>(parent[axis]) - coarse_domain.lo[axis]) * ratio[axis]);
    Real sum = Real(0);
    Index<Dim> child{};
    bool more = true;
    while (more) {
      Index<Dim> fine_index = base;
      for (int axis = 0; axis < Dim; ++axis)
        fine_index[axis] += child[axis];
      sum += fine(fine_index, 0);
      more = false;
      for (int axis = 0; axis < Dim; ++axis) {
        ++child[axis];
        if (child[axis] < ratio[axis]) {
          more = true;
          break;
        }
        child[axis] = 0;
      }
    }
    coarse(parent, 0) = sum * inverse_children;
  }
};

inline void validate_options(const CompositeFacOptions& options) {
  if (options.max_iters < 1 || options.fine_sweeps < 1 || options.coarse_cycles < 1 ||
      !std::isfinite(static_cast<double>(options.rel_tol)) || options.rel_tol <= Real(0) ||
      !std::isfinite(static_cast<double>(options.abs_tol)) || options.abs_tol < Real(0) ||
      !std::isfinite(static_cast<double>(options.coarse_rel_tol)) ||
      options.coarse_rel_tol <= Real(0) ||
      !std::isfinite(static_cast<double>(options.coarse_abs_tol)) ||
      options.coarse_abs_tol < Real(0))
    throw std::invalid_argument("partitioned FAC controls are invalid");
}

template <int Dim>
std::string exact_contract(const CompositeFacBuildRequest<Dim>& request,
                           const CompositeFacOptions& options, Real reaction) {
  ExactContractBuilder contract;
  contract.text("pops.elliptic.amr.partitioned-composite-fac")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .scalar(static_cast<std::uint64_t>(request.levels.size()));
  for (const auto& level : request.levels)
    contract.bytes(elliptic_contract_detail::build_request_contract(level));
  contract.scalar(static_cast<std::uint64_t>(request.ratios.size()));
  for (const auto& ratio : request.ratios)
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(ratio[axis]);
  const auto& budget = request.budget;
  contract.scalar(budget.levels)
      .scalar(budget.connections)
      .scalar(budget.parent_child_patch_pairs)
      .scalar(budget.interpolation_regions)
      .scalar(budget.local_scratch_cells)
      .scalar(budget.same_level_halo.layout.boxes)
      .scalar(budget.same_level_halo.layout.overlap_pairs)
      .scalar(budget.same_level_halo.box_image_pairs)
      .scalar(budget.same_level_halo.jobs)
      .scalar(budget.same_level_halo.periodic_images)
      .scalar(budget.same_level_halo.peer_plans)
      .scalar(budget.same_level_halo.local_elements)
      .scalar(budget.same_level_halo.send_elements)
      .scalar(budget.same_level_halo.receive_elements)
      .scalar(budget.parent_gather.canonical_jobs)
      .scalar(budget.parent_gather.peer_plans)
      .scalar(budget.parent_gather.local_elements)
      .scalar(budget.parent_gather.send_elements)
      .scalar(budget.parent_gather.receive_elements)
      .scalar(budget.fine_restriction.canonical_jobs)
      .scalar(budget.fine_restriction.peer_plans)
      .scalar(budget.fine_restriction.local_elements)
      .scalar(budget.fine_restriction.send_elements)
      .scalar(budget.fine_restriction.receive_elements)
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

}  // namespace fac_detail

/// Composite FAC over an exact partitioned hierarchy.
///
/// No replicated hierarchy is accepted. Same-level ghosts use HaloExchange on one duplicated
/// owning lane. Every coarse/fine gather and conservative restriction is described by a globally
/// canonical plan, staged before communication, authenticated on receive, and published only after
/// collective success. Spatial rank is a template argument and every stencil iterates over axes.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class CompositeFacPoisson {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "CompositeFacPoisson only supports dimensions 1, 2, and 3");
  static_assert(Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, MemorySpace>::accessible,
                "CompositeFacPoisson requires DefaultExecutionSpace access to MemorySpace");

  static constexpr int dimension = Dim;
  using field_type = MultiFab<Dim, MemorySpace>;
  using request_type = CompositeFacBuildRequest<Dim>;

  CompositeFacPoisson(request_type request, CompositeFacOptions options = {},
                      Real reaction = Real(0))
      : options_(options), reaction_(reaction) {
    std::exception_ptr local_error;
    try {
      fac_detail::validate_options(options_);
      if (!std::isfinite(static_cast<double>(reaction_)) || reaction_ < Real(0))
        throw std::invalid_argument("partitioned FAC reaction must be finite and non-negative");
      validate_request_(request, reaction_);
      lane_identity_ = "pops.elliptic.amr.composite-fac.nd" + std::to_string(Dim);
      exact_contract_ = fac_detail::exact_contract(request, options_, reaction_);
      build_levels_(request);
      build_connections_(request);
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L) != 0) {
      if (n_ranks() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error(
          "partitioned FAC metadata, budget, or reusable allocation failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("pops-partitioned-composite-fac"),
              std::string_view(exact_contract_)}}))
      throw std::invalid_argument(
          "partitioned FAC exact hierarchy contract differs between MPI ranks");
    for (std::size_t connection = 0; connection < connections_.size(); ++connection) {
      if (!all_ranks_agree_exact_ordered_byte_pairs(
              {{std::string_view("pops-fac-parent-gather"),
                std::string_view(connections_[connection]->gather_contract)},
               {std::string_view("pops-fac-fine-restriction"),
                std::string_view(connections_[connection]->restriction_contract)}}))
        throw std::invalid_argument(
            "partitioned FAC coarse/fine transfer plan differs between MPI ranks");
    }

    lane_.emplace(ExecutionLane::duplicate_world_collectively(lane_identity_));
    for (auto& connection : connections_)
      connection->attach_lane(*lane_);
    for (std::size_t level = 0; level < levels_.size(); ++level) {
      const bool remote = all_reduce_max(levels_[level]->halo_schedule.has_remote_jobs() ? 1L : 0L,
                                         lane_->communicator()) != 0;
      if (remote) {
        HaloExchangeContext context{};
        context.context_generation = level + 1;
        context.schedule_generation = level + 1;
        levels_[level]->halo_exchange.emplace(levels_[level]->halo_schedule, *lane_, context);
      }
    }
  }

  CompositeFacPoisson(const CompositeFacPoisson&) = delete;
  CompositeFacPoisson& operator=(const CompositeFacPoisson&) = delete;
  CompositeFacPoisson(CompositeFacPoisson&&) = delete;
  CompositeFacPoisson& operator=(CompositeFacPoisson&&) = delete;

  static constexpr CompositeFacCapabilities capabilities() noexcept { return {}; }
  static constexpr EllipticOperatorIdentity operator_identity() noexcept {
    return {"pops.elliptic.amr.partitioned-composite-fac.nd", 1};
  }

  std::string_view exact_prepared_contract() const noexcept { return exact_contract_; }
  int n_levels() const noexcept { return static_cast<int>(levels_.size()); }
  int maximum_iterations() const noexcept { return options_.max_iters; }
  const Geometry<Dim>& geom() const noexcept { return levels_.front()->geometry; }
  field_type& rhs() noexcept { return levels_.front()->rhs; }
  const field_type& rhs() const noexcept { return levels_.front()->rhs; }
  field_type& phi() noexcept { return levels_.front()->phi; }
  const field_type& phi() const noexcept { return levels_.front()->phi; }
  field_type& rhs_level(int level) { return levels_.at(static_cast<std::size_t>(level))->rhs; }
  const field_type& rhs_level(int level) const {
    return levels_.at(static_cast<std::size_t>(level))->rhs;
  }
  field_type& phi_level(int level) { return levels_.at(static_cast<std::size_t>(level))->phi; }
  const field_type& phi_level(int level) const {
    return levels_.at(static_cast<std::size_t>(level))->phi;
  }
  Real residual() const noexcept { return last_report_.residual_norm; }
  const SolveReport& last_solve_report() const noexcept { return last_report_; }
  bool owns_execution_lane() const noexcept {
    return lane_ && (n_ranks() == 1 || lane_->owns_communicator());
  }
  bool has_remote_same_level_halo() const noexcept {
    return std::any_of(levels_.begin(), levels_.end(),
                       [](const auto& level) { return level->halo_schedule.has_remote_jobs(); });
  }
  bool has_remote_parent_gather() const noexcept {
    return std::any_of(connections_.begin(), connections_.end(), [](const auto& connection) {
      return connection->gather->plan().has_remote_jobs();
    });
  }
  bool has_remote_fine_restriction() const noexcept {
    return std::any_of(connections_.begin(), connections_.end(), [](const auto& connection) {
      return connection->restriction->plan().has_remote_jobs();
    });
  }

  SolveReport solve() {
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
                         "partitioned_fac_non_finite_initial_residual");
      last_report_ = report;
      return last_report_;
    }
    if (reference <= stop) {
      fill_all_solution_ghosts_();
      report.mark_solved("partitioned_fac_initial_residual");
      last_report_ = report;
      return last_report_;
    }

    const int pre = (options_.fine_sweeps + 1) / 2;
    const int post = options_.fine_sweeps / 2;
    for (int iteration = 0; iteration < options_.max_iters; ++iteration) {
      for (std::size_t level = 1; level < levels_.size(); ++level)
        smooth_(level, levels_[level]->phi, levels_[level]->rhs, pre, true, false);

      compute_composite_residual_();
      restrict_residual_tower_();
      solve_coarse_correction_();
      report.step_norm = global_norm_inf_(levels_.front()->correction);
      add_active_(*levels_.front(), levels_.front()->correction);
      prolong_correction_tower_();
      for (std::size_t level = 1; level < levels_.size(); ++level)
        smooth_(level, levels_[level]->phi, levels_[level]->rhs, post, true, false);
      average_solution_down_();

      compute_composite_residual_();
      ++report.evaluations;
      report.iters = iteration + 1;
      report.residual_norm = composite_residual_norm_();
      report.rel_residual = report.residual_norm / reference;
      if (!std::isfinite(static_cast<double>(report.residual_norm))) {
        report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                           "partitioned_fac_non_finite_iteration");
        last_report_ = report;
        return last_report_;
      }
      if (report.residual_norm <= stop) {
        fill_all_solution_ghosts_();
        report.mark_solved("partitioned_fac_converged");
        last_report_ = report;
        return last_report_;
      }
    }
    report.mark_failed(SolveStatus::kIterationLimit, SolveAction::kFailRun,
                       "partitioned_fac_iteration_limit");
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
    field_type active;
    HaloSchedule<Dim> halo_schedule;
    PreparedPhysicalBoundary<Dim> physical_boundary;
    PreparedPhysicalBoundary<Dim> homogeneous_physical_boundary;
    std::optional<HaloExchange<Dim, MemorySpace>> halo_exchange{};

    Level(const EllipticBuildRequest<Dim>& request, bool full_domain,
          const HaloScheduleBudget& halo_budget)
        : geometry(request.geometry),
          boundary(request.boundary),
          phi(request.boxes, request.distribution, request.local_rank, 1,
              fac_detail::unit_ghosts<Dim>()),
          rhs(request.boxes, request.distribution, request.local_rank, 1, Extent<Dim>{}),
          residual(request.boxes, request.distribution, request.local_rank, 1, Extent<Dim>{}),
          scratch(request.boxes, request.distribution, request.local_rank, 1, Extent<Dim>{}),
          correction(request.boxes, request.distribution, request.local_rank, 1,
                     fac_detail::unit_ghosts<Dim>()),
          covered(request.boxes, request.distribution, request.local_rank, 1, Extent<Dim>{}),
          active(request.boxes, request.distribution, request.local_rank, 1, Extent<Dim>{}),
          halo_schedule(prepare_halo_schedule(
              phi, geometry.domain(), boundary.topology(),
              full_domain ? HaloLayoutCoverage::full_domain : HaloLayoutCoverage::sparse_level,
              halo_budget)),
          physical_boundary(prepare_physical_boundary(geometry.domain(),
                                                      fac_detail::unit_ghosts<Dim>(), boundary,
                                                      fac_detail::exact_boundary_budget<Dim>())),
          homogeneous_physical_boundary(
              prepare_physical_boundary(geometry.domain(), fac_detail::unit_ghosts<Dim>(),
                                        fac_detail::homogeneous_boundary(boundary, geometry),
                                        fac_detail::exact_boundary_budget<Dim>())) {
      phi.set_val(Real(0));
      rhs.set_val(Real(0));
      residual.set_val(Real(0));
      scratch.set_val(Real(0));
      correction.set_val(Real(0));
      covered.set_val(Real(0));
      active.set_val(Real(1));
    }
  };

  struct Connection {
    using transfer_job = partitioned_transfer::RegionTransferJob<Dim>;
    using transfer_plan = partitioned_transfer::RegionTransferPlan<Dim>;
    using transport_type = partitioned_transfer::RegionTransport<Dim, MemorySpace>;

    struct ScratchPatch {
      std::size_t fine_patch = 0;
      Fab<Dim, MemorySpace> parent_staging{};
      Fab<Dim, MemorySpace> restricted{};
      std::vector<Box<Dim>> ghost_regions{};
    };

    Level* parent = nullptr;
    Level* child = nullptr;
    ::pops::amr::RefinementRatio<Dim> ratio{};
    std::vector<ScratchPatch> scratch{};
    std::vector<std::size_t> scratch_by_fine_patch{};
    std::unique_ptr<transport_type> gather{};
    std::unique_ptr<transport_type> restriction{};
    std::string gather_contract{};
    std::string restriction_contract{};

    static constexpr std::size_t no_scratch = std::numeric_limits<std::size_t>::max();

    Connection(Level& parent_level, Level& child_level,
               ::pops::amr::RefinementRatio<Dim> level_ratio,
               const CompositeFacPreparationBudget& budget, std::size_t ordinal)
        : parent(&parent_level), child(&child_level), ratio(level_ratio) {
      const auto ratio_value = fac_detail::ratio_extent(ratio);
      scratch_by_fine_patch.assign(child->phi.layout().size(), no_scratch);
      scratch.reserve(child->phi.local_size());
      std::size_t local_cells = 0;
      std::size_t interpolation_regions = 0;
      for (std::size_t fine_patch = 0; fine_patch < child->phi.layout().size(); ++fine_patch) {
        if (!child->phi.contains_local(fine_patch))
          continue;
        const Box<Dim>& valid = child->phi.layout()[fine_patch];
        const Box<Dim> staging =
            coarsen(fac_detail::clipped_growth(valid, child->geometry.domain()), ratio_value);
        const Box<Dim> restricted_box = coarsen(valid, ratio_value);
        fac_detail::checked_add(local_cells, static_cast<std::size_t>(staging.numPts()),
                                budget.local_scratch_cells,
                                "partitioned FAC parent staging budget exceeded");
        fac_detail::checked_add(local_cells, static_cast<std::size_t>(restricted_box.numPts()),
                                budget.local_scratch_cells,
                                "partitioned FAC restriction scratch budget exceeded");
        ScratchPatch patch;
        patch.fine_patch = fine_patch;
        patch.parent_staging = Fab<Dim, MemorySpace>(staging, 1, Extent<Dim>{});
        patch.restricted = Fab<Dim, MemorySpace>(restricted_box, 1, Extent<Dim>{});
        std::vector<Box<Dim>> pending{fac_detail::clipped_growth(valid, child->geometry.domain())};
        for (const Box<Dim>& peer : child->phi.layout().boxes())
          fac_detail::subtract_from(pending, peer);
        interpolation_regions += pending.size();
        if (interpolation_regions > budget.interpolation_regions)
          throw std::length_error("partitioned FAC interpolation-region budget exceeded");
        patch.ghost_regions = std::move(pending);
        scratch_by_fine_patch[fine_patch] = scratch.size();
        scratch.push_back(std::move(patch));
      }

      std::vector<transfer_job> gather_jobs;
      std::vector<transfer_job> restriction_jobs;
      const std::size_t pair_count =
          fac_detail::checked_product(child->phi.layout().size(), parent->phi.layout().size(),
                                      "partitioned FAC parent-child pair count exceeds size_t");
      if (pair_count > budget.parent_child_patch_pairs)
        throw std::length_error("partitioned FAC parent-child pair budget exceeded");
      gather_jobs.reserve(pair_count);
      restriction_jobs.reserve(pair_count);
      for (std::size_t fine_patch = 0; fine_patch < child->phi.layout().size(); ++fine_patch) {
        const Box<Dim>& valid = child->phi.layout()[fine_patch];
        const Box<Dim> staging =
            coarsen(fac_detail::clipped_growth(valid, child->geometry.domain()), ratio_value);
        const Box<Dim> footprint = coarsen(valid, ratio_value);
        mesh::ExactCellCount gather_coverage;
        mesh::ExactCellCount restriction_coverage;
        for (std::size_t parent_patch = 0; parent_patch < parent->phi.layout().size();
             ++parent_patch) {
          const Box<Dim> gathered = staging.intersect(parent->phi.layout()[parent_patch]);
          if (!gathered.empty()) {
            if (!gather_coverage.add(mesh::ExactCellCount::from_box(gathered)))
              throw std::overflow_error("partitioned FAC gather coverage overflows");
            gather_jobs.push_back(transfer_job{
                parent_patch, fine_patch, parent->phi.distribution().owner(parent_patch),
                child->phi.distribution().owner(fine_patch), gathered, gathered});
          }
          const Box<Dim> restricted_region =
              footprint.intersect(parent->phi.layout()[parent_patch]);
          if (!restricted_region.empty()) {
            if (!restriction_coverage.add(mesh::ExactCellCount::from_box(restricted_region)))
              throw std::overflow_error("partitioned FAC restriction coverage overflows");
            restriction_jobs.push_back(transfer_job{fine_patch, parent_patch,
                                                    child->phi.distribution().owner(fine_patch),
                                                    parent->phi.distribution().owner(parent_patch),
                                                    restricted_region, restricted_region});
          }
        }
        if (gather_coverage != mesh::ExactCellCount::from_box(staging))
          throw std::invalid_argument(
              "partitioned FAC parent layout does not cover a child interpolation footprint");
        if (restriction_coverage != mesh::ExactCellCount::from_box(footprint))
          throw std::invalid_argument(
              "partitioned FAC parent layout does not cover a child restriction footprint");
      }
      gather = std::make_unique<transport_type>(
          transfer_plan{parent->phi.rank_space(), parent->phi.local_rank(), 1,
                        std::move(gather_jobs), budget.parent_gather});
      restriction = std::make_unique<transport_type>(
          transfer_plan{parent->phi.rank_space(), parent->phi.local_rank(), 1,
                        std::move(restriction_jobs), budget.fine_restriction});
      gather_contract = gather->plan().exact_contract("parent-gather/" + std::to_string(ordinal));
      restriction_contract =
          restriction->plan().exact_contract("fine-restriction/" + std::to_string(ordinal));
    }

    void attach_lane(const ExecutionLane& lane) {
      gather->attach_lane(lane);
      restriction->attach_lane(lane);
    }

    ScratchPatch& scratch_for(std::size_t fine_patch) {
      const std::size_t local = scratch_by_fine_patch.at(fine_patch);
      if (local == no_scratch)
        throw std::out_of_range("partitioned FAC scratch patch is not local");
      return scratch.at(local);
    }

    void gather_parent(const field_type& source) {
      auto source_view = [&source](const transfer_job& job) -> FieldView<const Real, Dim> {
        return source.fab_global(job.source_patch).view();
      };
      auto destination_view = [this](const transfer_job& job) -> FieldView<Real, Dim> {
        return scratch_for(job.destination_patch).parent_staging.view();
      };
      gather->execute(source_view, destination_view);
    }

    void interpolate_ghosts(field_type& destination) {
      for (ScratchPatch& patch : scratch) {
        const auto coarse = std::as_const(patch.parent_staging).view();
        auto fine = destination.fab_global(patch.fine_patch).view();
        for (const Box<Dim>& region : patch.ghost_regions)
          for_each_cell(region,
                        fac_detail::InjectionKernel<Dim>{coarse, fine, parent->geometry.domain(),
                                                         child->geometry.domain(), ratio});
      }
      Kokkos::fence();
    }

    void prolong_valid(field_type& destination) {
      for (ScratchPatch& patch : scratch) {
        const auto coarse = std::as_const(patch.parent_staging).view();
        auto fine = destination.fab_global(patch.fine_patch).view();
        for_each_cell(destination.fab_global(patch.fine_patch).box(),
                      fac_detail::InjectionKernel<Dim>{coarse, fine, parent->geometry.domain(),
                                                       child->geometry.domain(), ratio});
      }
      Kokkos::fence();
    }

    void restrict_into(const field_type& source, field_type& destination) {
      const Real inverse_children = Real(1) / static_cast<Real>(ratio.child_count());
      for (ScratchPatch& patch : scratch) {
        const auto fine = source.fab_global(patch.fine_patch).view();
        auto coarse = patch.restricted.view();
        for_each_cell(
            patch.restricted.box(),
            fac_detail::RestrictionKernel<Dim>{fine, coarse, parent->geometry.domain(),
                                               child->geometry.domain(), ratio, inverse_children});
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
  };

  static void validate_request_(const request_type& request, Real reaction) {
    if (request.levels.empty() || request.ratios.size() + 1 != request.levels.size())
      throw std::invalid_argument(
          "partitioned FAC requires one ratio between adjacent hierarchy levels");
    if (request.levels.size() > request.budget.levels ||
        request.ratios.size() > request.budget.connections)
      throw std::length_error("partitioned FAC hierarchy budget exceeded");
    const auto& rank_space = request.levels.front().distribution.rank_space();
    for (std::size_t level = 0; level < request.levels.size(); ++level) {
      const auto& current = request.levels[level];
      fac_detail::validate_boundary(current.geometry, current.boundary);
      for (int axis = 0; axis < Dim; ++axis)
        for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper})
          if (current.boundary.topology().is_periodic(Face<Dim>{axis, side}))
            throw std::invalid_argument(
                "partitioned FAC sparse periodic coarse/fine interpolation is not a registered "
                "capability");
      if (current.geometry.domain().empty() || current.boxes.empty() ||
          !current.distribution.matches_layout(current.boxes) ||
          current.distribution.replicated() || current.distribution.rank_space() != rank_space ||
          rank_space.size() != static_cast<std::size_t>(n_ranks()) ||
          !rank_space.contains(current.local_rank) ||
          rank_space.linear_rank(current.local_rank) != static_cast<std::size_t>(my_rank()) ||
          !current.boxes.is_disjoint_within(current.geometry.domain(), current.layout_budget))
        throw std::invalid_argument(
            "partitioned FAC level has an invalid non-replicated exact-rank layout");
      if (level == 0 &&
          !current.boxes.tiles_exactly(current.geometry.domain(), current.layout_budget))
        throw std::invalid_argument("partitioned FAC coarse level must tile its full domain");
      for (int axis = 0; axis < Dim; ++axis)
        if (current.rhs_ghosts[axis] != 0 || current.phi_ghosts[axis] < 1)
          throw std::invalid_argument(
              "partitioned FAC requires ghost-free RHS and at least one solution ghost");
      if (level == 0)
        continue;
      const auto& parent = request.levels[level - 1];
      const auto& ratio = request.ratios[level - 1];
      for (int axis = 0; axis < Dim; ++axis)
        if (ratio[axis] < 2)
          throw std::invalid_argument(
              "partitioned FAC refinement ratio must be at least two on every axis");
      const Extent<Dim> ratio_value = fac_detail::ratio_extent(ratio);
      if (current.geometry != parent.geometry.refine(ratio_value) ||
          current.boundary.topology() != parent.boundary.topology())
        throw std::invalid_argument(
            "partitioned FAC adjacent geometry or topology is not an exact refinement");
      for (const Box<Dim>& patch : current.boxes.boxes())
        if (refine(coarsen(patch, ratio_value), ratio_value) != patch)
          throw std::invalid_argument(
              "partitioned FAC fine patch is not aligned to its refinement ratio");
    }
    if (fac_detail::singular(request.levels.front().boundary, reaction))
      throw std::invalid_argument(
          "partitioned FAC singular nullspaces are not a registered capability; use reaction>0 "
          "or an anchoring physical boundary");
  }

  void build_levels_(const request_type& request) {
    levels_.reserve(request.levels.size());
    for (std::size_t level = 0; level < request.levels.size(); ++level)
      levels_.push_back(std::make_unique<Level>(request.levels[level], level == 0,
                                                request.budget.same_level_halo));
  }

  void build_connections_(const request_type& request) {
    connections_.reserve(request.ratios.size());
    for (std::size_t parent = 0; parent < request.ratios.size(); ++parent) {
      connections_.push_back(std::make_unique<Connection>(
          *levels_[parent], *levels_[parent + 1], request.ratios[parent], request.budget, parent));
      mark_coverage_(*levels_[parent], *levels_[parent + 1], request.ratios[parent]);
    }
  }

  static void mark_coverage_(Level& parent, const Level& child,
                             const ::pops::amr::RefinementRatio<Dim>& ratio) {
    const auto ratio_value = fac_detail::ratio_extent(ratio);
    for (const Box<Dim>& fine_patch : child.phi.layout().boxes()) {
      const Box<Dim> footprint = coarsen(fine_patch, ratio_value);
      for (std::size_t local = 0; local < parent.phi.local_size(); ++local) {
        const Box<Dim> region = parent.phi.box(local).intersect(footprint);
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

  void same_level_fill_(Level& level, field_type& field) {
    if (level.halo_exchange)
      level.halo_exchange->execute(field, *lane_);
    else
      fill_boundary(field, level.halo_schedule);
  }

  void fill_ghosts_(std::size_t level_index, field_type& field, bool homogeneous) {
    Level& level = *levels_[level_index];
    if (level_index > 0) {
      Connection& connection = *connections_[level_index - 1];
      const field_type& parent_field = &field == &level.correction
                                           ? levels_[level_index - 1]->correction
                                           : levels_[level_index - 1]->phi;
      connection.gather_parent(parent_field);
      connection.interpolate_ghosts(field);
    }
    same_level_fill_(level, field);
    fill_physical_boundary(
        field, homogeneous ? level.homogeneous_physical_boundary : level.physical_boundary);
  }

  void fill_all_solution_ghosts_() {
    for (std::size_t level = 0; level < levels_.size(); ++level)
      fill_ghosts_(level, levels_[level]->phi, false);
  }

  void smooth_(std::size_t level_index, field_type& iterate, const field_type& rhs, int sweeps,
               bool mask_covered, bool homogeneous) {
    if (sweeps <= 0)
      return;
    Level& level = *levels_[level_index];
    Real inverse_spacing_squared[Dim]{};
    Real diagonal = reaction_;
    for (int axis = 0; axis < Dim; ++axis) {
      const Real inverse = Real(1) / level.geometry.spacing(axis);
      inverse_spacing_squared[axis] = inverse * inverse;
      diagonal += Real(2) * inverse_spacing_squared[axis];
    }
    for (int sweep = 0; sweep < sweeps; ++sweep) {
      fill_ghosts_(level_index, iterate, homogeneous);
      for (std::size_t local = 0; local < iterate.local_size(); ++local) {
        fac_detail::JacobiKernel<Dim> kernel{level.scratch.fab(local).view(),
                                             std::as_const(iterate).fab(local).view(),
                                             rhs.fab(local).view(),
                                             std::as_const(level.covered).fab(local).view(),
                                             {},
                                             Real(1) / diagonal,
                                             reaction_,
                                             Real(2) / Real(3),
                                             mask_covered};
        for (int axis = 0; axis < Dim; ++axis)
          kernel.inverse_spacing_squared[axis] = inverse_spacing_squared[axis];
        for_each_cell(iterate.box(local), kernel);
      }
      Kokkos::fence();
      ::pops::elliptic::mg::copy_scalar_valid(level.scratch, iterate);
    }
  }

  void compute_level_residual_(std::size_t level_index) {
    Level& level = *levels_[level_index];
    fill_ghosts_(level_index, level.phi, false);
    ::pops::elliptic::mg::poisson_residual_valid(level.phi, level.rhs, level.geometry,
                                                 level.residual, reaction_);
    for (std::size_t local = 0; local < level.residual.local_size(); ++local)
      for_each_cell(level.residual.box(local), fac_detail::MaskResidualKernel<Dim>{
                                                   level.residual.fab(local).view(),
                                                   std::as_const(level.covered).fab(local).view()});
    Kokkos::fence();
  }

  void compute_composite_residual_() {
    for (std::size_t level = 0; level < levels_.size(); ++level)
      compute_level_residual_(level);
  }

  void restrict_residual_tower_() {
    for (std::size_t child = levels_.size(); child-- > 1;)
      connections_[child - 1]->restrict_into(levels_[child]->residual,
                                             levels_[child - 1]->residual);
  }

  void solve_coarse_correction_() {
    Level& coarse = *levels_.front();
    coarse.correction.set_val(Real(0));
    const Real reference = global_norm_inf_(coarse.residual);
    const Real stop = std::max(options_.coarse_abs_tol, options_.coarse_rel_tol * reference);
    for (int sweep = 0; sweep < options_.coarse_cycles; ++sweep) {
      smooth_(0, coarse.correction, coarse.residual, 1, false, true);
      if ((sweep + 1) % 8 == 0 || sweep + 1 == options_.coarse_cycles) {
        fill_ghosts_(0, coarse.correction, true);
        ::pops::elliptic::mg::poisson_residual_valid(coarse.correction, coarse.residual,
                                                     coarse.geometry, coarse.scratch, reaction_);
        if (global_norm_inf_(coarse.scratch) <= stop)
          break;
      }
    }
  }

  static void add_active_(Level& level, const field_type& correction) {
    for (std::size_t local = 0; local < level.phi.local_size(); ++local)
      for_each_cell(level.phi.box(local),
                    fac_detail::ActiveAddKernel<Dim>{
                        level.phi.fab(local).view(), correction.fab(local).view(),
                        std::as_const(level.active).fab(local).view()});
    Kokkos::fence();
  }

  void prolong_correction_tower_() {
    for (std::size_t parent = 0; parent < connections_.size(); ++parent) {
      Connection& connection = *connections_[parent];
      connection.gather_parent(levels_[parent]->correction);
      levels_[parent + 1]->correction.set_val(Real(0));
      connection.prolong_valid(levels_[parent + 1]->correction);
      add_active_(*levels_[parent + 1], levels_[parent + 1]->correction);
    }
  }

  void average_solution_down_() {
    for (std::size_t child = levels_.size(); child-- > 1;)
      connections_[child - 1]->restrict_into(levels_[child]->phi, levels_[child - 1]->phi);
  }

  Real global_norm_inf_(const field_type& field) const {
    return static_cast<Real>(
        all_reduce_max(static_cast<double>(norm_inf(field)), lane_->communicator()));
  }

  Real composite_residual_norm_() const {
    Real local = Real(0);
    for (const auto& level : levels_)
      local = std::max(local, norm_inf(level->residual));
    return static_cast<Real>(all_reduce_max(static_cast<double>(local), lane_->communicator()));
  }

  CompositeFacOptions options_{};
  Real reaction_ = Real(0);
  std::optional<ExecutionLane> lane_{};
  std::vector<std::unique_ptr<Level>> levels_{};
  std::vector<std::unique_ptr<Connection>> connections_{};
  std::string lane_identity_{};
  std::string exact_contract_{};
  SolveReport last_report_{};
};

}  // namespace pops::elliptic::amr
