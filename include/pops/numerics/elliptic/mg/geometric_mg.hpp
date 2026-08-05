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

  bool operator==(const GeometricMultigridOptions&) const = default;
};

struct GeometricMultigridCapabilities {
  bool scalar_constant_coefficient = true;
  bool scalar_reaction = true;
  bool variable_diagonal = false;
  bool cross_tensor = false;
  bool embedded_boundary = false;

  constexpr bool operator==(const GeometricMultigridCapabilities&) const = default;
};

namespace detail {

inline std::size_t checked_size_product(std::size_t left, std::size_t right,
                                        const char* operation) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(operation);
  return left * right;
}

inline std::size_t checked_size_sum(std::size_t left, std::size_t right,
                                    const char* operation) {
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
  const std::size_t pairs =
      layout.size() < 2
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
    return boxes < 2 ? std::size_t{0}
                     : checked_size_product(boxes, boxes - 1,
                                            "geometric MG overlap budget overflow") /
                           2;
  };
  const std::size_t pairs = checked_size_product(
      left.size(), right.size(), "geometric MG copy-pair budget overflow");
  return {pairs, pairs, overlap_pairs(left.size()), overlap_pairs(right.size())};
}

template <int Dim>
HaloScheduleBudget exact_halo_budget(const mesh::BoxArray<Dim>& layout,
                                     const Box<Dim>& domain) {
  const std::size_t boxes = layout.size();
  const std::size_t pairs =
      checked_size_product(boxes, boxes, "geometric MG halo-pair budget overflow");
  std::size_t images = 1;
  for (int axis = 0; axis < Dim; ++axis)
    images = checked_size_product(images, 3, "geometric MG image budget overflow");
  const std::size_t work =
      checked_size_product(pairs, images, "geometric MG halo-work budget overflow");
  const std::size_t jobs = checked_size_product(
      work, static_cast<std::size_t>(2 * Dim), "geometric MG halo-job budget overflow");
  const std::int64_t signed_cells = domain.numPts();
  if (signed_cells <= 0)
    throw std::invalid_argument("geometric MG halo domain must be non-empty");
  const std::size_t cells = static_cast<std::size_t>(signed_cells);
  const std::size_t elements = checked_size_product(
      jobs, cells, "geometric MG halo-element budget overflow");
  return {exact_layout_budget(layout), work, jobs, images,
          checked_size_product(boxes, std::size_t{2},
                               "geometric MG peer budget overflow"),
          elements, elements, elements};
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
    throw std::invalid_argument(
        "geometric MG coarsening must retain the global patch cardinality");
  if (model.replicated())
    return mesh::Distribution<Dim>::replicated(layout, model.rank_space());
  return mesh::Distribution<Dim>::partitioned(layout, model.rank_space(), model.owners());
}

template <int Dim>
PhysicalBoundaryConditions<Dim> boundary_for_geometry(
    const PhysicalBoundaryConditions<Dim>& source, const Geometry<Dim>& geometry,
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
      throw std::invalid_argument(
          "geometric MG boundary spacing differs from its exact geometry");
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> face{axis, side};
      const PhysicalBoundaryFace law = boundary.at(face);
      if (boundary.topology().is_periodic(face)) {
        if (law.kind != PhysicalBoundaryKind::external)
          throw std::invalid_argument(
              "geometric MG periodic faces cannot carry a physical law");
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

  GeometricMG(request_type request, GeometricMultigridOptions options = {})
      : geometry_(request.geometry),
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
          request.distribution.rank_space().size() != static_cast<std::size_t>(n_ranks()) ||
          request.distribution.rank_space().linear_rank(request.local_rank) !=
              static_cast<std::size_t>(my_rank()))
        throw std::invalid_argument("geometric MG build request is not an exact collective layout");
      for (int axis = 0; axis < Dim; ++axis)
        if (request.rhs_ghosts[axis] != 0 || request.phi_ghosts[axis] < 1)
          throw std::invalid_argument(
              "geometric MG requires a ghost-free RHS and at least one solution ghost");
    } catch (...) {
      validation_error = std::current_exception();
    }
    if (all_reduce_max(validation_error ? 1L : 0L) != 0) {
      if (n_ranks() == 1 && validation_error)
        std::rethrow_exception(validation_error);
      throw std::runtime_error("geometric MG preparation failed collectively");
    }

    build_hierarchy_(request);
    prepared_operator_contract_ = make_materialized_elliptic_operator_contract(
        operator_identity(), geometry_, boundary_, rhs(), phi(), detail::options_contract<Dim>(options_));
  }

  GeometricMG(const Geometry<Dim>& geometry, const mesh::BoxArray<Dim>& layout,
              const mesh::Distribution<Dim>& distribution, Index<Dim> local_rank,
              PhysicalBoundaryConditions<Dim> boundary,
              GeometricMultigridOptions options = {})
      : GeometricMG(request_type{geometry,
                                 layout,
                                 distribution,
                                 local_rank,
                                 std::move(boundary),
                                 Extent<Dim>{},
                                 detail::unit_ghosts<Dim>(),
                                 detail::exact_layout_budget(layout)},
                    options) {}

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
    return make_expected_elliptic_operator_contract(
        operator_identity(), request, detail::options_contract<Dim>(options));
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
  int maximum_iterations() const noexcept { return options_.maximum_cycles; }
  Real residual() const noexcept { return last_report_.residual_norm; }
  const field_type& residual_field() const noexcept { return levels_.front()->residual; }
  const SolveReport& last_solve_report() const noexcept { return last_report_; }

  SolveReport solve() {
    Level& fine = *levels_.front();
    if (singular_) {
      project_mean_(fine.rhs, fine.geometry.domain());
      project_mean_(fine.phi, fine.geometry.domain());
    }

    compute_residual_(fine);
    const Real reference = global_norm_inf_(fine.residual);
    SolveReport report;
    report.reference_residual_norm = reference;
    report.residual_norm = reference;
    report.rel_residual = reference > Real(0) ? Real(1) : Real(0);
    report.evaluations = 1;
    const Real stop =
        std::max(options_.absolute_tolerance, options_.relative_tolerance * reference);
    if (!std::isfinite(static_cast<double>(reference))) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                         "geometric_mg_non_finite_initial_residual");
      last_report_ = report;
      return last_report_;
    }
    if (reference <= stop) {
      fill_ghosts_(fine);
      report.mark_solved("geometric_mg_initial_residual");
      last_report_ = report;
      return last_report_;
    }

    for (int cycle = 0; cycle < options_.maximum_cycles; ++cycle) {
      v_cycle_(0);
      if (singular_)
        project_mean_(fine.phi, fine.geometry.domain());
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
    HaloSchedule<Dim> halo_schedule;
    PreparedPhysicalBoundary<Dim> physical_boundary;
    std::unique_ptr<ExecutionLane> lane;
    std::unique_ptr<HaloExchange<Dim, MemorySpace>> exchange;

    Level(Geometry<Dim> level_geometry, mesh::BoxArray<Dim> layout,
          mesh::Distribution<Dim> distribution, Index<Dim> local_rank,
          PhysicalBoundaryConditions<Dim> level_boundary, std::uint64_t generation)
        : geometry(level_geometry),
          boundary(std::move(level_boundary)),
          phi(layout, distribution, local_rank, 1, detail::unit_ghosts<Dim>()),
          rhs(layout, distribution, local_rank, 1, Extent<Dim>{}),
          residual(layout, distribution, local_rank, 1, Extent<Dim>{}),
          scratch(layout, distribution, local_rank, 1, Extent<Dim>{}),
          correction(layout, distribution, local_rank, 1, Extent<Dim>{}),
          halo_schedule(prepare_halo_schedule(phi, geometry.domain(), boundary.topology(),
                                               detail::exact_halo_budget(layout, geometry.domain()))),
          physical_boundary(prepare_physical_boundary(
              geometry.domain(), detail::unit_ghosts<Dim>(), boundary,
              detail::exact_boundary_budget<Dim>())) {
      const bool remote = all_reduce_max(halo_schedule.has_remote_jobs() ? 1L : 0L) != 0;
      if (remote) {
        lane = std::make_unique<ExecutionLane>(ExecutionLane::duplicate_world_collectively(
            "pops.geometric-mg.nd" + std::to_string(Dim) + "/level/" +
            std::to_string(generation)));
        HaloExchangeContext context{};
        context.context_generation = generation + 1;
        context.schedule_generation = generation + 1;
        exchange =
            std::make_unique<HaloExchange<Dim, MemorySpace>>(halo_schedule, *lane, context);
      }
    }
  };

  static bool coarsenable_(const Geometry<Dim>& geometry,
                           const mesh::BoxArray<Dim>& layout,
                           const GeometricMultigridOptions& options) {
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

  void build_hierarchy_(const request_type& request) {
    Geometry<Dim> level_geometry = request.geometry;
    mesh::BoxArray<Dim> level_layout = request.boxes;
    mesh::Distribution<Dim> level_distribution = request.distribution;
    std::uint64_t generation = 0;
    while (true) {
      const bool correction_level = !levels_.empty();
      auto level_boundary = detail::boundary_for_geometry(
          request.boundary, level_geometry, correction_level);
      levels_.push_back(std::make_unique<Level>(level_geometry, level_layout, level_distribution,
                                                request.local_rank, std::move(level_boundary),
                                                generation));
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
      level_geometry = Geometry<Dim>::from_bounds(coarse_domain, level_geometry.lower(),
                                                  level_geometry.upper());
      ++generation;
    }
  }

  void fill_ghosts_(Level& level) {
    if (level.exchange)
      level.exchange->execute(level.phi, *level.lane);
    else
      fill_boundary(level.phi, level.halo_schedule);
    fill_physical_boundary(level.phi, level.physical_boundary);
  }

  void smooth_(Level& level, int sweeps) {
    for (int sweep = 0; sweep < sweeps; ++sweep) {
      fill_ghosts_(level);
      damped_jacobi_update_valid(level.phi, level.rhs, level.geometry, level.scratch,
                                 options_.jacobi_relaxation, options_.reaction);
      copy_scalar_valid(level.scratch, level.phi);
      if (singular_)
        project_mean_(level.phi, level.geometry.domain());
    }
  }

  void compute_residual_(Level& level) {
    fill_ghosts_(level);
    poisson_residual_valid(level.phi, level.rhs, level.geometry, level.residual,
                           options_.reaction);
  }

  void v_cycle_(std::size_t level_index) {
    Level& level = *levels_.at(level_index);
    if (level_index + 1 == levels_.size()) {
      smooth_(level, options_.bottom_sweeps);
      return;
    }

    smooth_(level, options_.pre_sweeps);
    compute_residual_(level);
    Level& coarse = *levels_.at(level_index + 1);
    coarse.phi.set_val(Real(0));
    const CopyScheduleBudget restriction_budget =
        detail::exact_copy_budget(coarse.rhs.layout(), coarsen(level.residual.layout(), 2));
    average_down(level.residual, coarse.rhs, 2, restriction_budget);
    if (singular_)
      project_mean_(coarse.rhs, coarse.geometry.domain());
    v_cycle_(level_index + 1);

    level.correction.set_val(Real(0));
    const CopyScheduleBudget prolongation_budget =
        detail::exact_copy_budget(coarsen(level.correction.layout(), 2), coarse.phi.layout());
    interpolate(coarse.phi, level.correction, 2, prolongation_budget);
    saxpy(level.phi, Real(1), level.correction);
    smooth_(level, options_.post_sweeps);
  }

  Real global_norm_inf_(const field_type& field) const {
    return static_cast<Real>(all_reduce_max(static_cast<double>(norm_inf(field))));
  }

  Real global_sum_(const field_type& field) const {
    const Real local = reduce_sum_local(field);
    if (!field.distribution().replicated())
      return static_cast<Real>(all_reduce_sum(static_cast<double>(local)));
    const Real minimum = static_cast<Real>(all_reduce_min(static_cast<double>(local)));
    const Real maximum = static_cast<Real>(all_reduce_max(static_cast<double>(local)));
    if (minimum != maximum)
      throw std::runtime_error("replicated geometric MG field differs between MPI ranks");
    return local;
  }

  void project_mean_(field_type& field, const Box<Dim>& domain) const {
    const std::int64_t cells = domain.numPts();
    if (cells <= 0)
      throw std::logic_error("geometric MG cannot project an empty domain");
    add_scalar_valid(field, -global_sum_(field) / static_cast<Real>(cells));
  }

  Geometry<Dim> geometry_;
  PhysicalBoundaryConditions<Dim> boundary_;
  GeometricMultigridOptions options_{};
  bool singular_ = false;
  std::vector<std::unique_ptr<Level>> levels_{};
  EllipticOperatorContract prepared_operator_contract_{};
  SolveReport last_report_{};
};

static_assert(EllipticSolver<GeometricMG<1>>);
static_assert(EllipticSolver<GeometricMG<2>>);
static_assert(EllipticSolver<GeometricMG<3>>);

}  // namespace pops::elliptic::mg
