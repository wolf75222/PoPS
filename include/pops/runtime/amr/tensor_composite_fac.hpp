/// @file
/// @brief Prepared full-tensor composite FAC kernel for exact-ranked Cartesian hierarchies.

#pragma once

#include <pops/amr/refinement_ratio.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/halo_exchange.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/amr/partitioned_region_transfer.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_workspace.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/numerics/elliptic/nd/cartesian_tensor_operator.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>
#include <pops/parallel/execution_lane.hpp>

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
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::program::tensor_fac {

template <int Dim, class MemorySpace>
struct LevelBinding {
  const Geometry<Dim>* geometry = nullptr;
  const PhysicalBoundaryConditions<Dim>* boundary = nullptr;
  std::array<MultiFab<Dim, MemorySpace>*, static_cast<std::size_t>(Dim* Dim)> coefficients{};
  MultiFab<Dim, MemorySpace>* rhs = nullptr;
  MultiFab<Dim, MemorySpace>* initial_guess = nullptr;
  MultiFab<Dim, MemorySpace>* solution = nullptr;
};

struct Controls {
  Real relative_tolerance = Real(1e-10);
  Real absolute_tolerance = Real(0);
  int maximum_iterations = 100;
  int fine_sweeps = 4;
  Real coarse_relative_tolerance = Real(1e-11);
  Real coarse_absolute_tolerance = Real(0);
  int coarse_cycles = 128;
};

namespace detail {

inline std::size_t checked_product(std::size_t left, std::size_t right, const char* message) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(message);
  return left * right;
}

inline std::size_t checked_sum(std::size_t left, std::size_t right, const char* message) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error(message);
  return left + right;
}

template <int Dim>
Extent<Dim> one_ghost() {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = 1;
  return result;
}

template <int Dim>
Extent<Dim> ratio_extent(const ::pops::amr::RefinementRatio<Dim>& ratio) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = ratio[axis];
  return result;
}

template <int Dim>
mesh::BoxArrayValidationBudget exact_layout_budget(const mesh::BoxArray<Dim>& layout) {
  const std::size_t boxes = layout.size();
  const std::size_t pairs =
      boxes < 2 ? 0 : checked_product(boxes, boxes - 1, "tensor FAC layout budget overflow") / 2;
  return {boxes, pairs};
}

template <int Dim>
HaloScheduleBudget exact_halo_budget(const mesh::BoxArray<Dim>& layout, const Box<Dim>& domain) {
  const std::size_t boxes = layout.size();
  const std::size_t pairs = checked_product(boxes, boxes, "tensor FAC halo pair overflow");
  std::size_t images = 1;
  for (int axis = 0; axis < Dim; ++axis)
    images = checked_product(images, std::size_t{3}, "tensor FAC halo image overflow");
  const std::size_t work = checked_product(pairs, images, "tensor FAC halo work overflow");
  const std::size_t jobs = checked_product(work, 4, "tensor FAC halo job overflow");
  const std::int64_t signed_cells = domain.numPts();
  if (signed_cells <= 0)
    throw std::invalid_argument("tensor FAC halo domain is empty");
  const std::size_t elements = checked_product(jobs, static_cast<std::size_t>(signed_cells),
                                               "tensor FAC halo element overflow");
  return {exact_layout_budget<Dim>(layout),
          work,
          jobs,
          images,
          checked_product(boxes, 2, "tensor FAC peer budget overflow"),
          elements,
          elements,
          elements};
}

template <int Dim>
BoundaryScheduleBudget exact_boundary_budget() {
  std::size_t entries = 1;
  for (int axis = 0; axis < Dim; ++axis)
    entries = checked_product(entries, std::size_t{3}, "tensor FAC boundary budget overflow");
  return {entries - 1};
}

template <int Dim>
PhysicalBoundaryConditions<Dim> boundary_with_values(const PhysicalBoundaryConditions<Dim>& source,
                                                     const Geometry<Dim>& geometry,
                                                     bool homogeneous, bool coefficient) {
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    spacing[axis] = geometry.spacing(axis);
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> face{axis, side};
      PhysicalBoundaryFace law = source.at(face);
      if (coefficient && !source.topology().is_periodic(face)) {
        law = PhysicalBoundaryFace{PhysicalBoundaryKind::constant_extrapolation};
      } else if (homogeneous) {
        law.value = Real(0);
      }
      faces[static_cast<std::size_t>(face.ordinal())] = law;
    }
  }
  return PhysicalBoundaryConditions<Dim>{source.topology(), faces, spacing};
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
    auto pieces = subtract_box<Dim>(region, cut);
    next.insert(next.end(), pieces.begin(), pieces.end());
  }
  regions = std::move(next);
}

template <int Dim>
Box<Dim> clipped_growth(const Box<Dim>& valid, const Box<Dim>& domain) {
  return valid.grow(1).intersect(domain);
}

template <int Dim, class Value>
struct SetKernel {
  FieldView<Value, Dim> field{};
  Real value = Real(0);
  POPS_HD void operator()(const Index<Dim>& index) const { field(index, 0) = value; }
};

template <int Dim>
struct CopyKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> source{};
  POPS_HD void operator()(const Index<Dim>& index) const {
    destination(index, 0) = source(index, 0);
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
struct ActiveMomentKernel {
  FieldView<const Real, Dim> values{};
  FieldView<const Real, Dim> active{};
  Real measure = Real(1);
  POPS_HD Real operator()(const Index<Dim>& index) const {
    return active(index, 0) >= Real(0.5) ? values(index, 0) * measure : Real(0);
  }
};

template <int Dim>
struct ActiveVolumeKernel {
  FieldView<const Real, Dim> active{};
  Real measure = Real(1);
  POPS_HD Real operator()(const Index<Dim>& index) const {
    return active(index, 0) >= Real(0.5) ? measure : Real(0);
  }
};

template <int Dim>
struct ActiveShiftKernel {
  FieldView<Real, Dim> values{};
  FieldView<const Real, Dim> active{};
  Real shift = Real(0);
  POPS_HD void operator()(const Index<Dim>& index) const {
    if (active(index, 0) >= Real(0.5))
      values(index, 0) -= shift;
  }
};

template <int Dim>
struct MaskKernel {
  FieldView<Real, Dim> values{};
  FieldView<const Real, Dim> covered{};
  POPS_HD void operator()(const Index<Dim>& index) const {
    if (covered(index, 0) >= Real(0.5))
      values(index, 0) = Real(0);
  }
};

template <int Dim>
using TensorStencil = elliptic::nd::CartesianTensorOperator<
    Dim, elliptic::nd::CartesianTensorDivergenceSign::negative_divergence,
    elliptic::nd::SplitCartesianTensorCoefficients<Dim>>;

template <int Dim>
struct ResidualKernel {
  FieldView<Real, Dim> residual{};
  FieldView<const Real, Dim> rhs{};
  FieldView<const Real, Dim> covered{};
  TensorStencil<Dim> stencil{};
  bool mask_covered = true;
  POPS_HD void operator()(const Index<Dim>& index) const {
    residual(index, 0) = mask_covered && covered(index, 0) >= Real(0.5)
                             ? Real(0)
                             : rhs(index, 0) - stencil.image(index);
  }
};

/// One colour of a tensor-stencil Gauss-Seidel sweep.  A full tensor stencil couples diagonal
/// neighbours, so two colours are insufficient: every axis-parity combination is its own colour.
/// Updating one colour at a time keeps the in-place sweep race-free for Dim=1, 2, and 3.
template <int Dim>
struct ColoredGaussSeidelKernel {
  FieldView<Real, Dim> iterate{};
  FieldView<const Real, Dim> rhs{};
  FieldView<const Real, Dim> covered{};
  TensorStencil<Dim> stencil{};
  unsigned colour = 0;
  Real relaxation = Real(1);
  bool mask_covered = true;
  POPS_HD void operator()(const Index<Dim>& index) const {
    unsigned index_colour = 0;
    for (int axis = 0; axis < Dim; ++axis)
      index_colour |= (static_cast<unsigned>(index[axis]) & 1u) << axis;
    if (index_colour != colour)
      return;
    if (mask_covered && covered(index, 0) >= Real(0.5))
      return;
    const Real diagonal = stencil.diagonal(index);
    iterate(index, 0) += relaxation * (rhs(index, 0) - stencil.image(index)) / diagonal;
  }
};

template <int Dim>
struct LinearInterpolationKernel {
  FieldView<const Real, Dim> coarse{};
  FieldView<Real, Dim> fine{};
  Box<Dim> coarse_domain{};
  Box<Dim> fine_domain{};
  ::pops::amr::RefinementRatio<Dim> ratio{};
  POPS_HD void operator()(const Index<Dim>& index) const {
    Index<Dim> parent{};
    std::array<Real, static_cast<std::size_t>(Dim)> offset{};
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t relative = static_cast<std::int64_t>(index[axis]) - fine_domain.lo[axis];
      std::int64_t quotient = relative / ratio[axis];
      if (relative % ratio[axis] < 0)
        --quotient;
      parent[axis] = static_cast<int>(static_cast<std::int64_t>(coarse_domain.lo[axis]) + quotient);
      offset[axis] = (static_cast<Real>(relative) + Real(0.5)) / static_cast<Real>(ratio[axis]) -
                     (static_cast<Real>(quotient) + Real(0.5));
    }
    Real value = coarse(parent, 0);
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> lower = parent;
      Index<Dim> upper = parent;
      --lower[axis];
      ++upper[axis];
      Real slope = Real(0);
      if (parent[axis] == coarse_domain.lo[axis])
        slope = coarse(upper, 0) - coarse(parent, 0);
      else if (parent[axis] == coarse_domain.hi[axis])
        slope = coarse(parent, 0) - coarse(lower, 0);
      else
        slope = Real(0.5) * (coarse(upper, 0) - coarse(lower, 0));
      value += offset[axis] * slope;
    }
    fine(index, 0) = value;
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
    for (std::int64_t child = 0; child < ratio.child_count(); ++child) {
      std::int64_t ordinal = child;
      Index<Dim> fine_index = base;
      for (int axis = 0; axis < Dim; ++axis) {
        fine_index[axis] += static_cast<int>(ordinal % ratio[axis]);
        ordinal /= ratio[axis];
      }
      sum += fine(fine_index, 0);
    }
    coarse(parent, 0) = sum * inverse_children;
  }
};

template <int Dim>
inline void copy_region(FieldView<Real, Dim> destination, FieldView<const Real, Dim> source,
                        const Box<Dim>& region) {
  if (!region.empty())
    for_each_cell(region, CopyKernel<Dim>{destination, source});
}

template <int Dim, class MemorySpace>
void copy_valid(MultiFab<Dim, MemorySpace>& destination, const MultiFab<Dim, MemorySpace>& source) {
  if (destination.layout() != source.layout() ||
      destination.distribution() != source.distribution() ||
      destination.local_rank() != source.local_rank() || destination.ncomp() != 1 ||
      source.ncomp() != 1)
    throw std::invalid_argument("tensor FAC copy requires one exact scalar vector space");
  for (std::size_t local = 0; local < destination.local_size(); ++local)
    copy_region<Dim>(destination.fab(local).view(), std::as_const(source.fab(local)).view(),
                     destination.box(local));
  ::pops::device_fence();
}

inline void validate_controls(const Controls& controls) {
  if (controls.maximum_iterations < 1 || controls.fine_sweeps < 1 || controls.coarse_cycles < 1 ||
      !std::isfinite(static_cast<double>(controls.relative_tolerance)) ||
      controls.relative_tolerance <= Real(0) ||
      !std::isfinite(static_cast<double>(controls.absolute_tolerance)) ||
      controls.absolute_tolerance < Real(0) ||
      !std::isfinite(static_cast<double>(controls.coarse_relative_tolerance)) ||
      controls.coarse_relative_tolerance <= Real(0) ||
      !std::isfinite(static_cast<double>(controls.coarse_absolute_tolerance)) ||
      controls.coarse_absolute_tolerance < Real(0))
    throw std::invalid_argument("tensor FAC controls are invalid");
}

}  // namespace detail

/// A prepared dimension-generic FAC cycle for ``-div(A grad(phi)) = rhs``.
///
/// Coefficients are row-major ``A[row * Dim + column]``.  The conservative flux stencil includes
/// every cross derivative, so its support expands with the exact compile-time dimension.
/// Same-level halos and every partitioned coarse/fine transfer borrow one authenticated
/// ExecutionLane;
/// replicated level zero is retained as an explicit capability and refined contributions are
/// broadcast from their unique owners into that replicated parent.  No solve-time storage
/// allocation is permitted.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class FullTensorCompositeFac {
 public:
  using field_type = MultiFab<Dim, MemorySpace>;

  FullTensorCompositeFac(std::span<const LevelBinding<Dim, MemorySpace>> bindings,
                         std::span<const ::pops::amr::RefinementRatio<Dim>> ratios,
                         const ExecutionLane& lane)
      : bindings_(bindings.begin(), bindings.end()),
        ratios_(ratios.begin(), ratios.end()),
        lane_(&lane),
        lane_borrow_(lane.borrow_immutably()) {
    static_assert(
        Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, MemorySpace>::accessible,
        "FullTensorCompositeFac requires DefaultExecutionSpace access to its memory space");
    std::exception_ptr local_error;
    try {
      validate_bindings_();
      levels_.reserve(bindings_.size());
      for (std::size_t level = 0; level < bindings_.size(); ++level)
        levels_.push_back(std::make_unique<Level>(bindings_[level], level == 0));
      connections_.reserve(ratios_.size());
      for (std::size_t parent = 0; parent < ratios_.size(); ++parent) {
        connections_.push_back(std::make_unique<Connection>(*levels_[parent], *levels_[parent + 1],
                                                            ratios_[parent], parent));
        mark_coverage_(*levels_[parent], *levels_[parent + 1], ratios_[parent]);
      }
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("dimension-generic tensor FAC preparation failed collectively");
    }

    exact_contract_ = build_exact_contract_();
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"pops-nd-tensor-fac", std::string_view(exact_contract_)}}, lane))
      throw std::invalid_argument(
          "dimension-generic tensor FAC hierarchy differs between MPI ranks");
    for (std::size_t connection = 0; connection < connections_.size(); ++connection)
      if (!all_ranks_agree_exact_ordered_byte_pairs(
              {{"pops-nd-tensor-parent-gather",
                std::string_view(connections_[connection]->gather_contract)},
               {"pops-nd-tensor-fine-restriction",
                std::string_view(connections_[connection]->restriction_contract)}},
              lane))
        throw std::invalid_argument(
            "dimension-generic tensor FAC coarse/fine schedule differs between MPI ranks");

    // Singularity is a property of the authenticated boundary contract. Materialize its
    // nullspace workspace only after that contract has agreed, so no rank can enter a different
    // collective trace and no first solve can grow the resident footprint.
    prepare_nullspace_();
    for (auto& connection : connections_)
      connection->attach_lane(lane);
    for (std::size_t level = 0; level < levels_.size(); ++level) {
      const bool remote =
          all_reduce_max(levels_[level]->halo_schedule.has_remote_jobs() ? 1L : 0L, lane) != 0;
      if (remote) {
        HaloExchangeContext context{};
        context.context_generation = level + 1;
        context.schedule_generation = level + 1;
        levels_[level]->halo_exchange.emplace(levels_[level]->halo_schedule, lane, context);
      }
    }
  }

  FullTensorCompositeFac(const FullTensorCompositeFac&) = delete;
  FullTensorCompositeFac& operator=(const FullTensorCompositeFac&) = delete;
  FullTensorCompositeFac(FullTensorCompositeFac&&) = delete;
  FullTensorCompositeFac& operator=(FullTensorCompositeFac&&) = delete;

  std::string_view exact_prepared_contract() const noexcept { return exact_contract_; }

  bool borrows_execution_lane() const noexcept { return lane_ != nullptr; }

  bool has_remote_same_level_halo() const noexcept {
    return std::any_of(levels_.begin(), levels_.end(),
                       [](const auto& level) { return level->halo_schedule.has_remote_jobs(); });
  }

  bool has_remote_parent_gather() const noexcept {
    return std::any_of(connections_.begin(), connections_.end(), [](const auto& connection) {
      return connection->gather && connection->gather->plan().has_remote_jobs();
    });
  }

  bool has_remote_fine_restriction() const noexcept {
    return std::any_of(connections_.begin(), connections_.end(), [](const auto& connection) {
      return connection->restriction && connection->restriction->plan().has_remote_jobs();
    });
  }

  bool uses_replicated_parent_restriction() const noexcept {
    return std::any_of(connections_.begin(), connections_.end(),
                       [](const auto& connection) { return connection->replicated_parent; });
  }

  /// Logical storage retained by the native FAC preparation, excluding this object itself.
  /// A singular nullspace workspace is materialized at bind and included in the exact receipt.
  [[nodiscard]] PreparedResidentStorage resident_storage() const;

  /// Conservative finite bound for a not-yet-materialized FAC image.
  ///
  /// This is intentionally expressed in configured topology terms rather than accepting an
  /// already-built ``LevelBinding``.  It is used by the type-erased tensor-provider boundary to
  /// seal capacity before a forward hierarchy exists.  The result covers every allocation
  /// represented by ``resident_storage()`` plus the prepared transfer arenas which the latter
  /// deliberately treats as transport implementation detail.  Arithmetic overflow is a refusal,
  /// never a wraparound capacity promise.
  [[nodiscard]] static std::uint64_t configured_resident_storage_upper_bound(
      std::span<const std::uint64_t> level_cell_bounds, std::span<const std::uint64_t> patch_bounds,
      std::span<const std::uint64_t> parent_child_pair_bounds, std::uint64_t rank_bound,
      std::string_view execution_lane_identity);

  SolveReport solve(const Controls& controls, const ExecutionLane& lane) {
    if (all_reduce_max(&lane == lane_ ? 0L : 1L, *lane_) != 0)
      throw std::invalid_argument(
          "dimension-generic tensor FAC requires its prepared execution lane");
    detail::validate_controls(controls);
    for (std::size_t level = 0; level < levels_.size(); ++level)
      detail::copy_valid<Dim>(*levels_[level]->binding.solution,
                              *levels_[level]->binding.initial_guess);
    try {
      if (nullspace_workspace_) {
        nullspace_workspace_->require_compatible(nullspace_rhs_);
        nullspace_workspace_->apply_gauge(nullspace_candidates_);
      }
    } catch (const FieldNullspaceIncompatibleRhs& error) {
      SolveReport report;
      report.mark_failed(SolveStatus::kIncompatibleRhs, SolveAction::kFailRun, error.what());
      return report;
    } catch (const FieldNullspaceInvalidEvaluation& error) {
      SolveReport report;
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun, error.what());
      return report;
    }
    fill_all_coefficient_ghosts_();
    if (!coefficients_are_elliptic_()) {
      SolveReport report;
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                         "nd_tensor_fac_non_elliptic_coefficient");
      return report;
    }
    average_solution_down_();
    compute_composite_residual_();
    const Real reference = composite_residual_norm_();
    SolveReport report;
    report.reference_residual_norm = reference;
    report.residual_norm = reference;
    report.rel_residual = reference > Real(0) ? Real(1) : Real(0);
    report.evaluations = 1;
    const Real stop =
        std::max(controls.absolute_tolerance, controls.relative_tolerance * reference);
    if (!std::isfinite(static_cast<double>(reference))) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                         "nd_tensor_fac_non_finite_initial_residual");
      return report;
    }
    if (reference <= stop) {
      fill_all_solution_ghosts_();
      report.mark_solved("nd_tensor_fac_initial_residual");
      return report;
    }

    const int pre = (controls.fine_sweeps + 1) / 2;
    const int post = controls.fine_sweeps / 2;
    for (int iteration = 0; iteration < controls.maximum_iterations; ++iteration) {
      for (std::size_t level = 1; level < levels_.size(); ++level)
        smooth_(level, *levels_[level]->binding.solution, *levels_[level]->binding.rhs, pre, true,
                false);

      compute_composite_residual_();
      restrict_residual_tower_();
      solve_coarse_correction_(controls);
      report.step_norm = global_norm_inf_(levels_.front()->correction);
      add_active_(*levels_.front(), levels_.front()->correction);
      prolong_correction_tower_();
      for (std::size_t level = 1; level < levels_.size(); ++level)
        smooth_(level, *levels_[level]->binding.solution, *levels_[level]->binding.rhs, post, true,
                false);
      average_solution_down_();
      if (nullspace_workspace_)
        nullspace_workspace_->apply_gauge(nullspace_candidates_);

      compute_composite_residual_();
      ++report.evaluations;
      report.iters = iteration + 1;
      report.residual_norm = composite_residual_norm_();
      report.rel_residual = report.residual_norm / reference;
      if (!std::isfinite(static_cast<double>(report.residual_norm))) {
        report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                           "nd_tensor_fac_non_finite_iteration");
        return report;
      }
      if (report.residual_norm <= stop) {
        fill_all_solution_ghosts_();
        report.mark_solved("nd_tensor_fac_converged");
        return report;
      }
    }
    report.mark_failed(SolveStatus::kIterationLimit, SolveAction::kFailRun,
                       "nd_tensor_fac_iteration_limit");
    return report;
  }

 private:
  struct Level {
    LevelBinding<Dim, MemorySpace> binding;
    field_type residual;
    field_type scratch;
    field_type correction;
    field_type covered;
    field_type active;
    HaloSchedule<Dim> halo_schedule;
    PreparedPhysicalBoundary<Dim> physical_boundary;
    PreparedPhysicalBoundary<Dim> homogeneous_boundary;
    PreparedPhysicalBoundary<Dim> coefficient_boundary;
    std::optional<HaloExchange<Dim, MemorySpace>> halo_exchange{};

    Level(LevelBinding<Dim, MemorySpace> source, bool full_domain)
        : binding(source),
          residual(source.solution->layout(), source.solution->distribution(),
                   source.solution->local_rank(), 1, Extent<Dim>{}),
          scratch(source.solution->layout(), source.solution->distribution(),
                  source.solution->local_rank(), 1, Extent<Dim>{}),
          correction(source.solution->layout(), source.solution->distribution(),
                     source.solution->local_rank(), 1, detail::one_ghost<Dim>()),
          covered(source.solution->layout(), source.solution->distribution(),
                  source.solution->local_rank(), 1, Extent<Dim>{}),
          active(source.solution->layout(), source.solution->distribution(),
                 source.solution->local_rank(), 1, Extent<Dim>{}),
          halo_schedule(prepare_halo_schedule(
              *source.solution, source.geometry->domain(), source.boundary->topology(),
              full_domain ? HaloLayoutCoverage::full_domain : HaloLayoutCoverage::sparse_level,
              detail::exact_halo_budget<Dim>(source.solution->layout(),
                                             source.geometry->domain()))),
          physical_boundary(prepare_physical_boundary(source.geometry->domain(),
                                                      detail::one_ghost<Dim>(), *source.boundary,
                                                      detail::exact_boundary_budget<Dim>())),
          homogeneous_boundary(prepare_physical_boundary(
              source.geometry->domain(), detail::one_ghost<Dim>(),
              detail::boundary_with_values<Dim>(*source.boundary, *source.geometry, true, false),
              detail::exact_boundary_budget<Dim>())),
          coefficient_boundary(prepare_physical_boundary(
              source.geometry->domain(), detail::one_ghost<Dim>(),
              detail::boundary_with_values<Dim>(*source.boundary, *source.geometry, false, true),
              detail::exact_boundary_budget<Dim>())) {
      residual.set_val(Real(0));
      scratch.set_val(Real(0));
      correction.set_val(Real(0));
      covered.set_val(Real(0));
      active.set_val(Real(1));
    }
  };

  struct Connection {
    using transfer_job = elliptic::amr::partitioned_transfer::RegionTransferJob<Dim>;
    using transfer_plan = elliptic::amr::partitioned_transfer::RegionTransferPlan<Dim>;
    using transport_type = elliptic::amr::partitioned_transfer::RegionTransport<Dim, MemorySpace>;
    using host_mirror_type = typename Fab<Dim, MemorySpace>::host_mirror_type;

    struct ScratchPatch {
      std::size_t fine_patch = 0;
      Fab<Dim, MemorySpace> parent_staging{};
      Fab<Dim, MemorySpace> restricted{};
      std::vector<Box<Dim>> ghost_regions{};
      std::optional<host_mirror_type> restricted_host{};
      std::vector<Real> broadcast_buffer{};

      ScratchPatch(std::size_t patch, const Box<Dim>& staging, const Box<Dim>& restriction,
                   std::vector<Box<Dim>> regions, bool broadcast)
          : fine_patch(patch),
            parent_staging(staging, 1, Extent<Dim>{}),
            restricted(restriction, 1, Extent<Dim>{}),
            ghost_regions(std::move(regions)) {
        if (broadcast) {
          restricted_host.emplace(restricted.create_host_mirror());
          broadcast_buffer.resize(restricted.size());
        }
      }
    };

    Level* parent = nullptr;
    Level* child = nullptr;
    ::pops::amr::RefinementRatio<Dim> ratio{};
    bool replicated_parent = false;
    std::vector<ScratchPatch> scratch{};
    std::vector<std::size_t> scratch_by_fine_patch{};
    std::unique_ptr<transport_type> gather{};
    std::unique_ptr<transport_type> restriction{};
    std::string gather_contract{};
    std::string restriction_contract{};
    const ExecutionLane* lane = nullptr;

    static constexpr std::size_t no_scratch = std::numeric_limits<std::size_t>::max();

    Connection(Level& parent_level, Level& child_level,
               ::pops::amr::RefinementRatio<Dim> level_ratio, std::size_t ordinal)
        : parent(&parent_level),
          child(&child_level),
          ratio(level_ratio),
          replicated_parent(parent_level.binding.solution->distribution().replicated()) {
      const Extent<Dim> ratio_value = detail::ratio_extent<Dim>(ratio);
      const std::size_t fine_count = child->binding.solution->layout().size();
      scratch_by_fine_patch.assign(fine_count, no_scratch);
      scratch.reserve(replicated_parent ? fine_count : child->binding.solution->local_size());
      for (std::size_t fine_patch = 0; fine_patch < fine_count; ++fine_patch) {
        if (!replicated_parent && !child->binding.solution->contains_local(fine_patch))
          continue;
        const Box<Dim>& valid = child->binding.solution->layout()[fine_patch];
        const Box<Dim> staging =
            coarsen(detail::clipped_growth<Dim>(valid, child->binding.geometry->domain()),
                    ratio_value)
                .grow(1)
                .intersect(parent->binding.geometry->domain());
        const Box<Dim> restricted_box = coarsen(valid, ratio_value);
        std::vector<Box<Dim>> pending{
            detail::clipped_growth<Dim>(valid, child->binding.geometry->domain())};
        for (const Box<Dim>& peer : child->binding.solution->layout().boxes())
          detail::subtract_from<Dim>(pending, peer);
        scratch_by_fine_patch[fine_patch] = scratch.size();
        scratch.emplace_back(fine_patch, staging, restricted_box, std::move(pending),
                             replicated_parent);
      }

      if (replicated_parent) {
        gather_contract = "pops.nd-tensor-fac.replicated-parent-gather/" + std::to_string(ordinal);
        restriction_contract =
            "pops.nd-tensor-fac.replicated-parent-restriction/" + std::to_string(ordinal);
        return;
      }

      std::vector<transfer_job> gather_jobs;
      std::vector<transfer_job> restriction_jobs;
      const std::size_t parent_count = parent->binding.solution->layout().size();
      gather_jobs.reserve(
          detail::checked_product(fine_count, parent_count, "tensor FAC gather pair overflow"));
      restriction_jobs.reserve(gather_jobs.capacity());
      for (std::size_t fine_patch = 0; fine_patch < fine_count; ++fine_patch) {
        const Box<Dim>& valid = child->binding.solution->layout()[fine_patch];
        const Box<Dim> staging =
            coarsen(detail::clipped_growth<Dim>(valid, child->binding.geometry->domain()),
                    ratio_value)
                .grow(1)
                .intersect(parent->binding.geometry->domain());
        const Box<Dim> footprint = coarsen(valid, ratio_value);
        std::int64_t gathered_cells = 0;
        std::int64_t restricted_cells = 0;
        for (std::size_t parent_patch = 0; parent_patch < parent_count; ++parent_patch) {
          const Box<Dim> gathered =
              staging.intersect(parent->binding.solution->layout()[parent_patch]);
          if (!gathered.empty()) {
            gathered_cells += gathered.numPts();
            gather_jobs.push_back(transfer_job{
                parent_patch, fine_patch,
                parent->binding.solution->distribution().owner(parent_patch),
                child->binding.solution->distribution().owner(fine_patch), gathered, gathered});
          }
          const Box<Dim> restricted_region =
              footprint.intersect(parent->binding.solution->layout()[parent_patch]);
          if (!restricted_region.empty()) {
            restricted_cells += restricted_region.numPts();
            restriction_jobs.push_back(transfer_job{
                fine_patch, parent_patch, child->binding.solution->distribution().owner(fine_patch),
                parent->binding.solution->distribution().owner(parent_patch), restricted_region,
                restricted_region});
          }
        }
        if (gathered_cells != staging.numPts() || restricted_cells != footprint.numPts())
          throw std::invalid_argument(
              "dimension-generic tensor FAC parent layout does not cover a refined footprint");
      }
      const auto gather_budget = exact_transfer_budget_(gather_jobs);
      const auto restriction_budget = exact_transfer_budget_(restriction_jobs);
      gather = std::make_unique<transport_type>(transfer_plan{
          parent->binding.solution->rank_space(), parent->binding.solution->local_rank(), 1,
          std::move(gather_jobs), gather_budget});
      restriction = std::make_unique<transport_type>(transfer_plan{
          parent->binding.solution->rank_space(), parent->binding.solution->local_rank(), 1,
          std::move(restriction_jobs), restriction_budget});
      gather_contract =
          gather->plan().exact_contract("nd-tensor-parent-gather/" + std::to_string(ordinal));
      restriction_contract = restriction->plan().exact_contract("nd-tensor-fine-restriction/" +
                                                                std::to_string(ordinal));
    }

    static elliptic::amr::partitioned_transfer::RegionTransferBudget exact_transfer_budget_(
        const std::vector<transfer_job>& jobs) {
      std::size_t elements = 0;
      for (const transfer_job& job : jobs)
        elements =
            detail::checked_sum(elements, static_cast<std::size_t>(job.source_region.numPts()),
                                "tensor FAC transfer element budget overflow");
      return {jobs.size(),
              detail::checked_product(parent_rank_count_(jobs), 2,
                                      "tensor FAC transfer peer budget overflow"),
              elements, elements, elements};
    }

    static std::size_t parent_rank_count_(const std::vector<transfer_job>& jobs) {
      std::vector<Index<Dim>> ranks;
      ranks.reserve(detail::checked_product(jobs.size(), 2, "tensor FAC rank budget overflow"));
      for (const transfer_job& job : jobs) {
        ranks.push_back(job.source_rank);
        ranks.push_back(job.destination_rank);
      }
      std::sort(ranks.begin(), ranks.end(), [](const Index<Dim>& left, const Index<Dim>& right) {
        for (int axis = Dim; axis-- > 0;) {
          if (left[axis] != right[axis])
            return left[axis] < right[axis];
        }
        return false;
      });
      ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
      return std::max<std::size_t>(1, ranks.size());
    }

    void attach_lane(const ExecutionLane& execution_lane) {
      lane = &execution_lane;
      if (gather)
        gather->attach_lane(execution_lane);
      if (restriction)
        restriction->attach_lane(execution_lane);
    }

    ScratchPatch& scratch_for(std::size_t fine_patch) {
      const std::size_t local = scratch_by_fine_patch.at(fine_patch);
      if (local == no_scratch)
        throw std::out_of_range("dimension-generic tensor FAC scratch patch is not materialized");
      return scratch.at(local);
    }

    void gather_parent(const field_type& source) {
      if (replicated_parent) {
        for (ScratchPatch& patch : scratch) {
          patch.parent_staging.set_val(Real(0));
          for (std::size_t parent_patch = 0; parent_patch < source.layout().size();
               ++parent_patch) {
            const Box<Dim> region =
                patch.parent_staging.box().intersect(source.layout()[parent_patch]);
            detail::copy_region(patch.parent_staging.view(),
                                std::as_const(source.fab_global(parent_patch)).view(), region);
          }
        }
        ::pops::device_fence();
        return;
      }
      auto source_view = [&source](const transfer_job& job) -> FieldView<const Real, Dim> {
        return std::as_const(source.fab_global(job.source_patch)).view();
      };
      auto destination_view = [this](const transfer_job& job) -> FieldView<Real, Dim> {
        return scratch_for(job.destination_patch).parent_staging.view();
      };
      gather->execute(source_view, destination_view);
    }

    void interpolate_ghosts(field_type& destination) {
      for (ScratchPatch& patch : scratch) {
        if (!destination.contains_local(patch.fine_patch))
          continue;
        const auto coarse = std::as_const(patch.parent_staging).view();
        auto fine = destination.fab_global(patch.fine_patch).view();
        for (const Box<Dim>& region : patch.ghost_regions)
          for_each_cell(region, detail::LinearInterpolationKernel<Dim>{
                                    coarse, fine, parent->binding.geometry->domain(),
                                    child->binding.geometry->domain(), ratio});
      }
      ::pops::device_fence();
    }

    void prolong_valid(field_type& destination) {
      for (ScratchPatch& patch : scratch) {
        if (!destination.contains_local(patch.fine_patch))
          continue;
        const auto coarse = std::as_const(patch.parent_staging).view();
        auto fine = destination.fab_global(patch.fine_patch).view();
        for_each_cell(
            destination.fab_global(patch.fine_patch).box(),
            detail::LinearInterpolationKernel<Dim>{coarse, fine, parent->binding.geometry->domain(),
                                                   child->binding.geometry->domain(), ratio});
      }
      ::pops::device_fence();
    }

    void restrict_into(const field_type& source, field_type& destination) {
      const Real inverse_children = Real(1) / static_cast<Real>(ratio.child_count());
      for (ScratchPatch& patch : scratch) {
        if (!source.contains_local(patch.fine_patch))
          continue;
        for_each_cell(patch.restricted.box(),
                      detail::RestrictionKernel<Dim>{
                          std::as_const(source.fab_global(patch.fine_patch)).view(),
                          patch.restricted.view(), parent->binding.geometry->domain(),
                          child->binding.geometry->domain(), ratio, inverse_children});
      }
      ::pops::device_fence();
      if (replicated_parent) {
        broadcast_and_publish_restriction_(destination);
        return;
      }
      auto source_view = [this](const transfer_job& job) -> FieldView<const Real, Dim> {
        return std::as_const(scratch_for(job.source_patch).restricted).view();
      };
      auto destination_view = [&destination](const transfer_job& job) -> FieldView<Real, Dim> {
        return destination.fab_global(job.destination_patch).view();
      };
      restriction->execute(source_view, destination_view);
    }

    void broadcast_and_publish_restriction_(field_type& destination) {
      if (lane == nullptr)
        throw std::logic_error("dimension-generic tensor FAC replicated restriction has no lane");
      for (ScratchPatch& patch : scratch) {
        const Index<Dim>& owner = child->binding.solution->distribution().owner(patch.fine_patch);
        const int root = static_cast<int>(child->binding.solution->rank_space().linear_rank(owner));
        long packing_failure = 0;
        try {
          if (!patch.restricted_host ||
              patch.broadcast_buffer.size() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error(
                "dimension-generic tensor restriction has an invalid broadcast allocation");
          if (lane->rank() == root) {
            patch.restricted.copy_to_host(*patch.restricted_host);
            for (std::size_t element = 0; element < patch.broadcast_buffer.size(); ++element)
              patch.broadcast_buffer[element] = (*patch.restricted_host)(element);
          }
        } catch (...) {
          packing_failure = 1;
        }
        if (all_reduce_max(packing_failure, *lane) != 0)
          throw std::runtime_error(
              "dimension-generic tensor replicated restriction packing failed collectively");
#ifdef POPS_HAS_MPI
        if (lane->size() > 1) {
          const int code = MPI_Bcast(patch.broadcast_buffer.data(),
                                     static_cast<int>(patch.broadcast_buffer.size()),
                                     pops::mpi_real_datatype(), root, lane->native_handle());
          if (all_reduce_max(code == MPI_SUCCESS ? 0L : 1L, *lane) != 0)
            throw std::runtime_error(
                "dimension-generic tensor replicated restriction broadcast failed collectively");
        }
#endif
        long publication_failure = 0;
        try {
          if (lane->rank() != root) {
            for (std::size_t element = 0; element < patch.broadcast_buffer.size(); ++element)
              (*patch.restricted_host)(element) = patch.broadcast_buffer[element];
            patch.restricted.copy_from_host(*patch.restricted_host);
          }
          for (std::size_t parent_patch = 0; parent_patch < destination.layout().size();
               ++parent_patch) {
            const Box<Dim> region =
                patch.restricted.box().intersect(destination.layout()[parent_patch]);
            detail::copy_region(destination.fab_global(parent_patch).view(),
                                std::as_const(patch.restricted).view(), region);
          }
          ::pops::device_fence();
        } catch (...) {
          publication_failure = 1;
        }
        if (all_reduce_max(publication_failure, *lane) != 0)
          throw std::runtime_error(
              "dimension-generic tensor replicated restriction publication failed collectively");
      }
    }
  };

  void validate_bindings_() const {
    if (bindings_.size() < 2 || ratios_.size() + 1 != bindings_.size())
      throw std::invalid_argument(
          "dimension-generic tensor FAC requires a populated refined hierarchy");
    const auto& rank_space = bindings_.front().solution->rank_space();
    const Index<Dim> local_rank = bindings_.front().solution->local_rank();
    if (rank_space.size() != static_cast<std::size_t>(lane_->size()) ||
        rank_space.linear_rank(local_rank) != static_cast<std::size_t>(lane_->rank()))
      throw std::invalid_argument(
          "dimension-generic tensor FAC process space differs from its execution lane");
    for (std::size_t level = 0; level < bindings_.size(); ++level) {
      const auto& binding = bindings_[level];
      if (binding.geometry == nullptr || binding.boundary == nullptr || binding.rhs == nullptr ||
          binding.initial_guess == nullptr || binding.solution == nullptr ||
          std::any_of(binding.coefficients.begin(), binding.coefficients.end(),
                      [](const field_type* value) { return value == nullptr; }))
        throw std::invalid_argument("dimension-generic tensor FAC level binding is incomplete");
      const field_type& solution = *binding.solution;
      if (solution.ncomp() != 1 || binding.rhs->ncomp() != 1 ||
          binding.initial_guess->ncomp() != 1 || solution.layout().empty() ||
          solution.rank_space() != rank_space || solution.local_rank() != local_rank ||
          !solution.distribution().matches_layout(solution.layout()) ||
          !solution.layout().is_disjoint_within(
              binding.geometry->domain(), detail::exact_layout_budget<Dim>(solution.layout())))
        throw std::invalid_argument(
            "dimension-generic tensor FAC level has an invalid exact layout");
      if (level == 0 &&
          !solution.layout().tiles_exactly(binding.geometry->domain(),
                                           detail::exact_layout_budget<Dim>(solution.layout())))
        throw std::invalid_argument(
            "dimension-generic tensor FAC coarse layout must tile its domain");
      for (int axis = 0; axis < Dim; ++axis) {
        if (solution.ghosts()[axis] < 1 || binding.initial_guess->ghosts()[axis] < 1 ||
            binding.rhs->ghosts()[axis] != 0 ||
            binding.boundary->spacing()[axis] != binding.geometry->spacing(axis))
          throw std::invalid_argument(
              "dimension-generic tensor FAC field ghosts or boundary spacing are incompatible");
      }
      const auto exact_shape = [&solution](const field_type& field, bool require_ghost) {
        if (field.layout() != solution.layout() ||
            field.distribution() != solution.distribution() ||
            field.local_rank() != solution.local_rank() || field.ncomp() != 1)
          return false;
        if (require_ghost)
          for (int axis = 0; axis < Dim; ++axis)
            if (field.ghosts()[axis] < 1)
              return false;
        return true;
      };
      if (!exact_shape(*binding.rhs, false) || !exact_shape(*binding.initial_guess, true) ||
          std::any_of(binding.coefficients.begin(), binding.coefficients.end(),
                      [&](const field_type* field) { return !exact_shape(*field, true); }))
        throw std::invalid_argument(
            "dimension-generic tensor FAC coefficients and vectors do not share one exact level "
            "space");
      for (int axis = 0; axis < Dim; ++axis)
        for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
          const Face<Dim> face{axis, side};
          if (!binding.boundary->topology().is_periodic(face) &&
              binding.boundary->at(face).kind == PhysicalBoundaryKind::external)
            throw std::invalid_argument(
                "dimension-generic tensor FAC requires an authored law on every physical face");
        }
      if (level == 0)
        continue;
      if (solution.distribution().replicated())
        throw std::invalid_argument(
            "dimension-generic tensor FAC refined levels require unique partitioned ownership");
      const Extent<Dim> ratio = detail::ratio_extent(ratios_[level - 1]);
      if (ratios_[level - 1].is_identity())
        throw std::invalid_argument(
            "dimension-generic tensor FAC transition must refine at least one axis");
      if (*binding.geometry != bindings_[level - 1].geometry->refine(ratio) ||
          binding.boundary->topology() != bindings_[level - 1].boundary->topology())
        throw std::invalid_argument(
            "dimension-generic tensor FAC adjacent geometry or topology is not an exact "
            "refinement");
      for (const Box<Dim>& patch : solution.layout().boxes()) {
        if (refine(coarsen(patch, ratio), ratio) != patch)
          throw std::invalid_argument(
              "dimension-generic tensor FAC fine patch is not refinement-aligned");
        const Box<Dim> grown = patch.grow(1);
        for (int axis = 0; axis < Dim; ++axis)
          if ((grown.lo[axis] < binding.geometry->domain().lo[axis] ||
               grown.hi[axis] > binding.geometry->domain().hi[axis]) &&
              binding.boundary->topology().is_periodic(Face<Dim>{axis, BoundarySide::lower}))
            throw std::invalid_argument(
                "dimension-generic tensor FAC periodic sparse patches may not cross the domain "
                "seam");
      }
    }
  }

  std::string build_exact_contract_() const {
    ExactContractBuilder contract;
    contract.text("pops.runtime.amr.nd-tensor-composite-fac")
        .scalar(std::uint32_t{2})
        .scalar(std::int32_t{Dim})
        .text(lane_->identity())
        .scalar(static_cast<std::uint64_t>(bindings_.size()));
    for (const auto& binding : bindings_) {
      const auto& rank_space = binding.solution->rank_space();
      for (int axis = 0; axis < Dim; ++axis)
        contract.scalar(rank_space.origin()[axis]).scalar(rank_space.extent()[axis]);
      for (int axis = 0; axis < Dim; ++axis)
        contract.scalar(binding.geometry->domain().lo[axis])
            .scalar(binding.geometry->domain().hi[axis])
            .scalar(binding.geometry->lower()[axis])
            .scalar(binding.geometry->upper()[axis]);
      contract.scalar(binding.solution->distribution().mode())
          .sequence(binding.solution->layout().boxes(),
                    [](ExactContractBuilder& item, const Box<Dim>& patch) {
                      for (int axis = 0; axis < Dim; ++axis)
                        item.scalar(patch.lo[axis]).scalar(patch.hi[axis]);
                    })
          .sequence(binding.solution->distribution().owners(),
                    [](ExactContractBuilder& item, const Index<Dim>& owner) {
                      for (int axis = 0; axis < Dim; ++axis)
                        item.scalar(owner[axis]);
                    });
      for (int axis = 0; axis < Dim; ++axis)
        for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
          const Face<Dim> face{axis, side};
          const PhysicalBoundaryFace& law = binding.boundary->at(face);
          contract.scalar(binding.boundary->topology().is_periodic(face))
              .scalar(law.kind)
              .scalar(law.value)
              .scalar(law.alpha)
              .scalar(law.beta);
        }
    }
    contract.scalar(static_cast<std::uint64_t>(ratios_.size()));
    for (const auto& ratio : ratios_)
      for (int axis = 0; axis < Dim; ++axis)
        contract.scalar(ratio[axis]);
    return std::move(contract).release();
  }

  static void mark_coverage_(Level& parent, const Level& child,
                             const ::pops::amr::RefinementRatio<Dim>& ratio) {
    const Extent<Dim> ratio_value = detail::ratio_extent<Dim>(ratio);
    for (const Box<Dim>& fine_patch : child.binding.solution->layout().boxes()) {
      const Box<Dim> footprint = coarsen(fine_patch, ratio_value);
      for (std::size_t local = 0; local < parent.binding.solution->local_size(); ++local) {
        const Box<Dim> region = parent.binding.solution->box(local).intersect(footprint);
        if (!region.empty()) {
          for_each_cell(region,
                        detail::SetKernel<Dim, Real>{parent.covered.fab(local).view(), Real(1)});
          for_each_cell(region,
                        detail::SetKernel<Dim, Real>{parent.active.fab(local).view(), Real(0)});
        }
      }
    }
    ::pops::device_fence();
  }

  void same_level_fill_(Level& level, field_type& field) {
    if (level.halo_exchange)
      level.halo_exchange->execute(field, *lane_);
    else
      fill_boundary(field, level.halo_schedule);
  }

  void fill_ghosts_(std::size_t level_index, field_type& field, const field_type* parent_field,
                    const PreparedPhysicalBoundary<Dim>& boundary) {
    Level& level = *levels_[level_index];
    if (level_index > 0) {
      if (parent_field == nullptr)
        throw std::logic_error("dimension-generic tensor FAC fine fill has no parent field");
      Connection& connection = *connections_[level_index - 1];
      connection.gather_parent(*parent_field);
      connection.interpolate_ghosts(field);
    }
    same_level_fill_(level, field);
    fill_physical_boundary(field, boundary);
  }

  void fill_all_coefficient_ghosts_() {
    for (std::size_t level = 0; level < levels_.size(); ++level)
      for (std::size_t coefficient = 0; coefficient < static_cast<std::size_t>(Dim * Dim);
           ++coefficient) {
        const field_type* parent =
            level == 0 ? nullptr : levels_[level - 1]->binding.coefficients[coefficient];
        fill_ghosts_(level, *levels_[level]->binding.coefficients[coefficient], parent,
                     levels_[level]->coefficient_boundary);
      }
  }

  void fill_solution_ghosts_(std::size_t level, field_type& solution, bool homogeneous) {
    const field_type* parent = nullptr;
    if (level > 0)
      parent = &solution == &levels_[level]->correction ? &levels_[level - 1]->correction
                                                        : levels_[level - 1]->binding.solution;
    fill_ghosts_(
        level, solution, parent,
        homogeneous ? levels_[level]->homogeneous_boundary : levels_[level]->physical_boundary);
  }

  void fill_all_solution_ghosts_() {
    for (std::size_t level = 0; level < levels_.size(); ++level)
      fill_solution_ghosts_(level, *levels_[level]->binding.solution, false);
  }

  bool coefficients_are_elliptic_() const {
    long local_invalid = 0;
    for (const auto& level : levels_)
      for (std::size_t local = 0; local < level->binding.solution->local_size(); ++local) {
        using storage_type =
            std::remove_cvref_t<decltype(level->binding.coefficients[0]->fab(local).storage())>;
        std::array<storage_type, static_cast<std::size_t>(Dim * Dim)> coefficients{};
        for (std::size_t slot = 0; slot < coefficients.size(); ++slot)
          coefficients[slot] = level->binding.coefficients[slot]->fab(local).storage();
        const std::size_t count = coefficients[0].extent(0);
        long patch_invalid = 0;
        Kokkos::parallel_reduce(
            "pops_nd_tensor_ellipticity", Kokkos::RangePolicy<>(0, count),
            KOKKOS_LAMBDA(const std::size_t index, long& invalid) {
              constexpr Real infinity = std::numeric_limits<Real>::infinity();
              for (int row = 0; row < Dim; ++row) {
                const Real diagonal =
                    coefficients[static_cast<std::size_t>(row * Dim + row)](index);
                Real off_diagonal = Real(0);
                bool finite = diagonal == diagonal && diagonal != infinity && diagonal != -infinity;
                for (int column = 0; column < Dim; ++column) {
                  const Real a = coefficients[static_cast<std::size_t>(row * Dim + column)](index);
                  finite = finite && a == a && a != infinity && a != -infinity;
                  if (column != row)
                    off_diagonal += a < Real(0) ? -a : a;
                }
                if (!finite || !(diagonal > off_diagonal))
                  invalid = 1;
              }
            },
            Kokkos::Max<long>(patch_invalid));
        local_invalid = std::max(local_invalid, patch_invalid);
      }
    ::pops::device_fence();
    return all_reduce_max(local_invalid, *lane_) == 0;
  }

  detail::TensorStencil<Dim> stencil_(const Level& level, std::size_t local,
                                      const field_type& field) const {
    std::array<FieldView<const Real, Dim>, static_cast<std::size_t>(Dim * Dim)> coefficients{};
    for (std::size_t coefficient = 0; coefficient < static_cast<std::size_t>(Dim * Dim);
         ++coefficient)
      coefficients[coefficient] =
          std::as_const(level.binding.coefficients[coefficient]->fab(local)).view();
    return elliptic::nd::make_cartesian_tensor_operator<
        elliptic::nd::CartesianTensorDivergenceSign::negative_divergence>(
        std::as_const(field.fab(local)).view(),
        elliptic::nd::split_cartesian_tensor_coefficients<Dim>(coefficients),
        *level.binding.geometry);
  }

  void smooth_(std::size_t level_index, field_type& iterate, const field_type& rhs, int sweeps,
               bool mask_covered, bool homogeneous) {
    if (sweeps <= 0)
      return;
    Level& level = *levels_[level_index];
    for (int sweep = 0; sweep < sweeps; ++sweep) {
      constexpr unsigned colours = 1u << Dim;
      for (unsigned colour = 0; colour < colours; ++colour) {
        // Each colour consumes values written by the preceding one.  Refreshing here (rather than
        // once per sweep) also makes those in-place updates visible across local patch boundaries
        // and through the prepared same-level halo exchange.
        fill_solution_ghosts_(level_index, iterate, homogeneous);
        for (std::size_t local = 0; local < iterate.local_size(); ++local)
          for_each_cell(iterate.box(local),
                        detail::ColoredGaussSeidelKernel<Dim>{
                            iterate.fab(local).view(), std::as_const(rhs.fab(local)).view(),
                            std::as_const(level.covered.fab(local)).view(),
                            stencil_(level, local, iterate), colour, Real(1), mask_covered});
        ::pops::device_fence();
      }
    }
  }

  void compute_level_residual_(std::size_t level_index) {
    Level& level = *levels_[level_index];
    field_type& solution = *level.binding.solution;
    fill_solution_ghosts_(level_index, solution, false);
    for (std::size_t local = 0; local < solution.local_size(); ++local)
      for_each_cell(solution.box(local),
                    detail::ResidualKernel<Dim>{level.residual.fab(local).view(),
                                                std::as_const(level.binding.rhs->fab(local)).view(),
                                                std::as_const(level.covered.fab(local)).view(),
                                                stencil_(level, local, solution), true});
    ::pops::device_fence();
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

  void solve_coarse_correction_(const Controls& controls) {
    Level& coarse = *levels_.front();
    coarse.correction.set_val(Real(0));
    const Real reference = global_norm_inf_(coarse.residual);
    const Real stop = std::max(controls.coarse_absolute_tolerance,
                               controls.coarse_relative_tolerance * reference);
    for (int sweep = 0; sweep < controls.coarse_cycles; ++sweep) {
      smooth_(0, coarse.correction, coarse.residual, 1, false, true);
      if (nullspace_workspace_ && ((sweep + 1) % 8 == 0 || sweep + 1 == controls.coarse_cycles))
        subtract_active_mean_(coarse, coarse.correction);
      if ((sweep + 1) % 8 == 0 || sweep + 1 == controls.coarse_cycles) {
        fill_solution_ghosts_(0, coarse.correction, true);
        for (std::size_t local = 0; local < coarse.correction.local_size(); ++local)
          for_each_cell(
              coarse.correction.box(local),
              detail::ResidualKernel<Dim>{coarse.scratch.fab(local).view(),
                                          std::as_const(coarse.residual.fab(local)).view(),
                                          std::as_const(coarse.covered.fab(local)).view(),
                                          stencil_(coarse, local, coarse.correction), false});
        ::pops::device_fence();
        if (global_norm_inf_(coarse.scratch) <= stop)
          break;
      }
    }
  }

  static void add_active_(Level& level, const field_type& correction) {
    for (std::size_t local = 0; local < level.binding.solution->local_size(); ++local)
      for_each_cell(level.binding.solution->box(local),
                    detail::ActiveAddKernel<Dim>{level.binding.solution->fab(local).view(),
                                                 std::as_const(correction.fab(local)).view(),
                                                 std::as_const(level.active.fab(local)).view()});
    ::pops::device_fence();
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
      connections_[child - 1]->restrict_into(*levels_[child]->binding.solution,
                                             *levels_[child - 1]->binding.solution);
  }

  Real global_norm_inf_(const field_type& field) const {
    return static_cast<Real>(all_reduce_max(static_cast<double>(norm_inf(field)), *lane_));
  }

  bool singular_() const noexcept {
    const PhysicalBoundaryConditions<Dim>& boundary = *levels_.front()->binding.boundary;
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

  void subtract_active_mean_(const Level& level, field_type& field) const {
    Real measure = Real(1);
    for (int axis = 0; axis < Dim; ++axis)
      measure *= level.binding.geometry->spacing(axis);
    double local_sum = 0;
    double local_volume = 0;
    for (std::size_t local = 0; local < field.local_size(); ++local) {
      const auto values = std::as_const(field).fab(local).view();
      const auto active = std::as_const(level.active).fab(local).view();
      local_sum += static_cast<double>(for_each_cell_reduce_sum(
          field.box(local), detail::ActiveMomentKernel<Dim>{values, active, measure}));
      local_volume += static_cast<double>(for_each_cell_reduce_sum(
          field.box(local), detail::ActiveVolumeKernel<Dim>{active, measure}));
    }
    const double volume = all_reduce_sum(local_volume, *lane_);
    if (!(volume > 0))
      return;
    const Real shift = static_cast<Real>(all_reduce_sum(local_sum, *lane_) / volume);
    for (std::size_t local = 0; local < field.local_size(); ++local)
      for_each_cell(field.box(local), detail::ActiveShiftKernel<Dim>{
                                          field.fab(local).view(),
                                          std::as_const(level.active).fab(local).view(), shift});
    ::pops::device_fence();
  }

  void prepare_nullspace_() {
    if (nullspace_workspace_ || !singular_())
      return;
    std::optional<FieldNullspacePlan<Dim>> plan;
    std::vector<PreparedVectorDistribution<Dim>> distributions;
    std::vector<const MultiFab<Dim>*> rhs_layouts;
    std::vector<const MultiFab<Dim>*> retained_rhs;
    std::vector<MultiFab<Dim>*> candidates;
    std::exception_ptr local_error;
    try {
      plan.emplace(constant_mean_zero_nullspace<Dim>(
          "pops.runtime.amr.tensor-composite-fac.nullspace", "tensor-fac-composite", Real(1)));
      plan->bases[0].masks.clear();
      plan->bases[0].cell_measure.clear();
      plan->bases[0].masks.reserve(levels_.size());
      plan->bases[0].cell_measure.reserve(levels_.size());
      distributions.reserve(levels_.size());
      rhs_layouts.reserve(levels_.size());
      retained_rhs.reserve(levels_.size());
      candidates.reserve(levels_.size());
      for (const auto& level : levels_) {
        auto mask =
            std::make_shared<MultiFab<Dim>>(level->active.layout(), level->active.distribution(),
                                            level->active.local_rank(), 1, Extent<Dim>{});
        ::pops::elliptic::mg::copy_scalar_valid(level->active, *mask);
        plan->bases[0].masks.emplace_back(std::move(mask));
        Real measure = Real(1);
        for (int axis = 0; axis < Dim; ++axis)
          measure *= level->binding.geometry->spacing(axis);
        plan->bases[0].cell_measure.push_back(measure);
        distributions.push_back(level->binding.solution->distribution().replicated()
                                    ? PreparedVectorDistribution<Dim>::replicated()
                                    : PreparedVectorDistribution<Dim>::distributed());
        rhs_layouts.push_back(level->binding.rhs);
        retained_rhs.push_back(level->binding.rhs);
        candidates.push_back(level->binding.solution);
      }
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, *lane_) != 0) {
      if (lane_->size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error(
          "dimension-generic tensor FAC nullspace materialization failed collectively");
    }
    nullspace_workspace_ = FieldNullspaceWorkspace<Dim>::prepare_collectively(
        std::move(*plan), std::move(rhs_layouts), std::move(distributions), *lane_);
    nullspace_rhs_ = std::move(retained_rhs);
    nullspace_candidates_ = std::move(candidates);
  }

  Real composite_residual_norm_() const {
    Real local = Real(0);
    for (const auto& level : levels_)
      local = std::max(local, norm_inf(level->residual));
    return static_cast<Real>(all_reduce_max(static_cast<double>(local), *lane_));
  }

  std::vector<LevelBinding<Dim, MemorySpace>> bindings_;
  std::vector<::pops::amr::RefinementRatio<Dim>> ratios_;
  // The immutable borrow pins the external lane until every prepared communication object has
  // been destroyed. Member destruction is reverse declaration order, so the borrow follows them.
  const ExecutionLane* lane_ = nullptr;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  std::vector<std::unique_ptr<Level>> levels_;
  std::vector<std::unique_ptr<Connection>> connections_;
  std::string exact_contract_{};
  std::vector<const MultiFab<Dim>*> nullspace_rhs_{};
  std::vector<MultiFab<Dim>*> nullspace_candidates_{};
  std::unique_ptr<FieldNullspaceWorkspace<Dim>> nullspace_workspace_{};
};

template <int Dim, class MemorySpace>
PreparedResidentStorage FullTensorCompositeFac<Dim, MemorySpace>::resident_storage() const {
  // The singular nullspace workspace is materialized during preparation and contributes its exact
  // receipt below; it must never be omitted from a hard resident-storage decision.

  const auto add = [](std::uint64_t& total, std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - total)
      throw std::overflow_error("tensor FAC resident storage size overflows uint64");
    total += value;
  };
  const auto vector_bytes = [](const auto& values) -> std::uint64_t {
    using value_type = typename std::remove_reference_t<decltype(values)>::value_type;
    if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
      throw std::overflow_error("tensor FAC resident vector storage overflows uint64");
    return static_cast<std::uint64_t>(values.capacity()) * sizeof(value_type);
  };
  const auto external_string_bytes = [](const std::string& value) -> std::uint64_t {
    const auto object_begin = reinterpret_cast<std::uintptr_t>(&value);
    const auto object_end = object_begin + sizeof(value);
    const auto data = reinterpret_cast<std::uintptr_t>(value.data());
    if (data >= object_begin && data < object_end)
      return 0;
    if (value.capacity() == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("tensor FAC resident string storage overflows uint64");
    return static_cast<std::uint64_t>(value.capacity()) + 1U;
  };
  const auto fab_bytes = [&add](const Fab<Dim, MemorySpace>& fab) -> std::uint64_t {
    const std::uint64_t values = static_cast<std::uint64_t>(fab.size());
    if (values > std::numeric_limits<std::uint64_t>::max() / sizeof(Real))
      throw std::overflow_error("tensor FAC resident Fab storage overflows uint64");
    std::uint64_t result = 0;
    add(result, values * sizeof(Real));
    return result;
  };

  std::uint64_t total = 0;
  add(total, vector_bytes(bindings_));
  add(total, vector_bytes(ratios_));
  add(total, vector_bytes(levels_));
  for (const auto& level : levels_) {
    add(total, sizeof(Level));
    for (const field_type* field :
         {&level->residual, &level->scratch, &level->correction, &level->covered, &level->active})
      add(total, field->resident_storage_bytes());
    add(total, level->halo_schedule.resident_storage_bytes());
    add(total, level->physical_boundary.resident_storage_bytes());
    add(total, level->homogeneous_boundary.resident_storage_bytes());
    add(total, level->coefficient_boundary.resident_storage_bytes());
    if (level->halo_exchange)
      add(total, sizeof(HaloExchange<Dim, MemorySpace>));
    if (level->halo_exchange)
      add(total, level->halo_exchange->resident_storage_bytes());
  }
  add(total, vector_bytes(connections_));
  for (const auto& connection : connections_) {
    add(total, sizeof(Connection));
    add(total, vector_bytes(connection->scratch));
    add(total, vector_bytes(connection->scratch_by_fine_patch));
    add(total, external_string_bytes(connection->gather_contract));
    add(total, external_string_bytes(connection->restriction_contract));
    for (const auto& patch : connection->scratch) {
      add(total, fab_bytes(patch.parent_staging));
      add(total, fab_bytes(patch.restricted));
      add(total, vector_bytes(patch.ghost_regions));
      add(total, vector_bytes(patch.broadcast_buffer));
      if (patch.restricted_host) {
        const std::uint64_t values = static_cast<std::uint64_t>(patch.restricted_host->size());
        if (values > std::numeric_limits<std::uint64_t>::max() / sizeof(Real))
          throw std::overflow_error("tensor FAC resident host mirror storage overflows uint64");
        add(total, values * sizeof(Real));
      }
    }
    if (connection->gather) {
      add(total, sizeof(typename Connection::transport_type));
      add(total, connection->gather->resident_storage_bytes());
    }
    if (connection->restriction) {
      add(total, sizeof(typename Connection::transport_type));
      add(total, connection->restriction->resident_storage_bytes());
    }
  }
  add(total, external_string_bytes(exact_contract_));
  add(total, vector_bytes(nullspace_rhs_));
  add(total, vector_bytes(nullspace_candidates_));
  if (nullspace_workspace_) {
    const PreparedResidentStorage workspace_storage = nullspace_workspace_->resident_storage();
    if (!workspace_storage.is_exact())
      return PreparedResidentStorage::unknown();
    add(total, sizeof(*nullspace_workspace_));
    add(total, workspace_storage.bytes);
  }
  return PreparedResidentStorage::exact(total);
}

template <int Dim, class MemorySpace>
std::uint64_t FullTensorCompositeFac<Dim, MemorySpace>::configured_resident_storage_upper_bound(
    std::span<const std::uint64_t> level_cell_bounds, std::span<const std::uint64_t> patch_bounds,
    std::span<const std::uint64_t> parent_child_pair_bounds, std::uint64_t rank_bound,
    std::string_view execution_lane_identity) {
  const std::size_t levels = level_cell_bounds.size();
  if (levels < 2 || patch_bounds.size() != levels ||
      parent_child_pair_bounds.size() + 1U != levels || rank_bound == 0 ||
      execution_lane_identity.empty())
    throw std::invalid_argument("tensor FAC configured storage bounds are incomplete");
  if (std::any_of(level_cell_bounds.begin(), level_cell_bounds.end(),
                  [](std::uint64_t value) { return value == 0; }) ||
      std::any_of(patch_bounds.begin(), patch_bounds.end(),
                  [](std::uint64_t value) { return value == 0; }) ||
      std::any_of(parent_child_pair_bounds.begin(), parent_child_pair_bounds.end(),
                  [](std::uint64_t value) { return value == 0; }))
    throw std::invalid_argument("tensor FAC configured storage bounds contain zero capacity");

  const auto add = [](std::uint64_t& total, std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - total)
      throw std::overflow_error("tensor FAC configured storage size overflows uint64");
    total += value;
  };
  const auto product = [](std::uint64_t left, std::uint64_t right) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
      throw std::overflow_error("tensor FAC configured storage product overflows uint64");
    return left * right;
  };
  const auto power = [&product](std::uint64_t base, int exponent) {
    std::uint64_t result = 1;
    for (int index = 0; index < exponent; ++index)
      result = product(result, base);
    return result;
  };
  const auto vector_bytes = [&product](std::uint64_t count, std::size_t element_size) {
    return product(count, static_cast<std::uint64_t>(element_size));
  };
  const auto vector_capacity_bytes = [&product, &vector_bytes](std::uint64_t count,
                                                               std::size_t element_size) {
    return vector_bytes(product(2U, count), element_size);
  };
  const auto external_string_bytes = [&add](std::uint64_t characters) {
    std::uint64_t result = characters;
    add(result, 1U);
    return result;
  };
  const auto decimal_digits = [](std::uint64_t value) {
    std::uint64_t digits = 1;
    while (value >= 10) {
      value /= 10;
      ++digits;
    }
    return digits;
  };
  const std::uint64_t scalar_frame_bytes = 20;
  const std::uint64_t frame_bytes = 9;
  const std::uint64_t sequence_count_bytes = 8;
  const std::uint64_t ghost_growth = power(3, Dim);
  const std::uint64_t staging_growth = power(5, Dim);
  const std::uint64_t boundary_entries = ghost_growth - 1U;

  const auto multifab_bytes = [&](std::uint64_t cells, std::uint64_t patches, int components,
                                  bool ghosts) {
    if (components < 1)
      throw std::invalid_argument("tensor FAC configured field has invalid components");
    std::uint64_t result = 0;
    // A configured request owns only a cell bound and a patch bound.  Treat each admitted patch
    // as a full-domain image so this remains a bound before the concrete disjoint layout is
    // materialized (and therefore dominates the resident field receipt).
    const std::uint64_t patch_cells = product(cells, patches);
    const std::uint64_t values =
        product(product(patch_cells, ghosts ? ghost_growth : std::uint64_t{1}),
                static_cast<std::uint64_t>(components));
    add(result, product(values, static_cast<std::uint64_t>(sizeof(Real))));
    // MultiFab::resident_storage_bytes() charges two BoxArray images, the owner image, two index
    // maps, and the local Fab vector.  A configured rank may own every patch, which is the
    // conservative local shape.
    add(result, vector_bytes(patches, sizeof(Box<Dim>)));
    add(result, vector_bytes(patches, sizeof(Box<Dim>)));
    add(result, vector_bytes(patches, sizeof(Index<Dim>)));
    add(result, vector_bytes(patches, sizeof(std::size_t)));
    add(result, vector_bytes(patches, sizeof(std::size_t)));
    add(result, vector_bytes(patches, sizeof(Fab<Dim, MemorySpace>)));
    return result;
  };

  std::uint64_t total = 0;
  add(total,
      vector_bytes(static_cast<std::uint64_t>(levels), sizeof(LevelBinding<Dim, MemorySpace>)));
  add(total, vector_bytes(static_cast<std::uint64_t>(levels - 1U),
                          sizeof(::pops::amr::RefinementRatio<Dim>)));
  add(total, vector_bytes(static_cast<std::uint64_t>(levels), sizeof(std::unique_ptr<Level>)));
  for (std::size_t level = 0; level < levels; ++level) {
    const std::uint64_t cells = level_cell_bounds[level];
    const std::uint64_t patches = patch_bounds[level];
    add(total, sizeof(Level));
    // residual, scratch, covered, and active have no ghosts; correction carries one ghost layer.
    for (int field = 0; field < 4; ++field)
      add(total, multifab_bytes(cells, patches, 1, false));
    add(total, multifab_bytes(cells, patches, 1, true));

    // The halo/boundary components retain their copied layout, classified job/peer vectors and
    // fixed region schedules.  Charge growth capacity rather than only final work cardinality.
    const std::uint64_t halo_jobs = product(product(product(4U, patches), patches), ghost_growth);
    add(total, vector_capacity_bytes(product(2U, patches), sizeof(Box<Dim>)));
    add(total, vector_capacity_bytes(patches, sizeof(Index<Dim>)));
    add(total, vector_capacity_bytes(product(3U, halo_jobs), sizeof(HaloJob<Dim>)));
    add(total, vector_capacity_bytes(product(2U, rank_bound), sizeof(HaloPeerPlan<Dim>)));
    add(total,
        vector_capacity_bytes(product(3U, boundary_entries), sizeof(BoundaryRegionPlan<Dim>)));

    // Every configured level may be remote.  The exchange owns a local arena, four peer mirrors,
    // peer/request vectors, active-field pointers and its persistent fence label.
    const std::uint64_t halo_elements = product(halo_jobs, cells);
    add(total, sizeof(HaloExchange<Dim, MemorySpace>));
    add(total, product(5U, product(halo_elements, static_cast<std::uint64_t>(sizeof(Real)))));
    add(total, HaloExchange<Dim, MemorySpace>::configured_metadata_storage_upper_bound(rank_bound,
                                                                                       patches));
    add(total, external_string_bytes(static_cast<std::uint64_t>(
                   std::string_view{"pops.halo-exchange.fence"}.size())));
  }

  // The detached configured envelope does not yet contain physical boundary kinds. Reserve the
  // complete constant-mode nullspace image unconditionally so a later periodic materialization
  // cannot exceed the sealed ceiling. This mirrors prepare_nullspace_() with the same STL growth
  // operations and charges the provider-erased distributions, masks, and dense reduction arenas.
  FieldNullspacePlan<Dim> nullspace_plan = constant_mean_zero_nullspace<Dim>(
      "pops.runtime.amr.tensor-composite-fac.nullspace", "tensor-fac-composite", Real(1));
  nullspace_plan.bases[0].masks.clear();
  nullspace_plan.bases[0].cell_measure.clear();
  nullspace_plan.bases[0].masks.reserve(levels);
  nullspace_plan.bases[0].cell_measure.reserve(levels);
  for (std::size_t level = 0; level < levels; ++level) {
    nullspace_plan.bases[0].masks.emplace_back();
    nullspace_plan.bases[0].cell_measure.push_back(Real(1));
  }
  const auto string_capacity_bytes = [&add](const std::string& value) {
    std::uint64_t result = static_cast<std::uint64_t>(value.capacity());
    add(result, 1U);
    return result;
  };
  add(total, sizeof(FieldNullspaceWorkspace<Dim>));
  add(total, string_capacity_bytes(nullspace_plan.identity));
  add(total, string_capacity_bytes(nullspace_plan.layout_identity));
  add(total, vector_bytes(static_cast<std::uint64_t>(nullspace_plan.bases.capacity()),
                          sizeof(FieldNullspaceBasis<Dim>)));
  add(total, vector_bytes(static_cast<std::uint64_t>(nullspace_plan.gauges.capacity()),
                          sizeof(FieldGaugeConstraint)));
  for (const FieldNullspaceBasis<Dim>& basis : nullspace_plan.bases) {
    add(total, string_capacity_bytes(basis.identity));
    add(total, string_capacity_bytes(basis.provenance));
    add(total, string_capacity_bytes(basis.recipe_identity));
    add(total, vector_bytes(static_cast<std::uint64_t>(basis.masks.capacity()),
                            sizeof(std::shared_ptr<const MultiFab<Dim>>)));
    add(total, vector_bytes(static_cast<std::uint64_t>(basis.coverage.capacity()),
                            sizeof(std::shared_ptr<const MultiFab<Dim>>)));
    add(total,
        vector_bytes(static_cast<std::uint64_t>(basis.cell_measure.capacity()), sizeof(Real)));
  }
  for (const FieldGaugeConstraint& gauge : nullspace_plan.gauges)
    add(total, string_capacity_bytes(gauge.basis_identity));
  for (std::size_t level = 0; level < levels; ++level) {
    add(total, sizeof(MultiFab<Dim>));
    add(total, multifab_bytes(level_cell_bounds[level], patch_bounds[level], 1, false));
  }

  std::vector<const MultiFab<Dim>*> nullspace_layouts;
  nullspace_layouts.reserve(levels);
  nullspace_layouts.resize(levels, nullptr);
  const std::uint64_t nullspace_pointer_capacity =
      static_cast<std::uint64_t>(nullspace_layouts.capacity());
  // Workspace layouts plus the FAC-owned RHS and candidate pointer vectors.
  add(total, vector_bytes(nullspace_pointer_capacity, sizeof(const MultiFab<Dim>*)));
  add(total, vector_bytes(nullspace_pointer_capacity, sizeof(const MultiFab<Dim>*)));
  add(total, vector_bytes(nullspace_pointer_capacity, sizeof(MultiFab<Dim>*)));

  const PreparedVectorDistribution<Dim> distributed =
      PreparedVectorDistribution<Dim>::distributed();
  const PreparedVectorDistribution<Dim> replicated = PreparedVectorDistribution<Dim>::replicated();
  const PreparedResidentStorage distributed_storage = distributed.resident_storage();
  const PreparedResidentStorage replicated_storage = replicated.resident_storage();
  if (!distributed_storage.is_exact() || !replicated_storage.is_exact())
    throw std::logic_error("native nullspace distribution has no configured storage receipt");
  std::vector<PreparedVectorDistribution<Dim>> distributions;
  distributions.reserve(levels);
  for (std::size_t level = 0; level < levels; ++level)
    distributions.push_back(distributed);
  add(total, vector_bytes(static_cast<std::uint64_t>(distributions.capacity()),
                          sizeof(PreparedVectorDistribution<Dim>)));
  add(total, product(static_cast<std::uint64_t>(levels),
                     std::max(distributed_storage.bytes, replicated_storage.bytes)));

  constexpr std::uint64_t basis_count = 1;
  constexpr std::uint64_t value_capacity = 2;
  add(total,
      vector_bytes(product(static_cast<std::uint64_t>(levels), value_capacity), sizeof(double)));
  add(total, vector_bytes(value_capacity, sizeof(double)));
  add(total, vector_bytes(basis_count, sizeof(double)));
  add(total, vector_bytes(basis_count, sizeof(double)));
  add(total, static_cast<std::uint64_t>(replicated.validation_scratch_byte_count()));
  add(total, vector_bytes(static_cast<std::uint64_t>(replicated.reduction_scratch_value_count(
                              static_cast<std::size_t>(value_capacity))),
                          sizeof(double)));

  add(total,
      vector_bytes(static_cast<std::uint64_t>(levels - 1U), sizeof(std::unique_ptr<Connection>)));
  std::uint64_t fac_contract = 0;
  add(fac_contract,
      frame_bytes + static_cast<std::uint64_t>(
                        std::string_view{"pops.runtime.amr.nd-tensor-composite-fac"}.size()));
  add(fac_contract, product(3, scalar_frame_bytes));
  add(fac_contract, frame_bytes + static_cast<std::uint64_t>(execution_lane_identity.size()));
  add(fac_contract, scalar_frame_bytes);
  for (std::size_t level = 0; level < levels; ++level) {
    const std::uint64_t patches = patch_bounds[level];
    add(fac_contract, product(static_cast<std::uint64_t>(4 * Dim + 1), scalar_frame_bytes));
    // sequence(Box): outer frame + count plus one nested frame containing 2*Dim scalar frames.
    add(fac_contract, frame_bytes + sequence_count_bytes);
    add(fac_contract, product(patches, frame_bytes + product(static_cast<std::uint64_t>(2 * Dim),
                                                             scalar_frame_bytes)));
    // sequence(Index): outer frame + count plus one nested frame containing Dim scalar frames.
    add(fac_contract, frame_bytes + sequence_count_bytes);
    add(fac_contract, product(patches, frame_bytes + product(static_cast<std::uint64_t>(Dim),
                                                             scalar_frame_bytes)));
  }
  add(fac_contract, scalar_frame_bytes);
  add(fac_contract,
      product(static_cast<std::uint64_t>((levels - 1U) * static_cast<std::size_t>(Dim)),
              scalar_frame_bytes));
  add(total, external_string_bytes(fac_contract));

  constexpr std::string_view gather_prefix = "pops.nd-tensor-fac.replicated-parent-gather/";
  constexpr std::string_view restriction_prefix =
      "pops.nd-tensor-fac.replicated-parent-restriction/";
  for (std::size_t parent = 0; parent + 1U < levels; ++parent) {
    const std::uint64_t parent_cells = level_cell_bounds[parent];
    const std::uint64_t child_cells = level_cell_bounds[parent + 1U];
    const std::uint64_t child_patches = patch_bounds[parent + 1U];
    const std::uint64_t pairs = parent_child_pair_bounds[parent];
    add(total, sizeof(Connection));
    add(total, vector_bytes(child_patches, sizeof(typename Connection::ScratchPatch)));
    add(total, vector_bytes(child_patches, sizeof(std::size_t)));
    const std::uint64_t ordinal_digits = decimal_digits(static_cast<std::uint64_t>(parent));
    add(total,
        external_string_bytes(static_cast<std::uint64_t>(gather_prefix.size()) + ordinal_digits));
    add(total, external_string_bytes(static_cast<std::uint64_t>(restriction_prefix.size()) +
                                     ordinal_digits));

    // A partitioned parent owns two serialized RegionTransferPlan contracts.  Each records the
    // exact rank-space and every overlap row; the short replicated labels above are not a bound.
    constexpr std::string_view transfer_contract_prefix =
        "pops.elliptic.amr.partitioned-region-transfer";
    const std::uint64_t transfer_identity_bytes =
        static_cast<std::uint64_t>(
            std::max(std::string_view{"nd-tensor-parent-gather/"}.size(),
                     std::string_view{"nd-tensor-fine-restriction/"}.size())) +
        ordinal_digits;
    const std::uint64_t transfer_row_bytes = 24U + product(48U, static_cast<std::uint64_t>(Dim));
    std::uint64_t transfer_contract_bytes = 0;
    add(transfer_contract_bytes, 8U + static_cast<std::uint64_t>(transfer_contract_prefix.size()));
    add(transfer_contract_bytes, 8U * 4U);
    add(transfer_contract_bytes, 8U + transfer_identity_bytes);
    add(transfer_contract_bytes, product(16U, static_cast<std::uint64_t>(Dim)));
    add(transfer_contract_bytes, 16U);
    add(transfer_contract_bytes, product(pairs, transfer_row_bytes));
    add(total, external_string_bytes(transfer_contract_bytes));
    add(total, external_string_bytes(transfer_contract_bytes));

    // Every fine patch can induce one complete parent-domain staging/restriction image.  The
    // coarsened staging box is clipped to that parent domain, so ``parent_cells`` is the exact
    // per-patch ceiling; 5^Dim covers the surrounding-growth form, and the remaining three
    // images cover restriction plus replicated host/broadcast mirrors.  Charge all P admitted
    // fine patches because configured capacity precedes the disjoint-layout proof.
    const std::uint64_t scratch_cells = product(parent_cells, child_patches);
    add(total, product(product(staging_growth + 3U, scratch_cells),
                       static_cast<std::uint64_t>(sizeof(Real))));
    // A disjoint collection of P boxes induces at most (2P+1)^Dim grid cells in the complement
    // of one grown fine patch.  This bounds every ScratchPatch::ghost_regions vector without
    // relying on the order in which ``subtract_from`` splits it.
    std::uint64_t grid_extent = product(2U, child_patches);
    add(grid_extent, 1U);
    const std::uint64_t ghost_regions_per_patch = power(grid_extent, Dim);
    add(total, vector_bytes(product(child_patches, ghost_regions_per_patch), sizeof(Box<Dim>)));

    // The two RegionTransport instances retain canonical/local/peer job records and device/host
    // staging buffers.  Pairs and fine cells are explicit configured authorities, so no live
    // topology or implicit rank count is consulted here.  Four payload mirrors plus one local
    // arena cover a transport; there are gather and restriction transports on each edge.
    using transfer_job = typename Connection::transfer_job;
    using transfer_plan = typename Connection::transfer_plan;
    using transfer_peer_plan = typename transfer_plan::peer_plan_type;
    using transport_type = typename Connection::transport_type;
    add(total, vector_capacity_bytes(product(4U, pairs), sizeof(transfer_job)));
    add(total, vector_capacity_bytes(product(2U, rank_bound), sizeof(transfer_peer_plan)));
    add(total, product(product(product(10U, pairs), child_cells),
                       static_cast<std::uint64_t>(sizeof(Real))));
    add(total, product(2U, static_cast<std::uint64_t>(sizeof(transport_type))));
    add(total,
        product(2U, transport_type::configured_peer_metadata_storage_upper_bound(rank_bound)));
  }
  return total;
}

}  // namespace pops::runtime::program::tensor_fac
