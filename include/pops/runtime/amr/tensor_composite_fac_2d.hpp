/// @file
/// @brief Prepared full-tensor composite FAC kernel for the exact two-dimensional hierarchy route.

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
#include <pops/numerics/elliptic/linear/solve_report.hpp>
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
#include <utility>
#include <vector>

namespace pops::runtime::program::tensor_fac_2d {

template <class MemorySpace>
struct LevelBinding {
  const Geometry<2>* geometry = nullptr;
  const PhysicalBoundaryConditions<2>* boundary = nullptr;
  std::array<MultiFab<2, MemorySpace>*, 4> coefficients{};
  MultiFab<2, MemorySpace>* rhs = nullptr;
  MultiFab<2, MemorySpace>* initial_guess = nullptr;
  MultiFab<2, MemorySpace>* solution = nullptr;
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

inline Extent<2> one_ghost() {
  return Extent<2>{1, 1};
}

inline Extent<2> ratio_extent(const ::pops::amr::RefinementRatio<2>& ratio) {
  return Extent<2>{ratio[0], ratio[1]};
}

inline mesh::BoxArrayValidationBudget exact_layout_budget(const mesh::BoxArray<2>& layout) {
  const std::size_t boxes = layout.size();
  const std::size_t pairs =
      boxes < 2 ? 0 : checked_product(boxes, boxes - 1, "tensor FAC layout budget overflow") / 2;
  return {boxes, pairs};
}

inline HaloScheduleBudget exact_halo_budget(const mesh::BoxArray<2>& layout, const Box<2>& domain) {
  const std::size_t boxes = layout.size();
  const std::size_t pairs = checked_product(boxes, boxes, "tensor FAC halo pair overflow");
  constexpr std::size_t images = 9;
  const std::size_t work = checked_product(pairs, images, "tensor FAC halo work overflow");
  const std::size_t jobs = checked_product(work, 4, "tensor FAC halo job overflow");
  const std::int64_t signed_cells = domain.numPts();
  if (signed_cells <= 0)
    throw std::invalid_argument("tensor FAC halo domain is empty");
  const std::size_t elements = checked_product(jobs, static_cast<std::size_t>(signed_cells),
                                               "tensor FAC halo element overflow");
  return {exact_layout_budget(layout),
          work,
          jobs,
          images,
          checked_product(boxes, 2, "tensor FAC peer budget overflow"),
          elements,
          elements,
          elements};
}

inline BoundaryScheduleBudget exact_boundary_budget() {
  return {8};
}

inline PhysicalBoundaryConditions<2> boundary_with_values(
    const PhysicalBoundaryConditions<2>& source, const Geometry<2>& geometry, bool homogeneous,
    bool coefficient) {
  std::array<PhysicalBoundaryFace, 4> faces{};
  RealVector<2> spacing{};
  for (int axis = 0; axis < 2; ++axis) {
    spacing[axis] = geometry.spacing(axis);
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<2> face{axis, side};
      PhysicalBoundaryFace law = source.at(face);
      if (coefficient && !source.topology().is_periodic(face)) {
        law = PhysicalBoundaryFace{PhysicalBoundaryKind::constant_extrapolation};
      } else if (homogeneous) {
        law.value = Real(0);
      }
      faces[static_cast<std::size_t>(face.ordinal())] = law;
    }
  }
  return PhysicalBoundaryConditions<2>{source.topology(), faces, spacing};
}

inline std::vector<Box<2>> subtract_box(const Box<2>& subject, const Box<2>& cut) {
  const Box<2> overlap = subject.intersect(cut);
  if (overlap.empty())
    return subject.empty() ? std::vector<Box<2>>{} : std::vector<Box<2>>{subject};
  if (overlap == subject)
    return {};
  std::vector<Box<2>> result;
  result.reserve(4);
  Box<2> remainder = subject;
  for (int axis = 0; axis < 2; ++axis) {
    if (remainder.lo[axis] < overlap.lo[axis]) {
      Box<2> lower = remainder;
      lower.hi[axis] = overlap.lo[axis] - 1;
      result.push_back(lower);
      remainder.lo[axis] = overlap.lo[axis];
    }
    if (overlap.hi[axis] < remainder.hi[axis]) {
      Box<2> upper = remainder;
      upper.lo[axis] = overlap.hi[axis] + 1;
      result.push_back(upper);
      remainder.hi[axis] = overlap.hi[axis];
    }
  }
  return result;
}

inline void subtract_from(std::vector<Box<2>>& regions, const Box<2>& cut) {
  std::vector<Box<2>> next;
  for (const Box<2>& region : regions) {
    auto pieces = subtract_box(region, cut);
    next.insert(next.end(), pieces.begin(), pieces.end());
  }
  regions = std::move(next);
}

inline Box<2> clipped_growth(const Box<2>& valid, const Box<2>& domain) {
  return valid.grow(1).intersect(domain);
}

template <class Value>
struct SetKernel {
  FieldView<Value, 2> field{};
  Real value = Real(0);
  POPS_HD void operator()(const Index<2>& index) const { field(index, 0) = value; }
};

struct CopyKernel {
  FieldView<Real, 2> destination{};
  FieldView<const Real, 2> source{};
  POPS_HD void operator()(const Index<2>& index) const { destination(index, 0) = source(index, 0); }
};

struct ActiveAddKernel {
  FieldView<Real, 2> destination{};
  FieldView<const Real, 2> correction{};
  FieldView<const Real, 2> active{};
  POPS_HD void operator()(const Index<2>& index) const {
    if (active(index, 0) >= Real(0.5))
      destination(index, 0) += correction(index, 0);
  }
};

struct MaskKernel {
  FieldView<Real, 2> values{};
  FieldView<const Real, 2> covered{};
  POPS_HD void operator()(const Index<2>& index) const {
    if (covered(index, 0) >= Real(0.5))
      values(index, 0) = Real(0);
  }
};

POPS_HD inline Real harmonic(Real left, Real right) {
  const Real denominator = left + right;
  return denominator != Real(0) ? Real(2) * left * right / denominator : Real(0);
}

struct TensorStencil {
  FieldView<const Real, 2> phi{};
  std::array<FieldView<const Real, 2>, 4> coefficient{};
  Real inverse_dx = Real(0);
  Real inverse_dy = Real(0);

  POPS_HD Real image(const Index<2>& cell) const {
    const Index<2> xm{cell[0] - 1, cell[1]};
    const Index<2> xp{cell[0] + 1, cell[1]};
    const Index<2> ym{cell[0], cell[1] - 1};
    const Index<2> yp{cell[0], cell[1] + 1};
    const Index<2> xmym{cell[0] - 1, cell[1] - 1};
    const Index<2> xmyp{cell[0] - 1, cell[1] + 1};
    const Index<2> xpym{cell[0] + 1, cell[1] - 1};
    const Index<2> xpyp{cell[0] + 1, cell[1] + 1};

    const Real a_xx_p = harmonic(coefficient[0](cell, 0), coefficient[0](xp, 0));
    const Real a_xx_m = harmonic(coefficient[0](xm, 0), coefficient[0](cell, 0));
    const Real a_xy_p = Real(0.5) * (coefficient[1](cell, 0) + coefficient[1](xp, 0));
    const Real a_xy_m = Real(0.5) * (coefficient[1](xm, 0) + coefficient[1](cell, 0));
    const Real a_yx_p = Real(0.5) * (coefficient[2](cell, 0) + coefficient[2](yp, 0));
    const Real a_yx_m = Real(0.5) * (coefficient[2](ym, 0) + coefficient[2](cell, 0));
    const Real a_yy_p = harmonic(coefficient[3](cell, 0), coefficient[3](yp, 0));
    const Real a_yy_m = harmonic(coefficient[3](ym, 0), coefficient[3](cell, 0));

    const Real dx_phi_p = (phi(xp, 0) - phi(cell, 0)) * inverse_dx;
    const Real dx_phi_m = (phi(cell, 0) - phi(xm, 0)) * inverse_dx;
    const Real dy_at_xp =
        (phi(yp, 0) - phi(ym, 0) + phi(xpyp, 0) - phi(xpym, 0)) * (Real(0.25) * inverse_dy);
    const Real dy_at_xm =
        (phi(xmyp, 0) - phi(xmym, 0) + phi(yp, 0) - phi(ym, 0)) * (Real(0.25) * inverse_dy);
    const Real dy_phi_p = (phi(yp, 0) - phi(cell, 0)) * inverse_dy;
    const Real dy_phi_m = (phi(cell, 0) - phi(ym, 0)) * inverse_dy;
    const Real dx_at_yp =
        (phi(xp, 0) - phi(xm, 0) + phi(xpyp, 0) - phi(xmyp, 0)) * (Real(0.25) * inverse_dx);
    const Real dx_at_ym =
        (phi(xpym, 0) - phi(xmym, 0) + phi(xp, 0) - phi(xm, 0)) * (Real(0.25) * inverse_dx);

    const Real flux_x_p = a_xx_p * dx_phi_p + a_xy_p * dy_at_xp;
    const Real flux_x_m = a_xx_m * dx_phi_m + a_xy_m * dy_at_xm;
    const Real flux_y_p = a_yx_p * dx_at_yp + a_yy_p * dy_phi_p;
    const Real flux_y_m = a_yx_m * dx_at_ym + a_yy_m * dy_phi_m;
    return -(flux_x_p - flux_x_m) * inverse_dx - (flux_y_p - flux_y_m) * inverse_dy;
  }

  POPS_HD Real diagonal(const Index<2>& cell) const {
    const Index<2> xm{cell[0] - 1, cell[1]};
    const Index<2> xp{cell[0] + 1, cell[1]};
    const Index<2> ym{cell[0], cell[1] - 1};
    const Index<2> yp{cell[0], cell[1] + 1};
    return (harmonic(coefficient[0](xm, 0), coefficient[0](cell, 0)) +
            harmonic(coefficient[0](cell, 0), coefficient[0](xp, 0))) *
               inverse_dx * inverse_dx +
           (harmonic(coefficient[3](ym, 0), coefficient[3](cell, 0)) +
            harmonic(coefficient[3](cell, 0), coefficient[3](yp, 0))) *
               inverse_dy * inverse_dy;
  }
};

struct ResidualKernel {
  FieldView<Real, 2> residual{};
  FieldView<const Real, 2> rhs{};
  FieldView<const Real, 2> covered{};
  TensorStencil stencil{};
  bool mask_covered = true;
  POPS_HD void operator()(const Index<2>& index) const {
    residual(index, 0) = mask_covered && covered(index, 0) >= Real(0.5)
                             ? Real(0)
                             : rhs(index, 0) - stencil.image(index);
  }
};

struct JacobiKernel {
  FieldView<Real, 2> destination{};
  FieldView<const Real, 2> iterate{};
  FieldView<const Real, 2> rhs{};
  FieldView<const Real, 2> covered{};
  TensorStencil stencil{};
  Real relaxation = Real(0.65);
  bool mask_covered = true;
  POPS_HD void operator()(const Index<2>& index) const {
    if (mask_covered && covered(index, 0) >= Real(0.5)) {
      destination(index, 0) = iterate(index, 0);
      return;
    }
    const Real diagonal = stencil.diagonal(index);
    destination(index, 0) =
        iterate(index, 0) + relaxation * (rhs(index, 0) - stencil.image(index)) / diagonal;
  }
};

struct LinearInterpolationKernel {
  FieldView<const Real, 2> coarse{};
  FieldView<Real, 2> fine{};
  Box<2> coarse_domain{};
  Box<2> fine_domain{};
  ::pops::amr::RefinementRatio<2> ratio{};
  POPS_HD void operator()(const Index<2>& index) const {
    Index<2> parent{};
    Real offset[2]{};
    for (int axis = 0; axis < 2; ++axis) {
      const std::int64_t relative = static_cast<std::int64_t>(index[axis]) - fine_domain.lo[axis];
      std::int64_t quotient = relative / ratio[axis];
      if (relative % ratio[axis] < 0)
        --quotient;
      parent[axis] = static_cast<int>(static_cast<std::int64_t>(coarse_domain.lo[axis]) + quotient);
      offset[axis] = (static_cast<Real>(relative) + Real(0.5)) / static_cast<Real>(ratio[axis]) -
                     (static_cast<Real>(quotient) + Real(0.5));
    }
    Real value = coarse(parent, 0);
    for (int axis = 0; axis < 2; ++axis) {
      Index<2> lower = parent;
      Index<2> upper = parent;
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

struct RestrictionKernel {
  FieldView<const Real, 2> fine{};
  FieldView<Real, 2> coarse{};
  Box<2> coarse_domain{};
  Box<2> fine_domain{};
  ::pops::amr::RefinementRatio<2> ratio{};
  Real inverse_children = Real(1);
  POPS_HD void operator()(const Index<2>& parent) const {
    Index<2> base{};
    for (int axis = 0; axis < 2; ++axis)
      base[axis] = static_cast<int>(
          static_cast<std::int64_t>(fine_domain.lo[axis]) +
          (static_cast<std::int64_t>(parent[axis]) - coarse_domain.lo[axis]) * ratio[axis]);
    Real sum = Real(0);
    for (int y = 0; y < ratio[1]; ++y)
      for (int x = 0; x < ratio[0]; ++x)
        sum += fine(Index<2>{base[0] + x, base[1] + y}, 0);
    coarse(parent, 0) = sum * inverse_children;
  }
};

inline void copy_region(FieldView<Real, 2> destination, FieldView<const Real, 2> source,
                        const Box<2>& region) {
  if (!region.empty())
    for_each_cell(region, CopyKernel{destination, source});
}

template <class MemorySpace>
void copy_valid(MultiFab<2, MemorySpace>& destination, const MultiFab<2, MemorySpace>& source) {
  if (destination.layout() != source.layout() ||
      destination.distribution() != source.distribution() ||
      destination.local_rank() != source.local_rank() || destination.ncomp() != 1 ||
      source.ncomp() != 1)
    throw std::invalid_argument("tensor FAC copy requires one exact scalar vector space");
  for (std::size_t local = 0; local < destination.local_size(); ++local)
    copy_region(destination.fab(local).view(), std::as_const(source.fab(local)).view(),
                destination.box(local));
  Kokkos::fence();
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

/// A prepared rank-two FAC cycle for ``-div(A grad(phi)) = rhs``.
///
/// The four coefficient fields are ordered ``Axx, Axy, Ayx, Ayy``.  The conservative flux stencil
/// is genuinely nine-point when either cross coefficient is nonzero.  Same-level halos and every
/// partitioned coarse/fine transfer own one duplicated ExecutionLane; replicated level zero is
/// retained as an explicit capability and refined contributions are broadcast from their unique
/// owners into that replicated parent.  No solve-time storage allocation is permitted.
template <class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class TensorCompositeFac {
 public:
  using field_type = MultiFab<2, MemorySpace>;

  TensorCompositeFac(std::span<const LevelBinding<MemorySpace>> bindings,
                     std::span<const ::pops::amr::RefinementRatio<2>> ratios)
      : bindings_(bindings.begin(), bindings.end()), ratios_(ratios.begin(), ratios.end()) {
    static_assert(
        Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, MemorySpace>::accessible,
        "TensorCompositeFac requires DefaultExecutionSpace access to its memory space");
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
    if (all_reduce_max(local_error ? 1L : 0L) != 0) {
      if (n_ranks() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("rank-two tensor FAC preparation failed collectively");
    }

    exact_contract_ = build_exact_contract_();
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"pops-rank2-tensor-fac", std::string_view(exact_contract_)}}))
      throw std::invalid_argument("rank-two tensor FAC hierarchy differs between MPI ranks");
    for (std::size_t connection = 0; connection < connections_.size(); ++connection)
      if (!all_ranks_agree_exact_ordered_byte_pairs(
              {{"pops-rank2-tensor-parent-gather",
                std::string_view(connections_[connection]->gather_contract)},
               {"pops-rank2-tensor-fine-restriction",
                std::string_view(connections_[connection]->restriction_contract)}}))
        throw std::invalid_argument(
            "rank-two tensor FAC coarse/fine schedule differs between MPI ranks");

    lane_.emplace(
        ExecutionLane::duplicate_world_collectively("pops.runtime.amr.rank2-tensor-composite-fac"));
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

  TensorCompositeFac(const TensorCompositeFac&) = delete;
  TensorCompositeFac& operator=(const TensorCompositeFac&) = delete;
  TensorCompositeFac(TensorCompositeFac&&) = delete;
  TensorCompositeFac& operator=(TensorCompositeFac&&) = delete;

  std::string_view exact_prepared_contract() const noexcept { return exact_contract_; }

  bool owns_execution_lane() const noexcept {
    return lane_ && (n_ranks() == 1 || lane_->owns_communicator());
  }

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

  SolveReport solve(const Controls& controls) {
    detail::validate_controls(controls);
    for (std::size_t level = 0; level < levels_.size(); ++level)
      detail::copy_valid(*levels_[level]->binding.solution, *levels_[level]->binding.initial_guess);
    fill_all_coefficient_ghosts_();
    if (!coefficients_are_elliptic_()) {
      SolveReport report;
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                         "rank2_tensor_fac_non_elliptic_coefficient");
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
                         "rank2_tensor_fac_non_finite_initial_residual");
      return report;
    }
    if (reference <= stop) {
      fill_all_solution_ghosts_();
      report.mark_solved("rank2_tensor_fac_initial_residual");
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

      compute_composite_residual_();
      ++report.evaluations;
      report.iters = iteration + 1;
      report.residual_norm = composite_residual_norm_();
      report.rel_residual = report.residual_norm / reference;
      if (!std::isfinite(static_cast<double>(report.residual_norm))) {
        report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                           "rank2_tensor_fac_non_finite_iteration");
        return report;
      }
      if (report.residual_norm <= stop) {
        fill_all_solution_ghosts_();
        report.mark_solved("rank2_tensor_fac_converged");
        return report;
      }
    }
    report.mark_failed(SolveStatus::kIterationLimit, SolveAction::kFailRun,
                       "rank2_tensor_fac_iteration_limit");
    return report;
  }

 private:
  struct Level {
    LevelBinding<MemorySpace> binding;
    field_type residual;
    field_type scratch;
    field_type correction;
    field_type covered;
    field_type active;
    HaloSchedule<2> halo_schedule;
    PreparedPhysicalBoundary<2> physical_boundary;
    PreparedPhysicalBoundary<2> homogeneous_boundary;
    PreparedPhysicalBoundary<2> coefficient_boundary;
    std::optional<HaloExchange<2, MemorySpace>> halo_exchange{};

    Level(LevelBinding<MemorySpace> source, bool full_domain)
        : binding(source),
          residual(source.solution->layout(), source.solution->distribution(),
                   source.solution->local_rank(), 1, Extent<2>{}),
          scratch(source.solution->layout(), source.solution->distribution(),
                  source.solution->local_rank(), 1, Extent<2>{}),
          correction(source.solution->layout(), source.solution->distribution(),
                     source.solution->local_rank(), 1, detail::one_ghost()),
          covered(source.solution->layout(), source.solution->distribution(),
                  source.solution->local_rank(), 1, Extent<2>{}),
          active(source.solution->layout(), source.solution->distribution(),
                 source.solution->local_rank(), 1, Extent<2>{}),
          halo_schedule(prepare_halo_schedule(
              *source.solution, source.geometry->domain(), source.boundary->topology(),
              full_domain ? HaloLayoutCoverage::full_domain : HaloLayoutCoverage::sparse_level,
              detail::exact_halo_budget(source.solution->layout(), source.geometry->domain()))),
          physical_boundary(prepare_physical_boundary(source.geometry->domain(),
                                                      detail::one_ghost(), *source.boundary,
                                                      detail::exact_boundary_budget())),
          homogeneous_boundary(prepare_physical_boundary(
              source.geometry->domain(), detail::one_ghost(),
              detail::boundary_with_values(*source.boundary, *source.geometry, true, false),
              detail::exact_boundary_budget())),
          coefficient_boundary(prepare_physical_boundary(
              source.geometry->domain(), detail::one_ghost(),
              detail::boundary_with_values(*source.boundary, *source.geometry, false, true),
              detail::exact_boundary_budget())) {
      residual.set_val(Real(0));
      scratch.set_val(Real(0));
      correction.set_val(Real(0));
      covered.set_val(Real(0));
      active.set_val(Real(1));
    }
  };

  struct Connection {
    using transfer_job = elliptic::amr::partitioned_transfer::RegionTransferJob<2>;
    using transfer_plan = elliptic::amr::partitioned_transfer::RegionTransferPlan<2>;
    using transport_type = elliptic::amr::partitioned_transfer::RegionTransport<2, MemorySpace>;
    using host_mirror_type = typename Fab<2, MemorySpace>::host_mirror_type;

    struct ScratchPatch {
      std::size_t fine_patch = 0;
      Fab<2, MemorySpace> parent_staging{};
      Fab<2, MemorySpace> restricted{};
      std::vector<Box<2>> ghost_regions{};
      std::optional<host_mirror_type> restricted_host{};
      std::vector<Real> broadcast_buffer{};

      ScratchPatch(std::size_t patch, const Box<2>& staging, const Box<2>& restriction,
                   std::vector<Box<2>> regions, bool broadcast)
          : fine_patch(patch),
            parent_staging(staging, 1, Extent<2>{}),
            restricted(restriction, 1, Extent<2>{}),
            ghost_regions(std::move(regions)) {
        if (broadcast) {
          restricted_host.emplace(restricted.create_host_mirror());
          broadcast_buffer.resize(restricted.size());
        }
      }
    };

    Level* parent = nullptr;
    Level* child = nullptr;
    ::pops::amr::RefinementRatio<2> ratio{};
    bool replicated_parent = false;
    std::vector<ScratchPatch> scratch{};
    std::vector<std::size_t> scratch_by_fine_patch{};
    std::unique_ptr<transport_type> gather{};
    std::unique_ptr<transport_type> restriction{};
    std::string gather_contract{};
    std::string restriction_contract{};
    const ExecutionLane* lane = nullptr;

    static constexpr std::size_t no_scratch = std::numeric_limits<std::size_t>::max();

    Connection(Level& parent_level, Level& child_level, ::pops::amr::RefinementRatio<2> level_ratio,
               std::size_t ordinal)
        : parent(&parent_level),
          child(&child_level),
          ratio(level_ratio),
          replicated_parent(parent_level.binding.solution->distribution().replicated()) {
      const Extent<2> ratio_value = detail::ratio_extent(ratio);
      const std::size_t fine_count = child->binding.solution->layout().size();
      scratch_by_fine_patch.assign(fine_count, no_scratch);
      scratch.reserve(replicated_parent ? fine_count : child->binding.solution->local_size());
      for (std::size_t fine_patch = 0; fine_patch < fine_count; ++fine_patch) {
        if (!replicated_parent && !child->binding.solution->contains_local(fine_patch))
          continue;
        const Box<2>& valid = child->binding.solution->layout()[fine_patch];
        const Box<2> staging =
            coarsen(detail::clipped_growth(valid, child->binding.geometry->domain()), ratio_value)
                .grow(1)
                .intersect(parent->binding.geometry->domain());
        const Box<2> restricted_box = coarsen(valid, ratio_value);
        std::vector<Box<2>> pending{
            detail::clipped_growth(valid, child->binding.geometry->domain())};
        for (const Box<2>& peer : child->binding.solution->layout().boxes())
          detail::subtract_from(pending, peer);
        scratch_by_fine_patch[fine_patch] = scratch.size();
        scratch.emplace_back(fine_patch, staging, restricted_box, std::move(pending),
                             replicated_parent);
      }

      if (replicated_parent) {
        gather_contract =
            "pops.rank2-tensor-fac.replicated-parent-gather/" + std::to_string(ordinal);
        restriction_contract =
            "pops.rank2-tensor-fac.replicated-parent-restriction/" + std::to_string(ordinal);
        return;
      }

      std::vector<transfer_job> gather_jobs;
      std::vector<transfer_job> restriction_jobs;
      const std::size_t parent_count = parent->binding.solution->layout().size();
      gather_jobs.reserve(
          detail::checked_product(fine_count, parent_count, "tensor FAC gather pair overflow"));
      restriction_jobs.reserve(gather_jobs.capacity());
      for (std::size_t fine_patch = 0; fine_patch < fine_count; ++fine_patch) {
        const Box<2>& valid = child->binding.solution->layout()[fine_patch];
        const Box<2> staging =
            coarsen(detail::clipped_growth(valid, child->binding.geometry->domain()), ratio_value)
                .grow(1)
                .intersect(parent->binding.geometry->domain());
        const Box<2> footprint = coarsen(valid, ratio_value);
        std::int64_t gathered_cells = 0;
        std::int64_t restricted_cells = 0;
        for (std::size_t parent_patch = 0; parent_patch < parent_count; ++parent_patch) {
          const Box<2> gathered =
              staging.intersect(parent->binding.solution->layout()[parent_patch]);
          if (!gathered.empty()) {
            gathered_cells += gathered.numPts();
            gather_jobs.push_back(transfer_job{
                parent_patch, fine_patch,
                parent->binding.solution->distribution().owner(parent_patch),
                child->binding.solution->distribution().owner(fine_patch), gathered, gathered});
          }
          const Box<2> restricted_region =
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
              "rank-two tensor FAC parent layout does not cover a refined footprint");
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
          gather->plan().exact_contract("rank2-tensor-parent-gather/" + std::to_string(ordinal));
      restriction_contract = restriction->plan().exact_contract("rank2-tensor-fine-restriction/" +
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
      std::vector<Index<2>> ranks;
      ranks.reserve(detail::checked_product(jobs.size(), 2, "tensor FAC rank budget overflow"));
      for (const transfer_job& job : jobs) {
        ranks.push_back(job.source_rank);
        ranks.push_back(job.destination_rank);
      }
      std::sort(ranks.begin(), ranks.end(), [](const Index<2>& left, const Index<2>& right) {
        return left[1] < right[1] || (left[1] == right[1] && left[0] < right[0]);
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
        throw std::out_of_range("rank-two tensor FAC scratch patch is not materialized");
      return scratch.at(local);
    }

    void gather_parent(const field_type& source) {
      if (replicated_parent) {
        for (ScratchPatch& patch : scratch) {
          patch.parent_staging.set_val(Real(0));
          for (std::size_t parent_patch = 0; parent_patch < source.layout().size();
               ++parent_patch) {
            const Box<2> region =
                patch.parent_staging.box().intersect(source.layout()[parent_patch]);
            detail::copy_region(patch.parent_staging.view(),
                                std::as_const(source.fab_global(parent_patch)).view(), region);
          }
        }
        Kokkos::fence();
        return;
      }
      auto source_view = [&source](const transfer_job& job) -> FieldView<const Real, 2> {
        return std::as_const(source.fab_global(job.source_patch)).view();
      };
      auto destination_view = [this](const transfer_job& job) -> FieldView<Real, 2> {
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
        for (const Box<2>& region : patch.ghost_regions)
          for_each_cell(region, detail::LinearInterpolationKernel{
                                    coarse, fine, parent->binding.geometry->domain(),
                                    child->binding.geometry->domain(), ratio});
      }
      Kokkos::fence();
    }

    void prolong_valid(field_type& destination) {
      for (ScratchPatch& patch : scratch) {
        if (!destination.contains_local(patch.fine_patch))
          continue;
        const auto coarse = std::as_const(patch.parent_staging).view();
        auto fine = destination.fab_global(patch.fine_patch).view();
        for_each_cell(
            destination.fab_global(patch.fine_patch).box(),
            detail::LinearInterpolationKernel{coarse, fine, parent->binding.geometry->domain(),
                                              child->binding.geometry->domain(), ratio});
      }
      Kokkos::fence();
    }

    void restrict_into(const field_type& source, field_type& destination) {
      const Real inverse_children = Real(1) / static_cast<Real>(ratio.child_count());
      for (ScratchPatch& patch : scratch) {
        if (!source.contains_local(patch.fine_patch))
          continue;
        for_each_cell(
            patch.restricted.box(),
            detail::RestrictionKernel{std::as_const(source.fab_global(patch.fine_patch)).view(),
                                      patch.restricted.view(), parent->binding.geometry->domain(),
                                      child->binding.geometry->domain(), ratio, inverse_children});
      }
      Kokkos::fence();
      if (replicated_parent) {
        broadcast_and_publish_restriction_(destination);
        return;
      }
      auto source_view = [this](const transfer_job& job) -> FieldView<const Real, 2> {
        return std::as_const(scratch_for(job.source_patch).restricted).view();
      };
      auto destination_view = [&destination](const transfer_job& job) -> FieldView<Real, 2> {
        return destination.fab_global(job.destination_patch).view();
      };
      restriction->execute(source_view, destination_view);
    }

    void broadcast_and_publish_restriction_(field_type& destination) {
      if (lane == nullptr)
        throw std::logic_error("rank-two tensor FAC replicated restriction has no lane");
      for (ScratchPatch& patch : scratch) {
        const Index<2>& owner = child->binding.solution->distribution().owner(patch.fine_patch);
        const int root = static_cast<int>(child->binding.solution->rank_space().linear_rank(owner));
        long packing_failure = 0;
        try {
          if (!patch.restricted_host ||
              patch.broadcast_buffer.size() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error(
                "rank-two tensor restriction has an invalid broadcast allocation");
          if (lane->rank() == root) {
            patch.restricted.copy_to_host(*patch.restricted_host);
            for (std::size_t element = 0; element < patch.broadcast_buffer.size(); ++element)
              patch.broadcast_buffer[element] = (*patch.restricted_host)(element);
          }
        } catch (...) {
          packing_failure = 1;
        }
        if (all_reduce_max(packing_failure, lane->communicator()) != 0)
          throw std::runtime_error(
              "rank-two tensor replicated restriction packing failed collectively");
#ifdef POPS_HAS_MPI
        if (lane->size() > 1) {
          const int code = MPI_Bcast(patch.broadcast_buffer.data(),
                                     static_cast<int>(patch.broadcast_buffer.size()), MPI_DOUBLE,
                                     root, lane->native_handle());
          if (all_reduce_max(code == MPI_SUCCESS ? 0L : 1L, lane->communicator()) != 0)
            throw std::runtime_error(
                "rank-two tensor replicated restriction broadcast failed collectively");
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
            const Box<2> region =
                patch.restricted.box().intersect(destination.layout()[parent_patch]);
            detail::copy_region(destination.fab_global(parent_patch).view(),
                                std::as_const(patch.restricted).view(), region);
          }
          Kokkos::fence();
        } catch (...) {
          publication_failure = 1;
        }
        if (all_reduce_max(publication_failure, lane->communicator()) != 0)
          throw std::runtime_error(
              "rank-two tensor replicated restriction publication failed collectively");
      }
    }
  };

  void validate_bindings_() const {
    if (bindings_.size() < 2 || ratios_.size() + 1 != bindings_.size())
      throw std::invalid_argument("rank-two tensor FAC requires a populated refined hierarchy");
    const auto& rank_space = bindings_.front().solution->rank_space();
    const Index<2> local_rank = bindings_.front().solution->local_rank();
    if (rank_space.size() != static_cast<std::size_t>(n_ranks()) ||
        rank_space.linear_rank(local_rank) != static_cast<std::size_t>(my_rank()))
      throw std::invalid_argument("rank-two tensor FAC process space differs from MPI world");
    for (std::size_t level = 0; level < bindings_.size(); ++level) {
      const auto& binding = bindings_[level];
      if (binding.geometry == nullptr || binding.boundary == nullptr || binding.rhs == nullptr ||
          binding.initial_guess == nullptr || binding.solution == nullptr ||
          std::any_of(binding.coefficients.begin(), binding.coefficients.end(),
                      [](const field_type* value) { return value == nullptr; }))
        throw std::invalid_argument("rank-two tensor FAC level binding is incomplete");
      const field_type& solution = *binding.solution;
      if (solution.ncomp() != 1 || binding.rhs->ncomp() != 1 ||
          binding.initial_guess->ncomp() != 1 || solution.layout().empty() ||
          solution.rank_space() != rank_space || solution.local_rank() != local_rank ||
          !solution.distribution().matches_layout(solution.layout()) ||
          !solution.layout().is_disjoint_within(binding.geometry->domain(),
                                                detail::exact_layout_budget(solution.layout())))
        throw std::invalid_argument("rank-two tensor FAC level has an invalid exact layout");
      if (level == 0 &&
          !solution.layout().tiles_exactly(binding.geometry->domain(),
                                           detail::exact_layout_budget(solution.layout())))
        throw std::invalid_argument("rank-two tensor FAC coarse layout must tile its domain");
      for (int axis = 0; axis < 2; ++axis) {
        if (solution.ghosts()[axis] < 1 || binding.initial_guess->ghosts()[axis] < 1 ||
            binding.rhs->ghosts()[axis] != 0 ||
            binding.boundary->spacing()[axis] != binding.geometry->spacing(axis))
          throw std::invalid_argument(
              "rank-two tensor FAC field ghosts or boundary spacing are incompatible");
      }
      const auto exact_shape = [&solution](const field_type& field, bool require_ghost) {
        if (field.layout() != solution.layout() ||
            field.distribution() != solution.distribution() ||
            field.local_rank() != solution.local_rank() || field.ncomp() != 1)
          return false;
        if (require_ghost)
          for (int axis = 0; axis < 2; ++axis)
            if (field.ghosts()[axis] < 1)
              return false;
        return true;
      };
      if (!exact_shape(*binding.rhs, false) || !exact_shape(*binding.initial_guess, true) ||
          std::any_of(binding.coefficients.begin(), binding.coefficients.end(),
                      [&](const field_type* field) { return !exact_shape(*field, true); }))
        throw std::invalid_argument(
            "rank-two tensor FAC coefficients and vectors do not share one exact level space");
      for (int axis = 0; axis < 2; ++axis)
        for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
          const Face<2> face{axis, side};
          if (!binding.boundary->topology().is_periodic(face) &&
              binding.boundary->at(face).kind == PhysicalBoundaryKind::external)
            throw std::invalid_argument(
                "rank-two tensor FAC requires an authored law on every physical face");
        }
      if (level == 0)
        continue;
      if (solution.distribution().replicated())
        throw std::invalid_argument(
            "rank-two tensor FAC refined levels require unique partitioned ownership");
      const Extent<2> ratio = detail::ratio_extent(ratios_[level - 1]);
      for (int axis = 0; axis < 2; ++axis)
        if (ratios_[level - 1][axis] < 2)
          throw std::invalid_argument("rank-two tensor FAC refinement ratios must be at least two");
      if (*binding.geometry != bindings_[level - 1].geometry->refine(ratio) ||
          binding.boundary->topology() != bindings_[level - 1].boundary->topology())
        throw std::invalid_argument(
            "rank-two tensor FAC adjacent geometry or topology is not an exact refinement");
      for (const Box<2>& patch : solution.layout().boxes()) {
        if (refine(coarsen(patch, ratio), ratio) != patch)
          throw std::invalid_argument("rank-two tensor FAC fine patch is not refinement-aligned");
        const Box<2> grown = patch.grow(1);
        for (int axis = 0; axis < 2; ++axis)
          if ((grown.lo[axis] < binding.geometry->domain().lo[axis] ||
               grown.hi[axis] > binding.geometry->domain().hi[axis]) &&
              binding.boundary->topology().is_periodic(Face<2>{axis, BoundarySide::lower}))
            throw std::invalid_argument(
                "rank-two tensor FAC periodic sparse patches may not cross the domain seam");
      }
    }
  }

  std::string build_exact_contract_() const {
    ExactContractBuilder contract;
    contract.text("pops.runtime.amr.rank2-tensor-composite-fac")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{2})
        .scalar(static_cast<std::uint64_t>(bindings_.size()));
    for (const auto& binding : bindings_) {
      for (int axis = 0; axis < 2; ++axis)
        contract.scalar(binding.geometry->domain().lo[axis])
            .scalar(binding.geometry->domain().hi[axis])
            .scalar(binding.geometry->lower()[axis])
            .scalar(binding.geometry->upper()[axis]);
      contract.scalar(binding.solution->distribution().mode())
          .sequence(binding.solution->layout().boxes(),
                    [](ExactContractBuilder& item, const Box<2>& patch) {
                      for (int axis = 0; axis < 2; ++axis)
                        item.scalar(patch.lo[axis]).scalar(patch.hi[axis]);
                    })
          .sequence(binding.solution->distribution().owners(),
                    [](ExactContractBuilder& item, const Index<2>& owner) {
                      item.scalar(owner[0]).scalar(owner[1]);
                    });
    }
    contract.scalar(static_cast<std::uint64_t>(ratios_.size()));
    for (const auto& ratio : ratios_)
      contract.scalar(ratio[0]).scalar(ratio[1]);
    return std::move(contract).release();
  }

  static void mark_coverage_(Level& parent, const Level& child,
                             const ::pops::amr::RefinementRatio<2>& ratio) {
    const Extent<2> ratio_value = detail::ratio_extent(ratio);
    for (const Box<2>& fine_patch : child.binding.solution->layout().boxes()) {
      const Box<2> footprint = coarsen(fine_patch, ratio_value);
      for (std::size_t local = 0; local < parent.binding.solution->local_size(); ++local) {
        const Box<2> region = parent.binding.solution->box(local).intersect(footprint);
        if (!region.empty()) {
          for_each_cell(region, detail::SetKernel<Real>{parent.covered.fab(local).view(), Real(1)});
          for_each_cell(region, detail::SetKernel<Real>{parent.active.fab(local).view(), Real(0)});
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

  void fill_ghosts_(std::size_t level_index, field_type& field, const field_type* parent_field,
                    const PreparedPhysicalBoundary<2>& boundary) {
    Level& level = *levels_[level_index];
    if (level_index > 0) {
      if (parent_field == nullptr)
        throw std::logic_error("rank-two tensor FAC fine fill has no parent field");
      Connection& connection = *connections_[level_index - 1];
      connection.gather_parent(*parent_field);
      connection.interpolate_ghosts(field);
    }
    same_level_fill_(level, field);
    fill_physical_boundary(field, boundary);
  }

  void fill_all_coefficient_ghosts_() {
    for (std::size_t level = 0; level < levels_.size(); ++level)
      for (std::size_t coefficient = 0; coefficient < 4; ++coefficient) {
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
        const auto xx = level->binding.coefficients[0]->fab(local).storage();
        const auto xy = level->binding.coefficients[1]->fab(local).storage();
        const auto yx = level->binding.coefficients[2]->fab(local).storage();
        const auto yy = level->binding.coefficients[3]->fab(local).storage();
        const std::size_t count = xx.extent(0);
        long patch_invalid = 0;
        Kokkos::parallel_reduce(
            "pops_rank2_tensor_ellipticity", Kokkos::RangePolicy<>(0, count),
            KOKKOS_LAMBDA(const std::size_t index, long& invalid) {
              const Real a_xx = xx(index);
              const Real a_xy = xy(index);
              const Real a_yx = yx(index);
              const Real a_yy = yy(index);
              const Real symmetric_cross = Real(0.5) * (a_xy + a_yx);
              const Real determinant = a_xx * a_yy - symmetric_cross * symmetric_cross;
              constexpr Real infinity = std::numeric_limits<Real>::infinity();
              const bool finite = a_xx == a_xx && a_xy == a_xy && a_yx == a_yx && a_yy == a_yy &&
                                  a_xx != infinity && a_xx != -infinity && a_xy != infinity &&
                                  a_xy != -infinity && a_yx != infinity && a_yx != -infinity &&
                                  a_yy != infinity && a_yy != -infinity;
              if (!(a_xx > Real(0)) || !(a_yy > Real(0)) || !(determinant > Real(0)) || !finite)
                invalid = 1;
            },
            Kokkos::Max<long>(patch_invalid));
        local_invalid = std::max(local_invalid, patch_invalid);
      }
    Kokkos::fence();
    return all_reduce_max(local_invalid, lane_->communicator()) == 0;
  }

  detail::TensorStencil stencil_(const Level& level, std::size_t local,
                                 const field_type& field) const {
    detail::TensorStencil stencil;
    stencil.phi = std::as_const(field.fab(local)).view();
    for (std::size_t coefficient = 0; coefficient < 4; ++coefficient)
      stencil.coefficient[coefficient] =
          std::as_const(level.binding.coefficients[coefficient]->fab(local)).view();
    stencil.inverse_dx = Real(1) / level.binding.geometry->spacing(0);
    stencil.inverse_dy = Real(1) / level.binding.geometry->spacing(1);
    return stencil;
  }

  void smooth_(std::size_t level_index, field_type& iterate, const field_type& rhs, int sweeps,
               bool mask_covered, bool homogeneous) {
    if (sweeps <= 0)
      return;
    Level& level = *levels_[level_index];
    for (int sweep = 0; sweep < sweeps; ++sweep) {
      fill_solution_ghosts_(level_index, iterate, homogeneous);
      for (std::size_t local = 0; local < iterate.local_size(); ++local)
        for_each_cell(iterate.box(local),
                      detail::JacobiKernel{
                          level.scratch.fab(local).view(), std::as_const(iterate.fab(local)).view(),
                          std::as_const(rhs.fab(local)).view(),
                          std::as_const(level.covered.fab(local)).view(),
                          stencil_(level, local, iterate), Real(0.65), mask_covered});
      Kokkos::fence();
      detail::copy_valid(iterate, level.scratch);
    }
  }

  void compute_level_residual_(std::size_t level_index) {
    Level& level = *levels_[level_index];
    field_type& solution = *level.binding.solution;
    fill_solution_ghosts_(level_index, solution, false);
    for (std::size_t local = 0; local < solution.local_size(); ++local)
      for_each_cell(solution.box(local),
                    detail::ResidualKernel{level.residual.fab(local).view(),
                                           std::as_const(level.binding.rhs->fab(local)).view(),
                                           std::as_const(level.covered.fab(local)).view(),
                                           stencil_(level, local, solution), true});
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

  void solve_coarse_correction_(const Controls& controls) {
    Level& coarse = *levels_.front();
    coarse.correction.set_val(Real(0));
    const Real reference = global_norm_inf_(coarse.residual);
    const Real stop = std::max(controls.coarse_absolute_tolerance,
                               controls.coarse_relative_tolerance * reference);
    for (int sweep = 0; sweep < controls.coarse_cycles; ++sweep) {
      smooth_(0, coarse.correction, coarse.residual, 1, false, true);
      if ((sweep + 1) % 8 == 0 || sweep + 1 == controls.coarse_cycles) {
        fill_solution_ghosts_(0, coarse.correction, true);
        for (std::size_t local = 0; local < coarse.correction.local_size(); ++local)
          for_each_cell(coarse.correction.box(local),
                        detail::ResidualKernel{coarse.scratch.fab(local).view(),
                                               std::as_const(coarse.residual.fab(local)).view(),
                                               std::as_const(coarse.covered.fab(local)).view(),
                                               stencil_(coarse, local, coarse.correction), false});
        Kokkos::fence();
        if (global_norm_inf_(coarse.scratch) <= stop)
          break;
      }
    }
  }

  static void add_active_(Level& level, const field_type& correction) {
    for (std::size_t local = 0; local < level.binding.solution->local_size(); ++local)
      for_each_cell(level.binding.solution->box(local),
                    detail::ActiveAddKernel{level.binding.solution->fab(local).view(),
                                            std::as_const(correction.fab(local)).view(),
                                            std::as_const(level.active.fab(local)).view()});
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
      connections_[child - 1]->restrict_into(*levels_[child]->binding.solution,
                                             *levels_[child - 1]->binding.solution);
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

  std::vector<LevelBinding<MemorySpace>> bindings_;
  std::vector<::pops::amr::RefinementRatio<2>> ratios_;
  // The owning lane must outlive HaloExchange and RegionTransport borrows. Member destruction is
  // reverse declaration order, so it deliberately precedes every prepared communication object.
  std::optional<ExecutionLane> lane_{};
  std::vector<std::unique_ptr<Level>> levels_;
  std::vector<std::unique_ptr<Connection>> connections_;
  std::string exact_contract_{};
};

}  // namespace pops::runtime::program::tensor_fac_2d
