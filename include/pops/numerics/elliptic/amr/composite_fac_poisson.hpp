/// @file
/// @brief Exact-rank partitioned-MPI composite FAC Poisson solver.

#pragma once

#include <pops/amr/refinement_ratio.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/mesh/boundary/halo_exchange.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/amr/partitioned_region_transfer.hpp>
#include <pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp>
#include <pops/numerics/elliptic/mg/composite_fac_nlevel.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/elliptic/interface/elliptic_solver.hpp>
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/numerics/elliptic/interface/field_nonlinear.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_workspace.hpp>
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
  bool singular_nullspace = true;
  bool variable_coefficient = true;
  bool embedded_boundary = true;

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
struct ScaleKernel {
  FieldView<Real, Dim> values{};
  Real factor = Real(1);
  POPS_HD void operator()(const Index<Dim>& index) const { values(index, 0) *= factor; }
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
struct AddKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> increment{};
  POPS_HD void operator()(const Index<Dim>& index) const {
    destination(index, 0) += increment(index, 0);
  }
};

template <int Dim>
struct LinearInterpolationKernel {
  FieldView<const Real, Dim> coarse{};
  FieldView<Real, Dim> fine{};
  Box<Dim> coarse_domain{};
  Box<Dim> fine_domain{};
  ::pops::amr::RefinementRatio<Dim> ratio{};
  bool periodic[Dim]{};
  POPS_HD void operator()(const Index<Dim>& index) const {
    Index<Dim> parent{};
    Real offset[Dim]{};
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
      if (periodic[axis] ||
          (parent[axis] != coarse_domain.lo[axis] && parent[axis] != coarse_domain.hi[axis]))
        slope = Real(0.5) * (coarse(upper, 0) - coarse(lower, 0));
      else if (parent[axis] == coarse_domain.lo[axis])
        slope = coarse(upper, 0) - coarse(parent, 0);
      else
        slope = coarse(parent, 0) - coarse(lower, 0);
      value += offset[axis] * slope;
    }
    fine(index, 0) = value;
  }
};

template <int Dim>
struct CopyByIndexKernel {
  FieldView<const Real, Dim> source{};
  FieldView<Real, Dim> destination{};
  POPS_HD void operator()(const Index<Dim>& index) const {
    destination(index, 0) = source(index, 0);
  }
};

template <int Dim>
Index<Dim> index_from_ordinal(const Box<Dim>& box, std::size_t ordinal) {
  Index<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t length = static_cast<std::size_t>(box.length(axis));
    index[axis] = box.lo[axis] + static_cast<int>(ordinal % length);
    ordinal /= length;
  }
  return index;
}

template <int Dim>
std::size_t grown_offset(const Box<Dim>& grown, const Index<Dim>& index) {
  std::size_t offset = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    offset += static_cast<std::size_t>(index[axis] - grown.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(grown.length(axis));
  }
  return offset;
}

template <int Dim>
struct WrapStagingKernel {
  FieldView<Real, Dim> staging{};
  Box<Dim> domain{};
  Box<Dim> staging_box{};
  bool periodic[Dim]{};
  POPS_HD void operator()(const Index<Dim>& index) const {
    if (domain.contains(index))
      return;
    Index<Dim> wrapped = index;
    bool shifted = false;
    for (int axis = 0; axis < Dim; ++axis) {
      if (!periodic[axis])
        continue;
      const int length = static_cast<int>(domain.length(axis));
      if (length <= 0)
        continue;
      if (wrapped[axis] < domain.lo[axis]) {
        wrapped[axis] += length;
        shifted = true;
      } else if (wrapped[axis] > domain.hi[axis]) {
        wrapped[axis] -= length;
        shifted = true;
      }
    }
    if (shifted && domain.contains(wrapped) && staging_box.contains(wrapped))
      staging(index, 0) = staging(wrapped, 0);
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
  using nonlinear_workspace_type = AmrFieldNewtonKrylovWorkspace<Dim, MemorySpace>;
  using nonlinear_hierarchy_type = typename nonlinear_workspace_type::hierarchy_type;

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
                std::string_view(connections_[connection]->restriction_contract)},
               {std::string_view("pops-fac-flux-mismatch"),
                std::string_view(connections_[connection]->flux_contract)}}))
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
    build_coarse_solver_(request.levels.front());
    try_prepare_fft_coarse_();
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

  void install_newton(FieldNewtonOptions options) {
    if (newton_workspace_)
      throw std::logic_error("partitioned FAC Newton authority is already installed");
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
      throw std::logic_error("partitioned FAC boundary kernel is already installed");
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
      throw std::logic_error("partitioned FAC has no compiled dynamic boundary kernel");
    if (!contexts || contexts->size() != levels_.size())
      throw std::invalid_argument(
          "partitioned FAC requires one dynamic boundary context per live AMR level");
    for (const FieldBoundaryExecutionContext<Dim>& context : contexts->contexts())
      if (context.failure == nullptr)
        throw std::invalid_argument(
            "partitioned FAC dynamic boundary requires fallible execution channels");
    boundary_contexts_ = std::move(contexts);
  }

  void install_nullspace(FieldNullspacePlan<Dim> plan,
                         std::vector<PreparedVectorDistribution<Dim>> distributions) {
    if (nullspace_workspace_)
      throw std::logic_error("partitioned FAC nullspace authority is already installed");
    if (distributions.size() != levels_.size())
      throw std::invalid_argument(
          "partitioned FAC nullspace authority requires one distribution per hierarchy level");
    if (fac_detail::singular(levels_.front()->boundary, reaction_) != !plan.empty())
      throw std::invalid_argument(
          "partitioned FAC nullspace plan disagrees with the prepared operator kernel");

    std::vector<const MultiFab<Dim>*> rhs_layouts;
    std::vector<MultiFab<Dim>*> candidates;
    rhs_layouts.reserve(levels_.size());
    candidates.reserve(levels_.size());
    for (const auto& level : levels_) {
      rhs_layouts.push_back(&level->rhs);
      candidates.push_back(&level->phi);
    }
    nullspace_workspace_ =
        std::make_unique<FieldNullspaceWorkspace<Dim>>(plan, rhs_layouts, distributions, *lane_);
    nullspace_rhs_ = std::move(rhs_layouts);
    nullspace_candidates_ = std::move(candidates);
  }

  void install_coefficient(int level, const field_type& conductivity) {
    Level& target = *levels_.at(static_cast<std::size_t>(level));
    ::pops::elliptic::mg::WeightedPoissonFields<Dim, MemorySpace> probe;
    probe.coefficient = &conductivity;
    ::pops::elliptic::mg::validate_weighted_poisson_fields(target.phi, probe,
                                                           "partitioned FAC coefficient");
    if (!target.coefficient)
      target.coefficient.emplace(target.phi.layout(), target.phi.distribution(),
                                 target.phi.local_rank(), 1, fac_detail::unit_ghosts<Dim>());
    ::pops::elliptic::mg::copy_scalar_valid(conductivity, *target.coefficient);
    fill_coefficient_ghosts_(static_cast<std::size_t>(level));
    if (level == 0 && coarse_solver_)
      coarse_solver_->install_coefficient(*target.coefficient);
    if (level == 0) {
      fft_coarse_.reset();
      used_fft_coarse_ = false;
    }
  }

  void install_embedded_boundary(int level, const field_type& active,
                                 const field_type& inverse_volume,
                                 const field_type& aperture_lower,
                                 const field_type& aperture_upper) {
    Level& target = *levels_.at(static_cast<std::size_t>(level));
    ::pops::elliptic::mg::WeightedPoissonFields<Dim, MemorySpace> probe;
    probe.inverse_volume = &inverse_volume;
    probe.aperture_lower = &aperture_lower;
    probe.aperture_upper = &aperture_upper;
    probe.active = &active;
    ::pops::elliptic::mg::validate_weighted_poisson_fields(
        target.phi, probe, "partitioned FAC embedded boundary");
    if (nullspace_workspace_)
      throw std::logic_error(
          "partitioned FAC embedded-boundary install requires the metric before nullspace "
          "authority");
    if (target.inverse_volume)
      throw std::logic_error("partitioned FAC embedded-boundary authority is already installed");
    ::pops::elliptic::mg::copy_scalar_valid(active, target.active);
    target.inverse_volume.emplace(inverse_volume.layout(), inverse_volume.distribution(),
                                  inverse_volume.local_rank(), 1, Extent<Dim>{});
    ::pops::elliptic::mg::copy_scalar_valid(inverse_volume, *target.inverse_volume);
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
  }

  SolveReport solve() {
    if (newton_workspace_ || boundary_kernel_)
      return solve_dynamic_();
    try {
      ensure_nullspace_();
      if (nullspace_workspace_) {
        nullspace_workspace_->require_compatible(nullspace_rhs_);
        nullspace_workspace_->apply_gauge(nullspace_candidates_);
      }
      used_fft_coarse_ = false;
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
    compute_composite_residual_();
    const Real reference = composite_residual_norm_();
    SolveReport report;
    report.evaluations = 1;
    if (!std::isfinite(static_cast<double>(reference))) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                         "partitioned_fac_non_finite_initial_residual");
      last_report_ = report;
      return last_report_;
    }
    report.reference_residual_norm = reference;
    report.residual_norm = reference;
    report.rel_residual = reference > Real(0) ? Real(1) : Real(0);
    const Real stop = std::max(options_.abs_tol, options_.rel_tol * reference);
    if (reference <= stop) {
      fill_all_solution_ghosts_();
      report.mark_solved("partitioned_fac_initial_residual");
      last_report_ = report;
      return last_report_;
    }

    const int pre = (options_.fine_sweeps + 1) / 2;
    const int post = options_.fine_sweeps / 2;
    for (int iteration = 0; iteration < options_.max_iters; ++iteration) {
      for (std::size_t level = 0; level < levels_.size(); ++level)
        smooth_(level, levels_[level]->phi, levels_[level]->rhs, pre, true, false);

      compute_composite_residual_();
      restrict_residual_tower_();
      const SolveReport coarse_report = apply_fac_correction_();
      report.evaluations += coarse_report.evaluations;
      if (!coarse_report.solved()) {
        report.iters = iteration;
        report.residual_norm = composite_residual_norm_();
        report.rel_residual = report.residual_norm / reference;
        report.mark_failed(
            coarse_report.status, SolveAction::kFailRun,
            std::string("partitioned_fac_coarse_correction_failed:") + coarse_report.reason +
                " rel=" + std::to_string(static_cast<double>(coarse_report.rel_residual)) +
                " iters=" + std::to_string(coarse_report.iters));
        last_report_ = report;
        return last_report_;
      }
      report.step_norm = global_norm_inf_(levels_.front()->correction);
      for (std::size_t level = 0; level < levels_.size(); ++level)
        smooth_(level, levels_[level]->phi, levels_[level]->rhs, post, true, false);
      if (levels_.size() > 2)
        smooth_(levels_.size() - 1, levels_.back()->phi, levels_.back()->rhs, options_.fine_sweeps,
                true, false);
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
          residual_operator_view(request.boxes, request.distribution, request.local_rank, 1,
                                 fac_detail::unit_ghosts<Dim>()),
          direction_operator_view(request.boxes, request.distribution, request.local_rank, 1,
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
                                        fac_detail::exact_boundary_budget<Dim>())),
          coefficient_boundary(prepare_physical_boundary(
              geometry.domain(), fac_detail::unit_ghosts<Dim>(),
              ::pops::elliptic::mg::detail::coefficient_boundary_for_geometry(boundary, geometry),
              fac_detail::exact_boundary_budget<Dim>())) {
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
    using transfer_job = partitioned_transfer::RegionTransferJob<Dim>;
    using transfer_plan = partitioned_transfer::RegionTransferPlan<Dim>;
    using transport_type = partitioned_transfer::RegionTransport<Dim, MemorySpace>;

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
    ::pops::amr::RefinementRatio<Dim> ratio{};
    std::vector<ScratchPatch> scratch{};
    std::vector<std::size_t> scratch_by_fine_patch{};
    std::unique_ptr<transport_type> gather{};
    std::unique_ptr<transport_type> restriction{};
    std::unique_ptr<transport_type> flux{};
    std::string gather_contract{};
    std::string restriction_contract{};
    std::string flux_contract{};

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
        const Box<Dim> restricted_box = coarsen(valid, ratio_value);
        const Box<Dim> staging = restricted_box.grow(2);
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
        patch.flux_increment = Fab<Dim, MemorySpace>(staging, 1, Extent<Dim>{});
        patch.covered_staging = Fab<Dim, MemorySpace>(staging, 1, Extent<Dim>{});
        patch.flux_increment.set_val(Real(0));
        patch.covered_staging.set_val(Real(0));
        for_each_cell(restricted_box,
                      fac_detail::SetScalarKernel<Dim>{patch.covered_staging.view(), Real(1)});
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
      gather_jobs.reserve(pair_count * std::max<std::size_t>(parent->phi.rank_space().size(), 1));
      restriction_jobs.reserve(pair_count);
      const auto& parent_dist = parent->phi.distribution();
      const auto& child_dist = child->phi.distribution();
      const auto& rank_space = parent->phi.rank_space();
      auto unique_owner = [](const mesh::Distribution<Dim>& distribution, std::size_t patch,
                             const Index<Dim>& fallback) -> Index<Dim> {
        return distribution.replicated() ? fallback : distribution.owner(patch);
      };
      auto receive_ranks = [&](const mesh::Distribution<Dim>& distribution,
                               std::size_t patch) -> std::vector<Index<Dim>> {
        if (!distribution.replicated())
          return {distribution.owner(patch)};
        std::vector<Index<Dim>> ranks;
        ranks.reserve(rank_space.size());
        for (std::size_t rank = 0; rank < rank_space.size(); ++rank)
          ranks.push_back(rank_space.coordinate(rank));
        return ranks;
      };
      for (std::size_t fine_patch = 0; fine_patch < child->phi.layout().size(); ++fine_patch) {
        const Box<Dim>& valid = child->phi.layout()[fine_patch];
        const Box<Dim> footprint = coarsen(valid, ratio_value);
        const Box<Dim> staging = footprint.grow(2).intersect(parent->geometry.domain());
        mesh::ExactCellCount gather_coverage;
        mesh::ExactCellCount restriction_coverage;
        for (std::size_t parent_patch = 0; parent_patch < parent->phi.layout().size();
             ++parent_patch) {
          const Box<Dim> gathered = staging.intersect(parent->phi.layout()[parent_patch]);
          if (!gathered.empty()) {
            if (!gather_coverage.add(mesh::ExactCellCount::from_box(gathered)))
              throw std::overflow_error("partitioned FAC gather coverage overflows");
            for (const Index<Dim>& dest_rank : receive_ranks(child_dist, fine_patch)) {
              const Index<Dim> source_rank = unique_owner(parent_dist, parent_patch, dest_rank);
              gather_jobs.push_back(transfer_job{parent_patch, fine_patch, source_rank, dest_rank,
                                                 gathered, gathered});
            }
          }
          const Box<Dim> restricted_region =
              footprint.intersect(parent->phi.layout()[parent_patch]);
          if (!restricted_region.empty()) {
            if (!restriction_coverage.add(mesh::ExactCellCount::from_box(restricted_region)))
              throw std::overflow_error("partitioned FAC restriction coverage overflows");
            for (const Index<Dim>& dest_rank : receive_ranks(parent_dist, parent_patch)) {
              const Index<Dim> source_rank = unique_owner(child_dist, fine_patch, dest_rank);
              restriction_jobs.push_back(transfer_job{fine_patch, parent_patch, source_rank,
                                                      dest_rank, restricted_region,
                                                      restricted_region});
            }
          }
        }
        if (gather_coverage != mesh::ExactCellCount::from_box(staging))
          throw std::invalid_argument(
              "partitioned FAC parent layout does not cover a child interpolation footprint");
        if (restriction_coverage != mesh::ExactCellCount::from_box(footprint))
          throw std::invalid_argument(
              "partitioned FAC parent layout does not cover a child restriction footprint");
      }
      std::vector<transfer_job> flux_jobs;
      flux_jobs.reserve(gather_jobs.size());
      for (const transfer_job& job : gather_jobs)
        flux_jobs.push_back(transfer_job{job.destination_patch, job.source_patch, job.destination_rank,
                                         job.source_rank, job.destination_region, job.source_region});
      gather = std::make_unique<transport_type>(
          transfer_plan{parent->phi.rank_space(), parent->phi.local_rank(), 1,
                        std::move(gather_jobs), budget.parent_gather});
      restriction = std::make_unique<transport_type>(
          transfer_plan{parent->phi.rank_space(), parent->phi.local_rank(), 1,
                        std::move(restriction_jobs), budget.fine_restriction});
      flux = std::make_unique<transport_type>(
          transfer_plan{parent->phi.rank_space(), parent->phi.local_rank(), 1, std::move(flux_jobs),
                        budget.parent_gather});
      gather_contract = gather->plan().exact_contract("parent-gather/" + std::to_string(ordinal));
      restriction_contract =
          restriction->plan().exact_contract("fine-restriction/" + std::to_string(ordinal));
      flux_contract = flux->plan().exact_contract("flux-mismatch/" + std::to_string(ordinal));
    }

    void attach_lane(const ExecutionLane& lane) {
      gather->attach_lane(lane);
      restriction->attach_lane(lane);
      flux->attach_lane(lane);
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

    void interpolate_ghosts(field_type& destination) {
      const ::pops::amr::transfer::IndexMapping<Dim> mapping{parent->geometry.domain().lo,
                                                            child->geometry.domain().lo};
      for (ScratchPatch& patch : scratch) {
        const auto coarse = std::as_const(patch.parent_staging).view();
        auto fine = destination.fab_global(patch.fine_patch).view();
        for (const Box<Dim>& region : patch.ghost_regions)
          for_each_cell(region, ::pops::elliptic::mg::fac_detail::QuadraticInterpolationTransfer<Dim>{
                                    coarse, fine, region, ratio, mapping, child->geometry.domain()});
      }
      Kokkos::fence();
    }

    void prolong_valid(field_type& destination) {
      for (ScratchPatch& patch : scratch) {
        const auto coarse = std::as_const(patch.parent_staging).view();
        auto fine = destination.fab_global(patch.fine_patch).view();
        fac_detail::LinearInterpolationKernel<Dim> kernel{
            coarse, fine, parent->geometry.domain(), child->geometry.domain(), ratio, {}};
        for (int axis = 0; axis < Dim; ++axis)
          kernel.periodic[axis] =
              parent->boundary.topology().is_periodic(Face<Dim>{axis, BoundarySide::lower});
        for_each_cell(destination.fab_global(patch.fine_patch).box(), kernel);
      }
      Kokkos::fence();
    }

    void apply_flux_mismatch(const field_type& parent_phi, const field_type& child_phi,
                             field_type& parent_residual, field_type& parent_scratch) {
      gather_parent(parent_phi);
      for (ScratchPatch& patch : scratch) {
        patch.flux_increment.set_val(Real(0));
        const Box<Dim> footprint = patch.restricted.box();
        const auto parent_view = std::as_const(patch.parent_staging).view();
        const auto fine_view = std::as_const(child_phi).fab_global(patch.fine_patch).view();
        const auto covered = std::as_const(patch.covered_staging).view();
        auto increment = patch.flux_increment.view();
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
            ::pops::elliptic::mg::fac_detail::FluxMismatchTransfer<Dim> transfer{
                parent_view, fine_view, increment, covered, destination, ratio, axis, child_side,
                inverse_spacing_squared, fine_face_weight, Real(1), geometry_shift};
            if (child->coefficient)
              transfer.fine_coefficient =
                  std::as_const(*child->coefficient).fab_global(patch.fine_patch).view();
            if (child->aperture_lower)
              transfer.fine_aperture_lower =
                  std::as_const(*child->aperture_lower).fab_global(patch.fine_patch).view();
            if (child->aperture_upper)
              transfer.fine_aperture_upper =
                  std::as_const(*child->aperture_upper).fab_global(patch.fine_patch).view();
            for_each_cell(destination, transfer);
          }
        }
      }
      Kokkos::fence();
      parent_scratch.set_val(Real(0));
      auto source_view = [this](const transfer_job& job) -> FieldView<const Real, Dim> {
        return std::as_const(scratch_for(job.source_patch).flux_increment).view();
      };
      auto destination_view = [&parent_scratch](const transfer_job& job) -> FieldView<Real, Dim> {
        return parent_scratch.fab_global(job.destination_patch).view();
      };
      flux->execute(source_view, destination_view);
      for (std::size_t local = 0; local < parent_residual.local_size(); ++local) {
        for_each_cell(parent_residual.box(local),
                      fac_detail::AddKernel<Dim>{
                          parent_residual.fab(local).view(),
                          std::as_const(parent_scratch).fab(local).view()});
        for_each_cell(parent_residual.box(local),
                      fac_detail::MaskResidualKernel<Dim>{
                          parent_residual.fab(local).view(),
                          std::as_const(parent->covered).fab(local).view()});
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
      if (current.geometry.domain().empty() || current.boxes.empty())
        throw std::invalid_argument("partitioned FAC level has an empty domain or box array");
      if (!current.distribution.matches_layout(current.boxes))
        throw std::invalid_argument("partitioned FAC level distribution does not match its boxes");
      if (current.distribution.rank_space() != rank_space)
        throw std::invalid_argument("partitioned FAC levels must share one rank space");
      if (rank_space.size() != static_cast<std::size_t>(n_ranks()))
        throw std::invalid_argument("partitioned FAC rank space disagrees with MPI world size");
      if (!rank_space.contains(current.local_rank) ||
          rank_space.linear_rank(current.local_rank) != static_cast<std::size_t>(my_rank()))
        throw std::invalid_argument("partitioned FAC local rank disagrees with MPI world rank");
      if (!current.boxes.is_disjoint_within(current.geometry.domain(), current.layout_budget))
        throw std::invalid_argument("partitioned FAC boxes are not disjoint within the domain");
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
      if (ratio.is_identity())
        throw std::invalid_argument("partitioned FAC transition must refine at least one axis");
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
    (void)reaction;
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

  void fill_ghosts_(std::size_t level_index, field_type& field, bool homogeneous,
                    const field_type* parent_override = nullptr) {
    Level& level = *levels_[level_index];
    if (level_index > 0) {
      Connection& connection = *connections_[level_index - 1];
      const field_type& parent_field =
          parent_override != nullptr
              ? *parent_override
              : (homogeneous || &field == &level.correction ? levels_[level_index - 1]->correction
                                                            : levels_[level_index - 1]->phi);
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

  ::pops::elliptic::mg::WeightedPoissonFields<Dim, MemorySpace> weighted_fields_(
      Level& level) const {
    ::pops::elliptic::mg::WeightedPoissonFields<Dim, MemorySpace> fields;
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
    if (level.halo_exchange)
      level.halo_exchange->execute(*level.coefficient, *lane_);
    else
      fill_boundary(*level.coefficient, level.halo_schedule);
    for (std::size_t local = 0; local < level.coefficient->local_size(); ++local)
      for_each_cell(level.coefficient->fab(local).grown_box(),
                    ::pops::elliptic::mg::fac_detail::ExtrudeScalarValidToGhosts<Dim>{
                        level.coefficient->fab(local).view(), level.coefficient->box(local)});
    Kokkos::fence();
    fill_physical_boundary(*level.coefficient, level.coefficient_boundary);
  }

  static void copy_vector_valid_(const field_type& source, field_type& destination) {
    if (source.layout() != destination.layout() || source.ncomp() != destination.ncomp())
      throw std::invalid_argument("partitioned FAC vector copy requires one exact layout");
    for (std::size_t local = 0; local < source.local_size(); ++local) {
      const auto in = source.fab(local).view();
      const auto out = destination.fab(local).view();
      const int components = source.ncomp();
      for_each_cell(source.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        for (int component = 0; component < components; ++component)
          out(cell, component) = in(cell, component);
      });
    }
    Kokkos::fence();
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
      fill_coefficient_ghosts_(level_index);
      const field_type* effective_rhs = &rhs;
      if (level_index + 1 < levels_.size() && &iterate == &level.phi && &rhs == &level.rhs) {
        ::pops::elliptic::mg::copy_scalar_valid(level.rhs, level.residual);
        fill_ghosts_(level_index + 1, levels_[level_index + 1]->phi, false);
        connections_[level_index]->apply_flux_mismatch(iterate, levels_[level_index + 1]->phi,
                                                       level.residual, level.scratch);
        effective_rhs = &level.residual;
      }
      if (uses_weighted_operator_(level) && &iterate == &level.phi) {
        ::pops::elliptic::mg::damped_jacobi_weighted_update_valid(
            iterate, *effective_rhs, level.geometry, level.scratch, Real(2) / Real(3), reaction_,
            weighted_fields_(level));
      } else {
        for (std::size_t local = 0; local < iterate.local_size(); ++local) {
          fac_detail::JacobiKernel<Dim> kernel{level.scratch.fab(local).view(),
                                               std::as_const(iterate).fab(local).view(),
                                               effective_rhs->fab(local).view(),
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
      }
      ::pops::elliptic::mg::copy_scalar_valid(level.scratch, iterate);
    }
  }

  void compute_level_residual_(std::size_t level_index) {
    Level& level = *levels_[level_index];
    fill_ghosts_(level_index, level.phi, false);
    fill_coefficient_ghosts_(level_index);
    if (uses_weighted_operator_(level)) {
      ::pops::elliptic::mg::weighted_poisson_residual_valid(
          level.phi, level.rhs, level.geometry, level.residual, reaction_,
          weighted_fields_(level));
      return;
    }
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
    for (std::size_t connection = 0; connection < connections_.size(); ++connection)
      connections_[connection]->apply_flux_mismatch(
          levels_[connection]->phi, levels_[connection + 1]->phi, levels_[connection]->residual,
          levels_[connection]->scratch);
  }

  void restrict_residual_tower_() {
    for (std::size_t child = levels_.size(); child-- > 1;)
      connections_[child - 1]->restrict_into(levels_[child]->residual,
                                             levels_[child - 1]->residual);
  }

  void install_coarse_nullspace_(const mesh::Distribution<Dim>& distribution) {
    const PreparedVectorDistribution<Dim> prepared = distribution.replicated()
                                                         ? PreparedVectorDistribution<Dim>::replicated()
                                                         : PreparedVectorDistribution<Dim>::distributed();
    if (fac_detail::singular(levels_.front()->boundary, reaction_)) {
      Real measure = Real(1);
      for (int axis = 0; axis < Dim; ++axis)
        measure *= levels_.front()->geometry.spacing(axis);
      coarse_solver_->install_nullspace(
          constant_mean_zero_nullspace<Dim>(
              "pops.elliptic.amr.partitioned-composite-fac.coarse-nullspace",
              "partitioned-fac-coarse-correction", measure),
          prepared);
      return;
    }
    coarse_solver_->install_nullspace(FieldNullspacePlan<Dim>{}, prepared);
  }

  void build_coarse_solver_(const EllipticBuildRequest<Dim>& coarse_request,
                            bool allow_coarsening = true) {
    coarse_request_ = coarse_request;
    ::pops::elliptic::mg::GeometricMultigridOptions controls;
    controls.relative_tolerance = options_.coarse_rel_tol;
    controls.absolute_tolerance = options_.coarse_abs_tol;
    controls.maximum_cycles = options_.coarse_cycles;
    controls.reaction = reaction_;
    controls.allow_coarsening = allow_coarsening;
    if (coarse_request.boxes.size() > 2)
      return;
    EllipticBuildRequest<Dim> correction_request = coarse_request;
    correction_request.boundary = ::pops::elliptic::mg::detail::boundary_for_geometry(
        coarse_request.boundary, coarse_request.geometry, true);
    coarse_solver_ = std::make_unique<::pops::elliptic::mg::GeometricMG<Dim, MemorySpace>>(
        std::move(correction_request), *lane_, controls);
    install_coarse_nullspace_(coarse_request.distribution);
  }

  void rebuild_coarse_solver_single_level_() {
    if (!coarse_request_)
      throw std::logic_error("partitioned FAC coarse request is not prepared");
    build_coarse_solver_(*coarse_request_, false);
    if (levels_.front()->coefficient)
      coarse_solver_->install_coefficient(*levels_.front()->coefficient);
    try_prepare_fft_coarse_();
  }

  void try_prepare_fft_coarse_() {
    fft_coarse_.reset();
    used_fft_coarse_ = false;
    if (!coarse_request_ || !lane_)
      return;
    const Level& coarse = *levels_.front();
    EllipticBuildRequest<Dim> request = *coarse_request_;
    request.boundary = ::pops::elliptic::mg::detail::boundary_for_geometry(
        request.boundary, request.geometry, true);
    request.rhs_ghosts = {};
    request.phi_ghosts = fac_detail::unit_ghosts<Dim>();
    request.layout_budget =
        ::pops::elliptic::mg::detail::exact_layout_budget(request.boxes);
    fft_coarse_ = ::pops::elliptic::PoissonFftMultiFabAdapter<Dim>::try_make(
        request, *lane_, reaction_, coarse.coefficient.has_value(),
        coarse.inverse_volume.has_value());
  }

  void ensure_nullspace_() {
    if (nullspace_workspace_ || !fac_detail::singular(levels_.front()->boundary, reaction_))
      return;
    FieldNullspacePlan<Dim> plan = constant_mean_zero_nullspace<Dim>(
        "pops.elliptic.amr.partitioned-composite-fac.nullspace", "partitioned-fac-composite",
        Real(1));
    plan.bases[0].masks.clear();
    plan.bases[0].cell_measure.clear();
    std::vector<PreparedVectorDistribution<Dim>> distributions;
    distributions.reserve(levels_.size());
    for (const auto& level : levels_) {
      auto mask = std::make_shared<MultiFab<Dim>>(level->active.layout(), level->active.distribution(),
                                                 level->active.local_rank(), 1, Extent<Dim>{});
      ::pops::elliptic::mg::copy_scalar_valid(level->active, *mask);
      plan.bases[0].masks.emplace_back(std::move(mask));
      Real measure = Real(1);
      for (int axis = 0; axis < Dim; ++axis)
        measure *= level->geometry.spacing(axis);
      plan.bases[0].cell_measure.push_back(measure);
      distributions.push_back(level->phi.distribution().replicated()
                                  ? PreparedVectorDistribution<Dim>::replicated()
                                  : PreparedVectorDistribution<Dim>::distributed());
    }
    std::vector<const MultiFab<Dim>*> rhs_layouts;
    std::vector<MultiFab<Dim>*> candidates;
    rhs_layouts.reserve(levels_.size());
    candidates.reserve(levels_.size());
    for (const auto& level : levels_) {
      rhs_layouts.push_back(&level->rhs);
      candidates.push_back(&level->phi);
    }
    nullspace_workspace_ = std::make_unique<FieldNullspaceWorkspace<Dim>>(
        std::move(plan), rhs_layouts, std::move(distributions), *lane_);
    nullspace_rhs_ = std::move(rhs_layouts);
    nullspace_candidates_ = std::move(candidates);
  }

  void prolong_one_(std::size_t parent) {
    Connection& connection = *connections_.at(parent);
    connection.gather_parent(levels_[parent]->correction);
    levels_[parent + 1]->correction.set_val(Real(0));
    connection.prolong_valid(levels_[parent + 1]->correction);
    add_active_(*levels_[parent + 1], levels_[parent + 1]->correction);
  }

  SolveReport apply_fac_correction_() {
    SolveReport coarse_report;
    if (fft_coarse_) {
      coarse_report = fft_coarse_->apply(levels_.front()->residual, levels_.front()->correction);
      used_fft_coarse_ = true;
      if (coarse_report.solved())
        fill_ghosts_(0, levels_.front()->correction, true);
    } else if (coarse_solver_) {
      coarse_solver_->phi().set_val(Real(0));
      ::pops::elliptic::mg::copy_scalar_valid(levels_.front()->residual, coarse_solver_->rhs());
      coarse_report = coarse_solver_->solve();
      if (coarse_report.solved())
        ::pops::elliptic::mg::copy_scalar_valid(coarse_solver_->phi(),
                                                levels_.front()->correction);
    } else {
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
      coarse_report.mark_solved("partitioned_fac_coarse_correction");
    }
    if (!coarse_report.solved())
      return coarse_report;
    add_active_(*levels_.front(), levels_.front()->correction);
    prolong_correction_tower_();
    return coarse_report;
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
    for (std::size_t parent = 0; parent < connections_.size(); ++parent)
      prolong_one_(parent);
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
    ::pops::elliptic::mg::copy_scalar_valid(source, destination);
  }

  FieldBoundaryExecutionContext<Dim> boundary_context_at_(std::size_t level, int iteration) const {
    if (!boundary_contexts_ || boundary_contexts_->size() != levels_.size())
      throw std::logic_error("partitioned FAC dynamic boundary contexts are absent");
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
    fill_ghosts_(level_index, level.residual_operator_view, false);
    if (boundary_kernel_) {
      auto context = boundary_context_at_(level_index, iteration);
      context.failure->reset();
      for (int face = 0; face < 2 * Dim; ++face)
        boundary_kernel_->prepare_residual_view(face, level.phi, level.residual_operator_view,
                                                level.geometry, context);
      synchronize_boundary_failure_(
          context, "partitioned FAC dynamic boundary residual failed collectively");
    }
  }

  void fill_dynamic_jvp_ghosts_(std::size_t level_index, int iteration) {
    Level& level = *levels_.at(level_index);
    copy_valid_(level.correction, level.direction_operator_view);
    fill_ghosts_(level_index, level.direction_operator_view, true,
                 level_index > 0 ? &levels_[level_index - 1]->correction : nullptr);
    if (boundary_kernel_) {
      auto context = boundary_context_at_(level_index, iteration);
      context.failure->reset();
      for (int face = 0; face < 2 * Dim; ++face)
        boundary_kernel_->prepare_jvp_view(face, level.phi, level.correction,
                                           level.direction_operator_view, level.geometry, context);
      synchronize_boundary_failure_(context,
                                    "partitioned FAC dynamic boundary JVP failed collectively");
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

  void stage_iterate_(const nonlinear_hierarchy_type& iterate) {
    if (iterate.size() != levels_.size())
      throw std::invalid_argument("partitioned FAC nonlinear iterate has the wrong level count");
    for (std::size_t level = 0; level < levels_.size(); ++level)
      copy_valid_(iterate[level], levels_[level]->phi);
    average_solution_down_();
  }

  void stage_direction_(const nonlinear_hierarchy_type& direction) {
    if (direction.size() != levels_.size())
      throw std::invalid_argument("partitioned FAC nonlinear direction has the wrong level count");
    for (std::size_t level = 0; level < levels_.size(); ++level)
      copy_valid_(direction[level], levels_[level]->correction);
    for (std::size_t child = levels_.size(); child-- > 1;)
      connections_[child - 1]->restrict_into(levels_[child]->correction,
                                             levels_[child - 1]->correction);
  }

  void evaluate_dynamic_residual_(const nonlinear_hierarchy_type& iterate,
                                  nonlinear_hierarchy_type& output, int iteration) {
    if (output.size() != levels_.size())
      throw std::invalid_argument("partitioned FAC nonlinear residual has the wrong level count");
    stage_iterate_(iterate);
    for (std::size_t level_index = 0; level_index < levels_.size(); ++level_index) {
      Level& level = *levels_[level_index];
      fill_dynamic_residual_ghosts_(level_index, iteration);
      fill_coefficient_ghosts_(level_index);
      if (uses_weighted_operator_(level)) {
        ::pops::elliptic::mg::weighted_poisson_residual_valid(
            level.residual_operator_view, level.rhs, level.geometry, level.residual, reaction_,
            weighted_fields_(level));
      } else {
        ::pops::elliptic::mg::poisson_residual_valid(level.residual_operator_view, level.rhs,
                                                     level.geometry, level.residual, reaction_);
      }
      if (boundary_kernel_) {
        auto context = boundary_context_at_(level_index, iteration);
        context.failure->reset();
        for (int face = 0; face < 2 * Dim; ++face)
          boundary_kernel_->add_residual(face, level.phi, level.residual, level.geometry, context);
        synchronize_boundary_failure_(
            context, "partitioned FAC dynamic residual closure failed collectively");
      }
      mask_covered_(level, level.residual);
    }
    for (std::size_t level_index = 0; level_index < levels_.size(); ++level_index) {
      Level& level = *levels_[level_index];
      copy_valid_(level.residual, output[level_index]);
      dynamic_const_view_[level_index] = &output[level_index];
    }
    if (nullspace_workspace_)
      nullspace_workspace_->require_compatible(dynamic_const_view_);
  }

  void apply_dynamic_linearized_(const nonlinear_hierarchy_type& iterate,
                                 const nonlinear_hierarchy_type& direction,
                                 nonlinear_hierarchy_type& output, int iteration) {
    if (output.size() != levels_.size())
      throw std::invalid_argument("partitioned FAC nonlinear JVP has the wrong level count");
    stage_iterate_(iterate);
    stage_direction_(direction);
    for (std::size_t level_index = 0; level_index < levels_.size(); ++level_index) {
      Level& level = *levels_[level_index];
      fill_dynamic_jvp_ghosts_(level_index, iteration);
      fill_coefficient_ghosts_(level_index);
      if (uses_weighted_operator_(level)) {
        ::pops::elliptic::mg::apply_weighted_poisson_operator_valid(
            level.direction_operator_view, level.geometry, level.scratch, reaction_,
            weighted_fields_(level));
      } else {
        ::pops::elliptic::mg::apply_poisson_operator_valid(
            level.direction_operator_view, level.geometry, level.scratch, reaction_);
      }
      if (boundary_kernel_) {
        auto context = boundary_context_at_(level_index, iteration);
        context.failure->reset();
        for (int face = 0; face < 2 * Dim; ++face)
          boundary_kernel_->apply_jvp(face, level.phi, level.correction, level.scratch,
                                      level.geometry, context);
        synchronize_boundary_failure_(context,
                                      "partitioned FAC dynamic JVP closure failed collectively");
      }
      mask_covered_(level, level.scratch);
    }
    for (std::size_t level_index = 0; level_index < levels_.size(); ++level_index) {
      Level& level = *levels_[level_index];
      copy_valid_(level.scratch, output[level_index]);
    }
  }

  void apply_dynamic_gauge_(nonlinear_hierarchy_type& values) {
    if (!nullspace_workspace_)
      return;
    if (values.size() != levels_.size())
      throw std::invalid_argument("partitioned FAC nonlinear gauge has the wrong level count");
    for (std::size_t level = 0; level < levels_.size(); ++level)
      dynamic_mutable_view_[level] = &values[level];
    nullspace_workspace_->apply_gauge(dynamic_mutable_view_);
  }

  SolveReport solve_dynamic_() {
    if (boundary_kernel_ && (!boundary_contexts_ || boundary_contexts_->size() != levels_.size()))
      throw std::logic_error("partitioned FAC dynamic boundary has no level-qualified contexts");
    if (boundary_kernel_ && boundary_kernel_->observes_iteration && !newton_workspace_)
      throw std::logic_error(
          "iterate-dependent partitioned FAC boundary requires a prepared Newton authority");
    if (!lane_)
      throw std::logic_error("partitioned FAC dynamic solve has no prepared execution lane");
    auto* workspace = newton_workspace_ ? &*newton_workspace_ : &*linear_boundary_workspace_;
    if (nullspace_workspace_) {
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
    }
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
      if (nullspace_workspace_)
        nullspace_workspace_->apply_gauge(nullspace_candidates_);
      for (std::size_t level = 0; level < levels_.size(); ++level) {
        fill_dynamic_residual_ghosts_(level, report.iters);
        copy_valid_(levels_[level]->residual_operator_view, levels_[level]->phi);
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

  CompositeFacOptions options_{};
  Real reaction_ = Real(0);
  std::optional<ExecutionLane> lane_{};
  std::optional<EllipticBuildRequest<Dim>> coarse_request_{};
  std::unique_ptr<::pops::elliptic::mg::GeometricMG<Dim, MemorySpace>> coarse_solver_{};
  std::unique_ptr<::pops::elliptic::PoissonFftMultiFabAdapter<Dim>> fft_coarse_{};
  bool used_fft_coarse_ = false;
  std::vector<std::unique_ptr<Level>> levels_{};
  std::vector<std::unique_ptr<Connection>> connections_{};
  std::string lane_identity_{};
  std::string exact_contract_{};
  SolveReport last_report_{};
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
};

}  // namespace pops::elliptic::amr
