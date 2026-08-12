/// @file
/// @brief Prepared exact-ranked halo and physical-boundary session for Program scalar stencils.

#pragma once

#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/numerics/spatial/operators/cartesian_operator.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::program {

namespace scalar_boundary_detail {

inline std::size_t checked_product(std::size_t left, std::size_t right, const char* operation) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(operation);
  return left * right;
}

template <int Dim>
HaloScheduleBudget halo_budget(const MultiFab<Dim>& field, const Geometry<Dim>& geometry,
                               const BoundaryTopology<Dim>& topology) {
  const std::size_t boxes = field.layout().size();
  const std::size_t pairs = checked_product(boxes, boxes, "Program scalar halo pair overflow");
  std::size_t images = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    std::size_t count = 1;
    if (topology.is_periodic(Face<Dim>{axis, BoundarySide::lower}) && field.ghosts()[axis] > 0) {
      const std::int64_t length = geometry.domain().length(axis);
      if (length <= 0)
        throw std::invalid_argument("Program scalar halo has an empty periodic axis");
      const std::int64_t wraps = 1 + (field.ghosts()[axis] - 1) / length;
      count = 1 + checked_product(2, static_cast<std::size_t>(wraps),
                                  "Program scalar halo image overflow");
    }
    images = checked_product(images, count, "Program scalar halo image product overflow");
  }
  const std::size_t work = checked_product(pairs, images, "Program scalar halo work overflow");
  const std::size_t jobs =
      checked_product(work, static_cast<std::size_t>(2 * Dim), "Program scalar halo job overflow");
  const std::int64_t signed_cells = geometry.domain().numPts();
  if (signed_cells <= 0)
    throw std::invalid_argument("Program scalar halo requires a non-empty domain");
  const std::size_t elements = checked_product(
      checked_product(jobs, static_cast<std::size_t>(signed_cells),
                      "Program scalar halo element overflow"),
      static_cast<std::size_t>(field.ncomp()), "Program scalar halo component overflow");
  return {{boxes, pairs},
          work,
          jobs,
          images,
          checked_product(boxes, std::size_t{2}, "Program scalar halo peer overflow"),
          elements,
          elements,
          elements};
}

template <int Dim>
std::size_t boundary_entry_budget(const Geometry<Dim>& geometry, const Extent<Dim>& ghosts,
                                  const BoundaryTopology<Dim>& topology) {
  std::size_t regions = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    std::size_t segments = 1;
    if (ghosts[axis] > 0) {
      segments = 3;
      if (topology.is_periodic(Face<Dim>{axis, BoundarySide::lower})) {
        const std::int64_t length = geometry.domain().length(axis);
        if (length <= 0)
          throw std::invalid_argument("Program scalar boundary has an empty periodic axis");
        const std::int64_t wraps = 1 + (ghosts[axis] - 1) / length;
        segments = 1 + checked_product(2, static_cast<std::size_t>(wraps),
                                       "Program scalar boundary segment overflow");
      }
    }
    regions = checked_product(regions, segments, "Program scalar boundary region overflow");
  }
  return regions - 1;
}

template <int Dim>
PreparedPhysicalBoundary<Dim> physical_boundary(const Geometry<Dim>& geometry,
                                                const Extent<Dim>& ghosts,
                                                const BoundaryTopology<Dim>& topology) {
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    spacing[axis] = geometry.spacing(axis);
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> face{axis, side};
      faces[static_cast<std::size_t>(face.ordinal())].kind =
          topology.is_periodic(face) ? PhysicalBoundaryKind::external
                                     : PhysicalBoundaryKind::constant_extrapolation;
    }
  }
  PhysicalBoundaryConditions<Dim> conditions(topology, faces, spacing);
  return prepare_physical_boundary(
      geometry.domain(), ghosts, std::move(conditions),
      BoundaryScheduleBudget{boundary_entry_budget(geometry, ghosts, topology)});
}

}  // namespace scalar_boundary_detail

/// One immutable scalar-field boundary identity prepared outside a Krylov hot loop.
///
/// Periodic faces use the exact ranked halo transport. Non-periodic faces use the sanctioned
/// constant-extrapolation law of Cartesian Program scalar operators. A field family requiring
/// Dirichlet, Neumann, Robin, mapped or metric boundaries must install a different qualified
/// provider instead of being approximated here.
template <int Dim>
class PreparedScalarBoundarySession {
 private:
  struct UninitializedTag {};

 public:
  using field_type = MultiFab<Dim>;

  /// One block-qualified allocation image shared by the generated Uniform boundary evaluator,
  /// its residual/JVP linearization and the facade transaction journal.  The enclosing session
  /// mutex is held for every access, including nested generated calls.
  struct BoundaryScratch {
    explicit BoundaryScratch(const field_type& prototype)
        : characteristic_candidate(prototype.layout(), prototype.distribution(),
                                   prototype.local_rank(), prototype.ncomp(), prototype.ghosts()),
          transaction_state_snapshot(prototype.layout(), prototype.distribution(),
                                     prototype.local_rank(), prototype.ncomp(), prototype.ghosts()),
          transaction_result_snapshot(prototype.layout(), prototype.distribution(),
                                      prototype.local_rank(), prototype.ncomp(),
                                      prototype.ghosts()),
          transaction_candidate(prototype.layout(), prototype.distribution(),
                                prototype.local_rank(), prototype.ncomp(), prototype.ghosts()),
          detached_state(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                         prototype.ncomp(), prototype.ghosts()),
          residual_boundary_state(prototype.layout(), prototype.distribution(),
                                  prototype.local_rank(), prototype.ncomp(), prototype.ghosts()),
          residual_core_state(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                              prototype.ncomp(), prototype.ghosts()),
          residual_boundary_total(prototype.layout(), prototype.distribution(),
                                  prototype.local_rank(), prototype.ncomp(), prototype.ghosts()),
          residual_core_total(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                              prototype.ncomp(), prototype.ghosts()),
          jvp_perturbed(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                        prototype.ncomp(), prototype.ghosts()),
          jvp_base(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                   prototype.ncomp(), prototype.ghosts()),
          jvp_displaced(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                        prototype.ncomp(), prototype.ghosts()),
          generated_candidate(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                              prototype.ncomp(), prototype.ghosts()),
          generated_source(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                           prototype.ncomp(), prototype.ghosts()),
          generated_source_status(prototype.layout(), prototype.distribution(),
                                  prototype.local_rank(), 1, prototype.ghosts()),
          generated_faces(nd::make_face_flux_workspace(prototype)),
          cartesian_operator(prototype) {}

    field_type characteristic_candidate;
    field_type transaction_state_snapshot;
    field_type transaction_result_snapshot;
    field_type transaction_candidate;
    field_type detached_state;
    field_type residual_boundary_state;
    field_type residual_core_state;
    field_type residual_boundary_total;
    field_type residual_core_total;
    field_type jvp_perturbed;
    field_type jvp_base;
    field_type jvp_displaced;
    field_type generated_candidate;
    field_type generated_source;
    field_type generated_source_status;
    std::vector<nd::FaceField<Dim>> generated_faces;
    nd::PreparedCartesianOperatorScratch<Dim> cartesian_operator;
  };

  /// Existing AMR construction seam. Uniform callers use ``prepare`` so shared allocation
  /// failures are authenticated collectively before session initialization begins.
  PreparedScalarBoundarySession(const Geometry<Dim>& geometry,
                                const BoundaryTopology<Dim>& topology, const field_type& prototype,
                                const ExecutionLane& lane, std::uint64_t generation)
      : PreparedScalarBoundarySession(UninitializedTag{}, geometry, topology, lane, generation) {
    initialize_(prototype, false);
  }

  static std::shared_ptr<PreparedScalarBoundarySession> prepare(
      const Geometry<Dim>& geometry, const BoundaryTopology<Dim>& topology,
      const field_type& prototype, const ExecutionLane& lane, std::uint64_t generation) {
    return prepare_impl_(geometry, topology, prototype, lane, generation, false);
  }

  /// Prepare the additional transaction/evaluator scratch required by one installed block.
  static std::shared_ptr<PreparedScalarBoundarySession> prepare_block(
      const Geometry<Dim>& geometry, const BoundaryTopology<Dim>& topology,
      const field_type& prototype, const ExecutionLane& lane, std::uint64_t generation) {
    return prepare_impl_(geometry, topology, prototype, lane, generation, true);
  }

 private:
  static std::shared_ptr<PreparedScalarBoundarySession> prepare_impl_(
      const Geometry<Dim>& geometry, const BoundaryTopology<Dim>& topology,
      const field_type& prototype, const ExecutionLane& lane, std::uint64_t generation,
      bool prepare_block_scratch) {
    std::shared_ptr<PreparedScalarBoundarySession> session;
    std::exception_ptr local_error;
    try {
      session = std::shared_ptr<PreparedScalarBoundarySession>(new PreparedScalarBoundarySession(
          UninitializedTag{}, geometry, topology, lane, generation));
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane.communicator()) != 0) {
      if (lane.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("Program scalar boundary allocation failed collectively");
    }
    session->initialize_(prototype, prepare_block_scratch);
    return session;
  }

 public:
  PreparedScalarBoundarySession(const PreparedScalarBoundarySession&) = delete;
  PreparedScalarBoundarySession& operator=(const PreparedScalarBoundarySession&) = delete;
  PreparedScalarBoundarySession(PreparedScalarBoundarySession&&) = delete;
  PreparedScalarBoundarySession& operator=(PreparedScalarBoundarySession&&) = delete;

  const Geometry<Dim>& geometry() const noexcept { return geometry_; }
  const BoundaryTopology<Dim>& topology() const noexcept { return topology_; }
  std::uint64_t generation() const noexcept { return generation_; }

  void fill_halo(field_type& field) const {
    if (exchange_)
      exchange_->execute(field, *lane_);
    else
      pops::fill_boundary(field, *schedule_);
  }

  void fill(field_type& field) const {
    fill_halo(field);
    fill_physical_boundary(field, *physical_);
  }

  /// Execute one allocation-free characteristic fill against the candidate prepared from this
  /// session's exact prototype. The lock spans immediate consumption so no caller can retain or
  /// re-enter the mutable scratch while it contains another evaluation.
  template <class Operation>
  void with_characteristic_candidate(field_type& field, Operation&& operation) const {
    static_assert(std::is_void_v<std::invoke_result_t<Operation, field_type&>>,
                  "prepared characteristic scratch callbacks must consume the view immediately");
    with_boundary_scratch(field, [&](BoundaryScratch& scratch) {
      std::forward<Operation>(operation)(scratch.characteristic_candidate);
    });
  }

  /// Execute one immediate, allocation-free block operation while retaining the session lock.
  /// Returning a value/reference is rejected so prepared mutable storage cannot escape the guard.
  template <class Operation>
  void with_boundary_scratch(const field_type& field, Operation&& operation) const {
    static_assert(std::is_void_v<std::invoke_result_t<Operation, BoundaryScratch&>>,
                  "prepared boundary scratch callbacks must consume the view immediately");
    std::lock_guard lock(boundary_mutex_);
    require_boundary_scratch_(field);
    std::forward<Operation>(operation)(*boundary_scratch_);
  }

  [[nodiscard]] const ExecutionLane& lane() const noexcept { return *lane_; }

 private:
  PreparedScalarBoundarySession(UninitializedTag, const Geometry<Dim>& geometry,
                                const BoundaryTopology<Dim>& topology, const ExecutionLane& lane,
                                std::uint64_t generation)
      : geometry_(geometry),
        topology_(topology),
        lane_(&lane),
        lane_borrow_(lane.borrow_immutably()),
        generation_(generation) {}

  void initialize_(const field_type& prototype, bool prepare_block_scratch) {
    std::exception_ptr local_error;
    try {
      if (generation_ == 0)
        throw std::invalid_argument("Program scalar boundary generation must be non-zero");
      if (lane_->size() != static_cast<int>(prototype.rank_space().size()) ||
          lane_->rank() !=
              static_cast<int>(prototype.rank_space().linear_rank(prototype.local_rank())))
        throw std::invalid_argument(
            "Program scalar boundary lane differs from the prototype rank space");
      schedule_.emplace(prepare_halo_schedule(
          prototype, geometry_.domain(), topology_,
          scalar_boundary_detail::halo_budget(prototype, geometry_, topology_)));
      physical_.emplace(
          scalar_boundary_detail::physical_boundary(geometry_, prototype.ghosts(), topology_));
      if (prepare_block_scratch)
        boundary_scratch_.emplace(prototype);
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane_->communicator()) != 0) {
      if (lane_->size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("Program scalar boundary preparation failed collectively");
    }
    const bool distributed =
        all_reduce_max(schedule_->has_remote_jobs() ? 1L : 0L, lane_->communicator()) != 0;
    if (distributed) {
      local_error = nullptr;
      try {
        HaloExchangeContext context{};
        context.context_generation = generation_;
        context.schedule_generation = generation_;
        // Optional storage is allocated with the local session candidate.  Every rank therefore
        // enters HaloExchange's collective constructor before any rank can fail a wrapper heap
        // allocation.
        exchange_.emplace(*schedule_, *lane_, context);
      } catch (...) {
        local_error = std::current_exception();
      }
      if (all_reduce_max(local_error ? 1L : 0L, lane_->communicator()) != 0) {
        if (lane_->size() == 1 && local_error)
          std::rethrow_exception(local_error);
        throw std::runtime_error("Program scalar boundary exchange allocation failed collectively");
      }
    }
  }

  void require_boundary_scratch_(const field_type& field) const {
    if (!boundary_scratch_ || generation_ == 0)
      throw std::logic_error(
          "Program scalar boundary session has no prepared block scratch authority");
    const field_type& prepared = boundary_scratch_->transaction_state_snapshot;
    if (prepared.layout() != field.layout() || prepared.distribution() != field.distribution() ||
        prepared.local_rank() != field.local_rank() ||
        prepared.local_size() != field.local_size() || prepared.ncomp() != field.ncomp() ||
        prepared.ghosts() != field.ghosts())
      throw std::invalid_argument(
          "Program scalar boundary scratch differs from its authenticated block layout");
    boundary_scratch_->cartesian_operator.require_layout(field);
  }
  Geometry<Dim> geometry_;
  BoundaryTopology<Dim> topology_;
  std::optional<HaloSchedule<Dim>> schedule_;
  std::optional<PreparedPhysicalBoundary<Dim>> physical_;
  const ExecutionLane* lane_ = nullptr;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  std::uint64_t generation_ = 0;
  mutable std::optional<HaloExchange<Dim>> exchange_;
  mutable std::recursive_mutex boundary_mutex_;
  mutable std::optional<BoundaryScratch> boundary_scratch_;
};

}  // namespace pops::runtime::program
