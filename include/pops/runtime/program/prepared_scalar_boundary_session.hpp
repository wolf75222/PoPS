/// @file
/// @brief Prepared exact-ranked halo and physical-boundary session for Program scalar stencils.

#pragma once

#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

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
 public:
  using field_type = MultiFab<Dim>;

  static std::shared_ptr<PreparedScalarBoundarySession> prepare(
      const Geometry<Dim>& geometry, const BoundaryTopology<Dim>& topology,
      const field_type& prototype, const ExecutionLane& lane, std::uint64_t generation) {
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
    session->initialize_(prototype);
    return session;
  }

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

 private:
  struct UninitializedTag {};

  PreparedScalarBoundarySession(UninitializedTag, const Geometry<Dim>& geometry,
                                const BoundaryTopology<Dim>& topology, const ExecutionLane& lane,
                                std::uint64_t generation)
      : geometry_(geometry),
        topology_(topology),
        lane_(&lane),
        lane_borrow_(lane.borrow_immutably()),
        generation_(generation) {}

  void initialize_(const field_type& prototype) {
    std::exception_ptr local_error;
    try {
      if (generation_ == 0)
        throw std::invalid_argument("Program scalar boundary generation must be non-zero");
      schedule_.emplace(prepare_halo_schedule(
          prototype, geometry_.domain(), topology_,
          scalar_boundary_detail::halo_budget(prototype, geometry_, topology_)));
      physical_.emplace(
          scalar_boundary_detail::physical_boundary(geometry_, prototype.ghosts(), topology_));
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
      HaloExchangeContext context{};
      context.context_generation = generation_;
      context.schedule_generation = generation_;
      exchange_ = std::make_unique<HaloExchange<Dim>>(*schedule_, *lane_, context);
    }
  }
  Geometry<Dim> geometry_;
  BoundaryTopology<Dim> topology_;
  std::optional<HaloSchedule<Dim>> schedule_;
  std::optional<PreparedPhysicalBoundary<Dim>> physical_;
  const ExecutionLane* lane_ = nullptr;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  std::uint64_t generation_ = 0;
  mutable std::unique_ptr<HaloExchange<Dim>> exchange_;
};

}  // namespace pops::runtime::program
