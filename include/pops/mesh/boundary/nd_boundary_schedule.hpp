/// @file
/// @brief Backend-neutral 1D/2D/3D Cartesian boundary-region schedule.

#pragma once

#include <pops/mesh/index/box.hpp>
#include <pops/mesh/topology/boundary_topology.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {

enum class BoundaryRegionKind : unsigned char { face, edge, corner };

/// A canonical non-empty intersection of oriented boundary faces.  At most one side per axis may
/// participate.  Faces are stored in increasing axis/ordinal order, independent of authoring order.
template <int Dim>
struct BoundaryRegion {
  std::array<Face<Dim>, Dim> faces{};
  unsigned char count = 0;

  BoundaryRegion() = default;

  BoundaryRegion(std::array<Face<Dim>, Dim> region_faces, std::size_t region_count)
      : faces(region_faces), count(static_cast<unsigned char>(region_count)) {
    if (region_count == 0 || region_count > static_cast<std::size_t>(Dim))
      throw std::invalid_argument("pops::BoundaryRegion requires between one and Dim faces");
    std::sort(faces.begin(), faces.begin() + static_cast<std::ptrdiff_t>(region_count),
              face_less<Dim>);
    for (std::size_t index = 1; index < region_count; ++index) {
      if (faces[index - 1].axis == faces[index].axis)
        throw std::invalid_argument("pops::BoundaryRegion cannot contain two sides of one axis");
    }
  }

  std::size_t codimension() const noexcept { return count; }

  BoundaryRegionKind kind() const {
    if (count == 0)
      throw std::logic_error("pops::BoundaryRegion empty value has no boundary kind");
    if (count == 1)
      return BoundaryRegionKind::face;
    if constexpr (Dim == 3) {
      if (count == 2)
        return BoundaryRegionKind::edge;
    }
    return BoundaryRegionKind::corner;
  }

  /// Base-three identity, axis 0 fastest: interior=0, lower=1, upper=2.
  std::size_t ordinal() const noexcept {
    std::size_t result = 0;
    std::size_t stride = 1;
    std::size_t face_index = 0;
    for (int axis = 0; axis < Dim; ++axis) {
      if (face_index < count && faces[face_index].axis == axis) {
        result += stride *
                  (faces[face_index].side == BoundarySide::lower ? std::size_t{1} : std::size_t{2});
        ++face_index;
      }
      stride *= 3;
    }
    return result;
  }

  bool operator==(const BoundaryRegion&) const = default;
};

template <int Dim>
struct BoundaryOperation {
  Face<Dim> face{};
  BoundaryFaceKind kind = BoundaryFaceKind::physical;
  /// Translation applied to a destination ghost index to locate its periodic source.  Physical
  /// operations must carry the zero shift.
  Index<Dim> source_from_destination_shift{};

  bool operator==(const BoundaryOperation&) const = default;
};

/// One disjoint destination region and its canonical face/edge/corner composition.  This is a
/// fixed-size, trivially-copyable execution record: a backend may mirror it without interpreting a
/// host pointer, communicator, or callback.
template <int Dim>
struct BoundaryRegionPlan {
  BoundaryRegion<Dim> region{};
  Box<Dim> destination{};
  std::array<BoundaryOperation<Dim>, Dim> operations{};
  unsigned char operation_count = 0;
  Index<Dim> source_from_destination_shift{};

  bool has_physical() const noexcept {
    for (std::size_t index = 0; index < operation_count; ++index)
      if (operations[index].kind == BoundaryFaceKind::physical)
        return true;
    return false;
  }

  bool has_periodic() const noexcept {
    for (std::size_t index = 0; index < operation_count; ++index)
      if (operations[index].kind == BoundaryFaceKind::periodic)
        return true;
    return false;
  }

  bool operator==(const BoundaryRegionPlan&) const = default;
};

struct BoundaryScheduleBudget {
  std::size_t entries;
};

namespace boundary_schedule_detail {

inline int checked_index(std::int64_t value, const char* operation) {
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    throw std::overflow_error(operation);
  return static_cast<int>(value);
}

inline std::int64_t checked_add(std::int64_t left, std::int64_t right, const char* operation) {
  if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
      (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right))
    throw std::overflow_error(operation);
  return left + right;
}

inline std::int64_t checked_multiply(std::int64_t left, std::int64_t right, const char* operation) {
  if (left < 0 || right < 0)
    throw std::invalid_argument(operation);
  if (left != 0 && right > std::numeric_limits<std::int64_t>::max() / left)
    throw std::overflow_error(operation);
  return left * right;
}

inline std::size_t checked_axis_segment_count(std::int64_t ghosts, std::int64_t extent,
                                              bool periodic) {
  if (ghosts < 0)
    throw std::invalid_argument("pops::mesh boundary ghost depths must be non-negative");
  if (ghosts == 0)
    return 1;
  if (!periodic)
    return 3;
  if (extent <= 0)
    throw std::invalid_argument("pops::mesh periodic boundary requires a positive domain extent");
  const std::int64_t wraps = ghosts / extent + (ghosts % extent == 0 ? 0 : 1);
  if (wraps > static_cast<std::int64_t>((std::numeric_limits<std::size_t>::max() - 1) / 2))
    throw std::length_error("pops::mesh boundary axis schedule exceeds size_t");
  return 1 + 2 * static_cast<std::size_t>(wraps);
}

template <int Dim>
bool zero_shift(const Index<Dim>& shift) noexcept {
  for (int axis = 0; axis < Dim; ++axis)
    if (shift[axis] != 0)
      return false;
  return true;
}

template <int Dim>
struct AxisSlice {
  int lower = 0;
  int upper = -1;
  bool boundary = false;
  BoundaryOperation<Dim> operation{};
};

template <int Dim>
std::vector<AxisSlice<Dim>> make_axis_slices(const Box<Dim>& domain, const Extent<Dim>& ghosts,
                                             const BoundaryTopology<Dim>& topology, int axis) {
  std::vector<AxisSlice<Dim>> result;
  const std::int64_t depth = ghosts[axis];
  const std::int64_t extent = domain.length(axis);
  const Face<Dim> lower_face{axis, BoundarySide::lower};
  const Face<Dim> upper_face{axis, BoundarySide::upper};
  const bool periodic = topology.is_periodic(lower_face);
  const std::size_t count = checked_axis_segment_count(depth, extent, periodic);
  result.reserve(count);
  result.push_back(AxisSlice<Dim>{domain.lo[axis], domain.hi[axis], false, {}});
  if (depth == 0)
    return result;

  if (!periodic) {
    result.push_back(AxisSlice<Dim>{
        checked_index(checked_add(domain.lo[axis], -depth,
                                  "pops::mesh lower physical ghost region overflows int64_t"),
                      "pops::mesh lower physical ghost region exceeds native index range"),
        checked_index(static_cast<std::int64_t>(domain.lo[axis]) - 1,
                      "pops::mesh lower physical ghost region exceeds native index range"),
        true, BoundaryOperation<Dim>{lower_face, BoundaryFaceKind::physical, {}}});
    result.push_back(AxisSlice<Dim>{
        checked_index(static_cast<std::int64_t>(domain.hi[axis]) + 1,
                      "pops::mesh upper physical ghost region exceeds native index range"),
        checked_index(checked_add(domain.hi[axis], depth,
                                  "pops::mesh upper physical ghost region overflows int64_t"),
                      "pops::mesh upper physical ghost region exceeds native index range"),
        true, BoundaryOperation<Dim>{upper_face, BoundaryFaceKind::physical, {}}});
    return result;
  }

  const std::int64_t wraps = depth / extent + (depth % extent == 0 ? 0 : 1);
  for (std::int64_t wrap = 1; wrap <= wraps; ++wrap) {
    const std::int64_t previous =
        checked_multiply(wrap - 1, extent, "pops::mesh periodic wrap offset overflow");
    const std::int64_t reached =
        checked_multiply(wrap, extent, "pops::mesh periodic wrap offset overflow");
    const std::int64_t capped = std::min(depth, reached);
    const int shift = checked_index(reached, "pops::mesh periodic shift exceeds Index range");

    Index<Dim> lower_shift{};
    lower_shift[axis] = shift;
    result.push_back(AxisSlice<Dim>{
        checked_index(checked_add(domain.lo[axis], -capped,
                                  "pops::mesh lower periodic ghost region overflows int64_t"),
                      "pops::mesh lower periodic ghost region exceeds native index range"),
        checked_index(
            checked_add(checked_add(domain.lo[axis], -previous,
                                    "pops::mesh lower periodic ghost region overflows int64_t"),
                        -1, "pops::mesh lower periodic ghost region overflows int64_t"),
            "pops::mesh lower periodic ghost region exceeds native index range"),
        true, BoundaryOperation<Dim>{lower_face, BoundaryFaceKind::periodic, lower_shift}});

    Index<Dim> upper_shift{};
    upper_shift[axis] = -shift;
    result.push_back(AxisSlice<Dim>{
        checked_index(
            checked_add(checked_add(domain.hi[axis], previous,
                                    "pops::mesh upper periodic ghost region overflows int64_t"),
                        1, "pops::mesh upper periodic ghost region overflows int64_t"),
            "pops::mesh upper periodic ghost region exceeds native index range"),
        checked_index(checked_add(domain.hi[axis], capped,
                                  "pops::mesh upper periodic ghost region overflows int64_t"),
                      "pops::mesh upper periodic ghost region exceeds native index range"),
        true, BoundaryOperation<Dim>{upper_face, BoundaryFaceKind::periodic, upper_shift}});
  }
  return result;
}

}  // namespace boundary_schedule_detail

/// Canonicalizes the operation order and validates one composed record.  In particular it rejects
/// two sides of one axis, physical shifts, tangential shifts, and periodic zero shifts; callers can
/// therefore compose independently authored face rules without last-writer-wins behavior.
template <int Dim>
BoundaryRegionPlan<Dim> compose_boundary_region_plan(
    const Box<Dim>& destination,
    std::type_identity_t<std::array<BoundaryOperation<Dim>, static_cast<std::size_t>(Dim)>>
        operations,
    std::size_t operation_count) {
  if (destination.empty())
    throw std::invalid_argument("pops::mesh boundary plan destination must be non-empty");
  if (operation_count == 0 || operation_count > static_cast<std::size_t>(Dim))
    throw std::invalid_argument("pops::mesh boundary plan operation count is outside [1, Dim]");
  std::sort(operations.begin(), operations.begin() + static_cast<std::ptrdiff_t>(operation_count),
            [](const BoundaryOperation<Dim>& left, const BoundaryOperation<Dim>& right) {
              return face_less(left.face, right.face);
            });

  std::array<Face<Dim>, Dim> faces{};
  Index<Dim> composed{};
  for (std::size_t index = 0; index < operation_count; ++index) {
    const BoundaryOperation<Dim>& operation = operations[index];
    faces[index] = operation.face;
    if (index != 0 && operations[index - 1].face.axis == operation.face.axis)
      throw std::invalid_argument(
          "pops::mesh boundary composition has conflicting sides on one axis");
    if (operation.kind == BoundaryFaceKind::physical) {
      if (!boundary_schedule_detail::zero_shift(operation.source_from_destination_shift))
        throw std::invalid_argument("pops::mesh physical boundary operation carries a shift");
      continue;
    }
    if (operation.source_from_destination_shift[operation.face.axis] == 0)
      throw std::invalid_argument("pops::mesh periodic boundary operation carries a zero shift");
    for (int shift_axis = 0; shift_axis < Dim; ++shift_axis) {
      const int value = operation.source_from_destination_shift[shift_axis];
      if (shift_axis != operation.face.axis && value != 0)
        throw std::invalid_argument(
            "pops::mesh axis-translation boundary operation carries a tangential shift");
      if (value != 0 && composed[shift_axis] != 0)
        throw std::invalid_argument(
            "pops::mesh boundary composition has conflicting translation contributors");
      if (value != 0)
        composed[shift_axis] = value;
    }
  }

  return BoundaryRegionPlan<Dim>{BoundaryRegion<Dim>{faces, operation_count}, destination,
                                 operations, static_cast<unsigned char>(operation_count), composed};
}

/// Host owner for a backend-neutral array of fixed-size execution records.  It performs no MPI,
/// Kokkos, callback, or field access; those execution layers consume this authenticated plan.
template <int Dim>
class BoundarySchedule {
 public:
  BoundarySchedule(Box<Dim> domain, Extent<Dim> ghosts, BoundaryTopology<Dim> topology,
                   std::vector<BoundaryRegionPlan<Dim>> entries)
      : domain_(domain), ghosts_(ghosts), topology_(topology), entries_(std::move(entries)) {}

  const Box<Dim>& domain() const noexcept { return domain_; }
  const Extent<Dim>& ghosts() const noexcept { return ghosts_; }
  const BoundaryTopology<Dim>& topology() const noexcept { return topology_; }
  const std::vector<BoundaryRegionPlan<Dim>>& entries() const noexcept { return entries_; }
  std::size_t size() const noexcept { return entries_.size(); }

 private:
  Box<Dim> domain_{};
  Extent<Dim> ghosts_{};
  BoundaryTopology<Dim> topology_{};
  std::vector<BoundaryRegionPlan<Dim>> entries_{};
};

/// Enumerates disjoint face/edge/corner regions.  Axis 0 is the fastest Cartesian schedule
/// coordinate; deep periodic ghosts are split into exact wrap-width strips instead of being mapped
/// by one insufficient shift.
template <int Dim>
BoundarySchedule<Dim> prepare_boundary_schedule(const Box<Dim>& domain, const Extent<Dim>& ghosts,
                                                const BoundaryTopology<Dim>& topology,
                                                BoundaryScheduleBudget budget) {
  if (domain.empty())
    throw std::invalid_argument("pops::mesh boundary schedule requires a non-empty domain");

  std::array<std::vector<boundary_schedule_detail::AxisSlice<Dim>>, Dim> axes;
  std::array<std::size_t, Dim> axis_counts{};
  std::size_t cartesian_count = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    const Face<Dim> lower{axis, BoundarySide::lower};
    axis_counts[axis] = boundary_schedule_detail::checked_axis_segment_count(
        ghosts[axis], domain.length(axis), topology.is_periodic(lower));
    if (axis_counts[axis] > std::numeric_limits<std::size_t>::max() / cartesian_count)
      throw std::length_error("pops::mesh boundary schedule Cartesian size overflows size_t");
    cartesian_count *= axis_counts[axis];
  }
  const std::size_t entry_count = cartesian_count - 1;
  if (entry_count > budget.entries)
    throw std::length_error("pops::mesh boundary schedule exceeds its explicit entry budget");

  for (int axis = 0; axis < Dim; ++axis)
    axes[axis] = boundary_schedule_detail::make_axis_slices(domain, ghosts, topology, axis);

  std::vector<BoundaryRegionPlan<Dim>> entries;
  if (entry_count > entries.max_size())
    throw std::length_error("pops::mesh boundary schedule exceeds vector capacity");
  entries.reserve(entry_count);
  for (std::size_t ordinal = 1; ordinal < cartesian_count; ++ordinal) {
    std::size_t quotient = ordinal;
    Box<Dim> destination{};
    std::array<BoundaryOperation<Dim>, Dim> operations{};
    std::size_t operation_count = 0;
    for (int axis = 0; axis < Dim; ++axis) {
      const auto& axis_slices = axes[axis];
      const auto& slice = axis_slices[quotient % axis_slices.size()];
      quotient /= axis_slices.size();
      destination.lo[axis] = slice.lower;
      destination.hi[axis] = slice.upper;
      if (slice.boundary)
        operations[operation_count++] = slice.operation;
    }
    entries.push_back(compose_boundary_region_plan<Dim>(destination, operations, operation_count));
  }
  return BoundarySchedule<Dim>{domain, ghosts, topology, std::move(entries)};
}

static_assert(std::is_trivially_copyable_v<BoundaryRegion<3>>);
static_assert(std::is_trivially_copyable_v<BoundaryOperation<3>>);
static_assert(std::is_trivially_copyable_v<BoundaryRegionPlan<3>>);

}  // namespace pops
