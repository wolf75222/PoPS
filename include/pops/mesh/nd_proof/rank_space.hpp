/// @file
/// @brief Private compile-time-ranked process-coordinate layout proof.
///
/// Non-installed proof scaffolding.  It is promoted or deleted in the one-shot ND cutover.

#pragma once

#include <pops/mesh/index/extent.hpp>
#include <pops/mesh/index/index.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pops::mesh::nd_proof {

/// Half-open rank-coordinate box with axis 0 contiguous linearization.
template <int Dim>
class RankSpace {
  static_assert(Dim >= 1 && Dim <= 3, "nd_proof::RankSpace only supports dimensions 1, 2, and 3");

 public:
  RankSpace(Index<Dim> origin, Extent<Dim> extent) : origin_(origin), extent_(extent) {
    size_ = checked_size();
  }

  constexpr const Index<Dim>& origin() const noexcept { return origin_; }
  constexpr const Extent<Dim>& extent() const noexcept { return extent_; }
  constexpr std::size_t size() const noexcept { return size_; }
  constexpr bool empty() const noexcept { return size_ == 0; }

  bool contains(const Index<Dim>& coordinate) const noexcept {
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t offset = static_cast<std::int64_t>(coordinate[axis]) - origin_[axis];
      if (offset < 0 || offset >= extent_[axis])
        return false;
    }
    return !empty();
  }

  std::size_t linear_rank(const Index<Dim>& coordinate) const {
    if (!contains(coordinate))
      throw std::out_of_range("nd_proof::RankSpace coordinate is outside the rank space");
    std::size_t rank = 0;
    std::size_t stride = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      const std::size_t offset =
          static_cast<std::size_t>(static_cast<std::int64_t>(coordinate[axis]) - origin_[axis]);
      rank += offset * stride;
      stride *= static_cast<std::size_t>(extent_[axis]);
    }
    return rank;
  }

  Index<Dim> coord_from_linear(std::size_t rank) const {
    if (rank >= size_)
      throw std::out_of_range("nd_proof::RankSpace rank is outside the rank space");
    Index<Dim> coordinate{};
    for (int axis = 0; axis < Dim; ++axis) {
      const std::size_t axis_extent = static_cast<std::size_t>(extent_[axis]);
      const std::size_t offset = rank % axis_extent;
      rank /= axis_extent;
      coordinate[axis] = static_cast<int>(static_cast<std::int64_t>(origin_[axis]) + offset);
    }
    return coordinate;
  }

 private:
  std::size_t checked_size() const {
    bool has_empty_axis = false;
    for (int axis = 0; axis < Dim; ++axis) {
      if (extent_[axis] < 0)
        throw std::invalid_argument("nd_proof::RankSpace extents must be non-negative");
      if (extent_[axis] == 0) {
        has_empty_axis = true;
        continue;
      }
      const std::int64_t available =
          static_cast<std::int64_t>(std::numeric_limits<int>::max()) - origin_[axis];
      if (extent_[axis] - 1 > available)
        throw std::overflow_error("nd_proof::RankSpace coordinate extent exceeds signed indices");
    }
    if (has_empty_axis)
      return 0;

    std::size_t result = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      const std::uint64_t axis_extent = static_cast<std::uint64_t>(extent_[axis]);
      if (axis_extent > std::numeric_limits<std::size_t>::max() ||
          result > std::numeric_limits<std::size_t>::max() / axis_extent)
        throw std::overflow_error("nd_proof::RankSpace size exceeds size_t");
      result *= static_cast<std::size_t>(axis_extent);
    }
    return result;
  }

  Index<Dim> origin_;
  Extent<Dim> extent_;
  std::size_t size_ = 0;
};

}  // namespace pops::mesh::nd_proof
