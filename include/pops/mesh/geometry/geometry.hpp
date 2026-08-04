/// @file
/// @brief Compile-time-ranked Cartesian geometry over one exact integer domain.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/index/real_vector.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace pops {

namespace geometry_detail {

constexpr bool finite(Real value) {
  return value == value && value != std::numeric_limits<Real>::infinity() &&
         value != -std::numeric_limits<Real>::infinity();
}

}  // namespace geometry_detail

/// Uniform Cartesian index-to-physical mapping with an immutable compile-time spatial rank.
/// Cell and face coordinates remain defined for ghost indices outside ``domain``.
template <int Dim>
class Geometry {
  static_assert(Dim >= 1 && Dim <= 3, "Geometry only supports dimensions 1, 2, and 3");

 public:
  static constexpr int rank = Dim;

  static Geometry from_bounds(Box<Dim> domain, RealVector<Dim> lower, RealVector<Dim> upper) {
    if (domain.empty())
      throw std::invalid_argument("Geometry requires a non-empty index domain");
    RealVector<Dim> inverse_cells{};
    for (int axis = 0; axis < Dim; ++axis) {
      if (!geometry_detail::finite(lower[axis]) || !geometry_detail::finite(upper[axis]) ||
          !(upper[axis] > lower[axis]))
        throw std::invalid_argument(
            "Geometry physical bounds must be finite and strictly increasing");
      const std::int64_t cells = domain.length(axis);
      if (cells <= 0)
        throw std::invalid_argument("Geometry requires positive cell extents on every axis");
      const Real physical_extent = upper[axis] - lower[axis];
      const Real cell_spacing = physical_extent / static_cast<Real>(cells);
      if (!geometry_detail::finite(physical_extent) || !geometry_detail::finite(cell_spacing) ||
          !(cell_spacing > Real(0)))
        throw std::invalid_argument(
            "Geometry physical extents must produce finite positive cell spacing");
      inverse_cells[axis] = Real(1) / static_cast<Real>(cells);
    }
    return Geometry(domain, lower, upper, inverse_cells);
  }

  POPS_HD const Box<Dim>& domain() const noexcept { return domain_; }
  POPS_HD const RealVector<Dim>& lower() const noexcept { return lower_; }
  POPS_HD const RealVector<Dim>& upper() const noexcept { return upper_; }

  POPS_HD Real spacing(int axis) const {
    return (upper_[axis] - lower_[axis]) * inverse_cells_[axis];
  }

  POPS_HD Real cell_coordinate(int axis, int index) const {
    return lower_[axis] +
           (static_cast<Real>(index) - static_cast<Real>(domain_.lo[axis]) + Real(0.5)) *
               spacing(axis);
  }

  POPS_HD Real face_coordinate(int axis, int index) const {
    return lower_[axis] +
           (static_cast<Real>(index) - static_cast<Real>(domain_.lo[axis])) * spacing(axis);
  }

  POPS_HD RealVector<Dim> cell_center(const Index<Dim>& index) const {
    RealVector<Dim> result{};
    for (int axis = 0; axis < Dim; ++axis)
      result[axis] = cell_coordinate(axis, index[axis]);
    return result;
  }

  POPS_HD RealVector<Dim> lower_face(const Index<Dim>& index) const {
    RealVector<Dim> result{};
    for (int axis = 0; axis < Dim; ++axis)
      result[axis] = face_coordinate(axis, index[axis]);
    return result;
  }

  Geometry refine(const Extent<Dim>& ratio) const {
    Box<Dim> refined{};
    for (int axis = 0; axis < Dim; ++axis) {
      if (ratio[axis] <= 0 || ratio[axis] > std::numeric_limits<int>::max())
        throw std::invalid_argument(
            "Geometry refinement ratios must be positive signed-index values");
      refined.lo[axis] =
          detail::checked_box_index(static_cast<std::int64_t>(domain_.lo[axis]) * ratio[axis],
                                    "Geometry refined lower bound exceeds the signed index range");
      refined.hi[axis] = detail::checked_box_index(
          static_cast<std::int64_t>(domain_.hi[axis]) * ratio[axis] + ratio[axis] - 1,
          "Geometry refined upper bound exceeds the signed index range");
    }
    return from_bounds(refined, lower_, upper_);
  }

  bool operator==(const Geometry&) const = default;

 private:
  POPS_HD constexpr Geometry(Box<Dim> domain, RealVector<Dim> lower, RealVector<Dim> upper,
                             RealVector<Dim> inverse_cells)
      : domain_(domain), lower_(lower), upper_(upper), inverse_cells_(inverse_cells) {}

  Box<Dim> domain_;
  RealVector<Dim> lower_;
  RealVector<Dim> upper_;
  RealVector<Dim> inverse_cells_;
};

static_assert(std::is_trivially_copyable_v<Geometry<1>>);
static_assert(std::is_trivially_copyable_v<Geometry<2>>);
static_assert(std::is_trivially_copyable_v<Geometry<3>>);

}  // namespace pops
