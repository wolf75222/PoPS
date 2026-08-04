/// @file
/// @brief One compile-time-ranked spatial authority shared by native runtime configurations.

#pragma once

#include <pops/mesh/index/box.hpp>
#include <pops/mesh/index/extent.hpp>
#include <pops/mesh/index/real_vector.hpp>
#include <pops/mesh/layout/box_array.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pops {

namespace runtime_config_detail {

template <int Dim>
constexpr Extent<Dim> filled_extent(std::int64_t value) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
constexpr RealVector<Dim> filled_real_vector(double value) {
  RealVector<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
constexpr std::array<bool, Dim> filled_periodicity(bool value) {
  std::array<bool, Dim> result{};
  result.fill(value);
  return result;
}

template <int Dim>
constexpr std::string_view cartesian_coordinate_system() {
  static_assert(Dim >= 1 && Dim <= 3,
                "Cartesian coordinate identity only supports dimensions 1, 2, and 3");
  constexpr std::array<std::string_view, 3> identities{
      "pops://coordinates/cartesian-1d@1",
      "pops://coordinates/cartesian-2d@1",
      "pops://coordinates/cartesian-3d@1",
  };
  return identities[static_cast<std::size_t>(Dim - 1)];
}

}  // namespace runtime_config_detail

/// Exact domain facts carried unchanged from resolved Python geometry into one native artifact.
/// Every array has the immutable compile-time rank. Boxes use inclusive native bounds and are
/// ordered exactly as the resolved decomposition; an empty list denotes the single full domain.
template <int Dim>
struct RuntimeSpatialDomain {
  static_assert(Dim >= 1 && Dim <= 3, "RuntimeSpatialDomain only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  Extent<Dim> shape = runtime_config_detail::filled_extent<Dim>(64);
  RealVector<Dim> lower = runtime_config_detail::filled_real_vector<Dim>(0.0);
  RealVector<Dim> upper = runtime_config_detail::filled_real_vector<Dim>(1.0);
  std::array<bool, Dim> periodicity = runtime_config_detail::filled_periodicity<Dim>(true);
  std::vector<Box<Dim>> boxes{};
  std::string coordinate_system =
      std::string(runtime_config_detail::cartesian_coordinate_system<Dim>());

  Box<Dim> index_domain() const { return Box<Dim>::from_extents(shape); }

  std::vector<Box<Dim>> materialized_boxes() const {
    if (!boxes.empty())
      return boxes;
    return {index_domain()};
  }

  void validate_spatial_domain() const {
    const Box<Dim> domain = index_domain();
    for (int axis = 0; axis < Dim; ++axis) {
      if (shape[axis] < 1)
        throw std::invalid_argument("native runtime shape must be strictly positive on every axis");
      if (!std::isfinite(lower[axis]) || !std::isfinite(upper[axis]) ||
          !(upper[axis] > lower[axis]))
        throw std::invalid_argument(
            "native runtime physical bounds must be finite and strictly increasing");
    }
    if (coordinate_system.empty())
      throw std::invalid_argument("native runtime coordinate-system identity must be non-empty");
    const std::vector<Box<Dim>> materialized = materialized_boxes();
    const std::size_t box_count = materialized.size();
    if (box_count > 1 && box_count - 1 > std::numeric_limits<std::size_t>::max() / box_count)
      throw std::length_error("native runtime decomposition exceeds its validation budget");
    const mesh::BoxArrayValidationBudget budget{
        box_count, box_count < 2 ? 0 : box_count * (box_count - 1) / 2};
    const mesh::BoxArray<Dim> layout{materialized};
    if (!layout.tiles_exactly(domain, budget))
      throw std::invalid_argument("native runtime decomposition must tile its exact ranked domain");
  }
};

}  // namespace pops
