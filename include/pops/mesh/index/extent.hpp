/// @file
/// @brief Compile-time-ranked non-negative box extents.

#pragma once

#include <pops/core/foundation/types.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace pops {

namespace extent_detail {

template <class T, bool IsIntegral = std::is_integral_v<T> && !std::is_same_v<T, bool>>
struct lossless_extent_scalar_impl : std::false_type {};

template <class T>
struct lossless_extent_scalar_impl<T, true>
    : std::bool_constant<
          std::numeric_limits<T>::lowest() >= std::numeric_limits<std::int64_t>::lowest() &&
          std::numeric_limits<T>::max() <= std::numeric_limits<std::int64_t>::max()> {};

template <class T>
inline constexpr bool lossless_extent_scalar = lossless_extent_scalar_impl<std::decay_t<T>>::value;

}  // namespace extent_detail

/// Non-negative extent per spatial axis.  Construction and validation belong to the owning box.
template <int Dim>
struct Extent {
  static_assert(Dim >= 1 && Dim <= 3, "pops::Extent only supports dimensions 1, 2, and 3");

  static constexpr int rank = Dim;
  std::int64_t values[Dim]{};

  POPS_HD constexpr Extent() = default;

  template <class... Sizes,
            std::enable_if_t<sizeof...(Sizes) == Dim &&
                                 (extent_detail::lossless_extent_scalar<Sizes> && ...),
                             int> = 0>
  POPS_HD constexpr explicit Extent(Sizes... sizes) : values{static_cast<std::int64_t>(sizes)...} {}

  POPS_HD constexpr std::int64_t& operator[](int axis) { return values[axis]; }
  POPS_HD constexpr std::int64_t operator[](int axis) const { return values[axis]; }

  POPS_HD constexpr bool operator==(const Extent& other) const {
    for (int axis = 0; axis < Dim; ++axis)
      if (values[axis] != other.values[axis])
        return false;
    return true;
  }
};

}  // namespace pops
