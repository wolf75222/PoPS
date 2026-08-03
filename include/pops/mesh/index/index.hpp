/// @file
/// @brief Compile-time-ranked integer cell coordinates.

#pragma once

#include <pops/core/foundation/types.hpp>

#include <limits>
#include <type_traits>

namespace pops {

namespace index_detail {

template <class T, bool IsIntegral = std::is_integral_v<T> && !std::is_same_v<T, bool>>
struct lossless_index_scalar_impl : std::false_type {};

template <class T>
struct lossless_index_scalar_impl<T, true>
    : std::bool_constant<std::numeric_limits<T>::lowest() >= std::numeric_limits<int>::lowest() &&
                         std::numeric_limits<T>::max() <= std::numeric_limits<int>::max()> {};

template <class T>
inline constexpr bool lossless_index_scalar = lossless_index_scalar_impl<std::decay_t<T>>::value;

}  // namespace index_detail

/// Signed cell coordinate with a compile-time spatial rank.
template <int Dim>
struct Index {
  static_assert(Dim >= 1 && Dim <= 3, "pops::Index only supports dimensions 1, 2, and 3");

  static constexpr int rank = Dim;
  int values[Dim]{};

  POPS_HD constexpr Index() = default;

  template <class... Coordinates,
            std::enable_if_t<sizeof...(Coordinates) == Dim &&
                                 (index_detail::lossless_index_scalar<Coordinates> && ...),
                             int> = 0>
  POPS_HD constexpr explicit Index(Coordinates... coordinates)
      : values{static_cast<int>(coordinates)...} {}

  POPS_HD constexpr int& operator[](int axis) { return values[axis]; }
  POPS_HD constexpr int operator[](int axis) const { return values[axis]; }

  POPS_HD constexpr bool operator==(const Index& other) const {
    for (int axis = 0; axis < Dim; ++axis)
      if (values[axis] != other.values[axis])
        return false;
    return true;
  }
};

}  // namespace pops
