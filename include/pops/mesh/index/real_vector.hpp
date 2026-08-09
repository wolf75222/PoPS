/// @file
/// @brief Compile-time-ranked real Cartesian coordinates.

#pragma once

#include <pops/core/foundation/types.hpp>

#include <limits>
#include <type_traits>

namespace pops {

namespace real_vector_detail {

template <class T, bool IsIntegral = std::is_integral_v<T> && !std::is_same_v<T, bool>,
          bool IsFloating = std::is_floating_point_v<T>>
struct lossless_real_scalar_impl : std::false_type {};

template <class T>
struct lossless_real_scalar_impl<T, true, false>
    : std::bool_constant<std::numeric_limits<T>::digits <= std::numeric_limits<double>::digits> {};

template <class T>
struct lossless_real_scalar_impl<T, false, true>
    : std::bool_constant<
          std::numeric_limits<T>::digits <= std::numeric_limits<double>::digits &&
          std::numeric_limits<T>::max_exponent <= std::numeric_limits<double>::max_exponent &&
          std::numeric_limits<T>::min_exponent >= std::numeric_limits<double>::min_exponent> {};

template <class T>
inline constexpr bool lossless_real_scalar = lossless_real_scalar_impl<std::decay_t<T>>::value;

}  // namespace real_vector_detail

/// Double-precision Cartesian coordinate with a compile-time spatial rank.
template <int Dim>
struct RealVector {
  static_assert(Dim >= 1 && Dim <= 3, "pops::RealVector only supports dimensions 1, 2, and 3");

  static constexpr int rank = Dim;
  double values[Dim]{};

  POPS_HD constexpr RealVector() = default;

  template <class... Coordinates,
            std::enable_if_t<sizeof...(Coordinates) == Dim &&
                                 (real_vector_detail::lossless_real_scalar<Coordinates> && ...),
                             int> = 0>
  POPS_HD constexpr explicit RealVector(Coordinates... coordinates)
      : values{static_cast<double>(coordinates)...} {}

  POPS_HD constexpr double& operator[](int axis) { return values[axis]; }
  POPS_HD constexpr double operator[](int axis) const { return values[axis]; }

  POPS_HD constexpr bool operator==(const RealVector& other) const {
    for (int axis = 0; axis < Dim; ++axis)
      if (values[axis] != other.values[axis])
        return false;
    return true;
  }
};

}  // namespace pops
