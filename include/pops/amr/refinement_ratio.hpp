/// @file
/// @brief One validated compile-time-dimensional AMR refinement-ratio authority.

#pragma once

#include <pops/core/foundation/types.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace pops::amr {

/// A positive, immutable-by-interface refinement ratio for dimensions 1, 2, and 3.
///
/// The identity ratio is a valid hierarchy property (and is required at level zero).  Operations
/// that require a true coarse/fine transition must reject it at their own preparation boundary.
template <int Dim>
class RefinementRatio {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "ND refinement ratios only support dimensions 1, 2, and 3");

  POPS_HD constexpr RefinementRatio() {
    for (int axis = 0; axis < Dim; ++axis)
      values_[axis] = 1;
  }

  template <class... Ratios,
            std::enable_if_t<sizeof...(Ratios) == Dim &&
                                 (std::is_integral_v<std::decay_t<Ratios>> && ...) &&
                                 (!std::is_same_v<std::decay_t<Ratios>, bool> && ...),
                             int> = 0>
  explicit RefinementRatio(Ratios... ratios) {
    const std::array<std::int64_t, Dim> requested{checked_component(ratios)...};
    initialize(requested);
  }

  explicit RefinementRatio(const std::array<int, Dim>& ratios) {
    std::array<std::int64_t, Dim> requested{};
    for (int axis = 0; axis < Dim; ++axis)
      requested[static_cast<std::size_t>(axis)] = ratios[static_cast<std::size_t>(axis)];
    initialize(requested);
  }

  POPS_HD constexpr int operator[](int axis) const { return values_[axis]; }
  POPS_HD constexpr std::int64_t child_count() const { return child_count_; }

  POPS_HD constexpr bool refines_any_axis() const {
    for (int axis = 0; axis < Dim; ++axis)
      if (values_[axis] > 1)
        return true;
    return false;
  }

  POPS_HD constexpr bool is_identity() const { return !refines_any_axis(); }

  POPS_HD constexpr bool operator==(const RefinementRatio& other) const {
    for (int axis = 0; axis < Dim; ++axis)
      if (values_[axis] != other.values_[axis])
        return false;
    return true;
  }

 private:
  template <class T>
  static std::int64_t checked_component(T value) {
    if (std::cmp_less(value, 1) || std::cmp_greater(value, std::numeric_limits<int>::max()))
      throw std::invalid_argument(
          "ND refinement ratio components must lie in the positive signed-index range");
    return static_cast<std::int64_t>(value);
  }

  void initialize(const std::array<std::int64_t, Dim>& requested) {
    std::int64_t children = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t value = requested[static_cast<std::size_t>(axis)];
      if (value < 1 || value > std::numeric_limits<int>::max())
        throw std::invalid_argument(
            "ND refinement ratio components must lie in the positive signed-index range");
      if (children > std::numeric_limits<std::int64_t>::max() / value)
        throw std::overflow_error("ND refinement ratio child count exceeds int64_t");
      values_[axis] = static_cast<int>(value);
      children *= value;
    }
    child_count_ = children;
  }

  int values_[Dim]{};
  std::int64_t child_count_ = 1;
};

}  // namespace pops::amr
