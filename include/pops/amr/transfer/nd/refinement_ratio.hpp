/// @file
/// @brief Validated anisotropic refinement ratios for prepared ND transfers.

#pragma once

#include <pops/core/foundation/types.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace pops::amr::transfer::nd {

/// A positive per-axis AMR refinement ratio for compile-time dimensions 1, 2, and 3.
///
/// An axis ratio of one is allowed so a hierarchy can refine only selected axes.  At least one
/// axis must refine, which keeps an inter-level transfer distinct from an identity copy.  The
/// child count is validated once on the host and retained as a fixed-width scalar for allocation-
/// free device kernels.
template <int Dim>
class RefinementRatio {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "ND refinement ratios only support dimensions 1, 2, and 3");

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
    bool refines_an_axis = false;
    std::int64_t children = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t value = requested[static_cast<std::size_t>(axis)];
      if (value < 1 || value > std::numeric_limits<int>::max())
        throw std::invalid_argument(
            "ND refinement ratio components must lie in the positive signed-index range");
      refines_an_axis = refines_an_axis || value > 1;
      if (children > std::numeric_limits<std::int64_t>::max() / value)
        throw std::overflow_error("ND refinement ratio child count exceeds int64_t");
      values_[axis] = static_cast<int>(value);
      children *= value;
    }
    if (!refines_an_axis)
      throw std::invalid_argument("ND refinement ratio must refine at least one spatial axis");
    child_count_ = children;
  }

  int values_[Dim]{};
  std::int64_t child_count_ = 0;
};

static_assert(std::is_trivially_copyable_v<RefinementRatio<1>>);
static_assert(std::is_trivially_copyable_v<RefinementRatio<2>>);
static_assert(std::is_trivially_copyable_v<RefinementRatio<3>>);

}  // namespace pops::amr::transfer::nd
