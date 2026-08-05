/// @file
/// @brief Exact-ranked output contract for one solved named elliptic field.

#pragma once

#include <pops/core/foundation/types.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace pops::runtime::field {

/// Immutable publication layout selected after Python has resolved the native dimension.
///
/// The first component always receives the potential. A field either publishes no gradient or
/// exactly one gradient component per spatial axis. Keeping the fixed carrier ranked prevents a
/// native Dim=1/3 package from silently entering the historical ``phi/gx/gy`` ABI.
template <int Dim>
class NamedFieldOutput {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "NamedFieldOutput only supports dimensions 1, 2, and 3");

  static constexpr std::size_t potential_and_gradient_count = static_cast<std::size_t>(Dim + 1);

  NamedFieldOutput(const std::vector<int>& output_components, int gradient_sign)
      : has_gradients_(output_components.size() == potential_and_gradient_count),
        gradient_sign_(gradient_sign) {
    if (output_components.size() != 1 && !has_gradients_)
      throw std::invalid_argument(
          "named elliptic field outputs must contain one potential component or exactly " +
          std::to_string(potential_and_gradient_count) +
          " potential/gradient components for the selected dimension");
    components_.fill(-1);
    for (std::size_t slot = 0; slot < output_components.size(); ++slot)
      components_[slot] = output_components[slot];
    validate_active_components_();
  }

  NamedFieldOutput(const std::array<int, Dim + 1>& output_components, int gradient_sign)
      : components_(output_components), has_gradients_(true), gradient_sign_(gradient_sign) {
    validate_active_components_();
  }

  POPS_HD const std::array<int, Dim + 1>& components() const noexcept { return components_; }
  POPS_HD int potential_component() const noexcept { return components_[0]; }
  POPS_HD int gradient_component(int axis) const noexcept { return components_[axis + 1]; }
  POPS_HD bool has_gradients() const noexcept { return has_gradients_; }
  POPS_HD int gradient_sign() const noexcept { return gradient_sign_; }
  POPS_HD std::size_t component_count() const noexcept {
    return has_gradients_ ? potential_and_gradient_count : std::size_t{1};
  }
  constexpr bool operator==(const NamedFieldOutput&) const = default;

  void validate_width(int width, std::string_view owner) const {
    if (width < 1)
      throw std::invalid_argument(std::string(owner) + " has no auxiliary output components");
    for (std::size_t slot = 0; slot < component_count(); ++slot)
      if (components_[slot] >= width)
        throw std::out_of_range(std::string(owner) +
                                " named elliptic field output exceeds auxiliary width");
  }

 private:
  void validate_active_components_() const {
    if (gradient_sign_ != -1 && gradient_sign_ != 1)
      throw std::invalid_argument("named elliptic field gradient sign must be exactly -1 or 1");
    if (!has_gradients_ && gradient_sign_ != 1)
      throw std::invalid_argument(
          "a named elliptic field without gradient outputs must use gradient sign +1");
    for (std::size_t slot = 0; slot < component_count(); ++slot) {
      if (components_[slot] < 0)
        throw std::invalid_argument("named elliptic field output components must be non-negative");
      for (std::size_t previous = 0; previous < slot; ++previous)
        if (components_[previous] == components_[slot])
          throw std::invalid_argument(
              "named elliptic field output components must be pairwise distinct");
    }
  }

  std::array<int, Dim + 1> components_{};
  bool has_gradients_ = false;
  int gradient_sign_ = 1;
};

static_assert(std::is_trivially_copyable_v<NamedFieldOutput<1>>);
static_assert(std::is_trivially_copyable_v<NamedFieldOutput<2>>);
static_assert(std::is_trivially_copyable_v<NamedFieldOutput<3>>);

}  // namespace pops::runtime::field
