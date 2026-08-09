/// @file
/// @brief Exact-ranked output contract for one solved named elliptic field.

#pragma once

#include <pops/core/foundation/types.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace pops::runtime::field {

/// Immutable local publication layout selected after Python has resolved the native dimension.
///
/// The carrier is owned by the field, so its components are always compact: slot zero is the
/// potential and slots ``axis + 1`` are the optional gradient.  Global provider storage is resolved
/// separately from owner-qualified ComponentKeys; this type deliberately contains neither a raw
/// auxiliary index nor a package-local slot.
template <int Dim>
class NamedFieldOutput {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "NamedFieldOutput only supports dimensions 1, 2, and 3");

  static constexpr std::size_t potential_and_gradient_count = static_cast<std::size_t>(Dim + 1);

  explicit NamedFieldOutput(std::size_t output_count, int gradient_sign)
      : has_gradients_(output_count == potential_and_gradient_count),
        gradient_sign_(gradient_sign) {
    if (output_count != 1 && !has_gradients_)
      throw std::invalid_argument(
          "named elliptic field outputs must contain one potential component or exactly " +
          std::to_string(potential_and_gradient_count) +
          " potential/gradient components for the selected dimension");
    validate_();
  }

  POPS_HD int potential_component() const noexcept { return 0; }
  POPS_HD int gradient_component(int axis) const noexcept { return axis + 1; }
  POPS_HD bool has_gradients() const noexcept { return has_gradients_; }
  POPS_HD int gradient_sign() const noexcept { return gradient_sign_; }
  POPS_HD std::size_t component_count() const noexcept {
    return has_gradients_ ? potential_and_gradient_count : std::size_t{1};
  }
  constexpr bool operator==(const NamedFieldOutput&) const = default;

  void validate_width(int width, std::string_view owner) const {
    if (width != static_cast<int>(component_count()))
      throw std::invalid_argument(std::string(owner) +
                                  " named elliptic field output carrier has the wrong exact width");
  }

 private:
  void validate_() const {
    if (gradient_sign_ != -1 && gradient_sign_ != 1)
      throw std::invalid_argument("named elliptic field gradient sign must be exactly -1 or 1");
    if (!has_gradients_ && gradient_sign_ != 1)
      throw std::invalid_argument(
          "a named elliptic field without gradient outputs must use gradient sign +1");
  }

  bool has_gradients_ = false;
  int gradient_sign_ = 1;
};

static_assert(std::is_trivially_copyable_v<NamedFieldOutput<1>>);
static_assert(std::is_trivially_copyable_v<NamedFieldOutput<2>>);
static_assert(std::is_trivially_copyable_v<NamedFieldOutput<3>>);

}  // namespace pops::runtime::field
