/// @file
/// @brief Checked signed affine maps of a Cartesian index lattice.

#pragma once

#include <pops/mesh/index/box.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pops::mesh {
namespace affine_detail {

inline int checked_index(std::int64_t value, const char* operation) {
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    throw std::overflow_error(operation);
  return static_cast<int>(value);
}

inline std::int64_t checked_add(std::int64_t left, std::int64_t right, const char* operation) {
  if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
      (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right))
    throw std::overflow_error(operation);
  return left + right;
}

inline std::int64_t checked_negate(std::int64_t value, const char* operation) {
  if (value == std::numeric_limits<std::int64_t>::min())
    throw std::overflow_error(operation);
  return -value;
}

}  // namespace affine_detail

/// A signed source-axis -> target-axis permutation.
template <int Dim>
class SignedPermutation {
  static_assert(Dim >= 1 && Dim <= 3, "SignedPermutation only supports dimensions 1, 2, and 3");

 public:
  SignedPermutation() {
    for (int axis = 0; axis < Dim; ++axis) {
      target_axis_[axis] = axis;
      sign_[axis] = 1;
    }
  }

  SignedPermutation(std::array<int, Dim> target_axis, std::array<int, Dim> sign)
      : target_axis_(target_axis), sign_(sign) {
    validate();
  }

  const std::array<int, Dim>& target_axes() const noexcept { return target_axis_; }
  const std::array<int, Dim>& signs() const noexcept { return sign_; }

  bool is_identity() const noexcept {
    for (int axis = 0; axis < Dim; ++axis)
      if (target_axis_[axis] != axis || sign_[axis] != 1)
        return false;
    return true;
  }

  SignedPermutation inverse() const {
    std::array<int, Dim> inverse_axis{};
    std::array<int, Dim> inverse_sign{};
    for (int source = 0; source < Dim; ++source) {
      const int target = target_axis_[source];
      inverse_axis[target] = source;
      inverse_sign[target] = sign_[source];
    }
    return SignedPermutation{inverse_axis, inverse_sign};
  }

  /// Returns @p after composed after this map: ``after(this(source))``.
  SignedPermutation compose(const SignedPermutation& after) const {
    std::array<int, Dim> composed_axis{};
    std::array<int, Dim> composed_sign{};
    for (int source = 0; source < Dim; ++source) {
      const int intermediate = target_axis_[source];
      composed_axis[source] = after.target_axis_[intermediate];
      composed_sign[source] = sign_[source] * after.sign_[intermediate];
    }
    return SignedPermutation{composed_axis, composed_sign};
  }

  bool operator==(const SignedPermutation&) const = default;

 private:
  void validate() const {
    std::array<bool, Dim> seen{};
    for (int source = 0; source < Dim; ++source) {
      const int target = target_axis_[source];
      if (target < 0 || target >= Dim || seen[target])
        throw std::invalid_argument("SignedPermutation must be a bijection");
      if (sign_[source] != -1 && sign_[source] != 1)
        throw std::invalid_argument("SignedPermutation signs must be -1 or +1");
      seen[target] = true;
    }
  }

  std::array<int, Dim> target_axis_{};
  std::array<int, Dim> sign_{};
};

/// Checked affine source-index -> target-index map.  Offset components are indexed by target axis.
template <int Dim>
class AffineIndexTransform {
 public:
  AffineIndexTransform() = default;
  AffineIndexTransform(SignedPermutation<Dim> source_to_target,
                       std::array<std::int64_t, Dim> target_offset)
      : source_to_target_(std::move(source_to_target)), target_offset_(target_offset) {}

  const SignedPermutation<Dim>& signed_permutation() const noexcept { return source_to_target_; }
  const std::array<std::int64_t, Dim>& target_offsets() const noexcept { return target_offset_; }

  Index<Dim> apply(const Index<Dim>& source) const {
    Index<Dim> result;
    for (int source_axis = 0; source_axis < Dim; ++source_axis) {
      const int target_axis = source_to_target_.target_axes()[source_axis];
      const std::int64_t signed_source =
          static_cast<std::int64_t>(source_to_target_.signs()[source_axis]) * source[source_axis];
      result[target_axis] = affine_detail::checked_index(
          affine_detail::checked_add(signed_source, target_offset_[target_axis],
                                     "Cartesian affine index transform overflow"),
          "Cartesian affine index transform exceeds native index range");
    }
    return result;
  }

  Box<Dim> apply(const Box<Dim>& source) const {
    if (source.empty())
      return source;
    Box<Dim> result;
    for (int source_axis = 0; source_axis < Dim; ++source_axis) {
      const int target_axis = source_to_target_.target_axes()[source_axis];
      const std::int64_t first = affine_detail::checked_add(
          static_cast<std::int64_t>(source_to_target_.signs()[source_axis]) *
              source.lo[source_axis],
          target_offset_[target_axis], "Cartesian affine box transform overflow");
      const std::int64_t second = affine_detail::checked_add(
          static_cast<std::int64_t>(source_to_target_.signs()[source_axis]) *
              source.hi[source_axis],
          target_offset_[target_axis], "Cartesian affine box transform overflow");
      result.lo[target_axis] = affine_detail::checked_index(
          std::min(first, second), "Cartesian affine box transform exceeds native index range");
      result.hi[target_axis] = affine_detail::checked_index(
          std::max(first, second), "Cartesian affine box transform exceeds native index range");
    }
    return result;
  }

  AffineIndexTransform inverse() const {
    const SignedPermutation<Dim> inverse_permutation = source_to_target_.inverse();
    std::array<std::int64_t, Dim> inverse_offset{};
    for (int source_axis = 0; source_axis < Dim; ++source_axis) {
      const int target_axis = source_to_target_.target_axes()[source_axis];
      inverse_offset[source_axis] =
          source_to_target_.signs()[source_axis] == 1
              ? affine_detail::checked_negate(target_offset_[target_axis],
                                              "Cartesian affine inverse overflow")
              : target_offset_[target_axis];
    }
    return AffineIndexTransform{inverse_permutation, inverse_offset};
  }

  /// Apply this map and then @p after.
  AffineIndexTransform compose(const AffineIndexTransform& after) const {
    std::array<std::int64_t, Dim> offsets{};
    for (int axis = 0; axis < Dim; ++axis) {
      const int target = after.source_to_target_.target_axes()[axis];
      const auto signed_offset =
          after.source_to_target_.signs()[axis] == 1
              ? target_offset_[axis]
              : affine_detail::checked_negate(target_offset_[axis], "Cartesian affine composition");
      offsets[target] = affine_detail::checked_add(signed_offset, after.target_offset_[target],
                                                   "Cartesian affine composition");
    }
    return {source_to_target_.compose(after.source_to_target_), offsets};
  }

  bool operator==(const AffineIndexTransform&) const = default;

 private:
  SignedPermutation<Dim> source_to_target_;
  std::array<std::int64_t, Dim> target_offset_{};
};

}  // namespace pops::mesh
