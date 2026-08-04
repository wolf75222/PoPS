/// @file
/// @brief Private compile-time-ranked periodic topology and axis-translation image proof.
///
/// Non-installed proof scaffolding.  Mapped identifications are topology/affine values only;
/// axis-translation images deliberately reject them rather than approximating them as wraps.

#pragma once

#include <pops/mesh/index/box.hpp>
#include <pops/mesh/topology/boundary_topology.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pops::mesh::nd_proof {

namespace periodicity_detail {

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

inline std::int64_t checked_multiple(std::int64_t multiple, std::int64_t extent,
                                     const char* operation) {
  if (extent <= 0)
    throw std::invalid_argument("nd_proof periodic translation requires a positive extent");
  if ((multiple > 0 && multiple > std::numeric_limits<std::int64_t>::max() / extent) ||
      (multiple < 0 && multiple < std::numeric_limits<std::int64_t>::min() / extent))
    throw std::overflow_error(operation);
  return multiple * extent;
}

template <int Dim>
Box<Dim> translate_box(const Box<Dim>& source, const std::array<std::int64_t, Dim>& translation,
                       const char* operation) {
  if (source.empty())
    return source;
  Box<Dim> result;
  for (int axis = 0; axis < Dim; ++axis) {
    result.lo[axis] =
        checked_index(checked_add(source.lo[axis], translation[axis], operation), operation);
    result.hi[axis] =
        checked_index(checked_add(source.hi[axis], translation[axis], operation), operation);
  }
  return result;
}

template <int Dim>
Box<Dim> grow_box(const Box<Dim>& source, const Extent<Dim>& ghosts) {
  if (source.empty())
    return source;
  Box<Dim> result;
  for (int axis = 0; axis < Dim; ++axis) {
    if (ghosts[axis] < 0)
      throw std::invalid_argument("nd_proof destination ghost depths must be non-negative");
    result.lo[axis] = checked_index(
        checked_add(source.lo[axis], checked_negate(ghosts[axis], "nd_proof ghost lower overflow"),
                    "nd_proof ghost lower overflow"),
        "nd_proof ghost lower bound exceeds native index range");
    result.hi[axis] =
        checked_index(checked_add(source.hi[axis], ghosts[axis], "nd_proof ghost upper overflow"),
                      "nd_proof ghost upper bound exceeds native index range");
  }
  return result;
}

}  // namespace periodicity_detail

using Side = ::pops::BoundarySide;

using ::pops::Face;
using ::pops::face_less;

/// A signed source-axis -> target-axis permutation.
template <int Dim>
class SignedPermutation {
  static_assert(Dim >= 1 && Dim <= 3,
                "nd_proof::SignedPermutation only supports dimensions 1, 2, and 3");

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
        throw std::invalid_argument("nd_proof::SignedPermutation must be a bijection");
      if (sign_[source] != -1 && sign_[source] != 1)
        throw std::invalid_argument("nd_proof::SignedPermutation signs must be -1 or +1");
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
      result[target_axis] = periodicity_detail::checked_index(
          periodicity_detail::checked_add(signed_source, target_offset_[target_axis],
                                          "nd_proof affine index transform overflow"),
          "nd_proof affine index transform exceeds native index range");
    }
    return result;
  }

  Box<Dim> apply(const Box<Dim>& source) const {
    if (source.empty())
      return source;
    Box<Dim> result;
    for (int source_axis = 0; source_axis < Dim; ++source_axis) {
      const int target_axis = source_to_target_.target_axes()[source_axis];
      const std::int64_t first = periodicity_detail::checked_add(
          static_cast<std::int64_t>(source_to_target_.signs()[source_axis]) *
              source.lo[source_axis],
          target_offset_[target_axis], "nd_proof affine box transform overflow");
      const std::int64_t second = periodicity_detail::checked_add(
          static_cast<std::int64_t>(source_to_target_.signs()[source_axis]) *
              source.hi[source_axis],
          target_offset_[target_axis], "nd_proof affine box transform overflow");
      result.lo[target_axis] = periodicity_detail::checked_index(
          std::min(first, second), "nd_proof affine box transform exceeds native index range");
      result.hi[target_axis] = periodicity_detail::checked_index(
          std::max(first, second), "nd_proof affine box transform exceeds native index range");
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
              ? periodicity_detail::checked_negate(target_offset_[target_axis],
                                                   "nd_proof affine inverse overflow")
              : target_offset_[target_axis];
    }
    return AffineIndexTransform{inverse_permutation, inverse_offset};
  }

  bool operator==(const AffineIndexTransform&) const = default;

 private:
  SignedPermutation<Dim> source_to_target_;
  std::array<std::int64_t, Dim> target_offset_{};
};

/// One signed/permuted identification from a source face interior to a target face exterior.
template <int Dim>
class PeriodicIdentification {
 public:
  PeriodicIdentification(Face<Dim> source, Face<Dim> target,
                         SignedPermutation<Dim> source_to_target = {})
      : source_(source), target_(target), source_to_target_(std::move(source_to_target)) {
    validate_structure();
  }

  const Face<Dim>& source() const noexcept { return source_; }
  const Face<Dim>& target() const noexcept { return target_; }
  const SignedPermutation<Dim>& signed_permutation() const noexcept { return source_to_target_; }

  bool is_axis_translation() const noexcept {
    return source_.axis == target_.axis && source_to_target_.is_identity();
  }

  PeriodicIdentification canonical() const {
    if (!face_less(target_, source_))
      return *this;
    return PeriodicIdentification{target_, source_, source_to_target_.inverse()};
  }

  void validate(const Box<Dim>& domain) const {
    validate_structure();
    if (domain.empty())
      throw std::invalid_argument("nd_proof periodic topology requires a non-empty domain");
    for (int source_axis = 0; source_axis < Dim; ++source_axis) {
      if (source_axis == source_.axis)
        continue;
      const int target_axis = source_to_target_.target_axes()[source_axis];
      if (domain.length(source_axis) != domain.length(target_axis))
        throw std::invalid_argument(
            "nd_proof mapped periodic tangential extents must agree under the signed permutation");
    }
  }

  AffineIndexTransform<Dim> source_interior_to_target_exterior(const Box<Dim>& domain) const {
    validate(domain);
    std::array<std::int64_t, Dim> target_offset{};
    for (int source_axis = 0; source_axis < Dim; ++source_axis) {
      const int target_axis = source_to_target_.target_axes()[source_axis];
      const std::int64_t sign = source_to_target_.signs()[source_axis];
      if (source_axis == source_.axis) {
        const std::int64_t source_adjacent =
            source_.side == Side::lower ? domain.lo[source_axis] : domain.hi[source_axis];
        const std::int64_t target_first_exterior =
            target_.side == Side::lower ? static_cast<std::int64_t>(domain.lo[target_axis]) - 1
                                        : static_cast<std::int64_t>(domain.hi[target_axis]) + 1;
        target_offset[target_axis] =
            periodicity_detail::checked_add(target_first_exterior, -sign * source_adjacent,
                                            "nd_proof periodic normal affine offset overflow");
      } else if (sign == 1) {
        target_offset[target_axis] = static_cast<std::int64_t>(domain.lo[target_axis]) -
                                     static_cast<std::int64_t>(domain.lo[source_axis]);
      } else {
        target_offset[target_axis] =
            periodicity_detail::checked_add(domain.hi[target_axis], domain.lo[source_axis],
                                            "nd_proof periodic tangential affine offset overflow");
      }
    }
    return AffineIndexTransform<Dim>{source_to_target_, target_offset};
  }

  AffineIndexTransform<Dim> target_exterior_to_source_interior(const Box<Dim>& domain) const {
    return source_interior_to_target_exterior(domain).inverse();
  }

  bool operator==(const PeriodicIdentification&) const = default;

 private:
  void validate_structure() const {
    if (source_ == target_)
      throw std::invalid_argument("nd_proof periodic identification requires distinct faces");
    if (source_to_target_.target_axes()[source_.axis] != target_.axis)
      throw std::invalid_argument(
          "nd_proof periodic normal axis does not map to the target normal");
    const int source_outward = source_.side == Side::lower ? -1 : 1;
    const int target_outward = target_.side == Side::lower ? -1 : 1;
    const int required_sign = -source_outward * target_outward;
    if (source_to_target_.signs()[source_.axis] != required_sign)
      throw std::invalid_argument(
          "nd_proof periodic normal sign does not map source interior to target exterior");
  }

  Face<Dim> source_;
  Face<Dim> target_;
  SignedPermutation<Dim> source_to_target_;
};

/// Canonical topology identity.  It stores no domain-derived translation offsets.
template <int Dim>
class PeriodicTopology {
 public:
  PeriodicTopology() = default;
  explicit PeriodicTopology(std::vector<PeriodicIdentification<Dim>> identifications) {
    for (PeriodicIdentification<Dim>& identification : identifications)
      identification = identification.canonical();
    std::sort(
        identifications.begin(), identifications.end(),
        [](const PeriodicIdentification<Dim>& left, const PeriodicIdentification<Dim>& right) {
          if (left.source().ordinal() != right.source().ordinal())
            return left.source().ordinal() < right.source().ordinal();
          if (left.target().ordinal() != right.target().ordinal())
            return left.target().ordinal() < right.target().ordinal();
          if (left.signed_permutation().target_axes() != right.signed_permutation().target_axes())
            return left.signed_permutation().target_axes() <
                   right.signed_permutation().target_axes();
          return left.signed_permutation().signs() < right.signed_permutation().signs();
        });

    std::array<bool, 2 * Dim> assigned{};
    for (const PeriodicIdentification<Dim>& identification : identifications) {
      const int source = identification.source().ordinal();
      const int target = identification.target().ordinal();
      if (assigned[source] || assigned[target])
        throw std::invalid_argument("nd_proof periodic topology assigns one face more than once");
      assigned[source] = true;
      assigned[target] = true;
    }
    identifications_ = std::move(identifications);
  }

  static PeriodicTopology axis_translations(const std::array<bool, Dim>& periodic_axes) {
    std::vector<PeriodicIdentification<Dim>> identifications;
    for (int axis = 0; axis < Dim; ++axis)
      if (periodic_axes[axis])
        identifications.emplace_back(Face<Dim>{axis, Side::lower}, Face<Dim>{axis, Side::upper});
    return PeriodicTopology{std::move(identifications)};
  }

  const std::vector<PeriodicIdentification<Dim>>& identifications() const noexcept {
    return identifications_;
  }

  bool is_axis_translation_only() const noexcept {
    for (const PeriodicIdentification<Dim>& identification : identifications_)
      if (!identification.is_axis_translation())
        return false;
    return true;
  }

  bool axis_is_translation_periodic(int axis) const {
    if (axis < 0 || axis >= Dim)
      throw std::invalid_argument("nd_proof periodic axis is outside the compile-time rank");
    for (const PeriodicIdentification<Dim>& identification : identifications_)
      if (identification.is_axis_translation() && identification.source().axis == axis)
        return true;
    return false;
  }

  void validate(const Box<Dim>& domain) const {
    for (const PeriodicIdentification<Dim>& identification : identifications_)
      identification.validate(domain);
  }

  bool operator==(const PeriodicTopology&) const = default;

 private:
  std::vector<PeriodicIdentification<Dim>> identifications_;
};

/// Explicit cap for the finite catalogue of ordinary axis-translation images.
struct AxisTranslationImageBudget {
  std::size_t images;
};

template <int Dim>
struct AxisTranslationImage {
  std::array<std::int64_t, Dim> multiples{};
  std::array<std::int64_t, Dim> translation{};

  bool is_zero() const noexcept {
    for (const std::int64_t value : multiples)
      if (value != 0)
        return false;
    return true;
  }

  Index<Dim> apply(const Index<Dim>& source) const {
    Index<Dim> result;
    for (int axis = 0; axis < Dim; ++axis)
      result[axis] = periodicity_detail::checked_index(
          periodicity_detail::checked_add(source[axis], translation[axis],
                                          "nd_proof periodic index translation overflow"),
          "nd_proof periodic index translation exceeds native index range");
    return result;
  }

  Box<Dim> apply(const Box<Dim>& source) const {
    return periodicity_detail::translate_box<Dim>(source, translation,
                                                  "nd_proof periodic box translation overflow");
  }

  bool operator==(const AxisTranslationImage&) const = default;
};

/// Enumerates ordinary axis-translation images only.  For each axis the multiplier order is
/// ``0, -1, +1, -2, +2, ...``; Cartesian combinations use axis 0 as the fastest coordinate.
template <int Dim>
std::vector<AxisTranslationImage<Dim>> enumerate_axis_translation_images(
    const Box<Dim>& domain, const Extent<Dim>& ghosts, const PeriodicTopology<Dim>& topology,
    AxisTranslationImageBudget budget) {
  if (domain.empty())
    throw std::invalid_argument("nd_proof axis-translation images require a non-empty domain");
  topology.validate(domain);
  if (!topology.is_axis_translation_only())
    throw std::invalid_argument(
        "nd_proof axis-translation images do not support mapped periodic identifications");

  std::array<std::vector<std::int64_t>, Dim> axis_multiples;
  std::size_t image_count = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    if (ghosts[axis] < 0)
      throw std::invalid_argument("nd_proof periodic ghost depths must be non-negative");
    std::int64_t maximum_multiple = 0;
    const std::int64_t extent = domain.length(axis);
    if (topology.axis_is_translation_periodic(axis)) {
      maximum_multiple = ghosts[axis] / extent + (ghosts[axis] % extent == 0 ? 0 : 1);
      (void)periodicity_detail::checked_multiple(
          maximum_multiple, extent, "nd_proof periodic image translation overflows int64_t");
    }
    if (maximum_multiple >
        static_cast<std::int64_t>((std::numeric_limits<std::size_t>::max() - 1) / 2))
      throw std::length_error("nd_proof periodic image count exceeds size_t");
    const std::size_t axis_count = 1 + 2 * static_cast<std::size_t>(maximum_multiple);
    if (axis_count > budget.images || image_count > budget.images / axis_count)
      throw std::length_error("nd_proof periodic image count exceeds its explicit budget");
    image_count *= axis_count;

    std::vector<std::int64_t>& values = axis_multiples[axis];
    if (axis_count > values.max_size())
      throw std::length_error("nd_proof periodic image axis count exceeds vector capacity");
    values.reserve(axis_count);
    values.push_back(0);
    for (std::int64_t magnitude = 1; magnitude <= maximum_multiple;) {
      values.push_back(-magnitude);
      values.push_back(magnitude);
      if (magnitude == maximum_multiple)
        break;
      ++magnitude;
    }
  }

  std::vector<AxisTranslationImage<Dim>> images;
  if (image_count > images.max_size())
    throw std::length_error("nd_proof periodic image count exceeds vector capacity");
  images.reserve(image_count);
  for (std::size_t ordinal = 0; ordinal < image_count; ++ordinal) {
    AxisTranslationImage<Dim> image;
    std::size_t quotient = ordinal;
    for (int axis = 0; axis < Dim; ++axis) {
      const std::vector<std::int64_t>& values = axis_multiples[axis];
      const std::int64_t multiple = values[quotient % values.size()];
      quotient /= values.size();
      image.multiples[axis] = multiple;
      image.translation[axis] = periodicity_detail::checked_multiple(
          multiple, domain.length(axis), "nd_proof periodic image translation overflows int64_t");
    }
    images.push_back(image);
  }
  return images;
}

}  // namespace pops::mesh::nd_proof
