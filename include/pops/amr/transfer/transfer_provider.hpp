/// @file
/// @brief Allocation-free prepared AMR restriction and interpolation in 1D, 2D, and 3D.

#pragma once

#include <pops/amr/refinement_ratio.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/storage/field_view.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace pops::amr::transfer {

/// Logical location of transferred values.  The first ND substrate intentionally authenticates
/// only cell-centered transfers; the other values make unsupported routes explicit and fail-closed.
enum class Centering : unsigned char { Cell = 0, Node = 1, Face0 = 2, Face1 = 3, Face2 = 4 };

enum class TransferKind : unsigned char {
  ConservativeRestriction = 0,
  LinearProlongation = 1,
  CoarseFineGhostInterpolation = 2,
  ConstantInjection = 3,
  FifthOrderCoarseFineGhostInterpolation = 4,
  NodeMultilinearProlongation = 5,
  DivergencePreservingFaceProlongation = 6,
};

/// Limiter carried by an authenticated interpolation strategy.
enum class SlopeLimiter : unsigned char { None = 0, MonotonizedCentral = 1 };

struct TransferCapabilities {
  int interpolation_order = 0;
  int source_stencil_radius = 0;
  bool conservative = false;
  bool allocation_free_hot_path = false;
  SlopeLimiter slope_limiter = SlopeLimiter::None;

  constexpr bool operator==(const TransferCapabilities&) const = default;
};

/// Component interval bound during preparation.  A prepared kernel never owns a dynamic list.
struct ComponentRange {
  int source_begin = 0;
  int destination_begin = 0;
  int count = 1;

  constexpr bool operator==(const ComponentRange&) const = default;
};

/// Affine relationship between level index spaces.  Origins need not be zero or positive.
template <int Dim>
struct IndexMapping {
  Index<Dim> coarse_origin{};
  Index<Dim> fine_origin{};

  constexpr bool operator==(const IndexMapping&) const = default;
};

namespace detail {

inline int checked_transfer_index(std::int64_t value, const char* operation) {
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    throw std::overflow_error(operation);
  return static_cast<int>(value);
}

inline std::int64_t checked_transfer_add(std::int64_t left, std::int64_t right,
                                         const char* operation) {
  if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
      (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right))
    throw std::overflow_error(operation);
  return left + right;
}

inline std::int64_t checked_transfer_multiply(std::int64_t value, int positive_multiplier,
                                              const char* operation) {
  if (value > std::numeric_limits<std::int64_t>::max() / positive_multiplier ||
      value < std::numeric_limits<std::int64_t>::min() / positive_multiplier)
    throw std::overflow_error(operation);
  return value * positive_multiplier;
}

POPS_HD constexpr std::int64_t floor_div_positive(std::int64_t numerator, int denominator) {
  const std::int64_t quotient = numerator / denominator;
  const std::int64_t remainder = numerator % denominator;
  return remainder < 0 ? quotient - 1 : quotient;
}

POPS_HD constexpr std::int64_t ceil_div_positive(std::int64_t numerator, int denominator) {
  const std::int64_t quotient = numerator / denominator;
  const std::int64_t remainder = numerator % denominator;
  return remainder > 0 ? quotient + 1 : quotient;
}

template <int Dim>
struct ValidatedView {
  Box<Dim> box{};
  std::uintptr_t begin = 0;
  std::uintptr_t end = 0;
};

template <class T, int Dim>
ValidatedView<Dim> validate_view(const FieldView<T, Dim>& view) {
  if (view.data == nullptr || view.ncomp < 1 || view.component_stride < 1)
    throw std::invalid_argument("prepared ND transfer requires a valid non-empty FieldView");

  Box<Dim> box{};
  box.lo = view.origin;
  std::int64_t maximum_offset = 0;
  std::int64_t minimum_stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    const std::int64_t extent = view.extents[axis];
    const std::int64_t stride = view.strides[axis];
    if (extent < 1 || stride < minimum_stride)
      throw std::invalid_argument(
          "prepared ND transfer requires positive non-overlapping FieldView strides");
    box.hi[axis] = checked_transfer_index(
        checked_transfer_add(view.origin[axis], extent - 1,
                             "prepared ND transfer FieldView index range exceeds int64_t"),
        "prepared ND transfer FieldView index range exceeds signed coordinates");
    if (extent - 1 > (std::numeric_limits<std::int64_t>::max() - maximum_offset) / stride)
      throw std::overflow_error("prepared ND transfer FieldView spatial span exceeds int64_t");
    maximum_offset += (extent - 1) * stride;
    if (extent > std::numeric_limits<std::int64_t>::max() / stride)
      throw std::overflow_error("prepared ND transfer FieldView stride hierarchy exceeds int64_t");
    minimum_stride = extent * stride;
  }
  if (view.component_stride < minimum_stride)
    throw std::invalid_argument(
        "prepared ND transfer FieldView components overlap its spatial storage");
  const std::int64_t component_count = static_cast<std::int64_t>(view.ncomp) - 1;
  if (component_count >
      (std::numeric_limits<std::int64_t>::max() - maximum_offset) / view.component_stride)
    throw std::overflow_error("prepared ND transfer FieldView component span exceeds int64_t");
  maximum_offset += component_count * view.component_stride;
  if (maximum_offset == std::numeric_limits<std::int64_t>::max())
    throw std::overflow_error("prepared ND transfer FieldView element span exceeds int64_t");

  const auto elements = static_cast<std::uint64_t>(maximum_offset) + 1;
  if (elements > std::numeric_limits<std::uintptr_t>::max() / sizeof(std::remove_const_t<T>))
    throw std::overflow_error("prepared ND transfer FieldView byte span exceeds uintptr_t");
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(view.data);
  const std::uintptr_t bytes =
      static_cast<std::uintptr_t>(elements * sizeof(std::remove_const_t<T>));
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
    throw std::overflow_error("prepared ND transfer FieldView address span wraps uintptr_t");
  return {box, begin, begin + bytes};
}

template <int Dim>
void validate_components(const FieldView<const Real, Dim>& source,
                         const FieldView<Real, Dim>& destination,
                         const ComponentRange& components) {
  if (components.source_begin < 0 || components.destination_begin < 0 || components.count < 1 ||
      components.source_begin > source.ncomp - components.count ||
      components.destination_begin > destination.ncomp - components.count)
    throw std::invalid_argument("prepared ND transfer component interval is outside its fields");
}

template <int Dim>
Box<Dim> refined_source_box(const Box<Dim>& coarse_region, const RefinementRatio<Dim>& ratio,
                            const IndexMapping<Dim>& mapping) {
  Box<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::int64_t lower_relative =
        static_cast<std::int64_t>(coarse_region.lo[axis]) - mapping.coarse_origin[axis];
    const std::int64_t upper_relative =
        static_cast<std::int64_t>(coarse_region.hi[axis]) - mapping.coarse_origin[axis];
    const std::int64_t lower_scaled = checked_transfer_multiply(
        lower_relative, ratio[axis], "prepared ND restriction index mapping exceeds int64_t");
    const std::int64_t upper_scaled = checked_transfer_multiply(
        upper_relative, ratio[axis], "prepared ND restriction index mapping exceeds int64_t");
    const std::int64_t lower =
        checked_transfer_add(mapping.fine_origin[axis], lower_scaled,
                             "prepared ND restriction lower source index exceeds int64_t");
    const std::int64_t upper = checked_transfer_add(
        checked_transfer_add(mapping.fine_origin[axis], upper_scaled,
                             "prepared ND restriction upper source index exceeds int64_t"),
        static_cast<std::int64_t>(ratio[axis]) - 1,
        "prepared ND restriction upper source index exceeds int64_t");
    result.lo[axis] = checked_transfer_index(
        lower, "prepared ND restriction lower source index exceeds signed coordinates");
    result.hi[axis] = checked_transfer_index(
        upper, "prepared ND restriction upper source index exceeds signed coordinates");
  }
  return result;
}

template <int Dim>
Box<Dim> interpolation_source_box(const Box<Dim>& fine_region, const RefinementRatio<Dim>& ratio,
                                  const IndexMapping<Dim>& mapping, int stencil_radius) {
  Box<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::int64_t lower_relative =
        static_cast<std::int64_t>(fine_region.lo[axis]) - mapping.fine_origin[axis];
    const std::int64_t upper_relative =
        static_cast<std::int64_t>(fine_region.hi[axis]) - mapping.fine_origin[axis];
    std::int64_t lower = checked_transfer_add(
        mapping.coarse_origin[axis], floor_div_positive(lower_relative, ratio[axis]),
        "prepared ND interpolation lower source index exceeds int64_t");
    std::int64_t upper = checked_transfer_add(
        mapping.coarse_origin[axis], floor_div_positive(upper_relative, ratio[axis]),
        "prepared ND interpolation upper source index exceeds int64_t");
    if (ratio[axis] > 1 && stencil_radius > 0) {
      lower = checked_transfer_add(lower, -stencil_radius,
                                   "prepared ND interpolation lower stencil exceeds int64_t");
      upper = checked_transfer_add(upper, stencil_radius,
                                   "prepared ND interpolation upper stencil exceeds int64_t");
    }
    result.lo[axis] = checked_transfer_index(
        lower, "prepared ND interpolation lower stencil exceeds signed coordinates");
    result.hi[axis] = checked_transfer_index(
        upper, "prepared ND interpolation upper stencil exceeds signed coordinates");
  }
  return result;
}

template <int Dim>
Box<Dim> node_interpolation_source_box(const Box<Dim>& fine_region,
                                       const RefinementRatio<Dim>& ratio,
                                       const IndexMapping<Dim>& mapping) {
  Box<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::int64_t lower_relative =
        static_cast<std::int64_t>(fine_region.lo[axis]) - mapping.fine_origin[axis];
    const std::int64_t upper_relative =
        static_cast<std::int64_t>(fine_region.hi[axis]) - mapping.fine_origin[axis];
    const std::int64_t lower = checked_transfer_add(
        mapping.coarse_origin[axis], floor_div_positive(lower_relative, ratio[axis]),
        "prepared node interpolation lower source index exceeds int64_t");
    const std::int64_t upper = checked_transfer_add(
        mapping.coarse_origin[axis], ceil_div_positive(upper_relative, ratio[axis]),
        "prepared node interpolation upper source index exceeds int64_t");
    result.lo[axis] = checked_transfer_index(
        lower, "prepared node interpolation lower source index exceeds signed coordinates");
    result.hi[axis] = checked_transfer_index(
        upper, "prepared node interpolation upper source index exceeds signed coordinates");
  }
  return result;
}

template <int Dim>
Box<Dim> face_interpolation_source_box(const Box<Dim>& fine_face_region, int normal_axis,
                                       const RefinementRatio<Dim>& ratio,
                                       const IndexMapping<Dim>& mapping) {
  Box<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::int64_t lower_relative =
        static_cast<std::int64_t>(fine_face_region.lo[axis]) - mapping.fine_origin[axis];
    const std::int64_t upper_relative =
        static_cast<std::int64_t>(fine_face_region.hi[axis]) - mapping.fine_origin[axis];
    std::int64_t lower = checked_transfer_add(
        mapping.coarse_origin[axis], floor_div_positive(lower_relative, ratio[axis]),
        "prepared face interpolation lower source index exceeds int64_t");
    std::int64_t upper =
        checked_transfer_add(mapping.coarse_origin[axis],
                             axis == normal_axis ? ceil_div_positive(upper_relative, ratio[axis])
                                                 : floor_div_positive(upper_relative, ratio[axis]),
                             "prepared face interpolation upper source index exceeds int64_t");
    if (axis != normal_axis && ratio[axis] > 1) {
      lower = checked_transfer_add(lower, -1,
                                   "prepared face interpolation lower stencil exceeds int64_t");
      upper = checked_transfer_add(upper, 1,
                                   "prepared face interpolation upper stencil exceeds int64_t");
    }
    result.lo[axis] = checked_transfer_index(
        lower, "prepared face interpolation lower stencil exceeds signed coordinates");
    result.hi[axis] = checked_transfer_index(
        upper, "prepared face interpolation upper stencil exceeds signed coordinates");
  }
  return result;
}

template <int Dim>
Box<Dim> face_region(const Box<Dim>& cell_region, int normal_axis) {
  Box<Dim> result = cell_region;
  result.hi[normal_axis] = checked_transfer_index(
      checked_transfer_add(result.hi[normal_axis], 1,
                           "prepared face destination region exceeds int64_t"),
      "prepared face destination region exceeds signed coordinates");
  return result;
}

POPS_HD inline Real absolute_value(Real value) {
  return value < Real(0) ? -value : value;
}

/// Monotonized-central slope in coarse-cell index units.  For smooth affine data it returns the
/// exact centered slope; at extrema and discontinuities it collapses without changing the parent
/// average because every child offset remains antisymmetric about the parent center.
POPS_HD inline Real monotonized_central_slope(Real lower, Real center, Real upper) {
  const Real backward = center - lower;
  const Real centered = Real(0.5) * (upper - lower);
  const Real forward = upper - center;
  const bool positive = backward > Real(0) && centered > Real(0) && forward > Real(0);
  const bool negative = backward < Real(0) && centered < Real(0) && forward < Real(0);
  if (!positive && !negative)
    return Real(0);
  Real magnitude = absolute_value(Real(2) * backward);
  const Real centered_magnitude = absolute_value(centered);
  const Real forward_magnitude = absolute_value(Real(2) * forward);
  if (centered_magnitude < magnitude)
    magnitude = centered_magnitude;
  if (forward_magnitude < magnitude)
    magnitude = forward_magnitude;
  return positive ? magnitude : -magnitude;
}

POPS_HD inline Real integer_power(Real value, int exponent) {
  Real result = Real(1);
  for (int power = 0; power < exponent; ++power)
    result *= value;
  return result;
}

/// Entry of the inverse map from five adjacent unit-cell averages to the coefficients of the
/// unique quartic polynomial written about the central coarse-cell center.  Keeping this as a
/// fixed switch makes the device kernel allocation-free and avoids a per-cell matrix solve.
POPS_HD inline Real inverse_quartic_cell_moment(int degree, int stencil) {
  if (degree == 0) {
    if (stencil == 0 || stencil == 4)
      return Real(3) / Real(640);
    if (stencil == 1 || stencil == 3)
      return -Real(29) / Real(480);
    return Real(1067) / Real(960);
  }
  if (degree == 1) {
    if (stencil == 0)
      return Real(5) / Real(48);
    if (stencil == 1)
      return -Real(17) / Real(24);
    if (stencil == 3)
      return Real(17) / Real(24);
    if (stencil == 4)
      return -Real(5) / Real(48);
    return Real(0);
  }
  if (degree == 2) {
    if (stencil == 0 || stencil == 4)
      return -Real(1) / Real(16);
    if (stencil == 1 || stencil == 3)
      return Real(3) / Real(4);
    return -Real(11) / Real(8);
  }
  if (degree == 3) {
    if (stencil == 0)
      return -Real(1) / Real(12);
    if (stencil == 1)
      return Real(1) / Real(6);
    if (stencil == 3)
      return -Real(1) / Real(6);
    if (stencil == 4)
      return Real(1) / Real(12);
    return Real(0);
  }
  if (stencil == 0 || stencil == 4)
    return Real(1) / Real(24);
  if (stencil == 1 || stencil == 3)
    return -Real(1) / Real(6);
  return Real(1) / Real(4);
}

/// Conservative sub-cell-average weights for one child of a coarse cell.  They reconstruct the
/// unique quartic matching coarse averages at offsets [-2,2], then integrate that polynomial over
/// the exact child interval.  The weights therefore reproduce every polynomial through degree
/// four and sum over all children to the central parent average for every positive ratio.
POPS_HD inline void fifth_order_cell_average_weights(int child, int ratio, Real weights[5]) {
  if (ratio == 1) {
    weights[0] = Real(0);
    weights[1] = Real(0);
    weights[2] = Real(1);
    weights[3] = Real(0);
    weights[4] = Real(0);
    return;
  }
  const Real inverse_ratio = Real(1) / static_cast<Real>(ratio);
  const Real lower = -Real(0.5) + static_cast<Real>(child) * inverse_ratio;
  const Real upper = lower + inverse_ratio;
  Real averaged_monomials[5]{};
  averaged_monomials[0] = Real(1);
  for (int degree = 1; degree < 5; ++degree) {
    const int power = degree + 1;
    averaged_monomials[degree] = static_cast<Real>(ratio) *
                                 (integer_power(upper, power) - integer_power(lower, power)) /
                                 static_cast<Real>(power);
  }
  for (int stencil = 0; stencil < 5; ++stencil) {
    weights[stencil] = Real(0);
    for (int degree = 0; degree < 5; ++degree)
      weights[stencil] += averaged_monomials[degree] * inverse_quartic_cell_moment(degree, stencil);
  }
}

template <int Dim>
POPS_HD bool increment_child(Index<Dim>& child, const RefinementRatio<Dim>& ratio) {
  for (int axis = 0; axis < Dim; ++axis) {
    ++child[axis];
    if (child[axis] < ratio[axis])
      return true;
    child[axis] = 0;
  }
  return false;
}

template <int Dim>
POPS_HD void fine_parent_and_child(const Index<Dim>& fine, const RefinementRatio<Dim>& ratio,
                                   const IndexMapping<Dim>& mapping, Index<Dim>& parent,
                                   Index<Dim>& child) {
  for (int axis = 0; axis < Dim; ++axis) {
    const std::int64_t relative = static_cast<std::int64_t>(fine[axis]) - mapping.fine_origin[axis];
    const std::int64_t parent_relative = floor_div_positive(relative, ratio[axis]);
    parent[axis] =
        static_cast<int>(static_cast<std::int64_t>(mapping.coarse_origin[axis]) + parent_relative);
    child[axis] = static_cast<int>(relative - parent_relative * ratio[axis]);
  }
}

}  // namespace detail

template <int Dim>
class PreparedTransfer {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "PreparedTransfer only supports dimensions 1, 2, and 3");

  POPS_HD void operator()(const Index<Dim>& destination_index) const {
    if (kind_ == TransferKind::ConservativeRestriction)
      restrict_cell(destination_index);
    else if (kind_ == TransferKind::ConstantInjection)
      inject_cell(destination_index);
    else if (kind_ == TransferKind::NodeMultilinearProlongation)
      interpolate_node(destination_index);
    else if (kind_ == TransferKind::FifthOrderCoarseFineGhostInterpolation)
      interpolate_fifth_order_cell(destination_index);
    else
      interpolate_cell(destination_index);
  }

  POPS_HD TransferKind kind() const { return kind_; }
  POPS_HD const Box<Dim>& destination_region() const { return destination_region_; }
  POPS_HD const RefinementRatio<Dim>& refinement_ratio() const { return ratio_; }
  POPS_HD ComponentRange components() const { return components_; }
  POPS_HD SlopeLimiter slope_limiter() const {
    return kind_ == TransferKind::LinearProlongation ||
                   kind_ == TransferKind::CoarseFineGhostInterpolation
               ? SlopeLimiter::MonotonizedCentral
               : SlopeLimiter::None;
  }

 private:
  template <int, Centering>
  friend class TransferProvider;

  POPS_HD PreparedTransfer(TransferKind kind, RefinementRatio<Dim> ratio, IndexMapping<Dim> mapping,
                           ComponentRange components, FieldView<const Real, Dim> source,
                           FieldView<Real, Dim> destination, Box<Dim> destination_region)
      : kind_(kind),
        ratio_(ratio),
        mapping_(mapping),
        components_(components),
        source_(source),
        destination_(destination),
        destination_region_(destination_region) {}

  POPS_HD void restrict_cell(const Index<Dim>& coarse) const {
    Index<Dim> fine_base{};
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t coarse_relative =
          static_cast<std::int64_t>(coarse[axis]) - mapping_.coarse_origin[axis];
      fine_base[axis] = static_cast<int>(static_cast<std::int64_t>(mapping_.fine_origin[axis]) +
                                         coarse_relative * ratio_[axis]);
    }
    for (int component = 0; component < components_.count; ++component) {
      const int source_component = components_.source_begin + component;
      const int destination_component = components_.destination_begin + component;
      const Real anchor = source_(fine_base, source_component);
      Real correction = Real(0);
      Index<Dim> child{};
      do {
        Index<Dim> fine = fine_base;
        for (int axis = 0; axis < Dim; ++axis)
          fine[axis] += child[axis];
        correction += source_(fine, source_component) - anchor;
      } while (detail::increment_child(child, ratio_));
      destination_(coarse, destination_component) =
          anchor + correction / static_cast<Real>(ratio_.child_count());
    }
  }

  POPS_HD void interpolate_cell(const Index<Dim>& fine) const {
    Index<Dim> parent{};
    Index<Dim> child{};
    detail::fine_parent_and_child(fine, ratio_, mapping_, parent, child);
    for (int component = 0; component < components_.count; ++component) {
      const int source_component = components_.source_begin + component;
      const int destination_component = components_.destination_begin + component;
      Real value = source_(parent, source_component);
      for (int axis = 0; axis < Dim; ++axis) {
        if (ratio_[axis] == 1)
          continue;
        Index<Dim> lower = parent;
        Index<Dim> upper = parent;
        --lower[axis];
        ++upper[axis];
        const Real slope = detail::monotonized_central_slope(source_(lower, source_component),
                                                             source_(parent, source_component),
                                                             source_(upper, source_component));
        const std::int64_t offset_numerator = std::int64_t{2} * child[axis] + 1 - ratio_[axis];
        const std::int64_t offset_denominator = std::int64_t{2} * ratio_[axis];
        value +=
            slope * static_cast<Real>(offset_numerator) / static_cast<Real>(offset_denominator);
      }
      destination_(fine, destination_component) = value;
    }
  }

  POPS_HD void inject_cell(const Index<Dim>& fine) const {
    Index<Dim> parent{};
    Index<Dim> child{};
    detail::fine_parent_and_child(fine, ratio_, mapping_, parent, child);
    (void)child;
    for (int component = 0; component < components_.count; ++component)
      destination_(fine, components_.destination_begin + component) =
          source_(parent, components_.source_begin + component);
  }

  POPS_HD void interpolate_node(const Index<Dim>& fine) const {
    Index<Dim> parent{};
    Index<Dim> child{};
    detail::fine_parent_and_child(fine, ratio_, mapping_, parent, child);
    int active_axes[Dim]{};
    int active_count = 0;
    for (int axis = 0; axis < Dim; ++axis)
      if (child[axis] != 0)
        active_axes[active_count++] = axis;
    const int corner_count = 1 << active_count;
    for (int component = 0; component < components_.count; ++component) {
      Real value = Real(0);
      for (int corner = 0; corner < corner_count; ++corner) {
        Index<Dim> source_index = parent;
        Real weight = Real(1);
        for (int active = 0; active < active_count; ++active) {
          const int axis = active_axes[active];
          const Real upper_weight =
              static_cast<Real>(child[axis]) / static_cast<Real>(ratio_[axis]);
          if ((corner & (1 << active)) != 0) {
            ++source_index[axis];
            weight *= upper_weight;
          } else {
            weight *= Real(1) - upper_weight;
          }
        }
        value += weight * source_(source_index, components_.source_begin + component);
      }
      destination_(fine, components_.destination_begin + component) = value;
    }
  }

  POPS_HD void interpolate_fifth_order_cell(const Index<Dim>& fine) const {
    Index<Dim> parent{};
    Index<Dim> child{};
    detail::fine_parent_and_child(fine, ratio_, mapping_, parent, child);
    Real axis_weights[Dim][5]{};
    int stencil_count = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      detail::fifth_order_cell_average_weights(child[axis], ratio_[axis], axis_weights[axis]);
      if (ratio_[axis] > 1)
        stencil_count *= 5;
    }
    for (int component = 0; component < components_.count; ++component) {
      Real value = Real(0);
      for (int ordinal = 0; ordinal < stencil_count; ++ordinal) {
        int remainder = ordinal;
        Index<Dim> source_index = parent;
        Real weight = Real(1);
        for (int axis = 0; axis < Dim; ++axis) {
          if (ratio_[axis] == 1)
            continue;
          const int stencil = remainder % 5;
          remainder /= 5;
          source_index[axis] += stencil - 2;
          weight *= axis_weights[axis][stencil];
        }
        value += weight * source_(source_index, components_.source_begin + component);
      }
      destination_(fine, components_.destination_begin + component) = value;
    }
  }

  TransferKind kind_;
  RefinementRatio<Dim> ratio_;
  IndexMapping<Dim> mapping_{};
  ComponentRange components_{};
  FieldView<const Real, Dim> source_{};
  FieldView<Real, Dim> destination_{};
  Box<Dim> destination_region_{};
};

template <int Dim>
class DivergencePreservingFaceTransferProvider;

/// Authenticated construction boundary for one prepared ND transfer operation.
///
/// The provider performs all pointer, extent, component, stencil, centering and operation checks
/// on the host.  The returned value contains only fixed-size metadata and non-owning FieldViews;
/// invoking it for each destination index performs no allocation or dynamic dispatch.
template <int Dim, Centering Center>
class TransferProvider {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "TransferProvider only supports dimensions 1, 2, and 3");

  constexpr explicit TransferProvider(TransferKind kind) : kind_(kind) {}

  static constexpr TransferProvider conservative_restriction() {
    return TransferProvider(TransferKind::ConservativeRestriction);
  }

  static constexpr TransferProvider linear_prolongation() {
    return TransferProvider(TransferKind::LinearProlongation);
  }

  /// First-order parent injection is retained only as an explicitly selected strategy.  It is
  /// never used as a fallback when the conservative linear stencil is unavailable.
  static constexpr TransferProvider constant_injection() {
    return TransferProvider(TransferKind::ConstantInjection);
  }

  static constexpr TransferProvider coarse_fine_ghost_interpolation() {
    return TransferProvider(TransferKind::CoarseFineGhostInterpolation);
  }

  static constexpr TransferProvider fifth_order_coarse_fine_ghost_interpolation() {
    return TransferProvider(TransferKind::FifthOrderCoarseFineGhostInterpolation);
  }

  static constexpr TransferProvider node_multilinear_prolongation() {
    return TransferProvider(TransferKind::NodeMultilinearProlongation);
  }

  TransferCapabilities capabilities() const {
    require_supported_route();
    if (kind_ == TransferKind::ConservativeRestriction)
      return {1, 0, true, true, SlopeLimiter::None};
    if (kind_ == TransferKind::ConstantInjection)
      return {1, 0, true, true, SlopeLimiter::None};
    if (kind_ == TransferKind::LinearProlongation)
      return {2, 1, true, true, SlopeLimiter::MonotonizedCentral};
    if (kind_ == TransferKind::FifthOrderCoarseFineGhostInterpolation)
      return {5, 2, false, true, SlopeLimiter::None};
    if (kind_ == TransferKind::NodeMultilinearProlongation)
      return {2, 0, false, true, SlopeLimiter::None};
    return {2, 1, false, true, SlopeLimiter::MonotonizedCentral};
  }

  PreparedTransfer<Dim> prepare(FieldView<const Real, Dim> source, FieldView<Real, Dim> destination,
                                const Box<Dim>& destination_region, RefinementRatio<Dim> ratio,
                                IndexMapping<Dim> mapping = {},
                                ComponentRange components = {}) const {
    require_supported_route();
    if (!ratio.refines_any_axis())
      throw std::invalid_argument(
          "prepared ND transfer requires a non-identity inter-level refinement ratio");
    const auto source_view = detail::validate_view(source);
    const auto destination_view = detail::validate_view(destination);
    if (destination_region.empty() || !destination_view.box.contains(destination_region))
      throw std::invalid_argument(
          "prepared ND transfer destination region is empty or outside its FieldView");
    detail::validate_components(source, destination, components);
    if (source_view.begin < destination_view.end && destination_view.begin < source_view.end)
      throw std::invalid_argument("prepared ND transfer requires non-overlapping field storage");

    const int interpolation_radius = kind_ == TransferKind::ConstantInjection ? 0
                                     : kind_ == TransferKind::FifthOrderCoarseFineGhostInterpolation
                                         ? 2
                                         : 1;
    const Box<Dim> required_source =
        kind_ == TransferKind::ConservativeRestriction
            ? detail::refined_source_box(destination_region, ratio, mapping)
        : kind_ == TransferKind::NodeMultilinearProlongation
            ? detail::node_interpolation_source_box(destination_region, ratio, mapping)
            : detail::interpolation_source_box(destination_region, ratio, mapping,
                                               interpolation_radius);
    if (!source_view.box.contains(required_source))
      throw std::invalid_argument(
          "prepared ND transfer source FieldView does not contain the complete stencil");

    return PreparedTransfer<Dim>(kind_, ratio, mapping, components, source, destination,
                                 destination_region);
  }

 private:
  void require_supported_route() const {
    if constexpr (Center == Centering::Node) {
      if (kind_ == TransferKind::NodeMultilinearProlongation)
        return;
      throw std::invalid_argument(
          "node-centered ND transfer requires the multilinear native route");
    }
    if constexpr (Center != Centering::Cell)
      throw std::invalid_argument(
          "oriented face transfer requires the coupled divergence-preserving provider");
    switch (kind_) {
      case TransferKind::ConservativeRestriction:
      case TransferKind::LinearProlongation:
      case TransferKind::CoarseFineGhostInterpolation:
      case TransferKind::FifthOrderCoarseFineGhostInterpolation:
      case TransferKind::ConstantInjection:
        return;
      case TransferKind::NodeMultilinearProlongation:
      case TransferKind::DivergencePreservingFaceProlongation:
        break;
    }
    throw std::invalid_argument("ND transfer provider identity is not registered");
  }

  TransferKind kind_;
};

/// Coupled face-normal prolongation over one Cartesian vector field.
///
/// Coarse boundary-face averages are prolonged conservatively in every transverse direction.
/// Interior faces start from multilinear normal interpolation, then receive a deterministic
/// allocation-free flux correction on a lexicographic spanning tree of each refined parent cell.
/// The correction makes every fine-cell discrete divergence equal to its parent-cell divergence;
/// its terminal compatibility follows from the conservative boundary-face averages.
template <int Dim>
class PreparedDivergencePreservingFaceTransfer {
 public:
  static_assert(Dim >= 1 && Dim <= 3,
                "PreparedDivergencePreservingFaceTransfer supports dimensions 1, 2, and 3");

  POPS_HD void operator()(int normal_axis, const Index<Dim>& fine_face) const {
    for (int component = 0; component < components_.count; ++component) {
      const int source_component = components_.source_begin + component;
      const int destination_component = components_.destination_begin + component;
      destination_[normal_axis](fine_face, destination_component) =
          base_face_value_(normal_axis, fine_face, source_component) +
          divergence_correction_(normal_axis, fine_face, source_component);
    }
  }

  POPS_HD TransferKind kind() const { return TransferKind::DivergencePreservingFaceProlongation; }
  POPS_HD const Box<Dim>& destination_cell_region() const { return destination_cell_region_; }
  POPS_HD const Box<Dim>& destination_face_region(int normal_axis) const {
    return destination_face_regions_[normal_axis];
  }
  POPS_HD const RefinementRatio<Dim>& refinement_ratio() const { return ratio_; }
  POPS_HD ComponentRange components() const { return components_; }

 private:
  friend class DivergencePreservingFaceTransferProvider<Dim>;

  POPS_HD PreparedDivergencePreservingFaceTransfer(
      RefinementRatio<Dim> ratio, IndexMapping<Dim> mapping, ComponentRange components,
      std::array<FieldView<const Real, Dim>, Dim> source,
      std::array<FieldView<Real, Dim>, Dim> destination, Box<Dim> destination_cell_region,
      std::array<Box<Dim>, Dim> destination_face_regions)
      : ratio_(ratio),
        mapping_(mapping),
        components_(components),
        source_(source),
        destination_(destination),
        destination_cell_region_(destination_cell_region),
        destination_face_regions_(destination_face_regions) {}

  POPS_HD void parent_and_child_(const Index<Dim>& fine, Index<Dim>& parent,
                                 Index<Dim>& child) const {
    detail::fine_parent_and_child(fine, ratio_, mapping_, parent, child);
  }

  POPS_HD Real transverse_boundary_value_(int normal_axis, const Index<Dim>& parent,
                                          const Index<Dim>& child, int normal_face,
                                          int component) const {
    Index<Dim> center = parent;
    center[normal_axis] = normal_face;
    Real value = source_[normal_axis](center, component);
    for (int axis = 0; axis < Dim; ++axis) {
      if (axis == normal_axis || ratio_[axis] == 1)
        continue;
      Index<Dim> lower = center;
      Index<Dim> upper = center;
      --lower[axis];
      ++upper[axis];
      const Real slope = detail::monotonized_central_slope(source_[normal_axis](lower, component),
                                                           source_[normal_axis](center, component),
                                                           source_[normal_axis](upper, component));
      const std::int64_t offset_numerator = std::int64_t{2} * child[axis] + 1 - ratio_[axis];
      value += slope * static_cast<Real>(offset_numerator) /
               static_cast<Real>(std::int64_t{2} * ratio_[axis]);
    }
    return value;
  }

  POPS_HD Real base_face_value_(int normal_axis, const Index<Dim>& fine_face, int component) const {
    Index<Dim> parent{};
    Index<Dim> child{};
    parent_and_child_(fine_face, parent, child);
    const Real lower =
        transverse_boundary_value_(normal_axis, parent, child, parent[normal_axis], component);
    if (child[normal_axis] == 0)
      return lower;
    const Real upper =
        transverse_boundary_value_(normal_axis, parent, child, parent[normal_axis] + 1, component);
    const Real alpha =
        static_cast<Real>(child[normal_axis]) / static_cast<Real>(ratio_[normal_axis]);
    return lower + alpha * (upper - lower);
  }

  POPS_HD Index<Dim> fine_cell_(const Index<Dim>& parent, const Index<Dim>& child) const {
    Index<Dim> fine{};
    for (int axis = 0; axis < Dim; ++axis) {
      fine[axis] = static_cast<int>(
          static_cast<std::int64_t>(mapping_.fine_origin[axis]) +
          (static_cast<std::int64_t>(parent[axis]) - mapping_.coarse_origin[axis]) * ratio_[axis] +
          child[axis]);
    }
    return fine;
  }

  POPS_HD Real coarse_divergence_(const Index<Dim>& parent, int component) const {
    Real divergence = Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> upper = parent;
      ++upper[axis];
      divergence += source_[axis](upper, component) - source_[axis](parent, component);
    }
    return divergence;
  }

  POPS_HD Real base_fine_divergence_(const Index<Dim>& parent, const Index<Dim>& child,
                                     int component) const {
    const Index<Dim> fine = fine_cell_(parent, child);
    Real divergence = Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> upper = fine;
      ++upper[axis];
      divergence += static_cast<Real>(ratio_[axis]) * (base_face_value_(axis, upper, component) -
                                                       base_face_value_(axis, fine, component));
    }
    return divergence;
  }

  POPS_HD Real divergence_error_(const Index<Dim>& parent, const Index<Dim>& child,
                                 int component) const {
    return coarse_divergence_(parent, component) - base_fine_divergence_(parent, child, component);
  }

  POPS_HD Real divergence_correction_(int normal_axis, const Index<Dim>& fine_face,
                                      int component) const {
    Index<Dim> parent{};
    Index<Dim> child{};
    parent_and_child_(fine_face, parent, child);
    const int normal_child = child[normal_axis];
    if (normal_child == 0)
      return Real(0);
    for (int axis = 0; axis < normal_axis; ++axis)
      if (child[axis] != ratio_[axis] - 1)
        return Real(0);

    std::uint64_t sample_count = static_cast<std::uint64_t>(normal_child);
    for (int axis = 0; axis < normal_axis; ++axis)
      sample_count *= static_cast<std::uint64_t>(ratio_[axis]);
    Real prefix_error = Real(0);
    for (std::uint64_t ordinal = 0; ordinal < sample_count; ++ordinal) {
      std::uint64_t remainder = ordinal;
      Index<Dim> sample = child;
      for (int axis = 0; axis < normal_axis; ++axis) {
        sample[axis] = static_cast<int>(remainder % static_cast<std::uint64_t>(ratio_[axis]));
        remainder /= static_cast<std::uint64_t>(ratio_[axis]);
      }
      sample[normal_axis] = static_cast<int>(remainder);
      prefix_error += divergence_error_(parent, sample, component);
    }
    return prefix_error / static_cast<Real>(ratio_[normal_axis]);
  }

  RefinementRatio<Dim> ratio_;
  IndexMapping<Dim> mapping_{};
  ComponentRange components_{};
  std::array<FieldView<const Real, Dim>, Dim> source_{};
  std::array<FieldView<Real, Dim>, Dim> destination_{};
  Box<Dim> destination_cell_region_{};
  std::array<Box<Dim>, Dim> destination_face_regions_{};
};

template <int Dim>
class DivergencePreservingFaceTransferProvider {
 public:
  static_assert(Dim >= 1 && Dim <= 3,
                "DivergencePreservingFaceTransferProvider supports dimensions 1, 2, and 3");

  static constexpr TransferCapabilities capabilities() {
    return {2, 1, true, true, SlopeLimiter::MonotonizedCentral};
  }

  PreparedDivergencePreservingFaceTransfer<Dim> prepare(
      std::array<FieldView<const Real, Dim>, Dim> source,
      std::array<FieldView<Real, Dim>, Dim> destination, const Box<Dim>& destination_cell_region,
      RefinementRatio<Dim> ratio, IndexMapping<Dim> mapping = {},
      ComponentRange components = {}) const {
    if (!ratio.refines_any_axis())
      throw std::invalid_argument(
          "prepared divergence-preserving face transfer requires a non-identity ratio");
    if (destination_cell_region.empty())
      throw std::invalid_argument(
          "prepared divergence-preserving face transfer requires a non-empty cell region");

    std::array<detail::ValidatedView<Dim>, Dim> source_views{};
    std::array<detail::ValidatedView<Dim>, Dim> destination_views{};
    std::array<Box<Dim>, Dim> destination_face_regions{};
    for (int normal_axis = 0; normal_axis < Dim; ++normal_axis) {
      source_views[normal_axis] = detail::validate_view(source[normal_axis]);
      destination_views[normal_axis] = detail::validate_view(destination[normal_axis]);
      detail::validate_components(source[normal_axis], destination[normal_axis], components);
      destination_face_regions[normal_axis] =
          detail::face_region(destination_cell_region, normal_axis);
      if (!destination_views[normal_axis].box.contains(destination_face_regions[normal_axis]))
        throw std::invalid_argument(
            "prepared divergence-preserving face destination omits required oriented faces");
      const Box<Dim> required_source = detail::face_interpolation_source_box(
          destination_face_regions[normal_axis], normal_axis, ratio, mapping);
      if (!source_views[normal_axis].box.contains(required_source))
        throw std::invalid_argument(
            "prepared divergence-preserving face source omits a transverse stencil");
    }
    for (int destination_axis = 0; destination_axis < Dim; ++destination_axis) {
      for (int source_axis = 0; source_axis < Dim; ++source_axis)
        if (source_views[source_axis].begin < destination_views[destination_axis].end &&
            destination_views[destination_axis].begin < source_views[source_axis].end)
          throw std::invalid_argument(
              "prepared divergence-preserving face transfer requires candidate storage");
      for (int other_axis = destination_axis + 1; other_axis < Dim; ++other_axis)
        if (destination_views[other_axis].begin < destination_views[destination_axis].end &&
            destination_views[destination_axis].begin < destination_views[other_axis].end)
          throw std::invalid_argument(
              "prepared divergence-preserving face destinations must not overlap");
    }
    return PreparedDivergencePreservingFaceTransfer<Dim>(ratio, mapping, components, source,
                                                         destination, destination_cell_region,
                                                         destination_face_regions);
  }
};

}  // namespace pops::amr::transfer
