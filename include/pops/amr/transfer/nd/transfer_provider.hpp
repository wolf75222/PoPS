/// @file
/// @brief Allocation-free prepared AMR restriction and interpolation in 1D, 2D, and 3D.

#pragma once

#include <pops/amr/transfer/nd/refinement_ratio.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/storage/field_view.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace pops::amr::transfer::nd {

/// Logical location of transferred values.  The first ND substrate intentionally authenticates
/// only cell-centered transfers; the other values make unsupported routes explicit and fail-closed.
enum class Centering : unsigned char { Cell = 0, Node = 1, Face0 = 2, Face1 = 3, Face2 = 4 };

enum class TransferKind : unsigned char {
  ConservativeRestriction = 0,
  LinearProlongation = 1,
  CoarseFineGhostInterpolation = 2,
};

struct TransferCapabilities {
  int interpolation_order = 0;
  int source_stencil_radius = 0;
  bool conservative = false;
  bool allocation_free_hot_path = false;

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
                                  const IndexMapping<Dim>& mapping) {
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
    if (ratio[axis] > 1) {
      --lower;
      ++upper;
    }
    result.lo[axis] = checked_transfer_index(
        lower, "prepared ND interpolation lower stencil exceeds signed coordinates");
    result.hi[axis] = checked_transfer_index(
        upper, "prepared ND interpolation upper stencil exceeds signed coordinates");
  }
  return result;
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
    else
      interpolate_cell(destination_index);
  }

  POPS_HD TransferKind kind() const { return kind_; }
  POPS_HD const Box<Dim>& destination_region() const { return destination_region_; }
  POPS_HD const RefinementRatio<Dim>& refinement_ratio() const { return ratio_; }
  POPS_HD ComponentRange components() const { return components_; }

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
        const Real slope =
            Real(0.5) * (source_(upper, source_component) - source_(lower, source_component));
        const std::int64_t offset_numerator = std::int64_t{2} * child[axis] + 1 - ratio_[axis];
        const std::int64_t offset_denominator = std::int64_t{2} * ratio_[axis];
        value +=
            slope * static_cast<Real>(offset_numerator) / static_cast<Real>(offset_denominator);
      }
      destination_(fine, destination_component) = value;
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

  static constexpr TransferProvider coarse_fine_ghost_interpolation() {
    return TransferProvider(TransferKind::CoarseFineGhostInterpolation);
  }

  TransferCapabilities capabilities() const {
    require_supported_route();
    if (kind_ == TransferKind::ConservativeRestriction)
      return {1, 0, true, true};
    return {2, 1, false, true};
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

    const Box<Dim> required_source =
        kind_ == TransferKind::ConservativeRestriction
            ? detail::refined_source_box(destination_region, ratio, mapping)
            : detail::interpolation_source_box(destination_region, ratio, mapping);
    if (!source_view.box.contains(required_source))
      throw std::invalid_argument(
          "prepared ND transfer source FieldView does not contain the complete stencil");

    return PreparedTransfer<Dim>(kind_, ratio, mapping, components, source, destination,
                                 destination_region);
  }

 private:
  void require_supported_route() const {
    if constexpr (Center != Centering::Cell)
      throw std::invalid_argument(
          "ND transfer provider currently authenticates only cell-centered fields");
    switch (kind_) {
      case TransferKind::ConservativeRestriction:
      case TransferKind::LinearProlongation:
      case TransferKind::CoarseFineGhostInterpolation:
        return;
    }
    throw std::invalid_argument("ND transfer provider identity is not registered");
  }

  TransferKind kind_;
};

static_assert(std::is_trivially_copyable_v<PreparedTransfer<1>>);
static_assert(std::is_trivially_copyable_v<PreparedTransfer<2>>);
static_assert(std::is_trivially_copyable_v<PreparedTransfer<3>>);
static_assert(std::is_trivially_copyable_v<TransferProvider<1, Centering::Cell>>);
static_assert(std::is_trivially_copyable_v<TransferProvider<2, Centering::Cell>>);
static_assert(std::is_trivially_copyable_v<TransferProvider<3, Centering::Cell>>);

}  // namespace pops::amr::transfer::nd
