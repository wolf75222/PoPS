/// @file
/// @brief Exact-mask composite reductions over compile-time-ranked AMR fields.

#pragma once

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>

namespace pops::runtime::amr {

/// One level participating in a composite reduction.
///
/// `active` is an authenticated one-component mask on exactly the same layout and ownership as
/// `values`: one selects a valid composite cell and zero hides a cell shadowed by a finer level.
/// Reconstructing coverage from boxes inside a reduction would create a second topology authority,
/// so the prepared AMR owner must supply this mask explicitly.
template <int Dim, class MemorySpace>
struct CompositeLevelView {
  const MultiFab<Dim, MemorySpace>* values = nullptr;
  const MultiFab<Dim, MemorySpace>* active = nullptr;
  std::array<Real, Dim> cell_extent{};
  /// Optional exact relative cell measure.  Staircase EB supplies its binary active mask and
  /// cut-cell EB supplies kappa.  Hierarchy coverage remains the independent binary `active`
  /// authority, so refinement shadowing and physical volume cannot overwrite each other.
  const MultiFab<Dim, MemorySpace>* relative_measure = nullptr;
};

enum class CompositeReductionKind : unsigned char {
  Sum,
  AbsoluteSum,
  SumSquares,
  Minimum,
  Maximum,
  AbsoluteMaximum,
};

struct CompositeReductionResult {
  Real value = Real(0);
  Real active_measure = Real(0);
};

namespace composite_detail {

template <int Dim>
struct InvalidMaskValue {
  FieldView<const Real, Dim> active{};

  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real value = active(index);
    return value == Real(0) || value == Real(1) ? Real(0) : Real(1);
  }
};

template <int Dim>
struct InvalidSelectedValue {
  FieldView<const Real, Dim> values{};
  FieldView<const Real, Dim> active{};
  FieldView<const Real, Dim> relative{};
  bool has_relative = false;
  int component = 0;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    if (active(index) == Real(0) || (has_relative && relative(index) == Real(0)))
      return Real(0);
    const Real value = values(index, component);
    return value <= std::numeric_limits<Real>::max() && value >= std::numeric_limits<Real>::lowest()
               ? Real(0)
               : Real(1);
  }
};

template <int Dim>
struct InvalidRelativeMeasure {
  FieldView<const Real, Dim> relative{};

  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real value = relative(index);
    return value >= Real(0) && value <= Real(1) && Kokkos::isfinite(value) ? Real(0) : Real(1);
  }
};

template <int Dim>
struct ActiveCell {
  FieldView<const Real, Dim> active{};
  FieldView<const Real, Dim> relative{};
  bool has_relative = false;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    return active(index) == Real(1) ? (has_relative ? relative(index) : Real(1)) : Real(0);
  }
};

template <int Dim>
struct SumValue {
  FieldView<const Real, Dim> values{};
  FieldView<const Real, Dim> active{};
  FieldView<const Real, Dim> relative{};
  bool has_relative = false;
  int component = 0;
  CompositeReductionKind kind = CompositeReductionKind::Sum;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    if (active(index) == Real(0) || (has_relative && relative(index) == Real(0)))
      return Real(0);
    const Real value = values(index, component);
    const Real weight = has_relative ? relative(index) : Real(1);
    if (kind == CompositeReductionKind::AbsoluteSum)
      return (value < Real(0) ? -value : value) * weight;
    if (kind == CompositeReductionKind::SumSquares)
      return value * value * weight;
    return value * weight;
  }
};

template <int Dim>
struct ExtremumValue {
  FieldView<const Real, Dim> values{};
  FieldView<const Real, Dim> active{};
  FieldView<const Real, Dim> relative{};
  bool has_relative = false;
  int component = 0;
  bool absolute = false;
  bool negate = false;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    if (active(index) == Real(0) || (has_relative && relative(index) == Real(0)))
      return std::numeric_limits<Real>::lowest();
    Real value = values(index, component);
    if (absolute && value < Real(0))
      value = -value;
    return negate ? -value : value;
  }
};

template <int Dim, class MemorySpace>
void require_level_shape(const CompositeLevelView<Dim, MemorySpace>& level, int component) {
  if (level.values == nullptr || level.active == nullptr)
    throw std::invalid_argument("composite reduction requires values and an active-cell mask");
  const auto& values = *level.values;
  const auto& active = *level.active;
  if (component < 0 || component >= values.ncomp())
    throw std::out_of_range("composite reduction component is outside the level field");
  if (active.ncomp() != 1 || values.layout() != active.layout() ||
      values.distribution() != active.distribution() ||
      values.local_rank() != active.local_rank() || values.local_size() != active.local_size())
    throw std::invalid_argument(
        "composite reduction mask does not authenticate the field layout and ownership");
  if (level.relative_measure != nullptr) {
    const auto& relative = *level.relative_measure;
    if (relative.ncomp() != 1 || values.layout() != relative.layout() ||
        values.distribution() != relative.distribution() ||
        values.local_rank() != relative.local_rank() ||
        values.local_size() != relative.local_size())
      throw std::invalid_argument(
          "composite reduction relative measure does not authenticate the field ownership");
  }
  for (int axis = 0; axis < Dim; ++axis)
    if (!(level.cell_extent[static_cast<std::size_t>(axis)] > Real(0)) ||
        !std::isfinite(level.cell_extent[static_cast<std::size_t>(axis)]))
      throw std::invalid_argument("composite reduction cell extents must be finite and positive");
}

template <int Dim, class MemorySpace>
bool contributes_on_this_rank(const MultiFab<Dim, MemorySpace>& values) {
  return !values.distribution().replicated() ||
         values.local_rank() == values.rank_space().coordinate(0);
}

template <std::size_t Dim>
Real cell_measure(const std::array<Real, Dim>& extent) {
  Real measure = Real(1);
  for (std::size_t axis = 0; axis < Dim; ++axis)
    measure *= extent[axis];
  return measure;
}

inline bool is_sum_kind(CompositeReductionKind kind) {
  return kind == CompositeReductionKind::Sum || kind == CompositeReductionKind::AbsoluteSum ||
         kind == CompositeReductionKind::SumSquares;
}

}  // namespace composite_detail

/// Reduce an exact prepared composite view on the supplied execution lane.
///
/// Replicated levels contribute from the canonical process coordinate only; distributed levels
/// contribute all local owners.  The final collective therefore counts every physical cell once.
template <int Dim, class MemorySpace>
CompositeReductionResult composite_reduce(
    std::span<const CompositeLevelView<Dim, MemorySpace>> levels, int component,
    CompositeReductionKind kind, const ExecutionLane& lane = ExecutionLane::world()) {
  if (levels.empty())
    throw std::invalid_argument("composite reduction requires at least one prepared level");

  Real local_selected =
      composite_detail::is_sum_kind(kind) ? Real(0) : std::numeric_limits<Real>::lowest();
  Real local_active_measure = Real(0);
  Real local_invalid = Real(0);

  for (const auto& level : levels) {
    composite_detail::require_level_shape(level, component);
    const auto& values = *level.values;
    const auto& active = *level.active;
    if (!composite_detail::contributes_on_this_rank(values))
      continue;

    Real level_active = Real(0);
    Real level_selected =
        composite_detail::is_sum_kind(kind) ? Real(0) : std::numeric_limits<Real>::lowest();
    for (std::size_t local = 0; local < values.local_size(); ++local) {
      const Box<Dim>& cells = values.box(local);
      const auto value_view = values.fab(local).view();
      const auto active_view = active.fab(local).view();
      const bool has_relative = level.relative_measure != nullptr;
      const auto relative_view =
          has_relative ? level.relative_measure->fab(local).view() : FieldView<const Real, Dim>{};
      local_invalid = std::max(
          local_invalid,
          for_each_cell_reduce_max(cells, composite_detail::InvalidMaskValue<Dim>{active_view}));
      local_invalid = std::max(
          local_invalid,
          for_each_cell_reduce_max(
              cells, composite_detail::InvalidSelectedValue<Dim>{
                         value_view, active_view, relative_view, has_relative, component}));
      if (has_relative)
        local_invalid =
            std::max(local_invalid,
                     for_each_cell_reduce_max(
                         cells, composite_detail::InvalidRelativeMeasure<Dim>{relative_view}));
      level_active += for_each_cell_reduce_sum(
          cells, composite_detail::ActiveCell<Dim>{active_view, relative_view, has_relative});

      if (composite_detail::is_sum_kind(kind)) {
        level_selected += for_each_cell_reduce_sum(
            cells, composite_detail::SumValue<Dim>{value_view, active_view, relative_view,
                                                   has_relative, component, kind});
      } else {
        const bool absolute = kind == CompositeReductionKind::AbsoluteMaximum;
        const bool negate = kind == CompositeReductionKind::Minimum;
        level_selected = std::max(
            level_selected,
            for_each_cell_reduce_max(cells, composite_detail::ExtremumValue<Dim>{
                                                value_view, active_view, relative_view,
                                                has_relative, component, absolute, negate}));
      }
    }

    const Real measure = composite_detail::cell_measure(level.cell_extent);
    local_active_measure += level_active * measure;
    if (composite_detail::is_sum_kind(kind))
      local_selected += level_selected * measure;
    else
      local_selected = std::max(local_selected, level_selected);
  }

  if (all_reduce_max(local_invalid, lane) != Real(0))
    throw std::runtime_error("composite reduction observed a non-binary mask or non-finite value");
  const Real global_active_measure = all_reduce_sum(local_active_measure, lane);
  if (!(global_active_measure > Real(0)) || !std::isfinite(global_active_measure))
    throw std::runtime_error("composite reduction has no finite positive active measure");

  Real global_value = Real(0);
  if (composite_detail::is_sum_kind(kind)) {
    global_value = all_reduce_sum(local_selected, lane);
  } else {
    global_value = all_reduce_max(local_selected, lane);
    if (kind == CompositeReductionKind::Minimum)
      global_value = -global_value;
  }
  if (!std::isfinite(global_value))
    throw std::runtime_error("composite reduction result is not finite");
  return {global_value, global_active_measure};
}

template <int Dim, class MemorySpace>
CompositeReductionResult composite_l2_norm(
    std::span<const CompositeLevelView<Dim, MemorySpace>> levels, int component,
    const ExecutionLane& lane = ExecutionLane::world()) {
  CompositeReductionResult result =
      composite_reduce(levels, component, CompositeReductionKind::SumSquares, lane);
  result.value = std::sqrt(result.value);
  return result;
}

}  // namespace pops::runtime::amr
