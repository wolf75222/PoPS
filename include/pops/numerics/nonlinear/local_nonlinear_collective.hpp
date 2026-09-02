#pragma once

/// @file
/// @brief Exact collective selection of the first failed cell of a ranked local nonlinear solve.
///
/// Failure diagnostics must preserve arbitrary signed `Index<Dim>` coordinates. Packing the
/// coordinates, one component, and the failure priority into a binary floating-point value cannot
/// provide that contract. This helper instead performs exact staged integer reductions: the caller
/// selects the priority, each axis is selected in reverse lexicographic order, and the component is
/// selected at that exact index. The single axis loop is shared by dimensions 1, 2, and 3.

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace pops {

template <int Dim>
struct LocalNonlinearFailureLocation {
  int priority = 0;
  Index<Dim> index{};
  int component = -1;
  bool found = false;
};

namespace detail {

using LocalNonlinearMinimumView =
    Kokkos::View<int, typename Kokkos::DefaultExecutionSpace::memory_space>;

/// Persistent scalar slots for one exact failure-location reduction.  A prepared implicit-source
/// invocation consumes axes [0, Dim) and component [Dim], so it never materializes a scalar View
/// while reporting a rejected or failed candidate.
template <int Dim>
using LocalNonlinearFailureBuffer =
    Kokkos::View<int[Dim + 1], typename Kokkos::DefaultExecutionSpace::memory_space>;

template <int Dim, class MemorySpace, class Functor>
inline void local_nonlinear_failure_for_each(const Box<Dim>& box, const Functor& functor) {
  if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
    const Extent<Dim> extent = box.extent();
    for (std::int64_t ordinal = 0; ordinal < box.numPts(); ++ordinal) {
      std::int64_t remainder = ordinal;
      Index<Dim> index{};
      for (int axis = 0; axis < Dim; ++axis) {
        index[axis] = box.lo[axis] + static_cast<int>(remainder % extent[axis]);
        remainder /= extent[axis];
      }
      functor(index);
    }
  } else {
    for_each_cell(box, functor);
  }
}

template <int Dim>
struct LocalNonlinearFailureAxisMin {
  FieldView<const Real, Dim> values{};
  LocalNonlinearMinimumView minimum{};
  int priority = 0;
  int priority_component = 0;
  int axis = 0;
  Index<Dim> selected{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    bool matches = static_cast<int>(values(index, priority_component)) == priority;
    for (int selected_axis = axis + 1; selected_axis < Dim; ++selected_axis)
      matches = matches && index[selected_axis] == selected[selected_axis];
    if (matches)
      Kokkos::atomic_min(&minimum(), index[axis]);
  }
};

template <int Dim>
struct LocalNonlinearFailureComponentMin {
  FieldView<const Real, Dim> values{};
  LocalNonlinearMinimumView minimum{};
  int priority = 0;
  int priority_component = 0;
  int component_component = 0;
  Index<Dim> selected{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    bool matches = static_cast<int>(values(index, priority_component)) == priority;
    for (int axis = 0; axis < Dim; ++axis)
      matches = matches && index[axis] == selected[axis];
    if (matches)
      Kokkos::atomic_min(&minimum(), static_cast<int>(values(index, component_component)));
  }
};

template <int Dim, class MemorySpace, class Reducer>
inline int local_nonlinear_failure_min(const MultiFab<Dim, MemorySpace>& statistics,
                                       Reducer reducer, const ExecutionLane& lane) {
  detail::ensure_kokkos_initialized();
  LocalNonlinearMinimumView minimum("pops_local_nonlinear_failure_minimum");
  Kokkos::deep_copy(minimum, std::numeric_limits<int>::max());
  reducer.minimum = minimum;

  for (std::size_t local_index = 0; local_index < statistics.local_size(); ++local_index) {
    reducer.values = statistics.fab(local_index).view();
    local_nonlinear_failure_for_each<Dim, MemorySpace>(statistics.box(local_index), reducer);
  }

  int local = std::numeric_limits<int>::max();
  Kokkos::deep_copy(local, minimum);
  return static_cast<int>(all_reduce_min(static_cast<long>(local), lane));
}

template <int Dim, class MemorySpace, class Reducer>
inline int local_nonlinear_failure_min(const MultiFab<Dim, MemorySpace>& statistics,
                                       LocalNonlinearFailureBuffer<Dim>& buffer, int slot,
                                       Reducer reducer, const ExecutionLane& lane) {
  if (slot < 0 || slot > Dim)
    throw std::out_of_range("local nonlinear failure-buffer slot is out of range");
  detail::ensure_kokkos_initialized();
  const LocalNonlinearMinimumView minimum = Kokkos::subview(buffer, slot);
  if constexpr (std::is_same_v<typename Kokkos::DefaultExecutionSpace::memory_space,
                               Kokkos::HostSpace>)
    minimum() = std::numeric_limits<int>::max();
  else
    Kokkos::deep_copy(minimum, std::numeric_limits<int>::max());
  reducer.minimum = minimum;

  for (std::size_t local_index = 0; local_index < statistics.local_size(); ++local_index) {
    reducer.values = statistics.fab(local_index).view();
    local_nonlinear_failure_for_each<Dim, MemorySpace>(statistics.box(local_index), reducer);
  }

  int local = std::numeric_limits<int>::max();
  if constexpr (std::is_same_v<typename Kokkos::DefaultExecutionSpace::memory_space,
                               Kokkos::HostSpace>)
    local = minimum();
  else
    Kokkos::deep_copy(local, minimum);
  return static_cast<int>(all_reduce_min(static_cast<long>(local), lane));
}

}  // namespace detail

/// Select the first `(axis Dim-1, ..., axis 0, component)` carrying `priority` on all ranks.
/// `priority_component` and `component_component` identify scalar statistics components written by
/// the device kernel. A positive priority must occur at least once; otherwise the collective fails
/// closed instead of fabricating a diagnostic location.
template <int Dim, class MemorySpace>
inline LocalNonlinearFailureLocation<Dim> collective_first_local_nonlinear_failure(
    const MultiFab<Dim, MemorySpace>& statistics, int priority, int priority_component,
    int component_component, const ExecutionLane& lane) {
  if (priority <= 0)
    return {};
  if (priority_component < 0 || priority_component >= statistics.ncomp() ||
      component_component < 0 || component_component >= statistics.ncomp())
    throw std::invalid_argument("local nonlinear failure-statistics component is out of range");

  Index<Dim> selected{};
  for (int axis = Dim - 1; axis >= 0; --axis) {
    const int coordinate = detail::local_nonlinear_failure_min(
        statistics,
        detail::LocalNonlinearFailureAxisMin<Dim>{
            {}, {}, priority, priority_component, axis, selected},
        lane);
    if (coordinate == std::numeric_limits<int>::max())
      throw std::runtime_error("local nonlinear collective priority has no failing cell");
    selected[axis] = coordinate;
  }

  const int component = detail::local_nonlinear_failure_min(
      statistics,
      detail::LocalNonlinearFailureComponentMin<Dim>{
          {}, {}, priority, priority_component, component_component, selected},
      lane);
  if (component == std::numeric_limits<int>::max())
    throw std::runtime_error("local nonlinear collective selected cell has no component");

  return {priority, selected, component, true};
}

/// Allocation-free prepared variant using the caller-owned Dim+1 scalar reduction buffer.
template <int Dim, class MemorySpace>
inline LocalNonlinearFailureLocation<Dim> collective_first_local_nonlinear_failure(
    const MultiFab<Dim, MemorySpace>& statistics, detail::LocalNonlinearFailureBuffer<Dim>& buffer,
    int priority, int priority_component, int component_component, const ExecutionLane& lane) {
  if (priority <= 0)
    return {};
  if (priority_component < 0 || priority_component >= statistics.ncomp() ||
      component_component < 0 || component_component >= statistics.ncomp())
    throw std::invalid_argument("local nonlinear failure-statistics component is out of range");

  Index<Dim> selected{};
  for (int axis = Dim - 1; axis >= 0; --axis) {
    const int coordinate = detail::local_nonlinear_failure_min(
        statistics, buffer, axis,
        detail::LocalNonlinearFailureAxisMin<Dim>{
            {}, {}, priority, priority_component, axis, selected},
        lane);
    if (coordinate == std::numeric_limits<int>::max())
      throw std::runtime_error("local nonlinear collective priority has no failing cell");
    selected[axis] = coordinate;
  }

  const int component = detail::local_nonlinear_failure_min(
      statistics, buffer, Dim,
      detail::LocalNonlinearFailureComponentMin<Dim>{
          {}, {}, priority, priority_component, component_component, selected},
      lane);
  if (component == std::numeric_limits<int>::max())
    throw std::runtime_error("local nonlinear collective selected cell has no component");

  return {priority, selected, component, true};
}

}  // namespace pops
