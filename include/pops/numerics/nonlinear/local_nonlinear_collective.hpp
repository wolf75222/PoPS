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

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>

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
                                       Reducer reducer) {
  detail::ensure_kokkos_initialized();
  LocalNonlinearMinimumView minimum("pops_local_nonlinear_failure_minimum");
  Kokkos::deep_copy(minimum, std::numeric_limits<int>::max());
  reducer.minimum = minimum;

  for (std::size_t local_index = 0; local_index < statistics.local_size(); ++local_index) {
    reducer.values = statistics.fab(local_index).view();
    for_each_cell(statistics.box(local_index), reducer);
  }

  int local = std::numeric_limits<int>::max();
  Kokkos::deep_copy(local, minimum);
  return static_cast<int>(all_reduce_min(static_cast<long>(local)));
}

}  // namespace detail

/// Select the first `(axis Dim-1, ..., axis 0, component)` carrying `priority` on all ranks.
/// `priority_component` and `component_component` identify scalar statistics components written by
/// the device kernel. A positive priority must occur at least once; otherwise the collective fails
/// closed instead of fabricating a diagnostic location.
template <int Dim, class MemorySpace>
inline LocalNonlinearFailureLocation<Dim> collective_first_local_nonlinear_failure(
    const MultiFab<Dim, MemorySpace>& statistics, int priority, int priority_component,
    int component_component) {
  if (priority <= 0)
    return {};
  if (priority_component < 0 || priority_component >= statistics.ncomp() ||
      component_component < 0 || component_component >= statistics.ncomp())
    throw std::invalid_argument("local nonlinear failure-statistics component is out of range");

  Index<Dim> selected{};
  for (int axis = Dim - 1; axis >= 0; --axis) {
    const int coordinate = detail::local_nonlinear_failure_min(
        statistics, detail::LocalNonlinearFailureAxisMin<Dim>{
                        {}, {}, priority, priority_component, axis, selected});
    if (coordinate == std::numeric_limits<int>::max())
      throw std::runtime_error("local nonlinear collective priority has no failing cell");
    selected[axis] = coordinate;
  }

  const int component = detail::local_nonlinear_failure_min(
      statistics, detail::LocalNonlinearFailureComponentMin<Dim>{
                      {}, {}, priority, priority_component, component_component, selected});
  if (component == std::numeric_limits<int>::max())
    throw std::runtime_error("local nonlinear collective selected cell has no component");

  return {priority, selected, component, true};
}

}  // namespace pops
