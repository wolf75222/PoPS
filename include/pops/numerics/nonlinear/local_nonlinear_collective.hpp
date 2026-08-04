#pragma once

/// @file
/// @brief Exact collective selection of the first failed cell of a local nonlinear solve.
///
/// Failure diagnostics must preserve arbitrary signed `Box2D` indices. Packing two coordinates,
/// one component and the failure priority into a binary64 value cannot provide that contract: the
/// mantissa is too small, and negative coordinates can even escape the intended priority bin.
/// This helper instead performs exact staged reductions: priority is selected by the caller, then
/// minimum `j`, minimum `i` at that `j`, and minimum component at that exact cell. Integer Kokkos
/// reducers and integer MPI collectives preserve the full iterable `Box2D` index range independently
/// of the configured floating-point precision.

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>

#include <limits>
#include <stdexcept>

namespace pops {

struct LocalNonlinearFailureLocation {
  int priority = 0;
  int i = -1;
  int j = -1;
  int component = -1;
  bool found = false;
};

namespace detail {

struct LocalNonlinearFailurePresenceMax {
  ConstArray4 values;
  int priority = 0;
  int priority_component = 0;

  POPS_HD void operator()(int i, int j, int& result) const {
    if (static_cast<int>(values(i, j, priority_component)) == priority)
      result = 1;
  }
};

struct LocalNonlinearFailureJMin {
  ConstArray4 values;
  int priority = 0;
  int priority_component = 0;

  POPS_HD void operator()(int i, int j, int& result) const {
    if (static_cast<int>(values(i, j, priority_component)) == priority && j < result)
      result = j;
  }
};

struct LocalNonlinearFailureIMin {
  ConstArray4 values;
  int priority = 0;
  int priority_component = 0;
  int selected_j = 0;

  POPS_HD void operator()(int i, int j, int& result) const {
    if (j == selected_j && static_cast<int>(values(i, j, priority_component)) == priority &&
        i < result)
      result = i;
  }
};

struct LocalNonlinearFailureComponentMin {
  ConstArray4 values;
  int priority = 0;
  int priority_component = 0;
  int component_component = 0;
  int selected_i = 0;
  int selected_j = 0;

  POPS_HD void operator()(int i, int j, int& result) const {
    if (i == selected_i && j == selected_j &&
        static_cast<int>(values(i, j, priority_component)) == priority) {
      const int component = static_cast<int>(values(i, j, component_component));
      if (component < result)
        result = component;
    }
  }
};

template <class Reducer>
inline int local_nonlinear_failure_min(const MultiFab& statistics, Reducer reducer) {
  int local = std::numeric_limits<int>::max();
  for (int local_index = 0; local_index < statistics.local_size(); ++local_index) {
    reducer.values = statistics.fab(local_index).const_array();
    const Box2D box = statistics.box(local_index);
    if (box.empty())
      continue;
    require_iterable_box(box);
    ensure_kokkos_initialized();
    int selected = 0;
    Kokkos::parallel_reduce("pops_local_nonlinear_failure_min",
                            Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
                                {box.lo[0], box.lo[1]}, {box.hi[0] + 1, box.hi[1] + 1}),
                            reducer, Kokkos::Min<int>{selected});
    if (selected < local)
      local = selected;
  }
  return static_cast<int>(all_reduce_min(static_cast<long>(local)));
}

inline bool local_nonlinear_failure_exists(const MultiFab& statistics, int priority,
                                           int priority_component) {
  int local = 0;
  for (int local_index = 0; local_index < statistics.local_size(); ++local_index) {
    const Box2D box = statistics.box(local_index);
    if (box.empty())
      continue;
    require_iterable_box(box);
    ensure_kokkos_initialized();
    int found = 0;
    Kokkos::parallel_reduce(
        "pops_local_nonlinear_failure_presence",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
            {box.lo[0], box.lo[1]}, {box.hi[0] + 1, box.hi[1] + 1}),
        LocalNonlinearFailurePresenceMax{statistics.fab(local_index).const_array(), priority,
                                         priority_component},
        Kokkos::Max<int>{found});
    if (found != 0)
      local = 1;
  }
  return all_reduce_max(static_cast<long>(local)) != 0;
}

}  // namespace detail

/// Select the lexicographically first `(j, i, component)` carrying `priority` across all ranks.
/// `priority_component` and `component_component` identify scalar statistics components written by
/// the device kernel. A positive priority must occur at least once; otherwise the collective fails
/// closed instead of fabricating a diagnostic location.
inline LocalNonlinearFailureLocation collective_first_local_nonlinear_failure(
    const MultiFab& statistics, int priority, int priority_component, int component_component) {
  if (priority <= 0)
    return {};
  if (priority_component < 0 || priority_component >= statistics.ncomp() ||
      component_component < 0 || component_component >= statistics.ncomp())
    throw std::invalid_argument("local nonlinear failure-statistics component is out of range");
  if (!detail::local_nonlinear_failure_exists(statistics, priority, priority_component))
    throw std::runtime_error("local nonlinear collective priority has no failing cell");

  const int selected_j = detail::local_nonlinear_failure_min(
      statistics, detail::LocalNonlinearFailureJMin{{}, priority, priority_component});

  const int selected_i = detail::local_nonlinear_failure_min(
      statistics, detail::LocalNonlinearFailureIMin{{}, priority, priority_component, selected_j});

  const int selected_component = detail::local_nonlinear_failure_min(
      statistics,
      detail::LocalNonlinearFailureComponentMin{
          {}, priority, priority_component, component_component, selected_i, selected_j});

  return {priority, selected_i, selected_j, selected_component, true};
}

}  // namespace pops
