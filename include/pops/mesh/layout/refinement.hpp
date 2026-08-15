/// @file
/// @brief Compile-time-ranked layout redistribution and piecewise-constant AMR transfer.

#pragma once

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/copy_schedule.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {

namespace refinement_detail {

inline int checked_index(std::int64_t value, const char* operation) {
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    throw std::overflow_error(operation);
  return static_cast<int>(value);
}

template <int Dim>
void validate_ratio(const Extent<Dim>& ratio, const char* operation) {
  for (int axis = 0; axis < Dim; ++axis)
    if (ratio[axis] <= 0 || ratio[axis] > std::numeric_limits<int>::max())
      throw std::invalid_argument(std::string(operation) +
                                  ": refinement ratios must be positive native indices");
}

template <int Dim>
Extent<Dim> isotropic_ratio(int ratio, const char* operation) {
  if (ratio <= 0)
    throw std::invalid_argument(std::string(operation) +
                                ": refinement ratio must be strictly positive");
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = ratio;
  return result;
}

template <int Dim>
std::int64_t child_count(const Extent<Dim>& ratio, const char* operation) {
  validate_ratio(ratio, operation);
  std::int64_t result = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    if (result > std::numeric_limits<std::int64_t>::max() / ratio[axis])
      throw std::overflow_error(std::string(operation) + ": child count exceeds int64_t");
    result *= ratio[axis];
  }
  return result;
}

template <int Dim>
mesh::Distribution<Dim> rebind_distribution(const mesh::BoxArray<Dim>& layout,
                                            const mesh::Distribution<Dim>& model) {
  if (layout.size() != model.box_count())
    throw std::invalid_argument(
        "pops::refinement distribution rebind requires the same global patch count");
  if (model.replicated())
    return mesh::Distribution<Dim>::replicated(layout, model.rank_space());
  return mesh::Distribution<Dim>::partitioned(layout, model.rank_space(), model.owners());
}

template <int Dim>
struct CopyKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> source{};
  int destination_component = 0;
  int source_component = 0;
  int component_count = 0;

  POPS_HD void operator()(const Index<Dim>& index) const {
    for (int component = 0; component < component_count; ++component)
      destination(index, destination_component + component) =
          source(index, source_component + component);
  }
};

template <int Dim>
struct AverageDownKernel {
  FieldView<Real, Dim> coarse{};
  FieldView<const Real, Dim> fine{};
  Extent<Dim> ratio{};
  std::int64_t children = 0;
  int component_count = 0;

  POPS_HD void operator()(const Index<Dim>& coarse_index) const {
    for (int component = 0; component < component_count; ++component) {
      Real sum = 0;
      for (std::int64_t ordinal = 0; ordinal < children; ++ordinal) {
        std::int64_t remainder = ordinal;
        Index<Dim> fine_index{};
        for (int axis = 0; axis < Dim; ++axis) {
          const std::int64_t offset = remainder % ratio[axis];
          remainder /= ratio[axis];
          fine_index[axis] = static_cast<int>(
              static_cast<std::int64_t>(coarse_index[axis]) * ratio[axis] + offset);
        }
        sum += fine(fine_index, component);
      }
      coarse(coarse_index, component) = sum / static_cast<Real>(children);
    }
  }
};

template <int Dim>
struct InterpolateKernel {
  FieldView<Real, Dim> fine{};
  FieldView<const Real, Dim> coarse{};
  Extent<Dim> ratio{};
  int component_count = 0;

  POPS_HD void operator()(const Index<Dim>& fine_index) const {
    Index<Dim> coarse_index{};
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t coordinate = fine_index[axis];
      std::int64_t quotient = coordinate / ratio[axis];
      if (coordinate % ratio[axis] < 0)
        --quotient;
      coarse_index[axis] = static_cast<int>(quotient);
    }
    for (int component = 0; component < component_count; ++component)
      fine(fine_index, component) = coarse(coarse_index, component);
  }
};

template <int Dim, class MemorySpace>
void validate_transfer_fields(const MultiFab<Dim, MemorySpace>& fine,
                              const MultiFab<Dim, MemorySpace>& coarse, const Extent<Dim>& ratio,
                              const char* operation) {
  validate_ratio(ratio, operation);
  if (fine.ncomp() != coarse.ncomp())
    throw std::invalid_argument(std::string(operation) +
                                ": fine and coarse component counts differ");
  if (fine.rank_space() != coarse.rank_space() || fine.local_rank() != coarse.local_rank())
    throw std::invalid_argument(std::string(operation) +
                                ": fine and coarse rank identities differ");
}

}  // namespace refinement_detail

template <int Dim>
Box<Dim> coarsen(const Box<Dim>& box, const Extent<Dim>& ratio) {
  refinement_detail::validate_ratio(ratio, "pops::coarsen");
  if (box.empty())
    return box;
  Box<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    const int divisor = static_cast<int>(ratio[axis]);
    result.lo[axis] = detail::floor_div_index(box.lo[axis], divisor);
    result.hi[axis] = detail::floor_div_index(box.hi[axis], divisor);
  }
  return result;
}

template <int Dim>
Box<Dim> coarsen(const Box<Dim>& box, int ratio) {
  return coarsen(box, refinement_detail::isotropic_ratio<Dim>(ratio, "pops::coarsen"));
}

template <int Dim>
Box<Dim> refine(const Box<Dim>& box, const Extent<Dim>& ratio) {
  refinement_detail::validate_ratio(ratio, "pops::refine");
  if (box.empty())
    return box;
  Box<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    result.lo[axis] =
        refinement_detail::checked_index(static_cast<std::int64_t>(box.lo[axis]) * ratio[axis],
                                         "pops::refine lower bound exceeds Index range");
    result.hi[axis] = refinement_detail::checked_index(
        static_cast<std::int64_t>(box.hi[axis]) * ratio[axis] + ratio[axis] - 1,
        "pops::refine upper bound exceeds Index range");
  }
  return result;
}

template <int Dim>
Box<Dim> refine(const Box<Dim>& box, int ratio) {
  return refine(box, refinement_detail::isotropic_ratio<Dim>(ratio, "pops::refine"));
}

template <int Dim>
mesh::BoxArray<Dim> coarsen(const mesh::BoxArray<Dim>& layout, const Extent<Dim>& ratio) {
  std::vector<Box<Dim>> boxes;
  boxes.reserve(layout.size());
  for (const Box<Dim>& box : layout.boxes())
    boxes.push_back(coarsen(box, ratio));
  return mesh::BoxArray<Dim>{std::move(boxes)};
}

template <int Dim>
mesh::BoxArray<Dim> coarsen(const mesh::BoxArray<Dim>& layout, int ratio) {
  return coarsen(layout, refinement_detail::isotropic_ratio<Dim>(ratio, "pops::coarsen(layout)"));
}

template <int Dim>
mesh::BoxArray<Dim> refine(const mesh::BoxArray<Dim>& layout, const Extent<Dim>& ratio) {
  std::vector<Box<Dim>> boxes;
  boxes.reserve(layout.size());
  for (const Box<Dim>& box : layout.boxes())
    boxes.push_back(refine(box, ratio));
  return mesh::BoxArray<Dim>{std::move(boxes)};
}

template <int Dim>
mesh::BoxArray<Dim> refine(const mesh::BoxArray<Dim>& layout, int ratio) {
  return refine(layout, refinement_detail::isotropic_ratio<Dim>(ratio, "pops::refine(layout)"));
}

/// Replay an authenticated exact-overlap schedule.  Remote plans reject before any destination
/// kernel is submitted.
template <int Dim, class DestinationMemorySpace, class SourceMemorySpace>
void parallel_copy(MultiFab<Dim, DestinationMemorySpace>& destination,
                   const MultiFab<Dim, SourceMemorySpace>& source,
                   const CopySchedule<Dim>& schedule, int destination_component,
                   int source_component, int component_count) {
  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, DestinationMemorySpace>::accessible,
      "parallel_copy requires DefaultExecutionSpace access to destination MemorySpace");
  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, SourceMemorySpace>::accessible,
      "parallel_copy requires DefaultExecutionSpace access to source MemorySpace");
  schedule.authenticate(destination, source);
  if (destination_component < 0 || source_component < 0 || component_count <= 0 ||
      destination_component > destination.ncomp() - component_count ||
      source_component > source.ncomp() - component_count)
    throw std::invalid_argument("pops::parallel_copy component range is invalid");
  schedule.require_local_execution();
  for (const CopyJob<Dim>& job : schedule.local_jobs()) {
    auto& destination_fab = destination.fab_global(job.destination_box);
    const auto& source_fab = source.fab_global(job.source_box);
    for_each_cell(job.region, refinement_detail::CopyKernel<Dim>{
                                  destination_fab.view(), source_fab.view(), destination_component,
                                  source_component, component_count});
  }
  Kokkos::fence();
}

template <int Dim, class DestinationMemorySpace, class SourceMemorySpace>
void parallel_copy(MultiFab<Dim, DestinationMemorySpace>& destination,
                   const MultiFab<Dim, SourceMemorySpace>& source,
                   const CopySchedule<Dim>& schedule) {
  if (destination.ncomp() != source.ncomp())
    throw std::invalid_argument("pops::parallel_copy fields have different component counts");
  parallel_copy(destination, source, schedule, 0, 0, destination.ncomp());
}

template <int Dim, class DestinationMemorySpace, class SourceMemorySpace>
void parallel_copy(MultiFab<Dim, DestinationMemorySpace>& destination,
                   const MultiFab<Dim, SourceMemorySpace>& source, CopyScheduleBudget budget) {
  if (destination.ncomp() != source.ncomp())
    throw std::invalid_argument("pops::parallel_copy fields have different component counts");
  const CopySchedule<Dim> schedule = prepare_copy_schedule(destination, source, budget);
  parallel_copy(destination, source, schedule);
}

template <int Dim, class MemorySpace>
void average_down(const MultiFab<Dim, MemorySpace>& fine, MultiFab<Dim, MemorySpace>& coarse,
                  const Extent<Dim>& ratio, CopyScheduleBudget copy_budget) {
  static_assert(Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, MemorySpace>::accessible,
                "average_down requires DefaultExecutionSpace access to MemorySpace");
  refinement_detail::validate_transfer_fields(fine, coarse, ratio, "pops::average_down");
  const std::int64_t children = refinement_detail::child_count(ratio, "pops::average_down");
  const mesh::BoxArray<Dim> scratch_layout = coarsen(fine.layout(), ratio);
  if (refine(scratch_layout, ratio) != fine.layout())
    throw std::invalid_argument(
        "pops::average_down requires fine patches aligned to the refinement ratio");
  const mesh::Distribution<Dim> scratch_distribution =
      refinement_detail::rebind_distribution(scratch_layout, fine.distribution());
  MultiFab<Dim, MemorySpace> scratch{scratch_layout, scratch_distribution, fine.local_rank(),
                                     fine.ncomp(), Extent<Dim>{}};
  const CopySchedule<Dim> schedule = prepare_copy_schedule(coarse, scratch, copy_budget);
  schedule.require_local_execution();

  for (std::size_t local = 0; local < fine.local_size(); ++local) {
    const std::size_t global = fine.global_index(local);
    auto& coarse_fab = scratch.fab_global(global);
    const auto& fine_fab = fine.fab(local);
    for_each_cell(coarse_fab.box(),
                  refinement_detail::AverageDownKernel<Dim>{coarse_fab.view(), fine_fab.view(),
                                                            ratio, children, fine.ncomp()});
  }
  Kokkos::fence();
  parallel_copy(coarse, scratch, schedule);
}

template <int Dim, class MemorySpace>
void average_down(const MultiFab<Dim, MemorySpace>& fine, MultiFab<Dim, MemorySpace>& coarse,
                  int ratio, CopyScheduleBudget copy_budget) {
  average_down(fine, coarse, refinement_detail::isotropic_ratio<Dim>(ratio, "pops::average_down"),
               copy_budget);
}

template <int Dim, class MemorySpace>
void interpolate(const MultiFab<Dim, MemorySpace>& coarse, MultiFab<Dim, MemorySpace>& fine,
                 const Extent<Dim>& ratio, CopyScheduleBudget copy_budget) {
  static_assert(Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, MemorySpace>::accessible,
                "interpolate requires DefaultExecutionSpace access to MemorySpace");
  refinement_detail::validate_transfer_fields(fine, coarse, ratio, "pops::interpolate");
  const mesh::BoxArray<Dim> scratch_layout = coarsen(fine.layout(), ratio);
  if (refine(scratch_layout, ratio) != fine.layout())
    throw std::invalid_argument(
        "pops::interpolate requires fine patches aligned to the refinement ratio");
  const mesh::Distribution<Dim> scratch_distribution =
      refinement_detail::rebind_distribution(scratch_layout, fine.distribution());
  MultiFab<Dim, MemorySpace> scratch{scratch_layout, scratch_distribution, fine.local_rank(),
                                     fine.ncomp(), Extent<Dim>{}};
  const CopySchedule<Dim> schedule = prepare_copy_schedule(scratch, coarse, copy_budget);
  schedule.require_local_execution();
  parallel_copy(scratch, coarse, schedule);

  for (std::size_t local = 0; local < fine.local_size(); ++local) {
    const std::size_t global = fine.global_index(local);
    auto& fine_fab = fine.fab(local);
    const auto& coarse_fab = scratch.fab_global(global);
    for_each_cell(fine_fab.box(), refinement_detail::InterpolateKernel<Dim>{
                                      fine_fab.view(), coarse_fab.view(), ratio, fine.ncomp()});
  }
  Kokkos::fence();
}

template <int Dim, class MemorySpace>
void interpolate(const MultiFab<Dim, MemorySpace>& coarse, MultiFab<Dim, MemorySpace>& fine,
                 int ratio, CopyScheduleBudget copy_budget) {
  interpolate(coarse, fine, refinement_detail::isotropic_ratio<Dim>(ratio, "pops::interpolate"),
              copy_budget);
}

}  // namespace pops
