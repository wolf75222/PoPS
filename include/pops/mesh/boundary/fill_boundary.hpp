/// @file
/// @brief Exact local replay for authenticated compile-time-ranked halo schedules.

#pragma once

#include <pops/mesh/boundary/halo_exchange.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <Kokkos_Core.hpp>

#include <stdexcept>

namespace pops {

namespace fill_boundary_detail {

template <int Dim>
struct CopyShiftedKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> source{};
  Index<Dim> source_from_destination{};
  int component_count = 0;

  POPS_HD void operator()(const Index<Dim>& destination_index) const {
    Index<Dim> source_index = destination_index;
    for (int axis = 0; axis < Dim; ++axis)
      source_index[axis] += source_from_destination[axis];
    for (int component = 0; component < component_count; ++component)
      destination(destination_index, component) = source(source_index, component);
  }
};

template <int Dim, class MemorySpace>
void replay_local_job(MultiFab<Dim, MemorySpace>& fields, const HaloJob<Dim>& job) {
  auto& destination = fields.fab_global(job.destination_box);
  const auto& source = fields.fab_global(job.source_box);
  for_each_cell(job.destination_region,
                CopyShiftedKernel<Dim>{destination.view(), source.view(),
                                       job.source_from_destination, fields.ncomp()});
}

}  // namespace fill_boundary_detail

/// Replay one prepared schedule.  Remote work is rejected before any local kernel is submitted, so
/// callers never observe a partially filled field when production ND transport is unavailable.
template <int Dim, class MemorySpace>
void fill_boundary(MultiFab<Dim, MemorySpace>& fields, const HaloSchedule<Dim>& schedule) {
  schedule.authenticate(fields);
  schedule.require_local_execution();
  for (const HaloJob<Dim>& job : schedule.local_jobs())
    fill_boundary_detail::replay_local_job(fields, job);
  Kokkos::fence();
}

/// Execute one prepared asynchronous transport synchronously.  Callers that overlap interior work
/// invoke `exchange.begin(fields, lane)` and `exchange.complete(fields, lane)` directly.
template <int Dim, class MemorySpace>
void fill_boundary(MultiFab<Dim, MemorySpace>& fields, HaloExchange<Dim, MemorySpace>& exchange,
                   const ExecutionLane& lane) {
  exchange.execute(fields, lane);
}

/// Materialize a one-shot prepared transport on an explicit owning lane.
template <int Dim, class MemorySpace>
void fill_boundary(MultiFab<Dim, MemorySpace>& fields, const HaloSchedule<Dim>& schedule,
                   const ExecutionLane& lane, HaloExchangeContext context) {
  HaloExchange<Dim, MemorySpace> exchange(schedule, lane, context);
  exchange.execute(fields, lane);
}

/// Prepare and synchronously replay an ordinary Cartesian (axis-translation) topology.
template <int Dim, class MemorySpace>
void fill_boundary(MultiFab<Dim, MemorySpace>& fields, const Box<Dim>& domain,
                   const BoundaryTopology<Dim>& topology, HaloScheduleBudget budget) {
  const HaloSchedule<Dim> schedule = prepare_halo_schedule(fields, domain, topology, budget);
  fill_boundary(fields, schedule);
}

/// Prepare and execute one exact-ranked distributed halo transport.
template <int Dim, class MemorySpace>
void fill_boundary(MultiFab<Dim, MemorySpace>& fields, const Box<Dim>& domain,
                   const BoundaryTopology<Dim>& topology, HaloScheduleBudget budget,
                   const ExecutionLane& lane, HaloExchangeContext context) {
  const HaloSchedule<Dim> schedule = prepare_halo_schedule(fields, domain, topology, budget);
  fill_boundary(fields, schedule, lane, context);
}

}  // namespace pops
