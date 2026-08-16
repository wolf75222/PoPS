/// @file
/// @brief Ranked AMR transfer execution authority.

#pragma once

#include <pops/amr/transfer/temporal_interpolation_provider.hpp>
#include <pops/amr/transfer/transfer_provider.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/parallel/region_transfer.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>

#include <cstddef>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace pops::numerics::time::amr {

namespace detail {

template <int Dim>
struct PreparedTransferKernel {
  ::pops::amr::transfer::PreparedTransfer<Dim> transfer;

  POPS_HD void operator()(const Index<Dim>& index) const { transfer(index); }
};

template <int Dim>
struct PreparedFaceTransferKernel {
  ::pops::amr::transfer::PreparedDivergencePreservingFaceTransfer<Dim> transfer;
  int normal_axis = 0;

  POPS_HD void operator()(const Index<Dim>& index) const { transfer(normal_axis, index); }
};

template <int Dim>
struct PreparedTemporalTransferKernel {
  ::pops::amr::transfer::PreparedLinearTemporalInterpolation<Dim> transfer;

  POPS_HD void operator()(const Index<Dim>& index) const { transfer(index); }
};

}  // namespace detail

/// Execute an already authenticated transfer on an explicit execution-space instance.
template <class ExecutionSpace, int Dim>
void execute_prepared_transfer(const ExecutionSpace& execution,
                               const ::pops::amr::transfer::PreparedTransfer<Dim>& prepared) {
  for_each_cell(execution, prepared.destination_region(),
                detail::PreparedTransferKernel<Dim>{prepared});
}

/// Execute an already authenticated transfer on the default execution-space instance.
template <int Dim>
void execute_prepared_transfer(const ::pops::amr::transfer::PreparedTransfer<Dim>& prepared) {
  for_each_cell(prepared.destination_region(), detail::PreparedTransferKernel<Dim>{prepared});
}

template <class ExecutionSpace, int Dim>
void execute_prepared_transfer(
    const ExecutionSpace& execution,
    const ::pops::amr::transfer::PreparedDivergencePreservingFaceTransfer<Dim>& prepared) {
  for (int normal_axis = 0; normal_axis < Dim; ++normal_axis)
    for_each_cell(execution, prepared.destination_face_region(normal_axis),
                  detail::PreparedFaceTransferKernel<Dim>{prepared, normal_axis});
}

template <int Dim>
void execute_prepared_transfer(
    const ::pops::amr::transfer::PreparedDivergencePreservingFaceTransfer<Dim>& prepared) {
  for (int normal_axis = 0; normal_axis < Dim; ++normal_axis)
    for_each_cell(prepared.destination_face_region(normal_axis),
                  detail::PreparedFaceTransferKernel<Dim>{prepared, normal_axis});
}

template <class ExecutionSpace, int Dim>
void execute_prepared_transfer(
    const ExecutionSpace& execution,
    const ::pops::amr::transfer::PreparedLinearTemporalInterpolation<Dim>& prepared) {
  for_each_cell(execution, prepared.destination_region(),
                detail::PreparedTemporalTransferKernel<Dim>{prepared});
}

template <int Dim>
void execute_prepared_transfer(
    const ::pops::amr::transfer::PreparedLinearTemporalInterpolation<Dim>& prepared) {
  for_each_cell(prepared.destination_region(),
                detail::PreparedTemporalTransferKernel<Dim>{prepared});
}

/// Conservatively restrict a complete distributed level onto its parent carrier.
/// Fine and parent distributions may have different patch boundaries and owners.
template <int Dim, class MemorySpace>
void execute_average_down_collectively(
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime, std::size_t fine_level,
    const MultiFab<Dim, MemorySpace>& fine, MultiFab<Dim, MemorySpace>& parent,
    const ExecutionLane& lane) {
  using field_type = MultiFab<Dim, MemorySpace>;
  using transfer_job = ::pops::mesh::parallel::RegionTransferJob<Dim>;
  using transfer_plan = ::pops::mesh::parallel::RegionTransferPlan<Dim>;
  using transport_type = ::pops::mesh::parallel::RegionTransport<Dim, MemorySpace>;

  if (fine_level == 0 || fine_level >= runtime.hierarchy().num_levels())
    throw std::out_of_range("AMR average-down level is outside the live hierarchy");
  if (fine.ncomp() != parent.ncomp() || fine.ncomp() < 1 ||
      fine.rank_space() != parent.rank_space() || fine.local_rank() != parent.local_rank())
    throw std::invalid_argument("AMR average-down carriers have incompatible exact contracts");

  const auto& authority = runtime.hierarchy().layout(fine_level).ratio_from_parent();

  std::vector<Fab<Dim, MemorySpace>> restricted;
  std::vector<std::size_t> restricted_by_global;
  std::vector<::pops::amr::transfer::PreparedTransfer<Dim>> transfers;
  std::vector<transfer_job> jobs;
  std::size_t element_budget = 0;
  std::exception_ptr preparation_error;
  try {
    restricted.reserve(fine.local_size());
    restricted_by_global.assign(fine.layout().size(), field_type::not_local);
    transfers.reserve(fine.local_size());
    for (std::size_t fine_local = 0; fine_local < fine.local_size(); ++fine_local) {
      const std::size_t fine_global = fine.global_index(fine_local);
      const Box<Dim> footprint =
          ::pops::amr::hierarchy::coarsen_box(fine.layout()[fine_global], authority);
      restricted_by_global[fine_global] = restricted.size();
      restricted.emplace_back(footprint, fine.ncomp(), Extent<Dim>{});
      transfers.push_back(runtime.template prepare_transfer<::pops::amr::transfer::Centering::Cell>(
          fine_level, fine_level - 1, runtime.hierarchy().level(fine_level).spatial_contract(),
          runtime.hierarchy().level(fine_level - 1).spatial_contract(),
          ::pops::amr::transfer::TransferKind::ConservativeRestriction,
          std::as_const(fine.fab(fine_local)).view(), restricted.back().view(), footprint, {},
          {0, 0, fine.ncomp()}));
    }

    const auto& rank_space = fine.rank_space();
    for (std::size_t fine_global = 0; fine_global < fine.layout().size(); ++fine_global) {
      const Box<Dim> footprint =
          ::pops::amr::hierarchy::coarsen_box(fine.layout()[fine_global], authority);
      for (std::size_t parent_global = 0; parent_global < parent.layout().size(); ++parent_global) {
        const Box<Dim> region = footprint.intersect(parent.layout()[parent_global]);
        if (region.empty())
          continue;
        const auto append = [&](const Index<Dim>& source_rank, const Index<Dim>& destination_rank) {
          const std::size_t cells = static_cast<std::size_t>(region.numPts());
          const std::size_t components = static_cast<std::size_t>(fine.ncomp());
          if (cells > std::numeric_limits<std::size_t>::max() / components ||
              element_budget > std::numeric_limits<std::size_t>::max() - cells * components)
            throw std::overflow_error("AMR average-down transport budget exceeds size_t");
          element_budget += cells * components;
          jobs.push_back(
              {fine_global, parent_global, source_rank, destination_rank, region, region});
        };
        if (fine.distribution().replicated() && parent.distribution().replicated()) {
          for (std::size_t rank = 0; rank < rank_space.size(); ++rank) {
            const Index<Dim> coordinate = rank_space.coordinate(rank);
            append(coordinate, coordinate);
          }
        } else if (parent.distribution().replicated()) {
          const Index<Dim> source_rank = fine.distribution().owner(fine_global);
          for (std::size_t rank = 0; rank < rank_space.size(); ++rank)
            append(source_rank, rank_space.coordinate(rank));
        } else if (fine.distribution().replicated()) {
          const Index<Dim> destination_rank = parent.distribution().owner(parent_global);
          append(destination_rank, destination_rank);
        } else {
          append(fine.distribution().owner(fine_global),
                 parent.distribution().owner(parent_global));
        }
      }
    }
  } catch (...) {
    preparation_error = std::current_exception();
  }
  if (all_reduce_max(preparation_error ? 1L : 0L, lane.communicator()) != 0) {
    if (lane.size() == 1 && preparation_error)
      std::rethrow_exception(preparation_error);
    throw std::runtime_error("AMR average-down preparation failed collectively");
  }

  std::optional<transport_type> transport;
  std::exception_ptr transport_error;
  try {
    const std::size_t job_budget = jobs.size();
    transport.emplace(transfer_plan{
        fine.rank_space(),
        fine.local_rank(),
        fine.ncomp(),
        std::move(jobs),
        {job_budget, fine.rank_space().size(), element_budget, element_budget, element_budget}});
    transport->prepare_collectively(lane);
  } catch (...) {
    transport_error = std::current_exception();
  }
  if (all_reduce_max(transport_error ? 1L : 0L, lane.communicator()) != 0) {
    if (lane.size() == 1 && transport_error)
      std::rethrow_exception(transport_error);
    throw std::runtime_error("AMR average-down transport preparation failed collectively");
  }

  std::exception_ptr restriction_error;
  try {
    for (const auto& transfer : transfers)
      execute_prepared_transfer(transfer);
    Kokkos::fence();
  } catch (...) {
    restriction_error = std::current_exception();
  }
  if (all_reduce_max(restriction_error ? 1L : 0L, lane.communicator()) != 0) {
    if (lane.size() == 1 && restriction_error)
      std::rethrow_exception(restriction_error);
    throw std::runtime_error("AMR average-down restriction failed collectively");
  }

  auto source = [&](const transfer_job& job) -> FieldView<const Real, Dim> {
    const std::size_t local = restricted_by_global.at(job.source_patch);
    if (local == field_type::not_local)
      throw std::out_of_range("AMR average-down source patch is not local");
    return std::as_const(restricted.at(local)).view();
  };
  auto destination = [&](const transfer_job& job) -> FieldView<Real, Dim> {
    return parent.fab_global(job.destination_patch).view();
  };
  transport->execute(source, destination);
}

}  // namespace pops::numerics::time::amr
