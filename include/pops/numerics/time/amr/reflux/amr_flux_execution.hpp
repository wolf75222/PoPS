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
#include <string>
#include <utility>
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

/// Cold-prepared conservative restriction and distribution authority for one exact level pair.
/// Fine and parent distributions may have different patch boundaries and owners.
template <int Dim, class MemorySpace>
class PreparedAverageDown {
 public:
  using field_type = MultiFab<Dim, MemorySpace>;
  using transfer_job = ::pops::mesh::parallel::RegionTransferJob<Dim>;
  using transfer_plan = ::pops::mesh::parallel::RegionTransferPlan<Dim>;
  using transport_type = ::pops::mesh::parallel::RegionTransport<Dim, MemorySpace>;
  using runtime_type = ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>;
  using hierarchy_type = typename runtime_type::hierarchy_type;

  PreparedAverageDown() = default;
  PreparedAverageDown(const PreparedAverageDown&) = delete;
  PreparedAverageDown& operator=(const PreparedAverageDown&) = delete;
  PreparedAverageDown(PreparedAverageDown&&) = delete;
  PreparedAverageDown& operator=(PreparedAverageDown&&) = delete;

  void prepare(const runtime_type& runtime, std::size_t fine_level, const field_type& fine,
               field_type& parent, const ExecutionLane& lane) {
    prepare_impl_(runtime.hierarchy(), &runtime, runtime.topology_epoch(),
                  runtime.materialization_generation(), fine_level, fine, parent, lane);
  }

  /// Candidate-only preparation over an unpublished hierarchy.  The runtime pointer is retained
  /// solely as the eventual post-publication liveness witness; no runtime member is consulted
  /// while this method allocates the transfer carriers.
  void prepare_forward(const hierarchy_type& hierarchy, const runtime_type& eventual_runtime,
                       std::uint64_t topology_epoch,
                       std::uint64_t materialization_generation, std::size_t fine_level,
                       const field_type& fine, field_type& parent, const ExecutionLane& lane) {
    prepare_impl_(hierarchy, &eventual_runtime, topology_epoch, materialization_generation,
                  fine_level, fine, parent, lane);
  }

 private:
  void prepare_impl_(const hierarchy_type& hierarchy, const runtime_type* eventual_runtime,
                     std::uint64_t topology_epoch,
                     std::uint64_t materialization_generation, std::size_t fine_level,
                     const field_type& fine, field_type& parent, const ExecutionLane& lane) {
    if (prepared_)
      throw std::logic_error("AMR average-down authority is already prepared");
    validate_carriers_(hierarchy, fine_level, fine, parent, lane);

    const auto& authority = hierarchy.layout(fine_level).ratio_from_parent();
    std::vector<transfer_job> jobs;
    std::size_t element_budget = 0;
    std::exception_ptr preparation_error;
    try {
      restricted_.reserve(fine.local_size());
      restricted_by_global_.assign(fine.layout().size(), field_type::not_local);
      transfers_.reserve(fine.local_size());
      fine_globals_.reserve(fine.local_size());
      fine_storage_.reserve(fine.local_size());
      parent_globals_.reserve(parent.local_size());
      parent_storage_.reserve(parent.local_size());
      for (std::size_t fine_local = 0; fine_local < fine.local_size(); ++fine_local) {
        const std::size_t fine_global = fine.global_index(fine_local);
        const Box<Dim> footprint =
            ::pops::amr::hierarchy::coarsen_box(fine.layout()[fine_global], authority);
        restricted_by_global_[fine_global] = restricted_.size();
        restricted_.emplace_back(footprint, fine.ncomp(), Extent<Dim>{});
        transfers_.push_back(
            ::pops::amr::transfer::TransferProvider<
                Dim, ::pops::amr::transfer::Centering::Cell>(
                ::pops::amr::transfer::TransferKind::ConservativeRestriction)
                .prepare(std::as_const(fine.fab(fine_local)).view(), restricted_.back().view(),
                         footprint, authority, {}, {0, 0, fine.ncomp()}));
        fine_globals_.push_back(fine_global);
        fine_storage_.push_back(fine.fab(fine_local).view().data);
      }
      for (std::size_t parent_local = 0; parent_local < parent.local_size(); ++parent_local) {
        parent_globals_.push_back(parent.global_index(parent_local));
        parent_storage_.push_back(parent.fab(parent_local).view().data);
      }

      const auto& rank_space = fine.rank_space();
      for (std::size_t fine_global = 0; fine_global < fine.layout().size(); ++fine_global) {
        const Box<Dim> footprint =
            ::pops::amr::hierarchy::coarsen_box(fine.layout()[fine_global], authority);
        for (std::size_t parent_global = 0; parent_global < parent.layout().size();
             ++parent_global) {
          const Box<Dim> region = footprint.intersect(parent.layout()[parent_global]);
          if (region.empty())
            continue;
          const auto append = [&](const Index<Dim>& source_rank,
                                  const Index<Dim>& destination_rank) {
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

    std::exception_ptr transport_error;
    try {
      const std::size_t job_budget = jobs.size();
      transport_.emplace(transfer_plan{
          fine.rank_space(),
          fine.local_rank(),
          fine.ncomp(),
          std::move(jobs),
          {job_budget, fine.rank_space().size(), element_budget, element_budget, element_budget}});
      transport_->prepare_collectively(lane);
    } catch (...) {
      transport_error = std::current_exception();
    }
    if (all_reduce_max(transport_error ? 1L : 0L, lane.communicator()) != 0) {
      if (lane.size() == 1 && transport_error)
        std::rethrow_exception(transport_error);
      throw std::runtime_error("AMR average-down transport preparation failed collectively");
    }

    runtime_ = eventual_runtime;
    lane_ = &lane;
    fine_level_ = fine_level;
    topology_generation_ = topology_epoch;
    materialization_generation_ = materialization_generation;
    fine_ncomp_ = fine.ncomp();
    fine_ghosts_ = fine.ghosts();
    parent_ghosts_ = parent.ghosts();
    fine_layout_size_ = fine.layout().size();
    parent_layout_size_ = parent.layout().size();
    prepared_ = true;
  }

 public:

  void execute(const runtime_type& runtime, std::size_t fine_level, const field_type& fine,
               field_type& parent, const ExecutionLane& lane) {
    long validation_failure = 0;
    try {
      if (!prepared_ || runtime_ != &runtime || lane_ != &lane || fine_level_ != fine_level ||
          topology_generation_ != runtime.topology_epoch() ||
          materialization_generation_ != runtime.materialization_generation() ||
          fine.ncomp() != fine_ncomp_ || parent.ncomp() != fine_ncomp_ ||
          fine.ghosts() != fine_ghosts_ || parent.ghosts() != parent_ghosts_ ||
          fine.layout().size() != fine_layout_size_ ||
          parent.layout().size() != parent_layout_size_ ||
          fine.local_size() != fine_globals_.size() ||
          parent.local_size() != parent_globals_.size())
        validation_failure = 1;
      for (std::size_t local = 0; validation_failure == 0 && local < fine.local_size(); ++local)
        if (fine.global_index(local) != fine_globals_[local] ||
            fine.fab(local).view().data != fine_storage_[local])
          validation_failure = 1;
      for (std::size_t local = 0; validation_failure == 0 && local < parent.local_size(); ++local)
        if (parent.global_index(local) != parent_globals_[local] ||
            parent.fab(local).view().data != parent_storage_[local])
          validation_failure = 1;
    } catch (...) {
      validation_failure = 1;
    }
    if (all_reduce_max(validation_failure, lane.communicator()) != 0)
      throw std::invalid_argument(
          "AMR average-down prepared carrier binding is stale collectively");

    std::exception_ptr restriction_error;
    try {
      for (const auto& transfer : transfers_)
        execute_prepared_transfer(transfer);
      ::pops::device_fence(restriction_fence_label_);
    } catch (...) {
      restriction_error = std::current_exception();
    }
    if (all_reduce_max(restriction_error ? 1L : 0L, lane.communicator()) != 0) {
      if (lane.size() == 1 && restriction_error)
        std::rethrow_exception(restriction_error);
      throw std::runtime_error("AMR average-down restriction failed collectively");
    }

    auto source = [&](const transfer_job& job) -> FieldView<const Real, Dim> {
      const std::size_t local = restricted_by_global_.at(job.source_patch);
      if (local == field_type::not_local)
        throw std::out_of_range("AMR average-down source patch is not local");
      return std::as_const(restricted_.at(local)).view();
    };
    auto destination = [&](const transfer_job& job) -> FieldView<Real, Dim> {
      return parent.fab_global(job.destination_patch).view();
    };
    transport_->execute(source, destination);
  }

  /// Resident non-ledger payload used by the generic subcycling engine.  Region-transport owns
  /// its prepared transfer plan internally; its request/response buffers are warmed by that
  /// carrier and intentionally remain infrastructure rather than a generated Program value.
  [[nodiscard]] std::uint64_t resident_storage_bytes() const {
    const auto checked_add = [](std::uint64_t& total, std::uint64_t value) {
      if (value > std::numeric_limits<std::uint64_t>::max() - total)
        throw std::overflow_error("prepared average-down resident storage overflows uint64");
      total += value;
    };
    const auto vector_bytes = [](const auto& values) -> std::uint64_t {
      using value_type = typename std::remove_reference_t<decltype(values)>::value_type;
      if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
        throw std::overflow_error("prepared average-down vector storage overflows uint64");
      return static_cast<std::uint64_t>(values.capacity()) * sizeof(value_type);
    };
    const auto fab_payload_bytes = [](const Fab<Dim, MemorySpace>& fab) -> std::uint64_t {
      const auto element_count = static_cast<std::uint64_t>(fab.storage().extent(0));
      if (element_count > std::numeric_limits<std::uint64_t>::max() / sizeof(Real))
        throw std::overflow_error("prepared average-down Fab payload overflows uint64");
      return element_count * sizeof(Real);
    };
    std::uint64_t total = 0;
    checked_add(total, vector_bytes(restricted_));
    for (const auto& restricted : restricted_)
      checked_add(total, fab_payload_bytes(restricted));
    checked_add(total, vector_bytes(restricted_by_global_));
    checked_add(total, vector_bytes(transfers_));
    checked_add(total, vector_bytes(fine_globals_));
    checked_add(total, vector_bytes(fine_storage_));
    checked_add(total, vector_bytes(parent_globals_));
    checked_add(total, vector_bytes(parent_storage_));
    const auto begin = reinterpret_cast<std::uintptr_t>(&restriction_fence_label_);
    const auto end = begin + sizeof(restriction_fence_label_);
    const auto data = reinterpret_cast<std::uintptr_t>(restriction_fence_label_.data());
    if (!(data >= begin && data < end))
      checked_add(total, static_cast<std::uint64_t>(restriction_fence_label_.capacity()) + 1U);
    return total;
  }

 private:
  static void validate_carriers_(const hierarchy_type& hierarchy, std::size_t fine_level,
                                 const field_type& fine, const field_type& parent,
                                 const ExecutionLane& lane) {
    if (fine_level == 0 || fine_level >= hierarchy.num_levels())
      throw std::out_of_range("AMR average-down level is outside the live hierarchy");
    if (fine.ncomp() != parent.ncomp() || fine.ncomp() < 1 ||
        fine.rank_space() != parent.rank_space() || fine.local_rank() != parent.local_rank())
      throw std::invalid_argument("AMR average-down carriers have incompatible exact contracts");
    if (lane.size() != static_cast<int>(fine.rank_space().size()) ||
        lane.rank() != static_cast<int>(fine.rank_space().linear_rank(fine.local_rank())))
      throw std::invalid_argument("AMR average-down lane does not match the carrier rank space");
  }

  const runtime_type* runtime_ = nullptr;
  const ExecutionLane* lane_ = nullptr;
  std::size_t fine_level_ = 0;
  std::uint64_t topology_generation_ = 0;
  std::uint64_t materialization_generation_ = 0;
  int fine_ncomp_ = 0;
  Extent<Dim> fine_ghosts_{};
  Extent<Dim> parent_ghosts_{};
  std::size_t fine_layout_size_ = 0;
  std::size_t parent_layout_size_ = 0;
  std::vector<Fab<Dim, MemorySpace>> restricted_{};
  std::vector<std::size_t> restricted_by_global_{};
  std::vector<::pops::amr::transfer::PreparedTransfer<Dim>> transfers_{};
  std::vector<std::size_t> fine_globals_{};
  std::vector<const Real*> fine_storage_{};
  std::vector<std::size_t> parent_globals_{};
  std::vector<Real*> parent_storage_{};
  std::optional<transport_type> transport_{};
  std::string restriction_fence_label_{"pops.prepared-average-down.restriction-fence"};
  bool prepared_ = false;
};

/// Conservatively restrict a complete distributed level onto its parent carrier.
/// Fine and parent distributions may have different patch boundaries and owners.
template <int Dim, class MemorySpace>
void execute_average_down_collectively(
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime, std::size_t fine_level,
    const MultiFab<Dim, MemorySpace>& fine, MultiFab<Dim, MemorySpace>& parent,
    const ExecutionLane& lane) {
  PreparedAverageDown<Dim, MemorySpace> prepared;
  prepared.prepare(runtime, fine_level, fine, parent, lane);
  prepared.execute(runtime, fine_level, fine, parent, lane);
}

}  // namespace pops::numerics::time::amr
