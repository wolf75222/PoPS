/// @file
/// @brief Prepared MPI/Kokkos execution of an authenticated exact-rank CopySchedule.

#pragma once

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/copy_schedule.hpp>
#include <pops/mesh/parallel/region_transfer.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::mesh::parallel {

namespace copy_transport_detail {

template <int Dim>
struct LocalCopyKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> source{};
  int components = 0;

  POPS_HD void operator()(const Index<Dim>& index) const {
    for (int component = 0; component < components; ++component)
      destination(index, component) = source(index, component);
  }
};

}  // namespace copy_transport_detail

/// Reusable transport for the complete component range of one CopySchedule. The schedule remains
/// the sole overlap/ownership authority; this class only lowers its canonical jobs to the generic
/// region transport. A stable owning ExecutionLane must outlive the prepared transport.
template <int Dim, class BufferMemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PreparedCopyTransport {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "PreparedCopyTransport only supports dimensions 1, 2, and 3");
  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, BufferMemorySpace>::accessible,
      "PreparedCopyTransport requires DefaultExecutionSpace access to its buffer MemorySpace");

  using schedule_type = CopySchedule<Dim>;
  using region_job_type = RegionTransferJob<Dim>;
  using region_plan_type = RegionTransferPlan<Dim>;
  using region_transport_type = RegionTransport<Dim, BufferMemorySpace>;

  PreparedCopyTransport(schedule_type schedule, int components, RegionTransferBudget budget)
      : schedule_(std::move(schedule)), components_(components) {
    if (components_ < 1)
      throw std::invalid_argument("prepared copy transport requires a positive component count");
    if (schedule_.source_distribution().replicated())
      return;

    std::vector<region_job_type> jobs;
    if (schedule_.canonical_jobs().size() > jobs.max_size())
      throw std::length_error(
          "prepared copy transport canonical job count exceeds vector capacity");
    jobs.reserve(schedule_.canonical_jobs().size());
    for (const CopyJob<Dim>& job : schedule_.canonical_jobs())
      jobs.push_back(region_job_type{
          job.source_box, job.destination_box,
          schedule_.source_distribution().owner(job.source_box),
          schedule_.destination_distribution().owner(job.destination_box), job.region, job.region});
    region_transport_ = std::make_unique<region_transport_type>(
        region_plan_type{schedule_.source_distribution().rank_space(), schedule_.local_rank(),
                         components_, std::move(jobs), budget});
  }

  PreparedCopyTransport(const PreparedCopyTransport&) = delete;
  PreparedCopyTransport& operator=(const PreparedCopyTransport&) = delete;
  PreparedCopyTransport(PreparedCopyTransport&&) = delete;
  PreparedCopyTransport& operator=(PreparedCopyTransport&&) = delete;

  const schedule_type& schedule() const noexcept { return schedule_; }
  int components() const noexcept { return components_; }
  bool has_remote_jobs() const noexcept { return schedule_.has_remote_jobs(); }

  void attach_lane(const ExecutionLane& lane) {
    if (lane_ != nullptr)
      throw std::logic_error("prepared copy transport already has an ExecutionLane");
    long invalid = 0;
    try {
      invalid = !lane.active() ||
                        lane.size() !=
                            static_cast<int>(schedule_.source_distribution().rank_space().size()) ||
                        lane.rank() != static_cast<int>(
                                           schedule_.source_distribution().rank_space().linear_rank(
                                               schedule_.local_rank()))
                    ? 1L
                    : 0L;
    } catch (...) {
      invalid = 1;
    }
    if (all_reduce_max(invalid, lane.communicator()) != 0)
      throw std::invalid_argument(
          "prepared copy transport requires its exact ranked ExecutionLane");

    std::string contract;
    long serialization_failure = 0;
    try {
      contract = canonical_contract_();
    } catch (...) {
      serialization_failure = 1;
    }
    if (all_reduce_max(serialization_failure, lane.communicator()) != 0)
      throw std::runtime_error(
          "prepared copy transport contract serialization failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("pops-prepared-copy-transport-v1"), std::string_view(contract)}},
            lane.communicator()))
      throw std::invalid_argument(
          "prepared copy transport canonical schedule differs between MPI ranks");

    if (region_transport_)
      region_transport_->attach_lane(lane);
    lane_ = &lane;
    lane_borrow_.emplace(lane.borrow_immutably());
  }

  template <class DestinationMemorySpace, class SourceMemorySpace>
  void execute(MultiFab<Dim, DestinationMemorySpace>& destination,
               const MultiFab<Dim, SourceMemorySpace>& source) {
    static_assert(Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace,
                                             DestinationMemorySpace>::accessible,
                  "PreparedCopyTransport requires DefaultExecutionSpace destination access");
    static_assert(
        Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, SourceMemorySpace>::accessible,
        "PreparedCopyTransport requires DefaultExecutionSpace source access");
    if (lane_ == nullptr || !lane_borrow_)
      throw std::logic_error("prepared copy transport has no attached ExecutionLane");

    long invalid = 0;
    try {
      schedule_.authenticate(destination, source);
      if (destination.ncomp() != components_ || source.ncomp() != components_)
        throw std::invalid_argument(
            "prepared copy transport field component count changed after preparation");
    } catch (...) {
      invalid = 1;
    }
    if (all_reduce_max(invalid, lane_->communicator()) != 0)
      throw std::invalid_argument(
          "prepared copy transport source/destination binding failed collectively");

    if (region_transport_) {
      auto source_view = [&source](const region_job_type& job) -> FieldView<const Real, Dim> {
        return source.fab_global(job.source_patch).view();
      };
      auto destination_view = [&destination](const region_job_type& job) -> FieldView<Real, Dim> {
        return destination.fab_global(job.destination_patch).view();
      };
      region_transport_->execute(source_view, destination_view);
      return;
    }

    long execution_failure = 0;
    try {
      for (const CopyJob<Dim>& job : schedule_.local_jobs()) {
        auto& destination_fab = destination.fab_global(job.destination_box);
        const auto& source_fab = source.fab_global(job.source_box);
        for_each_cell(job.region, copy_transport_detail::LocalCopyKernel<Dim>{
                                      destination_fab.view(), source_fab.view(), components_});
      }
      Kokkos::fence();
    } catch (...) {
      execution_failure = 1;
    }
    if (all_reduce_max(execution_failure, lane_->communicator()) != 0)
      throw std::runtime_error("prepared replicated copy execution failed collectively");
  }

 private:
  std::string canonical_contract_() const {
    ExactContractBuilder contract;
    contract.text("pops.mesh.parallel.prepared-copy-transport")
        .scalar(std::uint32_t{1})
        .scalar(std::uint32_t{Dim})
        .scalar(std::int32_t{components_})
        .presence(static_cast<bool>(region_transport_));
    if (region_transport_) {
      contract.bytes(region_transport_->plan().exact_contract("copy-schedule"));
      return std::move(contract).release();
    }

    const auto& ranks = schedule_.source_distribution().rank_space();
    for (int axis = 0; axis < Dim; ++axis) {
      contract.scalar(std::int64_t{ranks.origin()[axis]});
      contract.scalar(std::int64_t{ranks.extent()[axis]});
    }
    contract.sequence(schedule_.canonical_jobs(), [](ExactContractBuilder& encoded,
                                                     const CopyJob<Dim>& job) {
      encoded.scalar(static_cast<std::uint64_t>(job.source_box))
          .scalar(static_cast<std::uint64_t>(job.destination_box));
      for (int axis = 0; axis < Dim; ++axis)
        encoded.scalar(std::int64_t{job.region.lo[axis]}).scalar(std::int64_t{job.region.hi[axis]});
    });
    return std::move(contract).release();
  }

  schedule_type schedule_;
  int components_ = 0;
  std::unique_ptr<region_transport_type> region_transport_;
  const ExecutionLane* lane_ = nullptr;
  std::optional<ExecutionLane::ImmutableBorrow> lane_borrow_;
};

}  // namespace pops::mesh::parallel
