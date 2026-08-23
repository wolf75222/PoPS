/// @file
/// @brief Prepared exact-rank point-to-point transport for composite AMR transfers.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/rank_space.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::elliptic::amr::partitioned_transfer {

struct RegionTransferBudget {
  std::size_t canonical_jobs = 0;
  std::size_t peer_plans = 0;
  std::size_t local_elements = 0;
  std::size_t send_elements = 0;
  std::size_t receive_elements = 0;
};

template <int Dim>
struct RegionTransferJob {
  std::size_t source_patch = 0;
  std::size_t destination_patch = 0;
  Index<Dim> source_rank{};
  Index<Dim> destination_rank{};
  Box<Dim> source_region{};
  Box<Dim> destination_region{};
  std::size_t offset = 0;
  std::size_t elements = 0;

  bool operator==(const RegionTransferJob&) const = default;
};

template <int Dim>
struct RegionTransferPeerPlan {
  Index<Dim> peer{};
  std::vector<RegionTransferJob<Dim>> jobs{};
  std::size_t elements = 0;

  bool operator==(const RegionTransferPeerPlan&) const = default;
};

namespace detail {

inline void checked_add(std::size_t& total, std::size_t value, std::size_t limit,
                        const char* message) {
  if (total > limit || value > limit - total)
    throw std::length_error(message);
  total += value;
}

template <int Dim>
std::size_t checked_elements(const Box<Dim>& source, const Box<Dim>& destination, int ncomp) {
  if (source.empty() || destination.empty() || source.extent() != destination.extent() || ncomp < 1)
    throw std::invalid_argument(
        "partitioned AMR transfer requires non-empty equal-rank source/destination regions");
  const std::int64_t cells = source.numPts();
  if (cells <= 0 || static_cast<std::uint64_t>(cells) >
                        std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(ncomp))
    throw std::overflow_error("partitioned AMR transfer payload exceeds size_t");
  return static_cast<std::size_t>(cells) * static_cast<std::size_t>(ncomp);
}

template <int Dim>
RegionTransferPeerPlan<Dim>& peer_plan(std::vector<RegionTransferPeerPlan<Dim>>& plans,
                                       const Index<Dim>& peer, std::size_t budget) {
  for (auto& plan : plans)
    if (plan.peer == peer)
      return plan;
  if (plans.size() >= budget || plans.size() >= plans.max_size())
    throw std::length_error("partitioned AMR transfer peer-plan budget exceeded");
  plans.push_back(RegionTransferPeerPlan<Dim>{peer, {}, 0});
  return plans.back();
}

inline void append_u64(std::string& bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<char>((value >> shift) & 0xffu));
}

inline void append_i64(std::string& bytes, std::int64_t value) {
  append_u64(bytes, static_cast<std::uint64_t>(value));
}

inline void append_text(std::string& bytes, std::string_view value) {
  append_u64(bytes, value.size());
  bytes.append(value.data(), value.size());
}

template <int Dim>
void append_index(std::string& bytes, const Index<Dim>& index) {
  for (int axis = 0; axis < Dim; ++axis)
    append_i64(bytes, index[axis]);
}

template <int Dim>
void append_box(std::string& bytes, const Box<Dim>& box) {
  append_index(bytes, box.lo);
  append_index(bytes, box.hi);
}

template <int Dim>
bool contains(const FieldView<const Real, Dim>& view, const Box<Dim>& region, int ncomp) noexcept {
  if (view.data == nullptr || view.ncomp != ncomp)
    return false;
  for (int axis = 0; axis < Dim; ++axis) {
    if (view.extents[axis] <= 0 || view.origin[axis] > region.lo[axis] ||
        static_cast<std::int64_t>(view.origin[axis]) + view.extents[axis] - 1 < region.hi[axis])
      return false;
  }
  return true;
}

template <int Dim>
bool contains(const FieldView<Real, Dim>& view, const Box<Dim>& region, int ncomp) noexcept {
  if (view.data == nullptr || view.ncomp != ncomp)
    return false;
  for (int axis = 0; axis < Dim; ++axis) {
    if (view.extents[axis] <= 0 || view.origin[axis] > region.lo[axis] ||
        static_cast<std::int64_t>(view.origin[axis]) + view.extents[axis] - 1 < region.hi[axis])
      return false;
  }
  return true;
}

}  // namespace detail

/// Immutable globally canonical region schedule classified for one exact process coordinate.
template <int Dim>
class RegionTransferPlan {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "RegionTransferPlan only supports dimensions 1, 2, and 3");

  using job_type = RegionTransferJob<Dim>;
  using peer_plan_type = RegionTransferPeerPlan<Dim>;

  RegionTransferPlan(mesh::RankSpace<Dim> rank_space, Index<Dim> local_rank, int ncomp,
                     std::vector<job_type> canonical_jobs, RegionTransferBudget budget)
      : rank_space_(rank_space), local_rank_(local_rank), ncomp_(ncomp) {
    if (!rank_space_.contains(local_rank_) || ncomp_ < 1)
      throw std::invalid_argument("partitioned AMR transfer rank space or components are invalid");
    if (canonical_jobs.size() > budget.canonical_jobs)
      throw std::length_error("partitioned AMR transfer canonical-job budget exceeded");
    canonical_.reserve(canonical_jobs.size());
    for (job_type job : canonical_jobs) {
      if (!rank_space_.contains(job.source_rank) || !rank_space_.contains(job.destination_rank))
        throw std::invalid_argument("partitioned AMR transfer job rank is outside its rank space");
      job.elements = detail::checked_elements(job.source_region, job.destination_region, ncomp_);
      canonical_.push_back(job);
      const bool source_local = job.source_rank == local_rank_;
      const bool destination_local = job.destination_rank == local_rank_;
      if (source_local && destination_local) {
        job.offset = local_elements_;
        detail::checked_add(local_elements_, job.elements, budget.local_elements,
                            "partitioned AMR transfer local-element budget exceeded");
        local_.push_back(std::move(job));
      } else if (source_local) {
        auto& plan = detail::peer_plan(send_, job.destination_rank, budget.peer_plans);
        job.offset = plan.elements;
        detail::checked_add(plan.elements, job.elements, budget.send_elements,
                            "partitioned AMR transfer peer send budget exceeded");
        detail::checked_add(send_elements_, job.elements, budget.send_elements,
                            "partitioned AMR transfer send-element budget exceeded");
        plan.jobs.push_back(std::move(job));
      } else if (destination_local) {
        auto& plan = detail::peer_plan(receive_, job.source_rank, budget.peer_plans);
        job.offset = plan.elements;
        detail::checked_add(plan.elements, job.elements, budget.receive_elements,
                            "partitioned AMR transfer peer receive budget exceeded");
        detail::checked_add(receive_elements_, job.elements, budget.receive_elements,
                            "partitioned AMR transfer receive-element budget exceeded");
        plan.jobs.push_back(std::move(job));
      }
    }
    const auto order = [this](const peer_plan_type& left, const peer_plan_type& right) {
      return rank_space_.linear_rank(left.peer) < rank_space_.linear_rank(right.peer);
    };
    std::sort(send_.begin(), send_.end(), order);
    std::sort(receive_.begin(), receive_.end(), order);
  }

  const mesh::RankSpace<Dim>& rank_space() const noexcept { return rank_space_; }
  const Index<Dim>& local_rank() const noexcept { return local_rank_; }
  int ncomp() const noexcept { return ncomp_; }
  const std::vector<job_type>& canonical_jobs() const noexcept { return canonical_; }
  const std::vector<job_type>& local_jobs() const noexcept { return local_; }
  const std::vector<peer_plan_type>& send_plans() const noexcept { return send_; }
  const std::vector<peer_plan_type>& receive_plans() const noexcept { return receive_; }
  std::size_t local_elements() const noexcept { return local_elements_; }
  std::size_t send_elements() const noexcept { return send_elements_; }
  std::size_t receive_elements() const noexcept { return receive_elements_; }
  bool has_remote_jobs() const noexcept { return !send_.empty() || !receive_.empty(); }

  static RegionTransferBudget budget_from_jobs(const std::vector<job_type>& jobs) {
    std::size_t elements = 0;
    for (const job_type& job : jobs) {
      const std::int64_t cells = job.source_region.numPts();
      if (cells <= 0)
        continue;
      if (elements > std::numeric_limits<std::size_t>::max() - static_cast<std::size_t>(cells))
        throw std::overflow_error("partitioned AMR transfer budget overflow");
      elements += static_cast<std::size_t>(cells);
    }
    const std::size_t peers = jobs.size() > std::numeric_limits<std::size_t>::max() / 2
                                  ? std::numeric_limits<std::size_t>::max()
                                  : jobs.size() * 2;
    return {jobs.size(), peers, elements, elements, elements};
  }

  std::string exact_contract(std::string_view identity) const {
    if (identity.empty())
      throw std::invalid_argument("partitioned AMR transfer identity is empty");
    std::string bytes;
    detail::append_text(bytes, "pops.elliptic.amr.partitioned-region-transfer");
    detail::append_u64(bytes, 1);
    detail::append_i64(bytes, Dim);
    detail::append_text(bytes, identity);
    detail::append_index(bytes, rank_space_.origin());
    for (int axis = 0; axis < Dim; ++axis)
      detail::append_i64(bytes, rank_space_.extent()[axis]);
    detail::append_i64(bytes, ncomp_);
    detail::append_u64(bytes, canonical_.size());
    for (const job_type& job : canonical_) {
      detail::append_u64(bytes, job.source_patch);
      detail::append_u64(bytes, job.destination_patch);
      detail::append_index(bytes, job.source_rank);
      detail::append_index(bytes, job.destination_rank);
      detail::append_box(bytes, job.source_region);
      detail::append_box(bytes, job.destination_region);
      detail::append_u64(bytes, job.elements);
    }
    return bytes;
  }

 private:
  mesh::RankSpace<Dim> rank_space_{};
  Index<Dim> local_rank_{};
  int ncomp_ = 0;
  std::vector<job_type> canonical_{};
  std::vector<job_type> local_{};
  std::vector<peer_plan_type> send_{};
  std::vector<peer_plan_type> receive_{};
  std::size_t local_elements_ = 0;
  std::size_t send_elements_ = 0;
  std::size_t receive_elements_ = 0;
};

/// Reusable staged transport. All allocations occur before attach_lane and therefore before any
/// solve-time collective. The owning lane is borrowed only after collective preparation succeeds.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class RegionTransport {
 public:
  static_assert(Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, MemorySpace>::accessible,
                "RegionTransport requires DefaultExecutionSpace access to MemorySpace");

  using plan_type = RegionTransferPlan<Dim>;
  using job_type = typename plan_type::job_type;
  using peer_plan_type = typename plan_type::peer_plan_type;
  using device_buffer_type = Kokkos::View<Real*, MemorySpace>;
  using pinned_buffer_type = Kokkos::View<Real*, Kokkos::SharedHostPinnedSpace>;
  using execution_index_type = std::int64_t;
  using execution_policy =
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<execution_index_type>>;

  explicit RegionTransport(plan_type plan) : plan_(std::move(plan)) { allocate_storage_(); }

  RegionTransport(const RegionTransport&) = delete;
  RegionTransport& operator=(const RegionTransport&) = delete;
  RegionTransport(RegionTransport&&) = delete;
  RegionTransport& operator=(RegionTransport&&) = delete;

  void attach_lane(const ExecutionLane& lane) {
    long invalid = 0;
    try {
      invalid = lane_ != nullptr || !lane.active() ||
                        lane.size() != static_cast<int>(plan_.rank_space().size()) ||
                        lane.rank() !=
                            static_cast<int>(plan_.rank_space().linear_rank(plan_.local_rank()))
                    ? 1L
                    : 0L;
#ifdef POPS_HAS_MPI
      if (plan_.has_remote_jobs() && !lane.owns_communicator())
        invalid = 1;
#endif
    } catch (...) {
      invalid = 1;
    }
    if (all_reduce_max(invalid, lane.communicator()) != 0)
      throw std::invalid_argument(
          "partitioned AMR transfer requires its exact owning duplicated ExecutionLane");
    lane_ = &lane;
    lane_borrow_.emplace(lane.borrow_immutably());
    remote_any_ = all_reduce_max(plan_.has_remote_jobs() ? 1L : 0L, lane.communicator()) != 0;
  }

  const plan_type& plan() const noexcept { return plan_; }
  bool sealed() const noexcept { return sealed_; }

  template <class SourceAccessor, class DestinationAccessor>
  void execute(SourceAccessor&& source, DestinationAccessor&& destination) {
    require_attached_();
    if (sealed_)
      throw std::runtime_error("partitioned AMR transfer is sealed after a collective failure");

    long validation_failure = 0;
    try {
      validate_jobs_(plan_.local_jobs(), source, destination);
      for (const auto& peer : peers_) {
        if (peer.send != nullptr)
          validate_jobs_(peer.send->jobs, source, destination, true, false);
        if (peer.receive != nullptr)
          validate_jobs_(peer.receive->jobs, source, destination, false, true);
      }
    } catch (...) {
      validation_failure = 1;
    }
    if (!gate_(validation_failure))
      throw std::invalid_argument(
          "partitioned AMR transfer source/destination binding failed collectively");

    long packing_failure = 0;
    try {
      pack_jobs_(plan_.local_jobs(), local_buffer_, source);
      for (PeerStorage& peer : peers_)
        if (peer.send != nullptr) {
          pack_jobs_(peer.send->jobs, peer.device_send, source);
          Kokkos::deep_copy(peer.host_send, peer.device_send);
        }
      Kokkos::fence();
    } catch (...) {
      packing_failure = 1;
    }
    if (!gate_(packing_failure))
      throw std::runtime_error("partitioned AMR transfer packing failed collectively");

#ifdef POPS_HAS_MPI
    if (remote_any_)
      exchange_remote_();
#else
    if (remote_any_)
      throw std::logic_error("remote partitioned AMR transfer requires an MPI build");
#endif

    long publication_failure = 0;
    try {
      unpack_jobs_(plan_.local_jobs(), local_buffer_, destination);
      for (PeerStorage& peer : peers_)
        if (peer.receive != nullptr)
          unpack_jobs_(peer.receive->jobs, peer.device_receive, destination);
      Kokkos::fence();
    } catch (...) {
      publication_failure = 1;
    }
    if (!gate_(publication_failure)) {
      sealed_ = true;
      std::terminate();
    }
  }

 private:
  struct PeerStorage {
    int mpi_rank = 0;
    const peer_plan_type* send = nullptr;
    const peer_plan_type* receive = nullptr;
    device_buffer_type device_send{};
    device_buffer_type device_receive{};
    pinned_buffer_type host_send{};
    pinned_buffer_type host_receive{};
  };

 public:
  // NVCC-generated Kokkos stubs must name these kernel types.  Transport state and all helpers
  // remain private below.
  struct KernelJob {
    int source_lower[Dim]{};
    int destination_lower[Dim]{};
    execution_index_type extent[Dim]{};
    execution_index_type cells = 0;
    execution_index_type offset = 0;
    execution_index_type elements = 0;
  };

  struct PackKernel {
    device_buffer_type buffer{};
    FieldView<const Real, Dim> source{};
    KernelJob job{};

    POPS_HD void operator()(execution_index_type element) const {
      const int component = static_cast<int>(element / job.cells);
      execution_index_type cell = element % job.cells;
      Index<Dim> index{};
      for (int axis = 0; axis < Dim; ++axis) {
        index[axis] = static_cast<int>(static_cast<execution_index_type>(job.source_lower[axis]) +
                                       cell % job.extent[axis]);
        cell /= job.extent[axis];
      }
      buffer(job.offset + element) = source(index, component);
    }
  };

  struct UnpackKernel {
    device_buffer_type buffer{};
    FieldView<Real, Dim> destination{};
    KernelJob job{};

    POPS_HD void operator()(execution_index_type element) const {
      const int component = static_cast<int>(element / job.cells);
      execution_index_type cell = element % job.cells;
      Index<Dim> index{};
      for (int axis = 0; axis < Dim; ++axis) {
        index[axis] =
            static_cast<int>(static_cast<execution_index_type>(job.destination_lower[axis]) +
                             cell % job.extent[axis]);
        cell /= job.extent[axis];
      }
      destination(index, component) = buffer(job.offset + element);
    }
  };

 private:
  KernelJob lower_(const job_type& job) const {
    const std::size_t execution_max =
        static_cast<std::size_t>(std::numeric_limits<execution_index_type>::max());
    if (job.elements == 0 || job.elements > execution_max || job.offset > execution_max ||
        job.elements > execution_max - job.offset ||
        job.elements % static_cast<std::size_t>(plan_.ncomp()) != 0)
      throw std::overflow_error("partitioned AMR transfer job exceeds execution range");
    KernelJob result{};
    result.cells =
        static_cast<execution_index_type>(job.elements / static_cast<std::size_t>(plan_.ncomp()));
    result.offset = static_cast<execution_index_type>(job.offset);
    result.elements = static_cast<execution_index_type>(job.elements);
    for (int axis = 0; axis < Dim; ++axis) {
      result.source_lower[axis] = job.source_region.lo[axis];
      result.destination_lower[axis] = job.destination_region.lo[axis];
      result.extent[axis] = job.source_region.length(axis);
    }
    return result;
  }

  template <class SourceAccessor, class DestinationAccessor>
  void validate_jobs_(const std::vector<job_type>& jobs, SourceAccessor& source,
                      DestinationAccessor& destination, bool validate_source = true,
                      bool validate_destination = true) const {
    for (const job_type& job : jobs) {
      if (validate_source && !detail::contains(FieldView<const Real, Dim>(source(job)),
                                               job.source_region, plan_.ncomp()))
        throw std::invalid_argument("partitioned AMR transfer source view is stale or incomplete");
      if (validate_destination && !detail::contains(FieldView<Real, Dim>(destination(job)),
                                                    job.destination_region, plan_.ncomp()))
        throw std::invalid_argument(
            "partitioned AMR transfer destination view is stale or incomplete");
    }
  }

  template <class SourceAccessor>
  void pack_jobs_(const std::vector<job_type>& jobs, device_buffer_type buffer,
                  SourceAccessor& source) const {
    for (const job_type& job : jobs) {
      const KernelJob lowered = lower_(job);
      Kokkos::parallel_for("pops_fac_transfer_pack", execution_policy(0, lowered.elements),
                           PackKernel{buffer, FieldView<const Real, Dim>(source(job)), lowered});
    }
  }

  template <class DestinationAccessor>
  void unpack_jobs_(const std::vector<job_type>& jobs, device_buffer_type buffer,
                    DestinationAccessor& destination) const {
    for (const job_type& job : jobs) {
      const KernelJob lowered = lower_(job);
      Kokkos::parallel_for("pops_fac_transfer_unpack", execution_policy(0, lowered.elements),
                           UnpackKernel{buffer, FieldView<Real, Dim>(destination(job)), lowered});
    }
  }

  bool gate_(long failure) const noexcept {
    try {
      return all_reduce_max(failure, lane_->communicator()) == 0;
    } catch (...) {
      return false;
    }
  }

  void require_attached_() const {
    if (lane_ == nullptr || !lane_borrow_)
      throw std::logic_error("partitioned AMR transfer has no prepared ExecutionLane");
  }

  void allocate_storage_() {
    local_buffer_ = device_buffer_type("pops_fac_transfer_local", plan_.local_elements());
    const std::size_t plans = plan_.send_plans().size() + plan_.receive_plans().size();
    if (plans > peers_.max_size())
      throw std::length_error("partitioned AMR transfer peer storage exceeds vector capacity");
    peers_.reserve(plans);
    const auto attach = [this](const peer_plan_type& plan, bool send) {
      const std::size_t linear = plan_.rank_space().linear_rank(plan.peer);
      if (linear > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          plan.elements > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("partitioned AMR transfer peer payload exceeds MPI int range");
      auto found = std::find_if(peers_.begin(), peers_.end(), [linear](const PeerStorage& peer) {
        return peer.mpi_rank == static_cast<int>(linear);
      });
      if (found == peers_.end()) {
        peers_.push_back(PeerStorage{static_cast<int>(linear)});
        found = std::prev(peers_.end());
      }
      const peer_plan_type*& slot = send ? found->send : found->receive;
      if (slot != nullptr)
        throw std::logic_error("partitioned AMR transfer has a duplicate peer plan");
      slot = &plan;
    };
    for (const auto& plan : plan_.send_plans())
      attach(plan, true);
    for (const auto& plan : plan_.receive_plans())
      attach(plan, false);
    std::sort(peers_.begin(), peers_.end(), [](const PeerStorage& left, const PeerStorage& right) {
      return left.mpi_rank < right.mpi_rank;
    });
    for (PeerStorage& peer : peers_) {
      const std::size_t sends = peer.send == nullptr ? 0 : peer.send->elements;
      const std::size_t receives = peer.receive == nullptr ? 0 : peer.receive->elements;
      peer.device_send = device_buffer_type("pops_fac_transfer_device_send", sends);
      peer.device_receive = device_buffer_type("pops_fac_transfer_device_receive", receives);
      peer.host_send = pinned_buffer_type("pops_fac_transfer_host_send", sends);
      peer.host_receive = pinned_buffer_type("pops_fac_transfer_host_receive", receives);
    }
#ifdef POPS_HAS_MPI
    receive_requests_.resize(plan_.receive_plans().size(), MPI_REQUEST_NULL);
    send_requests_.resize(plan_.send_plans().size(), MPI_REQUEST_NULL);
    receive_statuses_.resize(plan_.receive_plans().size());
#endif
  }

#ifdef POPS_HAS_MPI
  static int wait_all_(std::vector<MPI_Request>& requests) noexcept {
    if (requests.empty())
      return MPI_SUCCESS;
    if (requests.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      return MPI_ERR_COUNT;
    return MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
  }

  bool drain_() noexcept {
    bool safe = true;
    for (MPI_Request& request : receive_requests_)
      if (request != MPI_REQUEST_NULL && MPI_Cancel(&request) != MPI_SUCCESS)
        safe = false;
    if (wait_all_(send_requests_) != MPI_SUCCESS || wait_all_(receive_requests_) != MPI_SUCCESS)
      safe = false;
    const auto null_request = [](MPI_Request request) { return request == MPI_REQUEST_NULL; };
    return safe && std::all_of(send_requests_.begin(), send_requests_.end(), null_request) &&
           std::all_of(receive_requests_.begin(), receive_requests_.end(), null_request);
  }

  [[noreturn]] void unsafe_failure_(const char* message) {
    sealed_ = true;
    if (!drain_())
      std::terminate();
    throw std::runtime_error(message);
  }

  void exchange_remote_() {
    std::fill(receive_requests_.begin(), receive_requests_.end(), MPI_REQUEST_NULL);
    std::fill(send_requests_.begin(), send_requests_.end(), MPI_REQUEST_NULL);
    std::size_t receive_index = 0;
    int code = MPI_SUCCESS;
    for (PeerStorage& peer : peers_)
      if (peer.receive != nullptr) {
        if (code == MPI_SUCCESS)
          code = MPI_Irecv(peer.host_receive.data(), static_cast<int>(peer.receive->elements),
                           pops::mpi_real_datatype(), peer.mpi_rank, ExecutionLane::parallel_copy_message_tag,
                           lane_->native_handle(), &receive_requests_[receive_index]);
        ++receive_index;
      }
    if (!gate_(code == MPI_SUCCESS ? 0L : 1L))
      unsafe_failure_("partitioned AMR transfer receive posting failed collectively");

    std::size_t send_index = 0;
    code = MPI_SUCCESS;
    for (PeerStorage& peer : peers_)
      if (peer.send != nullptr) {
        if (code == MPI_SUCCESS)
          code = MPI_Isend(peer.host_send.data(), static_cast<int>(peer.send->elements), pops::mpi_real_datatype(),
                           peer.mpi_rank, ExecutionLane::parallel_copy_message_tag,
                           lane_->native_handle(), &send_requests_[send_index]);
        ++send_index;
      }
    if (!gate_(code == MPI_SUCCESS ? 0L : 1L))
      unsafe_failure_("partitioned AMR transfer send posting failed collectively");

    code = wait_all_(send_requests_);
    if (code == MPI_SUCCESS && !receive_requests_.empty())
      code = MPI_Waitall(static_cast<int>(receive_requests_.size()), receive_requests_.data(),
                         receive_statuses_.data());
    if (code == MPI_SUCCESS) {
      std::size_t status_index = 0;
      for (const PeerStorage& peer : peers_)
        if (peer.receive != nullptr) {
          int count = MPI_UNDEFINED;
          const MPI_Status& status = receive_statuses_[status_index++];
          if (MPI_Get_count(&status, pops::mpi_real_datatype(), &count) != MPI_SUCCESS ||
              status.MPI_SOURCE != peer.mpi_rank ||
              status.MPI_TAG != ExecutionLane::parallel_copy_message_tag ||
              count != static_cast<int>(peer.receive->elements)) {
            code = MPI_ERR_OTHER;
            break;
          }
        }
    }
    if (!gate_(code == MPI_SUCCESS ? 0L : 1L))
      unsafe_failure_(
          "partitioned AMR transfer receive authentication or MPI wait failed collectively");

    long staging_failure = 0;
    try {
      for (PeerStorage& peer : peers_)
        if (peer.receive != nullptr)
          Kokkos::deep_copy(peer.device_receive, peer.host_receive);
      Kokkos::fence();
    } catch (...) {
      staging_failure = 1;
    }
    if (!gate_(staging_failure)) {
      sealed_ = true;
      throw std::runtime_error(
          "partitioned AMR transfer receive staging failed collectively after a safe drain");
    }
  }
#endif

  plan_type plan_;
  const ExecutionLane* lane_ = nullptr;
  std::optional<ExecutionLane::ImmutableBorrow> lane_borrow_{};
  device_buffer_type local_buffer_{};
  std::vector<PeerStorage> peers_{};
  bool remote_any_ = false;
  bool sealed_ = false;
#ifdef POPS_HAS_MPI
  std::vector<MPI_Request> receive_requests_{};
  std::vector<MPI_Request> send_requests_{};
  std::vector<MPI_Status> receive_statuses_{};
#endif
};

}  // namespace pops::elliptic::amr::partitioned_transfer
