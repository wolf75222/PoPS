/// @file
/// @brief Prepared exact-rank point-to-point transport for distributed mesh regions.

#pragma once

#include <pops/core/foundation/kokkos_env.hpp>
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
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::mesh::parallel {

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
        "region transfer requires non-empty equal-rank source/destination regions");
  const std::int64_t cells = source.numPts();
  if (cells <= 0 || static_cast<std::uint64_t>(cells) >
                        std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(ncomp))
    throw std::overflow_error("region transfer payload exceeds size_t");
  return static_cast<std::size_t>(cells) * static_cast<std::size_t>(ncomp);
}

template <int Dim>
RegionTransferPeerPlan<Dim>& peer_plan(std::vector<RegionTransferPeerPlan<Dim>>& plans,
                                       const Index<Dim>& peer, std::size_t budget) {
  for (auto& plan : plans)
    if (plan.peer == peer)
      return plan;
  if (plans.size() >= budget || plans.size() >= plans.max_size())
    throw std::length_error("region transfer peer-plan budget exceeded");
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

/// Immutable globally canonical region input. Rank-local classification is deliberately deferred
/// until RegionTransport authenticates this exact input collectively on its borrowed lane.
template <int Dim>
class RegionTransferPlan {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "RegionTransferPlan only supports dimensions 1, 2, and 3");

  using job_type = RegionTransferJob<Dim>;
  using peer_plan_type = RegionTransferPeerPlan<Dim>;

  RegionTransferPlan(RankSpace<Dim> rank_space, Index<Dim> local_rank, int ncomp,
                     std::vector<job_type> canonical_jobs, RegionTransferBudget budget) noexcept
      : rank_space_(rank_space),
        local_rank_(local_rank),
        ncomp_(ncomp),
        canonical_input_(std::move(canonical_jobs)),
        budget_(budget) {}

  const RankSpace<Dim>& rank_space() const noexcept { return rank_space_; }
  const Index<Dim>& local_rank() const noexcept { return local_rank_; }
  int ncomp() const noexcept { return ncomp_; }
  const std::vector<job_type>& canonical_jobs() const noexcept {
    return prepared_ ? prepared_->canonical : canonical_input_;
  }
  const std::vector<job_type>& local_jobs() const { return prepared_classification_().local; }
  const std::vector<peer_plan_type>& send_plans() const { return prepared_classification_().send; }
  const std::vector<peer_plan_type>& receive_plans() const {
    return prepared_classification_().receive;
  }
  std::size_t local_elements() const { return prepared_classification_().local_elements; }
  std::size_t send_elements() const { return prepared_classification_().send_elements; }
  std::size_t receive_elements() const { return prepared_classification_().receive_elements; }
  bool has_remote_jobs() const noexcept {
    return std::any_of(canonical_input_.begin(), canonical_input_.end(),
                       [](const job_type& job) { return job.source_rank != job.destination_rank; });
  }

  std::string exact_contract(std::string_view identity) const {
    if (identity.empty())
      throw std::invalid_argument("region transfer identity is empty");
    std::string bytes;
    detail::append_text(bytes, "pops.mesh.parallel.region-transfer");
    detail::append_u64(bytes, 3);
    detail::append_i64(bytes, Dim);
    detail::append_text(bytes, identity);
    detail::append_index(bytes, rank_space_.origin());
    for (int axis = 0; axis < Dim; ++axis)
      detail::append_i64(bytes, rank_space_.extent()[axis]);
    detail::append_i64(bytes, ncomp_);
    detail::append_u64(bytes, canonical_input_.size());
    for (const job_type& job : canonical_input_) {
      detail::append_u64(bytes, job.source_patch);
      detail::append_u64(bytes, job.destination_patch);
      detail::append_index(bytes, job.source_rank);
      detail::append_index(bytes, job.destination_rank);
      detail::append_box(bytes, job.source_region);
      detail::append_box(bytes, job.destination_region);
      detail::append_u64(
          bytes, detail::checked_elements(job.source_region, job.destination_region, ncomp_));
    }
    detail::append_u64(bytes, budget_.canonical_jobs);
    detail::append_u64(bytes, budget_.peer_plans);
    detail::append_u64(bytes, budget_.local_elements);
    detail::append_u64(bytes, budget_.send_elements);
    detail::append_u64(bytes, budget_.receive_elements);
    return bytes;
  }

 private:
  template <int, class>
  friend class RegionTransport;

  struct PreparedClassification {
    std::vector<job_type> canonical{};
    std::vector<job_type> local{};
    std::vector<peer_plan_type> send{};
    std::vector<peer_plan_type> receive{};
    std::size_t local_elements = 0;
    std::size_t send_elements = 0;
    std::size_t receive_elements = 0;
  };

  const PreparedClassification& prepared_classification_() const {
    if (!prepared_)
      throw std::logic_error("region transfer plan is not collectively prepared");
    return *prepared_;
  }

  std::unique_ptr<PreparedClassification> classify_() const {
    if (!rank_space_.contains(local_rank_) || ncomp_ < 1)
      throw std::invalid_argument("region transfer rank space or components are invalid");
    if (canonical_input_.size() > budget_.canonical_jobs)
      throw std::length_error("region transfer canonical-job budget exceeded");

    auto prepared = std::make_unique<PreparedClassification>();
    prepared->canonical.reserve(canonical_input_.size());
    for (job_type job : canonical_input_) {
      if (!rank_space_.contains(job.source_rank) || !rank_space_.contains(job.destination_rank))
        throw std::invalid_argument("region transfer job rank is outside its rank space");
      job.elements = detail::checked_elements(job.source_region, job.destination_region, ncomp_);
      prepared->canonical.push_back(job);
      const bool source_local = job.source_rank == local_rank_;
      const bool destination_local = job.destination_rank == local_rank_;
      if (source_local && destination_local) {
        job.offset = prepared->local_elements;
        detail::checked_add(prepared->local_elements, job.elements, budget_.local_elements,
                            "region transfer local-element budget exceeded");
        prepared->local.push_back(std::move(job));
      } else if (source_local) {
        auto& plan = detail::peer_plan(prepared->send, job.destination_rank, budget_.peer_plans);
        job.offset = plan.elements;
        detail::checked_add(plan.elements, job.elements, budget_.send_elements,
                            "region transfer peer send budget exceeded");
        detail::checked_add(prepared->send_elements, job.elements, budget_.send_elements,
                            "region transfer send-element budget exceeded");
        plan.jobs.push_back(std::move(job));
      } else if (destination_local) {
        auto& plan = detail::peer_plan(prepared->receive, job.source_rank, budget_.peer_plans);
        job.offset = plan.elements;
        detail::checked_add(plan.elements, job.elements, budget_.receive_elements,
                            "region transfer peer receive budget exceeded");
        detail::checked_add(prepared->receive_elements, job.elements, budget_.receive_elements,
                            "region transfer receive-element budget exceeded");
        plan.jobs.push_back(std::move(job));
      }
    }
    const auto order = [this](const peer_plan_type& left, const peer_plan_type& right) {
      return rank_space_.linear_rank(left.peer) < rank_space_.linear_rank(right.peer);
    };
    std::sort(prepared->send.begin(), prepared->send.end(), order);
    std::sort(prepared->receive.begin(), prepared->receive.end(), order);
    return prepared;
  }

  RankSpace<Dim> rank_space_{};
  Index<Dim> local_rank_{};
  int ncomp_ = 0;
  std::vector<job_type> canonical_input_{};
  RegionTransferBudget budget_{};
  std::unique_ptr<PreparedClassification> prepared_{};
};

/// Reusable staged transport. The canonical plan is authenticated collectively before any
/// rank-local peer classification or buffer/request allocation. The owning lane is borrowed only
/// after collective preparation succeeds.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class RegionTransport {
 public:
  static_assert(Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, MemorySpace>::accessible,
                "RegionTransport requires DefaultExecutionSpace access to MemorySpace");

  using plan_type = RegionTransferPlan<Dim>;
  using job_type = typename plan_type::job_type;
  using peer_plan_type = typename plan_type::peer_plan_type;
  using classification_type = typename plan_type::PreparedClassification;
  using device_buffer_type = Kokkos::View<Real*, MemorySpace>;
  using pinned_buffer_type = Kokkos::View<Real*, Kokkos::SharedHostPinnedSpace>;
  using execution_space = Kokkos::DefaultExecutionSpace;
  using execution_index_type = std::int64_t;
  using execution_policy =
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<execution_index_type>>;

  explicit RegionTransport(plan_type plan) noexcept : plan_(std::move(plan)) {}

  RegionTransport(const RegionTransport&) = delete;
  RegionTransport& operator=(const RegionTransport&) = delete;
  RegionTransport(RegionTransport&&) = delete;
  RegionTransport& operator=(RegionTransport&&) = delete;

  ~RegionTransport() {
#ifdef POPS_HAS_MPI
    // Destruction is deliberately non-collective. Any request escaping execute() is a violated
    // lifetime invariant and cannot be repaired safely from an independently ordered destructor.
    if (storage_ && (!all_null_(storage_->send_requests) || !all_null_(storage_->receive_requests)))
      std::terminate();
#endif
  }

  void prepare_collectively(const ExecutionLane& lane) {
    long invalid = 0;
    try {
      invalid = lane_ != nullptr ||
#ifdef POPS_HAS_MPI
                        !lane.active() || !lane.owns_communicator() ||
#endif
                        lane.identity().empty() ||
                        plan_.rank_space().size() >
                            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                        !plan_.rank_space().contains(plan_.local_rank()) ||
                        lane.size() != static_cast<int>(plan_.rank_space().size()) ||
                        lane.rank() !=
                            static_cast<int>(plan_.rank_space().linear_rank(plan_.local_rank()))
                    ? 1L
                    : 0L;
    } catch (...) {
      invalid = 1;
    }
    if (all_reduce_max(invalid, lane.communicator()) != 0)
      throw std::invalid_argument(
          "region transfer requires its exact owning duplicated ExecutionLane");

    std::string contract;
    long serialization_failure = 0;
    try {
      contract = plan_.exact_contract("prepared-region-transport");
    } catch (...) {
      serialization_failure = 1;
    }
    if (all_reduce_max(serialization_failure, lane.communicator()) != 0)
      throw std::runtime_error("region transfer contract serialization failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("pops-region-transfer-v3"), std::string_view(contract)}},
            lane.communicator()))
      throw std::invalid_argument("region transfer canonical schedule differs between MPI ranks");

    const bool remote_any =
        all_reduce_max(plan_.has_remote_jobs() ? 1L : 0L, lane.communicator()) != 0;

    std::unique_ptr<classification_type> classification;
    std::unique_ptr<PreparedStorage> storage;
    const execution_space* execution = nullptr;
    long allocation_failure = 0;
    try {
      ::pops::detail::ensure_kokkos_initialized();
      execution = &::pops::detail::default_execution_space();
      classification = plan_.classify_();
      storage = prepare_storage_(*classification);
    } catch (...) {
      allocation_failure = 1;
    }
    if (all_reduce_max(allocation_failure, lane.communicator()) != 0)
      throw std::runtime_error(
          "region transfer classification or storage preparation failed collectively");

    plan_.prepared_ = std::move(classification);
    storage_ = std::move(storage);
    execution_ = execution;
    lane_ = &lane;
    lane_borrow_.emplace(lane.borrow_immutably());
    remote_any_ = remote_any;
  }

  const plan_type& plan() const noexcept { return plan_; }
  bool sealed() const noexcept { return sealed_; }

  template <class SourceAccessor, class DestinationAccessor>
  void execute(SourceAccessor&& source, DestinationAccessor&& destination) {
    require_attached_();
    if (sealed_)
      throw std::runtime_error("region transfer is sealed after a collective failure");

    long validation_failure = 0;
    try {
      validate_jobs_(plan_.local_jobs(), source, destination);
      for (const auto& peer : storage_->peers) {
        if (peer.send != nullptr)
          validate_jobs_(peer.send->jobs, source, destination, true, false);
        if (peer.receive != nullptr)
          validate_jobs_(peer.receive->jobs, source, destination, false, true);
      }
    } catch (...) {
      validation_failure = 1;
    }
    if (!gate_(validation_failure))
      throw std::invalid_argument("region transfer source/destination binding failed collectively");

    long packing_failure = 0;
    try {
      pack_jobs_(plan_.local_jobs(), storage_->local_buffer, source);
      for (PeerStorage& peer : storage_->peers)
        if (peer.send != nullptr)
          pack_jobs_(peer.send->jobs, peer.device_send, source);
      if constexpr (std::is_same_v<execution_space, Kokkos::DefaultHostExecutionSpace>) {
        for (PeerStorage& peer : storage_->peers)
          if (peer.send != nullptr)
            std::copy_n(peer.device_send.data(), peer.send->elements, peer.host_send.data());
      } else {
        for (PeerStorage& peer : storage_->peers)
          if (peer.send != nullptr)
            Kokkos::deep_copy(*execution_, peer.host_send, peer.device_send);
        execution_fence_();
      }
    } catch (...) {
      packing_failure = 1;
    }
    if (!gate_(packing_failure))
      throw std::runtime_error("region transfer packing failed collectively");

#ifdef POPS_HAS_MPI
    if (remote_any_)
      exchange_remote_();
#else
    if (remote_any_)
      throw std::logic_error("remote region transfer requires an MPI build");
#endif

    long publication_failure = 0;
    try {
      unpack_jobs_(plan_.local_jobs(), storage_->local_buffer, destination);
      for (PeerStorage& peer : storage_->peers)
        if (peer.receive != nullptr)
          unpack_jobs_(peer.receive->jobs, peer.device_receive, destination);
      execution_fence_();
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

  struct PreparedStorage {
    device_buffer_type local_buffer{};
    std::vector<PeerStorage> peers{};
#ifdef POPS_HAS_MPI
    std::vector<MPI_Request> receive_requests{};
    std::vector<MPI_Request> send_requests{};
    std::vector<MPI_Status> receive_statuses{};
#endif
  };

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

  KernelJob lower_(const job_type& job) const {
    const std::size_t execution_max =
        static_cast<std::size_t>(std::numeric_limits<execution_index_type>::max());
    if (job.elements == 0 || job.elements > execution_max || job.offset > execution_max ||
        job.elements > execution_max - job.offset ||
        job.elements % static_cast<std::size_t>(plan_.ncomp()) != 0)
      throw std::overflow_error("region transfer job exceeds execution range");
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
        throw std::invalid_argument("region transfer source view is stale or incomplete");
      if (validate_destination && !detail::contains(FieldView<Real, Dim>(destination(job)),
                                                    job.destination_region, plan_.ncomp()))
        throw std::invalid_argument("region transfer destination view is stale or incomplete");
    }
  }

  template <class SourceAccessor>
  void pack_jobs_(const std::vector<job_type>& jobs, device_buffer_type buffer,
                  SourceAccessor& source) const {
    for (const job_type& job : jobs) {
      const KernelJob lowered = lower_(job);
      const PackKernel kernel{buffer, FieldView<const Real, Dim>(source(job)), lowered};
#if defined(KOKKOS_ENABLE_OPENMP) && defined(_OPENMP)
      if constexpr (std::is_same_v<execution_space, Kokkos::OpenMP>) {
#pragma omp parallel for schedule(static)
        for (execution_index_type element = 0; element < lowered.elements; ++element)
          kernel(element);
        continue;
      }
#endif
#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL)
      if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible) {
        for (execution_index_type element = 0; element < lowered.elements; ++element)
          kernel(element);
        continue;
      }
#endif
      Kokkos::parallel_for(pack_kernel_label_, execution_policy(*execution_, 0, lowered.elements),
                           kernel);
    }
  }

  template <class DestinationAccessor>
  void unpack_jobs_(const std::vector<job_type>& jobs, device_buffer_type buffer,
                    DestinationAccessor& destination) const {
    for (const job_type& job : jobs) {
      const KernelJob lowered = lower_(job);
      const UnpackKernel kernel{buffer, FieldView<Real, Dim>(destination(job)), lowered};
#if defined(KOKKOS_ENABLE_OPENMP) && defined(_OPENMP)
      if constexpr (std::is_same_v<execution_space, Kokkos::OpenMP>) {
#pragma omp parallel for schedule(static)
        for (execution_index_type element = 0; element < lowered.elements; ++element)
          kernel(element);
        continue;
      }
#endif
#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL)
      if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible) {
        for (execution_index_type element = 0; element < lowered.elements; ++element)
          kernel(element);
        continue;
      }
#endif
      Kokkos::parallel_for(unpack_kernel_label_, execution_policy(*execution_, 0, lowered.elements),
                           kernel);
    }
  }

  void execution_fence_() const {
    if constexpr (!std::is_same_v<execution_space, Kokkos::DefaultHostExecutionSpace>)
      execution_->fence(execution_fence_label_);
  }

  bool gate_(long failure) const { return all_reduce_max(failure, lane_->communicator()) == 0; }

  void require_attached_() const {
    if (lane_ == nullptr || !lane_borrow_ || !storage_ || execution_ == nullptr)
      throw std::logic_error("region transfer has no prepared ExecutionLane");
  }

  std::unique_ptr<PreparedStorage> prepare_storage_(
      const classification_type& classification) const {
    auto storage = std::make_unique<PreparedStorage>();
    storage->local_buffer =
        device_buffer_type("pops_region_transfer_local", classification.local_elements);
    if (classification.send.size() >
        std::numeric_limits<std::size_t>::max() - classification.receive.size())
      throw std::overflow_error("region transfer peer storage count exceeds size_t");
    const std::size_t plans = classification.send.size() + classification.receive.size();
    if (plans > storage->peers.max_size())
      throw std::length_error("region transfer peer storage exceeds vector capacity");
    storage->peers.reserve(plans);
    const auto attach = [this, &storage](const peer_plan_type& plan, bool send) {
      const std::size_t linear = plan_.rank_space().linear_rank(plan.peer);
      if (linear > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          plan.elements > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("region transfer peer payload exceeds MPI int range");
      auto found = std::find_if(
          storage->peers.begin(), storage->peers.end(),
          [linear](const PeerStorage& peer) { return peer.mpi_rank == static_cast<int>(linear); });
      if (found == storage->peers.end()) {
        storage->peers.push_back(PeerStorage{static_cast<int>(linear)});
        found = std::prev(storage->peers.end());
      }
      const peer_plan_type*& slot = send ? found->send : found->receive;
      if (slot != nullptr)
        throw std::logic_error("region transfer has a duplicate peer plan");
      slot = &plan;
    };
    for (const auto& plan : classification.send)
      attach(plan, true);
    for (const auto& plan : classification.receive)
      attach(plan, false);
    std::sort(storage->peers.begin(), storage->peers.end(),
              [](const PeerStorage& left, const PeerStorage& right) {
                return left.mpi_rank < right.mpi_rank;
              });
    for (PeerStorage& peer : storage->peers) {
      const std::size_t sends = peer.send == nullptr ? 0 : peer.send->elements;
      const std::size_t receives = peer.receive == nullptr ? 0 : peer.receive->elements;
      peer.device_send = device_buffer_type("pops_region_transfer_device_send", sends);
      peer.device_receive = device_buffer_type("pops_region_transfer_device_receive", receives);
      peer.host_send = pinned_buffer_type("pops_region_transfer_host_send", sends);
      peer.host_receive = pinned_buffer_type("pops_region_transfer_host_receive", receives);
    }
#ifdef POPS_HAS_MPI
    storage->receive_requests.resize(classification.receive.size(), MPI_REQUEST_NULL);
    storage->send_requests.resize(classification.send.size(), MPI_REQUEST_NULL);
    storage->receive_statuses.resize(classification.receive.size());
#endif
    return storage;
  }

#ifdef POPS_HAS_MPI
  enum class RequestPhaseConsensus { accepted, rejected };

  /// A collective exception is not a shared rejection: phase alignment is then unknown and no
  /// further MPI operation is safe while requests may be live. Only a completed consensus may
  /// authorize the coordinated drain below.
  RequestPhaseConsensus request_phase_consensus_(long failure) noexcept {
    try {
      return gate_(failure) ? RequestPhaseConsensus::accepted : RequestPhaseConsensus::rejected;
    } catch (...) {
      sealed_ = true;
      std::terminate();
    }
  }

  static int wait_all_(std::vector<MPI_Request>& requests) noexcept {
    if (requests.empty())
      return MPI_SUCCESS;
    if (requests.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      return MPI_ERR_COUNT;
    return MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
  }

  static bool all_null_(const std::vector<MPI_Request>& requests) noexcept {
    return std::all_of(requests.begin(), requests.end(),
                       [](MPI_Request request) { return request == MPI_REQUEST_NULL; });
  }

  /// Called only after every lane rank has completed the same posting-consensus gate. Receives are
  /// globally known to precede sends, so every posted send has a matching receive. Drain sends
  /// first; only then may unmatched receives from a partial send phase be cancelled and waited.
  bool drain_after_posting_consensus_() noexcept {
    (void)wait_all_(storage_->send_requests);
    if (!all_null_(storage_->send_requests))
      return false;
    for (MPI_Request& request : storage_->receive_requests)
      if (request != MPI_REQUEST_NULL)
        (void)MPI_Cancel(&request);
    (void)wait_all_(storage_->receive_requests);
    return all_null_(storage_->receive_requests);
  }

  [[noreturn]] void coordinated_failure_after_consensus_(const char* message) {
    sealed_ = true;
    if (!drain_after_posting_consensus_())
      std::terminate();
    throw std::runtime_error(message);
  }

  void exchange_remote_() {
    std::fill(storage_->receive_requests.begin(), storage_->receive_requests.end(),
              MPI_REQUEST_NULL);
    std::fill(storage_->send_requests.begin(), storage_->send_requests.end(), MPI_REQUEST_NULL);
    std::size_t receive_index = 0;
    int code = MPI_SUCCESS;
    for (PeerStorage& peer : storage_->peers)
      if (peer.receive != nullptr) {
        if (code == MPI_SUCCESS)
          code = MPI_Irecv(peer.host_receive.data(), static_cast<int>(peer.receive->elements),
                           pops::mpi_real_datatype(), peer.mpi_rank,
                           ExecutionLane::parallel_copy_message_tag, lane_->native_handle(),
                           &storage_->receive_requests[receive_index]);
        ++receive_index;
      }
    if (request_phase_consensus_(code == MPI_SUCCESS ? 0L : 1L) == RequestPhaseConsensus::rejected)
      coordinated_failure_after_consensus_("region transfer receive posting failed collectively");

    std::size_t send_index = 0;
    code = MPI_SUCCESS;
    for (PeerStorage& peer : storage_->peers)
      if (peer.send != nullptr) {
        if (code == MPI_SUCCESS)
          code = MPI_Isend(peer.host_send.data(), static_cast<int>(peer.send->elements),
                           pops::mpi_real_datatype(), peer.mpi_rank,
                           ExecutionLane::parallel_copy_message_tag, lane_->native_handle(),
                           &storage_->send_requests[send_index]);
        ++send_index;
      }
    if (request_phase_consensus_(code == MPI_SUCCESS ? 0L : 1L) == RequestPhaseConsensus::rejected)
      coordinated_failure_after_consensus_("region transfer send posting failed collectively");

    const int send_wait_code = wait_all_(storage_->send_requests);
    const int receive_wait_code =
        storage_->receive_requests.empty()
            ? MPI_SUCCESS
            : MPI_Waitall(static_cast<int>(storage_->receive_requests.size()),
                          storage_->receive_requests.data(), storage_->receive_statuses.data());
    code = send_wait_code == MPI_SUCCESS && receive_wait_code == MPI_SUCCESS ? MPI_SUCCESS
                                                                             : MPI_ERR_OTHER;
    if (code == MPI_SUCCESS && all_null_(storage_->send_requests) &&
        all_null_(storage_->receive_requests)) {
      std::size_t status_index = 0;
      for (const PeerStorage& peer : storage_->peers)
        if (peer.receive != nullptr) {
          int count = MPI_UNDEFINED;
          const MPI_Status& status = storage_->receive_statuses[status_index++];
          if (MPI_Get_count(&status, pops::mpi_real_datatype(), &count) != MPI_SUCCESS ||
              status.MPI_SOURCE != peer.mpi_rank ||
              status.MPI_TAG != ExecutionLane::parallel_copy_message_tag ||
              count != static_cast<int>(peer.receive->elements)) {
            code = MPI_ERR_OTHER;
            break;
          }
        }
    } else {
      code = MPI_ERR_OTHER;
    }
    if (request_phase_consensus_(code == MPI_SUCCESS ? 0L : 1L) == RequestPhaseConsensus::rejected)
      coordinated_failure_after_consensus_(
          "region transfer receive authentication or MPI wait failed collectively");

    long staging_failure = 0;
    try {
      if constexpr (std::is_same_v<execution_space, Kokkos::DefaultHostExecutionSpace>) {
        for (PeerStorage& peer : storage_->peers)
          if (peer.receive != nullptr)
            std::copy_n(peer.host_receive.data(), peer.receive->elements,
                        peer.device_receive.data());
      } else {
        for (PeerStorage& peer : storage_->peers)
          if (peer.receive != nullptr)
            Kokkos::deep_copy(*execution_, peer.device_receive, peer.host_receive);
        execution_fence_();
      }
    } catch (...) {
      staging_failure = 1;
    }
    if (!gate_(staging_failure)) {
      sealed_ = true;
      throw std::runtime_error(
          "region transfer receive staging failed collectively after a safe drain");
    }
  }
#endif

  plan_type plan_;
  const ExecutionLane* lane_ = nullptr;
  std::optional<ExecutionLane::ImmutableBorrow> lane_borrow_{};
  std::unique_ptr<PreparedStorage> storage_{};
  const execution_space* execution_ = nullptr;
  std::string pack_kernel_label_{"pops_region_transfer_pack"};
  std::string unpack_kernel_label_{"pops_region_transfer_unpack"};
  std::string execution_fence_label_{"pops.region-transfer.fence"};
  bool remote_any_ = false;
  bool sealed_ = false;
};

}  // namespace pops::mesh::parallel
