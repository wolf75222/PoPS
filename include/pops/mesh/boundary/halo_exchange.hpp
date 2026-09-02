/// @file
/// @brief Prepared asynchronous MPI transport for exact-ranked halo schedules.

#pragma once

#include <pops/mesh/boundary/halo_schedule.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {

/// Immutable identity and private failure seams for one prepared halo transport.
struct HaloExchangeContext {
  std::uint64_t context_generation = 0;
  std::uint64_t schedule_generation = 0;
  int tag = ExecutionLane::halo_message_tag;
  int fail_allocation_rank = -1;
  int fail_receive_post_rank = -1;
  int fail_send_post_rank = -1;
  int fail_wait_rank = -1;
  int fail_staging_rank = -1;
  int fail_publication_rank = -1;
  int fail_drain_rank = -1;
};

enum class HaloExchangeDiagnosticStage : unsigned char {
  none,
  binding,
  packing,
  receive_post,
  send_post,
  wait,
  staging,
  publication,
};

/// Reusable begin/complete halo transport bound to one exact schedule and one owning MPI lane.
///
/// `begin` only packs and posts communication.  Neither local nor remote ghosts are published
/// until `complete` has authenticated every receive and reached collective staging consensus.
/// The borrowed schedule and lane must outlive this object.  Independent concurrent exchanges
/// require independent ExecutionLane communicators.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class HaloExchange {
  static_assert(Dim >= 1 && Dim <= 3, "pops::HaloExchange supports dimensions 1, 2, and 3");
  static_assert(Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, MemorySpace>::accessible,
                "pops::HaloExchange requires DefaultExecutionSpace access to MemorySpace");

 public:
  static constexpr int dimension = Dim;

  using schedule_type = HaloSchedule<Dim>;
  using multifab_type = MultiFab<Dim, MemorySpace>;
  using rank_type = Index<Dim>;
  using peer_plan_type = HaloPeerPlan<Dim>;
  using job_type = HaloJob<Dim>;
  using device_buffer_type = Kokkos::View<Real*, MemorySpace>;
  using pinned_buffer_type = Kokkos::View<Real*, Kokkos::SharedHostPinnedSpace>;
  using execution_space = Kokkos::DefaultExecutionSpace;
  using execution_index_type = std::int64_t;
  using execution_policy =
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<execution_index_type>>;

  HaloExchange(const schedule_type& schedule, const ExecutionLane& lane,
               HaloExchangeContext context)
      : schedule_(&schedule),
        lane_(&lane),
        lane_borrow_(lane.borrow_immutably()),
        context_(context) {
#ifdef POPS_HAS_MPI
    validate_and_prepare_collectively_();
#else
    (void)schedule;
    (void)lane;
    (void)context;
    throw std::logic_error("pops::HaloExchange requires an active owning MPI ExecutionLane");
#endif
  }

  HaloExchange(const HaloExchange&) = delete;
  HaloExchange& operator=(const HaloExchange&) = delete;
  HaloExchange(HaloExchange&&) = delete;
  HaloExchange& operator=(HaloExchange&&) = delete;

  ~HaloExchange() noexcept {
#ifdef POPS_HAS_MPI
    drain_noexcept_();
#endif
  }

  [[nodiscard]] const schedule_type& schedule() const noexcept { return *schedule_; }
  [[nodiscard]] const ExecutionLane& lane() const noexcept { return *lane_; }
  [[nodiscard]] const HaloExchangeContext& context() const noexcept { return context_; }
  [[nodiscard]] bool in_flight() const noexcept { return in_flight_; }
  [[nodiscard]] bool sealed() const noexcept { return sealed_; }
  [[nodiscard]] HaloExchangeDiagnosticStage diagnostic_stage() const noexcept {
    return diagnostic_stage_;
  }
  [[nodiscard]] std::size_t peer_count() const noexcept { return peers_.size(); }
  [[nodiscard]] std::size_t send_buffer_elements() const noexcept {
    return schedule_->send_elements();
  }
  [[nodiscard]] std::size_t receive_buffer_elements() const noexcept {
    return schedule_->receive_elements();
  }
  [[nodiscard]] std::size_t live_request_count() const noexcept {
#ifdef POPS_HAS_MPI
    std::size_t live = 0;
    for (const MPI_Request request : receive_requests_)
      if (request != MPI_REQUEST_NULL)
        ++live;
    for (const MPI_Request request : send_requests_)
      if (request != MPI_REQUEST_NULL)
        ++live;
    return live;
#else
    return 0;
#endif
  }

  /// Dynamic resident storage of this transport; the borrowed schedule is intentionally excluded.
  [[nodiscard]] std::uint64_t resident_storage_bytes() const {
    const auto add = [](std::uint64_t& total, std::uint64_t value) {
      if (value > std::numeric_limits<std::uint64_t>::max() - total)
        throw std::overflow_error("pops::HaloExchange resident storage overflows uint64");
      total += value;
    };
    const auto vector_bytes = [](const auto& values) {
      using value_type = typename std::remove_reference_t<decltype(values)>::value_type;
      if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
        throw std::overflow_error("pops::HaloExchange vector storage overflows uint64");
      return static_cast<std::uint64_t>(values.capacity()) * sizeof(value_type);
    };
    const auto view_bytes = [](const auto& view) {
      if (view.extent(0) > std::numeric_limits<std::uint64_t>::max() / sizeof(Real))
        throw std::overflow_error("pops::HaloExchange payload storage overflows uint64");
      return static_cast<std::uint64_t>(view.extent(0)) * sizeof(Real);
    };
    const auto string_bytes = [](const std::string& value) {
      const auto object = reinterpret_cast<std::uintptr_t>(&value);
      const auto data = reinterpret_cast<std::uintptr_t>(value.data());
      if (data >= object && data < object + sizeof(value))
        return std::uint64_t{0};
      if (value.capacity() == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("pops::HaloExchange string storage overflows uint64");
      return static_cast<std::uint64_t>(value.capacity()) + 1U;
    };
    std::uint64_t total = 0;
    add(total, string_bytes(execution_fence_label_));
    add(total, vector_bytes(peers_));
    add(total, view_bytes(local_buffer_));
    add(total, vector_bytes(active_storage_));
    for (const PeerStorage& peer : peers_) {
      add(total, view_bytes(peer.device_send));
      add(total, view_bytes(peer.device_receive));
      add(total, view_bytes(peer.host_send));
      add(total, view_bytes(peer.host_receive));
    }
#ifdef POPS_HAS_MPI
    add(total, vector_bytes(receive_requests_));
    add(total, vector_bytes(send_requests_));
    add(total, vector_bytes(receive_statuses_));
#endif
    return total;
  }

  /// Pack and post every remote transfer without publishing any destination ghost.
  void begin(multifab_type& fields, const ExecutionLane& lane) {
#ifdef POPS_HAS_MPI
    const long invalid_state = sealed_ || in_flight_ || active_fields_ != nullptr ||
                                       live_request_count() != 0 || &lane != lane_ ||
                                       !lane.active() || !lane.owns_communicator()
                                   ? 1L
                                   : 0L;
    if (!phase_gate_(invalid_state, HaloExchangeDiagnosticStage::binding)) {
      seal_(HaloExchangeDiagnosticStage::binding);
      throw std::runtime_error("pops::HaloExchange begin binding is invalid collectively");
    }

    long packing_failure = 0;
    try {
      schedule_->authenticate(fields);
      bind_storage_(fields);
      if (schedule_->local_elements() != 0)
        pack_jobs_(fields, schedule_->local_jobs(), schedule_->local_elements(), local_buffer_);
      for (PeerStorage& peer : peers_)
        if (peer.send_plan != nullptr) {
          pack_plan_(fields, *peer.send_plan, peer.device_send);
          if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>)
            std::copy_n(peer.device_send.data(), peer.device_send.extent(0), peer.host_send.data());
          else
            Kokkos::deep_copy(peer.host_send, peer.device_send);
        }
      ::pops::device_fence(execution_fence_label_);
    } catch (...) {
      packing_failure = 1;
    }
    if (!phase_gate_(packing_failure, HaloExchangeDiagnosticStage::packing)) {
      seal_(HaloExchangeDiagnosticStage::packing);
      active_storage_.clear();
      throw std::runtime_error(
          "pops::HaloExchange validation, packing, or host staging failed collectively");
    }

    int receive_post_code = MPI_SUCCESS;
    for (PeerStorage& peer : peers_) {
      if (peer.receive_plan == nullptr || receive_post_code != MPI_SUCCESS)
        continue;
      if (context_.fail_receive_post_rank == lane_->rank()) {
        receive_post_code = MPI_ERR_OTHER;
        break;
      }
      receive_requests_.push_back(MPI_REQUEST_NULL);
      receive_post_code =
          MPI_Irecv(peer.host_receive.data(), static_cast<int>(peer.receive_plan->elements),
                    mpi_real_datatype(), peer.mpi_rank, context_.tag, lane_->native_handle(),
                    &receive_requests_.back());
      if (receive_post_code != MPI_SUCCESS)
        receive_requests_.pop_back();
    }
    if (!phase_gate_(receive_post_code == MPI_SUCCESS ? 0L : 1L,
                     HaloExchangeDiagnosticStage::receive_post)) {
      seal_(HaloExchangeDiagnosticStage::receive_post);
      require_proven_drain_(drain_receives_());
      active_storage_.clear();
      throw std::runtime_error("pops::HaloExchange receive posting failed collectively");
    }

    int send_post_code = MPI_SUCCESS;
    for (PeerStorage& peer : peers_) {
      if (peer.send_plan == nullptr || send_post_code != MPI_SUCCESS)
        continue;
      if (context_.fail_send_post_rank == lane_->rank()) {
        send_post_code = MPI_ERR_OTHER;
        break;
      }
      send_requests_.push_back(MPI_REQUEST_NULL);
      send_post_code = MPI_Isend(peer.host_send.data(), static_cast<int>(peer.send_plan->elements),
                                 mpi_real_datatype(), peer.mpi_rank, context_.tag,
                                 lane_->native_handle(), &send_requests_.back());
      if (send_post_code != MPI_SUCCESS)
        send_requests_.pop_back();
    }
    if (!phase_gate_(send_post_code == MPI_SUCCESS ? 0L : 1L,
                     HaloExchangeDiagnosticStage::send_post)) {
      seal_(HaloExchangeDiagnosticStage::send_post);
      require_proven_drain_(drain_after_send_failure_());
      active_storage_.clear();
      throw std::runtime_error("pops::HaloExchange send posting failed collectively");
    }

    active_fields_ = &fields;
    in_flight_ = true;
#else
    (void)fields;
    (void)lane;
    throw std::logic_error("pops::HaloExchange requires an active owning MPI ExecutionLane");
#endif
  }

  /// Complete the authenticated transfers, then publish remote and local ghosts together.
  void complete(multifab_type& fields, const ExecutionLane& lane) {
#ifdef POPS_HAS_MPI
    long invalid_binding = sealed_ || !in_flight_ || active_fields_ != &fields || &lane != lane_ ||
                                   !lane.active() || !lane.owns_communicator()
                               ? 1L
                               : 0L;
    if (invalid_binding == 0 && !storage_matches_(fields))
      invalid_binding = 1;
    if (!phase_gate_(invalid_binding, HaloExchangeDiagnosticStage::binding)) {
      seal_(HaloExchangeDiagnosticStage::binding);
      require_proven_drain_(drain_after_wait_failure_());
      reset_in_flight_();
      throw std::runtime_error("pops::HaloExchange complete binding is invalid collectively");
    }

    int wait_code = wait_all_(send_requests_);
    if (wait_code == MPI_SUCCESS)
      wait_code = wait_receives_authenticated_();
    if (wait_code == MPI_SUCCESS && context_.fail_wait_rank == lane_->rank())
      wait_code = MPI_ERR_OTHER;
    if (!phase_gate_(wait_code == MPI_SUCCESS ? 0L : 1L, HaloExchangeDiagnosticStage::wait)) {
      seal_(HaloExchangeDiagnosticStage::wait);
      require_proven_drain_(drain_after_wait_failure_());
      reset_in_flight_();
      throw std::runtime_error(
          "pops::HaloExchange receive authentication or MPI_Waitall failed collectively");
    }
    receive_requests_.clear();
    send_requests_.clear();

    long staging_failure = 0;
    try {
      for (PeerStorage& peer : peers_)
        if (peer.receive_plan != nullptr) {
          if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>)
            std::copy_n(peer.host_receive.data(), peer.host_receive.extent(0),
                        peer.device_receive.data());
          else
            Kokkos::deep_copy(peer.device_receive, peer.host_receive);
        }
      ::pops::device_fence(execution_fence_label_);
      if (context_.fail_staging_rank == lane_->rank())
        throw std::runtime_error("pops::HaloExchange injected receive staging failure");
    } catch (...) {
      staging_failure = 1;
    }
    if (!phase_gate_(staging_failure, HaloExchangeDiagnosticStage::staging)) {
      seal_(HaloExchangeDiagnosticStage::staging);
      reset_in_flight_();
      throw std::runtime_error("pops::HaloExchange receive staging failed collectively");
    }

    long publication_failure = 0;
    try {
      for (PeerStorage& peer : peers_)
        if (peer.receive_plan != nullptr)
          unpack_plan_(fields, *peer.receive_plan, peer.device_receive);
      if (schedule_->local_elements() != 0)
        unpack_jobs_(fields, schedule_->local_jobs(), schedule_->local_elements(), local_buffer_);
      ::pops::device_fence(execution_fence_label_);
      if (context_.fail_publication_rank == lane_->rank())
        throw std::runtime_error("pops::HaloExchange injected publication failure");
    } catch (...) {
      publication_failure = 1;
    }
    if (!completion_gate_(publication_failure)) {
      seal_(HaloExchangeDiagnosticStage::publication);
      std::terminate();
    }
    reset_in_flight_();
#else
    (void)fields;
    (void)lane;
    throw std::logic_error("pops::HaloExchange requires an active owning MPI ExecutionLane");
#endif
  }

  void execute(multifab_type& fields, const ExecutionLane& lane) {
    begin(fields, lane);
    complete(fields, lane);
  }

 private:
  struct PeerStorage {
    rank_type coordinate{};
    int mpi_rank = 0;
    const peer_plan_type* send_plan = nullptr;
    const peer_plan_type* receive_plan = nullptr;
    device_buffer_type device_send{};
    device_buffer_type device_receive{};
    pinned_buffer_type host_send{};
    pinned_buffer_type host_receive{};
  };

 public:
  [[nodiscard]] static std::uint64_t configured_metadata_storage_upper_bound(
      std::uint64_t remote_peer_bound, std::uint64_t local_fab_bound) {
    const auto product = [](std::uint64_t left, std::uint64_t right) {
      if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
        throw std::overflow_error("pops::HaloExchange configured metadata overflows uint64");
      return left * right;
    };
    const auto add = [](std::uint64_t& total, std::uint64_t value) {
      if (value > std::numeric_limits<std::uint64_t>::max() - total)
        throw std::overflow_error("pops::HaloExchange configured metadata overflows uint64");
      total += value;
    };
    const std::uint64_t peer_slots = product(2U, remote_peer_bound);
    std::uint64_t total = product(peer_slots, sizeof(PeerStorage));
    add(total, product(local_fab_bound, sizeof(const Real*)));
#ifdef POPS_HAS_MPI
    add(total, product(remote_peer_bound, sizeof(MPI_Request)));
    add(total, product(remote_peer_bound, sizeof(MPI_Request)));
    add(total, product(remote_peer_bound, sizeof(MPI_Status)));
#endif
    return total;
  }

 private:
  struct KernelJob {
    int destination_lower[Dim]{};
    execution_index_type destination_extent[Dim]{};
    std::int64_t source_translation[Dim]{};
    int component_count = 0;
    execution_index_type cells_per_component = 0;
    execution_index_type offset = 0;
    execution_index_type elements = 0;
  };

  struct PackKernel {
    device_buffer_type buffer{};
    FieldView<const Real, Dim> source{};
    KernelJob job{};

    POPS_HD void operator()(execution_index_type element) const {
      const int component = static_cast<int>(element / job.cells_per_component);
      execution_index_type cell = element % job.cells_per_component;
      Index<Dim> source_index{};
      for (int axis = 0; axis < Dim; ++axis) {
        const std::int64_t coordinate = static_cast<std::int64_t>(job.destination_lower[axis]) +
                                        cell % job.destination_extent[axis];
        source_index[axis] = static_cast<int>(coordinate + job.source_translation[axis]);
        cell /= job.destination_extent[axis];
      }
      buffer(job.offset + element) = source(source_index, component);
    }
  };

  struct UnpackKernel {
    device_buffer_type buffer{};
    FieldView<Real, Dim> destination{};
    KernelJob job{};

    POPS_HD void operator()(execution_index_type element) const {
      const int component = static_cast<int>(element / job.cells_per_component);
      execution_index_type cell = element % job.cells_per_component;
      Index<Dim> destination_index{};
      for (int axis = 0; axis < Dim; ++axis) {
        const std::int64_t coordinate = static_cast<std::int64_t>(job.destination_lower[axis]) +
                                        cell % job.destination_extent[axis];
        destination_index[axis] = static_cast<int>(coordinate);
        cell /= job.destination_extent[axis];
      }
      destination(destination_index, component) = buffer(job.offset + element);
    }
  };

  KernelJob lower_job_(const job_type& job) const {
    const std::size_t execution_max =
        static_cast<std::size_t>(std::numeric_limits<execution_index_type>::max());
    if (job.elements == 0 || job.elements > execution_max || job.offset > execution_max ||
        job.elements > execution_max - job.offset ||
        job.elements % static_cast<std::size_t>(schedule_->ncomp()) != 0)
      throw std::overflow_error("pops::HaloExchange job exceeds execution range");
    KernelJob result{};
    result.component_count = schedule_->ncomp();
    result.cells_per_component = static_cast<execution_index_type>(
        job.elements / static_cast<std::size_t>(schedule_->ncomp()));
    result.offset = static_cast<execution_index_type>(job.offset);
    result.elements = static_cast<execution_index_type>(job.elements);
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t extent = job.destination_region.length(axis);
      if (extent <= 0)
        throw std::invalid_argument("pops::HaloExchange job region must be non-empty");
      result.destination_lower[axis] = job.destination_region.lo[axis];
      result.destination_extent[axis] = extent;
      result.source_translation[axis] = job.source_from_destination[axis];
    }
    return result;
  }

  void pack_jobs_(const multifab_type& fields, const std::vector<job_type>& jobs,
                  std::size_t elements, device_buffer_type buffer) const {
    if (buffer.extent(0) != elements)
      throw std::invalid_argument("pops::HaloExchange pack buffer does not match its job plan");
    for (const job_type& job : jobs) {
      const FieldView<const Real, Dim> source = fields.fab_global(job.source_box).view();
      const KernelJob lowered = lower_job_(job);
      Kokkos::parallel_for("pops_halo_pack", execution_policy(0, lowered.elements),
                           PackKernel{buffer, source, lowered});
    }
  }

  void pack_plan_(const multifab_type& fields, const peer_plan_type& plan,
                  device_buffer_type buffer) const {
    pack_jobs_(fields, plan.jobs, plan.elements, buffer);
  }

  void unpack_jobs_(multifab_type& fields, const std::vector<job_type>& jobs, std::size_t elements,
                    device_buffer_type buffer) const {
    if (buffer.extent(0) != elements)
      throw std::invalid_argument("pops::HaloExchange unpack buffer does not match its job plan");
    for (const job_type& job : jobs) {
      const FieldView<Real, Dim> destination = fields.fab_global(job.destination_box).view();
      const KernelJob lowered = lower_job_(job);
      Kokkos::parallel_for("pops_halo_unpack", execution_policy(0, lowered.elements),
                           UnpackKernel{buffer, destination, lowered});
    }
  }

  void unpack_plan_(multifab_type& fields, const peer_plan_type& plan,
                    device_buffer_type buffer) const {
    unpack_jobs_(fields, plan.jobs, plan.elements, buffer);
  }

  void bind_storage_(const multifab_type& fields) {
    active_storage_.clear();
    for (std::size_t local = 0; local < fields.local_size(); ++local)
      active_storage_.push_back(fields.fab(local).view().data);
  }

  bool storage_matches_(const multifab_type& fields) const noexcept {
    try {
      schedule_->authenticate(fields);
      if (fields.local_size() != active_storage_.size())
        return false;
      for (std::size_t local = 0; local < fields.local_size(); ++local)
        if (fields.fab(local).view().data != active_storage_[local])
          return false;
      return true;
    } catch (...) {
      return false;
    }
  }

  static void append_u64_(std::string& bytes, std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes.push_back(static_cast<char>((value >> shift) & 0xffu));
  }

  static void append_i64_(std::string& bytes, std::int64_t value) {
    append_u64_(bytes, static_cast<std::uint64_t>(value));
  }

  static void append_string_(std::string& bytes, std::string_view value) {
    append_u64_(bytes, value.size());
    bytes.append(value.data(), value.size());
  }

  static void append_index_(std::string& bytes, const Index<Dim>& index) {
    for (int axis = 0; axis < Dim; ++axis)
      append_i64_(bytes, index[axis]);
  }

  static void append_extent_(std::string& bytes, const Extent<Dim>& extent) {
    for (int axis = 0; axis < Dim; ++axis)
      append_i64_(bytes, extent[axis]);
  }

  static void append_box_(std::string& bytes, const Box<Dim>& box) {
    append_index_(bytes, box.lo);
    append_index_(bytes, box.hi);
  }

  std::string canonical_contract_() const {
    std::string bytes;
    append_string_(bytes, "pops-halo-exchange-v1");
    append_i64_(bytes, Dim);
    append_string_(bytes, lane_->identity());
    append_u64_(bytes, context_.context_generation);
    append_u64_(bytes, context_.schedule_generation);
    append_i64_(bytes, context_.tag);
    append_u64_(bytes, schedule_->layout().size());
    for (const Box<Dim>& box : schedule_->layout().boxes())
      append_box_(bytes, box);
    const auto& distribution = schedule_->distribution();
    append_i64_(bytes, static_cast<int>(distribution.mode()));
    append_index_(bytes, distribution.rank_space().origin());
    append_extent_(bytes, distribution.rank_space().extent());
    append_u64_(bytes, distribution.box_count());
    if (!distribution.replicated())
      for (std::size_t box = 0; box < distribution.box_count(); ++box)
        append_index_(bytes, distribution.owner(box));
    append_box_(bytes, schedule_->domain());
    append_i64_(bytes, static_cast<int>(schedule_->coverage()));
    for (const BoundaryFaceRecord<Dim>& record : schedule_->topology().faces()) {
      append_i64_(bytes, record.face.axis);
      append_i64_(bytes, static_cast<int>(record.face.side));
      append_i64_(bytes, static_cast<int>(record.kind));
      append_i64_(bytes, record.partner.axis);
      append_i64_(bytes, static_cast<int>(record.partner.side));
    }
    append_extent_(bytes, schedule_->ghosts());
    append_i64_(bytes, schedule_->ncomp());
    append_u64_(bytes, schedule_->canonical_jobs().size());
    for (const job_type& job : schedule_->canonical_jobs()) {
      append_u64_(bytes, job.source_box);
      append_u64_(bytes, job.destination_box);
      append_box_(bytes, job.destination_region);
      append_index_(bytes, job.source_from_destination);
      append_u64_(bytes, job.elements);
    }
    return bytes;
  }

#ifdef POPS_HAS_MPI
  void validate_and_prepare_collectively_() {
    long invalid = 0;
    try {
      invalid = lane_ == nullptr || !lane_->active() || !lane_->owns_communicator() ||
                        lane_->identity().empty() || schedule_ == nullptr ||
                        context_.context_generation == 0 || context_.schedule_generation == 0 ||
                        context_.tag != ExecutionLane::halo_message_tag
                    ? 1L
                    : 0L;
      if (invalid == 0) {
        const auto& ranks = schedule_->distribution().rank_space();
        if (ranks.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            lane_->size() != static_cast<int>(ranks.size()) ||
            lane_->rank() != static_cast<int>(ranks.linear_rank(schedule_->local_rank())))
          invalid = 1;
        int* tag_upper_bound = nullptr;
        int flag = 0;
        if (MPI_Comm_get_attr(lane_->native_handle(), MPI_TAG_UB, &tag_upper_bound, &flag) !=
                MPI_SUCCESS ||
            flag == 0 || tag_upper_bound == nullptr || context_.tag > *tag_upper_bound)
          invalid = 1;
      }
    } catch (...) {
      invalid = 1;
    }
    if (all_reduce_max(invalid, lane_->communicator()) != 0)
      throw std::invalid_argument("pops::HaloExchange lane or schedule binding is invalid");

    std::string contract;
    long serialization_failure = 0;
    try {
      contract = canonical_contract_();
    } catch (...) {
      serialization_failure = 1;
    }
    if (all_reduce_max(serialization_failure, lane_->communicator()) != 0)
      throw std::runtime_error(
          "pops::HaloExchange canonical contract serialization failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("pops-halo-exchange-v1"), std::string_view(contract)}},
            lane_->communicator()))
      throw std::invalid_argument(
          "pops::HaloExchange canonical schedule contract differs between ranks");

    long allocation_failure = 0;
    try {
      initialize_peers_();
      if (context_.fail_allocation_rank == lane_->rank())
        throw std::bad_alloc();
      allocate_peer_storage_();
    } catch (...) {
      allocation_failure = 1;
    }
    if (all_reduce_max(allocation_failure, lane_->communicator()) != 0)
      throw std::runtime_error(
          "pops::HaloExchange reusable staging preparation failed collectively");
  }

  void initialize_peers_() {
    const auto& ranks = schedule_->distribution().rank_space();
    const auto add = [this, &ranks](const peer_plan_type& plan, bool send) {
      const std::size_t linear = ranks.linear_rank(plan.peer);
      if (linear > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          plan.elements > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("pops::HaloExchange peer payload exceeds MPI int range");
      auto found = std::find_if(peers_.begin(), peers_.end(), [linear](const PeerStorage& peer) {
        return peer.mpi_rank == static_cast<int>(linear);
      });
      if (found == peers_.end()) {
        PeerStorage storage{};
        storage.coordinate = plan.peer;
        storage.mpi_rank = static_cast<int>(linear);
        peers_.push_back(std::move(storage));
        found = std::prev(peers_.end());
      }
      const peer_plan_type*& slot = send ? found->send_plan : found->receive_plan;
      if (slot != nullptr)
        throw std::logic_error("pops::HaloExchange duplicate peer plan");
      slot = &plan;
    };

    peers_.clear();
    const std::size_t send_plans = schedule_->send_plans().size();
    const std::size_t receive_plans = schedule_->receive_plans().size();
    if (send_plans > std::numeric_limits<std::size_t>::max() - receive_plans)
      throw std::overflow_error("pops::HaloExchange request count overflows size_t");
    const std::size_t request_count = send_plans + receive_plans;
    if (request_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        request_count > peers_.max_size() || receive_plans > receive_requests_.max_size() ||
        send_plans > send_requests_.max_size())
      throw std::length_error("pops::HaloExchange request capacity is invalid");
    peers_.reserve(request_count);
    for (const peer_plan_type& plan : schedule_->send_plans())
      add(plan, true);
    for (const peer_plan_type& plan : schedule_->receive_plans())
      add(plan, false);
    std::sort(peers_.begin(), peers_.end(), [](const PeerStorage& left, const PeerStorage& right) {
      return left.mpi_rank < right.mpi_rank;
    });
  }

  void allocate_peer_storage_() {
    local_buffer_ = device_buffer_type("pops_halo_local_staging", schedule_->local_elements());
    std::size_t local_fabs = 0;
    for (std::size_t box = 0; box < schedule_->layout().size(); ++box)
      if (schedule_->distribution().is_local(box, schedule_->local_rank()))
        ++local_fabs;
    active_storage_.reserve(local_fabs);
    for (PeerStorage& peer : peers_) {
      const std::size_t send_elements = peer.send_plan == nullptr ? 0 : peer.send_plan->elements;
      const std::size_t receive_elements =
          peer.receive_plan == nullptr ? 0 : peer.receive_plan->elements;
      peer.device_send = device_buffer_type("pops_halo_device_send", send_elements);
      peer.device_receive = device_buffer_type("pops_halo_device_receive", receive_elements);
      peer.host_send = pinned_buffer_type("pops_halo_host_send", send_elements);
      peer.host_receive = pinned_buffer_type("pops_halo_host_receive", receive_elements);
    }
    receive_requests_.reserve(schedule_->receive_plans().size());
    send_requests_.reserve(schedule_->send_plans().size());
    receive_statuses_.resize(schedule_->receive_plans().size());
  }

  bool phase_gate_(long local_failure, HaloExchangeDiagnosticStage stage) noexcept {
    try {
      return all_reduce_max(local_failure, lane_->communicator()) == 0;
    } catch (...) {
      seal_(stage);
      if (live_request_count() != 0)
        std::terminate();
      return false;
    }
  }

  bool completion_gate_(long local_failure) noexcept {
    try {
      return all_reduce_max(local_failure, lane_->communicator()) == 0;
    } catch (...) {
      seal_(HaloExchangeDiagnosticStage::publication);
      std::terminate();
    }
  }

  static bool all_null_(const std::vector<MPI_Request>& requests) noexcept {
    return std::all_of(requests.begin(), requests.end(),
                       [](MPI_Request request) { return request == MPI_REQUEST_NULL; });
  }

  static int wait_all_(std::vector<MPI_Request>& requests) noexcept {
    if (requests.empty())
      return MPI_SUCCESS;
    if (requests.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      return MPI_ERR_COUNT;
    return MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
  }

  int wait_receives_authenticated_() noexcept {
    if (receive_requests_.empty())
      return MPI_SUCCESS;
    if (receive_requests_.size() > receive_statuses_.size() ||
        receive_requests_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      return MPI_ERR_COUNT;
    const int wait_code = MPI_Waitall(static_cast<int>(receive_requests_.size()),
                                      receive_requests_.data(), receive_statuses_.data());
    if (wait_code != MPI_SUCCESS)
      return wait_code;

    std::size_t status_index = 0;
    for (const PeerStorage& peer : peers_) {
      if (peer.receive_plan == nullptr)
        continue;
      const MPI_Status& status = receive_statuses_[status_index++];
      int count = MPI_UNDEFINED;
      const int count_code = MPI_Get_count(&status, mpi_real_datatype(), &count);
      if (count_code != MPI_SUCCESS || count == MPI_UNDEFINED ||
          status.MPI_SOURCE != peer.mpi_rank || status.MPI_TAG != context_.tag ||
          count != static_cast<int>(peer.receive_plan->elements))
        return MPI_ERR_OTHER;
    }
    return status_index == receive_requests_.size() ? MPI_SUCCESS : MPI_ERR_OTHER;
  }

  bool drain_receives_() noexcept {
    bool drained = true;
    for (MPI_Request& request : receive_requests_)
      if (request != MPI_REQUEST_NULL && MPI_Cancel(&request) != MPI_SUCCESS)
        drained = false;
    if (wait_all_(receive_requests_) != MPI_SUCCESS)
      drained = false;
    if (context_.fail_drain_rank == lane_->rank())
      drained = false;
    if (drained && all_null_(receive_requests_))
      receive_requests_.clear();
    return drained && all_null_(receive_requests_) && send_requests_.empty();
  }

  bool drain_after_send_failure_() noexcept {
    if (wait_all_(send_requests_) != MPI_SUCCESS || !all_null_(send_requests_))
      return false;
    send_requests_.clear();
    return drain_receives_();
  }

  bool drain_after_wait_failure_() noexcept {
    if (!all_null_(send_requests_) &&
        (wait_all_(send_requests_) != MPI_SUCCESS || !all_null_(send_requests_)))
      return false;
    send_requests_.clear();
    return drain_receives_();
  }

  void seal_(HaloExchangeDiagnosticStage stage) noexcept {
    if (!sealed_) {
      sealed_ = true;
      diagnostic_stage_ = stage;
    }
  }

  static void require_proven_drain_(bool drained) {
    if (!drained)
      std::terminate();
  }

  void reset_in_flight_() noexcept {
    in_flight_ = false;
    active_fields_ = nullptr;
    active_storage_.clear();
  }

  void drain_noexcept_() noexcept {
    if (live_request_count() == 0)
      return;
    if (!detail::comm_active_unlocked())
      std::terminate();
    require_proven_drain_(drain_after_wait_failure_());
    reset_in_flight_();
  }
#endif

  const schedule_type* schedule_ = nullptr;
  const ExecutionLane* lane_ = nullptr;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  HaloExchangeContext context_{};
  std::string execution_fence_label_ = "pops.halo-exchange.fence";
  std::vector<PeerStorage> peers_{};
  device_buffer_type local_buffer_{};
  std::vector<const Real*> active_storage_{};
  multifab_type* active_fields_ = nullptr;
  bool in_flight_ = false;
  bool sealed_ = false;
  HaloExchangeDiagnosticStage diagnostic_stage_ = HaloExchangeDiagnosticStage::none;
#ifdef POPS_HAS_MPI
  std::vector<MPI_Request> receive_requests_{};
  std::vector<MPI_Request> send_requests_{};
  std::vector<MPI_Status> receive_statuses_{};
#endif
};

}  // namespace pops
