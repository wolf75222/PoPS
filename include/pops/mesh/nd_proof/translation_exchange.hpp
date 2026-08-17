/// @file
/// @brief Private blocking MPI lease for one exact ND translation schedule.
///
/// The borrowed ExecutionLane and TranslationSchedule must outlive this object.  This proof is
/// deliberately blocking: it has no begin/end state, pooling, mapped topology, GPUDirect, or
/// payload chunking.  An unsafe communication failure seals the lease permanently.

#pragma once

#include <pops/mesh/nd_proof/translation_schedule.hpp>
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
#include <utility>
#include <vector>

namespace pops::mesh::nd_proof {

struct TranslationExchangeContext {
  std::uint64_t context_generation = 0;
  std::uint64_t schedule_generation = 0;
  int tag = ExecutionLane::translation_message_tag;
  /// Private test seam. A non-negative rank makes that rank fail before any point-to-point post.
  int fail_allocation_rank = -1;
  int fail_receive_post_rank = -1;
  int fail_send_post_rank = -1;
  /// This injects after a real wait so ordinary failure tests never leave unmatched traffic.
  int fail_wait_rank = -1;
  /// Fail after real unpack/replay mutation but before completion publication.
  int fail_completion_rank = -1;
  /// Fail-stop-only seam: a selected rank makes cleanup unprovable and therefore terminates.
  int fail_drain_rank = -1;
};

enum class TranslationExchangeDiagnosticStage : unsigned char {
  none,
  receive_post,
  send_post,
  wait,
  completion,
};

template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class TranslationExchange {
 public:
  using schedule_type = TranslationSchedule<Dim, MemorySpace>;
  using multifab_type = ::pops::MultiFab<Dim, MemorySpace>;
  using rank_type = Index<Dim>;
  using device_buffer_type = typename schedule_type::buffer_type;
  using pinned_buffer_type = Kokkos::View<Real*, Kokkos::SharedHostPinnedSpace>;

  TranslationExchange(const schedule_type& schedule, const ExecutionLane& lane,
                      TranslationExchangeContext context)
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
    throw std::logic_error(
        "nd_proof::TranslationExchange requires an active owning MPI ExecutionLane");
#endif
  }

  TranslationExchange(const TranslationExchange&) = delete;
  TranslationExchange& operator=(const TranslationExchange&) = delete;
  TranslationExchange(TranslationExchange&&) = delete;
  TranslationExchange& operator=(TranslationExchange&&) = delete;

  ~TranslationExchange() noexcept {
#ifdef POPS_HAS_MPI
    drain_noexcept_();
#endif
  }

  [[nodiscard]] const schedule_type& schedule() const noexcept { return *schedule_; }
  [[nodiscard]] const ExecutionLane& lane() const noexcept { return *lane_; }
  [[nodiscard]] const TranslationExchangeContext& context() const noexcept { return context_; }
  [[nodiscard]] bool sealed() const noexcept { return sealed_; }
  [[nodiscard]] TranslationExchangeDiagnosticStage diagnostic_stage() const noexcept {
    return diagnostic_stage_;
  }
  [[nodiscard]] std::size_t peer_count() const noexcept { return peers_.size(); }
  [[nodiscard]] std::size_t send_buffer_elements() const noexcept { return send_elements_; }
  [[nodiscard]] std::size_t receive_buffer_elements() const noexcept { return receive_elements_; }
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

  void execute(multifab_type& fields, const ExecutionLane& lane) {
#ifdef POPS_HAS_MPI
    require_execute_lane_collectively_(lane);
    if (sealed_)
      throw std::runtime_error("nd_proof::TranslationExchange is sealed after an unsafe failure");
    if (live_request_count() != 0)
      throw std::logic_error("nd_proof::TranslationExchange has live MPI requests before execute");

    long prepost_failure = 0;
    try {
      schedule_->validate_fields(fields);
      for (PeerStorage& peer : peers_) {
        if (peer.send_elements != 0) {
          schedule_->pack(fields, peer.coordinate, peer.device_send);
          Kokkos::deep_copy(peer.host_send, peer.device_send);
        }
      }
      Kokkos::fence();
    } catch (...) {
      prepost_failure = 1;
    }
    if (all_reduce_max(prepost_failure, lane_->communicator()) != 0)
      throw std::runtime_error(
          "nd_proof::TranslationExchange pre-post validation, packing, or staging failed "
          "collectively");

    int receive_post_code = MPI_SUCCESS;
    for (PeerStorage& peer : peers_) {
      if (peer.receive_elements != 0 && receive_post_code == MPI_SUCCESS) {
        if (context_.fail_receive_post_rank == lane_->rank()) {
          receive_post_code = MPI_ERR_OTHER;
          break;
        }
        MPI_Request request = MPI_REQUEST_NULL;
        receive_post_code =
            MPI_Irecv(peer.host_receive.data(), static_cast<int>(peer.receive_elements), pops::mpi_real_datatype(),
                      peer.mpi_rank, context_.tag, lane_->native_handle(), &request);
        if (receive_post_code == MPI_SUCCESS)
          receive_requests_.push_back(request);
      }
    }
    if (!post_phase_gate_(receive_post_code == MPI_SUCCESS ? 0L : 1L,
                          TranslationExchangeDiagnosticStage::receive_post)) {
      seal_(TranslationExchangeDiagnosticStage::receive_post);
      require_proven_drain_(drain_receives_());
      throw std::runtime_error("nd_proof::TranslationExchange receive posting failed collectively");
    }

    int send_post_code = MPI_SUCCESS;
    for (PeerStorage& peer : peers_) {
      if (peer.send_elements != 0 && send_post_code == MPI_SUCCESS) {
        if (context_.fail_send_post_rank == lane_->rank()) {
          send_post_code = MPI_ERR_OTHER;
          break;
        }
        MPI_Request request = MPI_REQUEST_NULL;
        send_post_code =
            MPI_Isend(peer.host_send.data(), static_cast<int>(peer.send_elements), pops::mpi_real_datatype(),
                      peer.mpi_rank, context_.tag, lane_->native_handle(), &request);
        if (send_post_code == MPI_SUCCESS)
          send_requests_.push_back(request);
      }
    }
    if (!post_phase_gate_(send_post_code == MPI_SUCCESS ? 0L : 1L,
                          TranslationExchangeDiagnosticStage::send_post)) {
      seal_(TranslationExchangeDiagnosticStage::send_post);
      require_proven_drain_(drain_after_send_failure_());
      throw std::runtime_error("nd_proof::TranslationExchange send posting failed collectively");
    }

    int wait_code = wait_all_(send_requests_);
    if (wait_code == MPI_SUCCESS)
      wait_code = wait_all_(receive_requests_);
    if (wait_code == MPI_SUCCESS && context_.fail_wait_rank == lane_->rank())
      wait_code = MPI_ERR_OTHER;
    if (!post_phase_gate_(wait_code == MPI_SUCCESS ? 0L : 1L,
                          TranslationExchangeDiagnosticStage::wait)) {
      seal_(TranslationExchangeDiagnosticStage::wait);
      require_proven_drain_(drain_after_wait_failure_());
      throw std::runtime_error("nd_proof::TranslationExchange MPI_Waitall failed collectively");
    }
    receive_requests_.clear();
    send_requests_.clear();

    long completion_failure = 0;
    try {
      for (PeerStorage& peer : peers_)
        if (peer.receive_elements != 0) {
          Kokkos::deep_copy(peer.device_receive, peer.host_receive);
          Kokkos::fence();
          schedule_->unpack(fields, peer.coordinate, peer.device_receive);
        }
      schedule_->replay(fields);
      Kokkos::fence();
      if (context_.fail_completion_rank == lane_->rank())
        throw std::runtime_error("nd_proof::TranslationExchange injected completion failure");
    } catch (...) {
      completion_failure = 1;
    }
    if (!completion_phase_gate_(completion_failure)) {
      seal_(TranslationExchangeDiagnosticStage::completion);
      std::terminate();
    }
#else
    (void)fields;
    (void)lane;
    throw std::logic_error(
        "nd_proof::TranslationExchange requires an active owning MPI ExecutionLane");
#endif
  }

 private:
  struct PeerStorage {
    rank_type coordinate{};
    int mpi_rank = 0;
    std::size_t send_elements = 0;
    std::size_t receive_elements = 0;
    device_buffer_type device_send{};
    device_buffer_type device_receive{};
    pinned_buffer_type host_send{};
    pinned_buffer_type host_receive{};
  };

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
      append_i64_(bytes, index.values[axis]);
  }

  static void append_extent_(std::string& bytes, const Extent<Dim>& extent) {
    for (int axis = 0; axis < Dim; ++axis)
      append_i64_(bytes, extent.values[axis]);
  }

  static void append_box_(std::string& bytes, const Box<Dim>& box) {
    append_index_(bytes, box.lo);
    append_index_(bytes, box.hi);
  }

  std::string canonical_contract_() const {
    std::string bytes;
    append_string_(bytes, "nd-translation-v1");
    append_i64_(bytes, Dim);
    append_string_(bytes, lane_->identity());
    append_u64_(bytes, context_.context_generation);
    append_u64_(bytes, context_.schedule_generation);
    append_i64_(bytes, context_.tag);
    const auto& layout = schedule_->layout();
    append_u64_(bytes, layout.size());
    for (const Box<Dim>& box : layout.boxes())
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
    const PeriodicTopology<Dim>& topology = schedule_->topology();
    append_u64_(bytes, topology.identifications().size());
    for (const PeriodicIdentification<Dim>& identification : topology.identifications()) {
      append_i64_(bytes, identification.source().axis);
      append_i64_(bytes, static_cast<int>(identification.source().side));
      append_i64_(bytes, identification.target().axis);
      append_i64_(bytes, static_cast<int>(identification.target().side));
      for (int axis = 0; axis < Dim; ++axis) {
        append_i64_(bytes, identification.signed_permutation().target_axes()[axis]);
        append_i64_(bytes, identification.signed_permutation().signs()[axis]);
      }
    }
    append_extent_(bytes, schedule_->ghosts());
    append_i64_(bytes, schedule_->ncomp());
    append_i64_(bytes, schedule_->first_component());
    append_i64_(bytes, schedule_->component_count());
    append_u64_(bytes, schedule_->canonical_global_jobs().size());
    for (const auto& job : schedule_->canonical_global_jobs()) {
      append_u64_(bytes, job.ordinal);
      append_u64_(bytes, job.source_box);
      append_u64_(bytes, job.destination_box);
      append_box_(bytes, job.destination_region);
      for (int axis = 0; axis < Dim; ++axis)
        append_i64_(bytes, job.source_from_destination[axis]);
      append_u64_(bytes, job.elements);
    }
    return bytes;
  }

#ifdef POPS_HAS_MPI
  void validate_and_prepare_collectively_() {
    long invalid = 0;
    try {
      invalid = lane_ == nullptr || !lane_->active() || !lane_->owns_communicator() ||
                        lane_->identity().empty() || context_.context_generation == 0 ||
                        context_.schedule_generation == 0 ||
                        context_.tag != ExecutionLane::translation_message_tag ||
                        context_.tag != 2 || schedule_ == nullptr
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
      throw std::invalid_argument(
          "nd_proof::TranslationExchange lane or schedule binding is invalid");

    std::string contract;
    long serialization_failure = 0;
    try {
      contract = canonical_contract_();
    } catch (...) {
      serialization_failure = 1;
    }
    if (all_reduce_max(serialization_failure, lane_->communicator()) != 0)
      throw std::runtime_error(
          "nd_proof::TranslationExchange canonical contract serialization failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("nd-translation-v1"), std::string_view(contract)}},
            lane_->communicator()))
      throw std::invalid_argument(
          "nd_proof::TranslationExchange canonical schedule contract differs between ranks");

    long allocation_failure = 0;
    try {
      initialize_peers_();
      if (context_.fail_allocation_rank >= 0 && lane_->rank() == context_.fail_allocation_rank)
        throw std::bad_alloc();
      allocate_peer_storage_();
    } catch (...) {
      allocation_failure = 1;
    }
    if (all_reduce_max(allocation_failure, lane_->communicator()) != 0)
      throw std::runtime_error(
          "nd_proof::TranslationExchange reusable buffer preparation failed collectively");
  }

  void initialize_peers_() {
    const auto& ranks = schedule_->distribution().rank_space();
    const auto add = [this, &ranks](const typename schedule_type::PeerPlan& plan, bool send) {
      const std::size_t linear = ranks.linear_rank(plan.peer);
      if (linear > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          plan.elements > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error(
            "nd_proof::TranslationExchange peer payload exceeds MPI int range");
      auto found = std::find_if(peers_.begin(), peers_.end(), [linear](const PeerStorage& peer) {
        return peer.mpi_rank == static_cast<int>(linear);
      });
      if (found == peers_.end()) {
        peers_.push_back(PeerStorage{plan.peer, static_cast<int>(linear)});
        found = std::prev(peers_.end());
      }
      if (send)
        found->send_elements = plan.elements;
      else
        found->receive_elements = plan.elements;
    };
    peers_.clear();
    const std::size_t send_plans = schedule_->send_plan_count();
    const std::size_t receive_plans = schedule_->receive_plan_count();
    if (send_plans > std::numeric_limits<std::size_t>::max() - receive_plans)
      throw std::overflow_error("nd_proof::TranslationExchange request count overflows size_t");
    const std::size_t request_count = send_plans + receive_plans;
    if (request_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error(
          "nd_proof::TranslationExchange request count exceeds MPI int range");
    if (request_count > peers_.max_size() || receive_plans > receive_requests_.max_size() ||
        send_plans > send_requests_.max_size())
      throw std::length_error("nd_proof::TranslationExchange request vector capacity is invalid");
    peers_.reserve(request_count);
    for (const auto& plan : schedule_->send_plans())
      add(plan, true);
    for (const auto& plan : schedule_->receive_plans())
      add(plan, false);
    std::sort(peers_.begin(), peers_.end(), [](const PeerStorage& left, const PeerStorage& right) {
      return left.mpi_rank < right.mpi_rank;
    });
    send_elements_ = 0;
    receive_elements_ = 0;
    for (const PeerStorage& peer : peers_) {
      if (peer.send_elements > std::numeric_limits<std::size_t>::max() - send_elements_ ||
          peer.receive_elements > std::numeric_limits<std::size_t>::max() - receive_elements_)
        throw std::overflow_error(
            "nd_proof::TranslationExchange aggregate payload overflows size_t");
      send_elements_ += peer.send_elements;
      receive_elements_ += peer.receive_elements;
    }
    if (send_elements_ > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        receive_elements_ > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error(
          "nd_proof::TranslationExchange aggregate payload exceeds MPI int range");
  }

  void allocate_peer_storage_() {
    for (PeerStorage& peer : peers_) {
      peer.device_send = device_buffer_type("pops_nd_translation_send", peer.send_elements);
      peer.device_receive =
          device_buffer_type("pops_nd_translation_receive", peer.receive_elements);
      peer.host_send = pinned_buffer_type("pops_nd_translation_host_send", peer.send_elements);
      peer.host_receive =
          pinned_buffer_type("pops_nd_translation_host_receive", peer.receive_elements);
    }
    receive_requests_.reserve(schedule_->receive_plan_count());
    send_requests_.reserve(schedule_->send_plan_count());
  }

  void require_execute_lane_collectively_(const ExecutionLane& lane) const {
    long invalid = &lane != lane_ || !lane.active() || !lane.owns_communicator() ? 1L : 0L;
    if (all_reduce_max(invalid, lane_->communicator()) != 0)
      throw std::invalid_argument(
          "nd_proof::TranslationExchange execute requires its exact owning ExecutionLane object");
  }

  /// A phase consensus may itself fail while request handles are live.  At that point no
  /// cross-rank cleanup protocol can be trusted, so fail-stop preserves the buffers and handles.
  bool post_phase_gate_(long local_failure, TranslationExchangeDiagnosticStage stage) noexcept {
    try {
      return all_reduce_max(local_failure, lane_->communicator()) == 0;
    } catch (...) {
      seal_(stage);
      if (live_request_count() != 0)
        std::terminate();
      return false;
    }
  }

  /// Completion consensus occurs after field mutation.  If its collective transport is uncertain,
  /// no rank can safely infer whether another rank published the same field state.
  bool completion_phase_gate_(long local_failure) noexcept {
    try {
      return all_reduce_max(local_failure, lane_->communicator()) == 0;
    } catch (...) {
      seal_(TranslationExchangeDiagnosticStage::completion);
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

  void seal_(TranslationExchangeDiagnosticStage stage) noexcept {
    sealed_ = true;
    diagnostic_stage_ = stage;
  }

  static void require_proven_drain_(bool drained) {
    if (!drained)
      std::terminate();
  }

  void drain_noexcept_() noexcept {
    if (live_request_count() == 0)
      return;
    if (!detail::comm_active_unlocked())
      std::terminate();
    require_proven_drain_(drain_after_wait_failure_());
  }
#endif

  const schedule_type* schedule_ = nullptr;
  const ExecutionLane* lane_ = nullptr;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  TranslationExchangeContext context_{};
  std::vector<PeerStorage> peers_{};
  std::size_t send_elements_ = 0;
  std::size_t receive_elements_ = 0;
  bool sealed_ = false;
  TranslationExchangeDiagnosticStage diagnostic_stage_ = TranslationExchangeDiagnosticStage::none;
#ifdef POPS_HAS_MPI
  std::vector<MPI_Request> receive_requests_{};
  std::vector<MPI_Request> send_requests_{};
#endif
};

}  // namespace pops::mesh::nd_proof
