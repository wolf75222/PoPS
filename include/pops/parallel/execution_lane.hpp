#pragma once

/// @file
/// @brief Owning MPI communicator lane for independently ordered native execution.
///
/// MPI_THREAD_MULTIPLE permits concurrent MPI calls, but it does not make two unrelated collective
/// traces on the same communicator distinguishable.  An ExecutionLane therefore owns a duplicate
/// of an authenticated parent communicator with its own MPI context id. Prepared resources create
/// lanes collectively, in one canonical order, before worker threads start; numerical hot paths
/// only borrow the resulting communicator. Point-to-point operation tags are namespaced by that
/// duplicated communicator.

#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {

/// Borrowed communicator authority from which independent execution lanes are materialized. The
/// field storage rank space is currently the process-world rank space, so an embedding-owned
/// communicator must contain the same ranks in the same order (MPI_IDENT or MPI_CONGRUENT). The
/// owner keeps it alive only while children are duplicated; every resulting ExecutionLane owns its
/// own communicator and no hot path depends on the parent.
class ExecutionCommunicator {
 public:
  [[nodiscard]] static ExecutionCommunicator world() {
#ifdef POPS_HAS_MPI
    detail::ensure_mpi_world_initialized();
    return ExecutionCommunicator(StaticIdentity{}, "MPI_COMM_WORLD", MPI_COMM_WORLD);
#else
    return ExecutionCommunicator(StaticIdentity{}, "serial");
#endif
  }

#ifdef POPS_HAS_MPI
  [[nodiscard]] static ExecutionCommunicator borrowed(std::string_view identity,
                                                      MPI_Comm communicator) {
    detail::ensure_mpi_world_initialized();
    if (communicator == MPI_COMM_NULL)
      throw std::invalid_argument(
          "borrowed execution communicator requires a live supplied MPI_Comm");
    const CommunicatorView supplied{communicator};
    int world_relation = MPI_UNEQUAL;
    const int compare_code = MPI_Comm_compare(MPI_COMM_WORLD, communicator, &world_relation);
    if (all_reduce_max(compare_code == MPI_SUCCESS ? 0L : 1L, supplied) != 0)
      throw std::runtime_error(
          "MPI_Comm_compare(borrowed execution communicator) failed on a supplied rank");
    const long invalid_world_rank_space =
        world_relation == MPI_IDENT || world_relation == MPI_CONGRUENT ? 0L : 1L;
    if (all_reduce_max(invalid_world_rank_space, supplied) != 0)
      throw std::invalid_argument(
          "borrowed execution communicator must preserve the MPI_COMM_WORLD rank space and order "
          "used by field storage");
    if (all_reduce_max(identity.empty() ? 1L : 0L, supplied) != 0)
      throw std::invalid_argument("borrowed execution communicator requires an identity");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("execution-communicator"), identity}}, supplied))
      throw std::invalid_argument(
          "borrowed execution communicator identity differs between MPI ranks");
    std::string owned_identity;
    long identity_allocation_failure_local = 0;
    try {
      owned_identity.assign(identity);
    } catch (...) {
      identity_allocation_failure_local = 1;
    }
    if (all_reduce_max(identity_allocation_failure_local, supplied) != 0)
      throw std::bad_alloc();
    return ExecutionCommunicator(std::move(owned_identity), communicator);
  }

  [[nodiscard]] MPI_Comm native_handle() const noexcept { return communicator_; }
#endif

  [[nodiscard]] std::string_view identity() const noexcept {
    return static_identity_.empty() ? std::string_view(identity_) : static_identity_;
  }
  [[nodiscard]] CommunicatorView communicator() const noexcept {
#ifdef POPS_HAS_MPI
    return CommunicatorView{communicator_};
#else
    return CommunicatorView{};
#endif
  }
  [[nodiscard]] int rank() const { return communicator().rank(); }
  [[nodiscard]] int size() const { return communicator().size(); }

 private:
  friend class ExecutionLane;

  struct StaticIdentity {};

#ifdef POPS_HAS_MPI
  ExecutionCommunicator(StaticIdentity, std::string_view identity, MPI_Comm communicator) noexcept
      : static_identity_(identity), communicator_(communicator) {}
  ExecutionCommunicator(std::string identity, MPI_Comm communicator) noexcept
      : identity_(std::move(identity)), communicator_(communicator) {}
#else
  ExecutionCommunicator(StaticIdentity, std::string_view identity) noexcept
      : static_identity_(identity) {}
  explicit ExecutionCommunicator(std::string identity) noexcept : identity_(std::move(identity)) {}
#endif

  std::string identity_;
  std::string_view static_identity_;
#ifdef POPS_HAS_MPI
  MPI_Comm communicator_ = MPI_COMM_NULL;
#endif
};

class ExecutionLane {
 public:
  /// Non-copyable stable-address pin for a borrower that retains a lane pointer.  A lane with an
  /// active pin cannot be moved or destroyed: moving would invalidate the borrowed object address
  /// even though its communicator remains valid.
  class ImmutableBorrow {
   public:
    ImmutableBorrow() = delete;
    ImmutableBorrow(const ImmutableBorrow&) = delete;
    ImmutableBorrow& operator=(const ImmutableBorrow&) = delete;

    ImmutableBorrow(ImmutableBorrow&& other) noexcept
        : lane_(std::exchange(other.lane_, nullptr)) {}
    ImmutableBorrow& operator=(ImmutableBorrow&&) = delete;

    ~ImmutableBorrow() {
      if (lane_ != nullptr)
        lane_->release_immutable_borrow_();
    }

   private:
    friend class ExecutionLane;

    explicit ImmutableBorrow(const ExecutionLane& lane) noexcept : lane_(&lane) {
      lane_->acquire_immutable_borrow_();
    }

    const ExecutionLane* lane_ = nullptr;
  };

  /// Collective-lifetime object. Owning lanes must be materialized and destroyed in the same
  /// canonical order on every parent rank. PoPS runtime owners keep them in deterministic object
  /// graphs and convert every post-duplication construction failure into a uniform collective
  /// failure before unwinding; embedding code must provide the same lifetime discipline because
  /// MPI_Comm_free is collective even on MPI implementations that optimize it locally.
  /// Operation-local tags. Their actual namespace is the lane's duplicated MPI communicator, so
  /// the same values are safe in concurrent lanes without a process-global tag allocator.
  static constexpr int halo_message_tag = 0;
  static constexpr int parallel_copy_message_tag = 1;
  static constexpr int translation_message_tag = 2;

  /// Non-owning sequential view of MPI_COMM_WORLD for preparation/control paths. This explicitly
  /// initializes or validates MPI, but its destructor never frees the process communicator.
  [[nodiscard]] static ExecutionLane world() {
#ifdef POPS_HAS_MPI
    detail::ensure_mpi_world_initialized();
    return ExecutionLane(StaticIdentity{}, "MPI_COMM_WORLD", MPI_COMM_WORLD, false);
#else
    return ExecutionLane(StaticIdentity{}, "MPI_COMM_WORLD");
#endif
  }

  /// Named non-owning world view. Identity parts remain borrowed until every rank has entered the
  /// common gate; the owned diagnostic string is materialized only inside that gated operation.
  [[nodiscard]] static ExecutionLane world(std::string_view identity) {
    const std::array<std::string_view, 1> parts{identity};
    return world(std::span<const std::string_view>(parts));
  }

  [[nodiscard]] static ExecutionLane world(std::string_view prefix, std::string_view suffix) {
    const std::array<std::string_view, 2> parts{prefix, suffix};
    return world(std::span<const std::string_view>(parts));
  }

  [[nodiscard]] static ExecutionLane world(std::span<const std::string_view> identity_parts) {
    std::size_t identity_size = 0;
    long invalid_identity_local = identity_parts.empty() ? 1L : 0L;
    for (const std::string_view part : identity_parts) {
      if (part.size() > std::numeric_limits<std::size_t>::max() - identity_size) {
        invalid_identity_local = 1;
        break;
      }
      identity_size += part.size();
    }
    if (identity_size == 0)
      invalid_identity_local = 1;
#ifdef POPS_HAS_MPI
    detail::ensure_mpi_world_initialized();
    const CommunicatorView world{MPI_COMM_WORLD};
    if (all_reduce_max(invalid_identity_local, world) != 0)
      throw std::invalid_argument("named world execution lane requires a non-empty identity");
    std::string owned_identity;
    long identity_allocation_failure_local = 0;
    try {
      owned_identity.reserve(identity_size);
      for (const std::string_view part : identity_parts)
        owned_identity.append(part);
    } catch (...) {
      identity_allocation_failure_local = 1;
    }
    if (all_reduce_max(identity_allocation_failure_local, world) != 0)
      throw std::bad_alloc();
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("world-execution-lane"), std::string_view(owned_identity)}}, world))
      throw std::invalid_argument("world execution lane identity differs between MPI ranks");
    return ExecutionLane(std::move(owned_identity), MPI_COMM_WORLD, false);
#else
    if (invalid_identity_local != 0)
      throw std::invalid_argument("named world execution lane requires a non-empty identity");
    std::string owned_identity;
    owned_identity.reserve(identity_size);
    for (const std::string_view part : identity_parts)
      owned_identity.append(part);
    return ExecutionLane(std::move(owned_identity));
#endif
  }

  /// Collectively duplicate MPI_COMM_WORLD after validating the exact lane identity. Every rank
  /// must create lanes in the same canonical order. This is a materialization operation, never a
  /// solve-time/lazy operation. In a serial build it returns an identity-only serial lane.
  [[nodiscard]] static ExecutionLane duplicate_world_collectively(std::string_view identity) {
    return duplicate_collectively(ExecutionCommunicator::world(), identity);
  }

  /// Collectively duplicate an authenticated embedding-owned communicator. Every validation and
  /// allocation consensus remains on that exact supplied communicator; subgroup lanes never touch
  /// MPI_COMM_WORLD. The borrowed parent is not freed and must outlive this preparation call.
  [[nodiscard]] static ExecutionLane duplicate_collectively(const ExecutionCommunicator& parent,
                                                            std::string_view identity) {
    const long invalid_identity = all_reduce_max(
        identity.empty() || parent.identity().empty() ? 1L : 0L, parent.communicator());
    if (invalid_identity != 0)
      throw std::invalid_argument("execution lane requires exact parent and lane identities");
    const long invalid_qualified_size = all_reduce_max(
        parent.identity().size() >= std::numeric_limits<std::size_t>::max() - identity.size() ? 1L
                                                                                              : 0L,
        parent.communicator());
    if (invalid_qualified_size != 0)
      throw std::length_error("qualified execution lane identity exceeds size_t capacity");
    std::string qualified_identity;
    long identity_allocation_failure_local = 0;
    try {
      qualified_identity.reserve(parent.identity().size() + 1u + identity.size());
      qualified_identity.append(parent.identity());
      qualified_identity.push_back('/');
      qualified_identity.append(identity);
    } catch (...) {
      identity_allocation_failure_local = 1;
    }
    if (all_reduce_max(identity_allocation_failure_local, parent.communicator()) != 0)
      throw std::bad_alloc();
#ifdef POPS_HAS_MPI
    detail::ensure_mpi_world_initialized();
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("execution-parent"), parent.identity()},
             {std::string_view("execution-lane"), std::string_view(identity)}},
            parent.communicator()))
      throw std::invalid_argument("execution lane identity differs between MPI ranks");

    MPI_Comm communicator = MPI_COMM_NULL;
    const int duplicate_code = MPI_Comm_dup(parent.native_handle(), &communicator);
    const long duplicate_failure =
        all_reduce_max(duplicate_code == MPI_SUCCESS ? 0L : 1L, parent.communicator());
    if (duplicate_failure != 0) {
      const long has_handle = communicator == MPI_COMM_NULL ? 0L : 1L;
      const long minimum_handles = all_reduce_min(has_handle, parent.communicator());
      const long maximum_handles = all_reduce_max(has_handle, parent.communicator());
      // MPI_Comm_free is collective on the duplicated communicator. A conforming collective
      // duplication therefore permits either every rank to free its result or no rank to free one.
      // A partial result cannot be recovered portably: ranks without the opaque context cannot join
      // its free. Abort the exact supplied rank space instead of deadlocking or leaking that broken
      // MPI state.
      if (minimum_handles != maximum_handles) {
        (void)MPI_Abort(parent.native_handle(), duplicate_code == MPI_SUCCESS ? 1 : duplicate_code);
        std::terminate();
      }
      const int cleanup_code = minimum_handles == 0 ? MPI_SUCCESS : MPI_Comm_free(&communicator);
      const long cleanup_failure =
          all_reduce_max(cleanup_code == MPI_SUCCESS ? 0L : 1L, parent.communicator());
      if (cleanup_failure != 0)
        throw std::runtime_error(
            "MPI_Comm_dup(execution lane) failed and a duplicated local handle could not be "
            "released collectively");
      throw std::runtime_error("MPI_Comm_dup(execution lane) failed on at least one parent rank");
    }
    const int errhandler_code = MPI_Comm_set_errhandler(communicator, MPI_ERRORS_RETURN);
    const long errhandler_failure =
        all_reduce_max(errhandler_code == MPI_SUCCESS ? 0L : 1L, parent.communicator());
    if (errhandler_failure != 0) {
      const int cleanup_code = MPI_Comm_free(&communicator);
      const long cleanup_failure =
          all_reduce_max(cleanup_code == MPI_SUCCESS ? 0L : 1L, parent.communicator());
      if (cleanup_failure != 0)
        throw std::runtime_error(
            "MPI_Comm_set_errhandler(execution lane) failed and the duplicated communicator "
            "could not be released collectively");
      throw std::runtime_error(
          "MPI_Comm_set_errhandler(execution lane) failed on at least one parent rank");
    }
    return ExecutionLane(std::move(qualified_identity), communicator, true);
#else
    return ExecutionLane(std::move(qualified_identity));
#endif
  }

  /// Collectively duplicate an already-authenticated owning lane. This is the child-lane seam for
  /// runtime materialization: it reuses the exact parent communicator and identity without
  /// borrowed-authority reauthentication or comparison with MPI_COMM_WORLD.
  [[nodiscard]] static ExecutionLane duplicate_collectively(const ExecutionLane& parent,
                                                            std::string_view identity) {
    if (!parent.active())
      throw std::invalid_argument("execution child lane requires an active parent lane");
    const long invalid_parent = all_reduce_max(parent.identity().empty()
#ifdef POPS_HAS_MPI
                                                       || !parent.owns_communicator()
#endif
                                                   ? 1L
                                                   : 0L,
                                               parent.communicator());
    if (invalid_parent != 0)
      throw std::invalid_argument(
          "execution child lane requires an authenticated owning parent lane");
#ifdef POPS_HAS_MPI
    const ExecutionCommunicator authenticated_parent(ExecutionCommunicator::StaticIdentity{},
                                                     parent.identity(), parent.native_handle());
#else
    const ExecutionCommunicator authenticated_parent(ExecutionCommunicator::StaticIdentity{},
                                                     parent.identity());
#endif
    return duplicate_collectively(authenticated_parent, identity);
  }

  ExecutionLane(const ExecutionLane&) = delete;
  ExecutionLane& operator=(const ExecutionLane&) = delete;

  ExecutionLane(ExecutionLane&& other) noexcept { move_from_(std::move(other)); }
  /// Replacement preserves the historical movable-lane API only while neither object is borrowed.
  /// A borrowed lane has a stable address contract, so replacing either endpoint fails closed.
  ExecutionLane& operator=(ExecutionLane&& other) noexcept {
    if (this == &other)
      return *this;
    if (immutable_borrow_count_.load(std::memory_order_acquire) != 0 ||
        other.immutable_borrow_count_.load(std::memory_order_acquire) != 0)
      std::terminate();
    release_();
    move_from_(std::move(other));
    return *this;
  }

  ~ExecutionLane() { release_(); }

  [[nodiscard]] std::string_view identity() const noexcept {
    return static_identity_.empty() ? std::string_view(identity_) : static_identity_;
  }
  [[nodiscard]] CommunicatorView communicator() const noexcept {
#ifdef POPS_HAS_MPI
    return CommunicatorView{communicator_};
#else
    return CommunicatorView{};
#endif
  }
  [[nodiscard]] bool active() const noexcept {
#ifdef POPS_HAS_MPI
    return communicator().active();
#else
    // Serial lanes have no native communicator handle. Their identity is the live-state token,
    // which is cleared explicitly when ownership is moved away.
    return !identity().empty();
#endif
  }
  /// True only for a collectively duplicated MPI communicator. World and serial lanes borrow none.
  [[nodiscard]] bool owns_communicator() const noexcept {
#ifdef POPS_HAS_MPI
    return owns_communicator_;
#else
    return false;
#endif
  }
  /// Pins this exact lane object against move/destruction until the returned guard dies.
  [[nodiscard]] ImmutableBorrow borrow_immutably() const noexcept { return ImmutableBorrow(*this); }
  [[nodiscard]] int rank() const { return communicator().rank(); }
  [[nodiscard]] int size() const { return communicator().size(); }

  /// True only when both lanes contain the same ranks in the same order. This is the local,
  /// allocation-free compatibility predicate used before binding independently duplicated
  /// problem/workspace lanes. The following collective preflight makes a local MPI failure uniform.
  [[nodiscard]] bool congruent_with(const ExecutionLane& other) const {
#ifdef POPS_HAS_MPI
    int relation = MPI_UNEQUAL;
    detail::require_mpi_success(MPI_Comm_compare(communicator_, other.communicator_, &relation),
                                "MPI_Comm_compare(execution lanes)");
    return relation == MPI_IDENT || relation == MPI_CONGRUENT;
#else
    (void)other;
    return true;
#endif
  }

#ifdef POPS_HAS_MPI
  /// Native-only seam for MPI point-to-point operations. Python never owns or transports this value.
  [[nodiscard]] MPI_Comm native_handle() const noexcept { return communicator_; }
#endif

 private:
  friend class ObserverMpiLane;

  struct StaticIdentity {};

#ifdef POPS_HAS_MPI
  ExecutionLane(StaticIdentity, std::string_view identity, MPI_Comm communicator,
                bool owns_communicator) noexcept
      : static_identity_(identity),
        communicator_(communicator),
        owns_communicator_(owns_communicator) {}
  ExecutionLane(std::string identity, MPI_Comm communicator, bool owns_communicator) noexcept
      : identity_(std::move(identity)),
        communicator_(communicator),
        owns_communicator_(owns_communicator) {}
#else
  ExecutionLane(StaticIdentity, std::string_view identity) noexcept : static_identity_(identity) {}
  explicit ExecutionLane(std::string identity) noexcept : identity_(std::move(identity)) {}
#endif

  void move_from_(ExecutionLane&& other) noexcept {
    if (other.immutable_borrow_count_.load(std::memory_order_acquire) != 0)
      std::terminate();
    identity_ = std::move(other.identity_);
    other.identity_.clear();
    static_identity_ = std::exchange(other.static_identity_, std::string_view{});
#ifdef POPS_HAS_MPI
    communicator_ = std::exchange(other.communicator_, MPI_COMM_NULL);
    owns_communicator_ = std::exchange(other.owns_communicator_, false);
#endif
  }

  void release_() noexcept {
    if (immutable_borrow_count_.load(std::memory_order_acquire) != 0)
      std::terminate();
#ifdef POPS_HAS_MPI
    if (communicator_ != MPI_COMM_NULL && owns_communicator_) {
      if (detail::comm_active_unlocked())
        (void)MPI_Comm_free(&communicator_);
    }
    communicator_ = MPI_COMM_NULL;
    owns_communicator_ = false;
#endif
  }

  void acquire_immutable_borrow_() const noexcept {
    std::size_t current = immutable_borrow_count_.load(std::memory_order_relaxed);
    for (;;) {
      if (current == std::numeric_limits<std::size_t>::max())
        std::terminate();
      if (immutable_borrow_count_.compare_exchange_weak(
              current, current + 1, std::memory_order_acq_rel, std::memory_order_relaxed))
        return;
    }
  }

  void release_immutable_borrow_() const noexcept {
    const std::size_t previous = immutable_borrow_count_.fetch_sub(1, std::memory_order_acq_rel);
    if (previous == 0)
      std::terminate();
  }

  std::string identity_;
  std::string_view static_identity_;
  mutable std::atomic<std::size_t> immutable_borrow_count_{0};
#ifdef POPS_HAS_MPI
  MPI_Comm communicator_ = MPI_COMM_NULL;
  bool owns_communicator_ = false;
#endif
};

/// Explicit-lifetime communicator dedicated to a post-commit observer worker.
///
/// The communicator is duplicated collectively before the worker starts and must be closed
/// collectively after that worker has joined.  Its destructor deliberately performs no MPI call:
/// Python garbage collection is neither collectively ordered nor guaranteed to run before MPI
/// finalization.  Forgetting close_collectively() therefore keeps the duplicate alive until MPI
/// finalization instead of risking a process deadlock from MPI_Comm_free in a GC destructor.
class ObserverMpiLane {
 public:
  /// Reuse the canonical ExecutionLane materialization gate, then transfer its owned communicator
  /// into an object whose release is explicit rather than destructor-driven.
  [[nodiscard]] static ObserverMpiLane duplicate_world_collectively(std::string_view identity) {
    return ObserverMpiLane(ExecutionLane::duplicate_world_collectively(identity));
  }

  ObserverMpiLane(const ObserverMpiLane&) = delete;
  ObserverMpiLane& operator=(const ObserverMpiLane&) = delete;
  ObserverMpiLane& operator=(ObserverMpiLane&&) = delete;

  ObserverMpiLane(ObserverMpiLane&& other) noexcept
      : identity_(std::move(other.identity_)),
        static_identity_(std::exchange(other.static_identity_, std::string_view{})),
#ifdef POPS_HAS_MPI
        communicator_(std::exchange(other.communicator_, MPI_COMM_NULL)),
#endif
        closed_(std::exchange(other.closed_, true)) {
  }

  /// Intentionally non-owning at destruction time; see the class contract above.
  ~ObserverMpiLane() = default;

  [[nodiscard]] std::string_view identity() const noexcept {
    return static_identity_.empty() ? std::string_view(identity_) : static_identity_;
  }

  [[nodiscard]] bool active() const noexcept {
    if (closed_)
      return false;
#ifdef POPS_HAS_MPI
    return communicator_ != MPI_COMM_NULL && detail::comm_active_unlocked();
#else
    return true;
#endif
  }

  [[nodiscard]] bool closed() const noexcept { return closed_; }

  [[nodiscard]] CommunicatorView communicator() const noexcept {
#ifdef POPS_HAS_MPI
    return CommunicatorView{closed_ ? MPI_COMM_NULL : communicator_};
#else
    return CommunicatorView{};
#endif
  }

  [[nodiscard]] int rank() const {
    require_open_();
    return communicator().rank();
  }

  [[nodiscard]] int size() const {
    require_open_();
    return communicator().size();
  }

  [[nodiscard]] std::int64_t fortran_handle() const {
    require_open_();
#ifdef POPS_HAS_MPI
    return static_cast<std::int64_t>(MPI_Comm_c2f(communicator_));
#else
    throw std::runtime_error("observer MPI lane is unavailable in a serial PoPS build");
#endif
  }

  void barrier() const {
    require_open_();
    pops::barrier(communicator());
  }

  /// Broadcast arbitrary bytes over the duplicated communicator.  Large payloads are split into
  /// MPI-int-sized chunks and allocation failures are made uniform before payload traffic begins.
  [[nodiscard]] std::string broadcast_bytes(std::string payload, int root = 0) const {
    require_open_();
#ifdef POPS_HAS_MPI
    const CommunicatorView lane = communicator();
    int minimum_root = root;
    int maximum_root = root;
    detail::require_mpi_success(
        MPI_Allreduce(&root, &minimum_root, 1, MPI_INT, MPI_MIN, lane.native_handle()),
        "MPI_Allreduce(observer root minimum)");
    detail::require_mpi_success(
        MPI_Allreduce(&root, &maximum_root, 1, MPI_INT, MPI_MAX, lane.native_handle()),
        "MPI_Allreduce(observer root maximum)");
    if (minimum_root != maximum_root)
      throw std::invalid_argument("observer collective root differs across MPI ranks");
    const int ranks = lane.size();
    if (root < 0 || root >= ranks)
      throw std::out_of_range("observer collective root is outside the lane");
    const int me = lane.rank();

    long length_overflow = 0;
    if constexpr (sizeof(std::size_t) > sizeof(unsigned long long)) {
      if (me == root &&
          payload.size() > static_cast<std::size_t>(std::numeric_limits<unsigned long long>::max()))
        length_overflow = 1;
    }
    if (all_reduce_max(length_overflow, lane) != 0)
      throw std::overflow_error("observer broadcast payload exceeds the MPI length domain");

    unsigned long long length = me == root ? static_cast<unsigned long long>(payload.size()) : 0ULL;
    detail::require_mpi_success(
        MPI_Bcast(&length, 1, MPI_UNSIGNED_LONG_LONG, root, lane.native_handle()),
        "MPI_Bcast(observer payload length)");
    if (length > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max()))
      throw std::overflow_error("observer broadcast payload exceeds the local size_t domain");

    long allocation_failed = 0;
    if (me != root) {
      try {
        payload.resize(static_cast<std::size_t>(length));
      } catch (const std::bad_alloc&) {
        allocation_failed = 1;
      } catch (const std::length_error&) {
        allocation_failed = 1;
      }
    }
    if (all_reduce_max(allocation_failed, lane) != 0)
      throw std::runtime_error("an observer rank could not allocate a broadcast payload");

    unsigned long long offset = 0;
    while (offset < length) {
      const int count = static_cast<int>(std::min<unsigned long long>(
          length - offset, static_cast<unsigned long long>(std::numeric_limits<int>::max())));
      detail::require_mpi_success(MPI_Bcast(payload.data() + static_cast<std::size_t>(offset),
                                            count, MPI_BYTE, root, lane.native_handle()),
                                  "MPI_Bcast(observer payload chunk)");
      offset += static_cast<unsigned long long>(count);
    }
    return payload;
#else
    if (root != 0)
      throw std::out_of_range("serial observer broadcast root must be zero");
    return payload;
#endif
  }

  /// Return one rank-ordered payload per lane participant.  Observer traffic is control-plane
  /// traffic, so a canonical sequence of lane-local broadcasts favors simple large-count safety
  /// over a second bespoke Allgatherv chunk protocol.
  [[nodiscard]] std::vector<std::string> allgather_bytes(const std::string& payload) const {
    require_open_();
#ifdef POPS_HAS_MPI
    const CommunicatorView lane = communicator();
    const int ranks = lane.size();
    const int me = lane.rank();
    std::vector<std::string> result;
    long allocation_failed = 0;
    try {
      result.resize(static_cast<std::size_t>(ranks));
    } catch (const std::bad_alloc&) {
      allocation_failed = 1;
    } catch (const std::length_error&) {
      allocation_failed = 1;
    }
    if (all_reduce_max(allocation_failed, lane) != 0)
      throw std::runtime_error("an observer rank could not allocate allgather results");

    for (int source = 0; source < ranks; ++source) {
      std::string source_payload;
      long copy_failed = 0;
      if (me == source) {
        try {
          source_payload = payload;
        } catch (const std::bad_alloc&) {
          copy_failed = 1;
        } catch (const std::length_error&) {
          copy_failed = 1;
        }
      }
      if (all_reduce_max(copy_failed, lane) != 0)
        throw std::runtime_error("an observer rank could not stage its allgather payload");
      result[static_cast<std::size_t>(source)] = broadcast_bytes(std::move(source_payload), source);
    }
    return result;
#else
    return {payload};
#endif
  }

  /// Gather one rank-ordered payload vector on root; non-root ranks return std::nullopt.
  [[nodiscard]] std::optional<std::vector<std::string>> gather_bytes(const std::string& payload,
                                                                     int root = 0) const {
    require_open_();
#ifdef POPS_HAS_MPI
    const CommunicatorView lane = communicator();
    int minimum_root = root;
    int maximum_root = root;
    detail::require_mpi_success(
        MPI_Allreduce(&root, &minimum_root, 1, MPI_INT, MPI_MIN, lane.native_handle()),
        "MPI_Allreduce(observer gather root minimum)");
    detail::require_mpi_success(
        MPI_Allreduce(&root, &maximum_root, 1, MPI_INT, MPI_MAX, lane.native_handle()),
        "MPI_Allreduce(observer gather root maximum)");
    if (minimum_root != maximum_root)
      throw std::invalid_argument("observer collective root differs across MPI ranks");
    const int ranks = lane.size();
    if (root < 0 || root >= ranks)
      throw std::out_of_range("observer collective root is outside the lane");
    const int me = lane.rank();

    long length_overflow = 0;
    if constexpr (sizeof(std::size_t) > sizeof(unsigned long long)) {
      if (payload.size() > static_cast<std::size_t>(std::numeric_limits<unsigned long long>::max()))
        length_overflow = 1;
    }
    if (all_reduce_max(length_overflow, lane) != 0)
      throw std::overflow_error("consumer gather payload exceeds the MPI length domain");
    const unsigned long long local_length = static_cast<unsigned long long>(payload.size());

    std::vector<unsigned long long> lengths;
    long allocation_failed = 0;
    if (me == root) {
      try {
        lengths.resize(static_cast<std::size_t>(ranks), 0ULL);
      } catch (const std::bad_alloc&) {
        allocation_failed = 1;
      } catch (const std::length_error&) {
        allocation_failed = 1;
      }
    }
    if (all_reduce_max(allocation_failed, lane) != 0)
      throw std::runtime_error("consumer root could not allocate gathered lengths");
    detail::require_mpi_success(
        MPI_Gather(&local_length, 1, MPI_UNSIGNED_LONG_LONG, me == root ? lengths.data() : nullptr,
                   1, MPI_UNSIGNED_LONG_LONG, root, lane.native_handle()),
        "MPI_Gather(consumer payload lengths)");

    unsigned long long maximum_length = local_length;
    detail::require_mpi_success(
        MPI_Allreduce(MPI_IN_PLACE, &maximum_length, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                      lane.native_handle()),
        "MPI_Allreduce(maximum consumer gather length)");

    std::optional<std::vector<std::string>> result;
    std::vector<int> counts;
    std::vector<int> displacements;
    allocation_failed = 0;
    if (me == root) {
      try {
        result.emplace(static_cast<std::size_t>(ranks));
        counts.resize(static_cast<std::size_t>(ranks), 0);
        displacements.resize(static_cast<std::size_t>(ranks), 0);
        for (int rank = 0; rank < ranks; ++rank) {
          const unsigned long long length = lengths[static_cast<std::size_t>(rank)];
          if (length > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
            allocation_failed = 1;
            break;
          }
          (*result)[static_cast<std::size_t>(rank)].resize(static_cast<std::size_t>(length));
        }
      } catch (const std::bad_alloc&) {
        allocation_failed = 1;
      } catch (const std::length_error&) {
        allocation_failed = 1;
      }
    }
    if (all_reduce_max(allocation_failed, lane) != 0)
      throw std::runtime_error("consumer root could not allocate gathered payloads");

    const int capacity = detail::chunk_capacity(ranks);
    for (unsigned long long offset = 0; offset < maximum_length;
         offset += static_cast<unsigned long long>(capacity)) {
      int total = 0;
      if (me == root) {
        for (int rank = 0; rank < ranks; ++rank) {
          const unsigned long long length = lengths[static_cast<std::size_t>(rank)];
          const int count = offset < length
                                ? static_cast<int>(std::min<unsigned long long>(
                                      length - offset, static_cast<unsigned long long>(capacity)))
                                : 0;
          counts[static_cast<std::size_t>(rank)] = count;
          displacements[static_cast<std::size_t>(rank)] = total;
          total += count;
        }
      }
      std::vector<char> round;
      long round_allocation_failed = 0;
      if (me == root) {
        try {
          round.resize(static_cast<std::size_t>(total));
        } catch (const std::bad_alloc&) {
          round_allocation_failed = 1;
        } catch (const std::length_error&) {
          round_allocation_failed = 1;
        }
      }
      if (all_reduce_max(round_allocation_failed, lane) != 0)
        throw std::runtime_error("consumer root could not allocate a gathered chunk");
      const int send_count =
          offset < local_length
              ? static_cast<int>(std::min<unsigned long long>(
                    local_length - offset, static_cast<unsigned long long>(capacity)))
              : 0;
      detail::require_mpi_success(
          MPI_Gatherv(detail::chunk_pointer(payload, offset, send_count), send_count, MPI_BYTE,
                      me == root ? round.data() : nullptr, me == root ? counts.data() : nullptr,
                      me == root ? displacements.data() : nullptr, MPI_BYTE, root,
                      lane.native_handle()),
          "MPI_Gatherv(consumer payload chunk)");
      if (me != root)
        continue;
      for (int rank = 0; rank < ranks; ++rank) {
        const int count = counts[static_cast<std::size_t>(rank)];
        if (count == 0)
          continue;
        std::copy_n(
            round.data() + displacements[static_cast<std::size_t>(rank)], count,
            (*result)[static_cast<std::size_t>(rank)].data() + static_cast<std::size_t>(offset));
      }
    }
    return result;
#else
    if (root != 0)
      throw std::out_of_range("serial observer gather root must be zero");
    return std::vector<std::string>{payload};
#endif
  }

  /// Collectively release the duplicate.  The owner must call this only after its observer worker
  /// has joined.  Repeated calls after a successful close are local no-ops.
  void close_collectively() {
    if (closed_)
      return;
#ifdef POPS_HAS_MPI
    if (communicator_ != MPI_COMM_NULL && detail::comm_active_unlocked()) {
      detail::require_mpi_success(MPI_Comm_free(&communicator_), "MPI_Comm_free(observer lane)");
    } else {
      // MPI_Finalize has already reclaimed MPI resources; never invoke MPI from this state.
      communicator_ = MPI_COMM_NULL;
    }
#endif
    closed_ = true;
  }

 private:
  explicit ObserverMpiLane(ExecutionLane&& lane) noexcept
      : identity_(std::move(lane.identity_)),
        static_identity_(std::exchange(lane.static_identity_, std::string_view{})),
#ifdef POPS_HAS_MPI
        communicator_(std::exchange(lane.communicator_, MPI_COMM_NULL)),
#endif
        closed_(false) {
#ifdef POPS_HAS_MPI
    lane.owns_communicator_ = false;
#endif
  }

  void require_open_() const {
    if (closed_)
      throw std::logic_error("observer MPI lane is closed");
#ifdef POPS_HAS_MPI
    if (communicator_ == MPI_COMM_NULL || !detail::comm_active_unlocked())
      throw std::runtime_error("observer MPI lane is not active");
#endif
  }

  std::string identity_;
  std::string_view static_identity_;
#ifdef POPS_HAS_MPI
  MPI_Comm communicator_ = MPI_COMM_NULL;
#endif
  bool closed_ = true;
};

// Lane-shaped collective facade. These overloads keep the execution communicator explicit at each
// hot call site while sharing the exact implementations in comm.hpp.
inline void barrier(const ExecutionLane& lane) {
  barrier(lane.communicator());
}
inline double all_reduce_sum(double value, const ExecutionLane& lane) {
  return all_reduce_sum(value, lane.communicator());
}
inline double all_reduce_max(double value, const ExecutionLane& lane) {
  return all_reduce_max(value, lane.communicator());
}
inline double all_reduce_min(double value, const ExecutionLane& lane) {
  return all_reduce_min(value, lane.communicator());
}
inline long all_reduce_sum(long value, const ExecutionLane& lane) {
  return all_reduce_sum(value, lane.communicator());
}
inline long all_reduce_max(long value, const ExecutionLane& lane) {
  return all_reduce_max(value, lane.communicator());
}
inline long all_reduce_min(long value, const ExecutionLane& lane) {
  return all_reduce_min(value, lane.communicator());
}
inline void all_reduce_sum_inplace(double* buffer, int count, const ExecutionLane& lane) {
  all_reduce_sum_inplace(buffer, count, lane.communicator());
}
inline void all_reduce_sum_inplace(double* buffer, std::size_t count, const ExecutionLane& lane) {
  all_reduce_sum_inplace(buffer, count, lane.communicator());
}
inline void all_reduce_max_inplace(double* buffer, int count, const ExecutionLane& lane) {
  all_reduce_max_inplace(buffer, count, lane.communicator());
}
inline void all_reduce_max_inplace(double* buffer, std::size_t count, const ExecutionLane& lane) {
  all_reduce_max_inplace(buffer, count, lane.communicator());
}
inline void all_reduce_or_inplace(char* buffer, std::size_t count, const ExecutionLane& lane) {
  all_reduce_or_inplace(buffer, count, lane.communicator());
}
inline void all_reduce_min_inplace(char* buffer, std::size_t count, const ExecutionLane& lane) {
  all_reduce_min_inplace(buffer, count, lane.communicator());
}
inline void all_reduce_max_inplace(char* buffer, std::size_t count, const ExecutionLane& lane) {
  all_reduce_max_inplace(buffer, count, lane.communicator());
}
inline void all_reduce_min_inplace(long* buffer, std::size_t count, const ExecutionLane& lane) {
  all_reduce_min_inplace(buffer, count, lane.communicator());
}
inline void all_reduce_max_inplace(long* buffer, std::size_t count, const ExecutionLane& lane) {
  all_reduce_max_inplace(buffer, count, lane.communicator());
}
inline void broadcast_bytes_inplace(char* buffer, std::size_t count, const ExecutionLane& lane,
                                    int root = 0) {
  broadcast_bytes_inplace(buffer, count, root, lane.communicator());
}
inline bool all_ranks_agree_exact_ordered_byte_pairs(std::span<const ExactOrderedBytePair> values,
                                                     const ExecutionLane& lane) {
  return all_ranks_agree_exact_ordered_byte_pairs(values, lane.communicator());
}
inline bool all_ranks_agree_exact_ordered_byte_pairs(
    std::initializer_list<ExactOrderedBytePair> values, const ExecutionLane& lane) {
  return all_ranks_agree_exact_ordered_byte_pairs(values, lane.communicator());
}
inline bool all_ranks_agree_exact_ordered_byte_pairs(
    const std::vector<ExactOrderedBytePair>& values, const ExecutionLane& lane) {
  return all_ranks_agree_exact_ordered_byte_pairs(values, lane.communicator());
}

}  // namespace pops
