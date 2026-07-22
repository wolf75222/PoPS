#pragma once

/// @file
/// @brief Process-world MPI authority and byte transport owned by the native runtime.
///
/// This is the only object exposed to Python for distributed control-plane traffic.  It is a
/// non-constructible process singleton backed by exact ``MPI_COMM_WORLD``.  Creating it is an
/// explicit operation: on an MPI build it initializes MPI with ``MPI_Init_thread`` when no external
/// owner has done so, and registers an idempotent process-exit finalizer only for that PoPS-owned
/// initialization.  Merely reading runtime metadata remains effect-free.
///
/// The transport accepts bytes only.  Python may encode small structured control messages, but MPI
/// communicator/datatype objects and all collective operations remain native C++ values.

#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pops {

namespace detail {

#ifdef POPS_HAS_MPI

inline void require_active_mpi_world_unlocked() {
  int initialized = 0;
  int finalized = 0;
  require_mpi_success(MPI_Initialized(&initialized), "MPI_Initialized");
  require_mpi_success(MPI_Finalized(&finalized), "MPI_Finalized");
  if (!initialized || finalized)
    throw std::runtime_error("MPI_COMM_WORLD is not active");
}

inline void require_active_mpi_world() {
  require_active_mpi_world_unlocked();
}

inline int validated_collective_root(int root) {
  require_active_mpi_world_unlocked();
  int minimum = root;
  int maximum = root;
  require_mpi_success(MPI_Allreduce(&root, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD),
                      "MPI_Allreduce(root minimum)");
  require_mpi_success(MPI_Allreduce(&root, &maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
                      "MPI_Allreduce(root maximum)");
  if (minimum != maximum)
    throw std::invalid_argument("collective root differs across MPI ranks");
  int size = 1;
  require_mpi_success(MPI_Comm_size(MPI_COMM_WORLD, &size), "MPI_Comm_size");
  if (root < 0 || root >= size)
    throw std::out_of_range("collective root is outside MPI_COMM_WORLD");
  return root;
}

inline int chunk_capacity(int ranks) {
  const int divisor = std::max(1, ranks);
  return std::max(1, std::numeric_limits<int>::max() / divisor);
}

inline const char* chunk_pointer(const std::string& payload, unsigned long long offset, int count) {
  if (count == 0)
    return nullptr;
  return payload.data() + static_cast<std::size_t>(offset);
}

#endif

}  // namespace detail

/// Opaque native identity of MPI_DOUBLE.  Instances are obtainable only from WorldCommunicator.
class NativeMpiDatatype {
 public:
  NativeMpiDatatype(const NativeMpiDatatype&) = delete;
  NativeMpiDatatype& operator=(const NativeMpiDatatype&) = delete;

  [[nodiscard]] std::string_view identity() const noexcept {
#ifdef POPS_HAS_MPI
    return "MPI_DOUBLE";
#else
    return "none";
#endif
  }

  [[nodiscard]] std::int64_t fortran_handle() const {
#ifdef POPS_HAS_MPI
    detail::require_active_mpi_world_unlocked();
    return static_cast<std::int64_t>(MPI_Type_c2f(MPI_DOUBLE));
#else
    throw std::runtime_error("MPI_DOUBLE is unavailable in a serial PoPS build");
#endif
  }

 private:
  friend class WorldCommunicator;
  NativeMpiDatatype() = default;

  static const NativeMpiDatatype& float64_instance() {
    static const NativeMpiDatatype value;
    return value;
  }
};

/// Exact native process-world authority.  There is no public constructor and no custom communicator.
class WorldCommunicator {
 public:
  WorldCommunicator(const WorldCommunicator&) = delete;
  WorldCommunicator& operator=(const WorldCommunicator&) = delete;

  /// Return the unique process world, explicitly initializing MPI on an MPI build when necessary.
  static WorldCommunicator& world() {
#ifdef POPS_HAS_MPI
    detail::ensure_mpi_world_initialized();
#endif
    return instance();
  }

  [[nodiscard]] bool active() const noexcept {
#ifdef POPS_HAS_MPI
    return comm_active();
#else
    return false;
#endif
  }

  [[nodiscard]] std::string_view identity() const noexcept {
#ifdef POPS_HAS_MPI
    return "MPI_COMM_WORLD";
#else
    return "serial";
#endif
  }

  [[nodiscard]] int rank() const {
#ifdef POPS_HAS_MPI
    detail::require_active_mpi_world_unlocked();
    int value = 0;
    detail::require_mpi_success(MPI_Comm_rank(MPI_COMM_WORLD, &value), "MPI_Comm_rank");
    return value;
#else
    return 0;
#endif
  }

  [[nodiscard]] int size() const {
#ifdef POPS_HAS_MPI
    detail::require_active_mpi_world_unlocked();
    int value = 1;
    detail::require_mpi_success(MPI_Comm_size(MPI_COMM_WORLD, &value), "MPI_Comm_size");
    return value;
#else
    return 1;
#endif
  }

  /// Borrow the exact process-world communicator for native subsystems that also accept a
  /// duplicated observer lane. The returned view never owns or frees MPI_COMM_WORLD.
  [[nodiscard]] CommunicatorView communicator() const {
#ifdef POPS_HAS_MPI
    detail::require_active_mpi_world_unlocked();
    return CommunicatorView{MPI_COMM_WORLD};
#else
    return CommunicatorView{};
#endif
  }

  void require_active_mpi_world() const {
#ifdef POPS_HAS_MPI
    detail::require_active_mpi_world();
#else
    throw std::runtime_error("MPI_COMM_WORLD is unavailable in a serial PoPS build");
#endif
  }

  [[nodiscard]] bool initialized_by_pops() const noexcept { return mpi_initialized_by_pops(); }

  [[nodiscard]] bool atexit_finalize_registered() const noexcept {
    return mpi_atexit_finalize_registered();
  }

  [[nodiscard]] int thread_level() const noexcept { return mpi_thread_level(); }

  [[nodiscard]] std::int64_t fortran_handle() const {
#ifdef POPS_HAS_MPI
    detail::require_active_mpi_world_unlocked();
    return static_cast<std::int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD));
#else
    throw std::runtime_error("MPI_COMM_WORLD is unavailable in a serial PoPS build");
#endif
  }

  [[nodiscard]] const NativeMpiDatatype& datatype_float64() const {
    require_active_mpi_world();
    return NativeMpiDatatype::float64_instance();
  }

  [[nodiscard]] bool owns_float64_datatype(const NativeMpiDatatype& datatype) const noexcept {
    return &datatype == &NativeMpiDatatype::float64_instance();
  }

  void barrier() const {
#ifdef POPS_HAS_MPI
    detail::require_active_mpi_world_unlocked();
    detail::require_mpi_success(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
#endif
  }

  /// Broadcast arbitrary bytes from an agreed root.  Payloads larger than MPI's int count are
  /// transferred in fixed, identically ordered chunks.
  [[nodiscard]] std::string broadcast_bytes(std::string payload, int root = 0) const {
#ifdef POPS_HAS_MPI
    detail::require_active_mpi_world_unlocked();
    detail::validated_collective_root(root);
    int me = 0;
    detail::require_mpi_success(MPI_Comm_rank(MPI_COMM_WORLD, &me), "MPI_Comm_rank");

    int overflow = 0;
    if (me == root &&
        payload.size() > static_cast<std::size_t>(std::numeric_limits<unsigned long long>::max()))
      overflow = 1;
    int any_overflow = 0;
    detail::require_mpi_success(
        MPI_Allreduce(&overflow, &any_overflow, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(broadcast size overflow)");
    if (any_overflow)
      throw std::overflow_error("broadcast payload exceeds the native MPI length domain");

    unsigned long long length = me == root ? static_cast<unsigned long long>(payload.size()) : 0ULL;
    detail::require_mpi_success(MPI_Bcast(&length, 1, MPI_UNSIGNED_LONG_LONG, root, MPI_COMM_WORLD),
                                "MPI_Bcast(payload length)");
    if (length > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max()))
      throw std::overflow_error("broadcast payload cannot be represented by std::size_t");

    int allocation_failed = 0;
    if (me != root) {
      try {
        payload.resize(static_cast<std::size_t>(length));
      } catch (const std::bad_alloc&) {
        allocation_failed = 1;
      } catch (const std::length_error&) {
        allocation_failed = 1;
      }
    }
    int any_allocation_failed = 0;
    detail::require_mpi_success(MPI_Allreduce(&allocation_failed, &any_allocation_failed, 1,
                                              MPI_INT, MPI_MAX, MPI_COMM_WORLD),
                                "MPI_Allreduce(broadcast allocation)");
    if (any_allocation_failed)
      throw std::runtime_error("a rank could not allocate the broadcast payload");

    unsigned long long offset = 0;
    while (offset < length) {
      const int count = static_cast<int>(std::min<unsigned long long>(
          length - offset, static_cast<unsigned long long>(std::numeric_limits<int>::max())));
      detail::require_mpi_success(MPI_Bcast(payload.data() + static_cast<std::size_t>(offset),
                                            count, MPI_BYTE, root, MPI_COMM_WORLD),
                                  "MPI_Bcast(payload chunk)");
      offset += static_cast<unsigned long long>(count);
    }
    return payload;
#else
    if (root != 0)
      throw std::out_of_range("serial broadcast root must be zero");
    return payload;
#endif
  }

  /// Gather variable-sized bytes on every rank using chunked MPI_Allgatherv.
  [[nodiscard]] std::vector<std::string> allgather_bytes(const std::string& payload) const {
#ifdef POPS_HAS_MPI
    detail::require_active_mpi_world_unlocked();
    int ranks = 1;
    int me = 0;
    detail::require_mpi_success(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size");
    detail::require_mpi_success(MPI_Comm_rank(MPI_COMM_WORLD, &me), "MPI_Comm_rank");
    (void)me;

    int length_overflow = 0;
    if constexpr (sizeof(std::size_t) > sizeof(unsigned long long)) {
      if (payload.size() > static_cast<std::size_t>(std::numeric_limits<unsigned long long>::max()))
        length_overflow = 1;
    }
    int any_length_overflow = 0;
    detail::require_mpi_success(
        MPI_Allreduce(&length_overflow, &any_length_overflow, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(allgather length overflow)");
    if (any_length_overflow)
      throw std::overflow_error("an allgather payload exceeds the native MPI length domain");
    const unsigned long long local_length = static_cast<unsigned long long>(payload.size());
    std::vector<unsigned long long> lengths;
    int allocation_failed = 0;
    try {
      lengths.resize(static_cast<std::size_t>(ranks), 0ULL);
    } catch (const std::bad_alloc&) {
      allocation_failed = 1;
    } catch (const std::length_error&) {
      allocation_failed = 1;
    }
    int any_allocation_failed = 0;
    detail::require_mpi_success(MPI_Allreduce(&allocation_failed, &any_allocation_failed, 1,
                                              MPI_INT, MPI_MAX, MPI_COMM_WORLD),
                                "MPI_Allreduce(allgather length allocation)");
    if (any_allocation_failed)
      throw std::runtime_error("a rank could not allocate allgather lengths");
    detail::require_mpi_success(
        MPI_Allgather(&local_length, 1, MPI_UNSIGNED_LONG_LONG, lengths.data(), 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD),
        "MPI_Allgather(payload lengths)");

    allocation_failed = 0;
    std::vector<std::string> result;
    std::vector<int> counts;
    std::vector<int> displacements;
    try {
      result.resize(static_cast<std::size_t>(ranks));
      counts.resize(static_cast<std::size_t>(ranks), 0);
      displacements.resize(static_cast<std::size_t>(ranks), 0);
      for (int rank = 0; rank < ranks; ++rank) {
        if (lengths[static_cast<std::size_t>(rank)] >
            static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
          allocation_failed = 1;
          break;
        }
        result[static_cast<std::size_t>(rank)].resize(
            static_cast<std::size_t>(lengths[static_cast<std::size_t>(rank)]));
      }
    } catch (const std::bad_alloc&) {
      allocation_failed = 1;
    } catch (const std::length_error&) {
      allocation_failed = 1;
    }
    any_allocation_failed = 0;
    detail::require_mpi_success(MPI_Allreduce(&allocation_failed, &any_allocation_failed, 1,
                                              MPI_INT, MPI_MAX, MPI_COMM_WORLD),
                                "MPI_Allreduce(allgather allocation)");
    if (any_allocation_failed)
      throw std::runtime_error("a rank could not allocate allgather payloads");

    const unsigned long long maximum_length = *std::max_element(lengths.begin(), lengths.end());
    const int capacity = detail::chunk_capacity(ranks);
    for (unsigned long long offset = 0; offset < maximum_length;
         offset += static_cast<unsigned long long>(capacity)) {
      int total = 0;
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
      std::vector<char> round;
      int round_allocation_failed = 0;
      try {
        round.resize(static_cast<std::size_t>(total));
      } catch (const std::bad_alloc&) {
        round_allocation_failed = 1;
      } catch (const std::length_error&) {
        round_allocation_failed = 1;
      }
      int any_round_allocation_failed = 0;
      detail::require_mpi_success(
          MPI_Allreduce(&round_allocation_failed, &any_round_allocation_failed, 1, MPI_INT, MPI_MAX,
                        MPI_COMM_WORLD),
          "MPI_Allreduce(allgather chunk allocation)");
      if (any_round_allocation_failed)
        throw std::runtime_error("a rank could not allocate an allgather chunk");
      const int send_count =
          offset < local_length
              ? static_cast<int>(std::min<unsigned long long>(
                    local_length - offset, static_cast<unsigned long long>(capacity)))
              : 0;
      detail::require_mpi_success(MPI_Allgatherv(detail::chunk_pointer(payload, offset, send_count),
                                                 send_count, MPI_BYTE, round.data(), counts.data(),
                                                 displacements.data(), MPI_BYTE, MPI_COMM_WORLD),
                                  "MPI_Allgatherv(payload chunk)");
      for (int rank = 0; rank < ranks; ++rank) {
        const int count = counts[static_cast<std::size_t>(rank)];
        if (count == 0)
          continue;
        std::copy_n(
            round.data() + displacements[static_cast<std::size_t>(rank)], count,
            result[static_cast<std::size_t>(rank)].data() + static_cast<std::size_t>(offset));
      }
    }
    return result;
#else
    return {payload};
#endif
  }

  /// Gather variable-sized bytes only on root.  Non-root ranks return std::nullopt.
  [[nodiscard]] std::optional<std::vector<std::string>> gather_bytes(const std::string& payload,
                                                                     int root = 0) const {
#ifdef POPS_HAS_MPI
    detail::require_active_mpi_world_unlocked();
    detail::validated_collective_root(root);
    int ranks = 1;
    int me = 0;
    detail::require_mpi_success(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size");
    detail::require_mpi_success(MPI_Comm_rank(MPI_COMM_WORLD, &me), "MPI_Comm_rank");

    int length_overflow = 0;
    if constexpr (sizeof(std::size_t) > sizeof(unsigned long long)) {
      if (payload.size() > static_cast<std::size_t>(std::numeric_limits<unsigned long long>::max()))
        length_overflow = 1;
    }
    int any_length_overflow = 0;
    detail::require_mpi_success(
        MPI_Allreduce(&length_overflow, &any_length_overflow, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(gather length overflow)");
    if (any_length_overflow)
      throw std::overflow_error("a gather payload exceeds the native MPI length domain");
    const unsigned long long local_length = static_cast<unsigned long long>(payload.size());
    std::vector<unsigned long long> lengths;
    int allocation_failed = 0;
    if (me == root) {
      try {
        lengths.resize(static_cast<std::size_t>(ranks), 0ULL);
      } catch (const std::bad_alloc&) {
        allocation_failed = 1;
      } catch (const std::length_error&) {
        allocation_failed = 1;
      }
    }
    int any_allocation_failed = 0;
    detail::require_mpi_success(MPI_Allreduce(&allocation_failed, &any_allocation_failed, 1,
                                              MPI_INT, MPI_MAX, MPI_COMM_WORLD),
                                "MPI_Allreduce(gather length allocation)");
    if (any_allocation_failed)
      throw std::runtime_error("root could not allocate gathered lengths");
    detail::require_mpi_success(
        MPI_Gather(&local_length, 1, MPI_UNSIGNED_LONG_LONG, me == root ? lengths.data() : nullptr,
                   1, MPI_UNSIGNED_LONG_LONG, root, MPI_COMM_WORLD),
        "MPI_Gather(payload lengths)");
    unsigned long long maximum_length = local_length;
    detail::require_mpi_success(MPI_Allreduce(MPI_IN_PLACE, &maximum_length, 1,
                                              MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD),
                                "MPI_Allreduce(maximum gather length)");

    allocation_failed = 0;
    std::optional<std::vector<std::string>> result;
    std::vector<int> counts;
    std::vector<int> displacements;
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
    any_allocation_failed = 0;
    detail::require_mpi_success(MPI_Allreduce(&allocation_failed, &any_allocation_failed, 1,
                                              MPI_INT, MPI_MAX, MPI_COMM_WORLD),
                                "MPI_Allreduce(gather allocation)");
    if (any_allocation_failed)
      throw std::runtime_error("root could not allocate gathered payloads");

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
      int round_allocation_failed = 0;
      if (me == root) {
        try {
          round.resize(static_cast<std::size_t>(total));
        } catch (const std::bad_alloc&) {
          round_allocation_failed = 1;
        } catch (const std::length_error&) {
          round_allocation_failed = 1;
        }
      }
      int any_round_allocation_failed = 0;
      detail::require_mpi_success(
          MPI_Allreduce(&round_allocation_failed, &any_round_allocation_failed, 1, MPI_INT, MPI_MAX,
                        MPI_COMM_WORLD),
          "MPI_Allreduce(gather chunk allocation)");
      if (any_round_allocation_failed)
        throw std::runtime_error("root could not allocate a gathered chunk");
      const int send_count =
          offset < local_length
              ? static_cast<int>(std::min<unsigned long long>(
                    local_length - offset, static_cast<unsigned long long>(capacity)))
              : 0;
      detail::require_mpi_success(
          MPI_Gatherv(detail::chunk_pointer(payload, offset, send_count), send_count, MPI_BYTE,
                      me == root ? round.data() : nullptr, me == root ? counts.data() : nullptr,
                      me == root ? displacements.data() : nullptr, MPI_BYTE, root, MPI_COMM_WORLD),
          "MPI_Gatherv(payload chunk)");
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
      throw std::out_of_range("serial gather root must be zero");
    return std::vector<std::string>{payload};
#endif
  }

 private:
  WorldCommunicator() = default;

  static WorldCommunicator& instance() {
    static WorldCommunicator value;
    return value;
  }
};

}  // namespace pops
