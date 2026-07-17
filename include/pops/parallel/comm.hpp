#pragma once

/// @file
/// @brief Parallel seam: minimal MPI abstraction (rank/size + collectives) with serial fallback.
///
/// Layer: `include/pops/parallel`.
/// Role: expose my_rank()/n_ranks() and a fixed set of global reductions (sum/min/max on
/// double and long, in-place sum/max on a double array, in-place OR on a marker array)
/// behind a single facade. Without POPS_HAS_MPI everything compiles to a serial identity.  A few
/// performance-critical native algorithms use MPI directly; the process contract below therefore
/// requires full MPI thread support rather than pretending that a header-local lock can serialize
/// calls issued by every executable and shared object in the process.
/// Contract: every collective operates on MPI_COMM_WORLD; each rank must call them
/// in the same order (otherwise deadlock). The sum_inplace / or_inplace bricks feed
/// respectively the multi-patch AMR reflux and the gathering of regrid tags before
/// clustering; all_reduce_min guarantees a global dt identical on all ranks.
///
/// Invariants:
/// - even compiled with MPI, if MPI is not initialized my_rank()=0 / n_ranks()=1 (comm_active()
///   tests Initialized && !Finalized); call comm_init() at the start of main() for a distributed run;
/// - the all_reduce_* functions are COLLECTIVE: all ranks participate or none;
/// - in serial mode each function is the identity (no-op or returns the argument).

#ifdef POPS_HAS_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {

#ifdef POPS_HAS_MPI

namespace detail {

inline bool& mpi_initialized_by_pops_flag() {
  static bool value = false;
  return value;
}

inline bool& mpi_atexit_finalize_registered_flag() {
  static bool value = false;
  return value;
}

/// Local lifecycle serialization only.  This mutex is deliberately *not* presented as a
/// process-wide MPI-call lock: an inline header object cannot provide that guarantee across DSOs.
/// Concurrent MPI calls are instead covered by the process-global MPI_THREAD_MULTIPLE contract.
inline std::mutex& mpi_lifecycle_mutex() {
  static std::mutex value;
  return value;
}

[[noreturn]] inline void throw_mpi_error(int code, std::string_view operation) {
  char message[MPI_MAX_ERROR_STRING] = {};
  int length = 0;
  if (MPI_Error_string(code, message, &length) != MPI_SUCCESS || length <= 0)
    throw std::runtime_error(std::string(operation) + " failed with MPI error " +
                             std::to_string(code));
  throw std::runtime_error(std::string(operation) +
                           " failed: " + std::string(message, static_cast<std::size_t>(length)));
}

inline void require_mpi_success(int code, std::string_view operation) {
  if (code != MPI_SUCCESS)
    throw_mpi_error(code, operation);
}

inline bool comm_active_unlocked() noexcept {
  int initialized = 0;
  int finalized = 0;
  if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized)
    return false;
  if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized)
    return false;
  return true;
}

inline void finalize_owned_mpi_unlocked() noexcept {
  if (!mpi_initialized_by_pops_flag())
    return;
  if (!comm_active_unlocked())
    return;
  if (MPI_Finalize() == MPI_SUCCESS)
    mpi_initialized_by_pops_flag() = false;
}

inline void finalize_owned_mpi() noexcept {
  try {
    std::lock_guard<std::mutex> guard(mpi_lifecycle_mutex());
    finalize_owned_mpi_unlocked();
  } catch (...) {
    // Process-exit finalizers must never emit an exception.
  }
}

inline void ensure_mpi_world_initialized(int* argc = nullptr, char*** argv = nullptr) {
  // MPI_Init_thread/MPI_Finalize remain lifecycle operations: callers must establish the world
  // before starting worker threads and must not finalize it while native work is in flight.
  std::lock_guard<std::mutex> guard(mpi_lifecycle_mutex());
  int finalized = 0;
  require_mpi_success(MPI_Finalized(&finalized), "MPI_Finalized");
  if (finalized)
    throw std::runtime_error("MPI_COMM_WORLD cannot be reactivated after MPI_Finalize");

  int initialized = 0;
  require_mpi_success(MPI_Initialized(&initialized), "MPI_Initialized");
  if (!initialized) {
    if (!mpi_atexit_finalize_registered_flag()) {
      if (std::atexit(&finalize_owned_mpi) != 0)
        throw std::runtime_error("failed to register the PoPS-owned MPI finalizer");
      mpi_atexit_finalize_registered_flag() = true;
    }
    int provided = MPI_THREAD_SINGLE;
    const int code = MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
    if (code != MPI_SUCCESS)
      throw_mpi_error(code, "MPI_Init_thread");
    mpi_initialized_by_pops_flag() = true;
    if (provided < MPI_THREAD_MULTIPLE) {
      finalize_owned_mpi_unlocked();
      throw std::runtime_error("MPI_COMM_WORLD requires MPI_THREAD_MULTIPLE support");
    }
    return;
  }

  int provided = MPI_THREAD_SINGLE;
  require_mpi_success(MPI_Query_thread(&provided), "MPI_Query_thread");
  if (provided < MPI_THREAD_MULTIPLE)
    throw std::runtime_error("externally initialized MPI provides less than MPI_THREAD_MULTIPLE");
}

}  // namespace detail

inline bool comm_active() {
  return detail::comm_active_unlocked();
}

inline void comm_init(int* argc = nullptr, char*** argv = nullptr) {
  detail::ensure_mpi_world_initialized(argc, argv);
}

inline void comm_finalize() {
  // Never steal lifecycle ownership from an embedding application.  Calls after an external
  // MPI_Init are deliberately a no-op; PoPS closes only the MPI runtime it initialized itself.
  detail::finalize_owned_mpi();
}

inline int my_rank() {
  if (!detail::comm_active_unlocked())
    return 0;
  int r = 0;
  detail::require_mpi_success(MPI_Comm_rank(MPI_COMM_WORLD, &r), "MPI_Comm_rank");
  return r;
}

inline int n_ranks() {
  if (!detail::comm_active_unlocked())
    return 1;
  int s = 1;
  detail::require_mpi_success(MPI_Comm_size(MPI_COMM_WORLD, &s), "MPI_Comm_size");
  return s;
}

inline void barrier() {
  if (detail::comm_active_unlocked())
    detail::require_mpi_success(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
}

inline double all_reduce_sum(double x) {
  if (!detail::comm_active_unlocked())
    return x;
  double r = x;
  detail::require_mpi_success(MPI_Allreduce(&x, &r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
                              "MPI_Allreduce(double sum)");
  return r;
}

inline double all_reduce_max(double x) {
  if (!detail::comm_active_unlocked())
    return x;
  double r = x;
  detail::require_mpi_success(MPI_Allreduce(&x, &r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
                              "MPI_Allreduce(double max)");
  return r;
}

// Global min (counterpart of all_reduce_max). Brick for GLOBAL time step bounds
// (System::add_dt_bound): the host callback is evaluated PER RANK, the global min guarantees a dt
// IDENTICAL on all ranks (otherwise the collectives of the step -- Krylov, fill_boundary --
// would diverge -> deadlock). In serial: identity.
inline double all_reduce_min(double x) {
  if (!detail::comm_active_unlocked())
    return x;
  double r = x;
  detail::require_mpi_success(MPI_Allreduce(&x, &r, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD),
                              "MPI_Allreduce(double min)");
  return r;
}

// Element-by-element sum of an array, in place, on all ranks. Base brick of the
// distributed multi-patch AMR reflux: each rank fills the contributions of
// its local patches (0 elsewhere), all-reduce -> each rank has the complete register.
inline void all_reduce_sum_inplace(double* buf, int n) {
  if (!detail::comm_active_unlocked() || n <= 0)
    return;
  detail::require_mpi_success(
      MPI_Allreduce(MPI_IN_PLACE, buf, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
      "MPI_Allreduce(double inplace sum)");
}

/// Element-by-element maximum of an array, in place, on all ranks.  This batches related scalar
/// convergence witnesses into one collective without changing their individual MPI_MAX semantics.
inline void all_reduce_max_inplace(double* buf, int n) {
  if (!detail::comm_active_unlocked() || n <= 0)
    return;
  detail::require_mpi_success(
      MPI_Allreduce(MPI_IN_PLACE, buf, n, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
      "MPI_Allreduce(double inplace max)");
}

inline long all_reduce_sum(long x) {
  if (!detail::comm_active_unlocked())
    return x;
  long r = x;
  detail::require_mpi_success(MPI_Allreduce(&x, &r, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD),
                              "MPI_Allreduce(long sum)");
  return r;
}

inline long all_reduce_max(long x) {
  if (!detail::comm_active_unlocked())
    return x;
  long r = x;
  detail::require_mpi_success(MPI_Allreduce(&x, &r, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD),
                              "MPI_Allreduce(long max)");
  return r;
}

inline long all_reduce_min(long x) {
  if (!detail::comm_active_unlocked())
    return x;
  long r = x;
  detail::require_mpi_success(MPI_Allreduce(&x, &r, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD),
                              "MPI_Allreduce(long min)");
  return r;
}

// Element-by-element logical OR of a marker array (0/1), in place, on all ranks.
// Brick of the DISTRIBUTED-COARSE AMR regrid: each rank tags ONLY its local coarse
// boxes (tag_cells iterating over local_size()), so nobody sees the complete tag grid;
// the global OR gathers the tags on each rank before the Berger-Rigoutsos clustering, which then
// produces IDENTICAL fine patches everywhere (otherwise the fine BoxArray would differ per rank -> MPI
// desynchronized). See the note in tag_box.hpp ("the distributed tags will be gathered before clustering").
inline void all_reduce_or_inplace(char* buf, std::size_t n) {
  if (!detail::comm_active_unlocked() || n == 0)
    return;
  while (n != 0) {
    const int count =
        static_cast<int>(std::min(n, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    detail::require_mpi_success(
        MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_CHAR, MPI_BOR, MPI_COMM_WORLD),
        "MPI_Allreduce(char inplace or)");
    buf += count;
    n -= static_cast<std::size_t>(count);
  }
}

/// Batched exact-consensus witnesses for replicated byte arrays.
inline void all_reduce_min_inplace(char* buf, std::size_t n) {
  if (!detail::comm_active_unlocked() || n == 0)
    return;
  while (n != 0) {
    const int count =
        static_cast<int>(std::min(n, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    detail::require_mpi_success(
        MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_CHAR, MPI_MIN, MPI_COMM_WORLD),
        "MPI_Allreduce(char inplace min)");
    buf += count;
    n -= static_cast<std::size_t>(count);
  }
}

inline void all_reduce_max_inplace(char* buf, std::size_t n) {
  if (!detail::comm_active_unlocked() || n == 0)
    return;
  while (n != 0) {
    const int count =
        static_cast<int>(std::min(n, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    detail::require_mpi_success(
        MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_CHAR, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(char inplace max)");
    buf += count;
    n -= static_cast<std::size_t>(count);
  }
}

/// Batched structural consensus for canonical integral payloads.
inline void all_reduce_min_inplace(long* buf, std::size_t n) {
  if (!detail::comm_active_unlocked() || n == 0)
    return;
  while (n != 0) {
    const int count =
        static_cast<int>(std::min(n, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    detail::require_mpi_success(
        MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_LONG, MPI_MIN, MPI_COMM_WORLD),
        "MPI_Allreduce(long inplace min)");
    buf += count;
    n -= static_cast<std::size_t>(count);
  }
}

inline void all_reduce_max_inplace(long* buf, std::size_t n) {
  if (!detail::comm_active_unlocked() || n == 0)
    return;
  while (n != 0) {
    const int count =
        static_cast<int>(std::min(n, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    detail::require_mpi_success(
        MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_LONG, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(long inplace max)");
    buf += count;
    n -= static_cast<std::size_t>(count);
  }
}

#else  // ----- serial -----

inline bool comm_active() {
  return false;
}
inline void comm_init(int* = nullptr, char*** = nullptr) {}
inline void comm_finalize() {}
inline int my_rank() {
  return 0;
}
inline int n_ranks() {
  return 1;
}
inline void barrier() {}
inline double all_reduce_sum(double x) {
  return x;
}
inline double all_reduce_max(double x) {
  return x;
}
inline double all_reduce_min(double x) {
  return x;
}
inline long all_reduce_sum(long x) {
  return x;
}
inline long all_reduce_max(long x) {
  return x;
}
inline long all_reduce_min(long x) {
  return x;
}
inline void all_reduce_sum_inplace(double*, int) {}        // serial: identity
inline void all_reduce_max_inplace(double*, int) {}        // serial: identity
inline void all_reduce_or_inplace(char*, std::size_t) {}   // serial: identity
inline void all_reduce_min_inplace(char*, std::size_t) {}  // serial: identity
inline void all_reduce_max_inplace(char*, std::size_t) {}  // serial: identity
inline void all_reduce_min_inplace(long*, std::size_t) {}  // serial: identity
inline void all_reduce_max_inplace(long*, std::size_t) {}  // serial: identity

#endif

/// Read-only MPI lifecycle facts.  These observers never initialize MPI.
inline bool mpi_initialized_by_pops() noexcept {
#ifdef POPS_HAS_MPI
  try {
    std::lock_guard<std::mutex> guard(detail::mpi_lifecycle_mutex());
    return detail::mpi_initialized_by_pops_flag();
  } catch (...) {
    return false;
  }
#else
  return false;
#endif
}

inline bool mpi_atexit_finalize_registered() noexcept {
#ifdef POPS_HAS_MPI
  try {
    std::lock_guard<std::mutex> guard(detail::mpi_lifecycle_mutex());
    return detail::mpi_atexit_finalize_registered_flag();
  } catch (...) {
    return false;
  }
#else
  return false;
#endif
}

inline int mpi_thread_level() noexcept {
#ifdef POPS_HAS_MPI
  int initialized = 0;
  int finalized = 0;
  int provided = MPI_THREAD_SINGLE;
  if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized ||
      MPI_Finalized(&finalized) != MPI_SUCCESS || finalized ||
      MPI_Query_thread(&provided) != MPI_SUCCESS)
    return MPI_THREAD_SINGLE;
  return provided;
#else
  return 0;
#endif
}

/// Exact collective consensus for an already-canonical ordered sequence of byte pairs.
///
/// The sequence length and every component length are agreed with integral reductions before any
/// byte collective.  Ranks with a missing/extra pair or a differently-sized slot/identity therefore
/// return uniformly without entering incompatible payload collectives.  Once those structural
/// lengths agree, concatenation is unambiguous and element-wise minima/maxima provide an exact (not
/// hashed) equality witness.  Callers own canonical ordering; field-plan registries pass std::map
/// iteration order over (provider_slot, plan_identity).
inline bool all_ranks_agree_exact_ordered_byte_pairs(
    const std::vector<std::pair<std::string_view, std::string_view>>& values) {
  const long local_count = static_cast<long>(values.size());
  const long minimum_count = all_reduce_min(local_count);
  const long maximum_count = all_reduce_max(local_count);
  if (minimum_count != maximum_count)
    return false;

  std::vector<long> minimum_lengths;
  minimum_lengths.reserve(values.size() * 2);
  std::size_t payload_size = 0;
  for (const auto& value : values) {
    minimum_lengths.push_back(static_cast<long>(value.first.size()));
    minimum_lengths.push_back(static_cast<long>(value.second.size()));
    payload_size += value.first.size() + value.second.size();
  }
  std::vector<long> maximum_lengths(minimum_lengths);
  all_reduce_min_inplace(minimum_lengths.data(), minimum_lengths.size());
  all_reduce_max_inplace(maximum_lengths.data(), maximum_lengths.size());
  if (minimum_lengths != maximum_lengths)
    return false;

  std::vector<char> minimum;
  minimum.reserve(payload_size);
  for (const auto& value : values) {
    minimum.insert(minimum.end(), value.first.begin(), value.first.end());
    minimum.insert(minimum.end(), value.second.begin(), value.second.end());
  }
  std::vector<char> maximum(minimum);
  all_reduce_min_inplace(minimum.data(), minimum.size());
  all_reduce_max_inplace(maximum.data(), maximum.size());
  return minimum == maximum;
}

}  // namespace pops
