#pragma once

/// @file
/// @brief Preserve the first rank's exception diagnostic before leaving a collective phase.

#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <cstddef>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>

namespace pops {

/// All ranks enter with their local failure, or nullptr. Serial execution retains the original
/// exception type; a distributed failure selects the lowest failing rank and preserves its reason
/// on every rank. Callers select which failure class has priority before entering this boundary.
inline void collectively_rethrow_exception(std::exception_ptr local_error,
                                           const ExecutionLane& lane, std::string_view message) {
  const auto communicator = lane.communicator();
  const long failed = local_error ? 1L : 0L;
  if (all_reduce_max(failed, communicator) == 0)
    return;
  if (lane.size() == 1 && local_error)
    std::rethrow_exception(local_error);
  const long root = all_reduce_min(
      failed != 0 ? static_cast<long>(lane.rank()) : static_cast<long>(lane.size()), communicator);
  const bool authoritative = lane.rank() == root;
  std::string diagnostic;
  long preparation_failed = 0;
  try {
    if (authoritative) {
      diagnostic = std::string(message) + "; rank " + std::to_string(root) + ": ";
      try {
        std::rethrow_exception(local_error);
      } catch (const std::exception& error) {
        diagnostic += error.what();
      } catch (...) {
        diagnostic += "non-standard exception";
      }
    }
  } catch (...) {
    preparation_failed = 1;
  }
  if (all_reduce_max(preparation_failed, communicator) != 0)
    throw std::bad_alloc();
  const long invalid_length =
      diagnostic.size() > static_cast<std::size_t>(std::numeric_limits<long>::max()) ? 1L : 0L;
  if (all_reduce_max(invalid_length, communicator) != 0)
    throw std::length_error("collective exception diagnostic exceeds long capacity");
  const long length = all_reduce_max(static_cast<long>(diagnostic.size()), communicator);
  long allocation_failed = 0;
  try {
    diagnostic.resize(static_cast<std::size_t>(length));
  } catch (...) {
    allocation_failed = 1;
  }
  if (all_reduce_max(allocation_failed, communicator) != 0)
    throw std::bad_alloc();
  broadcast_bytes_inplace(diagnostic.data(), diagnostic.size(), static_cast<int>(root),
                          communicator);
  throw std::runtime_error(diagnostic);
}

}  // namespace pops
