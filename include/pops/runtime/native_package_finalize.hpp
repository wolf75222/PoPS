#pragma once

#include <exception>
#include <stdexcept>

namespace pops {

inline constexpr const char kNativePackageFinalizeCollectiveMessage[] =
    "System native package finalization rolled back collectively";

/// After collective rollback, emit the rank-local inner exception when present.
/// Ranks that only observed the collective failure still throw the generic
/// message. ``lane.size()`` must not hide a captured rank-0 diagnostic.
[[noreturn]] inline void rethrow_native_package_finalize_failure(std::exception_ptr failure) {
  if (failure)
    std::rethrow_exception(failure);
  throw std::runtime_error(kNativePackageFinalizeCollectiveMessage);
}

}  // namespace pops
