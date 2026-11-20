/// @file
/// @brief Host-side logical-clock authority for a prepared physical ghost application.

#pragma once

#include <cstdint>
#include <exception>
#include <string_view>

namespace pops::runtime::program {

template <int Dim>
class ProgramContext;

struct BoundaryGhostEvaluation {
  std::string_view clock;
  std::int64_t tick = 0;
};

/// One synchronous request. System resolves any local error inside the group preflight.
class PreparedChildBoundaryEvaluation {
 public:
  PreparedChildBoundaryEvaluation(const PreparedChildBoundaryEvaluation&) = delete;
  PreparedChildBoundaryEvaluation& operator=(const PreparedChildBoundaryEvaluation&) = delete;

  const BoundaryGhostEvaluation& require() const {
    if (error_)
      std::rethrow_exception(error_);
    return evaluation_;
  }

 private:
  template <int Dim>
  friend class ProgramContext;

  PreparedChildBoundaryEvaluation() = default;

  BoundaryGhostEvaluation evaluation_;
  std::exception_ptr error_;
};

}  // namespace pops::runtime::program
