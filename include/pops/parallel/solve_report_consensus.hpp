#pragma once

/// @file
/// @brief Allocation-free exact MPI consensus for provider-owned SolveReport values.

#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace pops {

/// Fixed scratch for exact communicator-wide agreement on one SolveReport.
///
/// Numeric fields are compared by their exact trivially-copyable representation, including the sign
/// bit of zero. The unbounded provider reason is compared in fixed-size chunks after its exact
/// length has reached consensus. Consequently this boundary neither hashes provider output nor
/// allocates in the solve hot path, and it remains independent of solver family and provider type.
class ExactSolveReportConsensusScratch {
 public:
  [[nodiscard]] bool agrees(const SolveReport& report, const ExecutionLane& lane) {
    fixed_minimum_.fill(char{0});
    std::size_t offset = 0;
    append(std::int64_t{report.iters}, offset);
    append(report.rel_residual, offset);
    append(report.reference_residual_norm, offset);
    append(report.residual_norm, offset);
    append(static_cast<std::int32_t>(report.status), offset);
    append(static_cast<std::int32_t>(report.action), offset);
    append(static_cast<std::uint64_t>(report.reason.size()), offset);

    fixed_maximum_ = fixed_minimum_;
    all_reduce_min_inplace(fixed_minimum_.data(), offset, lane);
    all_reduce_max_inplace(fixed_maximum_.data(), offset, lane);
    if (!std::equal(fixed_minimum_.begin(), fixed_minimum_.begin() + offset,
                    fixed_maximum_.begin()))
      return false;

    for (std::size_t begin = 0; begin < report.reason.size(); begin += kReasonChunkBytes) {
      const std::size_t count = std::min(kReasonChunkBytes, report.reason.size() - begin);
      std::memcpy(reason_minimum_.data(), report.reason.data() + begin, count);
      std::memcpy(reason_maximum_.data(), report.reason.data() + begin, count);
      all_reduce_min_inplace(reason_minimum_.data(), count, lane);
      all_reduce_max_inplace(reason_maximum_.data(), count, lane);
      if (!std::equal(reason_minimum_.begin(), reason_minimum_.begin() + count,
                      reason_maximum_.begin()))
        return false;
    }
    return true;
  }

  [[nodiscard]] bool agrees(const SolveReport& report) {
    const ExecutionLane lane = ExecutionLane::world();
    return agrees(report, lane);
  }

 private:
  template <class Value>
  void append(Value value, std::size_t& offset) noexcept {
    static_assert(std::is_trivially_copyable_v<Value>);
    static_assert(sizeof(Value) <= kFixedPayloadBytes);
    std::memcpy(fixed_minimum_.data() + offset, &value, sizeof(Value));
    offset += sizeof(Value);
  }

  static constexpr std::size_t kFixedPayloadBytes =
      sizeof(std::int64_t) + 3 * sizeof(Real) + 2 * sizeof(std::int32_t) + sizeof(std::uint64_t);
  static constexpr std::size_t kReasonChunkBytes = 1024;
  std::array<char, kFixedPayloadBytes> fixed_minimum_{};
  std::array<char, kFixedPayloadBytes> fixed_maximum_{};
  std::array<char, kReasonChunkBytes> reason_minimum_{};
  std::array<char, kReasonChunkBytes> reason_maximum_{};
};

}  // namespace pops
