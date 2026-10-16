/// @file
/// @brief Borrowed request for native accepted-continuation ownership
/// rematerialization.
#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace pops::runtime::program {
struct AmrProgramContinuationLevel;
struct AmrContinuationMigrationRequest {
  std::span<const std::vector<std::uint8_t>> source_states;
  std::span<const std::vector<int>> source_level_owners;
  std::span<const std::string> source_level_modes;
  std::uint64_t target_epoch, target_generation;
  double accepted_time;
  std::int64_t accepted_step;
  std::uint64_t workspace_byte_capacity;
};
using AmrContinuationRematerializer =
    std::function<std::vector<AmrProgramContinuationLevel>(const AmrContinuationMigrationRequest&)>;
}  // namespace pops::runtime::program
