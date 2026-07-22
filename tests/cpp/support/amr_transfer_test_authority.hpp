#pragma once

#include "load_balance_test_authority.hpp"

#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr/bootstrap_transfer_builtins.hpp>

#include <cstddef>

namespace pops::test {

inline void install_second_order_amr_transfer_authorities(
    AmrRuntime& runtime, std::size_t block_count, int refinement_ratio = kAmrRefRatio) {
  for (std::size_t block = 0; block < block_count; ++block) {
    runtime.set_block_transfer_authority(
        block, ::pops::runtime::amr::prepare_conservative_linear(),
        ::pops::runtime::amr::prepare_volume_average(),
        ::pops::runtime::amr::prepare_conservative_coarse_fine(),
        ::pops::runtime::amr::prepare_linear_time_interpolation(), refinement_ratio);
  }
}

}  // namespace pops::test
