#pragma once

#include "load_balance_test_authority.hpp"

#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr/bootstrap_transfer_builtins.hpp>

#include <cstddef>
#include <stdexcept>

namespace pops::test {

template <int Dim, class MemorySpace>
inline void install_second_order_amr_transfer_authorities(
    ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime, std::size_t block_count) {
  static_assert(Dim >= 1 && Dim <= 3);
  // AMR transfer preparation is now a typed operation on the exact source/destination views and
  // level spatial contracts.  The runtime owns no mutable block-wide transfer selector to install.
  // Keep this test seam only as an explicit witness that a live exact-ranked runtime has been
  // prepared before a test constructs those operations through `prepare_transfer`.
  if (block_count == 0 || runtime.hierarchy().num_levels() == 0)
    throw std::invalid_argument("test AMR transfer authority requires live blocks and levels");
}

}  // namespace pops::test
