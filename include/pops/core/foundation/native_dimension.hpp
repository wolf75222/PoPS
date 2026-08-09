#pragma once

namespace pops {

#ifndef POPS_NATIVE_DIM
#error "PoPS requires an explicit POPS_NATIVE_DIM=1, 2, or 3 compile-time specialization"
#endif

static_assert(POPS_NATIVE_DIM >= 1 && POPS_NATIVE_DIM <= 3,
              "POPS_NATIVE_DIM must be exactly 1, 2, or 3");

/// Immutable spatial rank of this compiled native artifact.
inline constexpr int kNativeDimension = POPS_NATIVE_DIM;

}  // namespace pops
