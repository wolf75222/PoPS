#pragma once

#include <pops/core/foundation/types.hpp>

#include <string>

namespace pops {

/// One authenticated member of an ordered FieldSolvePlan provider pack.  Runtime dispatch is a
/// direct lookup by owner + native key; no provider class/kind enum participates in the solve.
struct FieldProviderBinding {
  std::string identity;
  std::string owner_block;
  std::string native_key;
  Real coefficient = Real(1);
};

}  // namespace pops
