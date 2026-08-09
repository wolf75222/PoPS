#pragma once

/// @file
/// @brief Public exact-ranked Euler spelling.
///
/// The sole ideal-gas Euler implementation is `pops::nd::IdealGasEuler<Dim>` in the
/// dimension-generic conservation-law layer.  This header intentionally provides names only:
/// it does not wrap, adapt, or reimplement a second flux/EOS path.

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>

namespace pops {

template <int Dim>
using EulerND = nd::IdealGasEuler<Dim>;

using Euler = EulerND<kNativeDimension>;

}  // namespace pops
