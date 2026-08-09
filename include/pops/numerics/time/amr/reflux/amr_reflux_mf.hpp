/// @file
/// @brief Ranked AMR transfer, subcycling, and transactional metric-reflux umbrella.

#pragma once

#include <pops/amr/reflux/face_flux_ledger.hpp>
#include <pops/amr/reflux/metric_reflux.hpp>
#include <pops/numerics/time/amr/levels/amr_patch_range.hpp>
#include <pops/numerics/time/amr/levels/amr_subcycling.hpp>
#include <pops/numerics/time/amr/reflux/amr_flux_helpers.hpp>

// This header intentionally declares no compatibility storage.  Every exposed authority retains
// the compile-time rank selected by Python and carried by AmrRuntime<Dim, MemorySpace>.
