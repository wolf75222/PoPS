#pragma once

#include <pops/numerics/time/amr/reflux/amr_flux_helpers.hpp>  // mf_eval_rhs, average-down and coarse/fine interpolation
#include <pops/numerics/time/amr/levels/amr_patch_range.hpp>  // PatchRange, FluxRegister, CoverageMask, CoarseFineInterface, fill_periodic_local, mf_fill_fine_ghosts_multi, mf_average_down_multi
#include <pops/numerics/time/amr/levels/amr_subcycling.hpp>  // AmrLevelMP, prepared fill/reflux/average-down storage

/// @file
/// @brief Umbrella for the AMR MultiFab stack: includes the numerics/time sub-headers in
///        dependency order (flux_helpers -> patch_range -> hierarchy/reflux storage).
///
/// Layer: `include/pops/numerics/time`.
/// Role: single entry point for the AMR MultiFab/multi-patch stack. Every existing includer of
///        this header exposes spatial AMR building blocks only. ProgramGraph owns time integration.
