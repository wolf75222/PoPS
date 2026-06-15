#pragma once
// Parapluie AMR MultiFab : inclut les sous-entetes dans l'ordre de dependance.
// Tout inclueur existant de ce header continue de compiler sans modification.

#include <adc/numerics/time/amr_flux_helpers.hpp>   // mf_advance_faces, mf_apply_source*, mf_average_down, fill_cf_ghost_cell, mf_fill_fine_ghosts_t
#include <adc/numerics/time/amr_level.hpp>          // detail::AmrLevelMF, amr_step_2level_mf, subcycle_level_mf, amr_step_multilevel_mf
#include <adc/numerics/time/amr_patch_range.hpp>    // PatchRange, FluxRegister, CoverageMask, SubcyclingSchedule, CoarseFineInterface, fill_periodic_local, mf_fill_fine_ghosts_multi, mf_average_down_multi
#include <adc/numerics/time/amr_subcycling.hpp>     // AmrLevelMP, RegMP, mf_find_box, coarsen_grown, mf_fill_fine_ghosts_mb, mf_average_down_mb, amr_step_2level_multipatch, detail::subcycle_level_mp, detail::amr_step_multilevel_multipatch
#include <adc/numerics/time/amr_advance.hpp>        // OwnershipPolicy, LevelHierarchy, advance_amr

/// @file
/// @brief Parapluie de la pile AMR MultiFab : inclut les sous-entetes numerics/time dans l'ordre
///        de dependance (flux_helpers -> level -> patch_range -> subcycling -> advance).
///
/// Couche : `include/adc/numerics/time`.
/// Role : point d'entree unique de l'AMR MultiFab/multi-patch. Tout inclueur existant de ce
///        header continue de compiler sans modification apres l'eclatement en sous-entetes ;
///        l'API publique de production reste advance_amr (cf. amr_advance.hpp).
