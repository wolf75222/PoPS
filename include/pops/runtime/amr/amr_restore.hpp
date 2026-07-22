/// @file
/// @brief Out-of-line AmrRuntime member definitions kept out of amr_runtime.hpp (its line budget):
/// the ADC-542 level-composite reductions (composite_reduce + its folds), the checkpoint
/// hierarchy-rebuild seam (rebuild_hierarchy), and the regrid / clustering config setters
/// (set_regrid, set_clustering). Included at the END of amr_runtime.hpp, so the full AmrRuntime class
/// is visible; NOT a standalone header (it defines AmrRuntime members).
///
/// rebuild_hierarchy imposes a mid-run hierarchy from a v3 checkpoint by REUSING the regrid R6/R7
/// machinery (amr_runtime.hpp regrid()) MINUS tagging / clustering / prolong: the checkpoint supplies
/// the layout (BoxArrays + DistributionMappings) AND the data (the per-level state restore overwrites
/// every valid cell), so the divergence argument of the frozen-hierarchy limitation evaporates -- a
/// restart that imposes the exact mid-run hierarchy makes every post-restart regrid reproduce the
/// uninterrupted layout sequence (the determinism theorem, ADC-542 addendum B.2).

#pragma once

#include <pops/runtime/amr/amr_runtime.hpp>  // the class this file defines members of
#include <pops/runtime/amr/composite_reduction.hpp>

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

namespace pops {

// --- ADC-542 composite reductions (declared in amr_runtime.hpp) -----------------------------------

inline std::size_t AmrRuntime::block_index_by_name_(const std::string& name) const {
  for (std::size_t b = 0; b < blocks_.size(); ++b)
    if (blocks_[b].name == name)
      return b;
  throw std::runtime_error("AmrRuntime::composite_reduce : no block named '" + name + "'");
}

inline double AmrRuntime::composite_reduce(const std::string& block, const std::string& kind,
                                           int comp, const std::vector<int>& levels) const {
  const std::size_t b = block_index_by_name_(block);
  return runtime::amr::composite_reduce_levels(*blocks_[b].levels, replicated_coarse_, kind, comp,
                                               levels);
}

inline double AmrRuntime::composite_reduce_field(const std::string& provider_slot,
                                                 const std::string& kind, int comp,
                                                 const std::vector<int>& levels) {
  const int count = provider_potential_levels(provider_slot);
  if (blocks_.empty() || blocks_.front().levels == nullptr ||
      static_cast<int>(blocks_.front().levels->size()) != count)
    throw std::runtime_error(
        "AmrRuntime::composite_reduce_field: field and shared state hierarchies disagree");
  std::vector<const MultiFab*> values;
  std::vector<std::pair<Real, Real>> metrics;
  values.reserve(static_cast<std::size_t>(count));
  metrics.reserve(static_cast<std::size_t>(count));
  for (int level = 0; level < count; ++level) {
    values.push_back(&provider_potential_level(provider_slot, level));
    const AmrLevelMP& shared = blocks_.front().levels->at(static_cast<std::size_t>(level));
    metrics.emplace_back(shared.dx, shared.dy);
  }
  return runtime::amr::composite_reduce_fields(values, metrics, replicated_coarse_, kind, comp,
                                               levels);
}

inline std::vector<int> AmrRuntime::level_owner_ranks(int k) const {
  if (k < 0 || k >= nlev_)
    throw std::runtime_error("AmrRuntime::level_owner_ranks : level out of bounds");
  return hierarchy_.dm[static_cast<std::size_t>(k)].ranks();
}

// FULL shared aux of level k: ALL aux_ncomp_ components of aux_[k], LOCAL valid cells at
// level-domain-relative component-major flat indices (zeros outside the patches at a fine level) -- the
// exact layout of block_level_state, so the v3 checkpoint reader/writer share one convention. phi
// (comp 0) is included; the level-0 multigrid WARM START stays a separate phi_<k> payload
// (level_potential), which reads mg_.phi(), not aux_[0].
inline std::vector<double> AmrRuntime::level_aux_flat(int k) const {
  if (k < 0 || k >= nlev_)
    throw std::runtime_error("AmrRuntime::level_aux_flat : level out of bounds");
  const MultiFab& A = aux_[k];
  const int nc = A.ncomp();
  const Box2D level_domain = amr_level_index_domain(dom_, k);
  const std::size_t nx = static_cast<std::size_t>(level_domain.nx());
  const std::size_t cells = nx * static_cast<std::size_t>(level_domain.ny());
  std::vector<double> out(static_cast<std::size_t>(nc) * cells, 0.0);
  device_fence();
  for (int li = 0; li < A.local_size(); ++li) {
    const ConstArray4 a = A.fab(li).const_array();
    const Box2D v = A.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        for (int c = 0; c < nc; ++c)
          out[static_cast<std::size_t>(c) * cells +
              static_cast<std::size_t>(j - level_domain.lo[1]) * nx +
              static_cast<std::size_t>(i - level_domain.lo[0])] = a(i, j, c);
  }
  return out;
}

// Global form of level_aux_flat. Ownership-distributed levels are gathered from their disjoint
// per-rank contributions; replicated level 0 is already complete on every rank and must not be
// reduced (which would multiply the checkpoint payload by n_ranks).
inline std::vector<double> AmrRuntime::level_aux_flat_global(int k) const {
  std::vector<double> out = level_aux_flat(k);
  if (k > 0 || !replicated_coarse_)
    all_reduce_sum_inplace(out.data(), out.size());
  return out;
}

// Restores the FULL shared aux of level k from the flat layout above. Writes ONLY the VALID cells of
// the LOCAL fabs (owner-rank writes; a rank without a box is a no-op); the ghosts are redone by the
// next solve_fields, exactly like after a regrid.
inline void AmrRuntime::set_level_aux_flat(int k, const std::vector<double>& v) {
  if (k < 0 || k >= nlev_)
    throw std::runtime_error("AmrRuntime::set_level_aux_flat : level out of bounds");
  MultiFab& A = aux_[k];
  const int nc = A.ncomp();
  const Box2D level_domain = amr_level_index_domain(dom_, k);
  const std::size_t nx = static_cast<std::size_t>(level_domain.nx());
  const std::size_t cells = nx * static_cast<std::size_t>(level_domain.ny());
  if (v.size() != static_cast<std::size_t>(nc) * cells)
    throw std::runtime_error(
        "AmrRuntime::set_level_aux_flat : aux size differs from ncomp*level_cells");
  device_fence();
  for (int li = 0; li < A.local_size(); ++li) {
    Array4 a = A.fab(li).array();
    const Box2D b = A.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        for (int c = 0; c < nc; ++c)
          a(i, j, c) = v[static_cast<std::size_t>(c) * cells +
                         static_cast<std::size_t>(j - level_domain.lo[1]) * nx +
                         static_cast<std::size_t>(i - level_domain.lo[0])];
  }
}

// --- ADC-542 hierarchy rebuild (v3 checkpoint restore) --------------------------------------------

inline void AmrRuntime::rebuild_hierarchy(const std::vector<std::vector<PatchBox>>& level_boxes,
                                          const std::vector<std::vector<int>>& level_owner_ranks) {
  const int n_levels = static_cast<int>(level_boxes.size());
  if (n_levels < 1)
    throw std::runtime_error("AmrRuntime::rebuild_hierarchy : need at least the coarse level (0)");
  if (level_owner_ranks.size() != level_boxes.size())
    throw std::runtime_error(
        "AmrRuntime::rebuild_hierarchy : level_boxes and level_owner_ranks length mismatch");
  if (n_levels > max_levels())
    throw std::runtime_error(
        "AmrRuntime::rebuild_hierarchy : checkpoint has " + std::to_string(n_levels) +
        " active levels but the replayed composition resolves a maximum of " +
        std::to_string(max_levels()));

  // Validate and materialize the complete target topology before replacing any accepted storage.
  // Level zero is the composition-owned base layout and is intentionally absent from patch_boxes();
  // every fine level must be a non-empty contiguous prefix with an explicit owner per patch.
  std::vector<BoxArray> target_boxes(static_cast<std::size_t>(n_levels));
  std::vector<DistributionMapping> target_mappings(static_cast<std::size_t>(n_levels));
  std::vector<Real> target_dx(static_cast<std::size_t>(n_levels));
  std::vector<Real> target_dy(static_cast<std::size_t>(n_levels));
  target_boxes[0] = hierarchy_.ba[0];
  target_mappings[0] = hierarchy_.dm[0];
  target_dx[0] = hierarchy_.dx[0];
  target_dy[0] = hierarchy_.dy[0];
  if (!level_boxes[0].empty() || !level_owner_ranks[0].empty())
    throw std::runtime_error(
        "AmrRuntime::rebuild_hierarchy : level zero is owned by the resolved base layout");
  std::optional<RegridPhysicalGhostSupport> physical_support;
  if (n_levels > 1)
    physical_support = regrid_physical_ghost_support_();

  auto checked_refine_domain = [](const Box2D& domain, int ratio) {
    if (ratio != kAmrRefRatio)
      throw std::runtime_error(
          "AmrRuntime::rebuild_hierarchy : native AMR requires spatial refinement ratio 2");
    Box2D refined;
    for (int direction = 0; direction < 2; ++direction) {
      const std::int64_t lo = static_cast<std::int64_t>(domain.lo[direction]) * ratio;
      const std::int64_t hi = static_cast<std::int64_t>(domain.hi[direction]) * ratio + ratio - 1;
      if (lo < std::numeric_limits<int>::min() || lo > std::numeric_limits<int>::max() ||
          hi < std::numeric_limits<int>::min() || hi > std::numeric_limits<int>::max())
        throw std::runtime_error(
            "AmrRuntime::rebuild_hierarchy : refined index domain overflows native integers");
      refined.lo[direction] = static_cast<int>(lo);
      refined.hi[direction] = static_cast<int>(hi);
    }
    return refined;
  };

  Box2D parent_domain = dom_;
  for (int level = 1; level < n_levels; ++level) {
    const auto index = static_cast<std::size_t>(level);
    if (level_boxes[index].empty() ||
        level_boxes[index].size() != level_owner_ranks[index].size())
      throw std::runtime_error(
          "AmrRuntime::rebuild_hierarchy : every active fine level requires boxes and owners");
    const int ratio = maximum_refinement_ratios_[index - 1];
    const Box2D level_domain = checked_refine_domain(parent_domain, ratio);
    std::vector<Box2D> boxes;
    boxes.reserve(level_boxes[index].size());
    for (std::size_t patch = 0; patch < level_boxes[index].size(); ++patch) {
      const PatchBox& value = level_boxes[index][patch];
      const Box2D box{{value.ilo, value.jlo}, {value.ihi, value.jhi}};
      if (value.level != level || !level_domain.contains(box))
        throw std::runtime_error(
            "AmrRuntime::rebuild_hierarchy : checkpoint patch is outside its declared level");
      for (int direction = 0; direction < 2; ++direction) {
        const std::int64_t aligned_lo = static_cast<std::int64_t>(box.lo[direction]) -
                                        level_domain.lo[direction];
        const std::int64_t aligned_end = static_cast<std::int64_t>(box.hi[direction]) + 1 -
                                         level_domain.lo[direction];
        if (aligned_lo % ratio != 0 || aligned_end % ratio != 0)
          throw std::runtime_error(
              "AmrRuntime::rebuild_hierarchy : checkpoint patch is not aligned to parent cells");
      }
      for (const Box2D& prior : boxes)
        if (!prior.intersect(box).empty())
          throw std::runtime_error(
              "AmrRuntime::rebuild_hierarchy : checkpoint fine patches overlap");
      const int owner = level_owner_ranks[index][patch];
      if (owner < 0 || owner >= n_ranks())
        throw std::runtime_error(
            "AmrRuntime::rebuild_hierarchy : checkpoint owner rank is out of range");
      boxes.push_back(box);
    }
    target_boxes[index] = BoxArray(std::move(boxes));
    target_mappings[index] = DistributionMapping(level_owner_ranks[index]);
    validate_fine_layout_proper_nesting(
        target_boxes[index], target_boxes[index - 1], parent_domain, ratio, regrid_margin_,
        RegridPeriodicity{base_per_.x, base_per_.y},
        physical_support ? &*physical_support : nullptr);
    target_dx[index] = target_dx[index - 1] / Real(ratio);
    target_dy[index] = target_dy[index - 1] / Real(ratio);
    parent_domain = level_domain;
  }

  const StepSnapshot accepted = step_snapshot();
  try {
    device_fence();
    const int previous_levels = nlev_;
    if (n_levels < previous_levels)
      resize_history_levels_for_restore_(n_levels);

    hierarchy_.ba = std::move(target_boxes);
    hierarchy_.dm = std::move(target_mappings);
    hierarchy_.dx = std::move(target_dx);
    hierarchy_.dy = std::move(target_dy);
    hierarchy_.refinement_ratios.assign(
        maximum_refinement_ratios_.begin(),
        maximum_refinement_ratios_.begin() + static_cast<std::ptrdiff_t>(n_levels - 1));
    nlev_ = n_levels;
    refresh_active_temporal_plan_();

    // Checkpoint payloads overwrite every valid value. Reallocate the exact active prefix without
    // prolongation/restriction, retaining only level-zero accepted storage until its payload lands.
    for (auto& block : blocks_) {
      auto& levels = *block.levels;
      const int ncomp = levels.front().U.ncomp();
      const int ngrow = levels.front().U.n_grow();
      levels.resize(1);
      levels.reserve(static_cast<std::size_t>(n_levels));
      for (int level = 1; level < n_levels; ++level) {
        const auto index = static_cast<std::size_t>(level);
        MultiFab state(hierarchy_.ba[index], hierarchy_.dm[index], ncomp, ngrow);
        levels.push_back(
            AmrLevelMP{std::move(state), nullptr, hierarchy_.dx[index], hierarchy_.dy[index]});
      }
    }
    aux_.resize(1);
    aux_.reserve(static_cast<std::size_t>(n_levels));
    for (int level = 1; level < n_levels; ++level) {
      const auto index = static_cast<std::size_t>(level);
      aux_.emplace_back(hierarchy_.ba[index], hierarchy_.dm[index], aux_ncomp_, 1);
    }
    for (auto& block : blocks_)
      for (int level = 0; level < n_levels; ++level)
        (*block.levels)[static_cast<std::size_t>(level)].aux =
            &aux_[static_cast<std::size_t>(level)];

    // Existing rings may outlive a low-level rebuild. Reallocate every fine slot on the imposed
    // topology and append missing levels without interpolation; restore writes the authenticated
    // buffers immediately afterwards.
    for (int level = 1; level < n_levels; ++level)
      remap_history_rings_(hierarchy_.ba[static_cast<std::size_t>(level)],
                           hierarchy_.dm[static_cast<std::size_t>(level)], level, level - 1,
                           /*prolong=*/false);

    std::vector<std::vector<AmrLevelMP>> shared;
    shared.reserve(blocks_.size());
    for (const auto& block : blocks_)
      shared.push_back(*block.levels);
    detail::same_layout_or_throw(shared);

    invalidate_named_field_topology();
    record_topology_replacement_();
    materialize_boundary_sessions_();
  } catch (...) {
    restore_step_snapshot(accepted);
    throw;
  }
}

// --- regrid / clustering config setters (declared in amr_runtime.hpp) -----------------------------

inline void AmrRuntime::set_regrid(int every, int grow, int margin) {
  if (every < 0)
    throw std::runtime_error("AmrRuntime::set_regrid : regrid_every >= 0");
  regrid_every_ = every;
  regrid_grow_ = grow;
  regrid_margin_ = margin;
}

inline void AmrRuntime::set_clustering(double min_efficiency, int min_box_size, int max_box_size) {
  if (!(min_efficiency > 0.0 && min_efficiency <= 1.0))
    throw std::runtime_error("AmrRuntime::set_clustering : min_efficiency must be in (0, 1]");
  if (min_box_size < 1 || max_box_size < 1)
    throw std::runtime_error("AmrRuntime::set_clustering : box sizes must be >= 1");
  if (min_box_size > max_box_size)
    throw std::runtime_error("AmrRuntime::set_clustering : min_box_size <= max_box_size required");
  cluster_.min_efficiency = min_efficiency;
  cluster_.min_box_size = min_box_size;
  cluster_.max_box_size = max_box_size;
  if (!external_clustering_)
    clustering_provider_ = std::make_shared<const amr::BergerRigoutsosProvider>(cluster_);
}

}  // namespace pops
