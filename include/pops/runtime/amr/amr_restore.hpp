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
    const AmrLevelMP& shared =
        blocks_.front().levels->at(static_cast<std::size_t>(level));
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

// FULL shared aux of level k: ALL aux_ncomp_ components of aux_[k], LOCAL valid cells at GLOBAL
// component-major flat indices c*nf*nf + j*nf + i (zeros outside the patches at a fine level) -- the
// exact layout of block_level_state, so the v3 checkpoint reader/writer share one convention. phi
// (comp 0) is included; the level-0 multigrid WARM START stays a separate phi_<k> payload
// (level_potential), which reads mg_.phi(), not aux_[0].
inline std::vector<double> AmrRuntime::level_aux_flat(int k) const {
  if (k < 0 || k >= nlev_)
    throw std::runtime_error("AmrRuntime::level_aux_flat : level out of bounds");
  const MultiFab& A = aux_[k];
  const int nc = A.ncomp();
  const std::size_t nf = static_cast<std::size_t>(dom_.nx()) * level_refinement(k);
  std::vector<double> out(static_cast<std::size_t>(nc) * nf * nf, 0.0);
  device_fence();
  for (int li = 0; li < A.local_size(); ++li) {
    const ConstArray4 a = A.fab(li).const_array();
    const Box2D v = A.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        for (int c = 0; c < nc; ++c)
          out[static_cast<std::size_t>(c) * nf * nf + static_cast<std::size_t>(j) * nf +
              static_cast<std::size_t>(i)] = a(i, j, c);
  }
  return out;
}

// Global form of level_aux_flat. Ownership-distributed levels are gathered from their disjoint
// per-rank contributions; replicated level 0 is already complete on every rank and must not be
// reduced (which would multiply the checkpoint payload by n_ranks).
inline std::vector<double> AmrRuntime::level_aux_flat_global(int k) const {
  std::vector<double> out = level_aux_flat(k);
  if (k > 0 || !replicated_coarse_)
    all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
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
  const std::size_t nf = static_cast<std::size_t>(dom_.nx()) * level_refinement(k);
  if (v.size() != static_cast<std::size_t>(nc) * nf * nf)
    throw std::runtime_error("AmrRuntime::set_level_aux_flat : aux size != ncomp*nf*nf");
  device_fence();
  for (int li = 0; li < A.local_size(); ++li) {
    Array4 a = A.fab(li).array();
    const Box2D b = A.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        for (int c = 0; c < nc; ++c)
          a(i, j, c) = v[static_cast<std::size_t>(c) * nf * nf + static_cast<std::size_t>(j) * nf +
                         static_cast<std::size_t>(i)];
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
  // The level count cannot exceed what the composition allocated (nlev_ set at build from the config
  // / refinement). A checkpoint replayed against a DIFFERENT max-levels composition is refused verbatim.
  if (n_levels > nlev_)
    throw std::runtime_error(
        "AmrRuntime::rebuild_hierarchy : checkpoint has " + std::to_string(n_levels) +
        " levels but the replayed composition allocates " + std::to_string(nlev_) +
        " (a mismatched max-levels replay); replay the SAME composition before restart");

  // For each FINE level k >= 1: build BoxArray fb_k + DistributionMapping dmap_k from the manifest and
  // reallocate every block's level MultiFab on it (inherited ghost width, the R6 rule), then rebuild
  // the shared aux and rewire each block's aux pointer (the R7 steps). No prolong: the per-level state
  // restore overwrites every valid cell afterwards.
  for (int k = 1; k < n_levels; ++k) {
    std::vector<Box2D> boxes;
    boxes.reserve(level_boxes[k].size());
    for (const PatchBox& pb : level_boxes[k])
      boxes.push_back(Box2D{{pb.ilo, pb.jlo}, {pb.ihi, pb.jhi}});
    BoxArray fb(boxes);
    DistributionMapping dmap(level_owner_ranks[k]);
    // The hierarchy is the unique topology/ownership authority used by patch inspection, composite
    // masks, transfer contexts and the next regrid. Reallocating only the block MultiFabs would leave
    // those consumers on the seed layout even though checkpoint data was written onto the saved one.
    hierarchy_.ba[static_cast<std::size_t>(k)] = fb;
    hierarchy_.dm[static_cast<std::size_t>(k)] = dmap;

    // (R6) reallocate each block's level-k MultiFab on (fb, dmap) with its INHERITED ghost width.
    for (auto& b : blocks_) {
      auto& L = *b.levels;
      const int ngf = L[k].U.n_grow();
      L[k].U = MultiFab(fb, dmap, L[k].U.ncomp(), ngf);
    }
    // (R7) rebuild the shared aux on (fb, dmap) and rewire each block's aux pointer.
    aux_[k] = MultiFab(fb, dmap, aux_ncomp_, 1);
    for (auto& b : blocks_)
      (*b.levels)[k].aux = &aux_[k];

    // (R7b, ADC-631) reallocate every history ring's level-k slot on the imposed (fb, dmap) WITHOUT
    // interpolation: rebuild_hierarchy imposes both layout AND data (the per-slot restore overwrites
    // every valid cell afterwards). At a v3 restart the rings are registered lazily AFTER this, so
    // this is a no-op there; it holds the invariant when rings already exist at a rebuild.
    remap_history_rings_(fb, dmap, k, k - 1, /*prolong=*/false);
  }

  // (V3) shared-layout invariant: every block on the SAME (boxes, order, rank) after the rebuild.
  {
    std::vector<std::vector<AmrLevelMP>> ref;
    ref.reserve(blocks_.size());
    for (const auto& b : blocks_)
      ref.push_back(*b.levels);
    detail::same_layout_or_throw(ref);
  }
  for (auto& item : named_fields_) {
    item.second.mg.reset();
    item.second.level_mg.clear();
    item.second.fac.reset();
    item.second.nullspace = {};
    item.second.level_nullspace.clear();
    item.second.nullspace_ready = false;
  }
  record_topology_replacement_();
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
