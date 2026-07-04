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

// COLLECTIVE per-level extremum of U (min / max / abs_max), one component or ALL when full. Delegates
// to the proven pops:: collectives. UNMASKED: a covered coarse cell is the average of its children,
// within their [min, max], so including it never changes the global extrema.
inline double AmrRuntime::composite_extremum_(const MultiFab& U, const std::string& kind, int comp,
                                              bool full) {
  const int nc = U.ncomp();
  if (kind == "min") {
    double m = std::numeric_limits<double>::infinity();
    if (full)
      for (int c = 0; c < nc; ++c)
        m = std::min(m, static_cast<double>(reduce_min(U, c)));
    else
      m = static_cast<double>(reduce_min(U, comp));
    return m;
  }
  if (kind == "max") {
    double m = -std::numeric_limits<double>::infinity();
    if (full)
      for (int c = 0; c < nc; ++c)
        m = std::max(m, static_cast<double>(reduce_max(U, c)));
    else
      m = static_cast<double>(reduce_max(U, comp));
    return m;
  }
  // abs_max: collective max |U(.,.,c)| = all_reduce_max(norm_inf) (norm_inf is LOCAL by contract).
  double m = 0.0;
  if (full)
    for (int c = 0; c < nc; ++c)
      m = std::max(m, all_reduce_max(static_cast<double>(norm_inf(U, c))));
  else
    m = all_reduce_max(static_cast<double>(norm_inf(U, comp)));
  return m;
}

inline bool AmrRuntime::cell_covered_(int i, int j, const std::vector<Box2D>& covered) {
  for (const Box2D& c : covered)
    if (i >= c.lo[0] && i <= c.hi[0] && j >= c.lo[1] && j <= c.hi[1])
      return true;
  return false;
}

// COLLECTIVE per-level masked sum (sum / abs_sum / sum_sq) of block b at level k, one component or
// ALL when full, EXCLUDING coarse cells covered by the next finer level's patches. Raw cell sum (the
// dx*dy volume weight is applied by the caller). One all_reduce_sum.
inline double AmrRuntime::composite_level_sum_(std::size_t b, int k, const std::string& kind,
                                               int comp, bool full) const {
  const std::vector<AmrLevelMP>& L = *blocks_[b].levels;
  const MultiFab& U = L[k].U;
  const int nc = U.ncomp();
  // Covered region: the level-(k+1) patch boxes coarsened by ratio 2 into level-k index space. The
  // finer boxes come from the SHARED layout (block 0), identical for every block on this hierarchy.
  std::vector<Box2D> covered;
  if (k + 1 < static_cast<int>(L.size())) {
    for (const Box2D& fb : (*blocks_[0].levels)[k + 1].U.box_array().boxes())
      covered.push_back(Box2D{{fb.lo[0] >> 1, fb.lo[1] >> 1}, {fb.hi[0] >> 1, fb.hi[1] >> 1}});
  }
  device_fence();
  double s = 0.0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const Box2D v = U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        if (cell_covered_(i, j, covered))
          continue;
        const int c0 = full ? 0 : comp;
        const int c1 = full ? nc : comp + 1;
        for (int c = c0; c < c1; ++c) {
          const double val = u(i, j, c);
          if (kind == "sum")
            s += val;
          else if (kind == "abs_sum")
            s += val < 0 ? -val : val;
          else  // sum_sq
            s += val * val;
        }
      }
  }
  return all_reduce_sum(s);
}

inline double AmrRuntime::composite_reduce(const std::string& block, const std::string& kind,
                                           int comp) const {
  const std::size_t b = block_index_by_name_(block);
  const std::vector<AmrLevelMP>& L = *blocks_[b].levels;
  const int nlev = static_cast<int>(L.size());
  const bool full = kind.size() > 4 && kind.compare(kind.size() - 4, 4, "_all") == 0;
  const std::string base = full ? kind.substr(0, kind.size() - 4) : kind;

  if (base == "min") {
    double m = std::numeric_limits<double>::infinity();
    for (int k = 0; k < nlev; ++k)
      m = std::min(m, composite_extremum_(L[k].U, "min", comp, full));
    return m;
  }
  if (base == "max") {
    double m = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < nlev; ++k)
      m = std::max(m, composite_extremum_(L[k].U, "max", comp, full));
    return m;
  }
  if (base == "abs_max") {
    double m = 0.0;
    for (int k = 0; k < nlev; ++k)
      m = std::max(m, composite_extremum_(L[k].U, "abs_max", comp, full));
    return m;
  }
  if (base == "sum" || base == "abs_sum" || base == "sum_sq") {
    double acc = 0.0;
    for (int k = 0; k < nlev; ++k) {
      const Geometry g = level_geom(k);
      const double cell = static_cast<double>(g.dx()) * static_cast<double>(g.dy());
      acc += cell * composite_level_sum_(b, k, base, comp, full);
    }
    return acc;
  }
  throw std::runtime_error(
      "AmrRuntime::composite_reduce : unknown reduction kind '" + kind + "' for block '" + block +
      "' (expected sum / min / max / abs_sum / sum_sq / abs_max and their _all forms)");
}

inline std::vector<int> AmrRuntime::level_owner_ranks(int k) const {
  if (k < 0 || k >= nlev_)
    throw std::runtime_error("AmrRuntime::level_owner_ranks : level out of bounds");
  return (*blocks_[0].levels)[k].U.dmap().ranks();
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
  const std::size_t nf = static_cast<std::size_t>(dom_.nx()) << k;
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

// np>1 gather of level_aux_flat (all_reduce_sum of the disjoint per-rank contributions -- the AMR
// checkpoint gather pattern). COLLECTIVE: all ranks MUST call it. Mono-rank identity.
inline std::vector<double> AmrRuntime::level_aux_flat_global(int k) const {
  std::vector<double> out = level_aux_flat(k);
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
  const std::size_t nf = static_cast<std::size_t>(dom_.nx()) << k;
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
}

}  // namespace pops
