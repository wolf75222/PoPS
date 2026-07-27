#pragma once

#include <pops/core/model/coupled_system.hpp>
#include <pops/amr/hierarchy/refinement_ratio.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/coupling/amr/amr_coupler_mp.hpp>  // detail::coupler_inject_aux_mb
#include <pops/coupling/base/aux_fill.hpp>  // detail::derive_aux_bc + detail::fill_bz_box (shared)
#include <pops/coupling/base/elliptic_rhs.hpp>
#include <pops/numerics/elliptic/interface/elliptic_problem.hpp>  // field_postprocess, FieldPostProcess
#include <pops/numerics/elliptic/interface/elliptic_solver.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/time/amr/reflux/amr_reflux_mf.hpp>  // AmrLevelMP, prepared AMR transfers
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/parallel/comm.hpp>  // all_reduce_sum
#include <pops/parallel/prepared_load_balance.hpp>
#include <pops/parallel/prepared_provider_consensus.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

/// @file
/// @brief AmrSystemCoupler: multi-species SYSTEM coupler on AMR (milestone 2.3).
///
/// Carries a CoupledSystem on an AMR hierarchy: each block has ITS OWN level hierarchy, all species
/// SHARE the same AMR grid, the same aux field (phi, grad phi [, B_z, ...]) and the same coarse
/// Poisson. It prepares shared fields and hierarchy storage for spatial operators; ProgramGraph owns
/// every temporal composition. STRONG INVARIANT: all blocks live on EXACTLY the same grid per level
/// (the aux is shared); same_layout_or_throw checks this at the ctor.

namespace pops {

// EXPLICIT layout of a shared AMR hierarchy (point 2 of the multi-block capstone, first MINIMAL
// step). Single source of truth on the GRID that all blocks share: per level the BoxArray (the boxes
// AND their order), the DistributionMapping (rank per box), dx/dy, and the number of levels
// (= ba.size()). Today this information is implicit, scattered across each AmrLevelMP (U.box_array() /
// U.dmap() / dx,dy). This type only EXTRACTS it for the same_layout_or_throw guard: it does NOT
// replace EquationBlock / AmrLevelMP and introduces NO block abstraction (the wide AmrBlock of the
// design is a LATER step, and only if needed). The layout of a stack of levels is read via from_levels.
/// Single source of truth on the GRID shared by all blocks: per level the BoxArray (boxes AND order),
/// the DistributionMapping (rank per box) and dx/dy. EXTRACTED only (replaces neither EquationBlock nor
/// AmrLevelMP); used by the same_layout_or_throw guard.
struct AmrHierarchyLayout {
  std::vector<BoxArray> ba;             // [level]: boxes of the level (set AND order)
  std::vector<DistributionMapping> dm;  // [level], parallel to ba: MPI rank per box
  std::vector<Real> dx, dy;             // [level]: grid spacing (= dx_coarse / 2^k)
  std::vector<int> refinement_ratios;   // transition k -> k+1, resolved by hierarchy manifest
  std::shared_ptr<const PreparedLoadBalanceAuthority>
      load_balance;  // immutable owner selection authority for every future layout

  /// Number of levels (= ba.size()).
  int nlev() const { return static_cast<int>(ba.size()); }

  // Reads the layout carried by the level stack of ONE block (each AmrLevelMP carries
  // U.box_array() / U.dmap() / dx,dy). No copy of field data: only the grid.
  /// Extracts the layout (BoxArray + DistributionMapping + dx/dy per level) from the level stack of ONE
  /// block. No copy of field data, only the grid.
  static AmrHierarchyLayout from_levels(
      const std::vector<AmrLevelMP>& levels,
      std::shared_ptr<const PreparedLoadBalanceAuthority> load_balance) {
    if (!load_balance)
      throw std::invalid_argument(
          "AmrHierarchyLayout::from_levels requires a prepared load-balance authority");
    AmrHierarchyLayout L;
    L.load_balance = std::move(load_balance);
    const int n = static_cast<int>(levels.size());
    L.ba.reserve(n);
    L.dm.reserve(n);
    L.dx.reserve(n);
    L.dy.reserve(n);
    for (const auto& lv : levels) {
      L.ba.push_back(lv.U.box_array());
      L.dm.push_back(lv.U.dmap());
      L.dx.push_back(lv.dx);
      L.dy.push_back(lv.dy);
    }
    for (std::size_t level = 1; level < levels.size(); ++level) {
      const int ratio = static_cast<int>(levels[level - 1].dx / levels[level].dx);
      if (ratio < 2 || levels[level - 1].dx != levels[level].dx * Real(ratio) ||
          levels[level - 1].dy != levels[level].dy * Real(ratio))
        throw std::runtime_error("AmrHierarchyLayout requires exact isotropic refinement ratios");
      L.refinement_ratios.push_back(ratio);
    }
    return L;
  }
};

namespace detail {
template <class>
inline constexpr bool amr_always_false_v = false;

// EXACT comparison of the grids of two levels (point 1): same BoxArray (boxes AND order), same
// DistributionMapping (rank per box), same dx/dy (bit-for-bit). Returns true if everything matches.
// dx/dy are the level spacings, identical by construction if the boxes are; we compare them anyway to
// catch a mis-wired geometry.
inline bool same_level_layout(const BoxArray& a_ba, const DistributionMapping& a_dm, Real a_dx,
                              Real a_dy, const BoxArray& b_ba, const DistributionMapping& b_dm,
                              Real b_dx, Real b_dy) {
  return a_ba.boxes() == b_ba.boxes() && a_dm.ranks() == b_dm.ranks() && a_dx == b_dx &&
         a_dy == b_dy;
}

inline DistributionMapping amr_system_authoritative_coarse_mapping(
    const BoxArray& coarse_boxes, const std::vector<std::vector<AmrLevelMP>>& block_levels) {
  if (block_levels.empty() || block_levels.front().empty())
    throw std::invalid_argument("AmrSystemCoupler requires a coarse level");
  const MultiFab& coarse = block_levels.front().front().U;
  if (coarse.box_array().boxes() != coarse_boxes.boxes())
    throw std::invalid_argument(
        "AmrSystemCoupler coarse BoxArray disagrees with the authoritative level field");
  return coarse.dmap();
}

// LAYOUT CONSISTENCY guard between blocks (point 1 of the capstone). The aux is SHARED per level: all
// blocks MUST live on EXACTLY the same grid at each level, otherwise the rewiring
// levels[k].aux = &aux_[k] and spatial operators read an inconsistent grid (silent out-of-bound
// access).
// The old check only compared the NUMBER of boxes (.size()); here we compare EXACTLY: number of
// levels, then per level BoxArray (boxes AND order), DistributionMapping and dx/dy. Throws a clear
// error at the FIRST discrepancy (block and level located). A single block matches itself trivially ->
// single-block path strictly bit-identical (the loop over the other blocks is empty).
inline void same_layout_or_throw(const std::vector<std::vector<AmrLevelMP>>& block_levels) {
  if (block_levels.empty())
    return;
  const auto& ref = block_levels[0];
  const int nlev = static_cast<int>(ref.size());
  for (std::size_t b = 1; b < block_levels.size(); ++b) {
    const auto& cur = block_levels[b];
    if (static_cast<int>(cur.size()) != nlev)
      throw std::runtime_error(
          "AmrSystemCoupler: all blocks must have the same number of levels "
          "(shared AMR layout)");
    for (int k = 0; k < nlev; ++k) {
      if (!same_level_layout(cur[k].U.box_array(), cur[k].U.dmap(), cur[k].dx, cur[k].dy,
                             ref[k].U.box_array(), ref[k].U.dmap(), ref[k].dx, ref[k].dy))
        throw std::runtime_error(
            "AmrSystemCoupler: inconsistent AMR layout between blocks (the shared aux requires the "
            "SAME BoxArray [boxes and order], the SAME DistributionMapping and the SAME dx/dy per "
            "level)");
    }
  }
}
}  // namespace detail

/// Multi-species system coupler on AMR. @tparam System: CoupledSystem (blocks/species).
/// @tparam RhsAssembler: assembler of the Poisson RHS (f = Sum_s q_s n_s, e.g. ChargeDensityRhs).
/// @tparam Elliptic: elliptic backend (EllipticSolver concept, default GeometricMG). PRECONDITION:
/// all blocks share EXACTLY the same AMR layout per level (checked at the ctor).
template <CoupledSystemLike System, class RhsAssembler, class Elliptic = GeometricMG>
class AmrSystemCoupler {
  static_assert(EllipticSolver<Elliptic>, "the elliptic backend must model EllipticSolver");

 public:
  // block_levels[b] = hierarchy of block b (level 0 = coarse on ba_coarse, levels
  // > 0 = fine patches). The AmrLevelMP carry U + dx/dy per level; their aux pointer
  // is (re)wired here to the SHARED aux. The ctor also re-points block.state to the
  // coarse level of its hierarchy, so that the system RHS (ChargeDensityRhs) reads
  // the coarse densities correctly.
  // bz: out-of-plane magnetic field B_z(x, y) provided by the user (constant or field),
  // shared by ALL blocks. Set on the B_z component (index kAuxBaseComps) of the SHARED aux
  // channel of EACH level, from the cell centers OF THAT LEVEL (each level has its own
  // geometry / dx). AMR analog of the bz_ of SystemAssembler (non-AMR path). A block that
  // reads B_z (n_aux=4) sees it at all levels, a base block (3) ignores the component. Without
  // a block with an extra field (width 3) or if bz is empty: no-op -> bit-identical to history.
  template <class FactoryT = DefaultEllipticFactory<Elliptic>>
    requires pops::EllipticFactory<FactoryT, Elliptic>
  AmrSystemCoupler(System system, const Geometry& geom, const BoxArray& ba_coarse,
                   const BCRec& bcPhi, RhsAssembler rhs_assembler,
                   std::vector<std::vector<AmrLevelMP>> block_levels,
                   Periodicity base_per = Periodicity{true, true}, bool replicated_coarse = true,
                   ActiveRegionProvider2D active = {}, ScalarFieldProvider2D bz = {},
                   FactoryT elliptic_factory = {})
      : system_(std::move(system)),
        rhs_assembler_(std::move(rhs_assembler)),
        geom_(geom),
        dom_(geom.domain),
        base_per_(base_per),
        bcPhi_(bcPhi),
        aux_bc_(detail::derive_aux_bc(bcPhi)),
        replicated_coarse_(replicated_coarse),
        coarse_mapping_(detail::amr_system_authoritative_coarse_mapping(ba_coarse, block_levels)),
        mg_(make_elliptic_solver<Elliptic>(
            {geom_, ba_coarse, coarse_mapping_, bcPhi_, std::move(active),
             replicated_coarse ? FieldDistribution::Replicated : FieldDistribution::Distributed},
            std::move(elliptic_factory))),
        block_levels_(std::move(block_levels)),
        bz_(std::move(bz)) {
    require_prepared_provider_collective_consensus(bz_);
    // Construction checks (Codex review): without them, a malformed hierarchy
    // causes a silent out-of-bound access in the wiring / spatial evaluations.
    if (block_levels_.size() != System::n_blocks)
      throw std::runtime_error(
          "AmrSystemCoupler: block_levels must have one level vector per block "
          "(size != n_blocks)");
    nlev_ = block_levels_.empty() ? 0 : static_cast<int>(block_levels_[0].size());
    if (nlev_ == 0)
      throw std::runtime_error("AmrSystemCoupler: at least one level (coarse) required");
    // EXACT layout consistency between blocks (the aux is shared per level): same number of
    // levels, and per level same BoxArray (boxes AND order), same DistributionMapping, same
    // dx/dy. Replaces the old check that only compared the NUMBER of boxes (.size()).
    // Single-block: the check is trivial (a single block) -> bit-identical to history.
    detail::same_layout_or_throw(block_levels_);
    // SHARED aux: one MultiFab (phi, grad phi [, B_z, ...]) per level, on the common grid.
    // Sized once -> stable addresses for the blocks' aux pointers. Width =
    // max of aux_comps<Model> over the blocks (at least 3): a block reading B_z (n_aux > 3) has
    // the room at EACH level, a base block ignores the extra components. Without a block with an
    // extra field -> width 3 -> allocation strictly bit-identical to history.
    aux_ncomp_ = system_aux_comps(system_);
    aux_.resize(nlev_);
    for (int k = 0; k < nlev_; ++k)
      aux_[k] =
          MultiFab(block_levels_[0][k].U.box_array(), block_levels_[0][k].U.dmap(), aux_ncomp_, 1);
    for (auto& levels : block_levels_)
      for (int k = 0; k < nlev_; ++k)
        levels[k].aux = &aux_[k];

    // re-point each block to ITS coarse level (block.U() = coarse of the block).
    std::size_t b = 0;
    system_.for_each_block([&](auto& block) {
      block.state = &block_levels_[b][0].U;
      ++b;
    });

    prepare_aux_transfer_workspaces_();
    fill_bz();  // populates B_z per level (no-op if no block requests it or if bz is empty)
  }

  // Collective setter (parity with the collective ctor): all ranks must call it with an identical
  // prepared-provider contract. Immediately re-populates the aux channel of each level. Effective
  // no-op if the aux width <= base.
  void set_bz(ScalarFieldProvider2D bz) {
    require_prepared_provider_collective_consensus(bz);
    bz_ = std::move(bz);
    fill_bz();
  }

  System& system() { return system_; }
  const System& system() const { return system_; }
  MultiFab& phi() { return mg_.phi(); }
  int nlev() const { return nlev_; }
  const MultiFab& aux(int k) const { return aux_[k]; }
  // WRITE access to the shared aux channel of level k (parity with SystemAssembler::aux()):
  // allows populating an extra component (B_z, ...) that field_postprocess does not touch
  // (it only writes phi/grad, comp 0..2). The width is aux_ncomp_ (max aux_comps of the blocks).
  MultiFab& aux(int k) { return aux_[k]; }
  int aux_ncomp() const { return aux_ncomp_; }
  std::vector<AmrLevelMP>& levels(std::size_t b) { return block_levels_[b]; }
  MultiFab& coarse(std::size_t b) { return block_levels_[b][0].U; }
  const MultiFab& coarse(std::size_t b) const { return block_levels_[b][0].U; }
  // Number of Poisson solves since construction.
  int solve_count() const { return solve_count_; }

  // sync_down (per block) + coarse system Poisson + coarse aux + fine injection.
  /// Solves the fields: average_down per block, coarse system Poisson (RHS = Sum_s q_s n_s),
  /// coarse aux (phi, grad phi) then injection to the fine levels + re-sets B_z per level. Increments
  /// solve_count().
  void solve_fields() {
    ++solve_count_;
    for (std::size_t block = 0; block < block_levels_.size(); ++block) {
      auto& levels = block_levels_[block];
      auto& plan = average_down_plans_.at(block);
      if (!plan)
        throw std::logic_error("AMR system average-down plan was not prepared");
      for (int k = nlev_ - 1; k >= 1; --k)
        mf_average_down_mb(levels[k].U, levels[k - 1].U, plan->transition_for_child(k),
                           plan->topology_generation(), world_communicator_view());
    }

    rhs_assembler_(system_, mg_.rhs());  // f = Sum_s q_s n_s on the coarse level
    mg_.solve();

    // coarse aux = (phi, grad phi) via the SAME clean path as the single-level
    // SystemCoupler (Codex review 9.4): fill the ghosts of phi according to bcPhi_, then
    // field_postprocess, then fill the ghosts of aux according to aux_bc_ (derived from bcPhi_).
    // Handles the non-periodic case (Foextrap) instead of a hard-coded periodic fill_boundary.
    fill_ghosts(mg_.phi(), dom_, bcPhi_);
    const Real cx = Real(1) / (2 * geom_.dx()), cy = Real(1) / (2 * geom_.dy());
    field_postprocess(mg_.phi(), aux_[0], cx, cy,
                      FieldPostProcess{FieldPostProcess::GradSign::Plus, true});
    fill_ghosts(aux_[0], dom_, aux_bc_);
    for (int k = 1; k < nlev_; ++k) {
      auto& workspace = aux_transfer_workspaces_.at(static_cast<std::size_t>(k - 1));
      if (!workspace)
        throw std::logic_error("AMR system aux transfer workspace was not prepared");
      const bool replicated_parent = k == 1 && replicated_coarse_;
      const CommunicatorView communicator =
          replicated_parent ? CommunicatorView{} : world_communicator_view();
      workspace->apply(aux_[k - 1], aux_[k], transfer_topology_generation_, communicator);
    }

    // B_z PER LEVEL (not just propagated): coupler_inject_aux_mb copies ALL the components
    // of the parent (including B_z) to the fine levels, which would overwrite the fine B_z with a
    // coarse B_z injected (constant per coarse cell). So we re-set B_z from bz_ at the FINE centers
    // after the injection, so that a spatially varying B_z is sampled at the level resolution.
    // Static and cheap; no-op if the aux width <= base or bz empty (constant B_z: this re-fill is
    // idempotent, the injection would have sufficed).
    fill_bz();
  }

  // mass of component 0 of the coarse level of block b (sum u*dV over local fabs;
  // replicated coarse -> local sum = total, otherwise all_reduce).
  Real mass(std::size_t b) const {
    const MultiFab& U = block_levels_[b][0].U;
    const Real dV = geom_.dx() * geom_.dy();
    Real M = 0;
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 u = U.fab(li).const_array();
      M += for_each_cell_reduce_sum(U.box(li),
                                    [u, dV] POPS_HD(int i, int j) { return u(i, j, 0) * dV; });
    }
    return replicated_coarse_ ? M : all_reduce_sum(M);
  }

 private:
  void prepare_aux_transfer_workspaces_() {
    std::vector<std::optional<detail::PreparedConservativeLinearTransferWorkspace>> prepared(
        static_cast<std::size_t>(std::max(0, nlev_ - 1)));
    const Periodicity periodicity{
        aux_bc_.xlo == BCType::Periodic && aux_bc_.xhi == BCType::Periodic,
        aux_bc_.ylo == BCType::Periodic && aux_bc_.yhi == BCType::Periodic};
    for (int level = 1; level < nlev_; ++level) {
      const bool replicated_parent = level == 1 && replicated_coarse_;
      const CommunicatorView communicator =
          replicated_parent ? CommunicatorView{} : world_communicator_view();
      prepared[static_cast<std::size_t>(level - 1)].emplace(
          detail::PreparedConservativeLinearTransferWorkspace::prepare(
              aux_[level - 1], aux_[level], amr_level_index_domain(dom_, level - 1),
              amr_level_index_domain(dom_, level), replicated_parent,
              detail::ConservativeCellFillRegion::ValidAndGhost, periodicity,
              transfer_topology_generation_, communicator));
    }
    std::vector<std::optional<PreparedAmrAverageDownPlan>> average_down(block_levels_.size());
    for (std::size_t block = 0; block < block_levels_.size(); ++block)
      average_down[block].emplace(
          PreparedAmrAverageDownPlan::prepare(block_levels_[block], transfer_topology_generation_));
    aux_transfer_workspaces_.swap(prepared);
    average_down_plans_.swap(average_down);
  }

  System system_;
  RhsAssembler rhs_assembler_;
  // Width of the SHARED aux channel: maximum of aux_comps<Model> over all the blocks (at least
  // kAuxBaseComps). The shared channel per level must be at least as wide as the most demanding
  // block so that load_aux<aux_comps<Model>> never reads out of bound in AMR spatial operators; a
  // less demanding block simply ignores the extra
  // components. Exact analog of SystemAssembler::system_aux_comps (non-AMR path). Without a
  // block with an extra field, the width stays 3 -> allocation strictly bit-identical to history.
  static int system_aux_comps(const System& sys) {
    int w = kAuxBaseComps;
    sys.for_each_block([&](const auto& b) {
      using Model = std::decay_t<decltype(b.model)>;
      const int c = aux_comps<Model>();
      if (c > w)
        w = c;
    });
    return w;
  }
  // Populates the aux B_z component (index kAuxBaseComps) of the shared channel of EACH level from
  // bz_(x, y). B_z is static (external to the elliptic): set once (at the ctor / set_bz),
  // preserved by solve_fields (field_postprocess only writes phi/grad, comp 0..2; we re-set
  // after the coarse->fine injection which would copy a coarse B_z) and by Program execution
  // (spatial state updates do not touch aux). Each level has ITS geometry: level k = geom_.refine(1 << k),
  // same physical extents but refined index domain, so x_cell/y_cell point to the physical
  // center of the FINE cell. We fill the GROWN box (valid + halos) directly from
  // bz_(x, y): bz_ being a pure function of the physical position, its evaluation at the ghost
  // centers gives the physically correct B_z there too (independent of the BC of the fine patch,
  // without periodicity ambiguity on a patch domain). No-op if the aux width <= kAuxBaseComps (no
  // block reads B_z) or if bz_ is empty: RUNTIME guard (the width is only known at construction) ->
  // base model strictly bit-identical to history.
  void fill_bz() {
    if (!bz_ || aux_ncomp_ <= kAuxBaseComps)
      return;
    for (int k = 0; k < nlev_; ++k) {
      const Geometry gk = geom_.refine(1 << k);  // geometry of level k (dx = dx_coarse / 2^k)
      MultiFab& A = aux_[k];
      for (int li = 0; li < A.local_size(); ++li) {
        Fab2D& f = A.fab(li);
        // grown box (valid + halos): B_z(x,y) correct everywhere, geometry of level k.
        detail::fill_bz_box(f, f.grown_box(), gk, bz_);
      }
    }
  }

  Geometry geom_;
  Box2D dom_;
  Periodicity base_per_;
  BCRec bcPhi_, aux_bc_;
  bool replicated_coarse_;
  mutable int solve_count_ = 0;
  DistributionMapping coarse_mapping_;
  Elliptic mg_;
  std::vector<std::vector<AmrLevelMP>> block_levels_;  // [block][level]
  std::vector<MultiFab> aux_;                          // [level], shared
  std::uint64_t transfer_topology_generation_ = 1;
  std::vector<std::optional<detail::PreparedConservativeLinearTransferWorkspace>>
      aux_transfer_workspaces_;
  std::vector<std::optional<PreparedAmrAverageDownPlan>> average_down_plans_;
  int aux_ncomp_ =
      kAuxBaseComps;  // width of the shared aux channel (max aux_comps over the blocks)
  int nlev_ = 0;
  ScalarFieldProvider2D bz_;  // prepared external B_z(x, y) (empty if not provided)
};

}  // namespace pops
