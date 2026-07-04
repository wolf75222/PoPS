#pragma once

#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>

#include <pops/mesh/boundary/fill_boundary.hpp>  // fill_boundary (intra-level fine-fine + MPI adjacency)
#include <pops/parallel/comm.hpp>                // all_reduce_max (collective residual norm)

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

/// @file
/// @brief ADC-636 general FAC: the N-level / adjacent-patch / MPI composite elliptic path of
///        CompositeFacPoisson. Tail-included by composite_fac_poisson.hpp; every definition here is an
///        out-of-line member so the class API (declared in the main header) is untouched.
///
/// The 2-level non-adjacent mono-rank envelope is dispatched to the VERBATIM legacy body
/// (solve_two_level_legacy_) and never reaches this file. This is the general path taken by N > 2
/// levels, adjacent fine patches, or n_ranks() > 1.
///
/// ALGORITHM (McCormick multiplicative FAC / block Gauss-Seidel over the level chain). The base
/// multigrid (mg_) annihilates the smooth coarse error; per-level red-black SOR annihilates the local
/// high-frequency error; a conservative coarse-fine FLUX CORRECTION at every interface couples the
/// levels two-way. Order 2 is a property of the composite discretization at convergence (order-2
/// bilerp C/F Dirichlet + conservative fine-flux correction at every interface + average-down
/// consistency), not of the iteration. At L == 2 the driver reduces algebraically to the legacy loop
/// (mg_-correct(0) -> relax_level(1) == refresh_fine -> composite_residual(0) ==
/// composite_coarse_residual), the cross-check documented in test_composite_fac_nlevel.

namespace pops {

// ------------------------------------------------------------------------------------------------
// build_extra_levels_ : allocate levels k >= 2 + the uniform per-level coverage/footprint metadata
// (index 0..L-1) and the intermediate-level correction multigrids.
// ------------------------------------------------------------------------------------------------
inline void CompositeFacPoisson::build_extra_levels_(const std::vector<BoxArray>& level_boxes) {
  const int L = n_levels_;  // = 1 + level_boxes.size()
  for (int k = 2; k < L; ++k) {
    const Geometry gk = geom_c_.refine(1 << k);
    const BoxArray& bak = level_boxes[k - 1];  // level_boxes[k-1] is the level-k tiling
    // CO-LOCATION (ADC-636 MPI): place each level-k patch on the rank owning its PARENT (level k-1)
    // patch, so the C/F bilerp reads a LOCAL parent fab (no inter-rank read inside the bilerp). The
    // parent of child g is the level-(k-1) patch whose footprint contains the child's grown footprint
    // (validate_nesting_ guarantees a single parent). Level 1 reads the replicated coarse (local
    // everywhere), so only k >= 2 needs this.
    const BoxArray& parent_ba = (k == 2) ? ba_f_ : ba_lv_[(k - 1) - 2];
    const DistributionMapping& parent_dm = (k == 2) ? dm_f_ : dm_lv_[(k - 1) - 2];
    std::vector<int> ranks(bak.size(), 0);
    for (int g = 0; g < bak.size(); ++g) {
      const Box2D foot = PatchRange(bak[g]).box();  // child footprint on level k-1
      int owner = 0;
      for (int p = 0; p < parent_ba.size(); ++p)
        if (parent_ba[p].contains(foot)) {
          owner = parent_dm.ranks()[p];
          break;
        }
      ranks[g] = owner;
    }
    DistributionMapping dmk(std::move(ranks));
    geom_lv_.push_back(gk);
    ba_lv_.push_back(bak);
    dm_lv_.push_back(dmk);
    MultiFab phik(bak, dmk, 1, 1), fk(bak, dmk, 1, 0), rk(bak, dmk, 1, 0);
    MultiFab epk(bak, dmk, 1, 1), axk(bak, dmk, 1, 1), ayk(bak, dmk, 1, 1);
    phik.set_val(Real(0));
    fk.set_val(Real(0));
    rk.set_val(Real(0));
    epk.set_val(Real(1));
    axk.set_val(Real(0));
    ayk.set_val(Real(0));
    phi_lv_.push_back(std::move(phik));
    f_lv_.push_back(std::move(fk));
    res_lv_.push_back(std::move(rk));
    eps_lv_.push_back(std::move(epk));
    axy_lv_.push_back(std::move(axk));
    ayx_lv_.push_back(std::move(ayk));
  }
  // Level-1 residual buffer: needed as a correction source only when level 1 is intermediate (L > 2).
  if (L > 2)
    res_f_ = MultiFab(ba_f_, dm_f_, 1, 0);
  finalize_hierarchy_metadata_();
}

// finalize_hierarchy_metadata_ : the uniform per-level coverage cov_of_[m] (level m covered by level
// m+1), the parent footprints foot_of_[m][g], and the intermediate-level correction multigrids
// level_mg_[m] (1 <= m <= L-2). Built from the current hierarchy, so a 2-level input reaches the
// general path with valid metadata (the cross-check test hook, and later MPI).
inline void CompositeFacPoisson::finalize_hierarchy_metadata_() {
  const int L = n_levels_;
  cov_of_.clear();
  foot_of_.assign(L, {});
  for (int m = 0; m < L; ++m) {
    const Geometry& gm = geom_level(m);
    CoverageMask cm(Box2D::from_extents(gm.domain.nx(), gm.domain.ny()));
    if (m + 1 < L) {
      const BoxArray& child = (m + 1 == 1) ? ba_f_ : ba_lv_[(m + 1) - 2];
      for (int g = 0; g < child.size(); ++g)
        cm.mark(PatchRange(child[g]).box());
    }
    cov_of_.push_back(std::move(cm));
    if (m >= 1) {
      const BoxArray& bam = (m == 1) ? ba_f_ : ba_lv_[m - 2];
      for (int g = 0; g < bam.size(); ++g)
        foot_of_[m].push_back(PatchRange(bam[g]).box());
    }
  }
  // Intermediate-level correction multigrids. The patch is solved with HOMOGENEOUS DIRICHLET at its
  // own edges (the C/F correction e_m = 0 at the interface), so the mg domain must be the PATCH box
  // (not the whole refined grid) -- otherwise the patch edges would be seen as interior and the mg
  // would mis-handle the boundary. Built per patch level for a single-patch level (the common nested
  // tower); a multi-patch intermediate level uses the patch-union BoxArray on the refined geometry.
  level_mg_.clear();
  level_mg_.resize(L);
  BCRec dir;  // homogeneous Dirichlet on every side (the correction vanishes at the C/F interface)
  dir.xlo = dir.xhi = dir.ylo = dir.yhi = BCType::Dirichlet;
  for (int m = 1; m + 1 < L; ++m) {
    const Geometry& gm = geom_level(m);
    const BoxArray& bam = (m == 1) ? ba_f_ : ba_lv_[m - 2];
    if (bam.size() == 1) {
      // single patch: domain = the patch box, physical extent = the patch region on level m.
      const Box2D pb = bam[0];
      Geometry gp{pb, gm.xlo + pb.lo[0] * gm.dx(), gm.xlo + (pb.hi[0] + 1) * gm.dx(),
                  gm.ylo + pb.lo[1] * gm.dy(), gm.ylo + (pb.hi[1] + 1) * gm.dy()};
      level_mg_[m] = std::make_unique<GeometricMG>(gp, bam, dir, std::function<bool(Real, Real)>{},
                                                   /*replicated=*/true);
    } else {
      level_mg_[m] = std::make_unique<GeometricMG>(gm, bam, dir, std::function<bool(Real, Real)>{},
                                                   /*replicated=*/false);
    }
  }
}

// dst <- src, component 0, valid cells (== the legacy copy0).
inline void CompositeFacPoisson::copy0_(MultiFab& dst, const MultiFab& src) {
  device_fence();
  for (int li = 0; li < dst.local_size(); ++li) {
    Array4 d = dst.fab(li).array();
    const ConstArray4 s = src.fab(li).const_array();
    const Box2D b = dst.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        d(i, j, 0) = s(i, j, 0);
  }
}

// phi += e on the level-m cells NOT covered by level m+1 (== the legacy add_uncovered, per level).
inline void CompositeFacPoisson::add_uncovered_level_(int m, MultiFab& phi, const MultiFab& e) {
  device_fence();
  const CoverageMask& cov = cov_of_[m];
  for (int li = 0; li < phi.local_size(); ++li) {
    Array4 p = phi.fab(li).array();
    const ConstArray4 ec = e.fab(li).const_array();
    const Box2D b = phi.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        if (!cov.covered(i, j))
          p(i, j, 0) += ec(i, j, 0);
  }
}

// ------------------------------------------------------------------------------------------------
// setup_level_coeffs_ : legacy coefficient setup (:189-215) per level. eps ghosts on the coarse are
// Foextrap; every patch level inherits its coefficient ghosts by bilerp of its parent.
// ------------------------------------------------------------------------------------------------
inline void CompositeFacPoisson::setup_level_coeffs_() {
  const int L = n_levels_;
  if (has_eps_) {
    device_fence();
    fill_ghosts(eps_c_, geom_c_.domain, coeff_bc(bc_));
    for (int k = 1; k < L; ++k)
      fill_cf_field_(k, eps_level(k), eps_level(k - 1));
    mg_.set_epsilon(eps_c_);
  }
  if (has_cross_) {
    device_fence();
    fill_ghosts(axy_c_, geom_c_.domain, coeff_bc(bc_));
    fill_ghosts(ayx_c_, geom_c_.domain, coeff_bc(bc_));
    for (int k = 1; k < L; ++k) {
      fill_cf_field_(k, a_xy_level(k), a_xy_level(k - 1));
      fill_cf_field_(k, a_yx_level(k), a_yx_level(k - 1));
    }
    mg_.set_cross_terms(axy_c_, ayx_c_);
  }
}

namespace detail {
// Find the local parent fab whose GROWN box covers the coarsened footprint the child's ghost ring
// reads. Level 0 (parent) is mono-box replicated -> fab(0) on every rank. For k >= 2 the child is
// CO-LOCATED with its parent (build_extra_levels_), so the covering parent fab is LOCAL. Returns a
// null array (ptr == nullptr) if no local parent covers it (a rank that owns neither -> nothing to do).
inline ConstArray4 parent_array_for_(const MultiFab& parent, const Box2D& child_valid, int r,
                                     int ng) {
  // the ghost ring reads coarse cells around the coarsened child box; grow by 1 coarse cell for the
  // bilerp stencil, on the coarsened (child-grown) footprint.
  const Box2D need = Box2D{{(child_valid.lo[0] - ng) / r - 1, (child_valid.lo[1] - ng) / r - 1},
                           {(child_valid.hi[0] + ng) / r + 1, (child_valid.hi[1] + ng) / r + 1}};
  for (int li = 0; li < parent.local_size(); ++li)
    if (parent.fab(li).grown_box().contains(need))
      return parent.fab(li).const_array();
  return ConstArray4{};
}
}  // namespace detail

// fill_cf_field_ : ghosts of a level-k COEFFICIENT field by bilerp of the parent (level k-1). Generic
// (eps, a_xy, a_yx); reads the CO-LOCATED local parent fab (mono-box replicated at level 0).
inline void CompositeFacPoisson::fill_cf_field_(int /*k*/, MultiFab& fine, const MultiFab& parent) {
  device_fence();
  const int ng = fine.n_grow();
  for (int li = 0; li < fine.local_size(); ++li) {
    Array4 F = fine.fab(li).array();
    const Box2D vb = fine.box(li);
    const ConstArray4 C = detail::parent_array_for_(parent, vb, ratio_, ng);
    if (C.p == nullptr)
      continue;  // no local parent (should not happen under co-location); skip
    for (int j = vb.lo[1] - ng; j <= vb.hi[1] + ng; ++j)
      for (int i = vb.lo[0] - ng; i <= vb.hi[0] + ng; ++i) {
        const bool inside = (i >= vb.lo[0] && i <= vb.hi[0] && j >= vb.lo[1] && j <= vb.hi[1]);
        if (inside)
          continue;
        F(i, j, 0) = detail::fac_bilerp_coarse(C, i, j, ratio_);
      }
  }
}

// fill_cf_phi_ : the canonical ghost-fill order (design 3b) for phi_level(k), k >= 1:
//   1. parent phi ghosts already valid (physical BC on level 0 done by the caller);
//   2. fill_cf(phi_k <- phi_{k-1}) : order-2 bilerp into every ghost of every level-k patch;
//   3. fill_boundary(phi_k) : overwrite the ghosts overlapping a sibling patch's valid cells.
// For non-adjacent patches step 3 is a no-op -> identical to the legacy fill_cf_ghosts alone.
inline void CompositeFacPoisson::fill_cf_phi_(int k) {
  device_fence();
  MultiFab& phik = phi_level(k);
  const MultiFab& parent = phi_level(k - 1);
  const int ng = phik.n_grow();
  for (int li = 0; li < phik.local_size(); ++li) {
    Array4 F = phik.fab(li).array();
    const Box2D vb = phik.box(li);
    const ConstArray4 C = detail::parent_array_for_(parent, vb, ratio_, ng);  // co-located parent fab
    if (C.p == nullptr)
      continue;
    for (int j = vb.lo[1] - ng; j <= vb.hi[1] + ng; ++j)
      for (int i = vb.lo[0] - ng; i <= vb.hi[0] + ng; ++i) {
        const bool inside = (i >= vb.lo[0] && i <= vb.hi[0] && j >= vb.lo[1] && j <= vb.hi[1]);
        if (inside)
          continue;
        F(i, j, 0) = detail::fac_bilerp_coarse(C, i, j, ratio_);
      }
  }
  device_fence();
  fill_boundary(phik, geom_level(k).domain);  // step 3: fine-fine sibling exchange (adjacency + MPI)
}

// fine_sor_level_ : red-black SOR on level m with FROZEN ghosts, over cells NOT covered by level m+1
// (the finest level has no coverage). Variable-coeff face harmonic + explicit cross_div. Body is the
// legacy fine_sor per level, with a supplied effective RHS.
inline void CompositeFacPoisson::fine_sor_level_(int m, const MultiFab& f_eff, int sweeps) {
  MultiFab& phim = phi_level(m);
  const Geometry& gm = geom_level(m);
  const Real idx2 = Real(1) / (gm.dx() * gm.dx());
  const Real idy2 = Real(1) / (gm.dy() * gm.dy());
  const Real idx = Real(1) / gm.dx(), idy = Real(1) / gm.dy();
  const bool he = has_eps_, hc = has_cross_;
  const CoverageMask& cov = cov_of_[m];
  for (int li = 0; li < phim.local_size(); ++li) {
    const Box2D vb = phim.box(li);
    const Real omega = sor_omega_(vb);
    Array4 P = phim.fab(li).array();
    const ConstArray4 Pc = phim.fab(li).const_array();
    const ConstArray4 F = f_eff.fab(li).const_array();
    const ConstArray4 E = eps_level(m).fab(li).const_array();
    const ConstArray4 AXY = a_xy_level(m).fab(li).const_array();
    const ConstArray4 AYX = a_yx_level(m).fab(li).const_array();
    for (int s = 0; s < sweeps; ++s)
      for (int color = 0; color < 2; ++color)
        for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
          for (int i = vb.lo[0]; i <= vb.hi[0]; ++i) {
            if (((i + j) & 1) != color)
              continue;
            if (cov.covered(i, j))
              continue;  // covered by a finer patch: value from average-down, not relaxed
            const Real exm = he ? eps_harmonic(E(i, j, 0), E(i - 1, j, 0)) : Real(1);
            const Real exp = he ? eps_harmonic(E(i, j, 0), E(i + 1, j, 0)) : Real(1);
            const Real eym = he ? eps_harmonic(E(i, j, 0), E(i, j - 1, 0)) : Real(1);
            const Real eyp = he ? eps_harmonic(E(i, j, 0), E(i, j + 1, 0)) : Real(1);
            const Real diag = (exm + exp) * idx2 + (eym + eyp) * idy2;
            const Real nb = (exm * P(i - 1, j, 0) + exp * P(i + 1, j, 0)) * idx2 +
                            (eym * P(i, j - 1, 0) + eyp * P(i, j + 1, 0)) * idy2;
            const Real cross =
                hc ? detail::cross_div(Pc, true, AXY, true, AYX, i, j, idx, idy) : Real(0);
            const Real pgs = (nb + cross - F(i, j, 0)) / diag;
            P(i, j, 0) = (Real(1) - omega) * P(i, j, 0) + omega * pgs;
          }
  }
}

// add_flux_correction_ : accumulate the two-way C-F flux correction of the level-m/level-(m+1)
// interface into dst (level-m index space), reading the fine fluxes from level m+1. Enumerated from
// the UNCOVERED (coarse) side (design 4c): for each level-m cell (I,J) NOT covered by level m+1 with a
// covered neighbor, the shared face is a C-F face; add (coarse_flux - fine_flux). Interior
// fine-fine faces (covered on both sides) are never visited, so adjacency never double-counts. The
// per-face value is the identical (coarse - fine) as the legacy composite_coarse_residual; only the
// enumeration differs. Under MPI the fine fluxes come from possibly-remote patches: routed through a
// single-writer per-(cell,direction) FluxRegister in commit 4, exact gather + fixed fold.
inline void CompositeFacPoisson::add_flux_correction_(int m, MultiFab& dst) {
  const int L = n_levels_;
  if (m + 1 >= L)
    return;
  const Geometry& gm = geom_level(m);
  const CoverageMask& cov = cov_of_[m];
  const bool he = has_eps_;
  const Real idx2 = Real(1) / (gm.dx() * gm.dx());
  const Real idy2 = Real(1) / (gm.dy() * gm.dy());
  const int r = ratio_;
  // Coarse-side (level m) reader: at level 0 (mono-box replicated) fab(0) holds the whole grid; at an
  // intermediate multi-box level the bordering cell is inside a level-m patch by proper nesting, so we
  // read from the level-m fab whose GROWN box contains it. This keeps the fold correct for both the
  // mono-box level and a distributed intermediate level.
  MultiFab& phi_m = phi_level(m);
  MultiFab& eps_m = eps_level(m);
  auto read_m = [&](const MultiFab& mf, int I, int J) -> Real {
    for (int li = 0; li < mf.local_size(); ++li)
      if (mf.fab(li).grown_box().contains(I, J))
        return mf.fab(li).const_array()(I, J, 0);
    return Real(0);  // unreachable under proper nesting (the border cell is inside a level-m patch)
  };
  const BoxArray& child = (m + 1 == 1) ? ba_f_ : ba_lv_[(m + 1) - 2];
  MultiFab& phi_child = phi_level(m + 1);
  MultiFab& eps_child = eps_level(m + 1);
  // FluxRegister over the level-m grid: single-writer per (cell,direction) slot (4/cell), gather()
  // sums (x,0,0,..) exactly, then a fixed-order fold. Serial: gather() is the identity -> bit-identical.
  const Box2D region = Box2D::from_extents(gm.domain.nx(), gm.domain.ny());
  FluxRegister reg(region, 4);  // slot k in {0:xm, 1:xp, 2:ym, 3:yp}
  for (int g = 0; g < child.size(); ++g) {
    // level-m footprint of child patch g.
    const Box2D pc = foot_of_[m + 1][g].empty() ? PatchRange(child[g]).box() : foot_of_[m + 1][g];
    const int Ic0 = pc.lo[0], Ic1 = pc.hi[0], Jc0 = pc.lo[1], Jc1 = pc.hi[1];
    // only the rank owning this child patch writes its slots (single writer). Serial: always local.
    const int li = phi_child.local_index_of(g);
    if (li < 0)
      continue;
    const ConstArray4 PF = phi_child.fab(li).const_array();
    const ConstArray4 EF = eps_child.fab(li).const_array();
    // x-normal faces: left border column I = Ic0-1 (fine face i = r*Ic0), right I = Ic1+1.
    for (int J = Jc0; J <= Jc1; ++J) {
      if (!cov.covered(Ic0 - 1, J)) {
        const int I = Ic0 - 1;
        const Real efc = he ? eps_harmonic(read_m(eps_m, I, J), read_m(eps_m, I + 1, J)) : Real(1);
        const Real coarse_c = efc * (read_m(phi_m, I + 1, J) - read_m(phi_m, I, J)) * idx2;
        Real fine_sum = Real(0);
        for (int t = 0; t < r; ++t) {
          const int jf = r * J + t;
          const Real eff = he ? eps_harmonic(EF(r * Ic0 - 1, jf, 0), EF(r * Ic0, jf, 0)) : Real(1);
          fine_sum += eff * (PF(r * Ic0, jf, 0) - PF(r * Ic0 - 1, jf, 0));
        }
        reg.add(I, J, 1, coarse_c - fine_sum * idx2);  // +x face of (I,J)
      }
      if (!cov.covered(Ic1 + 1, J)) {
        const int I = Ic1 + 1;
        const Real efc = he ? eps_harmonic(read_m(eps_m, I, J), read_m(eps_m, I - 1, J)) : Real(1);
        const Real coarse_c = efc * (read_m(phi_m, I - 1, J) - read_m(phi_m, I, J)) * idx2;
        Real fine_sum = Real(0);
        for (int t = 0; t < r; ++t) {
          const int jf = r * J + t;
          const Real eff =
              he ? eps_harmonic(EF(r * Ic1 + r - 1, jf, 0), EF(r * Ic1 + r, jf, 0)) : Real(1);
          fine_sum += eff * (PF(r * Ic1 + r - 1, jf, 0) - PF(r * Ic1 + r, jf, 0));
        }
        reg.add(I, J, 0, coarse_c - fine_sum * idx2);  // -x face of (I,J)
      }
    }
    // y-normal faces: bottom border row J = Jc0-1, top J = Jc1+1.
    for (int I = Ic0; I <= Ic1; ++I) {
      if (!cov.covered(I, Jc0 - 1)) {
        const int J = Jc0 - 1;
        const Real efc = he ? eps_harmonic(read_m(eps_m, I, J), read_m(eps_m, I, J + 1)) : Real(1);
        const Real coarse_c = efc * (read_m(phi_m, I, J + 1) - read_m(phi_m, I, J)) * idy2;
        Real fine_sum = Real(0);
        for (int t = 0; t < r; ++t) {
          const int iff = r * I + t;
          const Real eff = he ? eps_harmonic(EF(iff, r * Jc0 - 1, 0), EF(iff, r * Jc0, 0)) : Real(1);
          fine_sum += eff * (PF(iff, r * Jc0, 0) - PF(iff, r * Jc0 - 1, 0));
        }
        reg.add(I, J, 3, coarse_c - fine_sum * idy2);  // +y face of (I,J)
      }
      if (!cov.covered(I, Jc1 + 1)) {
        const int J = Jc1 + 1;
        const Real efc = he ? eps_harmonic(read_m(eps_m, I, J), read_m(eps_m, I, J - 1)) : Real(1);
        const Real coarse_c = efc * (read_m(phi_m, I, J - 1) - read_m(phi_m, I, J)) * idy2;
        Real fine_sum = Real(0);
        for (int t = 0; t < r; ++t) {
          const int iff = r * I + t;
          const Real eff =
              he ? eps_harmonic(EF(iff, r * Jc1 + r - 1, 0), EF(iff, r * Jc1 + r, 0)) : Real(1);
          fine_sum += eff * (PF(iff, r * Jc1 + r - 1, 0) - PF(iff, r * Jc1 + r, 0));
        }
        reg.add(I, J, 2, coarse_c - fine_sum * idy2);  // -y face of (I,J)
      }
    }
  }
  reg.gather();  // all_reduce_sum of single-writer slots -> exact; serial identity.
  // fold the 4 slots into dst in a FIXED direction order (xm, xp, ym, yp) -> np-invariant bits.
  for (int li = 0; li < dst.local_size(); ++li) {
    Array4 D = dst.fab(li).array();
    const Box2D b = dst.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        if (!cov.covered(i, j))
          D(i, j, 0) +=
              reg.at(i, j, 0) + reg.at(i, j, 1) + reg.at(i, j, 2) + reg.at(i, j, 3);
  }
}

// ------------------------------------------------------------------------------------------------
// composite_residual_(m) : res_m = f_m - L_m(phi_m) on uncovered cells, 0 on covered; + the C-F flux
// correction of the level-m/level-(m+1) interface; return ||res_m||_inf over uncovered cells, reduced
// by all_reduce_max (collective, so the stopping criterion is synchronized across ranks). At m == 0,
// L == 2 this is algebraically identical to the legacy composite_coarse_residual.
// ------------------------------------------------------------------------------------------------
inline Real CompositeFacPoisson::composite_residual_(int m) {
  const Geometry& gm = geom_level(m);
  MultiFab& phim = phi_level(m);
  MultiFab& resm = res_level_(m);
  device_fence();
  // ghosts for the operator: level 0 uses the physical BC (it is the whole domain); a patch level uses
  // the C/F bilerp + fine-fine order (its edges are INTERIOR to the refined domain, not physical).
  if (m == 0) {
    fill_ghosts(phim, gm.domain, bc_);
  } else {
    if (m - 1 == 0)
      fill_ghosts(phi_c_, geom_c_.domain, bc_);
    fill_cf_phi_(m);
  }
  MultiFab lap(phim.box_array(), phim.dmap(), 1, 0);
  apply_laplacian(phim, gm, lap, /*coef=*/nullptr, has_eps_ ? &eps_level(m) : nullptr,
                  /*kappa=*/nullptr, /*eps_y=*/nullptr, has_cross_ ? &a_xy_level(m) : nullptr,
                  has_cross_ ? &a_yx_level(m) : nullptr);
  device_fence();
  const CoverageMask& cov = cov_of_[m];
  for (int li = 0; li < resm.local_size(); ++li) {
    Array4 R = resm.fab(li).array();
    const ConstArray4 LAP = lap.fab(li).const_array();
    const ConstArray4 FM = rhs_level(m).fab(li).const_array();
    const Box2D b = resm.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        R(i, j, 0) = cov.covered(i, j) ? Real(0) : (FM(i, j, 0) - LAP(i, j, 0));
  }
  add_flux_correction_(m, resm);  // += (coarse - fine) on the bordering cells

  Real nrm = Real(0);
  for (int li = 0; li < resm.local_size(); ++li) {
    const ConstArray4 R = resm.fab(li).const_array();
    const Box2D b = resm.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        if (!cov.covered(i, j))
          nrm = std::fmax(nrm, std::fabs(R(i, j, 0)));
  }
  return Real(all_reduce_max(static_cast<double>(nrm)));
}

// ------------------------------------------------------------------------------------------------
// relax_level_(m), 1 <= m <= L-1 (design 3d) :
//   1. physical ghosts of the parent (level 0) if m-1 == 0; canonical ghost order (3b) for phi_m;
//   2. effective RHS: f_eff = f_m (finest, m == L-1, bit-identical to the legacy fine_sor input) or,
//      for an intermediate level, f_m plus the (coarse-fine) flux fold at the level-(m+1) interface
//      so the level-m SOR uses the level-(m+1) fine flux at that face -> two-way at EVERY interface;
//   3. fine_sor over uncovered cells with frozen ghosts;
//   4. average_down(phi_m -> phi_{m-1}) over the covered parent cells (consistency).
// At L == 2 (m == 1) this is exactly refresh_fine: fill_ghosts(phi_0) + fill_cf + fine_sor(f_1) +
// average_down(phi_1 -> phi_0), bit-identical to the legacy path.
// ------------------------------------------------------------------------------------------------
inline void CompositeFacPoisson::relax_level_(int m, int sweeps) {
  device_fence();
  if (m - 1 == 0)
    fill_ghosts(phi_c_, geom_c_.domain, bc_);  // parent physical ghosts (bilerp reads to the border)
  const MultiFab& phim = phi_level(m);
  const bool multibox = phim.box_array().size() > 1;  // adjacency: fine-fine sibling ghosts matter
  if (!multibox) {
    // single patch: bilerp C/F ghosts once, then SOR with frozen ghosts. Bit-identical to the legacy
    // refresh_fine (fill_cf_ghosts + fine_sor) at L == 2, non-adjacent.
    fill_cf_phi_(m);
    fine_sor_level_(m, rhs_level(m), sweeps);
  } else {
    // ADJACENT patches: the shared fine-fine ghost must be RE-EXCHANGED between sweep batches so the
    // patches couple Gauss-Seidel (not once-per-relax block-Jacobi, which is unstable across a shared
    // face). Re-run the canonical ghost order (bilerp + fill_boundary) between short SOR batches. The
    // C/F bilerp ghosts at the outer boundary are re-derived from the (unchanged) parent each batch --
    // idempotent -- while the shared-face ghosts pick up the sibling's freshest interior.
    const int nbatch = 8;
    const int per = std::max(1, sweeps / nbatch);
    int done = 0;
    while (done < sweeps) {
      const int s = std::min(per, sweeps - done);
      fill_cf_phi_(m);
      fine_sor_level_(m, rhs_level(m), s);
      done += s;
    }
  }
  // NB: the average-down is NOT done here. It is a SEPARATE fine-to-coarse cascade (cascade_avgdown_)
  // run after ALL levels are relaxed, so a coarser parent's covered cells reflect the FINAL finer
  // values (avoids the staleness of avgdown m->m-1 before level m+1 updated level m).
}

// cascade_avgdown_ : fine-to-coarse average-down of the whole tower (L-1 -> L-2 -> ... -> 0), so each
// parent's covered cells equal the fine average of the FINAL child (conservation to ulp at every
// interface). At L == 2 this is the single average_down(phi_1 -> phi_0) of the legacy refresh_fine.
inline void CompositeFacPoisson::cascade_avgdown_() {
  for (int m = n_levels_ - 1; m >= 1; --m)
    average_down_level_(m);
}

// average_down_level_(m) : phi_level(m) -> phi_level(m-1) over the covered parent cells. For an
// intermediate distributed parent (m-1 >= 1) the standard average_down/parallel_copy is correct. For
// the REPLICATED coarse parent (m-1 == 0) under MPI, parallel_copy only updates the src-owner rank
// (every rank claims coarse box 0), so we instead compute each covered coarse cell's 2x2 fine average
// into a single-writer FluxRegister and gather() it: each covered cell has exactly one owner (the rank
// holding the covering fine patch), so the all_reduce_sum of (x,0,..) is exact and every rank's
// replicated coarse gets the identical value. Serial: gather() is the identity and this equals the
// legacy average_down bit-for-bit.
inline void CompositeFacPoisson::average_down_level_(int m) {
  MultiFab& parent = phi_level(m - 1);
  MultiFab& child = phi_level(m);
  if (m - 1 != 0 || n_ranks() == 1) {
    average_down(child, parent, ratio_);  // distributed parent, or serial -> legacy path (bit-identical)
    return;
  }
  // replicated coarse under MPI: single-writer FluxRegister over the coarse grid.
  const Geometry& gc = geom_level(0);
  const Box2D region = Box2D::from_extents(gc.domain.nx(), gc.domain.ny());
  FluxRegister reg(region, 1);
  const int r = ratio_;
  for (int li = 0; li < child.local_size(); ++li) {
    const ConstArray4 F = child.fab(li).const_array();
    const PatchRange pr(child.box(li));  // coarse footprint of this fine patch
    for (int J = pr.J0; J <= pr.J1; ++J)
      for (int I = pr.I0; I <= pr.I1; ++I) {
        Real avg = Real(0);
        for (int b = 0; b < r; ++b)
          for (int a = 0; a < r; ++a)
            avg += F(r * I + a, r * J + b, 0);
        reg.set(I, J, 0, avg / Real(r * r));  // single writer per covered coarse cell
      }
  }
  reg.gather();  // all_reduce_sum of single-writer slots -> exact; every rank now has the averages.
  Array4 P = parent.fab(0).array();  // replicated coarse (fab(0) on every rank)
  // write every covered coarse cell from the (now global) register, using the GLOBAL child footprints
  // so every rank writes the same cells (not just its local patches).
  const BoxArray& child_ba = (m == 1) ? ba_f_ : ba_lv_[m - 2];
  for (int g = 0; g < child_ba.size(); ++g) {
    const PatchRange pr(child_ba[g]);
    for (int J = pr.J0; J <= pr.J1; ++J)
      for (int I = pr.I0; I <= pr.I1; ++I)
        P(I, J, 0) = reg.at(I, J, 0);
  }
}

// ------------------------------------------------------------------------------------------------
// correct_level_(m), 0 <= m <= L-2 (design 3e) : solve L_m e_m = res_m (homogeneous Dirichlet) and
// add e_m to phi_m on uncovered cells.
//   m == 0     : base mg_ (identical to the legacy coarse correction :225-228);
//   1<=m<=L-2  : the patch-union multigrid level_mg_[m].
// ------------------------------------------------------------------------------------------------
inline void CompositeFacPoisson::correct_level_(int m) {
  if (m == 0) {
    copy0_(mg_.rhs(), res_c_);
    mg_.phi().set_val(Real(0));
    mg_.solve(options_.coarse_rel_tol, options_.coarse_cycles);
    add_uncovered_level_(0, phi_c_, mg_.phi());
    return;
  }
  GeometricMG& mgk = *level_mg_[m];
  const BoxArray& bam = (m == 1) ? ba_f_ : ba_lv_[m - 2];
  const bool single_patch_replicated = (bam.size() == 1);  // the mg is replicated (finalize_...)
  if (single_patch_replicated && n_ranks() > 1) {
    // MPI: the level-m patch is co-located on ONE rank but the correction mg is REPLICATED (every rank
    // solves it). Broadcast the distributed patch's residual + coefficients to a replicated copy on
    // every rank (single owner -> all_reduce_sum of the owner's data + 0 elsewhere = an exact
    // broadcast), solve the replicated mg identically on all ranks, then write the correction back to
    // the distributed patch on the owning rank. Deterministic + bit-identical across np.
    const BoxArray& rba = mgk.rhs().box_array();
    const DistributionMapping& rdm = mgk.rhs().dmap();  // replicated (every rank owns the box)
    auto broadcast = [&](const MultiFab& dist) {
      MultiFab rep(rba, rdm, 1, dist.n_grow());
      rep.set_val(Real(0));
      // the owner writes its patch cells; others contribute 0.
      for (int li = 0; li < dist.local_size(); ++li) {
        Array4 R = rep.fab(0).array();
        const ConstArray4 S = dist.fab(li).const_array();
        const Box2D b = dist.box(li);
        for (int j = b.lo[1]; j <= b.hi[1]; ++j)
          for (int i = b.lo[0]; i <= b.hi[0]; ++i)
            R(i, j, 0) = S(i, j, 0);
      }
      all_reduce_sum_inplace(rep.fab(0).array().p, static_cast<int>(rep.fab(0).size()));
      return rep;
    };
    MultiFab res_rep = broadcast(res_level_(m));
    if (has_eps_)
      mgk.set_epsilon(broadcast(eps_level(m)));
    if (has_cross_)
      mgk.set_cross_terms(broadcast(a_xy_level(m)), broadcast(a_yx_level(m)));
    copy0_(mgk.rhs(), res_rep);
    mgk.phi().set_val(Real(0));
    mgk.solve(options_.coarse_rel_tol, options_.coarse_cycles);
    // write the correction back to the distributed patch on its owner (add_uncovered on the local fab).
    const CoverageMask& cov = cov_of_[m];
    MultiFab& phim = phi_level(m);
    const ConstArray4 E = mgk.phi().fab(0).const_array();
    for (int li = 0; li < phim.local_size(); ++li) {
      Array4 P = phim.fab(li).array();
      const Box2D b = phim.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          if (!cov.covered(i, j))
            P(i, j, 0) += E(i, j, 0);
    }
    return;
  }
  // serial (or a distributed multi-patch mg): feed the level fields directly.
  if (has_eps_)
    mgk.set_epsilon(eps_level(m));
  if (has_cross_)
    mgk.set_cross_terms(a_xy_level(m), a_yx_level(m));
  copy0_(mgk.rhs(), res_level_(m));
  mgk.phi().set_val(Real(0));
  mgk.solve(options_.coarse_rel_tol, options_.coarse_cycles);
  add_uncovered_level_(m, phi_level(m), mgk.phi());
}

// ------------------------------------------------------------------------------------------------
// solve_composite_nlevel_ : the general FAC driver (design 3f). At L == 2 it is a single-iteration
// loop that reduces algebraically to the legacy 2-level loop.
// ------------------------------------------------------------------------------------------------
inline Real CompositeFacPoisson::solve_composite_nlevel_(int max_iters, int fine_sweeps, Real tol) {
  const int L = n_levels_;
  setup_level_coeffs_();  // == the legacy :189-215 coefficient setup, per level

  // 0) initial coarse solve (base level).
  copy0_(mg_.rhs(), f_c_);
  mg_.phi().set_val(Real(0));
  mg_.solve(options_.coarse_rel_tol, options_.coarse_cycles);
  copy0_(phi_c_, mg_.phi());

  // 1) relax every patch level coarse-to-fine, then a fine-to-coarse average-down cascade (==
  //    refresh_fine at L == 2: fill_cf + fine_sor + average_down).
  for (int m = 1; m < L; ++m)
    relax_level_(m, fine_sweeps);
  cascade_avgdown_();

  diagnostics_.clear();
  Real rnorm = composite_residual_(0);
  record_residual(-1, rnorm);
  if (!two_way_) {
    last_residual_ = rnorm;
    return rnorm;
  }

  // 2) FAC two-way iterations -- a MULTIPLICATIVE (Gauss-Seidel over the level chain) coarse-to-fine
  // sweep. For each interface m/(m+1) in order: recompute the level-m composite residual (flux-
  // corrected at its child interface, so it is FRESH after the coarser corrections), correct level m
  // (base multigrid at m == 0, patch multigrid otherwise), then re-relax the immediately finer level
  // on the corrected parent. This carries every interface two-way by the SAME mechanism the legacy
  // 2-level loop uses for the 0-1 interface, and recomputing each residual before its correction (not
  // an additive Jacobi batch) is what drives ||res_0|| below tol. A fine-to-coarse average-down
  // cascade closes the sweep (conservation to ulp). At L == 2 this is exactly correct(0) +
  // relax(1) + average_down + residual, bit-identical to the legacy loop.
  for (int it = 0; it < max_iters; ++it) {
    if (rnorm < tol)
      break;
    for (int m = 0; m + 1 < L; ++m) {
      if (m > 0)
        composite_residual_(m);  // fresh level-m residual (res_c_ for m==0 is already current)
      correct_level_(m);
      relax_level_(m + 1, fine_sweeps);  // re-establish the finer level on the corrected parent
    }
    cascade_avgdown_();
    rnorm = composite_residual_(0);
    record_residual(it, rnorm);
  }
  last_residual_ = rnorm;
  return rnorm;
}

}  // namespace pops
