#pragma once

#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>

#include <pops/mesh/boundary/fill_boundary.hpp>  // fill_boundary (intra-level fine-fine + MPI adjacency)
#include <pops/parallel/comm.hpp>                // all_reduce_max (collective residual norm)

#include <algorithm>
#include <cmath>
#include <memory>
#include <type_traits>
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

namespace detail {

/// Fill only the coarse/fine ghost ring of one patch.  Every field captured here is a POD device
/// handle or a scalar; the host MultiFab and its ownership metadata never cross the kernel seam.
struct FacFillCoarseFineGhostKernel {
  Array4 fine;
  ConstArray4 coarse;
  Box2D valid;
  int ratio;

  POPS_HD void operator()(int i, int j) const {
    const bool inside =
        i >= valid.lo[0] && i <= valid.hi[0] && j >= valid.lo[1] && j <= valid.hi[1];
    if (!inside)
      fine(i, j, 0) = fac_bilerp_coarse(coarse, i, j, ratio);
  }
};

/// One red or black five-point SOR color.  A five-point stencil has no dependency between cells of
/// the same color, hence this parallel kernel preserves the sequential red/black arithmetic.  The
/// nine-point cross-term path deliberately remains host-ordered below because its diagonal stencil
/// couples cells of the same color.
struct FacRedBlackFivePointSorKernel {
  Array4 phi;
  ConstArray4 rhs;
  ConstArray4 eps;
  ConstArray4 eps_y;
  Real idx2;
  Real idy2;
  Real omega;
  Real reaction;
  int color;
  bool has_eps;

  POPS_HD void operator()(int i, int j) const {
    if (((i + j) & 1) != color)
      return;
    const Real exm = has_eps ? eps_harmonic(eps(i, j, 0), eps(i - 1, j, 0)) : Real(1);
    const Real exp = has_eps ? eps_harmonic(eps(i, j, 0), eps(i + 1, j, 0)) : Real(1);
    const Real eym = has_eps ? eps_harmonic(eps_y(i, j, 0), eps_y(i, j - 1, 0)) : Real(1);
    const Real eyp = has_eps ? eps_harmonic(eps_y(i, j, 0), eps_y(i, j + 1, 0)) : Real(1);
    const Real diag = (exm + exp) * idx2 + (eym + eyp) * idy2 + reaction;
    const Real nb = (exm * phi(i - 1, j, 0) + exp * phi(i + 1, j, 0)) * idx2 +
                    (eym * phi(i, j - 1, 0) + eyp * phi(i, j + 1, 0)) * idy2;
    const Real cross = Real(0);
    const Real pgs = (nb + cross - rhs(i, j, 0)) / diag;
    phi(i, j, 0) = (Real(1) - omega) * phi(i, j, 0) + omega * pgs;
  }
};

/// Unmasked residual on the finest active level (which, by definition, has no covered cells).
struct FacFinestResidualKernel {
  Array4 residual;
  ConstArray4 rhs;
  ConstArray4 laplacian;

  POPS_HD void operator()(int i, int j) const {
    residual(i, j, 0) = rhs(i, j, 0) - laplacian(i, j, 0);
  }
};

/// Exact infinity-norm reducer for a finest-level residual.  Every non-finite sample maps to +inf
/// before Kokkos::Max, so neither NaN operand order nor the backend reduction tree can hide it.
struct FacFinestNormInfKernel {
  ConstArray4 residual;

  POPS_HD void operator()(int i, int j, Real& acc) const {
    const Real v = residual(i, j, 0);
    const Real av = v < Real(0) ? -v : v;
    if (!(av <= std::numeric_limits<Real>::max())) {
      acc = std::numeric_limits<Real>::infinity();
      return;
    }
    if (av > acc)
      acc = av;
  }
};

struct FacMaskedAddKernel {
  Array4 destination;
  ConstArray4 correction;
  CoverageMaskView coverage;

  POPS_HD void operator()(int i, int j) const {
    if (!coverage.covered(i, j))
      destination(i, j, 0) += correction(i, j, 0);
  }
};

/// Red/black five-point SOR and four-color nine-point SOR share one exact kernel.  The cross
/// stencil touches diagonal neighbours, so two colors are not an independent set; four colors are
/// required for a race-free device Gauss-Seidel update.  The diagonal-only route remains the
/// historical two-color ordering.
struct FacMaskedSorKernel {
  Array4 phi;
  ConstArray4 phi_read;
  ConstArray4 rhs;
  ConstArray4 eps;
  ConstArray4 eps_y;
  ConstArray4 a_xy;
  ConstArray4 a_yx;
  CoverageMaskView coverage;
  Real idx2;
  Real idy2;
  Real idx;
  Real idy;
  Real omega;
  Real reaction;
  int color;
  bool has_eps;
  bool has_cross;

  POPS_HD void operator()(int i, int j) const {
    const int cell_color = has_cross ? (((i & 1) << 1) | (j & 1)) : ((i + j) & 1);
    if (cell_color != color || coverage.covered(i, j))
      return;
    const Real exm = has_eps ? eps_harmonic(eps(i, j, 0), eps(i - 1, j, 0)) : Real(1);
    const Real exp = has_eps ? eps_harmonic(eps(i, j, 0), eps(i + 1, j, 0)) : Real(1);
    const Real eym = has_eps ? eps_harmonic(eps_y(i, j, 0), eps_y(i, j - 1, 0)) : Real(1);
    const Real eyp = has_eps ? eps_harmonic(eps_y(i, j, 0), eps_y(i, j + 1, 0)) : Real(1);
    const Real diagonal = (exm + exp) * idx2 + (eym + eyp) * idy2 + reaction;
    const Real neighbours =
        (exm * phi(i - 1, j, 0) + exp * phi(i + 1, j, 0)) * idx2 +
        (eym * phi(i, j - 1, 0) + eyp * phi(i, j + 1, 0)) * idy2;
    const Real cross = has_cross
                           ? cross_div(phi_read, true, a_xy, true, a_yx, i, j, idx, idy)
                           : Real(0);
    const Real candidate = (neighbours + cross - rhs(i, j, 0)) / diagonal;
    phi(i, j, 0) = (Real(1) - omega) * phi(i, j, 0) + omega * candidate;
  }
};

struct FacMaskedResidualKernel {
  Array4 residual;
  ConstArray4 rhs;
  ConstArray4 laplacian;
  CoverageMaskView coverage;

  POPS_HD void operator()(int i, int j) const {
    residual(i, j, 0) =
        coverage.covered(i, j) ? Real(0) : rhs(i, j, 0) - laplacian(i, j, 0);
  }
};

struct FacMaskedNormInfKernel {
  ConstArray4 residual;
  CoverageMaskView coverage;

  POPS_HD void operator()(int i, int j, Real& acc) const {
    if (coverage.covered(i, j))
      return;
    const Real value = residual(i, j, 0);
    const Real magnitude = value < Real(0) ? -value : value;
    if (!(magnitude <= std::numeric_limits<Real>::max())) {
      acc = std::numeric_limits<Real>::infinity();
      return;
    }
    if (magnitude > acc)
      acc = magnitude;
  }
};

struct FacFluxFoldKernel {
  Array4 destination;
  CoverageMaskView coverage;
  FluxRegisterView flux;

  POPS_HD void operator()(int i, int j) const {
    if (!coverage.covered(i, j))
      destination(i, j, 0) += flux.at(i, j, 0) + flux.at(i, j, 1) + flux.at(i, j, 2) +
                              flux.at(i, j, 3);
  }
};

struct FacFluxCorrectionKernel {
  ConstArray4 coarse_phi;
  ConstArray4 coarse_eps;
  ConstArray4 coarse_eps_y;
  ConstArray4 fine_phi;
  ConstArray4 fine_eps;
  ConstArray4 fine_eps_y;
  CoverageMaskView coverage;
  FluxRegisterView flux;
  Box2D footprint;
  Real idx2;
  Real idy2;
  int ratio;
  bool has_eps;

  POPS_HD void operator()(int i, int j) const {
    if (coverage.covered(i, j))
      return;
    const int ilo = footprint.lo[0];
    const int ihi = footprint.hi[0];
    const int jlo = footprint.lo[1];
    const int jhi = footprint.hi[1];
    if (i == ilo - 1 && j >= jlo && j <= jhi) {
      const Real coarse_face =
          (has_eps ? eps_harmonic(coarse_eps(i, j, 0), coarse_eps(i + 1, j, 0)) : Real(1)) *
          (coarse_phi(i + 1, j, 0) - coarse_phi(i, j, 0)) * idx2;
      Real fine_sum = Real(0);
      for (int t = 0; t < ratio; ++t) {
        const int jf = ratio * j + t;
        const Real face = has_eps
                              ? eps_harmonic(fine_eps(ratio * ilo - 1, jf, 0),
                                             fine_eps(ratio * ilo, jf, 0))
                              : Real(1);
        fine_sum += face *
                    (fine_phi(ratio * ilo, jf, 0) - fine_phi(ratio * ilo - 1, jf, 0));
      }
      flux.add(i, j, 1, coarse_face - fine_sum * idx2);
      return;
    }
    if (i == ihi + 1 && j >= jlo && j <= jhi) {
      const Real coarse_face =
          (has_eps ? eps_harmonic(coarse_eps(i, j, 0), coarse_eps(i - 1, j, 0)) : Real(1)) *
          (coarse_phi(i - 1, j, 0) - coarse_phi(i, j, 0)) * idx2;
      Real fine_sum = Real(0);
      for (int t = 0; t < ratio; ++t) {
        const int jf = ratio * j + t;
        const Real face =
            has_eps ? eps_harmonic(fine_eps(ratio * ihi + ratio - 1, jf, 0),
                                    fine_eps(ratio * ihi + ratio, jf, 0))
                    : Real(1);
        fine_sum += face * (fine_phi(ratio * ihi + ratio - 1, jf, 0) -
                            fine_phi(ratio * ihi + ratio, jf, 0));
      }
      flux.add(i, j, 0, coarse_face - fine_sum * idx2);
      return;
    }
    if (j == jlo - 1 && i >= ilo && i <= ihi) {
      const Real coarse_face =
          (has_eps ? eps_harmonic(coarse_eps_y(i, j, 0), coarse_eps_y(i, j + 1, 0))
                   : Real(1)) *
          (coarse_phi(i, j + 1, 0) - coarse_phi(i, j, 0)) * idy2;
      Real fine_sum = Real(0);
      for (int t = 0; t < ratio; ++t) {
        const int fi = ratio * i + t;
        const Real face = has_eps
                              ? eps_harmonic(fine_eps_y(fi, ratio * jlo - 1, 0),
                                             fine_eps_y(fi, ratio * jlo, 0))
                              : Real(1);
        fine_sum += face *
                    (fine_phi(fi, ratio * jlo, 0) - fine_phi(fi, ratio * jlo - 1, 0));
      }
      flux.add(i, j, 3, coarse_face - fine_sum * idy2);
      return;
    }
    if (j == jhi + 1 && i >= ilo && i <= ihi) {
      const Real coarse_face =
          (has_eps ? eps_harmonic(coarse_eps_y(i, j, 0), coarse_eps_y(i, j - 1, 0))
                   : Real(1)) *
          (coarse_phi(i, j - 1, 0) - coarse_phi(i, j, 0)) * idy2;
      Real fine_sum = Real(0);
      for (int t = 0; t < ratio; ++t) {
        const int fi = ratio * i + t;
        const Real face =
            has_eps ? eps_harmonic(fine_eps_y(fi, ratio * jhi + ratio - 1, 0),
                                    fine_eps_y(fi, ratio * jhi + ratio, 0))
                    : Real(1);
        fine_sum += face * (fine_phi(fi, ratio * jhi + ratio - 1, 0) -
                            fine_phi(fi, ratio * jhi + ratio, 0));
      }
      flux.add(i, j, 2, coarse_face - fine_sum * idy2);
    }
  }
};

struct FacAverageDownRegisterKernel {
  ConstArray4 fine;
  FluxRegisterView register_view;
  int ratio;

  POPS_HD void operator()(int i, int j) const {
    Real sum = Real(0);
    for (int dj = 0; dj < ratio; ++dj)
      for (int di = 0; di < ratio; ++di)
        sum += fine(ratio * i + di, ratio * j + dj, 0);
    register_view.set(i, j, 0, sum / Real(ratio * ratio));
  }
};

struct FacCopyCoveredRegisterKernel {
  Array4 destination;
  CoverageMaskView coverage;
  FluxRegisterView register_view;

  POPS_HD void operator()(int i, int j) const {
    if (coverage.covered(i, j))
      destination(i, j, 0) = register_view.at(i, j, 0);
  }
};

static_assert(std::is_trivially_copyable_v<FacFillCoarseFineGhostKernel>);
static_assert(std::is_trivially_copyable_v<FacRedBlackFivePointSorKernel>);
static_assert(std::is_trivially_copyable_v<FacFinestResidualKernel>);
static_assert(std::is_trivially_copyable_v<FacFinestNormInfKernel>);
static_assert(std::is_trivially_copyable_v<FacMaskedAddKernel>);
static_assert(std::is_trivially_copyable_v<FacMaskedSorKernel>);
static_assert(std::is_trivially_copyable_v<FacMaskedResidualKernel>);
static_assert(std::is_trivially_copyable_v<FacMaskedNormInfKernel>);
static_assert(std::is_trivially_copyable_v<FacFluxFoldKernel>);
static_assert(std::is_trivially_copyable_v<FacFluxCorrectionKernel>);
static_assert(std::is_trivially_copyable_v<FacAverageDownRegisterKernel>);
static_assert(std::is_trivially_copyable_v<FacCopyCoveredRegisterKernel>);

}  // namespace detail

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
    MultiFab phik(bak, dmk, 1, 1), fk(bak, dmk, 1, 0), rk(bak, dmk, 1, 0), lapk(bak, dmk, 1, 0);
    MultiFab epk(bak, dmk, 1, 1), eyk(bak, dmk, 1, 1), axk(bak, dmk, 1, 1),
        ayk(bak, dmk, 1, 1);
    phik.set_val(Real(0));
    fk.set_val(Real(0));
    rk.set_val(Real(0));
    lapk.set_val(Real(0));
    epk.set_val(Real(1));
    eyk.set_val(Real(1));
    axk.set_val(Real(0));
    ayk.set_val(Real(0));
    phi_lv_.push_back(std::move(phik));
    f_lv_.push_back(std::move(fk));
    res_lv_.push_back(std::move(rk));
    lap_lv_.push_back(std::move(lapk));
    eps_lv_.push_back(std::move(epk));
    eps_y_lv_.push_back(std::move(eyk));
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
      level_mg_[m] = std::make_unique<GeometricMG>(gp, bam, dir, ActiveRegionProvider2D{},
                                                   FieldDistribution::Replicated);
    } else {
      level_mg_[m] = std::make_unique<GeometricMG>(gm, bam, dir, ActiveRegionProvider2D{},
                                                   FieldDistribution::Distributed);
    }
  }

  // FAC residual / correction scratch is part of the hierarchy allocation, not the iteration.  A
  // FluxRegister owns host storage, therefore constructing one in composite_residual_ would allocate
  // on every residual evaluation (and becomes especially visible for FAS).  One register per
  // coarse/fine interface is sufficient because residual evaluation is sequential by level.
  flux_registers_.clear();
  flux_registers_.resize(static_cast<std::size_t>(L));
  for (int m = 0; m + 1 < L; ++m) {
    const Geometry& gm = geom_level(m);
    const Box2D region = Box2D::from_extents(gm.domain.nx(), gm.domain.ny());
    flux_registers_[static_cast<std::size_t>(m)] = std::make_unique<FluxRegister>(region, 4);
  }
  const Box2D coarse_region = Box2D::from_extents(geom_c_.domain.nx(), geom_c_.domain.ny());
  coarse_average_register_ = std::make_unique<FluxRegister>(coarse_region, 1);

  correction_residual_replicated_.clear();
  correction_eps_replicated_.clear();
  correction_eps_y_replicated_.clear();
  correction_axy_replicated_.clear();
  correction_ayx_replicated_.clear();
  correction_residual_replicated_.resize(static_cast<std::size_t>(L));
  correction_eps_replicated_.resize(static_cast<std::size_t>(L));
  correction_eps_y_replicated_.resize(static_cast<std::size_t>(L));
  correction_axy_replicated_.resize(static_cast<std::size_t>(L));
  correction_ayx_replicated_.resize(static_cast<std::size_t>(L));
  for (int m = 1; m + 1 < L; ++m) {
    GeometricMG& mgk = *level_mg_[static_cast<std::size_t>(m)];
    const BoxArray& rba = mgk.rhs().box_array();
    const DistributionMapping& rdm = mgk.rhs().dmap();
    correction_residual_replicated_[static_cast<std::size_t>(m)] = MultiFab(rba, rdm, 1, 0);
    correction_eps_replicated_[static_cast<std::size_t>(m)] = MultiFab(rba, rdm, 1, 1);
    correction_eps_y_replicated_[static_cast<std::size_t>(m)] = MultiFab(rba, rdm, 1, 1);
    correction_axy_replicated_[static_cast<std::size_t>(m)] = MultiFab(rba, rdm, 1, 1);
    correction_ayx_replicated_[static_cast<std::size_t>(m)] = MultiFab(rba, rdm, 1, 1);
  }
  prepare_fully_refined_solver_();
}

inline void CompositeFacPoisson::prepare_fully_refined_solver_() {
  fully_refined_solver_.reset();
  for (int level = 0; level + 1 < n_levels_; ++level)
    if (cov_of_[static_cast<std::size_t>(level)].covered_cell_count() !=
        static_cast<std::size_t>(geom_level(level).domain.num_cells()))
      return;

  const int finest = n_levels_ - 1;
  MultiFab& layout = phi_level(finest);
  fully_refined_solver_ = std::make_unique<GeometricMG>(
      geom_level(finest), layout.box_array(), layout.dmap(), bc_, ActiveRegionProvider2D{},
      FieldDistribution::Distributed);
  if (has_reaction_)
    fully_refined_solver_->set_reaction(constant_scalar_field_provider(reaction_));
  if (has_boundary_kernel_)
    fully_refined_solver_->set_boundary_kernel(boundary_kernel_, boundary_context_);
}

inline Real CompositeFacPoisson::solve_fully_refined_hierarchy_(
    int max_iters, Real rel_tol, Real abs_tol) {
  const int finest = n_levels_ - 1;
  GeometricMG& solver = *fully_refined_solver_;
  if (has_eps_) {
    if (has_eps_y_)
      solver.set_epsilon_anisotropic(eps_level(finest), eps_y_level(finest));
    else
      solver.set_epsilon(eps_level(finest));
  }
  if (has_cross_)
    solver.set_cross_terms(a_xy_level(finest), a_yx_level(finest));
  if (has_boundary_kernel_ && boundary_kernel_.observes_iteration)
    solver.set_boundary_context(boundary_context_);
  copy0_(solver.rhs(), rhs_level(finest));
  copy0_(solver.phi(), phi_level(finest));
  Real residual = Real(0);
  try {
    solver.solve(rel_tol, max_iters, abs_tol);
    residual = solver.last_residual();
  } catch (...) {
    last_solve_report_ = solver.last_solve_report();
    throw;
  }
  last_solve_report_ = solver.last_solve_report();
  copy0_(phi_level(finest), solver.phi());
  cascade_avgdown_();
  last_residual_ = residual;
  record_residual(last_solve_report_.iters, residual);
  return residual;
}

// dst <- src, component 0, valid cells (== the legacy copy0).
inline void CompositeFacPoisson::copy0_(MultiFab& dst, const MultiFab& src) {
  for (int li = 0; li < dst.local_size(); ++li) {
    Array4 d = dst.fab(li).array();
    const ConstArray4 s = src.fab(li).const_array();
    const Box2D b = dst.box(li);
    for_each_cell(b, detail::FacCopyAllKernel{d, s, 0});
  }
}

// phi += e on the level-m cells NOT covered by level m+1 (== the legacy add_uncovered, per level).
inline void CompositeFacPoisson::add_uncovered_level_(int m, MultiFab& phi, const MultiFab& e) {
  const CoverageMaskView coverage = cov_of_[m].view();
  for (int li = 0; li < phi.local_size(); ++li) {
    Array4 p = phi.fab(li).array();
    const ConstArray4 ec = e.fab(li).const_array();
    const Box2D b = phi.box(li);
    for_each_cell(b, detail::FacMaskedAddKernel{p, ec, coverage});
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
    if (has_eps_y_) {
      fill_ghosts(eps_y_c_, geom_c_.domain, coeff_bc(bc_));
      for (int k = 1; k < L; ++k)
        fill_cf_field_(k, eps_y_level(k), eps_y_level(k - 1));
      mg_.set_epsilon_anisotropic(eps_c_, eps_y_c_);
    } else {
      mg_.set_epsilon(eps_c_);
    }
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

inline Array4 writable_array_covering_(MultiFab& parent, const Box2D& footprint) {
  for (int li = 0; li < parent.local_size(); ++li)
    if (parent.box(li).contains(footprint))
      return parent.fab(li).array();
  return Array4{};
}

inline ConstArray4 readable_array_covering_(const MultiFab& parent, const Box2D& footprint) {
  for (int li = 0; li < parent.local_size(); ++li)
    if (parent.fab(li).grown_box().contains(footprint))
      return parent.fab(li).const_array();
  return ConstArray4{};
}
}  // namespace detail

// fill_cf_field_ : ghosts of a level-k COEFFICIENT field by bilerp of the parent (level k-1). Generic
// (eps, a_xy, a_yx); reads the CO-LOCATED local parent fab (mono-box replicated at level 0).
inline void CompositeFacPoisson::fill_cf_field_(int /*k*/, MultiFab& fine, const MultiFab& parent) {
  for (int li = 0; li < fine.local_size(); ++li) {
    Array4 F = fine.fab(li).array();
    const Box2D vb = fine.box(li);
    const ConstArray4 C = detail::parent_array_for_(parent, vb, ratio_, fine.n_grow());
    if (C.p == nullptr)
      continue;  // no local parent (should not happen under co-location); skip
    for_each_cell(fine.fab(li).grown_box(), detail::FacFillCoarseFineGhostKernel{F, C, vb, ratio_});
  }
}

// fill_cf_phi_ : the canonical ghost-fill order (design 3b) for phi_level(k), k >= 1:
//   1. parent phi ghosts already valid (physical BC on level 0 done by the caller);
//   2. fill_cf(phi_k <- phi_{k-1}) : order-2 bilerp into every ghost of every level-k patch;
//   3. fill_boundary(phi_k) : overwrite the ghosts overlapping a sibling patch's valid cells.
// For non-adjacent patches step 3 is a no-op -> identical to the legacy fill_cf_ghosts alone.
inline void CompositeFacPoisson::fill_cf_phi_(int k) {
  MultiFab& phik = phi_level(k);
  const MultiFab& parent = phi_level(k - 1);
  for (int li = 0; li < phik.local_size(); ++li) {
    Array4 F = phik.fab(li).array();
    const Box2D vb = phik.box(li);
    const ConstArray4 C = detail::parent_array_for_(parent, vb, ratio_, phik.n_grow());
    if (C.p == nullptr)
      continue;
    for_each_cell(phik.fab(li).grown_box(), detail::FacFillCoarseFineGhostKernel{F, C, vb, ratio_});
  }
  fill_boundary(phik,
                geom_level(k).domain);  // step 3: fine-fine sibling exchange (adjacency + MPI)
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
  const CoverageMaskView coverage = cov_of_[m].view();
  const bool finest_unmasked = m + 1 == n_levels_;
  for (int li = 0; li < phim.local_size(); ++li) {
    const Box2D vb = phim.box(li);
    const Real omega = sor_omega_(vb);
    Array4 P = phim.fab(li).array();
    const ConstArray4 Pc = phim.fab(li).const_array();
    const ConstArray4 F = f_eff.fab(li).const_array();
    const ConstArray4 E = eps_level(m).fab(li).const_array();
    const ConstArray4 EY = has_eps_y_ ? eps_y_level(m).fab(li).const_array()
                                     : eps_level(m).fab(li).const_array();
    const ConstArray4 AXY = a_xy_level(m).fab(li).const_array();
    const ConstArray4 AYX = a_yx_level(m).fab(li).const_array();
    if (finest_unmasked && !hc) {
      for (int s = 0; s < sweeps; ++s)
        for (int color = 0; color < 2; ++color)
          for_each_cell(
              vb, detail::FacRedBlackFivePointSorKernel{
                      P, F, E, EY, idx2, idy2, omega,
                      has_reaction_ ? reaction_ : Real(0), color, he});
      continue;
    }
    const int color_count = hc ? 4 : 2;
    for (int s = 0; s < sweeps; ++s)
      for (int color = 0; color < color_count; ++color)
        for_each_cell(vb, detail::FacMaskedSorKernel{
                              P, Pc, F, E, EY, AXY, AYX, coverage, idx2, idy2, idx, idy, omega,
                              has_reaction_ ? reaction_ : Real(0), color, he, hc});
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
  const CoverageMaskView coverage = cov_of_[m].view();
  const bool he = has_eps_;
  const Real idx2 = Real(1) / (gm.dx() * gm.dx());
  const Real idy2 = Real(1) / (gm.dy() * gm.dy());
  const int r = ratio_;
  MultiFab& phi_m = phi_level(m);
  MultiFab& eps_m = eps_level(m);
  MultiFab& eps_y_m = has_eps_y_ ? eps_y_level(m) : eps_level(m);
  const BoxArray& child = (m + 1 == 1) ? ba_f_ : ba_lv_[(m + 1) - 2];
  MultiFab& phi_child = phi_level(m + 1);
  MultiFab& eps_child = eps_level(m + 1);
  MultiFab& eps_y_child = has_eps_y_ ? eps_y_level(m + 1) : eps_level(m + 1);
  // FluxRegister over the level-m grid: single-writer per (cell,direction) slot (4/cell), gather()
  // sums (x,0,0,..) exactly, then a fixed-order fold. Serial: gather() is the identity -> bit-identical.
  FluxRegister& reg = *flux_registers_[static_cast<std::size_t>(m)];
  reg.clear_on_device();
  const FluxRegisterView register_view = reg.view();
  for (int g = 0; g < child.size(); ++g) {
    // level-m footprint of child patch g.
    const Box2D pc = foot_of_[m + 1][g].empty() ? PatchRange(child[g]).box() : foot_of_[m + 1][g];
    // only the rank owning this child patch writes its slots (single writer). Serial: always local.
    const int li = phi_child.local_index_of(g);
    if (li < 0)
      continue;
    const ConstArray4 PF = phi_child.fab(li).const_array();
    const ConstArray4 EF = eps_child.fab(li).const_array();
    const ConstArray4 EYF = eps_y_child.fab(li).const_array();
    const Box2D coarse_need = pc.grow(1);
    const ConstArray4 PC = detail::readable_array_covering_(phi_m, coarse_need);
    const ConstArray4 EC = detail::readable_array_covering_(eps_m, coarse_need);
    const ConstArray4 EYC = detail::readable_array_covering_(eps_y_m, coarse_need);
    if (PC.p == nullptr || (he && (EC.p == nullptr || EYC.p == nullptr)))
      throw std::runtime_error(
          "CompositeFacPoisson: co-located coarse interface storage is missing");
    for_each_cell(coarse_need,
                  detail::FacFluxCorrectionKernel{PC, EC, EYC, PF, EF, EYF, coverage,
                                                  register_view, pc, idx2, idy2, r, he});
  }
  reg.gather();  // all_reduce_sum of single-writer slots -> exact; serial identity.
  // fold the 4 slots into dst in a FIXED direction order (xm, xp, ym, yp) -> np-invariant bits.
  const FluxRegisterView flux = reg.view();
  for (int li = 0; li < dst.local_size(); ++li) {
    Array4 D = dst.fab(li).array();
    const Box2D b = dst.box(li);
    for_each_cell(b, detail::FacFluxFoldKernel{D, coverage, flux});
  }
  device_fence();  // the local register owns the pinned storage captured by the fold kernels
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
    prepare_field_residual_view(phim, has_boundary_kernel_ ? &boundary_view_c_ : nullptr, gm, bc_,
                                has_boundary_kernel_ ? &boundary_kernel_ : nullptr,
                                has_boundary_kernel_ ? &boundary_context_ : nullptr);
  } else {
    if (m - 1 == 0)
      fill_ghosts(phi_c_, geom_c_.domain, bc_);
    fill_cf_phi_(m);
  }
  MultiFab& lap = lap_level_(m);
  MultiFab& operator_view = (m == 0 && has_boundary_kernel_) ? boundary_view_c_ : phim;
  apply_laplacian(operator_view, gm, lap, /*coef=*/nullptr, has_eps_ ? &eps_level(m) : nullptr,
                  /*kappa=*/nullptr, has_eps_y_ ? &eps_y_level(m) : nullptr,
                  has_cross_ ? &a_xy_level(m) : nullptr,
                  has_cross_ ? &a_yx_level(m) : nullptr);
  if (has_reaction_)
    apply_constant_reaction_(lap, operator_view);
  if (m + 1 == n_levels_) {
    Real nrm = Real(0);
    for (int li = 0; li < resm.local_size(); ++li) {
      Array4 R = resm.fab(li).array();
      const ConstArray4 LAP = lap.fab(li).const_array();
      const ConstArray4 FM = rhs_level(m).fab(li).const_array();
      const Box2D b = resm.box(li);
      for_each_cell(b, detail::FacFinestResidualKernel{R, FM, LAP});
      nrm = std::max(
          nrm, reduce_max_cell(b, detail::FacFinestNormInfKernel{resm.fab(li).const_array()}));
    }
    return Real(all_reduce_max(static_cast<double>(nrm)));
  }

  const CoverageMaskView coverage = cov_of_[m].view();
  for (int li = 0; li < resm.local_size(); ++li) {
    Array4 R = resm.fab(li).array();
    const ConstArray4 LAP = lap.fab(li).const_array();
    const ConstArray4 FM = rhs_level(m).fab(li).const_array();
    const Box2D b = resm.box(li);
    for_each_cell(b, detail::FacMaskedResidualKernel{R, FM, LAP, coverage});
  }
  add_flux_correction_(m, resm);  // += (coarse - fine) on the bordering cells

  Real nrm = Real(0);
  for (int li = 0; li < resm.local_size(); ++li) {
    const Box2D b = resm.box(li);
    nrm = std::max(nrm, reduce_max_cell(
                            b, detail::FacMaskedNormInfKernel{resm.fab(li).const_array(),
                                                              coverage}));
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
    fill_ghosts(phi_c_, geom_c_.domain,
                bc_);  // parent physical ghosts (bilerp reads to the border)
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
    // Every child is co-located with its unique parent (build_extra_levels_), and same-level
    // footprints never overlap. Write the conservative average directly into that parent: this is
    // the same AverageDownKernel arithmetic as average_down(), without its per-call scratch MultiFab.
    const Real inv = Real(1) / Real(ratio_ * ratio_);
    for (int li = 0; li < child.local_size(); ++li) {
      const ConstArray4 F = child.fab(li).const_array();
      const Box2D footprint = PatchRange(child.box(li)).box();
      const Array4 P = detail::writable_array_covering_(parent, footprint);
      if (P.p == nullptr)
        throw std::runtime_error(
            "CompositeFacPoisson: co-located parent missing during average-down");
      for_each_cell(footprint, detail::AverageDownKernel{F, P, inv, ratio_, /*component=*/0});
    }
    return;
  }
  // replicated coarse under MPI: single-writer FluxRegister over the coarse grid.
  FluxRegister& reg = *coarse_average_register_;
  reg.clear_on_device();
  const FluxRegisterView register_view = reg.view();
  const int r = ratio_;
  for (int li = 0; li < child.local_size(); ++li) {
    const ConstArray4 F = child.fab(li).const_array();
    const PatchRange pr(child.box(li));  // coarse footprint of this fine patch
    for_each_cell(pr.box(), detail::FacAverageDownRegisterKernel{F, register_view, r});
  }
  reg.gather();  // all_reduce_sum of single-writer slots -> exact; every rank now has the averages.
  Array4 P = parent.fab(0).array();  // replicated coarse (fab(0) on every rank)
  for_each_cell(parent.box(0), detail::FacCopyCoveredRegisterKernel{
                                   P, cov_of_[0].view(), register_view});
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
    mg_.solve(options_.coarse_rel_tol, options_.coarse_cycles, options_.coarse_abs_tol);
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
    auto broadcast = [&](MultiFab& rep, const MultiFab& dist) {
      rep.set_val(Real(0));
      // the owner writes its patch cells; others contribute 0.
      for (int li = 0; li < dist.local_size(); ++li) {
        Array4 R = rep.fab(0).array();
        const ConstArray4 S = dist.fab(li).const_array();
        const Box2D b = dist.box(li);
        for_each_cell(b, detail::FacCopyAllKernel{R, S, 0});
      }
      device_fence();  // MPI consumes the managed allocation from the host.
      all_reduce_sum_inplace(rep.fab(0).array().p,
                             static_cast<std::size_t>(rep.fab(0).size()));
    };
    MultiFab& res_rep = correction_residual_replicated_[static_cast<std::size_t>(m)];
    broadcast(res_rep, res_level_(m));
    if (has_eps_) {
      MultiFab& eps_rep = correction_eps_replicated_[static_cast<std::size_t>(m)];
      broadcast(eps_rep, eps_level(m));
      if (has_eps_y_) {
        MultiFab& eps_y_rep = correction_eps_y_replicated_[static_cast<std::size_t>(m)];
        broadcast(eps_y_rep, eps_y_level(m));
        mgk.set_epsilon_anisotropic(eps_rep, eps_y_rep);
      } else {
        mgk.set_epsilon(eps_rep);
      }
    }
    if (has_cross_) {
      MultiFab& axy_rep = correction_axy_replicated_[static_cast<std::size_t>(m)];
      MultiFab& ayx_rep = correction_ayx_replicated_[static_cast<std::size_t>(m)];
      broadcast(axy_rep, a_xy_level(m));
      broadcast(ayx_rep, a_yx_level(m));
      mgk.set_cross_terms(axy_rep, ayx_rep);
    }
    copy0_(mgk.rhs(), res_rep);
    mgk.phi().set_val(Real(0));
    mgk.solve(options_.coarse_rel_tol, options_.coarse_cycles, options_.coarse_abs_tol);
    // write the correction back to the distributed patch on its owner (add_uncovered on the local fab).
    const CoverageMaskView coverage = cov_of_[m].view();
    MultiFab& phim = phi_level(m);
    const ConstArray4 E = mgk.phi().fab(0).const_array();
    for (int li = 0; li < phim.local_size(); ++li) {
      Array4 P = phim.fab(li).array();
      const Box2D b = phim.box(li);
      for_each_cell(b, detail::FacMaskedAddKernel{P, E, coverage});
    }
    return;
  }
  // serial (or a distributed multi-patch mg): feed the level fields directly.
  if (has_eps_) {
    if (has_eps_y_)
      mgk.set_epsilon_anisotropic(eps_level(m), eps_y_level(m));
    else
      mgk.set_epsilon(eps_level(m));
  }
  if (has_cross_)
    mgk.set_cross_terms(a_xy_level(m), a_yx_level(m));
  copy0_(mgk.rhs(), res_level_(m));
  mgk.phi().set_val(Real(0));
  mgk.solve(options_.coarse_rel_tol, options_.coarse_cycles, options_.coarse_abs_tol);
  add_uncovered_level_(m, phi_level(m), mgk.phi());
}

// ------------------------------------------------------------------------------------------------
// solve_composite_nlevel_ : the general FAC driver (design 3f). At L == 2 it is a single-iteration
// loop that reduces algebraically to the legacy 2-level loop.
// ------------------------------------------------------------------------------------------------
inline CompositeFacPoisson::LinearSolveResult CompositeFacPoisson::solve_composite_nlevel_(
    int max_iters, int fine_sweeps, Real stop) {
  const int L = n_levels_;

  // 0) initial coarse solve (base level).
  copy0_(mg_.rhs(), f_c_);
  mg_.phi().set_val(Real(0));
  mg_.solve(options_.coarse_rel_tol, options_.coarse_cycles, options_.coarse_abs_tol);
  copy0_(phi_c_, mg_.phi());

  // 1) relax every patch level coarse-to-fine, then a fine-to-coarse average-down cascade (==
  //    refresh_fine at L == 2: fill_cf + fine_sor + average_down).
  for (int m = 1; m < L; ++m)
    relax_level_(m, fine_sweeps);
  cascade_avgdown_();

  diagnostics_.clear();
  Real rnorm = composite_residual_norm_(/*general=*/true, /*prepare_cf=*/false);
  record_residual(-1, rnorm);
  if (!two_way_) {
    last_residual_ = rnorm;
    return {rnorm, 0};
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
  int iterations = 0;
  for (int it = 0; it < max_iters; ++it) {
    if (!std::isfinite(static_cast<double>(rnorm)) || rnorm <= stop)
      break;
    for (int m = 0; m + 1 < L; ++m) {
      if (m > 0)
        composite_residual_(m);  // fresh level-m residual (res_c_ for m==0 is already current)
      correct_level_(m);
      relax_level_(m + 1, fine_sweeps);  // re-establish the finer level on the corrected parent
    }
    cascade_avgdown_();
    rnorm = composite_residual_norm_(/*general=*/true, /*prepare_cf=*/false);
    ++iterations;
    record_residual(it, rnorm);
  }
  last_residual_ = rnorm;
  return {rnorm, iterations};
}

}  // namespace pops
