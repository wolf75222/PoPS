#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/diagnostics/runtime_diagnostics.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/layout/refinement.hpp>  // average_down, coarsen_index
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>  // coarse solver (geometric multigrid)
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>  // apply_laplacian (residual, reads the already-filled ghosts)
#include <pops/numerics/time/amr/levels/amr_patch_range.hpp>  // PatchRange, CoverageMask (coarse footprint of a patch)
#include <pops/parallel/comm.hpp>  // my_rank / n_ranks (replicated coarse dmap, MPI dispatch + gather)
#include <pops/runtime/numerical_defaults.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

/// @file
/// @brief CompositeFacPoisson: AMR COMPOSITE elliptic solver (Fast Adaptive Composite, FAC) for
///        ``div(A grad phi) - kappa phi = f`` across a nested hierarchy. ``kappa=0`` is Poisson;
///        a positive constant is the screened-Poisson/Helmholtz route.
///
/// MOTIVATION (amr-schur path). The current AMR Poisson (Option A) solves the elliptic only on the
/// coarse level then injects grad phi (piecewise constant) onto the fine patches: the patches refine
/// the TRANSPORT but NOT the elliptic coupling. A COMPOSITE solver makes the fine patch ACTUALLY
/// REFINE the elliptic solution (more accurate phi/grad phi near the patch). This is the AMR fidelity
/// lock (the composite Poisson coupling (FAC) that amr_reflux.hpp explicitly leaves to this solver).
///
/// 2-LEVEL FAC ALGORITHM (McCormick), one fine patch INTERIOR to the coarse domain. Composite solution
/// phi = phi_f on the patch, phi_c elsewhere:
///   0. initial coarse solve: GeometricMG(Lap phi_c = f_c, Dirichlet);
///   it. repeat:
///      1. C-F ghosts: fill the patch ghost ring by BILINEAR INTERPOLATION of phi_c (order
///         2 vs the constant injection of Option A) -> cell-centered C-F Dirichlet condition;
///      2. fine solve: red-black GS on the patch with FROZEN ghosts (Lap phi_f = f_f);
///      3. average_down phi_f -> phi_c on the COVERED coarse cells (consistency);
///      4. composite coarse residual: r_c = f_c - Lap phi_c (NON covered cells), 0 on covered ones,
///         + C-F FLUX CORRECTION: on the coarse cells BORDERING the patch, the flux through the
///         C-F face is replaced by the FINE flux (conservative sum of the 2 fine faces) -> two-way coupling;
///      5. coarse correction: GeometricMG(Lap e_c = r_c, homogeneous Dirichlet); phi_c += e_c (non covered);
///   until ||r_c|| (composite residual norm) below tolerance.
///
/// SCOPE (ADC-636, generalized envelope). Cartesian, ratio 2, an arbitrary NESTED hierarchy: N
/// levels, 1..N fine patches per level, ADJACENT (edge/corner-touching) patches allowed, and MPI
/// (REPLICATED coarse + DISTRIBUTED fine). The 2-level non-adjacent mono-rank case is dispatched to
/// the VERBATIM legacy body (solve_two_level_legacy_) and is byte-for-byte unchanged; the general
/// path (composite_fac_nlevel.hpp) serves every other shape. Distributed equals replicated
/// bit-identically at fixed np by construction. Only ratio != 2 (ADC-602 declared capability) and
/// overlapping / non-nested / misaligned patches (semantically impossible) are refused.
///
/// MULTI-PATCH. Each fine patch has its own box (fine BoxArray); the FINE operations (bilinear C-F
/// ghosts, SOR, C-F flux correction) loop OVER EACH local patch. The coarse coverage (CoverageMask)
/// is the UNION of the coarse footprints of all patches: it tells which coarse cells are shadowed
/// (residual set to 0, average_down) and lets the flux correction skip a face covered on both sides.
/// Adjacent patches share a fine face: the fine-fine join is realized by fill_boundary before the C/F
/// bilerp (the shared ghost takes the sibling's valid data), and the two-way flux correction is
/// enumerated from the uncovered coarse side so a shared interior face gets no correction.

namespace pops {

namespace detail {

struct FacCopyAllKernel {
  Array4 dst;
  ConstArray4 src;
  int comp;
  POPS_HD void operator()(int i, int j) const { dst(i, j, comp) = src(i, j, comp); }
};

struct FacSetAllKernel {
  Array4 dst;
  Real value;
  int comp;
  POPS_HD void operator()(int i, int j) const { dst(i, j, comp) = value; }
};

struct FacApplyConstantReactionKernel {
  Array4 value;
  ConstArray4 phi;
  Real reaction;

  POPS_HD void operator()(int i, int j) const { value(i, j, 0) -= reaction * phi(i, j, 0); }
};

/// BILINEAR interpolation of the coarse potential (cell-centered, @p C with ghosts) at the CENTER of the
/// fine cell (i, j). Ratio @p r. The fine center has abscissa (i+0.5)/r in coarse-step units, i.e.
/// the coarse center-index fx = (i+0.5)/r - 0.5; we interpolate the 4 surrounding coarse centers.
/// INTERIOR patch -> Ic, Ic+1, Jc, Jc+1 are in the coarse domain (ghosts included).
POPS_HD inline Real fac_bilerp_coarse(const ConstArray4& C, int i, int j, int r) {
  const Real fx = (Real(i) + Real(0.5)) / Real(r) - Real(0.5);
  const Real fy = (Real(j) + Real(0.5)) / Real(r) - Real(0.5);
  const int Ic = static_cast<int>(std::floor(fx));
  const int Jc = static_cast<int>(std::floor(fy));
  const Real tx = fx - Real(Ic), ty = fy - Real(Jc);
  const Real c00 = C(Ic, Jc, 0), c10 = C(Ic + 1, Jc, 0);
  const Real c01 = C(Ic, Jc + 1, 0), c11 = C(Ic + 1, Jc + 1, 0);
  return (Real(1) - tx) * (Real(1) - ty) * c00 + tx * (Real(1) - ty) * c10 +
         (Real(1) - tx) * ty * c01 + tx * ty * c11;
}

struct FacLegacyFillCoarseFineGhostKernel {
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

struct FacLegacyMaskedAddKernel {
  Array4 destination;
  ConstArray4 correction;
  CoverageMaskView coverage;

  POPS_HD void operator()(int i, int j) const {
    if (!coverage.covered(i, j))
      destination(i, j, 0) += correction(i, j, 0);
  }
};

struct FacLegacySorKernel {
  Array4 phi;
  ConstArray4 phi_read;
  ConstArray4 rhs;
  ConstArray4 eps;
  ConstArray4 eps_y;
  ConstArray4 a_xy;
  ConstArray4 a_yx;
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
    if (cell_color != color)
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

struct FacLegacyMaskedResidualKernel {
  Array4 residual;
  ConstArray4 rhs;
  ConstArray4 laplacian;
  CoverageMaskView coverage;

  POPS_HD void operator()(int i, int j) const {
    residual(i, j, 0) =
        coverage.covered(i, j) ? Real(0) : rhs(i, j, 0) - laplacian(i, j, 0);
  }
};

struct FacLegacyMaskedNormKernel {
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

struct FacLegacyFluxCorrectionKernel {
  Array4 residual;
  ConstArray4 coarse_phi;
  ConstArray4 coarse_eps;
  ConstArray4 coarse_eps_y;
  ConstArray4 fine_phi;
  ConstArray4 fine_eps;
  ConstArray4 fine_eps_y;
  CoverageMaskView coverage;
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
      residual(i, j, 0) += coarse_face - fine_sum * idx2;
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
      residual(i, j, 0) += coarse_face - fine_sum * idx2;
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
      residual(i, j, 0) += coarse_face - fine_sum * idy2;
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
      residual(i, j, 0) += coarse_face - fine_sum * idy2;
    }
  }
};

static_assert(std::is_trivially_copyable_v<FacLegacyFillCoarseFineGhostKernel>);
static_assert(std::is_trivially_copyable_v<FacLegacyMaskedAddKernel>);
static_assert(std::is_trivially_copyable_v<FacLegacySorKernel>);
static_assert(std::is_trivially_copyable_v<FacLegacyMaskedResidualKernel>);
static_assert(std::is_trivially_copyable_v<FacLegacyMaskedNormKernel>);
static_assert(std::is_trivially_copyable_v<FacLegacyFluxCorrectionKernel>);

}  // namespace detail

/// Composite FAC Poisson/Helmholtz solver (scalar). Built on the coarse layout (replicated mono-box)
/// + the fine patch (mono-box). The caller provides f_c (coarse) and f_f (fine); the solver returns
/// phi_c (coarse, covered = average_down of the fine) and phi_f (fine).
class CompositeFacPoisson {
 public:
  /// MONO-PATCH CTOR (Phase 1): DELEGATES to the multi-patch ctor with a fine BoxArray of a single box, so
  /// BIT-IDENTICAL to the old path. Kept for existing callers (AmrCouplerMP Option-A composite,
  /// mono-patch MMS tests).
  /// @p geom_c: coarse geometry (whole domain). @p ba_c: coarse BoxArray (mono-box covering
  ///             the domain). @p bc: domain BC (Dirichlet for this milestone). @p fine_box: box of the
  ///             fine patch (FINE index space, ratio 2, strictly interior). @p ratio: 2.
  CompositeFacPoisson(const Geometry& geom_c, const BoxArray& ba_c, const BCRec& bc,
                      const Box2D& fine_box, int ratio = 2)
      : CompositeFacPoisson(geom_c, ba_c, bc, BoxArray(std::vector<Box2D>{fine_box}), ratio) {}

  /// MULTI-PATCH CTOR (Phase 4a). @p fine_boxes: tiling of the fine level (1..N disjoint patches, FINE
  /// index space, ratio 2, strictly interior, aligned lo even / hi odd, SEPARATED by at least one
  /// coarse cell). The coarse stays replicated mono-box (single-rank). N == 1 -> mono-patch path.
  CompositeFacPoisson(const Geometry& geom_c, const BoxArray& ba_c, const BCRec& bc,
                      const BoxArray& fine_boxes, int ratio = 2)
      : geom_c_(geom_c),
        geom_f_(geom_c.refine(ratio)),
        ba_c_(ba_c),
        // REPLICATED coarse (ADC-636): the mono-box coarse lives on EVERY rank (each rank owns
        // fab(0)), which is what all the .fab(0) coarse reads assume and what GeometricMG(replicated)
        // expects. At np=1 my_rank()==0 -> identical to the historical round-robin (size, n_ranks())
        // that also placed the single box on rank 0: MONO-RANK bit-identical.
        dm_c_(std::vector<int>(static_cast<std::size_t>(ba_c.size()), my_rank())),
        bc_(bc),
        ratio_(ratio),
        ba_f_(fine_boxes),
        dm_f_(fine_boxes.size(), n_ranks()),
        mg_(geom_c, ba_c, bc, {}, FieldDistribution::Replicated),
        phi_c_(ba_c, dm_c_, 1, 1),
        phi_f_(ba_f_, dm_f_, 1, 1),
        f_c_(ba_c, dm_c_, 1, 0),
        f_f_(ba_f_, dm_f_, 1, 0),
        res_c_(ba_c, dm_c_, 1, 0),
        lap_c_(ba_c, dm_c_, 1, 0),
        lap_f_(ba_f_, dm_f_, 1, 0),
        boundary_view_c_(ba_c, dm_c_, 1, 1),
        eps_c_(ba_c, dm_c_, 1, 1),
        eps_f_(ba_f_, dm_f_, 1, 1),
        eps_y_c_(ba_c, dm_c_, 1, 1),
        eps_y_f_(ba_f_, dm_f_, 1, 1),
        axy_c_(ba_c, dm_c_, 1, 1),
        ayx_c_(ba_c, dm_c_, 1, 1),
        axy_f_(ba_f_, dm_f_, 1, 1),
        ayx_f_(ba_f_, dm_f_, 1, 1),
        cov_(Box2D::from_extents(geom_c.domain.nx(), geom_c.domain.ny())) {
    // ADC-636: validate the level-1 patches (aligned lo-even/hi-odd, non-overlapping) and DETECT
    // adjacency. Adjacent (edge/corner-touching) patches are now legal -- the fine-fine join is
    // handled by fill_boundary before the C/F bilerp with an uncovered-side flux ownership rule
    // (composite_fac_nlevel.hpp), and a touching hierarchy routes solve() to the general path. Only
    // overlapping/misaligned patches are refused (validate_level_patches_); inter-level nesting is
    // checked by the N-level ctor (validate_nesting_).
    validate_level_patches_(fine_boxes);
    // coarse footprints (covered cells) PER PATCH: PatchRange (lo/2 .. (hi-1)/2). The global coarse
    // coverage = UNION of the footprints (any gap between disjoint patches stays NON covered).
    for (int g = 0; g < fine_boxes.size(); ++g)
      patch_coarse_.push_back(PatchRange(fine_boxes[g]).box());
    for (const Box2D& pc : patch_coarse_)
      cov_.mark(pc);
    phi_c_.set_val(Real(0));
    phi_f_.set_val(Real(0));
    eps_c_.set_val(Real(1));  // default permittivity 1 -> operator = Laplacian (scalar)
    eps_f_.set_val(Real(1));
    eps_y_c_.set_val(Real(1));
    eps_y_f_.set_val(Real(1));
    axy_c_.set_val(Real(0));  // default cross terms 0 -> diagonal block only
    ayx_c_.set_val(Real(0));
    axy_f_.set_val(Real(0));
    ayx_f_.set_val(Real(0));
    // The finest level participates in the scientific composite residual even for a two-level
    // hierarchy. Deeper constructors reuse this buffer as level 1's residual storage.
    res_f_ = MultiFab(ba_f_, dm_f_, 1, 0);
    res_f_.set_val(Real(0));
    // ADC-636: build the uniform per-level metadata (coverage / footprints / intermediate mg) so the
    // general path is reachable for a 2-level input too (the cross-check hook and the MPI path). The
    // legacy 2-level dispatch does not use it -- it keeps cov_/patch_coarse_ as before, untouched.
    finalize_hierarchy_metadata_();
    initialize_probe_storage_();
  }

  /// N-LEVEL CTOR (ADC-636). @p level_boxes[k] = the fine BoxArray of level k+1 (in that level's index
  /// space, ratio 2 over level k), so level_boxes[0] = the level-1 patches (== the 2-level fine_boxes),
  /// level_boxes[1] = the level-2 patches, ... The 2-level ctor is the level_boxes.size() == 1 case;
  /// for it this DELEGATES to the multi-patch ctor above (identical level-0/1 allocation), so a
  /// single-patch-level hierarchy stays bit-identical. For deeper hierarchies the extra levels are
  /// allocated here (geom refined per level, per-level coverage and parent footprints).
  CompositeFacPoisson(const Geometry& geom_c, const BoxArray& ba_c, const BCRec& bc,
                      const std::vector<BoxArray>& level_boxes, int ratio = 2)
      : CompositeFacPoisson(geom_c, ba_c, bc,
                            level_boxes.empty() ? BoxArray(std::vector<Box2D>{}) : level_boxes[0],
                            ratio) {
    if (level_boxes.empty())
      throw std::runtime_error(
          "CompositeFacPoisson: the N-level ctor needs at least one patch level (level_boxes "
          "non-empty).");
    n_levels_ = 1 + static_cast<int>(level_boxes.size());
    build_extra_levels_(level_boxes);
    validate_nesting_(level_boxes);  // refuse non-nested patches (C/F bilerp source undefined)
    initialize_probe_storage_();
  }

  MultiFab& rhs_coarse() {
    return f_c_;
  }  ///< coarse RHS for div(eps grad phi_c) - kappa phi_c = f_c
  MultiFab& rhs_fine() { return f_f_; }  ///< fine RHS for the same composite operator
  MultiFab& phi_coarse() { return phi_c_; }
  MultiFab& phi_fine() { return phi_f_; }
  /// VARIABLE permittivity eps (at cell centers) PER LEVEL. Fill + use_variable_coefficient(true)
  /// to go from Lap phi = f to div(eps grad phi) = f -- the condensed Schur operator at B_z = 0
  /// (eps = 1 + theta^2 dt^2 alpha rho). eps unfilled / not enabled -> scalar (Phase 1), bit-identical.
  MultiFab& eps_coarse() { return eps_c_; }
  MultiFab& eps_fine() { return eps_f_; }
  void use_variable_coefficient(bool v) { has_eps_ = v; }
  /// Install a spatially uniform reaction coefficient. The internal FAC operator is
  /// ``div(eps grad phi) - kappa phi``; public ``-div(eps grad phi) + kappa phi`` equations negate
  /// their RHS at the runtime boundary. The same kappa is installed on every correction MG and in
  /// every composite residual/SOR level, so refined solves remain one exact Helmholtz operator.
  void set_reaction(Real reaction) {
    if (!std::isfinite(static_cast<double>(reaction)) || reaction <= Real(0))
      throw std::invalid_argument(
          "CompositeFacPoisson reaction must be finite and strictly positive");
    reaction_ = reaction;
    has_reaction_ = true;
    mg_.set_reaction(constant_scalar_field_provider(reaction));
    if (fully_refined_solver_)
      fully_refined_solver_->set_reaction(constant_scalar_field_provider(reaction));
    for (auto& level : level_mg_)
      if (level)
        level->set_reaction(constant_scalar_field_provider(reaction));
  }
  /// Cross terms a_xy / a_yx (at cell centers) PER LEVEL: FULL tensor A = diag(eps,eps) +
  /// [[0,a_xy],[a_yx,0]]. This is the condensed Schur operator at B_z != 0 (a_xy = c rho w/det,
  /// a_yx = -a_xy, w = theta dt B_z) -- antisymmetric, NON self-adjoint. Small for the Schur step
  /// (c = theta^2 dt^2 alpha) -> convergent SOR/V-cycle (EXPLICIT cross terms). Not enabled -> diagonal
  /// block only (Phase 3a/1), bit-identical. Requires use_variable_coefficient(true) (the diagonal block).
  MultiFab& a_xy_coarse() { return axy_c_; }
  MultiFab& a_yx_coarse() { return ayx_c_; }
  MultiFab& a_xy_fine() { return axy_f_; }
  MultiFab& a_yx_fine() { return ayx_f_; }
  void use_cross_terms(bool v) { has_cross_ = v; }
  /// Coarse footprint of the FIRST fine patch (mono-patch compat). Multi-patch: see patch_coarse(g).
  const Box2D& patch_coarse() const { return patch_coarse_[0]; }
  /// Coarse footprint of fine patch @p g (0 <= g < n_fine_patches()).
  const Box2D& patch_coarse(int g) const { return patch_coarse_[g]; }
  /// Number of fine patches (size of the fine BoxArray).
  int n_fine_patches() const { return ba_f_.size(); }
  /// Number of levels in the composite hierarchy. The historical 2-level ctors give 2; the N-level
  /// ctor (ADC-636, composite_fac_nlevel.hpp) gives 1 + number of patch levels.
  int n_levels() const { return n_levels_; }

  /// N-LEVEL ACCESSORS (ADC-636). Uniform field access by level index k (0 = coarse, 1 = first patch
  /// level, ...). The 2-level accessors above alias _level(0)/_level(1) so callers can use either. For
  /// k >= 2 the fields live in the per-level vectors allocated by the N-level ctor.
  MultiFab& rhs_level(int k) { return k == 0 ? f_c_ : (k == 1 ? f_f_ : f_lv_[k - 2]); }
  MultiFab& phi_level(int k) { return k == 0 ? phi_c_ : (k == 1 ? phi_f_ : phi_lv_[k - 2]); }
  MultiFab& eps_level(int k) { return k == 0 ? eps_c_ : (k == 1 ? eps_f_ : eps_lv_[k - 2]); }
  MultiFab& eps_y_level(int k) {
    return k == 0 ? eps_y_c_ : (k == 1 ? eps_y_f_ : eps_y_lv_[k - 2]);
  }
  void use_anisotropic_coefficient(bool value) {
    has_eps_y_ = value;
    has_eps_ = has_eps_ || value;
  }
  MultiFab& a_xy_level(int k) { return k == 0 ? axy_c_ : (k == 1 ? axy_f_ : axy_lv_[k - 2]); }
  MultiFab& a_yx_level(int k) { return k == 0 ? ayx_c_ : (k == 1 ? ayx_f_ : ayx_lv_[k - 2]); }
  /// Geometry of level k (k == 0 coarse, k == 1 fine, k >= 2 refined 2^k over the coarse).
  const Geometry& geom_level(int k) const {
    return k == 0 ? geom_c_ : (k == 1 ? geom_f_ : geom_lv_[k - 2]);
  }

  void set_verbose(bool v) { verbose_ = v; }
  const RuntimeDiagnosticsReport& diagnostics_report() const { return diagnostics_; }
  void reset_diagnostics() { diagnostics_.clear(); }
  /// true: iterate the FAC two-way coupling (C-F flux correction + coarse correction). false:
  /// ONE-WAY path (coarse solve + fine solve with bilinear C-F ghosts) -- the patch refines locally.
  void set_two_way(bool v) { two_way_ = v; }

  /// Install the composite-FAC knobs (outer iterations / fine sweeps / mixed relative+absolute
  /// composite stop / internal coarse GeometricMG rel_tol+cycles / verbose).
  void set_options(const CompositeFacOptions& o) {
    options_ = o;
    verbose_ = o.verbose;
  }
  const CompositeFacOptions& options() const { return options_; }

  void set_boundary_kernel(const CompiledFieldBoundaryKernel& kernel,
                           const FieldBoundaryExecutionContext& context = {}) {
    kernel.validate();
    boundary_kernel_ = kernel;
    boundary_context_ = context;
    boundary_context_.failure = &boundary_failure_;
    has_boundary_kernel_ = true;
    mg_.set_boundary_kernel(boundary_kernel_, boundary_context_);
    if (fully_refined_solver_)
      fully_refined_solver_->set_boundary_kernel(boundary_kernel_, boundary_context_);
  }

  void set_boundary_context(const FieldBoundaryExecutionContext& context) {
    if (!has_boundary_kernel_)
      throw std::runtime_error("CompositeFacPoisson boundary context has no installed kernel");
    boundary_context_ = context;
    boundary_context_.failure = &boundary_failure_;
    if (!boundary_kernel_.observes_iteration)
      boundary_context_.point.iteration = 0;
    mg_.set_boundary_context(boundary_context_);
    if (fully_refined_solver_)
      fully_refined_solver_->set_boundary_context(boundary_context_);
  }

  void set_field_nonlinear_options(const FieldNewtonOptions& options) {
    validate_field_newton_options(options);
    field_nonlinear_options_ = options;
    has_field_nonlinear_options_ = true;
  }

  const SolveReport& last_solve_report() const { return last_solve_report_; }

  /// Solves the composite system with the installed mixed-tolerance options. Failed linear solves do
  /// not publish a value through this throwing convenience overload; callers that consume a
  /// SolveReport directly use the explicit overload below.
  Real solve() {
    if (has_boundary_kernel_ && boundary_kernel_.observes_iteration) {
      if (!has_field_nonlinear_options_)
        throw std::runtime_error(
            "iterate-dependent composite field boundary requires a nonlinear FAS outer plan");
      last_solve_report_ = solve_boundary_fas(field_nonlinear_options_);
      if (!last_solve_report_.solved())
        throw std::runtime_error(std::string("field FAS solve failed: ") +
                                 last_solve_report_.status_name());
      return last_residual_;
    }
    const Real result =
        solve(options_.max_iters, options_.fine_sweeps, options_.rel_tol, options_.abs_tol);
    if (!last_solve_report_.solved()) {
      throw std::runtime_error(std::string("field composite solve failed: ") +
                               last_solve_report_.status_name());
    }
    return result;
  }

  /// Solves the composite system. @return the final composite infinity-norm residual.
  /// Stops at max(@p rel_tol * ||R(0)||inf, @p abs_tol), where R(0) is evaluated through the exact
  /// composite operator on every active level with the installed BCs, masks and coefficients. The
  /// report denominator is ||R(0)||inf for every nonzero forcing, however small, and one only when it
  /// is exactly zero. @p max_iters counts FAC two-way iterations; @p fine_sweeps is per fine solve.
  ///
  /// DISPATCH (ADC-636). The 2-level, NON adjacent, MONO-RANK envelope routes to the VERBATIM legacy
  /// body (solve_two_level_legacy_ below) -- same bytes, hence same bits, gated by the golden. Every
  /// genuinely new shape (N > 2 levels, adjacent fine patches, or n_ranks() > 1) routes to the general
  /// FAC (solve_composite_nlevel_, composite_fac_nlevel.hpp). At L == 2 / non-adjacent / mono-rank the
  /// general path reduces algebraically to the legacy loop (cross-checked, not gated on).
  Real solve(int max_iters, int fine_sweeps, Real rel_tol, Real abs_tol) {
    if (max_iters < 0 || fine_sweeps < 0)
      throw std::invalid_argument("CompositeFacPoisson iteration budgets must be nonnegative");
    if (rel_tol < Real(0) || !std::isfinite(static_cast<double>(rel_tol)))
      throw std::invalid_argument("CompositeFacPoisson rel_tol must be finite and nonnegative");
    if (abs_tol < Real(0) || !std::isfinite(static_cast<double>(abs_tol)))
      throw std::invalid_argument("CompositeFacPoisson abs_tol must be finite and nonnegative");

    last_solve_report_ = {};
    diagnostics_.clear();
    if (fully_refined_solver_)
      return solve_fully_refined_hierarchy_(max_iters, rel_tol, abs_tol);
    const bool fallible_linear_boundary =
        has_boundary_kernel_ && !boundary_kernel_.observes_iteration;
    if (fallible_linear_boundary)
      boundary_failure_.reset();
    const bool general = force_general_ || n_levels_ != 2 || adjacent_ || n_ranks() != 1;
    if (general)
      setup_level_coeffs_();
    else
      setup_two_level_coeffs_();

    const Real forcing_norm = exact_zero_composite_residual_(general);
    if (fallible_linear_boundary && boundary_failure_.synchronize_across_ranks())
      throw std::runtime_error("composite field boundary evaluation failed at face " +
                               std::to_string(boundary_failure_.face) + " cell (" +
                               std::to_string(boundary_failure_.i) + "," +
                               std::to_string(boundary_failure_.j) + ")");
    if (!std::isfinite(static_cast<double>(forcing_norm)))
      return mark_invalid_linear_solve_(0);

    const Real report_denominator = forcing_norm == Real(0) ? Real(1) : forcing_norm;
    const Real relative_stop = rel_tol * forcing_norm;
    if (!std::isfinite(static_cast<double>(relative_stop)))
      return mark_invalid_linear_solve_(0);
    const Real stop = relative_stop > abs_tol ? relative_stop : abs_tol;

    // Respect a converged incoming composite iterate without mutating its valid cells. This matters
    // for repeated solves and for caller-provided level guesses; the denominator remains R(0), not the
    // warm-start defect.
    if (fallible_linear_boundary)
      boundary_failure_.reset();
    const Real incoming_residual = composite_residual_norm_(general, /*prepare_cf=*/true);
    if (fallible_linear_boundary && boundary_failure_.synchronize_across_ranks())
      throw std::runtime_error("composite field boundary evaluation failed at face " +
                               std::to_string(boundary_failure_.face) + " cell (" +
                               std::to_string(boundary_failure_.i) + "," +
                               std::to_string(boundary_failure_.j) + ")");
    if (!std::isfinite(static_cast<double>(incoming_residual)))
      return mark_invalid_linear_solve_(0);
    record_residual(-1, incoming_residual);
    if (incoming_residual <= stop) {
      last_residual_ = incoming_residual;
      last_solve_report_.iters = 0;
      last_solve_report_.rel_residual = incoming_residual / report_denominator;
      if (!std::isfinite(static_cast<double>(last_solve_report_.rel_residual)))
        return mark_invalid_linear_solve_(0);
      last_solve_report_.mark_solved();
      return incoming_residual;
    }

    if (fallible_linear_boundary)
      boundary_failure_.reset();
    const LinearSolveResult outcome = general
                                          ? solve_composite_nlevel_(max_iters, fine_sweeps, stop)
                                          : solve_two_level_legacy_(max_iters, fine_sweeps, stop);
    if (fallible_linear_boundary && boundary_failure_.synchronize_across_ranks())
      throw std::runtime_error("composite field boundary evaluation failed at face " +
                               std::to_string(boundary_failure_.face) + " cell (" +
                               std::to_string(boundary_failure_.i) + "," +
                               std::to_string(boundary_failure_.j) + ")");
    if (!std::isfinite(static_cast<double>(outcome.residual)))
      return mark_invalid_linear_solve_(outcome.iterations);

    last_residual_ = outcome.residual;
    last_solve_report_.iters = outcome.iterations;
    last_solve_report_.rel_residual = outcome.residual / report_denominator;
    if (!std::isfinite(static_cast<double>(last_solve_report_.rel_residual)))
      return mark_invalid_linear_solve_(outcome.iterations);
    if (outcome.residual <= stop)
      last_solve_report_.mark_solved();
    else
      last_solve_report_.mark_failed(SolveStatus::kIterationLimit, SolveAction::kRejectAttempt);
    return outcome.residual;
  }

  /// TEST HOOK (ADC-636): route a 2-level non-adjacent mono-rank input through the GENERAL path so the
  /// cross-check test can assert general == legacy (array_equal). Never set in production; the
  /// shipping 2-level path always dispatches to the verbatim legacy body.
  void force_general_path_for_test(bool v) { force_general_ = v; }

 private:
  struct LinearSolveResult {
    Real residual;
    int iterations;
  };

  void setup_two_level_coeffs_() {
    if (has_eps_) {
      device_fence();
      fill_ghosts(eps_c_, geom_c_.domain, coeff_bc(bc_));
      fill_cf_coarse_to_fine(eps_c_, eps_f_);
      if (has_eps_y_) {
        fill_ghosts(eps_y_c_, geom_c_.domain, coeff_bc(bc_));
        fill_cf_coarse_to_fine(eps_y_c_, eps_y_f_);
        mg_.set_epsilon_anisotropic(eps_c_, eps_y_c_);
      } else {
        mg_.set_epsilon(eps_c_);
      }
    }
    if (has_cross_) {
      device_fence();
      fill_ghosts(axy_c_, geom_c_.domain, coeff_bc(bc_));
      fill_ghosts(ayx_c_, geom_c_.domain, coeff_bc(bc_));
      fill_cf_coarse_to_fine(axy_c_, axy_f_);
      fill_cf_coarse_to_fine(ayx_c_, ayx_f_);
      mg_.set_cross_terms(axy_c_, ayx_c_);
    }
  }

  Real composite_residual_norm_(bool general, bool prepare_cf) {
    // FAC's scientific residual is the level-0 composite defect: the uncovered coarse residual plus
    // the conservative flux replacement from every child interface.  This is the defect the FAC
    // correction actually solves.  Fine-level equations are relaxation subproblems; folding their
    // transient post-smoothing defects into the outer stop would change the algorithm into a
    // different (and non-contractive) iteration after the composite defect has converged.
    // A zero-probe or an arbitrary incoming iterate needs its derived C/F ghosts prepared. During
    // FAC itself, refresh_fine()/relax_level_() have already frozen those ghosts before smoothing;
    // rebuilding them after average-down would change the interface flux seen by the correction and
    // turns the historical contractive iteration into a divergent one.
    if (prepare_cf) {
      fill_ghosts(phi_c_, geom_c_.domain, bc_);
      if (general) {
        for (int level = 1; level < n_levels_; ++level)
          fill_cf_phi_(level);
      } else {
        fill_cf_ghosts();
      }
    }
    const Real norm = general ? composite_residual_(0) : composite_coarse_residual();
    return std::isfinite(static_cast<double>(norm)) ? norm : std::numeric_limits<Real>::infinity();
  }

  static void copy_all_cells_(MultiFab& dst, const MultiFab& src) {
    for (int li = 0; li < dst.local_size(); ++li) {
      Array4 d = dst.fab(li).array();
      const ConstArray4 s = src.fab(li).const_array();
      const Box2D grown = dst.fab(li).grown_box();
      for (int comp = 0; comp < dst.ncomp(); ++comp)
        for_each_cell(grown, detail::FacCopyAllKernel{d, s, comp});
    }
  }

  static void set_all_cells_(MultiFab& dst, Real value) {
    for (int li = 0; li < dst.local_size(); ++li) {
      Array4 d = dst.fab(li).array();
      const Box2D grown = dst.fab(li).grown_box();
      for (int comp = 0; comp < dst.ncomp(); ++comp)
        for_each_cell(grown, detail::FacSetAllKernel{d, value, comp});
    }
  }

  void initialize_probe_storage_() {
    phi_probe_snapshot_.clear();
    phi_probe_snapshot_.reserve(static_cast<std::size_t>(n_levels_));
    for (int level = 0; level < n_levels_; ++level) {
      MultiFab& phi = phi_level(level);
      phi_probe_snapshot_.emplace_back(phi.box_array(), phi.dmap(), phi.ncomp(), phi.n_grow());
    }
    boundary_probe_snapshot_ = MultiFab(ba_c_, dm_c_, 1, boundary_view_c_.n_grow());
    phi_published_snapshot_.clear();
    phi_published_snapshot_.reserve(static_cast<std::size_t>(n_levels_));
    for (int level = 0; level < n_levels_; ++level) {
      MultiFab& phi = phi_level(level);
      phi_published_snapshot_.emplace_back(phi.box_array(), phi.dmap(), phi.ncomp(), phi.n_grow());
    }
  }

  Real exact_zero_composite_residual_(bool general) {
    for (int level = 0; level < n_levels_; ++level) {
      MultiFab& phi = phi_level(level);
      copy_all_cells_(phi_probe_snapshot_[static_cast<std::size_t>(level)], phi);
      set_all_cells_(phi, Real(0));
    }
    if (has_boundary_kernel_)
      copy_all_cells_(boundary_probe_snapshot_, boundary_view_c_);
    auto restore = [&]() {
      for (int level = 0; level < n_levels_; ++level)
        copy_all_cells_(phi_level(level), phi_probe_snapshot_[static_cast<std::size_t>(level)]);
      if (has_boundary_kernel_)
        copy_all_cells_(boundary_view_c_, boundary_probe_snapshot_);
    };
    try {
      const Real norm = composite_residual_norm_(general, /*prepare_cf=*/true);
      restore();
      device_fence();
      return norm;
    } catch (...) {
      restore();
      device_fence();
      throw;
    }
  }

  Real mark_invalid_linear_solve_(int iterations) {
    last_residual_ = std::numeric_limits<Real>::infinity();
    last_solve_report_.iters = iterations;
    last_solve_report_.rel_residual = std::numeric_limits<Real>::infinity();
    last_solve_report_.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kRejectAttempt);
    return last_residual_;
  }

  /// VERBATIM historical 2-level FAC driver (moved unchanged from solve(); ADC-636 dispatch). This is
  /// the non-regression anchor: the 2-level non-adjacent mono-rank path executes exactly these bytes.
  LinearSolveResult solve_two_level_legacy_(int max_iters, int fine_sweeps, Real stop) {
    // 0) initial coarse solve (gives a phi_c for the 1st C-F ghost). ADC-614: the internal coarse
    // GeometricMG rel_tol / max_cycles come from the installed options (default = kFAC* constants).
    copy0(mg_.rhs(), f_c_);
    mg_.phi().set_val(Real(0));
    mg_.solve(options_.coarse_rel_tol, options_.coarse_cycles, options_.coarse_abs_tol);
    copy0(phi_c_, mg_.phi());

    // 1) bilinear C-F ghosts + fine solve (base ONE-WAY).
    refresh_fine(fine_sweeps);

    diagnostics_.clear();
    Real rnorm = composite_residual_norm_(/*general=*/false, /*prepare_cf=*/false);
    record_residual(-1, rnorm);
    if (!two_way_) {
      last_residual_ = rnorm;
      return {rnorm, 0};
    }

    // 2) FAC two-way iterations: coarse correction (C-F flux) then re-solve fine.
    int iterations = 0;
    for (int it = 0; it < max_iters; ++it) {
      if (!std::isfinite(static_cast<double>(rnorm)) || rnorm <= stop)
        break;
      // coarse correction: Lap e_c = r_c (homogeneous Dirichlet), phi_c += e_c (non covered). The
      // correction solve uses the SAME internal coarse tolerance/cycles as the initial solve (ADC-614).
      copy0(mg_.rhs(), res_c_);
      mg_.phi().set_val(Real(0));
      mg_.solve(options_.coarse_rel_tol, options_.coarse_cycles, options_.coarse_abs_tol);
      add_uncovered(phi_c_, mg_.phi());
      // re-ghost + re-solve fine on the corrected phi_c.
      refresh_fine(fine_sweeps);
      rnorm = composite_residual_norm_(/*general=*/false, /*prepare_cf=*/false);
      ++iterations;
      record_residual(it, rnorm);
    }
    last_residual_ = rnorm;
    return {rnorm, iterations};
  }

 public:
  Real last_residual() const { return last_residual_; }

  /// Nonlinear full-approximation outer loop for iterate-dependent physical closures.  The FAC
  /// hierarchy is the nonlinear preconditioner/correction engine; only a converged hierarchy is
  /// published.  Every failed evaluation, iteration limit or exception restores the immutable
  /// pre-solve snapshots on all levels.
  SolveReport solve_boundary_fas(const FieldNewtonOptions& nonlinear) {
    if (!has_boundary_kernel_ || !boundary_kernel_.observes_iteration)
      return SolveReport::capability_failure();
    validate_field_newton_options(nonlinear);
    for (int level = 0; level < n_levels_; ++level) {
      MultiFab& phi = phi_level(level);
      copy_all_cells_(phi_published_snapshot_[static_cast<std::size_t>(level)], phi);
    }
    auto restore = [&]() {
      for (int level = 0; level < n_levels_; ++level)
        copy_all_cells_(phi_level(level), phi_published_snapshot_[static_cast<std::size_t>(level)]);
    };

    SolveReport report;
    Real base = Real(1);
    try {
      for (int iteration = 0; iteration < nonlinear.max_iterations; ++iteration) {
        boundary_context_.point.iteration = iteration;
        mg_.set_boundary_context(boundary_context_);
        boundary_failure_.reset();
        const Real residual =
            solve(options_.max_iters, options_.fine_sweeps, options_.rel_tol, options_.abs_tol);
        const bool failed = boundary_failure_.synchronize_across_ranks();
        if (failed || !std::isfinite(static_cast<double>(residual))) {
          report.iters = iteration + 1;
          report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kRejectAttempt);
          restore();
          return report;
        }
        if (iteration == 0)
          base = residual > Real(0) ? residual : Real(1);
        report.iters = iteration + 1;
        report.rel_residual = residual / base;
        last_residual_ = residual;
        if (residual <= nonlinear.tolerance * base) {
          report.mark_solved();
          return report;
        }
      }
    } catch (...) {
      restore();
      throw;
    }
    report.mark_failed(SolveStatus::kIterationLimit, SolveAction::kRejectAttempt);
    restore();
    return report;
  }

 private:
  /// dst <- src (component 0, valid cells).
  void copy0(MultiFab& dst, const MultiFab& src) {
    device_fence();
    for (int li = 0; li < dst.local_size(); ++li) {
      Array4 d = dst.fab(li).array();
      const ConstArray4 s = src.fab(li).const_array();
      const Box2D b = dst.box(li);
      for_each_cell(b, detail::FacCopyAllKernel{d, s, 0});
    }
  }

  /// phi_c += e_c on the NON covered cells (the correction does not touch the covered = average_down).
  void add_uncovered(MultiFab& phi, const MultiFab& e) {
    const CoverageMaskView coverage = cov_.view();
    for (int li = 0; li < phi.local_size(); ++li) {
      Array4 p = phi.fab(li).array();
      const ConstArray4 ec = e.fab(li).const_array();
      const Box2D b = phi.box(li);
      for_each_cell(b, detail::FacLegacyMaskedAddKernel{p, ec, coverage});
    }
  }

  /// ADC-636 ctor validation of a level's patch tiling. Refuses only the semantically impossible:
  /// MISALIGNED patches (not lo-even / hi-odd under ratio 2) and OVERLAPPING same-level patches
  /// (footprints intersect). Adjacent (edge/corner-touching) patches are ALLOWED and set adjacent_ so
  /// solve() takes the general path (fill_boundary fine-fine join + uncovered-side flux ownership).
  void validate_level_patches_(const BoxArray& boxes) {
    const int N = boxes.size();
    for (int g = 0; g < N; ++g) {
      const Box2D& fb = boxes[g];
      if ((fb.lo[0] % ratio_) != 0 || (fb.lo[1] % ratio_) != 0 || ((fb.hi[0] + 1) % ratio_) != 0 ||
          ((fb.hi[1] + 1) % ratio_) != 0)
        throw std::runtime_error(
            "CompositeFacPoisson: misaligned fine patch (require lo even / hi odd under ratio 2).");
    }
    for (int g = 0; g < N; ++g) {
      const Box2D ag = PatchRange(boxes[g]).box();
      for (int h = g + 1; h < N; ++h) {
        const Box2D bh = PatchRange(boxes[h]).box();
        if (!ag.intersect(bh).empty())
          throw std::runtime_error(
              "CompositeFacPoisson: overlapping fine patches (coarse footprints intersect).");
        // touching (grown-by-one footprints intersect but the footprints themselves do not) = adjacent.
        if (!ag.grow(1).intersect(bh).empty())
          adjacent_ = true;
      }
    }
  }

  /// ADC-636 inter-level nesting check (design 4a). Refuses a NON-NESTED patch: a level-(k+1) patch
  /// whose GROWN level-k footprint is not contained in the covered/footprint region of a single
  /// level-k patch, so its C/F ghosts have no parent to bilerp. Called by the N-level ctor.
  void validate_nesting_(const std::vector<BoxArray>& level_boxes) {
    const int L = n_levels_;
    for (int k = 1; k + 1 < L; ++k) {
      // parents = level-k patches (level_boxes[k-1]); children = level-(k+1) patches (level_boxes[k]).
      const BoxArray& parents = level_boxes[k - 1];
      const BoxArray& children = level_boxes[k];
      for (int g = 0; g < children.size(); ++g) {
        // child footprint on level k = PatchRange(child) grown by one (the C/F ghost ring reads it).
        const Box2D foot = PatchRange(children[g]).box().grow(1);
        bool nested = false;
        for (int p = 0; p < parents.size(); ++p)
          if (parents[p].contains(foot)) {
            nested = true;
            break;
          }
        if (!nested)
          throw std::runtime_error(
              "CompositeFacPoisson: non-nested fine patch (a level-(k+1) patch grown footprint is "
              "not contained in a single parent patch; its coarse-fine bilerp has no source).");
      }
    }
  }

  /// Fills the ghost ring of EACH fine patch by bilerp of phi_c (cell-centered C-F Dirichlet).
  /// Since the patches are separated by at least one coarse cell, the ghost ring of a patch never
  /// overlaps the valid cells of another -> read from the coarse only (no fine-fine exchange).
  void fill_cf_ghosts() {
    const ConstArray4 C = phi_c_.fab(0).const_array();  // replicated mono-box coarse
    for (int li = 0; li < phi_f_.local_size(); ++li) {
      Array4 F = phi_f_.fab(li).array();
      const Box2D vb = phi_f_.box(li);
      for_each_cell(phi_f_.fab(li).grown_box(),
                    detail::FacLegacyFillCoarseFineGhostKernel{F, C, vb, ratio_});
    }
  }

  /// Fills the ghosts of a fine COEFFICIENT field (@p fine) by bilerp of the coarse field (@p coarse):
  /// coefficient consistency at the C-F interface (the coefficient face at the patch border mixes the fine
  /// interior coeff and the injected coarse coeff). Generic (eps, a_xy, a_yx).
  void fill_cf_coarse_to_fine(const MultiFab& coarse, MultiFab& fine) {
    const ConstArray4 C = coarse.fab(0).const_array();  // replicated mono-box coarse
    for (int li = 0; li < fine.local_size(); ++li) {
      Array4 F = fine.fab(li).array();
      const Box2D vb = fine.box(li);
      for_each_cell(fine.fab(li).grown_box(),
                    detail::FacLegacyFillCoarseFineGhostKernel{F, C, vb, ratio_});
    }
  }

  /// Coefficient (eps) BC: periodic preserved, physical border -> zero-gradient (Foextrap), like
  /// the Schur builder (coeff_bc) -- the coefficient carries no Dirichlet.
  static BCRec coeff_bc(const BCRec& b) {
    auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
    BCRec c;
    c.xlo = fo(b.xlo);
    c.xhi = fo(b.xhi);
    c.ylo = fo(b.ylo);
    c.yhi = fo(b.yhi);
    return c;
  }

  /// SOR over-relaxation factor ~ optimal for a patch (2/(1+sin(pi/N))) -> O(N) sweeps convergence
  /// instead of O(N^2) for GS. N = largest side of box @p b (computed per patch in multi-patch).
  Real sor_omega(const Box2D& b) const {
    const int N = std::max(b.nx(), b.ny());
    return Real(2) / (Real(1) + std::sin(Real(kPi_) / Real(N)));
  }

  /// Re-fills the bilinear C-F ghosts from phi_c then relaxes EACH fine patch (SOR) with FROZEN ghosts.
  void refresh_fine(int sweeps) {
    device_fence();
    fill_ghosts(phi_c_, geom_c_.domain,
                bc_);  // phi_c physical ghosts (the bilerp reads up to the border)
    fill_cf_ghosts();
    fine_sor(sweeps);
    average_down(phi_f_, phi_c_,
                 ratio_);  // consistency: coarse covered = fine average (multi-box OK)
  }

  /// Red-black SOR over EACH fine patch: div(eps grad phi_f)-kappa phi_f=f_f, FROZEN
  /// ghosts (no re-filling). eps == 1 everywhere (scalar) -> Laplacian, bit-identical to Phase 1.
  /// The over-relaxation factor is computed PER PATCH (own size). Since the patches are separated, the
  /// 9-point stencil of a patch never reads the valid cells of another (frozen ghosts only).
  void fine_sor(int sweeps) {
    const Real idx2 = Real(1) / (geom_f_.dx() * geom_f_.dx());
    const Real idy2 = Real(1) / (geom_f_.dy() * geom_f_.dy());
    const bool he = has_eps_;
    const bool hc = has_cross_;
    const Real idx = Real(1) / geom_f_.dx(), idy = Real(1) / geom_f_.dy();  // cross_div: 1/dx, 1/dy
    for (int li = 0; li < phi_f_.local_size(); ++li) {
      const Box2D vb = phi_f_.box(li);
      const Real omega = sor_omega(vb);
      Array4 P = phi_f_.fab(li).array();
      const ConstArray4 Pc =
          phi_f_.fab(li).const_array();  // const view (same memory) for cross stencil
      const ConstArray4 F = f_f_.fab(li).const_array();
      const ConstArray4 E = eps_f_.fab(li).const_array();
      const ConstArray4 EY =
          has_eps_y_ ? eps_y_f_.fab(li).const_array() : eps_f_.fab(li).const_array();
      const ConstArray4 AXY = axy_f_.fab(li).const_array();
      const ConstArray4 AYX = ayx_f_.fab(li).const_array();
      const int color_count = hc ? 4 : 2;
      for (int s = 0; s < sweeps; ++s)
        for (int color = 0; color < color_count; ++color)
          for_each_cell(vb, detail::FacLegacySorKernel{
                                P, Pc, F, E, EY, AXY, AYX, idx2, idy2, idx, idy, omega,
                                has_reaction_ ? reaction_ : Real(0), color, he, hc});
    }
  }

  /// Composite coarse residual: r_c = f_c - div(eps grad phi_c) (non covered), 0 (covered), + C-F
  /// FLUX correction on the cells bordering the patch. @return ||r_c||_inf (NON covered cells).
  Real composite_coarse_residual() {
    MultiFab& operator_view = prepare_field_residual_view(
        phi_c_, has_boundary_kernel_ ? &boundary_view_c_ : nullptr, geom_c_, bc_,
        has_boundary_kernel_ ? &boundary_kernel_ : nullptr,
        has_boundary_kernel_ ? &boundary_context_ : nullptr);
    // r_c = f_c - div(A grad phi_c) (apply_laplacian reads the already-filled ghosts; eps + cross if active).
    // The cross terms are read also on the COVERED cells (= fine average after average_down) -> the
    // 9-point stencil stays consistent at the interface; only the NORMAL flux is explicitly joined C-F
    // (the cross flux, tangential and small for the Schur step, is carried by the volume stencil).
    apply_laplacian(operator_view, geom_c_, lap_c_, /*coef=*/nullptr, has_eps_ ? &eps_c_ : nullptr,
                    /*kappa=*/nullptr, has_eps_y_ ? &eps_y_c_ : nullptr,
                    has_cross_ ? &axy_c_ : nullptr, has_cross_ ? &ayx_c_ : nullptr);
    if (has_reaction_)
      apply_constant_reaction_(lap_c_, operator_view);
    Array4 R = res_c_.fab(0).array();
    const ConstArray4 LAP = lap_c_.fab(0).const_array();
    const ConstArray4 FC = f_c_.fab(0).const_array();
    const Box2D b = res_c_.box(0);
    const CoverageMaskView coverage = cov_.view();
    for_each_cell(b, detail::FacLegacyMaskedResidualKernel{R, FC, LAP, coverage});

    // C-F FLUX CORRECTION, PER FINE PATCH. On each coarse cell BORDERING a patch (non covered,
    // covered neighbor), we REPLACE the contribution of the C-F face in div(eps grad phi_c) by the
    // FINE contribution (conservative sum of the r fine faces, harmonic face eps): r_c += (coarse
    // - fine). Since the patches are separated by at least one coarse cell, each border is a TRUE
    // coarse-fine join; the test !cov_.covered(I, J) defensively skips a bordering cell that would be
    // covered by ANOTHER patch (impossible under the guard, but robust: a covered bordering
    // cell is already interior to another patch, its residual stays 0). A cell SEPARATING two
    // patches (right border of one, left border of the other) gets TWO corrections, one per face: correct.
    const ConstArray4 PC = phi_c_.fab(0).const_array();
    const ConstArray4 EC = eps_c_.fab(0).const_array();
    const ConstArray4 EYC =
        has_eps_y_ ? eps_y_c_.fab(0).const_array() : eps_c_.fab(0).const_array();
    const bool he = has_eps_;
    const Real idx2 = Real(1) / (geom_c_.dx() * geom_c_.dx());
    const Real idy2 = Real(1) / (geom_c_.dy() * geom_c_.dy());
    const int r = ratio_;
    for (int g = 0; g < phi_f_.local_size(); ++g) {
      const ConstArray4 PF = phi_f_.fab(g).const_array();
      const ConstArray4 EF = eps_f_.fab(g).const_array();
      const ConstArray4 EYF =
          has_eps_y_ ? eps_y_f_.fab(g).const_array() : eps_f_.fab(g).const_array();
      for_each_cell(patch_coarse_[g].grow(1),
                    detail::FacLegacyFluxCorrectionKernel{
                        R, PC, EC, EYC, PF, EF, EYF, coverage, patch_coarse_[g], idx2, idy2, r,
                        he});
    }

    // inf norm of the residual over the NON covered cells.
    return reduce_max_cell(b, detail::FacLegacyMaskedNormKernel{res_c_.fab(0).const_array(),
                                                                coverage});
  }

  void record_residual(int iteration, Real residual) {
    if (!verbose_)
      return;
    (void)diagnostics_.try_record("elliptic.fac.residual", "CompositeFacPoisson", "info",
                                  iteration < 0 ? "initial composite hierarchy residual"
                                                : "FAC iteration composite hierarchy residual",
                                  iteration, static_cast<double>(residual));
  }

  Geometry geom_c_, geom_f_;
  BoxArray ba_c_;
  DistributionMapping dm_c_;
  BCRec bc_;
  int ratio_;
  BoxArray ba_f_;
  DistributionMapping dm_f_;
  GeometricMG mg_;  ///< coarse solver (initial + corrections), homogeneous Dirichlet
  MultiFab phi_c_, phi_f_, f_c_, f_f_, res_c_, lap_c_, lap_f_, boundary_view_c_;
  MultiFab eps_c_, eps_f_;                  ///< x-normal diagonal coefficient per level
  MultiFab eps_y_c_, eps_y_f_;              ///< optional y-normal diagonal coefficient per level
  MultiFab axy_c_, ayx_c_, axy_f_, ayx_f_;  ///< cross terms per level (full tensor, Schur B_z!=0)
  std::vector<Box2D> patch_coarse_;  ///< covered coarse footprint PER fine patch (multi-patch)
  CoverageMask cov_;
  Real last_residual_ = 0;
  RuntimeDiagnosticsReport diagnostics_ =
      make_runtime_diagnostics_report("pops.numerics.elliptic.composite_fac_poisson");
  bool has_eps_ = false;    ///< true: div(eps grad phi) operator; false: scalar Laplacian (Phase 1)
  bool has_eps_y_ = false;  ///< true: y faces use eps_y; false: isotropic y faces reuse eps
  bool has_cross_ = false;  ///< true: adds the cross terms a_xy/a_yx (full tensor, Schur B_z!=0)
  bool has_reaction_ = false;  ///< true: constant Helmholtz term -reaction_*phi on every level
  Real reaction_ = Real(0);
  bool has_boundary_kernel_ = false;
  CompiledFieldBoundaryKernel boundary_kernel_{};
  FieldBoundaryExecutionContext boundary_context_{};
  FieldBoundaryFailure boundary_failure_{};
  std::vector<MultiFab> phi_probe_snapshot_;  ///< persistent full-state snapshots for exact R(0)
  MultiFab boundary_probe_snapshot_;          ///< persistent generated-boundary view snapshot
  std::vector<MultiFab> phi_published_snapshot_;  ///< persistent rollback state for boundary FAS
  bool has_field_nonlinear_options_ = false;
  FieldNewtonOptions field_nonlinear_options_{};
  SolveReport last_solve_report_{};
  bool verbose_ = false;
  bool two_way_ = true;
  CompositeFacOptions options_;  ///< Installed FAC budgets, mixed tolerances and diagnostics knobs.
  int n_levels_ =
      2;  ///< ADC-636: hierarchy depth; 2 for the historical ctors, 1+patch-levels for N-level.
  bool adjacent_ =
      false;  ///< ADC-636: true when the hierarchy has edge/corner-touching fine patches (general path).
  bool force_general_ =
      false;  ///< ADC-636 test hook: route the 2-level input through the general path.
  static constexpr Real kPi_ = Real(3.14159265358979323846);

  // ADC-636 N-level storage (levels k >= 2; levels 0/1 keep the members above). One entry per
  // extra patch level, index 0 == level 2. Allocated by the N-level ctor; empty on the 2-level path
  // so the historical allocation is untouched. geom_lv_[k-2] = geom_c_.refine(2^k); cov_lv_[k-2] is
  // the coverage of level k by level k+1 (empty for the finest); foot_lv_[k-2][g] is the coarse
  // (level k-1) footprint of patch g. mg_lv_[k-2] serves the intermediate-level correction solve.
  MultiFab
      res_f_;  ///< level-1 composite residual buffer (report/stop norm and N-level correction).
  std::vector<Geometry> geom_lv_;  ///< geom_lv_[k-2] = geom_c_.refine(2^k) for level k >= 2
  std::vector<BoxArray> ba_lv_;    ///< ba_lv_[k-2] = the level-k patch tiling
  std::vector<DistributionMapping> dm_lv_;
  std::vector<MultiFab> phi_lv_, f_lv_, res_lv_, lap_lv_, eps_lv_, eps_y_lv_, axy_lv_, ayx_lv_;
  // Uniform per-level metadata, index m in [0, L-1] (covers level 0/1 as well as k >= 2 so the driver
  // loops without special-casing). cov_of_[m] = coverage of level m by level m+1 (finest: none).
  // foot_of_[m][g] = PatchRange of patch g of level m on level m-1 (empty at m == 0).
  std::vector<CoverageMask> cov_of_;
  std::vector<std::vector<Box2D>> foot_of_;
  // Intermediate-level correction multigrid: level_mg_[m] serves level m for 1 <= m <= L-2 (the
  // finest patch level is relaxed by SOR only). [0] (base mg_) and [L-1] stay null.
  std::vector<std::unique_ptr<GeometricMG>> level_mg_;
  // A hierarchy whose every transition covers the complete parent domain has no coarse/fine
  // interface and no uncovered coarse unknown.  Its exact composite operator is therefore the
  // uniform operator on the finest level, followed by conservative restriction to its covered
  // parents.  Materialize that solver with the hierarchy (never in solve()) so the residual cannot
  // silently collapse to zero merely because the interface set is empty.
  std::unique_ptr<GeometricMG> fully_refined_solver_;
  // Hot-path MPI communication scratch.  Built with the hierarchy, then reset/reused by every
  // residual/correction: no FluxRegister or MultiFab allocation is allowed inside solve().
  std::vector<std::unique_ptr<FluxRegister>> flux_registers_;
  std::unique_ptr<FluxRegister> coarse_average_register_;
  std::vector<MultiFab> correction_residual_replicated_, correction_eps_replicated_,
      correction_eps_y_replicated_, correction_axy_replicated_, correction_ayx_replicated_;

  // ADC-636: the general FAC (N levels / adjacent patches / MPI). Declared here; DEFINED out-of-line
  // in composite_fac_nlevel.hpp (tail-included below) so composite_fac_poisson.hpp keeps the legacy
  // body + dispatch and the general machinery lives in the mg/ layer per ADC-334.
  LinearSolveResult solve_composite_nlevel_(int max_iters, int fine_sweeps, Real stop);
  void build_extra_levels_(const std::vector<BoxArray>& level_boxes);
  // Build the uniform per-level metadata (cov_of_[m] = level m covered by level m+1, foot_of_[m][g] =
  // level-(m-1) footprint of patch g, level_mg_[m] = intermediate correction multigrid) from the
  // current hierarchy. Called by EVERY ctor (the 2-level ctor too) so the general path can be reached
  // for any shape, including a 2-level input via the test hook.
  void finalize_hierarchy_metadata_();
  void prepare_fully_refined_solver_();
  Real solve_fully_refined_hierarchy_(int max_iters, Real rel_tol, Real abs_tol);
  void setup_level_coeffs_();
  void fill_cf_field_(int k, MultiFab& fine,
                      const MultiFab& parent);  // C-F bilerp parent -> level k
  void fill_cf_phi_(int k);                     // ghost order 3b for phi_level(k)
  void relax_level_(int m, int sweeps);         // C-F ghost + fill_boundary + SOR (no avgdown)
  void cascade_avgdown_();                      // fine-to-coarse average-down of the whole tower
  void correct_level_(int m);                   // L_m e_m = res_m; phi_m += e_m on uncovered
  Real composite_residual_(int m);              // res_m + C/F flux correction; return ||.||_inf
  void fine_sor_level_(int m, const MultiFab& f_eff, int sweeps);  // red-black SOR on level m
  // Accumulate the level-m/level-(m+1) two-way C-F flux correction into dst (level-m residual or
  // effective RHS), enumerated from the uncovered coarse side (design 4c). single_writer_gather
  // routes remote fine fluxes through a per-face FluxRegister (ADC-636 commit 4) for MPI bit-identity.
  void add_flux_correction_(int m, MultiFab& dst);
  MultiFab& res_level_(int m) { return m == 0 ? res_c_ : (m == 1 ? res_f_ : res_lv_[m - 2]); }
  MultiFab& lap_level_(int m) { return m == 0 ? lap_c_ : (m == 1 ? lap_f_ : lap_lv_[m - 2]); }
  void add_uncovered_level_(int m, MultiFab& phi,
                            const MultiFab& e);     // phi += e on uncovered cells
  void copy0_(MultiFab& dst, const MultiFab& src);  // dst <- src (comp 0, valid)
  void apply_constant_reaction_(MultiFab& value, const MultiFab& phi) const {
    for (int li = 0; li < value.local_size(); ++li)
      for_each_cell(value.box(li),
                    detail::FacApplyConstantReactionKernel{value.fab(li).array(),
                                                           phi.fab(li).const_array(), reaction_});
  }
  // Average-down phi_level(m) -> phi_level(m-1). When the parent (m-1) is the REPLICATED coarse under
  // MPI, parallel_copy would only update the src-owner rank; this routes the covered-cell averages
  // through a single-writer FluxRegister so every rank's replicated parent gets the same values
  // (identity at np=1, so mono-rank stays bit-identical to the legacy average_down).
  void average_down_level_(int m);
  Real sor_omega_(const Box2D& b) const {  // per-patch over-relaxation (== legacy sor_omega)
    const int N = std::max(b.nx(), b.ny());
    return Real(2) / (Real(1) + std::sin(Real(kPi_) / Real(N)));
  }
};

}  // namespace pops

// ADC-636: the general N-level / adjacent / MPI FAC lives here (mg/ layering, ADC-334). Tail-included
// so composite_fac_poisson.hpp above keeps only the legacy body + dispatch + N-level ctor/accessors.
#include <pops/numerics/elliptic/mg/composite_fac_nlevel.hpp>  // solve_composite_nlevel_ + helpers
