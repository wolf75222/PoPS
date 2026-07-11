#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include <pops/amr/hierarchy/refinement_ratio.hpp>  // kAmrRefRatio (ratio 2)
#include <pops/core/foundation/kokkos_env.hpp>       // device_fence
#include <pops/mesh/layout/box_array.hpp>            // BoxArray / Box2D
#include <pops/mesh/storage/mf_arith.hpp>            // pops::lincomb (device-clean copy / negate)
#include <pops/mesh/storage/multifab.hpp>            // MultiFab / DistributionMapping
#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>  // CompositeFacPoisson (composite FAC elliptic)
#include <pops/parallel/comm.hpp>                    // pops::n_ranks (MPI-multilevel refusal)
#include <pops/runtime/amr/amr_runtime.hpp>          // AmrRuntime (the engine this helper reads)

/// @file
/// @brief AmrCondensedElliptic -- the composite tensor-coefficient elliptic driver a compiled
///        condensed-implicit time Program routes to on a REFINED AMR hierarchy (ADC-633 / ADC-637).
///
/// The compiled condensed-implicit Program lowers, per AMR level, to inline block-inverse assembly
/// kernels (no coupling/schur call, ADC-637). On a FLAT hierarchy those run the emitted matrix-free
/// BiCGStab on level 0 -- bit-identical to the uniform Program. On a REFINED hierarchy (>= one fine
/// patch) the single-level matrix-free solve cannot address the fine levels, so the tensor elliptic is
/// solved COMPOSITELY: this helper owns per-level tensor-coefficient buffers (eps_x / eps_y / a_xy /
/// a_yx), a per-level right-hand side and a per-level potential, plus a lazily built, box-cached
/// pops::CompositeFacPoisson (two-way, variable coefficient + cross terms). The emitted assembly ops
/// write THROUGH AmrProgramContext::assembly_target into these level-shaped buffers (the level-0-bound
/// emitted scratch is unusable on a fine level); solve_composite() copies them into the FAC's per-level
/// fields, solves, and publishes each level's potential for the emitted reconstruction to READ through
/// AmrProgramContext::assembly_source.
///
/// GENERIC LAYER (owner directive, ADC-637): this driver names ONLY mathematical objects -- tensor
/// coefficients, right-hand side, potential, composite FAC. No B_z / Lorentz / electrostatic / Schur
/// vocabulary: the physics is authored in the DSL and emitted inline; this helper just co-distributes
/// the level buffers and drives the composite solve.
///
/// SCOPE. Inherited verbatim from pops::CompositeFacPoisson (ADC-636 generalized envelope): N levels,
/// 1..N disjoint fine patches (nested, ratio 2), replicated mono-box coarse. MPI multilevel is refused
/// precisely (mono-rank only) -- the composite path is a mono-rank driver here. Beyond that the FAC
/// ctor refuses (non-nested / misaligned patches) with a precise message; no silent partial solve.
/// Single block (the AMR Program v1 block scope). theta == 1 (the macro's theta gate is upstream).

namespace pops {
namespace runtime {
namespace program {

/// Per-level tensor-coefficient buffers + a cached composite FAC solve, for one AMR block's condensed
/// tensor elliptic on a refined hierarchy. Owned by AmrProgramContext (one per installed Program on the
/// refined path); rebuilt lazily when the fine tiling changes. Indexed by AMR level (0 = coarse).
class AmrCondensedElliptic {
 public:
  /// @p eng: the AMR engine (levels / geom / bc); @p block: the AMR block index (sys_block-resolved by
  /// the caller). Buffers are allocated lazily on ensure_level_buffers() so a flat hierarchy (never
  /// refined) allocates nothing.
  AmrCondensedElliptic(AmrRuntime* eng, int block) : eng_(eng), block_(block) {}

  /// True iff there is >= one populated fine level (level 1 carries >= one patch for this block). The
  /// AmrProgramContext gates the flat (matrix-free BiCGStab) vs composite (FAC) branch on this.
  bool has_fine_patches() const {
    if (eng_->nlev() < 2)
      return false;
    return eng_->level_state(static_cast<std::size_t>(block_), 1).box_array().size() > 0;
  }

  /// The level-shaped WRITE target for an assembly field of @p role at level @p k. The emitted
  /// assembly kernel reaches it via AmrProgramContext::assembly_target so its per-cell write lands in
  /// the composite buffer instead of the level-0-bound emitted scratch. Roles map to the AssemblyFieldRole
  /// enum in coeff_elliptic_ops.hpp (eps_x / eps_y / a_xy / a_yx / rhs / flux).
  MultiFab& target(int role, int k) {
    ensure_level_buffers(k);
    LevelBuffers& lb = levels_[static_cast<std::size_t>(k)];
    switch (role) {
      case 0: return lb.eps_x;   // kEpsX
      case 1: return lb.eps_y;   // kEpsY
      case 2: return lb.a_xy;    // kAxy
      case 3: return lb.a_yx;    // kAyx
      case 4: return lb.rhs;     // kRhs
      case 5: return lb.flux;    // kFlux (transient explicit-flux scratch)
      default:
        throw std::runtime_error("AmrCondensedElliptic::target: unknown AssemblyFieldRole wire id " +
                                 std::to_string(role));
    }
  }

  /// The published composite potential of level @p k (filled by solve_composite): the emitted
  /// reconstruction reads it as phi^{n+theta} on that level (via AmrProgramContext::assembly_source).
  MultiFab& phi(int k) {
    ensure_level_buffers(k);
    return levels_[static_cast<std::size_t>(k)].phi;
  }

  /// Solve the composite tensor elliptic across the whole nested tower: build/reuse the FAC on the fine
  /// tilings, copy the per-level coefficient / RHS buffers into the FAC's level fields, enable variable
  /// coefficient + cross terms + two-way, solve, then publish each level's potential into phi(k). REUSES
  /// pops::CompositeFacPoisson wholesale (the SAME composite solver the native source-stage AMR route
  /// drives, ADC-636), so a refined Program matches that route where the coefficient / RHS feed
  /// coincides. The FAC's own operator / iteration decide the solve; the emitted matrix-free apply /
  /// precond are UNUSED on this branch (documented). MPI multilevel is refused precisely (mono-rank).
  void solve_composite() {
    const int L = eng_->nlev();
    if (L < 2)
      return;  // flat: the caller never reaches this branch (has_fine_patches() is false).
    if (pops::n_ranks() != 1)
      throw std::runtime_error(
          "AmrCondensedElliptic::solve_composite: the composite condensed-implicit elliptic on a "
          "refined hierarchy is mono-rank (the inherited CompositeFacPoisson envelope); MPI multilevel "
          "is deferred pending ADC-648 (use System, or the native AMR source-stage route).");
    for (int k = 0; k < L; ++k)
      ensure_level_buffers(k);

    // The fine tilings (levels 1..L-1) key the FAC build; rebuild only when a tiling changes.
    std::vector<BoxArray> level_boxes;
    for (int k = 1; k < L; ++k)
      level_boxes.push_back(eng_->level_state(static_cast<std::size_t>(block_), k).box_array());
    ensure_fac(level_boxes);

    fac_->use_variable_coefficient(true);
    fac_->use_cross_terms(true);
    fac_->set_two_way(true);
    for (int k = 0; k < L; ++k) {
      LevelBuffers& lb = levels_[static_cast<std::size_t>(k)];
      // The tensor coefficient A = [[eps_x, a_xy], [a_yx, eps_y]] per level (the FAC diagonal block uses
      // eps_x; eps_y is symmetric for the condensed operator and folded into the diagonal by the FAC).
      copy0(fac_->eps_level(k), lb.eps_x);
      copy0(fac_->a_xy_level(k), lb.a_xy);
      copy0(fac_->a_yx_level(k), lb.a_yx);
      // the emitted condensed_rhs builds -Lap phi^n - g div(F): the matrix-free operator sign is
      // -div(A grad); the FAC solves div(eps grad phi) = f, so f = -rhs (the sign convention #126).
      negate_into(fac_->rhs_level(k), lb.rhs);
    }

    fac_->solve();

    for (int k = 0; k < L; ++k)
      copy0(levels_[static_cast<std::size_t>(k)].phi, fac_->phi_level(k));
  }

 private:
  struct LevelBuffers {
    MultiFab eps_x, eps_y, a_xy, a_yx;  ///< tensor coefficient A = [[eps_x, a_xy], [a_yx, eps_y]]
    MultiFab rhs;                        ///< condensed right-hand side (-Lap phi^n - g div F)
    MultiFab flux;                       ///< transient explicit-flux scratch (2-comp, if the body uses it)
    MultiFab phi;                        ///< published composite potential of this level
    bool built = false;
  };

  /// Allocate level @p k's buffers on that level's grid (co-distributed with its state), once. eps /
  /// coefficient / phi carry 1 ghost (the operator face mean + the centered gradient); rhs 0 ghost.
  void ensure_level_buffers(int k) {
    if (k >= static_cast<int>(levels_.size()))
      levels_.resize(static_cast<std::size_t>(k) + 1);
    LevelBuffers& lb = levels_[static_cast<std::size_t>(k)];
    if (lb.built)
      return;
    const MultiFab& U = eng_->level_state(static_cast<std::size_t>(block_), k);
    const BoxArray ba = U.box_array();
    const DistributionMapping dm = U.dmap();
    lb.eps_x = MultiFab(ba, dm, 1, 1);
    lb.eps_y = MultiFab(ba, dm, 1, 1);
    lb.a_xy = MultiFab(ba, dm, 1, 1);
    lb.a_yx = MultiFab(ba, dm, 1, 1);
    lb.rhs = MultiFab(ba, dm, 1, 0);
    lb.flux = MultiFab(ba, dm, 2, 1);
    lb.phi = MultiFab(ba, dm, 1, 1);
    lb.eps_x.set_val(Real(0));
    lb.eps_y.set_val(Real(0));
    lb.a_xy.set_val(Real(0));
    lb.a_yx.set_val(Real(0));
    lb.rhs.set_val(Real(0));
    lb.flux.set_val(Real(0));
    lb.phi.set_val(Real(0));
    lb.built = true;
  }

  /// Build (or rebuild on a fine-tiling change) the composite FAC over ALL fine levels -- the verbatim
  /// ensure_fac idiom of the native source stepper (compare per-level boxes + order, rebuild only on a
  /// change). A single fine level uses the 2-level ctor (bit-identical), deeper towers the N-level ctor;
  /// the FAC ctor refuses ratio != 2 / non-nested / misaligned patches, precisely.
  void ensure_fac(const std::vector<BoxArray>& level_boxes) {
    std::vector<std::vector<Box2D>> key;
    key.reserve(level_boxes.size());
    for (const BoxArray& ba : level_boxes)
      key.push_back(ba.boxes());
    if (fac_ && fac_level_boxes_ == key)
      return;
    const Geometry geom_c = eng_->level_geom(0);
    const BoxArray coarse_ba =
        eng_->level_state(static_cast<std::size_t>(block_), 0).box_array();
    if (level_boxes.size() == 1)
      fac_ = std::make_unique<CompositeFacPoisson>(geom_c, coarse_ba, eng_->poisson_bc(),
                                                   level_boxes[0], kAmrRefRatio);
    else
      fac_ = std::make_unique<CompositeFacPoisson>(geom_c, coarse_ba, eng_->poisson_bc(), level_boxes,
                                                   kAmrRefRatio);
    fac_level_boxes_ = std::move(key);
  }

  static void copy0(MultiFab& dst, const MultiFab& src) {
    device_fence();
    pops::lincomb(dst, Real(1), src, Real(0), src);  // dst <- src (comp 0), device-clean
  }
  static void negate_into(MultiFab& dst, const MultiFab& src) {
    device_fence();
    pops::lincomb(dst, Real(-1), src, Real(0), src);  // dst <- -src
  }

  AmrRuntime* eng_;
  int block_;
  std::vector<LevelBuffers> levels_;
  std::unique_ptr<CompositeFacPoisson> fac_;
  std::vector<std::vector<Box2D>> fac_level_boxes_;
};

}  // namespace program
}  // namespace runtime
}  // namespace pops
